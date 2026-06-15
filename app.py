#!/usr/bin/env python3
"""
app.py — cloud mobile dashboard for the NSE scanner (Streamlit).

Designed to run on Streamlit Community Cloud (always-on, no PC needed). From your
phone you can: store + TEST your Dhan token, run a LIVE single-stock check, and run
a LIVE watchlist scan (trend / near-high / momentum). It also displays the last
Engine A / Engine B watchlists if their CSVs are committed to the repo.

What it CANNOT do (needs the 2000-stock cache + ~20 min, not phone/free-cloud work):
the full-universe scan and cross-sectional RS percentile. Those run on the PC
(run_scan.ps1 / run_engine_b.ps1); push their output CSVs to the repo to view here.

Deploy: push repo to GitHub -> share.streamlit.io -> deploy app.py. Open URL on phone.
"""
import base64
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import nse_scanner as ns

st.set_page_config(page_title="NSE Scanner", page_icon="📈", layout="centered")


def _token_expiry(tok):
    try:
        p = tok.split(".")[1]; p += "=" * (-len(p) % 4)
        exp = json.loads(base64.urlsafe_b64decode(p)).get("exp")
        return datetime.fromtimestamp(exp) if exp else None
    except Exception:
        return None


@st.cache_data(show_spinner="Loading Dhan scrip master ...")
def secmap():
    return ns.load_dhan_security_map()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_hist(cid, tok, sid, days=730, seg="NSE_EQ", it="EQUITY"):
    d = ns.dhanhq(cid, tok)
    r = d.historical_daily_data(sid, seg, it,
                                (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                                datetime.now().strftime("%Y-%m-%d"))
    if r.get("status") != "success":
        return None
    return ns._dhan_to_dataframe(r.get("data"))


STRONG_THEMES = {"Capital Goods", "Construction", "Realty", "Power", "Metals & Mining"}


@st.cache_data(show_spinner=False)
def quarterly_candidates():
    """Persistent-sales-acceleration names from the shipped quarterly.csv (no prices)."""
    path = "data_cache/quarterly.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    q = pd.read_csv(path)
    q["idx"] = q["q_end"].str.slice(0, 4).astype(int) * 12 + q["q_end"].str.slice(5, 7).astype(int)
    ind = {}
    if os.path.exists("data_cache/industry_map.csv"):
        im = pd.read_csv("data_cache/industry_map.csv")
        ind = dict(zip(im["Symbol"].str.strip(), im["Industry"].str.strip()))
    rows = []
    for sym, g in q.groupby("symbol"):
        d = {r["idx"]: (r["sales"], r["net_profit"]) for _, r in g.iterrows()}

        def yoy(i):
            return d[i][0] / d[i - 12][0] - 1 if (i in d and i - 12 in d and d[i - 12][0] > 0) else None
        i = max(d)
        gg = [yoy(i - 3 * j) for j in range(4)]
        c = 0
        for a, b in zip(gg, gg[1:]):
            if a is not None and b is not None and a > b:
                c += 1
            else:
                break
        sy = yoy(i)
        if c >= 2 and sy is not None and 0.15 <= sy <= 3.0 and d[i][1] > 0:
            rows.append({"symbol": str(sym).strip(), "sales_yoy%": round(sy * 100, 1),
                         "consec": c, "industry": ind.get(str(sym).strip(), "Unmapped")})
    return pd.DataFrame(rows)


def run_discovery_live(cid, tok, cap=120):
    """Engine-B-lite: take fundamental candidates, fetch live prices for THOSE only,
    add 'still quiet' context (RS vs NIFTY, distance from high). Cloud/phone runnable."""
    cands = quarterly_candidates()
    if cands.empty:
        return cands
    cands = cands.head(cap)
    sm = secmap()
    nfh = fetch_hist(cid, tok, "13", 400, seg="IDX_I", it="INDEX")
    nret6 = (nfh["Close"].astype(float).iloc[-1] / nfh["Close"].astype(float).iloc[-126] - 1) \
        if (nfh is not None and len(nfh) > 126) else 0.0
    out, prog = [], st.progress(0.0, text="Fetching live prices for candidates ...")
    for k, (_, r) in enumerate(cands.iterrows(), 1):
        prog.progress(k / len(cands))
        sid = sm.get(r["symbol"])
        if not sid:
            continue
        df = fetch_hist(cid, tok, sid)
        if df is None or len(df) < 220:
            continue
        c = df["Close"].astype(float)
        v = df["Volume"].astype(float)
        price = c.iloc[-1]; hi = c.tail(252).max()
        ret6 = price / c.iloc[-126] - 1
        turn = (c * v).tail(50).median() / 1e7
        if turn < 2.0:
            continue
        out.append({
            "symbol": r["symbol"], "industry": r["industry"],
            "theme": "★" if r["industry"] in STRONG_THEMES else "",
            "sales_yoy%": r["sales_yoy%"], "consec": r["consec"],
            "from_high%": round((hi - price) / hi * 100, 1),
            "rs_vs_nifty%": round((ret6 - nret6) * 100, 1),
            "above200": "✅" if price > c.rolling(200).mean().iloc[-1] else "",
            "turnover_cr": round(float(turn), 1), "price": round(price, 1),
        })
    prog.empty()
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # discovery = still quiet (low RS-vs-NIFTY) + strong theme; sort quietest first
    df["early_score"] = (df["theme"] == "★").astype(int) * 20 - df["rs_vs_nifty%"]
    return df.sort_values("early_score", ascending=False).drop(columns="early_score")


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------- Sidebar: token ----------------
st.sidebar.header("Dhan connection")
cid = st.sidebar.text_input("Client ID", value=os.getenv("DHAN_CLIENT_ID", ""))
tok = st.sidebar.text_input("Access token", value="", type="password")
ready = bool(cid and tok)
if tok:
    exp = _token_expiry(tok)
    if exp:
        st.sidebar.write(("🟢 valid until " if exp > datetime.now() else "🔴 EXPIRED ")
                         + exp.strftime("%Y-%m-%d %H:%M"))
if st.sidebar.button("Test connection", disabled=not ready):
    try:
        d = ns.dhanhq(cid, tok)
        r = d.historical_daily_data("13", "IDX_I", "INDEX",
                                    (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                                    datetime.now().strftime("%Y-%m-%d"))
        (st.sidebar.success("✅ Connected.") if r.get("status") == "success"
         else st.sidebar.error(f"❌ {r.get('remarks', r.get('data'))}"))
    except Exception as e:
        st.sidebar.error(f"❌ {e}")
st.sidebar.caption("Token is used only in your session for read-only data calls. "
                   "It expires ~daily. Re-enter after regenerating on Dhan.")

st.title("📈 NSE Scanner")
t_live, t_wl, t_disc, t_exec, t_track = st.tabs(
    ["🔎 Live", "📋 Watchlist", "🔭 Discovery", "🎯 Execution", "📊 Tracker"])


def flags(df):
    t = ns.analyze(df)
    if not t:
        return None
    return {
        "price": round(t["price"], 1),
        "from_high%": round(t["pct_from_high"] * 100, 1),
        "6m%": round(t["rs_raw"], 1),
        "stacked": "✅" if t["price"] > t["dma50"] > t["dma150"] > t["dma200"] else "",
        "vol_x": round(t["vol_ratio"], 1),
        "setup": t.get("setup", ""),
    }


# ---------------- Live single check ----------------
with t_live:
    sym = st.text_input("NSE symbol", value="RELIANCE").strip().upper()
    if st.button("Check", disabled=not ready):
        sid = secmap().get(sym)
        if not sid:
            st.error(f"{sym} not found.")
        else:
            df = fetch_hist(cid, tok, sid)
            f = flags(df) if df is not None else None
            if not f:
                st.error("No / insufficient data.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Price", f"₹{f['price']:,}")
                c2.metric("From 52w high", f"-{f['from_high%']}%")
                c1.metric("6-month return", f"{f['6m%']:+}%")
                c2.metric("Trend stacked", f["stacked"] or "❌")
                st.caption(f"setup: {f['setup']} · volume {f['vol_x']}x avg · "
                           "(cross-sectional RS needs the full PC scan)")

# ---------------- Live watchlist scan ----------------
with t_wl:
    st.caption("Paste up to 50 NSE symbols — live trend / near-high / momentum check.")
    txt = st.text_area("Symbols", "RELIANCE, TCS, PARAS, POLYMED, NETWEB")
    if st.button("Scan watchlist", disabled=not ready):
        syms = [s.strip().upper() for s in txt.replace(",", " ").split()][:50]
        sm = secmap()
        rows, prog = [], st.progress(0.0)
        for i, s in enumerate(syms, 1):
            prog.progress(i / len(syms))
            sid = sm.get(s)
            if not sid:
                continue
            df = fetch_hist(cid, tok, sid)
            f = flags(df) if df is not None else None
            if f:
                rows.append({"symbol": s, **f})
        prog.empty()
        if rows:
            out = pd.DataFrame(rows).sort_values("6m%", ascending=False)
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.warning("No results (check symbols / token).")

# ---------------- Display committed watchlists ----------------
with t_disc:
    st.caption("Engine B: early names — sales accelerating, in working themes, still quiet. "
               "Driven by quarterly earnings, so weekly is plenty (no need to run daily).")
    if st.button("🔭 Run Discovery LIVE", disabled=not ready):
        with st.spinner("Running Engine B (lite) — fundamentals + live prices ..."):
            res = run_discovery_live(cid, tok)
        if res is None or res.empty:
            st.warning("No candidates (ensure quarterly.csv is in the repo + token valid).")
        else:
            st.success(f"{len(res)} discovery candidates (quietest / strongest-theme first).")
            st.dataframe(res, use_container_width=True, hide_index=True)
            st.caption("'rs_vs_nifty%' low = still undiscovered; ★ = leadership theme. "
                       "These are watchlist names, not buy signals — let them prove themselves.")
    st.divider()
    df = _csv("discovery_watchlist.csv")
    if df is not None:
        st.caption("Last full Engine B run from PC (cross-sectional RS):")
        cols = [c for c in ["symbol", "industry", "theme_strength", "sales_yoy%",
                            "consec_accel", "rs_pctl", "Score"] if c in df.columns]
        st.dataframe(df[cols].head(40), use_container_width=True, hide_index=True)

with t_exec:
    df = _csv("scan_full_strict.csv")
    if df is None:
        df = _csv("scan_breakout.csv")
    if df is None:
        st.info("No execution scan in repo. Run run_scan.ps1 on PC and push the CSV.")
    else:
        cols = [c for c in ["Rank", "Symbol", "Score", "conviction", "price",
                            "rs_pctl", "pct_from_high", "setup"] if c in df.columns]
        st.dataframe(df[cols].head(40), use_container_width=True, hide_index=True)

with t_track:
    log = _csv("discovery_log.csv")
    if log is None or log.empty:
        st.info("No tracker log in repo yet.")
    else:
        n = len(log); mig = int(log.get("migrated", pd.Series(dtype=bool)).sum())
        rel = pd.to_numeric(log.get("rel_ret_pct", 0), errors="coerce").median()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tracked", n)
        c2.metric("→ RS80", f"{mig} ({mig/n*100:.0f}%)" if n else "0")
        c3.metric("vs NIFTY", f"{rel:+.1f}%")
        show = [c for c in ["symbol", "industry", "rs_at_discovery", "rs_current",
                            "migrated", "ret_6m", "rel_ret_pct"] if c in log.columns]
        st.dataframe(log[show], use_container_width=True, hide_index=True)
