#!/usr/bin/env python3
"""
NSE Momentum Scanner — 3-Phase Stock Screener
================================================

Phase 1 : Technical trend template (Minervini-style) + liquidity floor
Phase 2A: Fundamental pass/fail filter (from a screener.in export)
Phase 3 : Composite momentum ranking score

DATA SOURCES
------------
- Prices/volume : DhanHQ historical API (default; set DHAN_CLIENT_ID and
                  DHAN_ACCESS_TOKEN env vars). Yahoo Finance via yfinance is
                  available as a fallback with --source yahoo.
- Fundamentals  : CSV/XLSX export from screener.in (see below). Yahoo's India
                  fundamentals are unreliable, so Phase 2A reads a screener export.

SETUP
-----
    pip install dhanhq yfinance pandas numpy openpyxl
    # DhanHQ tokens are short-lived (≈24h). Regenerate at web.dhan.co and:
    setx DHAN_ACCESS_TOKEN "<token>"   (then open a NEW terminal)

USAGE
-----
    python nse_scanner.py --all-nse --no-fundamentals       # technicals only, all NSE
    python nse_scanner.py --all-nse --fundamentals my.csv   # full 3-phase
    python nse_scanner.py --relax                           # loosen Phase 1 filters
    python nse_scanner.py --source yahoo --universe nifty500.csv

FUNDAMENTALS — screener.in export
---------------------------------
1. Build a screen at screener.in with a query such as:
       Market Capitalization > 500 AND
       Return on capital employed > 15 AND
       Sales growth 3Years > 10 AND
       Profit growth 3Years > 10 AND
       Debt to equity < 1
2. Add those same ratios as columns, then "Export to Excel".
3. Pass the file:  --fundamentals screener_export.csv  (.xlsx also works)

The export has a company *Name* (no NSE ticker), so rows are matched to NSE
symbols via the Dhan scrip master. Unmatched names are reported at runtime.
See sample_screener_export.csv for the expected column layout.

Thresholds live in CONFIG and are overridable: --min-price, --min-turnover,
--min-vol-multiple, --near-high-pct.

Run it from cron each evening after market close (~6 PM IST):
    30 18 * * 1-5  cd /path && /usr/bin/python3 nse_scanner.py --all-nse >> scan.log 2>&1
"""

import argparse
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None


# ----------------------------------------------------------------------------
# CONFIG — tweak thresholds here
# ----------------------------------------------------------------------------
CONFIG = {
    # Phase 1 — technical
    "vol_multiple": 1.5,        # Volume > 1.5x average
    "vol_avg_period": 50,       # average volume lookback
    "near_high_pct": 0.15,      # within 15% of 52-week high
    "min_history_days": 220,    # need >200 for 200DMA
    "min_price": 20.0,          # exclude penny stocks (rupees)
    "min_turnover_cr": 5.0,     # min avg daily turnover (price x avg_vol), crores

    # Phase 2A — fundamental pass/fail filter (from screener.in CSV)
    "min_roce": 15.0,           # ROCE %
    "min_sales_growth": 10.0,   # 3Y sales growth %
    "min_profit_growth": 10.0,  # 3Y profit growth %
    "max_debt_equity": 1.0,     # Debt/Equity
    "min_mcap_cr": 500.0,       # market cap in crores

    # Base / pivot detection ("where did it rally from")
    "bb_lookback": 180,         # bars of history to scan for the base
    "bb_base_lens": (25, 35, 50),  # candidate consolidation lengths (bars)
    "bb_max_base_depth": 0.30,  # base must be tighter than 30% high-low range
    "bb_vol_confirm": 1.3,      # breakout-day volume vs avg for a "confirmed" break
    "bb_fresh_days": 5,         # breakout <= N bars ago = still fresh
    "bb_buy_zone_pct": 0.08,    # <= 8% above pivot = still in buy range
    "bb_extended_pct": 0.20,    # > 20% above pivot = extended / chasing
    "bb_near_pivot_pct": 0.05,  # within 5% below pivot = coiling / pre-breakout

    # Market breadth regime (the dominant driver of breakout success — see research)
    "breadth_riskoff": 45.0,    # % above 200DMA below this = RISK-OFF
    "breadth_riskon": 60.0,     # % above 200DMA above this = RISK-ON
    "breadth_min_sample": 100,  # need this many fetched stocks for a reading
    "nifty_secid": "13",        # Dhan security id for NIFTY 50 (IDX_I)
    "rs_window": 126,           # ~6m relative-return window for RS percentile
    "rs_strong_pctl": 80.0,     # validated RS threshold (research)

    # Phase 3 — ranking weights (must sum to 1.0)
    "w_rel_strength": 0.40,
    "w_volume": 0.20,
    "w_earnings": 0.20,
    "w_tech_trend": 0.20,

    # Run behaviour
    "batch_size": 50,           # yfinance download batch
    "sleep_between": 1.0,       # politeness delay (seconds)

    # DhanHQ data source
    "dhan_history_days": 730,   # ~2y of daily history to pull
    "dhan_sleep": 0.25,         # per-symbol delay (Dhan data API rate limit)
    "dhan_scrip_cache": "dhan_scrip_master.csv",  # local cache of scrip master
}


# ----------------------------------------------------------------------------
# UNIVERSE
# ----------------------------------------------------------------------------
def load_universe(path=None):
    """Return a list of NSE symbols (without .NS suffix).

    If a CSV path is given it must have a 'Symbol' column (NSE's own
    equity list format works directly). Otherwise a small built-in
    sample is used so the script runs out-of-the-box.
    """
    if path:
        df = pd.read_csv(path)
        col = next((c for c in df.columns if c.strip().lower() == "symbol"), df.columns[0])
        syms = df[col].astype(str).str.strip().str.upper().tolist()
        return [s for s in syms if s and s != "NAN"]

    # Built-in sample (liquid large/mid caps) so the tool runs immediately.
    # Replace with the full list — see download_full_universe() below.
    return [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
        "SBIN", "LT", "ITC", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
        "HINDUNILVR", "MARUTI", "SUNPHARMA", "TITAN", "ASIANPAINT",
        "NESTLEIND", "ULTRACEMCO", "WIPRO", "ADANIENT", "TMPV", "TMCV",
        "POWERGRID", "NTPC", "TATASTEEL", "JSWSTEEL", "M&M", "HCLTECH",
        "BAJAJFINSV", "DRREDDY", "CIPLA", "GRASIM", "COALINDIA", "BPCL",
        "DIVISLAB", "EICHERMOT", "BRITANNIA", "HEROMOTOCO", "PIDILITIND",
        "DMART", "VEDL", "GODREJCP", "DABUR", "SIEMENS", "HAVELLS",
        "BANKBARODA", "PNB", "TRENT", "PERSISTENT", "POLYCAB",
    ]


def load_universe_all_nse(refresh=False):
    """Return all NSE mainboard cash-equity symbols (SEM_SERIES == 'EQ').

    Sources the list from Dhan's scrip master, excluding SME, trade-for-trade,
    govt securities, ETFs and other non-rolling segments.
    """
    import os
    cache = CONFIG["dhan_scrip_cache"]
    if refresh or not os.path.exists(cache):
        if dhanhq is None:
            print("ERROR: dhanhq not installed. Run: pip install dhanhq")
            sys.exit(1)
        print("  fetching Dhan scrip master ...")
        master = pd.read_csv(dhanhq.COMPACT_CSV_URL, low_memory=False)
        master.to_csv(cache, index=False)
    else:
        master = pd.read_csv(cache, low_memory=False)

    eq = master[
        (master["SEM_EXM_EXCH_ID"] == "NSE")
        & (master["SEM_INSTRUMENT_NAME"] == "EQUITY")
        & (master["SEM_SERIES"].astype(str).str.upper() == "EQ")
        # 'ES' = equity shares; excludes ETF / MF / liquid-fund instruments
        # that NSE also files under the EQUITY/EQ classification.
        & (master["SEM_EXCH_INSTRUMENT_TYPE"].astype(str).str.upper() == "ES")
    ]
    syms = sorted(eq["SEM_TRADING_SYMBOL"].astype(str).str.strip().str.upper().unique())
    return [s for s in syms if s and s != "NAN"]


def yahoo_ticker(symbol):
    """NSE symbol -> Yahoo Finance ticker."""
    return f"{symbol.strip().upper()}.NS"


# ----------------------------------------------------------------------------
# PHASE 1 — TECHNICAL TREND TEMPLATE
# ----------------------------------------------------------------------------
def compute_technicals(df):
    """Given a single stock's OHLCV DataFrame, compute indicators.

    Returns a dict of the metrics needed for Phase 1 and Phase 3,
    or None if there is not enough history.
    """
    if df is None or len(df) < CONFIG["min_history_days"]:
        return None

    close = df["Close"].dropna()
    vol = df["Volume"].dropna()
    if len(close) < CONFIG["min_history_days"]:
        return None

    price = float(close.iloc[-1])
    dma50 = float(close.rolling(50).mean().iloc[-1])
    dma150 = float(close.rolling(150).mean().iloc[-1])
    dma200 = float(close.rolling(200).mean().iloc[-1])

    avg_vol = float(vol.rolling(CONFIG["vol_avg_period"]).mean().iloc[-1])
    today_vol = float(vol.iloc[-1])
    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0

    # Avg daily traded value in crores (liquidity proxy): price x avg volume.
    turnover_cr = (price * avg_vol) / 1e7

    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    pct_from_high = (high_52w - price) / high_52w if high_52w > 0 else 1.0

    # 200DMA slope (trending up?) — compare to ~1 month ago
    dma200_series = close.rolling(200).mean()
    dma200_prev = float(dma200_series.iloc[-22]) if len(dma200_series) >= 222 else dma200
    dma200_rising = dma200 > dma200_prev

    # Relative strength proxy: price change over ~6 months (126 trading days)
    if len(close) >= 126:
        rs_raw = (price / float(close.iloc[-126]) - 1.0) * 100
    else:
        rs_raw = (price / float(close.iloc[0]) - 1.0) * 100

    # Today's move (for advance/decline breadth)
    day_change = (price / float(close.iloc[-2]) - 1.0) * 100 if len(close) >= 2 else 0.0

    return {
        "price": price,
        "dma50": dma50,
        "dma150": dma150,
        "dma200": dma200,
        "dma200_rising": dma200_rising,
        "avg_vol": avg_vol,
        "vol_ratio": vol_ratio,
        "turnover_cr": turnover_cr,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": pct_from_high,
        "rs_raw": rs_raw,
        "day_change": day_change,
    }


def detect_base_breakout(df):
    """Locate the most recent base -> pivot -> breakout and measure how far
    price has travelled from it. This answers "where did the rally start?".

    A *base* is a tight consolidation (high-low range under bb_max_base_depth).
    The *pivot* is the base's resistance (its highest high). A *breakout* is the
    first close above that pivot. Returns a dict of pivot metrics + a `setup`
    label, or None if there isn't enough data.
    """
    if df is None or len(df) < 60:
        return None
    c = CONFIG
    d = df.tail(c["bb_lookback"]).reset_index(drop=True)
    close = d["Close"].astype(float)
    high = d["High"].astype(float) if "High" in d else close
    low = d["Low"].astype(float) if "Low" in d else close
    vol = d["Volume"].astype(float) if "Volume" in d else None
    n = len(close)
    cur = float(close.iloc[-1])
    avg_vol = float(vol.tail(50).mean()) if vol is not None else float("nan")

    # --- Find the most recent confirmed breakout across candidate base lengths
    best_t, best = -1, None
    for B in c["bb_base_lens"]:
        if n < B + 3:
            continue
        base_hi = high.rolling(B).max().shift(1)
        base_lo = low.rolling(B).min().shift(1)
        depth = (base_hi - base_lo) / base_lo
        crossed = (close > base_hi) & (close.shift(1) <= base_hi) \
            & (depth <= c["bb_max_base_depth"]) & (base_lo > 0)
        idxs = np.where(crossed.fillna(False).values)[0]
        if len(idxs):
            t = int(idxs[-1])
            if t > best_t:
                volr = (float(vol.iloc[t]) / avg_vol
                        if vol is not None and avg_vol > 0 else float("nan"))
                best_t, best = t, {
                    "pivot_price": float(base_hi.iloc[t]),
                    "pivot_low": float(base_lo.iloc[t]),
                    "base_depth_pct": float(depth.iloc[t] * 100),
                    "base_len": B,
                    "breakout_vol_ratio": volr,
                }

    out = {
        "pivot_price": np.nan, "pivot_low": np.nan, "base_depth_pct": np.nan,
        "base_len": np.nan, "breakout_vol_ratio": np.nan,
        "days_since_breakout": np.nan, "pct_from_pivot": np.nan,
        "gain_from_base_low": np.nan, "setup": "none",
    }

    if best is not None:
        days_since = n - 1 - best_t
        pct_from_pivot = (cur - best["pivot_price"]) / best["pivot_price"] * 100
        out.update(best)
        out["days_since_breakout"] = int(days_since)
        out["pct_from_pivot"] = float(pct_from_pivot)
        if best["pivot_low"] > 0:
            out["gain_from_base_low"] = (cur - best["pivot_low"]) / best["pivot_low"] * 100
        # classify how late we are relative to the breakout
        if pct_from_pivot > c["bb_extended_pct"] * 100:
            out["setup"] = "extended"
        elif (days_since <= c["bb_fresh_days"]
              and 0.0 <= pct_from_pivot <= c["bb_buy_zone_pct"] * 100):
            out["setup"] = "breakout"        # fresh + still above/at pivot = actionable
        else:
            # includes failed breakouts that fell back below the pivot
            out["setup"] = "in_trend"

    # --- Pre-breakout: currently coiling tightly just under a resistance line
    B = c["bb_base_lens"][1]
    if n >= B + 1:
        res = float(high.iloc[-B:-1].max())
        flo = float(low.iloc[-B:-1].min())
        if res > 0 and flo > 0:
            depth = (res - flo) / flo
            dist = (cur - res) / res  # negative => below resistance
            tight = depth <= c["bb_max_base_depth"]
            below = -c["bb_near_pivot_pct"] <= dist <= 0.0
            if tight and below and out["setup"] in ("none", "in_trend"):
                out["setup"] = "pre_breakout"
                out["pivot_price"] = res
                out["pivot_low"] = flo
                out["base_depth_pct"] = depth * 100
                out["pct_from_pivot"] = dist * 100  # negative
                out["days_since_breakout"] = 0
    return out


def analyze(df):
    """Full per-stock analysis: trend/volume technicals + base/pivot metrics."""
    t = compute_technicals(df)
    if t is None:
        return None
    bb = detect_base_breakout(df)
    if bb:
        t.update(bb)
    return t


def passes_breakout(t):
    """Prospective 'Base Breakout' filter: catch stocks AT the launch point
    (fresh breakout or coiling just under the pivot), not yet extended, and
    liquid enough to trade."""
    if t is None:
        return False
    c = CONFIG
    if t["price"] < c["min_price"] or t.get("turnover_cr", 0.0) < c["min_turnover_cr"]:
        return False
    # Trend-health gate: a base breakout in a healthy context, not a
    # falling-knife dead-cat bounce. Require price above the 200DMA.
    if t["price"] <= t.get("dma200", float("inf")):
        return False
    return t.get("setup") in ("breakout", "pre_breakout")


def passes_phase1(t):
    """Apply the Phase 1 conditions to a technicals dict."""
    if t is None:
        return False
    c = CONFIG
    return (
        t["price"] > t["dma50"]
        and t["dma50"] > t["dma150"]
        and t["dma150"] > t["dma200"]
        and t["vol_ratio"] > c["vol_multiple"]
        and t["pct_from_high"] <= c["near_high_pct"]
        and t["price"] >= c["min_price"]
        and t.get("turnover_cr", 0.0) >= c["min_turnover_cr"]
    )


def fetch_prices(symbols):
    """Download daily history for all symbols. Returns {symbol: technicals}."""
    if yf is None:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    results = {}
    tickers = [yahoo_ticker(s) for s in symbols]
    batch = CONFIG["batch_size"]

    for i in range(0, len(tickers), batch):
        chunk_syms = symbols[i:i + batch]
        chunk_tk = tickers[i:i + batch]
        print(f"  downloading {i+1}-{i+len(chunk_tk)} of {len(tickers)} ...")
        try:
            data = yf.download(
                chunk_tk, period="2y", interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=True,
            )
        except Exception as e:
            print(f"   batch failed: {e}")
            time.sleep(CONFIG["sleep_between"])
            continue

        for sym, tk in zip(chunk_syms, chunk_tk):
            try:
                sub = data[tk] if len(chunk_tk) > 1 else data
                t = analyze(sub)
                if t:
                    results[sym] = t
            except Exception:
                continue
        time.sleep(CONFIG["sleep_between"])

    return results


# ----------------------------------------------------------------------------
# DATA SOURCE — DhanHQ (historical daily candles)
# ----------------------------------------------------------------------------
def _dhan_client():
    """Build an authenticated DhanHQ client from env vars."""
    import os
    if dhanhq is None:
        print("ERROR: dhanhq not installed. Run: pip install dhanhq")
        sys.exit(1)
    cid = os.getenv("DHAN_CLIENT_ID")
    tok = os.getenv("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        print("ERROR: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables.")
        sys.exit(1)
    return dhanhq(cid, tok)


def load_dhan_security_map(refresh=False):
    """Return {NSE symbol -> security_id} for NSE cash-equity scrips.

    Downloads Dhan's compact scrip master (and caches it locally) so symbols
    can be resolved to the numeric security_id the historical API needs.
    """
    import os
    cache = CONFIG["dhan_scrip_cache"]
    if refresh or not os.path.exists(cache):
        print("  fetching Dhan scrip master ...")
        master = pd.read_csv(dhanhq.COMPACT_CSV_URL, low_memory=False)
        master.to_csv(cache, index=False)
    else:
        master = pd.read_csv(cache, low_memory=False)

    eq = master[
        (master["SEM_EXM_EXCH_ID"] == "NSE")
        & (master["SEM_INSTRUMENT_NAME"] == "EQUITY")
    ]
    out = {}
    for _, r in eq.iterrows():
        sym = str(r["SEM_TRADING_SYMBOL"]).strip().upper()
        if sym and sym != "NAN":
            out[sym] = str(r["SEM_SMST_SECURITY_ID"]).strip()
    return out


def _dhan_to_dataframe(data):
    """Convert Dhan's historical_daily_data payload to an OHLCV DataFrame
    with the Close/Volume columns compute_technicals() expects."""
    if not isinstance(data, dict):
        return None
    # Dhan returns parallel arrays keyed by ohlcv names (lower-case).
    keys = {k.lower(): k for k in data.keys()}
    close_k = keys.get("close")
    vol_k = keys.get("volume")
    if not close_k or not data.get(close_k):
        return None
    df = pd.DataFrame({
        "Open": data.get(keys.get("open", ""), []),
        "High": data.get(keys.get("high", ""), []),
        "Low": data.get(keys.get("low", ""), []),
        "Close": data[close_k],
        "Volume": data.get(vol_k, []),
    })
    # Attach a DatetimeIndex if a timestamp array is present (epoch seconds).
    ts_k = keys.get("timestamp") or keys.get("start_time")
    if ts_k and data.get(ts_k):
        try:
            df.index = pd.to_datetime(data[ts_k], unit="s")
        except (ValueError, TypeError):
            pass
    return df


def fetch_prices_dhan(symbols):
    """Download daily history via DhanHQ. Returns {symbol: technicals}."""
    from datetime import timedelta

    d = _dhan_client()
    secmap = load_dhan_security_map()
    print(f"  scrip master: {len(secmap)} NSE equity symbols mapped")

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now()
                 - timedelta(days=CONFIG["dhan_history_days"])).strftime("%Y-%m-%d")

    results = {}
    missing = []
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        sid = secmap.get(sym.strip().upper())
        if not sid:
            missing.append(sym)
            continue
        if i % 25 == 0 or i == n:
            print(f"  downloading {i}/{n} ...")
        try:
            r = d.historical_daily_data(sid, dhanhq.NSE, "EQUITY", from_date, to_date)
            if r.get("status") != "success":
                # Surface auth/rate errors clearly instead of silently skipping.
                rem = r.get("remarks", r.get("data"))
                if isinstance(rem, dict) and rem.get("error_type") == "Invalid_Authentication":
                    print(f"\nERROR: DhanHQ auth failed (DH-901) - access token "
                          f"invalid/expired. Refresh DHAN_ACCESS_TOKEN.\n  {rem}")
                    sys.exit(1)
                continue
            df = _dhan_to_dataframe(r.get("data"))
            t = analyze(df)
            if t:
                results[sym] = t
        except SystemExit:
            raise
        except Exception:
            pass
        time.sleep(CONFIG["dhan_sleep"])

    if missing:
        print(f"  note: {len(missing)} symbol(s) not found in Dhan scrip master: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    return results


# ----------------------------------------------------------------------------
# PHASE 2 — FUNDAMENTAL QUALITY (from Screener.in CSV)
# ----------------------------------------------------------------------------
_CORP_STOPWORDS = (
    " LIMITED", " LTD", " PVT", " PRIVATE", " CORPORATION", " CORP",
    " COMPANY", " INDUSTRIES", " INDUSTRY", " ENTERPRISES", " HOLDINGS",
)


_CONNECTOR_TOKENS = {"AND", "THE", "OF"}


def _norm_name(s):
    """Normalise a company name for fuzzy matching."""
    import re
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)          # drop &, ., - etc.
    for w in _CORP_STOPWORDS:
        s = s.replace(w, " ")
    toks = [t for t in s.split() if t not in _CONNECTOR_TOKENS]
    return " ".join(toks)


def _find_header_row(rows):
    """Index of the first row that looks like a screener header
    (contains a 'Name' cell or an 'S.No' cell). Defaults to 0."""
    for i, cells in enumerate(rows[:10]):
        low = [str(x).strip().lower() for x in cells]
        if any(c == "name" for c in low) or any("s.no" in c for c in low):
            return i
    return 0


def _read_screener_table(path):
    """Read a screener.in export (.csv or .xlsx), skipping any title/blank
    rows so the real header row (the one containing 'Name') becomes columns.

    Screener exports often have a title line with fewer fields than the
    data rows, so we locate the header explicitly before parsing."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None)
        hdr = _find_header_row(raw.values.tolist())
        return pd.read_excel(path, header=hdr)
    # CSV: use the csv module for header detection (handles quoted commas).
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    hdr = _find_header_row(rows)
    return pd.read_csv(path, skiprows=hdr)


def _build_name_index():
    """Return {normalised company name -> NSE symbol} from the Dhan master,
    restricted to mainboard equity shares. Empty if the master isn't cached."""
    import os
    cache = CONFIG["dhan_scrip_cache"]
    if not os.path.exists(cache):
        return {}
    m = pd.read_csv(cache, low_memory=False)
    eq = m[(m["SEM_EXM_EXCH_ID"] == "NSE")
           & (m["SEM_INSTRUMENT_NAME"] == "EQUITY")
           & (m["SEM_EXCH_INSTRUMENT_TYPE"].astype(str).str.upper() == "ES")]
    idx = {}
    for _, r in eq.iterrows():
        sym = str(r["SEM_TRADING_SYMBOL"]).strip().upper()
        nm = _norm_name(r.get("SM_SYMBOL_NAME", ""))
        if nm:
            idx.setdefault(nm, sym)
    return idx


def _resolve_symbol(name, name_index):
    """Map a screener company name to an NSE symbol via the name index.
    Strategy: exact-normalised -> token-prefix (handles screener truncation)
    -> difflib fuzzy >= 0.86. Returns symbol or None."""
    import difflib
    q = _norm_name(name)
    if not q:
        return None
    if q in name_index:
        return name_index[q]
    qtok = q.split()
    # token-prefix: align leading tokens, tolerating truncation on EITHER side
    # (both the screener export and the Dhan master abbreviate long names).
    best = None
    for nm, sym in name_index.items():
        mtok = nm.split()
        k = min(len(qtok), len(mtok))
        if k == 0:
            continue
        if all(mtok[i].startswith(qtok[i]) or qtok[i].startswith(mtok[i])
               for i in range(k)):
            # prefer the longest token-overlap match
            if best is None or k > best[0]:
                best = (k, sym)
    if best:
        return best[1]
    match = difflib.get_close_matches(q, list(name_index.keys()), n=1, cutoff=0.86)
    return name_index[match[0]] if match else None


def load_fundamentals(path):
    """Load a fundamentals export from screener.in (.csv or .xlsx).

    Resolves each row to an NSE symbol using an explicit symbol/NSE-code
    column if present, otherwise by matching the company Name against the
    Dhan scrip master.

    Returns {symbol: {roce, sales_growth, profit_growth, debt_equity, mcap_cr}}
    (empty if the file is missing).
    """
    import os
    if not path or not os.path.exists(path):
        return {}

    df = _read_screener_table(path)
    cols = {str(c).lower().strip(): c for c in df.columns}

    def find(*subs):
        # exact key first, then substring contains
        for s in subs:
            if s in cols:
                return cols[s]
        for s in subs:
            for k, orig in cols.items():
                if s in k:
                    return orig
        return None

    c_sym = find("nse code", "symbol", "ticker", "bse code", "code")
    c_name = find("name")
    c_roce = find("roce", "return on capital")
    c_sales = find("sales growth 3", "sales growth", "sales var", "sales gr")
    c_profit = find("profit growth 3", "profit growth", "profit var", "profit gr")
    c_de = find("debt to equity", "debt / equity", "debt/equity", "d/e")
    c_mcap = find("mar cap", "market cap", "market capital", "mcap")

    name_index = _build_name_index() if (c_sym is None and c_name) else {}

    def num(row, col):
        if col is None:
            return np.nan
        try:
            return float(str(row[col]).replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return np.nan

    out, unmatched = {}, []
    for _, row in df.iterrows():
        if c_sym is not None:
            sym = str(row[c_sym]).strip().upper()
        elif c_name is not None:
            sym = _resolve_symbol(row[c_name], name_index)
            if not sym:
                nm = str(row[c_name]).strip()
                if nm and nm.lower() != "nan":
                    unmatched.append(nm)
                continue
        else:
            continue
        if not sym or sym == "NAN":
            continue
        out[sym] = {
            "roce": num(row, c_roce),
            "sales_growth": num(row, c_sales),
            "profit_growth": num(row, c_profit),
            "debt_equity": num(row, c_de),
            "mcap_cr": num(row, c_mcap),
        }

    matched_by = "symbol column" if c_sym is not None else "company name"
    print(f"  fundamentals: parsed {len(out)} rows (matched by {matched_by})")
    if unmatched:
        print(f"  fundamentals: {len(unmatched)} name(s) could not be matched to an "
              f"NSE symbol: {', '.join(unmatched[:8])}{' ...' if len(unmatched) > 8 else ''}")
    return out


def passes_phase2(f):
    """Apply Phase 2 quality filters. Missing data -> fails (conservative)."""
    if not f:
        return False
    c = CONFIG
    try:
        return (
            f["roce"] >= c["min_roce"]
            and f["sales_growth"] >= c["min_sales_growth"]
            and f["profit_growth"] >= c["min_profit_growth"]
            and f["debt_equity"] <= c["max_debt_equity"]
            and f["mcap_cr"] >= c["min_mcap_cr"]
        )
    except (TypeError, KeyError):
        return False


def phase2_report(p1, funds):
    """Print a per-criterion breakdown so output quality can be judged:
    how many Phase 1 candidates have fundamentals, and how many clear each
    individual threshold."""
    c = CONFIG
    n = len(p1)
    have = [s for s in p1 if s in funds]
    no_data = [s for s in p1 if s not in funds]

    def cnt(key, ok):
        return sum(1 for s in have if ok(funds[s].get(key)))

    import math
    def ge(thr):
        return lambda v: v is not None and not math.isnan(v) and v >= thr
    def le(thr):
        return lambda v: v is not None and not math.isnan(v) and v <= thr

    print(f"  coverage: {len(have)}/{n} candidates have fundamentals "
          f"({len(no_data)} missing -> auto-fail)")
    if no_data:
        print(f"    missing: {', '.join(no_data[:12])}"
              f"{' ...' if len(no_data) > 12 else ''}")
    print(f"  pass per criterion (of {len(have)} with data):")
    print(f"    ROCE>={c['min_roce']:<5}      {cnt('roce', ge(c['min_roce']))}")
    print(f"    Sales3Y>={c['min_sales_growth']:<5}  {cnt('sales_growth', ge(c['min_sales_growth']))}")
    print(f"    Profit3Y>={c['min_profit_growth']:<5} {cnt('profit_growth', ge(c['min_profit_growth']))}")
    print(f"    D/E<={c['max_debt_equity']:<6}     {cnt('debt_equity', le(c['max_debt_equity']))}")
    print(f"    MCap>={c['min_mcap_cr']:<6}Cr  {cnt('mcap_cr', ge(c['min_mcap_cr']))}")


# ----------------------------------------------------------------------------
# PHASE 3 — RANKING SCORE
# ----------------------------------------------------------------------------
def _pct_rank(series):
    """Percentile rank 0-100 across the surviving set."""
    return series.rank(pct=True) * 100.0


def build_ranking(rows):
    """rows: list of dicts with technicals + fundamentals merged.
    Returns a ranked DataFrame with a composite Score (0-100).
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    c = CONFIG

    # Component 1: Relative strength (price momentum) — percentile of rs_raw
    df["score_rs"] = _pct_rank(df["rs_raw"])

    # Component 2: Volume surge — percentile of vol_ratio
    df["score_vol"] = _pct_rank(df["vol_ratio"])

    # Component 3: Earnings growth — percentile of profit_growth
    # (falls back to 50 if fundamentals absent)
    if "profit_growth" in df and df["profit_growth"].notna().any():
        df["score_earn"] = _pct_rank(df["profit_growth"].fillna(df["profit_growth"].min()))
    else:
        df["score_earn"] = 50.0

    # Component 4: Technical trend strength —
    # how far price sits above the stacked MAs + 200DMA rising bonus
    spread = ((df["price"] - df["dma200"]) / df["dma200"] * 100).clip(lower=0)
    df["score_tech"] = _pct_rank(spread) * 0.8 + df["dma200_rising"].astype(float) * 20

    df["Score"] = (
        c["w_rel_strength"] * df["score_rs"]
        + c["w_volume"] * df["score_vol"]
        + c["w_earnings"] * df["score_earn"]
        + c["w_tech_trend"] * df["score_tech"]
    ).round(2)

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def build_breakout_ranking(rows):
    """Rank base-breakout candidates. Rewards: tight base, volume confirmation,
    proximity to the pivot (not extended), and relative strength. Fresh
    breakouts rank above pre-breakouts."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # tighter base is better -> invert depth
    df["score_tight"] = _pct_rank(-df["base_depth_pct"].fillna(df["base_depth_pct"].max()))
    # closer to the pivot is better (small |extension|)
    df["score_prox"] = _pct_rank(-df["pct_from_pivot"].abs().fillna(99))
    # volume confirmation on the breakout (fall back to today's vol surge)
    volc = df.get("breakout_vol_ratio")
    if volc is None:
        volc = df["vol_ratio"]
    df["score_volc"] = _pct_rank(volc.fillna(df["vol_ratio"]).fillna(0))
    df["score_rs"] = _pct_rank(df["rs_raw"])

    setup_bonus = df["setup"].map({"breakout": 8.0, "pre_breakout": 0.0}).fillna(0.0)
    df["Score"] = (
        0.30 * df["score_rs"]
        + 0.25 * df["score_tight"]
        + 0.25 * df["score_volc"]
        + 0.20 * df["score_prox"]
        + setup_bonus
    ).round(2)

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def apply_fundamentals(cands, args):
    """Phase 2A fundamental pass/fail filter, shared by both modes.
    Returns (survivors_dict, funds_dict)."""
    if args.no_fundamentals:
        return cands, {}
    c = CONFIG
    print(f"\n[Phase 2A] Fundamental filter "
          f"(ROCE>={c['min_roce']}, sales3Y>={c['min_sales_growth']}, "
          f"profit3Y>={c['min_profit_growth']}, D/E<={c['max_debt_equity']}, "
          f"mcap>={c['min_mcap_cr']}Cr) ...")
    funds = load_fundamentals(args.fundamentals)
    if not funds:
        print(f"[Phase 2A] No fundamentals file at '{args.fundamentals}' — skipping. "
              "Export one from screener.in (see README) or use --no-fundamentals.")
        return cands, funds
    phase2_report(cands, funds)
    survivors = {s: t for s, t in cands.items() if passes_phase2(funds.get(s))}
    print(f"[Phase 2A] {len(survivors)} of {len(cands)} candidates passed")
    return survivors, funds


# ----------------------------------------------------------------------------
# MARKET BREADTH — the regime that governs breakout success (per research)
# ----------------------------------------------------------------------------
def compute_breadth(tech):
    """Market-wide breadth from the already-fetched universe (no extra calls).
    Returns a dict, or None if the sample is too small to be meaningful."""
    vals = [t for t in tech.values() if t]
    n = len(vals)
    if n < CONFIG["breadth_min_sample"]:
        return None
    above200 = sum(1 for t in vals if t["price"] > t.get("dma200", 1e18))
    above50 = sum(1 for t in vals if t["price"] > t.get("dma50", 1e18))
    new_hi = sum(1 for t in vals if t.get("pct_from_high", 1.0) <= 0.001)
    new_lo = sum(1 for t in vals
                 if t.get("low_52w", 0) > 0 and t["price"] <= t["low_52w"] * 1.001)
    adv = sum(1 for t in vals if t.get("day_change", 0.0) > 0)
    dec = sum(1 for t in vals if t.get("day_change", 0.0) < 0)

    pa200 = above200 / n * 100
    if pa200 < CONFIG["breadth_riskoff"]:
        state = "RISK-OFF"
    elif pa200 < CONFIG["breadth_riskon"]:
        state = "NEUTRAL"
    else:
        state = "RISK-ON"
    return {
        "n": n, "pct_above_200": pa200, "pct_above_50": above50 / n * 100,
        "new_highs": new_hi, "new_lows": new_lo, "adv": adv, "dec": dec,
        "ad_ratio": (adv / dec if dec else float("inf")), "state": state,
    }


def print_breadth_banner(tech):
    """Print the market-state banner and return the breadth dict (or None).
    Research finding: breakouts win ~22% when breadth>60% vs ~13% below — and
    relative strength compounds with it (RS + healthy breadth ~1.4x base rate)."""
    b = compute_breadth(tech)
    print("\n" + "-" * 60)
    if b is None:
        print("MARKET STATE: n/a (need a larger universe; use --all-nse)")
        print("-" * 60)
        return None
    print(f"MARKET STATE: {b['state']}   (breadth {b['pct_above_200']:.0f}% > 200DMA, "
          f"{b['pct_above_50']:.0f}% > 50DMA, sample {b['n']})")
    print(f"  52w highs: {b['new_highs']}   52w lows: {b['new_lows']}   "
          f"adv/dec: {b['adv']}/{b['dec']} ({b['ad_ratio']:.2f})")
    if b["state"] == "RISK-OFF":
        print("  >> Weak breadth: breakouts historically win ~13% here. "
              "Treat candidates as WATCHLIST, not new positions.")
    elif b["state"] == "RISK-ON":
        print("  >> Healthy breadth: the regime that favours breakouts (~22% win rate).")
    print("-" * 60)
    return b


def _dhan_nifty_return(window=None):
    """Trailing return of NIFTY 50 via Dhan, for relative-strength calc."""
    from datetime import timedelta
    window = window or CONFIG["rs_window"]
    d = _dhan_client()
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now()
                 - timedelta(days=int((window + 40) * 1.7))).strftime("%Y-%m-%d")
    try:
        r = d.historical_daily_data(CONFIG["nifty_secid"], "IDX_I", "INDEX",
                                    from_date, to_date)
        if r.get("status") == "success":
            c = r["data"].get("close") or []
            if len(c) > window:
                return float(c[-1]) / float(c[-1 - window]) - 1.0
    except Exception:
        pass
    return None


def compute_rs_percentile(tech, nifty_ret):
    """Inject cross-sectional RS-percentile (0-100) vs NIFTY into each technicals
    dict — the IBD-style relative-strength rank the research validated (RS>=80
    + RISK-ON breadth ~ 1.4x base breakout win-rate)."""
    if nifty_ret is None:
        return
    items = [(s, t) for s, t in tech.items() if t and t.get("rs_raw") is not None]
    if len(items) < CONFIG["breadth_min_sample"]:
        return
    rel = np.array([(1 + t["rs_raw"] / 100.0) / (1 + nifty_ret) for _, t in items])
    pct = rel.argsort().argsort().astype(float) / (len(rel) - 1) * 100.0
    for (s, t), p in zip(items, pct):
        t["rs_pctl"] = float(p)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="NSE 3-Phase Momentum Scanner")
    ap.add_argument("--universe", help="CSV with a 'Symbol' column (NSE codes)")
    ap.add_argument("--fundamentals", default="fundamentals.csv",
                    help="Screener.in CSV for Phase 2 (default: fundamentals.csv)")
    ap.add_argument("--no-fundamentals", action="store_true",
                    help="Skip Phase 2; rank on technicals only")
    ap.add_argument("--out", default="scan_results.csv", help="Output CSV path")
    ap.add_argument("--min-vol-multiple", type=float, default=None,
                    help=f"Override Phase 1 volume multiple "
                         f"(default {CONFIG['vol_multiple']})")
    ap.add_argument("--near-high-pct", type=float, default=None,
                    help=f"Override 'within X of 52w high' as fraction 0-1 "
                         f"(default {CONFIG['near_high_pct']})")
    ap.add_argument("--relax", action="store_true",
                    help="Loosen Phase 1: vol_multiple=1.0, near_high_pct=0.25")
    ap.add_argument("--source", choices=["yahoo", "dhan"], default="dhan",
                    help="Price data source (default: dhan)")
    ap.add_argument("--all-nse", action="store_true",
                    help="Scan all NSE mainboard equities (series EQ) from Dhan master")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the universe to the first N symbols (testing)")
    ap.add_argument("--min-price", type=float, default=None,
                    help=f"Min share price in rupees (default {CONFIG['min_price']})")
    ap.add_argument("--min-turnover", type=float, default=None,
                    help=f"Min avg daily turnover in crores "
                         f"(default {CONFIG['min_turnover_cr']})")
    ap.add_argument("--mode", choices=["momentum", "breakout"], default="momentum",
                    help="momentum = trend template (already-rallied); "
                         "breakout = stocks AT the base/pivot launch point")
    ap.add_argument("--buy-zone-pct", type=float, default=None,
                    help=f"breakout mode: max %% above pivot still 'buyable', "
                         f"as fraction (default {CONFIG['bb_buy_zone_pct']})")
    args = ap.parse_args()

    # Apply threshold overrides (explicit flags win over --relax)
    if args.relax:
        CONFIG["vol_multiple"] = 1.0
        CONFIG["near_high_pct"] = 0.25
    if args.min_vol_multiple is not None:
        CONFIG["vol_multiple"] = args.min_vol_multiple
    if args.near_high_pct is not None:
        CONFIG["near_high_pct"] = args.near_high_pct
    if args.min_price is not None:
        CONFIG["min_price"] = args.min_price
    if args.min_turnover is not None:
        CONFIG["min_turnover_cr"] = args.min_turnover
    if args.buy_zone_pct is not None:
        CONFIG["bb_buy_zone_pct"] = args.buy_zone_pct

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n=== NSE {args.mode.title()} Scan @ {stamp} ===")
    print(f"Data source: {args.source} | min_price={CONFIG['min_price']}, "
          f"min_turnover_cr={CONFIG['min_turnover_cr']}")

    if args.all_nse:
        symbols = load_universe_all_nse()
    else:
        symbols = load_universe(args.universe)
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"Universe: {len(symbols)} symbols")

    # ---- Fetch + analyse (trend/volume technicals + base/pivot metrics)
    print(f"\n[Fetch] Downloading & analysing ...")
    tech = fetch_prices_dhan(symbols) if args.source == "dhan" else fetch_prices(symbols)
    if args.source == "dhan":
        compute_rs_percentile(tech, _dhan_nifty_return())

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    if args.mode == "breakout":
        run_breakout(tech, args)
    else:
        run_momentum(tech, args)


def mark_conviction(ranked, breadth):
    """Flag the research-validated high-conviction setup: relative-strength
    leader (RS percentile >= threshold) in a RISK-ON breadth regime."""
    if ranked.empty or "rs_pctl" not in ranked.columns:
        return ranked
    risk_on = bool(breadth) and breadth["state"] == "RISK-ON"
    ranked["conviction"] = np.where(
        (ranked["rs_pctl"] >= CONFIG["rs_strong_pctl"]) & risk_on, "HIGH", "")
    return ranked


def _print_conviction_summary(ranked, breadth):
    if "conviction" not in ranked.columns:
        if "rs_pctl" not in ranked.columns:
            print("\n(RS-percentile/conviction needs the full universe — use --all-nse.)")
        return
    hi = int((ranked["conviction"] == "HIGH").sum())
    thr = int(CONFIG["rs_strong_pctl"])
    if bool(breadth) and breadth["state"] == "RISK-ON":
        print(f"HIGH-conviction (RS>={thr} pctl + RISK-ON breadth): {hi} of {len(ranked)} "
              f"— the ~1.4x edge from the research. Focus here.")
    else:
        state = breadth["state"] if breadth else "n/a"
        print(f"No HIGH-conviction tags: breadth is {state}, not RISK-ON. "
              f"The validated edge only applies in healthy breadth.")


def run_momentum(tech, args):
    """Trend-template screen: price stacked above MAs, near highs, volume surge.
    Output is annotated with where each name rallied from (pivot/extension)."""
    breadth = print_breadth_banner(tech)
    p1 = {s: t for s, t in tech.items() if passes_phase1(t)}
    print(f"\n[Phase 1] {len(p1)} passed (price>50>150>200 DMA, "
          f"vol>{CONFIG['vol_multiple']}x, <{CONFIG['near_high_pct']*100:.0f}% from high)")
    if not p1:
        print("No candidates. Exiting.")
        return

    survivors, funds = apply_fundamentals(p1, args)
    if not survivors:
        print("No candidates after Phase 2A. Exiting.")
        return

    rows = []
    for s, t in survivors.items():
        row = {"Symbol": s, **t}
        if s in funds:
            row.update(funds[s])
        rows.append(row)

    print("\n[Phase 3] Building composite ranking ...")
    ranked = build_ranking(rows)
    ranked = mark_conviction(ranked, breadth)

    # Annotate with where the rally started + whether it's still buyable.
    show_cols = ["Rank", "Symbol", "Score", "conviction", "price", "turnover_cr",
                 "pct_from_high", "rs_pctl", "vol_ratio", "setup", "pct_from_pivot",
                 "days_since_breakout", "pivot_price"]
    show_cols = [c for c in show_cols if c in ranked.columns]
    print("\n" + ranked[show_cols].to_string(index=False))
    if "setup" in ranked:
        early = (ranked["setup"] == "breakout").sum()
        ext = (ranked["setup"] == "extended").sum()
        print(f"\nEntry context: {early} still near pivot (actionable), "
              f"{ext} extended (chasing). See 'pct_from_pivot'.")
    _print_conviction_summary(ranked, breadth)

    ranked.to_csv(args.out, index=False)
    print(f"\nSaved full results -> {args.out}")


def run_breakout(tech, args):
    """Prospective screen: stocks AT the launch point — fresh base breakouts
    or coiling just under the pivot — before they become extended."""
    breadth = print_breadth_banner(tech)
    cands = {s: t for s, t in tech.items() if passes_breakout(t)}
    n_bo = sum(1 for t in cands.values() if t.get("setup") == "breakout")
    n_pre = sum(1 for t in cands.values() if t.get("setup") == "pre_breakout")
    print(f"\n[Breakout] {len(cands)} candidates "
          f"({n_bo} fresh breakouts within {CONFIG['bb_buy_zone_pct']*100:.0f}% of pivot, "
          f"{n_pre} coiling pre-breakout)")
    if not cands:
        print("No base-breakout setups today. Exiting.")
        return

    survivors, funds = apply_fundamentals(cands, args)
    if not survivors:
        print("No candidates after Phase 2A. Exiting.")
        return

    rows = []
    for s, t in survivors.items():
        row = {"Symbol": s, **t}
        if s in funds:
            row.update(funds[s])
        rows.append(row)

    print("\n[Rank] Scoring base-breakout quality ...")
    ranked = build_breakout_ranking(rows)
    ranked = mark_conviction(ranked, breadth)

    show_cols = ["Rank", "Symbol", "Score", "conviction", "setup", "price",
                 "pivot_price", "pct_from_pivot", "days_since_breakout",
                 "base_depth_pct", "rs_pctl", "breakout_vol_ratio", "turnover_cr"]
    show_cols = [c for c in show_cols if c in ranked.columns]
    print("\n" + ranked[show_cols].to_string(index=False))
    _print_conviction_summary(ranked, breadth)

    ranked.to_csv(args.out, index=False)
    print(f"\nSaved full results -> {args.out}")


if __name__ == "__main__":
    main()
