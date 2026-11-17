#!/usr/bin/env python
"""
Rebuild every event-window return in data/features.csv against the S&P 500 TOTAL return, from the 2008
daily CRSP pull, and add the CAR[-1,+1] reaction window (AR_M11).

  AR(i,t) = dlyret(i,t) - vwretd(S&P 500, t)       crsp_dsf_v2 / crsp_dsp500_v2; point-in-time index,
                                                    dlyret already includes delisting returns
  AR_M11  = CAR[-1,+1]   AR01 = CAR[0,+1]   CAR_2_5, CAR_2_20, CAR_2_60   (CAR_6_20, CAR_21_60 by difference)

Why total return, not the price return (sprtrn) the legacy panel used: dlyret includes dividends, so
subtracting a price-only index return leaves the index dividend yield inside every AR. Each CAR then
carries an upward bias of roughly yield x days / 252 (about +0.18 pp over [+2,+20] here, +0.5 pp over
[+2,+60]) that has nothing to do with the firm: it inflates the "long" drift, makes a zero forecast look
wrong, and grows with the horizon, so horizons are not comparable. Total return on both sides removes it.
It barely moves cross-sectional ranks (the shift is the same for all firms on a date), so Rank IC is
nearly unchanged; levels, OOS R^2 vs zero, and backtest P&L are what it fixes. The S&P 500 hedge in the
backtest also earns dividends (futures or an ETF), so total return matches the trade.

Day 0 is taken from data/features.csv, so windows line up with the panel. A Day 0 that is not a trading
day rolls forward. A window needs every session with a return, otherwise it is missing.

Checks (prefix px_, never used): the same windows against sprtrn, compared with the panel's AR01 and CAR_2_h
(they match to ~1e-6, which shows the panel used sprtrn). LOGVOL_2_20 stays the panel's: a benchmark shift
that is nearly constant within the window barely changes a volatility.

Output: data/reaction_windows.csv. common.load_events replaces the panel's AR01 and CAR_* with these
total-return versions (keeping the panel's as <name>_PX) and adds AR_M11.

Usage: python scripts/build_reaction_windows.py [--raw data/raw/sp500_pit_2008]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

log = C.get_logger("reaction")
WINDOWS = {"AR_M11": (-1, 1), "AR01": (0, 1), "CAR_2_5": (2, 5), "CAR_2_20": (2, 20), "CAR_2_60": (2, 60)}
PANEL = ["AR01", "CAR_2_5", "CAR_2_20", "CAR_2_60", "CAR_6_20", "CAR_21_60"]   # rebuilt on total return
REPO = Path(__file__).resolve().parents[1]


def load_daily(raw: Path, tickers: set[str], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / "crsp_dsf_v2_*.parquet")))
    if not files:
        raise SystemExit(f"no crsp_dsf_v2_*.parquet in {raw}")
    cols = ["permno", "dlycaldt", "dlyret", "ticker"]
    keep = []
    for f in files:
        year = int(Path(f).stem.rsplit("_", 1)[1])
        if year < first.year - 1 or year > last.year + 1:
            continue
        d = pd.read_parquet(f, columns=cols)
        keep.append(d[d["ticker"].isin(tickers)])
    d = pd.concat(keep, ignore_index=True)
    d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
    return d


def permno_map(daily: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    """Ticker -> PERMNO on each event's Day 0 (ticker as CRSP reported it that day)."""
    t = daily[["ticker", "dlycaldt", "permno"]].drop_duplicates().astype({"ticker": str})
    e = ev[["transcriptid", "ticker", "day0"]].astype({"ticker": str})
    m = pd.merge_asof(e.sort_values("day0"),
                      t.sort_values("dlycaldt"), left_on="day0", right_on="dlycaldt", by="ticker",
                      direction="nearest", tolerance=pd.Timedelta("10D"))
    return m.set_index("transcriptid")["permno"]


def window_sum(ar: pd.DataFrame, pos0: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Sum of AR over positions pos0+lo .. pos0+hi of one stock's series; NaN if any return is missing."""
    v = ar.to_numpy(float)
    out = np.full(len(pos0), np.nan)
    for i, p in enumerate(pos0):
        a, b = p + lo, p + hi
        if p >= 0 and a >= 0 and b < len(v):
            w = v[a:b + 1]
            if not np.isnan(w).any():
                out[i] = w.sum()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "data" / "raw" / "sp500_pit_2008"))
    ap.add_argument("--out", default=str(C.REACTION_PATH))
    args = ap.parse_args()
    raw = Path(args.raw)

    ev = pd.read_csv(C.FEATURES_PATH, usecols=["transcriptid", "ticker", "day0"] + PANEL, parse_dates=["day0"])
    daily = load_daily(raw, set(ev["ticker"]), ev["day0"].min(), ev["day0"].max())
    idx = pd.read_parquet(raw / "crsp_dsp500_v2.parquet", columns=["caldt", "vwretd", "sprtrn"])
    idx["caldt"] = pd.to_datetime(idx["caldt"])
    sessions = pd.DatetimeIndex(idx["caldt"].sort_values().unique())

    ev["permno"] = ev["transcriptid"].map(permno_map(daily, ev)).astype("Int64")
    # Day 0 on the trading calendar: roll a non-trading day forward
    ev["day0_session"] = sessions[np.minimum(sessions.searchsorted(ev["day0"], side="left"), len(sessions) - 1)]
    rolled = (ev["day0_session"] != ev["day0"]).sum()

    out = []
    for permno, g in ev.dropna(subset=["permno"]).groupby("permno"):
        st = daily[daily.permno == permno].set_index("dlycaldt")["dlyret"]
        st = st[~st.index.duplicated()].reindex(sessions)                     # missing day -> NaN
        bench = idx.set_index("caldt").reindex(sessions)
        pos = sessions.get_indexer(g["day0_session"])
        cols = {}
        for prefix, b in (("", "vwretd"), ("px_", "sprtrn")):
            ar = st - bench[b]
            for name, (lo, hi) in WINDOWS.items():
                cols[prefix + name] = window_sum(ar, pos, lo, hi)
        out.append(g[["transcriptid", "permno", "day0"]].assign(**cols))
    res = pd.concat(out, ignore_index=True)
    for p in ("", "px_"):
        res[p + "CAR_6_20"] = res[p + "CAR_2_20"] - res[p + "CAR_2_5"]
        res[p + "CAR_21_60"] = res[p + "CAR_2_60"] - res[p + "CAR_2_20"]

    m = res.merge(ev[["transcriptid"] + PANEL], on="transcriptid", suffixes=("", "_panel"))
    for c in PANEL:
        px = (m["px_" + c] - m[c + "_panel"]).abs()
        tr = m[c] - m[c + "_panel"]
        log.info("%-9s n=%d | price-return rebuild vs panel: max |diff| %.6f | total-return minus panel: "
                 "mean %+.4f pp", c, tr.notna().sum(), px.max(), tr.mean() * 100)
    log.info("events %d | no PERMNO %d | Day 0 rolled %d | missing: %s", len(ev), ev["permno"].isna().sum(),
             rolled, {c: int(res[c].isna().sum()) for c in ["AR_M11"] + PANEL})

    keep = ["transcriptid", "permno", "day0", "AR_M11"] + PANEL + ["px_AR_M11"] + ["px_" + c for c in PANEL]
    res[keep].sort_values("day0").to_csv(args.out, index=False)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
