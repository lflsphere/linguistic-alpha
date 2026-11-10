"""Firm characteristics: size, book-to-market, momentum, pre-event volatility, CAR[0,+1].

Implemented from CRSP daily data and Compustat quarterly fundamentals.

Definitions:
  size    = ln(|prc| * shrout * 1000) at the last trading day of the month before the call
  bm      = book equity / market cap, book equity from the latest fundq with rdq < Day 0;
            BE = seqq + txditcq - pstkq (fallback ceqq); negative BE -> NaN
  mom     = prod(1+ret) over months -12..-2 relative to the call month
  vol_pre = std(AR, -60..-5) * sqrt(252)
  car_0_1 = sum AR over Day 0 and Day +1 (control, not target)

Fixes vs. the legacy notebook (docs/data_audit.md §Controls): missing returns inside the
momentum window are not filled with 0; fundq rows with NULL rdq use datadate + 90 days lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_dates(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()


def firm_controls(events: pd.DataFrame, dsf: pd.DataFrame, fundq: pd.DataFrame,
                  ar: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time size, B/M, momentum, announcement CAR and pre-event volatility."""
    e = events.copy()
    s = dsf.copy()
    f = fundq.copy()
    a = ar.copy()
    _normalize_dates(s, ["date"])
    _normalize_dates(f, ["datadate", "rdq"])
    _normalize_dates(a, ["date"])
    s["ret"] = pd.to_numeric(s["ret"], errors="coerce")
    s["prc"] = pd.to_numeric(s["prc"], errors="coerce")
    s["shrout"] = pd.to_numeric(s["shrout"], errors="coerce")
    f["rdq"] = f["rdq"].fillna(f["datadate"] + pd.Timedelta(days=90))
    def numeric_col(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
        return pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(default, index=frame.index)

    f["book_equity"] = (numeric_col(f, "seqq") + numeric_col(f, "txditcq", 0).fillna(0) -
                        numeric_col(f, "pstkq", 0).fillna(0))
    fallback = pd.to_numeric(f.get("ceqq"), errors="coerce")
    f["book_equity"] = f["book_equity"].where(f["book_equity"].notna(), fallback)
    stocks = {p: g.sort_values("date") for p, g in s.groupby("permno")}
    factors = {p: g.set_index("date")["ar"].sort_index() for p, g in a.groupby("permno")}
    fundies = {k: g.sort_values("rdq") for k, g in f.groupby(f["gvkey"].astype("string").str.zfill(6))}
    rows = []
    for r in e.itertuples(index=False):
        day0 = pd.Timestamp(r.day0).normalize() if pd.notna(r.day0) else pd.NaT
        call = pd.Timestamp(getattr(r, "call_date_et", getattr(r, "call_date", day0))).normalize()
        stock = stocks.get(r.permno)
        arstock = factors.get(r.permno)
        row = {"keydevid": r.keydevid}
        if stock is not None and pd.notna(call):
            month_start = call.replace(day=1)
            prior = stock[stock["date"] < month_start]
            if not prior.empty:
                last = prior.iloc[-1]
                mcap = abs(last["prc"]) * last["shrout"] / 1000  # CRSP shares are in thousands; $ millions
                row["mcap_m"] = float(mcap) if pd.notna(mcap) and mcap > 0 else np.nan
                row["size"] = float(np.log(mcap * 1_000_000)) if pd.notna(mcap) and mcap > 0 else np.nan
                row["mcap_date"] = last["date"]
            start = month_start - pd.DateOffset(months=12)
            end = month_start - pd.DateOffset(months=2)
            mom = stock.loc[stock["date"].between(start, end), "ret"]
            if mom.notna().sum() >= 120:
                row["mom"] = float((1 + mom.dropna()).prod() - 1)
        if arstock is not None and pd.notna(day0):
            cal = pd.DatetimeIndex(arstock.index).sort_values()
            i = cal.get_indexer([day0])[0]
            if i >= 60:
                pre = arstock.reindex(cal[i - 60:i - 4])
                row["vol_pre"] = float(pre.std(ddof=1) * np.sqrt(252)) if pre.notna().sum() == 56 else np.nan
            if i >= 0 and i + 1 < len(cal):
                immediate = arstock.reindex(cal[i:i + 2])
                row["car_0_1"] = float(immediate.sum()) if immediate.notna().all() else np.nan
        gvkey = str(getattr(r, "gvkey", "")).zfill(6)
        fq = fundies.get(gvkey)
        if fq is not None and pd.notna(day0) and pd.notna(row.get("mcap_m")):
            available = fq[fq["rdq"] < day0]
            if not available.empty:
                be = available.iloc[-1]["book_equity"]
                if pd.notna(be) and be > 0:
                    row["bm"] = float(be / row["mcap_m"])
                    row["book_equity_date"] = available.iloc[-1]["rdq"]
        rows.append(row)
    return pd.DataFrame(rows)
