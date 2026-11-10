"""Cumulative abnormal returns and post-call volatility.

Status: specified, not implemented (waits on crsp_dsf / crsp_dsi / crsp_dsedelist).

Definitions (project scope §1, §4):
  AR(i,t)          = R(i,t) - R(bench,t); bench = S&P 500 (crsp.dsi), FF4 for robustness.
  CAR[a,b]         = sum of AR over sessions Day0+a .. Day0+b (offsets via events.timing).
  vol_post_2_20    = std(AR, +2..+20) * sqrt(252)
  vol_ratio_2_20   = vol_post_2_20 / std(AR, -60..-5) * sqrt(252)   (scope definition)

Fixes vs. the legacy notebook (docs/data_audit.md §Targets):
  * Merge delisting returns: R = (1+ret)(1+dlret)-1 on the delisting date.
  * Benchmark consistency: ``sprtrn`` is a price return while CRSP ``ret`` includes dividends;
    use ``vwretd`` or a total-return S&P series, or document the drift (~2%/yr).
  * Coverage rule (``targets.min_window_coverage``) is reported per event, not only applied.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def abnormal_returns(dsf: pd.DataFrame, bench: pd.DataFrame, delist: pd.DataFrame,
                     model: str = "sp500") -> pd.DataFrame:
    """Daily ``permno, date, ret, ar`` with delisting returns folded in."""
    s = dsf.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s["ret"] = _numeric(s["ret"])
    d = delist[["permno", "dlstdt", "dlret"]].copy()
    d["date"] = pd.to_datetime(d.pop("dlstdt"), errors="coerce").dt.normalize()
    d["dlret"] = _numeric(d["dlret"])
    s = s.merge(d[["permno", "date", "dlret"]], on=["permno", "date"], how="left")
    # CRSP delisting return is a second component of the holding-period return.
    s["ret_total"] = (1 + s["ret"].fillna(0)) * (1 + s["dlret"].fillna(0)) - 1
    b = bench.copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
    if model == "sp500":
        col = "vwretd" if "vwretd" in b else "sprtrn"
        b["bench_ret"] = _numeric(b[col])
    elif model == "ff4":
        required = ["mktrf", "smb", "hml", "umd", "rf"]
        missing = set(required) - set(b.columns)
        if missing:
            raise ValueError(f"FF4 benchmark missing columns: {sorted(missing)}")
        # Factor-model abnormal returns require estimated stock-specific betas and are
        # intentionally not approximated here; callers can pass precomputed expected_ret.
        if "expected_ret" not in b:
            raise ValueError("ff4 requires a precomputed expected_ret column")
        b["bench_ret"] = _numeric(b["expected_ret"])
    else:
        raise ValueError(f"unknown abnormal-return model: {model}")
    out = s.merge(b[["date", "bench_ret"]], on="date", how="left")
    out["ar"] = out["ret_total"] - out["bench_ret"]
    return out[["permno", "date", "ret_total", "bench_ret", "ar"]].sort_values(
        ["permno", "date"]).reset_index(drop=True)


def event_car(events: pd.DataFrame, ar: pd.DataFrame, windows: dict[str, list[int]],
              min_coverage: float) -> pd.DataFrame:
    """One row per keydevid with every configured CAR window."""
    e = events[["keydevid", "permno", "day0"]].copy()
    e["day0"] = pd.to_datetime(e["day0"], errors="coerce").dt.normalize()
    a = ar[["permno", "date", "ar"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    calendar = pd.DatetimeIndex(a["date"].dropna().unique()).sort_values()
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    grouped = {p: g.set_index("date")["ar"] for p, g in a.groupby("permno")}
    max_offset = max(b for _, b in windows.values())
    rows = []
    for row in e.itertuples(index=False):
        base = positions.get(row.day0, np.nan)
        stock = grouped.get(row.permno)
        record = {"keydevid": row.keydevid}
        for name, (lo, hi) in windows.items():
            dates = (calendar[max(0, int(base) + lo):min(len(calendar), int(base) + hi + 1)]
                     if pd.notna(base) else pd.DatetimeIndex([]))
            vals = stock.reindex(dates) if stock is not None else pd.Series(dtype=float)
            n = int(vals.notna().sum())
            coverage = n / (hi - lo + 1)
            record[name] = float(vals.sum()) if coverage >= min_coverage else np.nan
            record[f"{name}_coverage"] = coverage
        rows.append(record)
    return pd.DataFrame(rows)


def post_call_volatility(events: pd.DataFrame, ar: pd.DataFrame, post: list[int],
                         pre: list[int]) -> pd.DataFrame:
    e = events[["keydevid", "permno", "day0"]].copy()
    e["day0"] = pd.to_datetime(e["day0"], errors="coerce").dt.normalize()
    a = ar[["permno", "date", "ar"]].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    calendar = pd.DatetimeIndex(a["date"].dropna().unique()).sort_values()
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    grouped = {p: g.set_index("date")["ar"] for p, g in a.groupby("permno")}
    rows = []
    for row in e.itertuples(index=False):
        base = positions.get(row.day0, np.nan)
        stock = grouped.get(row.permno)
        def window(bounds):
            lo, hi = bounds
            if pd.isna(base):
                return np.nan
            dates = calendar[max(0, int(base) + lo):min(len(calendar), int(base) + hi + 1)]
            vals = stock.reindex(dates) if stock is not None else []
            return float(vals.std(ddof=1)) if len(vals) and pd.Series(vals).notna().sum() == hi - lo + 1 else np.nan
        post_vol, pre_vol = window(post), window(pre)
        rows.append({"keydevid": row.keydevid,
                     "vol_post_2_20": post_vol * np.sqrt(252) if pd.notna(post_vol) else np.nan,
                     "vol_ratio_2_20": post_vol / pre_vol if pd.notna(post_vol) and pre_vol > 0 else np.nan})
    return pd.DataFrame(rows)
