"""Corrected market-day CAR for A2; the archived target path stays separate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def financial_abnormal_returns(dsf: pd.DataFrame, bench: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """Daily ``permno, date, ret, ar`` with delisting returns folded in."""
    s = dsf.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s["ret"] = numeric(s["ret"]).where(lambda x: x.ge(-1))
    d = delist[["permno", "dlstdt", "dlret"]].copy()
    d["date"] = pd.to_datetime(d.pop("dlstdt"), errors="coerce").dt.normalize()
    d["dlret"] = numeric(d["dlret"]).where(lambda x: x.ge(-1))
    d = d.dropna(subset=["permno", "date"]).drop_duplicates()
    s = s.merge(d, on=["permno", "date"], how="outer", validate="one_to_one")
    s["return_missing_both"] = s["ret"].isna() & s["dlret"].isna()
    s["ret_total"] = ((1 + s["ret"].fillna(0)) * (1 + s["dlret"].fillna(0)) - 1).mask(
        s["return_missing_both"]
    )
    b = bench.copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
    b["bench_ret"] = numeric(b["vwretd"])
    out = s.merge(b[["date", "bench_ret"]], on="date", how="left", validate="many_to_one")
    out["ar"] = out["ret_total"] - out["bench_ret"]
    out["return_missing_both"] = out["return_missing_both"].astype("boolean")
    return out[["permno", "date", "ret_total", "bench_ret", "ar", "return_missing_both"]].sort_values(
        ["permno", "date"]).reset_index(drop=True)


def market_days(dates) -> pd.DatetimeIndex:
    """Unique, sorted exchange session labels, expressed as local dates."""
    days = pd.DatetimeIndex(pd.to_datetime(dates)).dropna().normalize()
    if days.tz is not None:
        raise ValueError("Market session labels must be timezone-naive dates")
    return days.unique().sort_values()


def market_date_at(day0: pd.Series, calendar: pd.DatetimeIndex, offset: int) -> pd.Series:
    base = calendar.get_indexer(day0)
    pos = base + offset
    valid = (base >= 0) & (pos >= 0) & (pos < len(calendar))
    out = pd.Series(pd.NaT, index=day0.index, dtype="datetime64[ns]")
    out.iloc[np.flatnonzero(valid)] = calendar[pos[valid]].to_numpy()
    return out


def event_car(events: pd.DataFrame, ar: pd.DataFrame, windows: dict[str, list[int]],
              min_coverage: float, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Measure each window on market sessions; no imputation or partial-window scaling."""
    if not 0 < min_coverage <= 1:
        raise ValueError("Minimum coverage must be in (0, 1]")
    calendar = market_days(trading_days)
    output = pd.DataFrame({"keydevid": events["keydevid"].to_numpy()})
    base = calendar.get_indexer(pd.to_datetime(events["day0"]).dt.normalize())
    a = ar.assign(date=pd.to_datetime(ar["date"]).dt.normalize())
    stock_groups = a.groupby("permno").indices
    event_groups = events.groupby("permno", dropna=False).indices
    for lo, hi in windows.values():
        if lo < 2 or hi < lo:
            raise ValueError("Financial baseline targets must start at Day +2 or later")
    for name, (lo, hi) in windows.items():
        expected = hi - lo + 1
        total = np.zeros(len(events))
        valid_days = np.zeros(len(events), dtype=int)
        left = np.clip(base + lo, 0, len(calendar))
        right = np.clip(base + hi + 1, 0, len(calendar))
        right = np.where(base >= 0, right, left)
        date_count = np.where(base >= 0, right - left, 0)
        complete = date_count == expected
        for permno, indices in event_groups.items():
            source_indices = stock_groups.get(permno)
            if source_indices is None:
                continue
            rows = a.iloc[source_indices].set_index("date").reindex(calendar)
            values = rows["ar"].to_numpy(dtype=float, na_value=np.nan)
            sums = np.r_[0.0, np.nancumsum(values)]
            counts = np.r_[0, np.cumsum(~np.isnan(values))]
            i, j = left[indices], right[indices]
            total[indices] = sums[j] - sums[i]
            valid_days[indices] = counts[j] - counts[i]
        coverage = valid_days / expected
        output[name] = np.where(complete & (coverage >= min_coverage), total, np.nan)
        output[f"{name}_coverage"] = np.where(complete, coverage, np.nan)
        output[f"{name}_valid_days"] = pd.Series(valid_days, dtype="Int64").where(base >= 0)
    return output
