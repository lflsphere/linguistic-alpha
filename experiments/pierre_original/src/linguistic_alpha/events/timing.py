"""Event timing: call timestamp -> New York call date -> Day 0 trading session.

Day 0 is the first session in which the market can trade on the call: the call's own session
if the call starts before the close (New York time) on a trading day, otherwise the next one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def call_timestamp_utc(date_utc: pd.Series, time_utc: pd.Series) -> pd.Series:
    """Combine CIQ ``mostimportantdateutc`` + ``mostimportanttimeutc`` into a UTC timestamp."""
    date = pd.to_datetime(date_utc, errors="coerce").dt.strftime("%Y-%m-%d")
    time = time_utc.astype("string").str.strip()
    ts = pd.to_datetime(date + " " + time.fillna(""), errors="coerce", utc=True)
    # a date with no time would parse as midnight UTC; treat it as unknown instead
    return ts.where(time.fillna("").ne("") & ~time.str.lower().isin(["nan", "none", "nat"]))


def assign_day0(
    call_ts_utc: pd.Series,
    trading_days: pd.DatetimeIndex,
    call_date_fallback: pd.Series | None = None,
    market_close_et: str = "16:00",
    missing_call_time: str = "next_session",
) -> pd.DataFrame:
    """Return ``call_date_et``, ``after_close`` and ``day0`` aligned to ``call_ts_utc``.

    The trading date is taken in New York time: a call at 01:00 UTC is the previous evening
    in New York (the legacy notebook used the UTC date, which shifts such calls a day late).
    When the time is missing, ``call_date_fallback`` gives the date and ``missing_call_time``
    decides the session ("next_session" is conservative: no same-day reaction assumed).
    """
    ts_et = call_ts_utc.dt.tz_convert(NY)
    date_et = ts_et.dt.tz_localize(None).dt.normalize()
    close = pd.Timedelta(market_close_et + ":00")
    after_close = (ts_et.dt.tz_localize(None) - date_et) >= close

    missing = call_ts_utc.isna()
    if call_date_fallback is not None:
        date_et = date_et.where(~missing, pd.to_datetime(call_date_fallback).dt.normalize())
    after_close = after_close.where(~missing, missing_call_time == "next_session").astype(bool)

    days = np.asarray(pd.DatetimeIndex(trading_days).normalize().sort_values(), "datetime64[ns]")
    d = date_et.to_numpy("datetime64[ns]")
    idx = np.where(after_close.to_numpy(),
                   np.searchsorted(days, d, side="right"),   # strictly after the call date
                   np.searchsorted(days, d, side="left"))    # call date itself if it trades
    valid = (idx < len(days)) & ~np.isnat(d)
    day0 = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    day0[valid] = days[idx[valid]]

    return pd.DataFrame({"call_date_et": date_et, "after_close": after_close,
                         "day0": pd.to_datetime(day0)}, index=call_ts_utc.index)


def event_day_index(day0: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Position of Day 0 in the trading calendar (so windows are offsets in sessions)."""
    days = pd.DatetimeIndex(trading_days).normalize().sort_values()
    pos = days.get_indexer(pd.to_datetime(day0))
    return pd.Series(pos, index=day0.index).replace(-1, pd.NA).astype("Int64")
