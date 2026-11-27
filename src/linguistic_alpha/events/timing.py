"""Event timing: event timestamp -> New York date -> Day 0 trading session.

The earnings release and the call count as one event (``events.anchor: release``): Day 0 is
the first session whose close comes after the **release**, i.e. the release's own session if
it comes before the close (New York time) on a trading day, otherwise the next one. The call
is then checked to fall in Day 0 or Day +1 (``events.max_call_lag_sessions``), so the target
window starting at Day +2 never begins before the call text is out.

``assign_day0`` works on any UTC timestamp: pass the release timestamp for the main sample,
or the call timestamp (``anchor: call``) for the robustness check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def call_timestamp_utc(date_utc: pd.Series, time_utc: pd.Series) -> pd.Series:
    """Combine CIQ ``mostimportantdateutc`` + ``mostimportanttimeutc`` into a UTC timestamp."""
    date = pd.to_datetime(date_utc, errors="coerce").dt.normalize()
    time = time_utc.astype("string").str.strip()
    # date + parsed time offset, row by row (parsing "date time" strings lets pandas infer one
    # format from the first row, which can turn every other row into NaT). A date with no time
    # would be midnight UTC; it is unknown instead.
    known = time.notna() & time.ne("") & ~time.str.lower().isin(["nan", "none", "nat"])
    offset = pd.to_timedelta(time.where(known), errors="coerce")
    return (date + offset).dt.tz_localize("UTC")


def release_timestamp_utc(anndats: pd.Series, anntims: pd.Series,
                          tz: str = NY) -> pd.Series:
    """I/B/E/S ``anndats`` + ``anntims`` (local clock time, ``tz``) -> UTC timestamp.

    ``anntims`` of 00:00:00 or missing is treated as unknown (NaT), so ``assign_day0`` falls
    back to the date with ``missing_call_time``.
    """
    date = pd.to_datetime(anndats, errors="coerce").dt.normalize()
    time = anntims.astype("string").str.strip()
    known = time.notna() & ~time.isin(["", "00:00:00", "nan", "None", "NaT"])
    local = date + pd.to_timedelta(time.where(known), errors="coerce")
    local = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return local.dt.tz_convert("UTC").where(known)


def call_lag_sessions(call_day0: pd.Series, day0: pd.Series,
                      trading_days: pd.DatetimeIndex) -> pd.Series:
    """Sessions from Day 0 (release) to the call's own first tradeable session."""
    return event_day_index(call_day0, trading_days) - event_day_index(day0, trading_days)


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
