"""Market-session timing for the audited financial baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from linguistic_alpha.events.timing import NY


def align_market_day(
    call_ts_utc: pd.Series,
    trading_days: pd.DatetimeIndex,
    call_date_fallback: pd.Series | None = None,
    market_close_et: str = "16:00",
    missing_call_time: str = "exclude",
    *,
    intraday: str = "next_session",
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``call_date_et``, ``after_close`` and ``day0`` aligned to ``call_ts_utc``.

    The trading date is taken in New York time: a call at 01:00 UTC is the previous evening
    in New York (the legacy notebook used the UTC date, which shifts such calls a day late).
    When the time is missing, ``call_date_fallback`` gives the date and ``missing_call_time``
    decides the session ("next_session" is conservative: no same-day reaction assumed).
    """
    ts_et = call_ts_utc.dt.tz_convert(NY)
    date_et = ts_et.dt.tz_localize(None).dt.normalize()
    if intraday not in ("same_session", "next_session"):
        raise ValueError("Invalid intraday policy")
    if missing_call_time not in ("next_session", "same_session", "exclude"):
        raise ValueError("Invalid missing-time policy")
    calendar = pd.DatetimeIndex(trading_days).normalize().dropna().unique().sort_values()
    opens, closes = session_hours(calendar, schedule, market_close_et)
    clock = ts_et.dt.tz_localize(None) - date_et
    is_session = date_et.isin(calendar)
    pre = is_session & clock.lt(date_et.map(opens))
    after_close = is_session & clock.ge(date_et.map(closes))
    during = is_session & clock.ge(date_et.map(opens)) & ~after_close
    session = pd.Series("unknown", index=ts_et.index)
    session.loc[pre] = "PRE"
    session.loc[during] = "INTRADAY"
    session.loc[after_close] = "POST"
    session.loc[ts_et.notna() & ~is_session] = "NON_TRADING"

    missing = call_ts_utc.isna()
    if call_date_fallback is not None:
        date_et = date_et.where(~missing, pd.to_datetime(call_date_fallback).dt.normalize())
    after_close = after_close.where(~missing, missing_call_time == "next_session")
    shift = after_close | (during & (intraday == "next_session"))
    days = calendar.to_numpy("datetime64[ns]")
    d = date_et.to_numpy("datetime64[ns]")
    idx = np.where(shift.to_numpy(),
                   np.searchsorted(days, d, side="right"),   # strictly after the call date
                   np.searchsorted(days, d, side="left"))    # call date itself if it trades
    valid = (idx < len(days)) & ~np.isnat(d)
    if missing_call_time == "exclude":
        valid &= ~missing.to_numpy()
    day0 = np.full(len(d), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    day0[valid] = days[idx[valid]]

    return pd.DataFrame({"call_date_et": date_et, "after_close": after_close,
                         "call_session": session, "day0": pd.to_datetime(day0)}, index=call_ts_utc.index)


def session_hours(calendar, schedule, regular_close="16:00"):
    """Schedule has one row per market date and local open_et/close_et clocks."""
    if schedule is None:
        return (
            pd.Series(pd.Timedelta("09:30:00"), index=calendar),
            pd.Series(pd.Timedelta(regular_close + ":00"), index=calendar),
        )
    schedule = schedule.set_index(pd.to_datetime(schedule["date"]).dt.normalize())
    within_range = schedule.index[
        (schedule.index >= calendar.min()) & (schedule.index <= calendar.max())
    ]
    if schedule.index.has_duplicates or not calendar.equals(within_range.sort_values()):
        raise ValueError("Session schedule must cover each market date exactly once")
    hours = []
    for name in ("open_et", "close_et"):
        clock = schedule[name].astype("string").reindex(calendar)
        clock = clock.where(clock.str.count(":").ne(1), clock + ":00")
        hours.append(pd.to_timedelta(clock))
    opens, closes = hours
    if not (opens.ge(pd.Timedelta(0)) & closes.gt(opens) & closes.lt(pd.Timedelta(days=1))).all():
        raise ValueError("Invalid market session hours")
    return opens, closes


def session_close_utc(dates, closes):
    local = dates + dates.map(closes)
    return local.dt.tz_localize(NY).dt.tz_convert("UTC")


def nyse_hours(calendar):
    """XNYS hours overlay for 2008¨C2024; CRSP DSI supplies the session dates."""
    if calendar.min().year < 2008 or calendar.max().year > 2024:
        raise ValueError("Provide --schedule for dates outside the reviewed 2008¨C2024 range")
    dow, year = calendar.dayofweek, calendar.year
    july3 = (calendar.month == 7) & (calendar.day == 3)
    july5 = (calendar.month == 7) & (calendar.day == 5)
    thanksgiving_friday = (calendar.month == 11) & (dow == 4) & (calendar.day >= 23) & (calendar.day <= 29)
    christmas_eve = (calendar.month == 12) & (calendar.day == 24)
    early = (thanksgiving_friday | christmas_eve | (july3 & np.isin(dow, [0, 1, 3]))
             | (july3 & (dow == 2) & (year >= 2013))
             | (july5 & (dow == 4) & (year < 2013)))
    return pd.DataFrame({"date": calendar, "open_et": "09:30",
                         "close_et": np.where(early, "13:00", "16:00")})
