import pandas as pd

from linguistic_alpha.events.timing import assign_day0, call_timestamp_utc

# Thu 2024-01-11, Fri 01-12, (weekend, MLK Mon 01-15 closed), Tue 01-16
DAYS = pd.DatetimeIndex(["2024-01-11", "2024-01-12", "2024-01-16"])


def day0(date, time, **kw):
    ts = call_timestamp_utc(pd.Series([date]), pd.Series([time]))
    return assign_day0(ts, DAYS, call_date_fallback=pd.Series([pd.Timestamp(date)]), **kw).iloc[0]


def test_before_open_same_session():
    assert day0("2024-01-11", "13:30:00")["day0"] == pd.Timestamp("2024-01-11")  # 08:30 ET


def test_after_close_next_session():
    assert day0("2024-01-11", "21:30:00")["day0"] == pd.Timestamp("2024-01-12")  # 16:30 ET


def test_utc_date_after_midnight_uses_new_york_date():
    # 01:00 UTC Fri = 20:00 ET Thu -> NY date Thu, after close -> Day 0 Fri (not Tue)
    r = day0("2024-01-12", "01:00:00")
    assert r["call_date_et"] == pd.Timestamp("2024-01-11")
    assert r["day0"] == pd.Timestamp("2024-01-12")


def test_friday_after_close_skips_weekend_and_holiday():
    assert day0("2024-01-12", "21:30:00")["day0"] == pd.Timestamp("2024-01-16")


def test_missing_time_is_conservative():
    r = day0("2024-01-11", None)
    assert r["after_close"]
    assert r["day0"] == pd.Timestamp("2024-01-12")
    r = day0("2024-01-11", None, missing_call_time="same_session")
    assert r["day0"] == pd.Timestamp("2024-01-11")


def test_beyond_calendar_is_nat():
    assert pd.isna(day0("2024-01-16", "22:00:00")["day0"])


def test_release_before_open_and_after_close():
    from linguistic_alpha.events.timing import release_timestamp_utc
    ts = release_timestamp_utc(pd.Series(["2024-01-11", "2024-01-11", "2024-01-12"]),
                               pd.Series(["07:00:00", "16:05:00", "00:00:00"]))
    out = assign_day0(ts, DAYS, call_date_fallback=pd.Series(pd.to_datetime(
        ["2024-01-11", "2024-01-11", "2024-01-12"])))
    assert list(out["day0"]) == list(pd.to_datetime(["2024-01-11", "2024-01-12", "2024-01-16"]))


def test_call_lag_sessions():
    from linguistic_alpha.events.timing import call_lag_sessions
    lag = call_lag_sessions(pd.Series(pd.to_datetime(["2024-01-12"])),
                            pd.Series(pd.to_datetime(["2024-01-11"])), DAYS)
    assert lag.iloc[0] == 1


def test_missing_time_in_first_row_does_not_blank_the_others():
    from linguistic_alpha.events.timing import release_timestamp_utc
    d = pd.Series(["2024-01-11", "2024-01-11", "2024-01-12"])
    t = pd.Series([None, "13:30:00", "21:30:00"])
    ts = call_timestamp_utc(d, t)
    assert ts.isna().tolist() == [True, False, False]
    assert ts.iloc[1] == pd.Timestamp("2024-01-11 13:30", tz="UTC")
    r = release_timestamp_utc(d, pd.Series(["", "07:00:00", "16:30:00"]))
    assert r.isna().tolist() == [True, False, False]
    assert r.iloc[1] == pd.Timestamp("2024-01-11 12:00", tz="UTC")   # 07:00 EST
