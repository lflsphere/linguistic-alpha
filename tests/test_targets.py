"""targets/car.py on synthetic returns with known CARs."""

import numpy as np
import pandas as pd
import pytest

from linguistic_alpha.targets.car import (abnormal_returns, benchmark_series, build_targets,
                                          event_car, post_call_volatility)

SESSIONS = pd.bdate_range("2020-01-01", periods=400)
D0 = 300          # Day 0 position


def _bench(r=0.001):
    return pd.DataFrame({"caldt": SESSIONS.strftime("%Y-%m-%d"), "vwretd": r})


def _dsf(ret, permno=1, sessions=SESSIONS):
    return pd.DataFrame({"permno": permno, "dlycaldt": sessions.strftime("%Y-%m-%d"),
                         "dlyret": ret})


def _events(pos=D0, permno=1, keydevid=10):
    return pd.DataFrame({"keydevid": [keydevid], "permno": [permno], "day0": [SESSIONS[pos]]})


def test_abnormal_returns_grid_and_missing_days():
    dsf = _dsf(0.002).drop(index=[5, 6])          # two sessions absent from CRSP
    dsf.loc[10, "dlyret"] = None                  # and one null return
    ar = abnormal_returns(dsf, _bench())
    assert len(ar) == len(SESSIONS)
    assert ar["ret"].isna().sum() == 3
    assert np.allclose(ar["ar"].dropna(), 0.001)


def test_car_known_value_compounded_per_leg():
    ar = abnormal_returns(_dsf(0.002), _bench(0.001))
    out = event_car(_events(), ar, {"car_2_20": [2, 20], "car_2_5": [2, 5]}, 0.8)
    assert out.loc[0, "car_2_20"] == pytest.approx(1.002 ** 19 - 1.001 ** 19, abs=1e-12)
    assert out.loc[0, "car_2_5"] == pytest.approx(1.002 ** 4 - 1.001 ** 4, abs=1e-12)
    assert out.loc[0, "car_2_20_cov"] == 1.0


def test_car_uses_exact_window_sessions():
    ret = np.zeros(len(SESSIONS))
    ret[D0 + 2] = 0.10                            # inside [+2, +5]
    ret[D0 + 1] = 0.50                            # outside: Day +1 is not in the target
    ret[D0 + 6] = -0.20                           # outside [+2, +5]
    ar = abnormal_returns(_dsf(ret), _bench(0.0))
    out = event_car(_events(), ar, {"car_2_5": [2, 5]}, 0.8)
    assert out.loc[0, "car_2_5"] == pytest.approx(0.10)


def test_day0_on_non_session_rolls_forward():
    pos = next(i for i in range(D0, D0 + 7) if SESSIONS[i].weekday() == 0)   # a Monday
    ret = np.zeros(len(SESSIONS))
    ret[pos + 2] = 0.05
    ar = abnormal_returns(_dsf(ret), _bench(0.0))
    ev = _events(pos=pos)
    ev["day0"] = [SESSIONS[pos] - pd.Timedelta(days=2)]  # the Saturday before -> Monday
    out = event_car(ev, ar, {"car_2_2": [2, 2]}, 0.8)
    assert out.loc[0, "car_2_2"] == pytest.approx(0.05)


@pytest.mark.parametrize("n_null,expect_missing", [(3, False), (4, True)])
def test_coverage_threshold(n_null, expect_missing):
    # [+2, +20] has 19 days; 0.8 * 19 = 15.2, so 16 valid days pass (3 null) and 15 fail (4 null)
    ret = np.full(len(SESSIONS), 0.002)
    null_pos = D0 + 2 + np.arange(n_null) * 3
    ret[null_pos] = np.nan
    ar = abnormal_returns(_dsf(ret), _bench(0.001))
    out = event_car(_events(), ar, {"car_2_20": [2, 20]}, 0.8)
    assert out.loc[0, "car_2_20_cov"] == pytest.approx((19 - n_null) / 19)
    if expect_missing:
        assert np.isnan(out.loc[0, "car_2_20"])
    else:   # null stock days skipped (not filled with 0), index over every day
        assert out.loc[0, "car_2_20"] == pytest.approx(1.002 ** (19 - n_null) - 1.001 ** 19)


def test_coverage_exactly_at_threshold_is_kept():
    # [+2, +11] = 10 days; 8 valid = exactly 0.8
    ret = np.full(len(SESSIONS), 0.0)
    ret[[D0 + 3, D0 + 7]] = np.nan
    ar = abnormal_returns(_dsf(ret), _bench(0.0))
    out = event_car(_events(), ar, {"w": [2, 11]}, 0.8)
    assert out.loc[0, "w_cov"] == pytest.approx(0.8)
    assert out.loc[0, "w"] == pytest.approx(0.0)


def test_window_past_data_end_is_missing():
    ar = abnormal_returns(_dsf(0.002), _bench(0.001))
    out = event_car(_events(pos=len(SESSIONS) - 30), ar, {"car_2_20": [2, 20],
                                                          "car_2_60": [2, 60]}, 0.8)
    assert not np.isnan(out.loc[0, "car_2_20"])
    assert np.isnan(out.loc[0, "car_2_60"])


def test_delisting_inside_window():
    # permno delists at Day +10 with a 20% delisting return already in dlyret; CRSP has no
    # rows after it. Later sessions are held in the benchmark (AR = 0), so the CAR exists.
    last = D0 + 10
    ret = np.full(last + 1, 0.0)
    ret[last] = 0.20
    dsf = _dsf(ret, sessions=SESSIONS[: last + 1])
    bench = _bench(0.001)
    ar = abnormal_returns(dsf, bench)
    b = benchmark_series(bench)
    dd = pd.Series([SESSIONS[last]], index=[1])
    out = event_car(_events(), ar, {"car_2_20": [2, 20]}, 0.8, bench=b, delist_dates=dd)
    stock = 1.20 * 1.001 ** 10                   # +2..+10 own returns, +11..+20 in the index
    assert out.loc[0, "car_2_20"] == pytest.approx(stock - 1.001 ** 19)
    assert out.loc[0, "car_2_20_cov"] == 1.0
    assert out.loc[0, "delist_rel"] == 10
    # the delisting return is counted once (no second splice)
    no_dd = event_car(_events(), ar, {"car_2_10": [2, 10]}, 0.8, bench=b)
    assert no_dd.loc[0, "car_2_10"] == pytest.approx(1.20 - 1.001 ** 9)
    # without the delisting date the missing tail fails coverage
    miss = event_car(_events(), ar, {"car_2_20": [2, 20]}, 0.8, bench=b)
    assert np.isnan(miss.loc[0, "car_2_20"])
    # volatility: post-delisting days are missing, so coverage fails
    v = post_call_volatility(_events(), ar, [2, 20], [-60, -5], 0.8, bench=b, delist_dates=dd)
    assert np.isnan(v.loc[0, "vol_post_2_20"])


def test_market_model_recovers_alpha_beta():
    rng = np.random.default_rng(0)
    rb = rng.normal(0.0005, 0.01, len(SESSIONS))
    rs = 0.0002 + 1.5 * rb
    bench = pd.DataFrame({"caldt": SESSIONS, "vwretd": rb})
    ar = abnormal_returns(_dsf(rs), bench)
    out = event_car(_events(), ar, {"car_2_20": [2, 20]}, 0.8,
                    market_model={"estimation_window": [-250, -30], "min_coverage": 0.8})
    assert out.loc[0, "mm_beta"] == pytest.approx(1.5)
    assert out.loc[0, "mm_alpha"] == pytest.approx(0.0002)
    assert out.loc[0, "car_2_20_mm"] == pytest.approx(0.0, abs=1e-12)
    w = slice(D0 + 2, D0 + 21)
    assert out.loc[0, "car_2_20"] == pytest.approx(np.prod(1 + rs[w]) - np.prod(1 + rb[w]))


def test_market_model_needs_estimation_coverage():
    ret = np.full(len(SESSIONS), 0.001)
    ret[D0 - 250: D0 - 100] = np.nan              # 150 of 221 estimation days missing
    ar = abnormal_returns(_dsf(ret), _bench(0.001))
    out = event_car(_events(), ar, {"car_2_20": [2, 20]}, 0.8,
                    market_model={"estimation_window": [-250, -30], "min_coverage": 0.8})
    assert np.isnan(out.loc[0, "car_2_20_mm"])
    assert not np.isnan(out.loc[0, "car_2_20"])


def test_post_call_volatility():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.02, len(SESSIONS))
    ar = abnormal_returns(_dsf(a + 0.001), _bench(0.001))
    v = post_call_volatility(_events(), ar, [2, 20], [-60, -5], 0.8)
    post = np.std(a[D0 + 2: D0 + 21], ddof=1) * np.sqrt(252)
    pre = np.std(a[D0 - 60: D0 - 4], ddof=1) * np.sqrt(252)
    assert v.loc[0, "vol_post_2_20"] == pytest.approx(post)
    assert v.loc[0, "vol_ratio_2_20"] == pytest.approx(post / pre)


def test_unknown_permno_all_missing_and_build_targets_schema():
    from linguistic_alpha import schemas
    tcfg = {"benchmark": "sp500", "benchmark_column": "vwretd",
            "market_model": {"estimation_window": [-250, -30], "min_coverage": 0.8},
            "car_windows": {"car_2_20": [2, 20], "car_2_5": [2, 5], "car_2_60": [2, 60]},
            "min_window_coverage": 0.8, "vol_post_window": [2, 20], "vol_pre_window": [-60, -5]}
    ev = pd.concat([_events(), _events(permno=99, keydevid=11)], ignore_index=True)
    out = build_targets(ev, _dsf(0.002), _bench(0.001), None, tcfg)
    schemas.validate(out, schemas.TARGETS, "targets", key="keydevid")
    assert out.loc[1, ["car_2_20", "car_2_5", "car_2_60", "vol_post_2_20"]].isna().all()
    assert out.loc[0, "car_2_20"] == pytest.approx(1.002 ** 19 - 1.001 ** 19)
