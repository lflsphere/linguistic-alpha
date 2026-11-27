"""la controls: synthetic I/B/E/S, Compustat and CRSP rows (docs/handoff_controls.md, Checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linguistic_alpha.controls.build import lookahead_check
from linguistic_alpha.controls.earnings import (earnings_surprise, eps_surprise_history,
                                                flag_and_zero, release_guidance, revenue_surprise,
                                                split_factor)
from linguistic_alpha.controls.firm import book_to_market, firm_controls
from linguistic_alpha.targets.car import abnormal_returns, benchmark_series

NO_ADJ = pd.DataFrame({"ticker": [], "spdates": [], "adj": []})


def _actuals(rows):
    return pd.DataFrame([{"ticker": t, "measure": "EPS", "pdicity": "QTR", "pends": p,
                          "anndats": a, "value": v} for t, p, a, v in rows])


def _summary(rows):
    return pd.DataFrame([{"ticker": t, "measure": "EPS", "fiscalp": "QTR", "fpi": "6",
                          "statpers": s, "fpedats": f, "medest": m, "stdev": sd, "numest": n,
                          "numup": up, "numdown": dn, "highest": hi, "lowest": lo}
                         for t, s, f, m, sd, n, up, dn, hi, lo in rows])


def _event(ticker="AAA", pends="2015-03-31", anndats="2015-04-20", day0="2015-04-20", k=1):
    return pd.DataFrame({"keydevid": [k], "ibes_ticker": [ticker], "release_pends": [pends],
                         "release_anndats": [pd.Timestamp(anndats)], "day0": [pd.Timestamp(day0)]})


def test_consensus_is_last_before_release_and_later_forecast_ignored():
    act = _actuals([("AAA", "2015-03-31", "2015-04-20", 1.10)])
    cons = _summary([
        ("AAA", "2015-03-19", "2015-03-31", 1.00, 0.05, 10, 2, 1, 1.2, 0.9),
        ("AAA", "2015-04-16", "2015-03-31", 1.02, 0.04, 12, 3, 1, 1.1, 0.95),   # last before
        ("AAA", "2015-04-20", "2015-03-31", 1.50, 0.10, 12, 9, 0, 2.0, 1.0),    # same day: no
        ("AAA", "2015-05-21", "2015-03-31", 1.60, 0.10, 12, 9, 0, 2.0, 1.0),    # after Day 0
        ("AAA", "2015-04-16", "2015-06-30", 9.99, 0.10, 12, 9, 0, 2.0, 1.0),    # other period
    ])
    hist = eps_surprise_history(act, cons, NO_ADJ)
    out = earnings_surprise(_event(), hist, pd.Series({1: 40.0}))
    r = out.iloc[0]
    assert r["eps_consensus"] == pytest.approx(1.02)
    assert r["eps_surprise"] == pytest.approx(0.08)
    assert r["sue"] == pytest.approx(0.08 / 40.0)
    assert r["sue_std"] == pytest.approx(0.08 / 0.04)
    assert r["n_analysts"] == 12
    assert r["dispersion"] == pytest.approx(0.04 / 1.02)
    assert r["net_revisions"] == pytest.approx((3 - 1) / 12)
    assert r["forecast_range"] == pytest.approx((1.1 - 0.95) / 40.0)
    assert r["src_consensus_statpers"] == pd.Timestamp("2015-04-16")


def test_zero_stdev_gives_nan_sue_std():
    act = _actuals([("AAA", "2015-03-31", "2015-04-20", 1.10)])
    cons = _summary([("AAA", "2015-04-16", "2015-03-31", 1.0, 0.0, 1, 0, 0, 1.0, 1.0)])
    out = earnings_surprise(_event(), eps_surprise_history(act, cons, NO_ADJ), pd.Series({1: 40.0}))
    assert np.isnan(out.loc[0, "sue_std"])
    assert out.loc[0, "eps_surprise"] == pytest.approx(0.10)


def test_split_between_consensus_and_release_is_not_a_surprise():
    # 2-for-1 split on 2015-04-01: consensus 2.00 on the old basis = 1.00 on the new basis
    adj = pd.DataFrame({"ticker": ["AAA", "AAA"], "spdates": ["2000-01-01", "2015-04-01"],
                        "adj": [2.0, 1.0]})
    act = _actuals([("AAA", "2015-03-31", "2015-04-20", 1.00)])
    cons = _summary([("AAA", "2015-03-19", "2015-03-31", 2.00, 0.10, 5, 0, 0, 2.2, 1.8)])
    out = earnings_surprise(_event(), eps_surprise_history(act, cons, adj), pd.Series({1: 40.0}))
    assert out.loc[0, "eps_consensus"] == pytest.approx(1.00)
    assert out.loc[0, "eps_surprise"] == pytest.approx(0.0)
    assert split_factor(adj, ["AAA", "AAA", "ZZZ"], ["1990-01-01", "2016-01-01", "2016-01-01"]
                        ).tolist() == [2.0, 1.0, 1.0]


def test_sue_ts_uses_only_prior_surprises():
    q = pd.period_range("2013Q1", periods=7, freq="Q")
    surprises = [0.01, -0.02, 0.03, 0.00, 0.02, 0.05, 0.50]
    act_rows, cons_rows = [], []
    for p, s in zip(q, surprises):
        pe = p.end_time.normalize()
        ann = pe + pd.Timedelta(days=20)
        act_rows.append(("AAA", str(pe.date()), str(ann.date()), 1.0 + s))
        cons_rows.append(("AAA", str((ann - pd.Timedelta(days=5)).date()), str(pe.date()),
                          1.0, 0.02, 5, 0, 0, 1.1, 0.9))
    hist = eps_surprise_history(_actuals(act_rows), _summary(cons_rows), NO_ADJ)
    hist = hist.sort_values("anndats").reset_index(drop=True)
    assert hist["sue_ts"].iloc[:4].isna().all()          # fewer than 4 prior surprises
    sd = np.std(surprises[:6], ddof=1)
    assert hist["sue_ts"].iloc[6] == pytest.approx(0.50 / sd)   # own surprise not in the scale
    assert (hist["sue_ts_scale_date"].dropna() < hist["anndats"][hist["sue_ts_scale_date"].notna()]).all()


def test_revenue_surprise_and_late_actual():
    actx = pd.DataFrame({"ticker": ["AAA", "BBB"], "measure": ["SAL", "SAL"],
                         "pdicity": ["QTR", "QTR"], "pends": ["2015-03-31"] * 2,
                         "anndats": ["2015-04-20", "2015-05-30"], "value": [110.0, 50.0]})
    sx = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "measure": ["SAL"] * 3, "fpi": ["6"] * 3,
                       "statpers": ["2015-04-16", "2015-04-23", "2015-04-16"],
                       "fpedats": ["2015-03-31"] * 3, "medest": [100.0, 200.0, 40.0]})
    ev = pd.concat([_event(), _event(ticker="BBB", k=2)], ignore_index=True)
    out = revenue_surprise(ev, actx, sx).set_index("keydevid")
    assert out.loc[1, "rev_sue"] == pytest.approx(0.10)
    assert np.isnan(out.loc[2, "rev_sue"])               # reported after Day 0


def _guidance(rows):
    return pd.DataFrame([{"ticker": "AAA", "measure": m, "pdicity": pd_, "anndats": a,
                          "anntims": "08:00:00", "prd_yr": y, "prd_mon": mo, "val_1": v1,
                          "val_2": v2, "mean_at_date": mad}
                         for m, pd_, a, y, mo, v1, v2, mad in rows])


def test_guidance_metrics_and_change():
    g = _guidance([
        ("EPS", "QTR", "2015-01-20", 2015, 6, 1.00, 1.10, 1.00),    # previous guidance for Q2
        ("EPS", "QTR", "2015-04-20", 2015, 6, 1.10, 1.30, 1.10),    # at the release: raised
        ("EPS", "ANN", "2015-04-20", 2015, 12, 4.0, 4.4, 4.1),      # later period: not chosen
        ("EPS", "QTR", "2015-04-22", 2015, 6, 0.50, 0.60, 1.10),    # after Day 0: ignored
    ])
    out = release_guidance(_event(), g, NO_ADJ, pd.Series({1: 50.0})).iloc[0]
    assert out["guidance_given"] == 1
    assert out["guidance_vs_consensus"] == pytest.approx((1.20 - 1.10) / 50.0)
    assert out["guidance_width"] == pytest.approx(0.20 / 50.0)
    assert out["guidance_change"] == 1
    assert out["src_guidance_anndats"] == pd.Timestamp("2015-04-20")


def test_no_guidance_is_flagged_and_zeroed():
    g = _guidance([("SAL", "QTR", "2015-04-20", 2015, 6, 100.0, None, None),
                   ("EPS", "QTR", "2014-01-20", 2014, 3, 1.0, None, 1.0)])
    ev = pd.concat([_event(), _event(ticker="ZZZ", k=2)], ignore_index=True)
    out = release_guidance(ev, g, NO_ADJ, pd.Series({1: 50.0, 2: 50.0})).set_index("keydevid")
    assert out.loc[1, "guidance_given"] == 1                # SAL guidance only: given, no EPS
    assert np.isnan(out.loc[1, "guidance_width"])
    assert out.loc[2, "guidance_given"] == 0
    out["rev_sue"] = [np.nan, 0.05]
    metrics = ["rev_sue", "guidance_given", "guidance_vs_consensus", "guidance_width",
               "guidance_change"]
    f = flag_and_zero(out, metrics)
    assert f.loc[1, "rev_sue"] == 0 and f.loc[1, "rev_sue_missing"] == 1
    assert f.loc[2, "rev_sue"] == pytest.approx(0.05) and f.loc[2, "rev_sue_missing"] == 0
    assert (f[["guidance_vs_consensus", "guidance_width", "guidance_change"]] == 0).all().all()
    assert (f["guidance_width_missing"] == 1).all()
    assert "guidance_given_missing" not in f.columns
    assert not f[metrics].isna().any().any()


def test_point_guidance_has_zero_width():
    g = _guidance([("EPS", "QTR", "2015-04-20", 2015, 6, 1.20, None, 1.10)])
    out = release_guidance(_event(), g, NO_ADJ, pd.Series({1: 50.0})).iloc[0]
    assert out["guidance_width"] == 0
    assert np.isnan(out["guidance_change"])               # no earlier guidance to compare


def test_book_to_market_uses_reports_before_day0_only():
    fq = pd.DataFrame({"gvkey": ["001"] * 3,
                       "datadate": ["2014-09-30", "2014-12-31", "2015-03-31"],
                       "rdq": ["2014-10-20", None, "2015-04-20"],      # last one on Day 0
                       "seqq": [100.0, 120.0, 999.0], "txditcq": [10.0, None, 0.0],
                       "pstkq": [5.0, 0.0, 0.0], "ceqq": [90.0, 110.0, 999.0]})
    ev = pd.DataFrame({"gvkey": ["001"], "day0": [pd.Timestamp("2015-04-20")]})
    bm, avail = book_to_market(ev, fq, np.array([1_200_000.0]))   # $1,200m in thousands
    # 2014-12-31 has NULL rdq -> available 2015-03-31 (< Day 0); BE = 120
    assert bm[0] == pytest.approx(120.0 / 1200.0)
    assert pd.Timestamp(avail[0]) == pd.Timestamp("2015-03-31")


def _crsp(n_days=320, null_on=None):
    sessions = pd.bdate_range("2014-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0005, 0.01, n_days)
    bret = rng.normal(0.0003, 0.008, n_days)
    dsf = pd.DataFrame({"permno": 10, "permco": 7, "dlycaldt": sessions, "dlyret": ret,
                        "dlyprc": -25.0, "dlycap": 1_000_000.0})
    dsf2 = dsf.assign(permno=11, dlycap=500_000.0)           # second share class, same firm
    if null_on is not None:
        dsf.loc[dsf["dlycaldt"] == null_on, "dlyret"] = np.nan
    bench = pd.DataFrame({"caldt": sessions, "vwretd": bret})
    return pd.concat([dsf, dsf2], ignore_index=True), bench, sessions, ret, bret


TCFG = {"reaction_window": {"car_m1_1": [-1, 1], "car_0_1": [0, 1]},
        "vol_pre_window": [-60, -5], "min_window_coverage": 0.8}


def _firm(dsf, bench, day0):
    ev = pd.DataFrame({"keydevid": [1], "permno": [10], "gvkey": ["001"], "day0": [day0]})
    ar = abnormal_returns(dsf[dsf["permno"] == 10], bench)
    fq = pd.DataFrame({"gvkey": ["001"], "datadate": ["2014-09-30"], "rdq": ["2014-10-20"],
                       "seqq": [600.0], "txditcq": [0.0], "pstkq": [0.0], "ceqq": [600.0]})
    return firm_controls(ev, dsf, fq, ar, benchmark_series(bench), TCFG).iloc[0]


def test_firm_controls_windows():
    dsf, bench, sessions, ret, bret = _crsp()
    day0 = sessions[300]
    r = _firm(dsf, bench, day0)
    i = 300
    exp = np.prod(1 + ret[i - 1:i + 2]) - np.prod(1 + bret[i - 1:i + 2])
    assert r["car_m1_1"] == pytest.approx(exp)
    assert r["car_0_1"] == pytest.approx(np.prod(1 + ret[i:i + 2]) - np.prod(1 + bret[i:i + 2]))
    assert r["price_m1"] == 25.0
    assert r["size"] == pytest.approx(np.log(1_500_000.0 * 1000))   # firm = both classes
    assert r["bm"] == pytest.approx(600.0 / 1500.0)
    ar = ret - bret
    assert r["vol_pre"] == pytest.approx(np.std(ar[i - 60:i - 4], ddof=1) * np.sqrt(252))
    m0 = day0.to_period("M").to_timestamp()
    s = pd.Series(ret, index=sessions)
    w = s[(s.index >= m0 - pd.DateOffset(months=12)) & (s.index < m0 - pd.DateOffset(months=1))]
    assert r["mom"] == pytest.approx(np.prod(1 + w) - 1)
    assert r["src_mom_end"] < m0 - pd.DateOffset(months=1)
    assert r["src_size_date"] < m0 and r["src_price_date"] < day0


def test_null_day_in_reaction_window_sets_it_missing():
    dsf, bench, sessions, *_ = _crsp(null_on=pd.bdate_range("2014-01-01", periods=320)[299])
    r = _firm(dsf, bench, sessions[300])
    assert np.isnan(r["car_m1_1"])          # Day -1 is null
    assert not np.isnan(r["car_0_1"])


def test_lookahead_check_counts_violations():
    sessions = pd.bdate_range("2015-04-01", periods=20)
    day0 = sessions[10]
    from linguistic_alpha.controls.build import LOOKAHEAD
    row = {c: sessions[9] for c in LOOKAHEAD}
    row.update(src_eps_anndats=day0, src_reaction_end=sessions[11], day0=day0)
    ok = pd.DataFrame([row])
    assert lookahead_check(ok, sessions)["violations"].sum() == 0
    bad = ok.assign(src_consensus_statpers=day0, src_reaction_end=sessions[12])
    v = lookahead_check(bad, sessions).set_index("input")["violations"]
    assert v["src_consensus_statpers"] == 1 and v["src_reaction_end"] == 1 and v.sum() == 2


def test_sue_ts_scale_excludes_same_day_releases():
    # two quarters announced on the same day must not enter each other's scale
    rows, cons = [], []
    for k, (pe, ann, s) in enumerate([("2013-03-31", "2013-04-20", 0.01),
                                      ("2013-06-30", "2013-07-20", 0.03),
                                      ("2013-09-30", "2013-10-20", -0.02),
                                      ("2013-12-31", "2014-01-20", 0.02),
                                      ("2014-03-31", "2014-04-20", 0.40),
                                      ("2014-06-30", "2014-04-20", 0.00)]):
        rows.append(("AAA", pe, ann, 1.0 + s))
        cons.append(("AAA", str((pd.Timestamp(ann) - pd.Timedelta(days=5)).date()), pe,
                     1.0, 0.02, 5, 0, 0, 1.1, 0.9))
    h = eps_surprise_history(_actuals(rows), _summary(cons), NO_ADJ).set_index("pends")
    sd = np.std([0.01, 0.03, -0.02, 0.02], ddof=1)
    assert h.loc[pd.Timestamp("2014-06-30"), "sue_ts"] == pytest.approx(0.0 / sd)
    assert h.loc[pd.Timestamp("2014-03-31"), "sue_ts"] == pytest.approx(0.40 / sd)
    assert (h["sue_ts_scale_date"].dropna() < h["anndats"][h["sue_ts_scale_date"].notna()]).all()


def test_bad_mean_at_date_is_missing():
    g = _guidance([("EPS", "ANN", "2015-04-20", 2015, 12, 3.0, 3.2, 23625.0)])
    out = release_guidance(_event(), g, NO_ADJ, pd.Series({1: 25.0})).iloc[0]
    assert np.isnan(out["guidance_vs_consensus"])
    assert out["guidance_width"] == pytest.approx(0.2 / 25.0)
