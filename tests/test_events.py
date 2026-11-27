"""la events: link rules, share classes, keep_first (placeholder time, 3-day window), release
match and a small end-to-end build on synthetic tables."""

import pandas as pd
from linguistic_alpha.config import load_config
from linguistic_alpha.data import linking, transcripts, universe
from linguistic_alpha.events import build

D = pd.Timestamp


# ---------------------------------------------------------------- links
def test_ciq_gvkey_date_valid():
    calls = pd.DataFrame({"keydevid": [1, 2], "companyid": [7, 7],
                          "call_date_et": [D("2010-06-01"), D("2020-06-01")]})
    g = pd.DataFrame({"companyid": [7.0, 7.0], "gvkey": ["000001", "000002"],
                      "startdate": [None, "2012-01-01"], "enddate": ["2011-12-31", None],
                      "primaryflag": [1, 1]})
    out = linking.link_ciq_to_gvkey(calls, g).set_index("keydevid")["gvkey"]
    assert out.to_dict() == {1: "000001", 2: "000002"}


def test_ccm_linktype_dates_and_candidates():
    ev = pd.DataFrame({"keydevid": [1], "gvkey": ["000001"], "call_date_et": [D("2015-04-20")]})
    ccm = pd.DataFrame({
        "gvkey": ["000001"] * 5,
        "permno": [10.0, 11.0, 12.0, 13.0, 10.0],
        "linktype": ["LC", "LC", "LN", "LU", "LU"],
        "linkprim": ["P", "J", "P", "P", "C"],
        "linkdt": ["2000-01-01", "2000-01-01", "2000-01-01", "2016-01-01", "2000-01-01"],
        "linkenddt": [None, None, None, None, None]})
    out = linking.link_gvkey_to_permno(ev, ccm).set_index("permno")
    # LN dropped, 13 not yet valid; permno 10 keeps its best link (P, LC); J kept as a candidate
    assert sorted(out.index) == [10, 11]
    assert out.loc[10, "linkprim"] == "P" and out.loc[10, "linktype"] == "LC"
    assert out.loc[11, "linkprim"] == "J"


def test_ibes_score_and_open_end():
    ev = pd.DataFrame({"keydevid": [1, 2], "permno": [10, 20],
                       "call_date_et": [D("2026-04-20"), D("2015-04-20")]})
    ic = pd.DataFrame({"ticker": ["AAA", "AAB", "BBB", "BBX"], "permno": [10, 10, 20, 20],
                       "sdate": ["2000-01-01"] * 4,
                       "edate": ["2025-12-31", "2025-12-31", "2025-12-31", "2025-12-31"],
                       "score": [1.0, 2.0, 5.0, 1.0]})
    out = linking.link_permno_to_ibes(ev, ic).set_index("keydevid")
    assert out.loc[1, "ibes_ticker"] == "AAA"      # 2026 date: refresh-date edate is open
    assert out.loc[2, "ibes_ticker"] == "BBX"      # score 5 dropped
    strict = linking.link_permno_to_ibes(ev, ic, open_end=None)
    assert 1 not in set(strict["keydevid"])


# ---------------------------------------------------------------- share classes
def _cands(rows):
    return pd.DataFrame(rows, columns=["keydevid", "permno", "linkprim", "mktcap_m1", "eligible"])


def test_share_class_primary_wins_over_mktcap():
    c = _cands([(1, 10, "P", 1.0, True), (1, 11, "J", 9.0, True)])
    r = universe.choose_share_class(c).iloc[0]
    assert r["permno"] == 10 and r["share_class_choice"] == "primary"


def test_share_class_mktcap_when_no_or_several_primary():
    c = _cands([(1, 10, "C", 1.0, True), (1, 11, "C", 9.0, True),
                (2, 20, "P", 1.0, True), (2, 21, "P", 5.0, True)])
    r = universe.choose_share_class(c).set_index("keydevid")
    assert r.loc[1, "permno"] == 11 and r.loc[1, "share_class_choice"] == "mktcap"
    assert r.loc[2, "permno"] == 21 and r.loc[2, "share_class_choice"] == "mktcap"


def test_share_class_only_member_counts():
    # the primary class is not a member on Day 0: the member secondary class is kept
    c = _cands([(1, 10, "P", 9.0, False), (1, 11, "J", 1.0, True)])
    r = universe.choose_share_class(c).iloc[0]
    assert r["permno"] == 11 and r["share_class_choice"] == "single"


def test_member_on_day0():
    sp = pd.DataFrame({"permno": [10], "mbrstartdt": ["2015-01-01"], "mbrenddt": ["2026-09-30"]})
    ev = pd.DataFrame({"permno": [10, 10, 11],
                       "day0": [D("2014-12-31"), D("2015-01-01"), D("2015-06-01")]})
    assert universe.sp500_member_on(ev, sp).tolist() == [False, True, False]


# ---------------------------------------------------------------- keep_first
def _calls(rows):
    c = pd.DataFrame(rows, columns=["keydevid", "gvkey", "headline", "call_date", "call_time_utc"])
    c["call_date"] = pd.to_datetime(c["call_date"])
    return c


def test_fiscal_label():
    h = pd.Series(["X Corp., Q3 2010 Earnings Call, Oct 26, 2010",
                   "Peabody Energy, 2011 Earnings Call, Jan 24, 2012",
                   "Y Inc., FY 2019 Earnings Call, Mar 1, 2020", "no label"])
    assert universe.fiscal_label(h).tolist()[:3] == ["Q3 2010", "Q4 2011", "Q4 2019"]
    assert pd.isna(universe.fiscal_label(h).iloc[3])


def test_placeholder_time_sorts_last():
    c = _calls([(1, "g", "A, Q1 2015 Earnings Call", "2015-04-20", "00:00:00"),
                (2, "g", "A, Q1 2015 Earnings Call", "2015-04-20", "21:00:00")])
    on = universe.duplicate_calls(c, 3, placeholder_last=True)
    assert on.iloc[0] == 2 and pd.isna(on.iloc[1])         # placeholder call dropped
    off = universe.duplicate_calls(c, 3, placeholder_last=False)
    assert pd.isna(off.iloc[0]) and off.iloc[1] == 1       # 00:00 read as a real time: first


def test_duplicate_window_days():
    c = _calls([(1, "g", "A, Q4 2015 Earnings Call", "2015-01-20", "13:00:00"),
                (2, "g", "A, Q4 2015 Earnings Call", "2015-01-23", "13:00:00"),   # 3 days: dup
                (3, "g", "A, Q4 2015 Earnings Call", "2015-11-20", "13:00:00"),   # new run
                (4, "h", "B, Q4 2015 Earnings Call", "2015-01-21", "13:00:00"),   # other firm
                (5, "g", "no label", "2015-01-20", "14:00:00")])                  # never a dup
    w = universe.duplicate_calls(c, 3)
    assert w.tolist()[1] == 1 and w.isna().tolist() == [True, False, True, True, True]
    anyd = universe.duplicate_calls(c, None)
    assert anyd.iloc[1] == 1 and anyd.iloc[2] == 1


# ---------------------------------------------------------------- release match
def test_release_match_prefers_on_or_before():
    rel = build.release_table(pd.DataFrame({
        "ticker": ["AAA"] * 4, "measure": ["EPS"] * 4, "pdicity": ["QTR"] * 4,
        "pends": ["2014-12-31", "2015-03-31", "2015-03-31", "2014-12-31"],
        "anndats": ["2015-04-17", "2015-04-21", "2015-04-30", "2015-04-21"],
        "anntims": ["07:00:00"] * 4}))
    assert len(rel) == 3                                    # two periods on 04-21: latest pends
    assert rel.loc[rel["anndats"].eq(D("2015-04-21")), "pends"].item() == "2015-03-31"
    c = pd.DataFrame({"keydevid": [1, 2, 3], "permno": [10, 10, 10], "ibes_ticker": ["AAA"] * 3,
                      "call_date_et": [D("2015-04-20"), D("2015-04-21"), D("2015-05-15")]})
    m = build.match_releases(c, rel, (-3, 1)).set_index("keydevid")
    assert m.loc[1, "release_anndats"] == D("2015-04-17")   # before (-3) beats after (+1)
    assert m.loc[2, "release_gap_days"] == 0
    assert 3 not in m.index                                 # nothing within the window


def test_placeholder_unknown_in_transcript_cutoff():
    days = pd.bdate_range("2015-04-01", "2015-04-30")
    v = pd.DataFrame({"keydevid": [1], "mostimportantdateutc": ["2015-04-21"],
                      "mostimportanttimeutc": ["00:00:00"]})
    real = transcripts.day0_close_utc(v, days).iloc[0]
    unk = transcripts.day0_close_utc(v, days, placeholder_unknown=True).iloc[0]
    assert real == D("2015-04-21 20:00", tz="UTC")          # 20:00 ET on 04-20 -> 04-21 close
    assert unk == D("2015-04-22 20:00", tz="UTC")           # date 04-21, next session


# ---------------------------------------------------------------- end to end
def _raw():
    days = pd.bdate_range("2015-01-02", "2015-06-30")
    dsf = pd.concat([pd.DataFrame({"permno": p, "dlycaldt": days.strftime("%Y-%m-%d"),
                                   "dlyret": 0.01, "dlycap": cap})
                     for p, cap in [(10, 1.0), (11, 5.0), (20, 1.0), (30, 1.0)]])
    return {
        "ciq_wrds_gvkey": pd.DataFrame({"companyid": [1.0, 2.0, 3.0],
                                        "gvkey": ["000001", "000002", "000003"],
                                        "startdate": None, "enddate": None, "primaryflag": 1}),
        "crsp_ccmxpf_lnkhist": pd.DataFrame({
            "gvkey": ["000001", "000001", "000002", "000003"], "permno": [10.0, 11.0, 20.0, 30.0],
            "linktype": ["LC", "LC", "LU", "LC"], "linkprim": ["P", "J", "P", "P"],
            "linkdt": "2000-01-01", "linkenddt": None}),
        "wrdsapps_ibcrsphist": pd.DataFrame({
            "ticker": ["AAA", "AAB", "CCC"], "permno": [10, 11, 30], "sdate": "2000-01-01",
            "edate": "2025-12-31", "score": [1.0, 1.0, 1.0]}),
        "crsp_msp500list_v2": pd.DataFrame({"permno": [10, 11, 20], "mbrstartdt": "2000-01-01",
                                            "mbrenddt": "2026-09-30"}),
        "comp_co_hgic": pd.DataFrame({"gvkey": ["000001"], "gsector": ["45"], "ggroup": ["4520"],
                                      "gind": ["452020"], "gsubind": ["45202030"],
                                      "indfrom": "2000-01-01", "indthru": None}),
        "ibes_actu_epsus": pd.DataFrame({
            "ticker": ["AAA", "AAB", "CCC"], "measure": "EPS", "pdicity": "QTR",
            "pends": "2015-03-31", "anndats": "2015-04-20", "anntims": ["07:00:00", "07:00:00",
                                                                       None]}),
        "crsp_dsf_v2": dsf,
    }


def _versions():
    rows = [  # keydevid, companyid, date, time, headline, transcriptid, created date, time
        (1, 1, "2015-04-20", "13:00:00", "Alpha Inc., Q1 2015 Earnings Call, Apr 20, 2015", 100,
         "2015-04-20", "18:00:00"),
        (2, 1, "2015-04-21", "14:00:00", "Alpha Inc., Q1 2015 Earnings Call, Apr 21, 2015", 200,
         "2015-04-21", "18:00:00"),
        (3, 2, "2015-04-20", "13:00:00", "Beta Co., Q1 2015 Earnings Call, Apr 20, 2015", 300,
         "2015-04-20", "18:00:00"),
        (4, 3, "2015-04-20", "13:00:00", "Gamma, Q1 2015 Earnings Call, Apr 20, 2015", 400,
         "2015-04-20", "18:00:00"),
        (5, 9, "2015-04-20", "13:00:00", "Delta, Q1 2015 Earnings Call, Apr 20, 2015", 500,
         "2015-04-20", "18:00:00"),
    ]
    v = pd.DataFrame(rows, columns=["keydevid", "companyid", "mostimportantdateutc",
                                    "mostimportanttimeutc", "headline", "transcriptid",
                                    "transcriptcreationdate_utc", "transcriptcreationtime_utc"])
    v["keydeveventtypename"] = "Earnings Calls"
    v["companyname"] = v["headline"].str.split(",").str[0]
    v["transcriptpresentationtypename"] = "Final"
    v["transcriptcollectiontypename"] = "Edited"
    return v


def test_build_events_end_to_end():
    cfg = load_config().raw
    cfg = {**cfg, "sample": {**cfg["sample"], "start": "2015-01-01", "end": "2015-12-31"}}
    ev, ex, log = build.build_events(_versions(), _raw(), cfg)
    assert ev["keydevid"].tolist() == [1]
    r = ev.iloc[0]
    assert r["permno"] == 10 and r["ibes_ticker"] == "AAA" and r["transcriptid"] == 100
    assert r["day0"] == D("2015-04-20") and r["call_lag_sessions"] == 0
    assert r["gsector"] == "45"
    reasons = ex.set_index("keydevid")["exclusion_reason"].to_dict()
    assert reasons == {2: "duplicate_call", 3: "no_ibes_ticker", 4: "not_sp500_member",
                       5: "no_gvkey"}
    att = pd.DataFrame(log["attrition"], columns=["step", "before", "after"])
    assert att["after"].iloc[-1] == 1 and att["before"].iloc[0] == 5


def test_apply_exclusions_call_lag_and_history():
    base = dict(event_date=D("2015-04-20"), gvkey="g", permno=10, sp500_member=True,
                ibes_ticker="AAA", release_anndats=D("2015-04-20"), release_pends="2015-03-31",
                headline="A, Q1 2015 Earnings Call", call_date=D("2015-04-20"),
                call_time_utc="13:00:00", gind="452020", transcriptid=100, n_prior_returns=70,
                call_lag_sessions=0)
    rows = [dict(base, keydevid=1),
            dict(base, keydevid=2, gvkey="h", ibes_ticker="BBB", call_lag_sessions=2),
            dict(base, keydevid=3, gvkey="i", ibes_ticker="CCC", call_lag_sessions=-1),
            dict(base, keydevid=4, gvkey="j", ibes_ticker="DDD", n_prior_returns=10),
            dict(base, keydevid=5, gvkey="k", ibes_ticker="EEE", transcriptid=pd.NA),
            dict(base, keydevid=6, gvkey="l", ibes_ticker="AAA", headline="other label")]
    cfg = load_config().raw
    out = universe.apply_exclusions(pd.DataFrame(rows), cfg).set_index("keydevid")
    got = out["exclusion_reason"]
    assert pd.isna(got[1])
    assert got[2] == "call_lag" and got[3] == "call_lag"
    assert got[4] == "short_return_history" and got[5] == "no_transcript_version"
    assert got[6] == "duplicate_release" and out.loc[6, "duplicate_release_of_keydevid"] == 1


def test_config_switches_exist():
    ev = load_config().raw["events"]
    assert {"placeholder_time_last", "duplicate_window_days", "release_match_days"} <= set(ev)


def test_joint_call_picks_the_member_company():
    # keydevid 1 also listed under company 3 (not an S&P 500 member): company 1 is kept
    v = _versions()
    extra = v[v["keydevid"].eq(1)].assign(companyid=3, companyname="Gamma")
    v = pd.concat([extra, v], ignore_index=True)
    cfg = load_config().raw
    cfg = {**cfg, "sample": {**cfg["sample"], "start": "2015-01-01", "end": "2015-12-31"}}
    ev, ex, log = build.build_events(v, _raw(), cfg)
    r = ev.set_index("keydevid").loc[1]
    assert r["companyid"] == 1 and r["permno"] == 10 and r["companyname"] == "Alpha Inc."
    assert log["joint_calls"] == 1
