"""
sp500_pipeline.py  -  final data pipeline for the earnings-call (linguistic alpha) project

Builds ONE modeling table for every S&P 500 earnings call from START_YEAR on:
one row per call with company IDs, sector, Day 0, targets (CAR, volatility),
controls (SUE, size, book-to-market, momentum, ...) and language features,
plus the transcript text in a separate file.

Stages (each saves its output and is skipped next time, so a crash just means re-running):
  1. pull     download raw data from WRDS in small pieces            -> raw/
  2. panel    one row per earnings call, all company IDs linked      -> event_panel.parquet
  3. targets  Day 0, CAR targets, controls, SUE, sector              -> events_numeric.parquet
  4. text     transcript text + language features, one year at a time -> text/, features/
  5. final    merge everything                                        -> model_table.*

WHERE TO RUN
  Your own computer:  python sp500_pipeline.py            (see the LOCAL_SETUP guide)
  WRDS Cloud:         qsub sp500_pipeline.sh
  Quick test:         python sp500_pipeline.py --tickers AAPL MSFT

Needs: Python 3, pandas, numpy, pyarrow, wrds (stage 1 only).
Batch jobs need a ~/.pgpass file (run wrds.Connection() once interactively first).

Optional: the Loughran-McDonald Master Dictionary CSV (University of Notre Dame,
Software Repository for Accounting and Finance) in the project folder or the
folder you run from. Without it, tone/hedging features are skipped (re-run the
text stage later; files built without it are rebuilt automatically).
"""

import gc
import glob
import os
import re
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Settings
START_YEAR = 2010
END_YEAR = 2026
TICKERS = None                    # None = all S&P 500 members; or e.g. ["AAPL", "MSFT"] for a test
WRDS_USERNAME = "bhuvighosh3"
STAGES = ["pull", "panel", "targets", "text", "final"]   # remove "pull" to rebuild from saved files
# Time split (by call date). Returns end 2024-12-31, so targets exist for calls up to about Nov 2024.
HOLDOUT_START = "2023-01-01"      # 2010-2022 = train, 2023-2024 = holdout test (used once, at the end)
MIN_HISTORY_CALLS = 4             # a company is tested only if it has at least this many earlier calls
                                  # with a target (also needed for the change-vs-prior-calls features)
TEXT_BATCH = 300                  # transcripts per WRDS query (smaller = less risk of timeouts)
WRITE_TEXT_CSV = False            # True also writes a CSV with transcript text (very large for all S&P 500)

ROOT = "/content" if os.path.isdir("/content") else os.path.expanduser("~")

# Command-line options (optional; the settings above are used when they are not given), e.g.
#   python sp500_pipeline.py                                   all S&P 500, all stages
#   python sp500_pipeline.py --tickers AAPL MSFT               a quick test
#   python sp500_pipeline.py --folder D:/earnings_data         choose where data is saved
#   python sp500_pipeline.py --stages panel targets text final rebuild without downloading
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Earnings-call pipeline for S&P 500 companies (WRDS).")
    ap.add_argument("--tickers", nargs="+", help="only these tickers (default: all S&P 500 members)")
    ap.add_argument("--stages", nargs="+", choices=["pull", "panel", "targets", "text", "final"],
                    help="stages to run (default: all)")
    ap.add_argument("--folder", help="folder for the project data (default: your home folder)")
    ap.add_argument("--user", help="WRDS username")
    ap.add_argument("--start-year", type=int, help="first year of calls (default 2010)")
    args = ap.parse_args()
    if args.tickers:
        TICKERS = [t.upper() for t in args.tickers]
    if args.stages:
        STAGES = args.stages
    if args.folder:
        ROOT = os.path.expanduser(args.folder)
    if args.user:
        WRDS_USERNAME = args.user
    if args.start_year:
        START_YEAR = args.start_year

NAME = "sp500" if not TICKERS else "_".join(sorted(TICKERS))[:60]
BASE = os.environ.get("EARNINGS_BASE", os.path.join(ROOT, f"earnings_{NAME}"))
RAW, TEXT_DIR, FEAT_DIR = f"{BASE}/raw", f"{BASE}/text", f"{BASE}/features"
for d in (RAW, TEXT_DIR, FEAT_DIR):
    os.makedirs(d, exist_ok=True)

GICS = {"10": "Energy", "15": "Materials", "20": "Industrials", "25": "Consumer Discretionary",
        "30": "Consumer Staples", "35": "Health Care", "40": "Financials",
        "45": "Information Technology", "50": "Communication Services",
        "55": "Utilities", "60": "Real Estate"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def path(name):
    return os.path.join(RAW, f"{name}.parquet")


def read_many(pattern):
    files = sorted(glob.glob(os.path.join(RAW, pattern)))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_dt(df, *cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df


# 1. PULL
def stage_pull():
    import wrds
    state = {"db": wrds.Connection(wrds_username=WRDS_USERNAME)}

    def sql(query, params=None, tries=4):
        for i in range(tries):
            try:
                return state["db"].raw_sql(query, params=params)
            except Exception as e:                      # timeouts, "conflict with recovery", dropped link
                log(f"  query failed ({str(e).splitlines()[0][:90]}); retry {i + 1}/{tries - 1}")
                time.sleep(20 * (i + 1))
                try:
                    state["db"].close()
                except Exception:
                    pass
                state["db"] = wrds.Connection(wrds_username=WRDS_USERNAME)
        raise RuntimeError("query kept failing; re-run later (saved pieces are kept)")

    def pull(name, query, params=None):
        if os.path.exists(path(name)):
            return
        df = sql(query, params)
        for c in df.select_dtypes("float64").columns:
            if c not in ("prc", "cfacpr"):
                df[c] = df[c].astype("float32")
        df.to_parquet(path(name), index=False)
        log(f"saved {name}: {len(df):,} rows")
        del df
        gc.collect()

    a, b = f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"
    # Universe: every stock that was in the S&P 500 at any time from START_YEAR on
    permnos = f"select distinct permno from crsp.msp500list where ending >= '{a}' and start <= '{b}'"
    if TICKERS:
        names = sql("select permno, ticker, comnam, nameenddt from crsp.stocknames where ticker in %(t)s",
                    {"t": tuple(TICKERS)})
        names = names.sort_values("nameenddt").groupby("ticker").tail(1)
        log("companies found:\n" + names[["ticker", "permno", "comnam"]].to_string(index=False))
        permnos += " and permno in (" + ", ".join(str(int(p)) for p in names["permno"]) + ")"
    gvkeys = f"""select distinct gvkey from crsp.ccmxpf_lnkhist where lpermno in ({permnos})
                 and linktype in ('LU','LC') and linkprim in ('P','C')"""
    tickers = f"select distinct ticker from wrdsapps.ibcrsphist where permno in ({permnos})"
    ciq_ids = f"select distinct companyid from ciq.wrds_gvkey where gvkey in ({gvkeys})"

    # Universe, links, company info
    pull("sp500_membership", f"select permno, start, ending from crsp.msp500list where permno in ({permnos})")
    pull("link_ccm", f"""select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt
                         from crsp.ccmxpf_lnkhist where lpermno in ({permnos})
                         and linktype in ('LU','LC') and linkprim in ('P','C')""")
    pull("link_ibes_crsp", f"select ticker, permno, sdate, edate, score from wrdsapps.ibcrsphist "
                           f"where permno in ({permnos})")
    pull("link_ciq_gvkey", f"select * from ciq.wrds_gvkey where gvkey in ({gvkeys})")
    pull("company", f"select gvkey, conm, gsector, gind, sic from comp.company where gvkey in ({gvkeys})")

    # Returns (daily, one year per file) and the S&P 500 benchmark
    pull("sp500_index_daily", "select date, sprtrn from crsp.dsi where date >= %(d)s",
         {"d": f"{START_YEAR - 1}-01-01"})
    for y in range(START_YEAR - 1, END_YEAR + 1):
        pull(f"crsp_daily_{y}", f"""select permno, date, ret, prc, vol, shrout, cfacpr from crsp.dsf
                                    where date between %(a)s and %(b)s and permno in ({permnos})""",
             {"a": f"{y}-01-01", "b": f"{y}-12-31"})
    pull("crsp_delist", f"select permno, dlstdt, dlret from crsp.dsedelist where permno in ({permnos})")

    # Fundamentals
    pull("compustat_fundq", f"""select gvkey, datadate, fyearq, fqtr, rdq, epspxq, ceqq, cshoq, prccq
                                from comp.fundq where gvkey in ({gvkeys}) and datadate >= %(d)s
                                and indfmt = 'INDL' and datafmt = 'STD' and consol = 'C' and popsrc = 'D'""",
         {"d": f"{START_YEAR - 3}-01-01"})

    # Analyst data: UNADJUSTED = EPS as reported at the time (no split restating / rounding)
    pull("ibesu_actuals", f"""select ticker, pends, pdicity, value as actual_eps, anndats, anntims
                              from ibes.actu_epsus where ticker in ({tickers}) and pdicity = 'QTR'
                              and anndats >= %(d)s""", {"d": f"{START_YEAR - 1}-06-01"})
    for y in range(START_YEAR - 1, END_YEAR + 1):
        pull(f"ibesu_consensus_{y}", f"""select ticker, statpers, fpedats, medest, meanest, stdev, numest
                                         from ibes.statsumu_epsus where ticker in ({tickers}) and fpi = '6'
                                         and statpers between %(a)s and %(b)s""",
             {"a": f"{y}-01-01", "b": f"{y}-12-31"})

    # Transcripts: section codes, list of earnings calls, then text in batches
    pull("component_types", "select * from ciq.ciqtranscriptcomponenttype")
    pull("transcript_calls", f"""select companyid, companyname, keydevid, transcriptid, headline,
                                        mostimportantdateutc as call_date, mostimportanttimeutc as call_time_utc
                                 from ciq.wrds_transcript_detail
                                 where keydeveventtypeid = 48 and companyid in ({ciq_ids})
                                   and mostimportantdateutc >= %(d)s""", {"d": a})
    calls = latest_versions(pd.read_parquet(path("transcript_calls")))
    calls["year"] = pd.to_datetime(calls["call_date"]).dt.year
    for y, grp in calls.groupby("year"):
        ids = sorted(int(t) for t in grp["transcriptid"])
        for k in range(0, len(ids), TEXT_BATCH):
            pull(f"transcript_text_{y}_part{k // TEXT_BATCH:03d}",
                 """select transcriptid, componentorder, transcriptcomponenttypeid, transcriptpersonid,
                           componenttext from ciq.ciqtranscriptcomponent where transcriptid in %(ids)s""",
                 {"ids": tuple(ids[k:k + TEXT_BATCH])})
        if not os.path.exists(path(f"transcript_people_{y}")):
            pids = read_many(f"transcript_text_{y}_part*.parquet")["transcriptpersonid"].dropna()
            pids = sorted(pids.astype(int).unique().tolist())
            parts = [sql("""select transcriptpersonid, transcriptpersonname as speaker,
                                   speakertypename as role, companyofperson as speaker_company
                            from ciq.wrds_transcript_person where transcriptpersonid in %(ids)s""",
                         {"ids": tuple(pids[i:i + 5000])}) for i in range(0, len(pids), 5000)]
            people = pd.concat(parts) if parts else pd.DataFrame(
                columns=["transcriptpersonid", "speaker", "role", "speaker_company"])
            people.drop_duplicates("transcriptpersonid").to_parquet(path(f"transcript_people_{y}"), index=False)
            log(f"saved transcript_people_{y}: {len(people):,} rows")
    state["db"].close()
    log("pull complete")


def latest_versions(calls):
    """One row per earnings call: the latest transcript version (highest transcriptid)."""
    return calls.sort_values("transcriptid").groupby("keydevid").tail(1)


# 2. PANEL
def stage_panel():
    calls = latest_versions(pd.read_parquet(path("transcript_calls")))
    calls = to_dt(calls, "call_date")

    # Capital IQ company -> Compustat gvkey (date-valid when the table has dates)
    g = pd.read_parquet(path("link_ciq_gvkey"))
    keep = ["companyid", "gvkey"] + [c for c in ("startdate", "enddate") if c in g.columns]
    panel = calls.merge(g[keep].drop_duplicates(), on="companyid")
    if "startdate" in panel.columns:
        sd = pd.to_datetime(panel["startdate"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        ed = pd.to_datetime(panel["enddate"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        panel = panel[(panel["call_date"] >= sd) & (panel["call_date"] <= ed)]

    # gvkey -> CRSP permno (primary share class, link valid on the call date)
    ccm = to_dt(pd.read_parquet(path("link_ccm")), "linkdt", "linkenddt")
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    ccm["prim"] = (ccm["linkprim"] == "P").astype(int)
    panel = panel.merge(ccm[["gvkey", "permno", "linkdt", "linkenddt", "prim"]], on="gvkey")
    panel = panel[(panel["call_date"] >= panel["linkdt"]) & (panel["call_date"] <= panel["linkenddt"])]

    # In the S&P 500 on the call date. The last membership spell in CRSP ends at the data's end date;
    # treat it as continuing so calls after CRSP's last update are kept (they just have no returns yet).
    sp = to_dt(pd.read_parquet(path("sp500_membership")), "start", "ending")
    sp.loc[sp["ending"] >= sp["ending"].max(), "ending"] = pd.Timestamp("2099-12-31")
    panel = panel.merge(sp, on="permno")
    panel = panel[(panel["call_date"] >= panel["start"]) & (panel["call_date"] <= panel["ending"])]
    panel = (panel.sort_values(["keydevid", "prim"], ascending=[True, False])
                  .drop_duplicates("keydevid"))

    # permno -> I/B/E/S ticker valid on the call date (best match score)
    ib = to_dt(pd.read_parquet(path("link_ibes_crsp")), "sdate", "edate")
    m = panel[["keydevid", "permno", "call_date"]].merge(ib, on="permno")
    m = m[(m["call_date"] >= m["sdate"]) & (m["call_date"] <= m["edate"])]
    m = m.sort_values(["keydevid", "score"]).drop_duplicates("keydevid")[["keydevid", "ticker"]]
    panel = panel.merge(m, on="keydevid", how="left")

    # Sector
    comp = pd.read_parquet(path("company"))
    comp["sector"] = comp["gsector"].astype(str).str[:2].map(GICS)
    panel = panel.merge(comp[["gvkey", "sector", "gind", "sic"]].drop_duplicates("gvkey"), on="gvkey", how="left")

    cols = ["keydevid", "transcriptid", "companyid", "companyname", "headline", "call_date",
            "call_time_utc", "gvkey", "permno", "ticker", "sector", "gind", "sic"]
    panel = panel[cols].sort_values(["call_date", "permno"]).reset_index(drop=True)
    panel.to_parquet(f"{BASE}/event_panel.parquet", index=False)
    log(f"event panel: {len(panel):,} calls, {panel['permno'].nunique():,} companies, "
        f"{panel['ticker'].isna().sum():,} without I/B/E/S ticker")


# 3. TARGETS AND CONTROLS (vectorized)
def stage_targets():
    panel = to_dt(pd.read_parquet(f"{BASE}/event_panel.parquet"), "call_date")
    spx = to_dt(pd.read_parquet(path("sp500_index_daily")), "date").sort_values("date")
    days = spx["date"].values.astype("datetime64[ns]")
    nd = len(days)
    crsp = to_dt(read_many("crsp_daily_*.parquet"), "date")
    crsp = crsp[crsp["date"].isin(spx["date"])]

    # Add delisting returns on the delisting day so failed firms' final losses are counted
    dl = to_dt(pd.read_parquet(path("crsp_delist")), "dlstdt").dropna(subset=["dlret", "dlstdt"])
    dl = dl[dl["permno"].isin(panel["permno"])]
    delisted_day = {}
    if len(dl):
        di = np.searchsorted(days, dl["dlstdt"].values)            # first trading day on/after delisting
        dl = dl[di < nd].assign(date=days[di[di < nd]])
        have = crsp.merge(dl[["permno", "date"]], on=["permno", "date"])[["permno", "date"]]
        extra = dl.merge(have, on=["permno", "date"], how="left", indicator=True)
        extra = extra[extra["_merge"] == "left_only"][["permno", "date"]].assign(ret=0.0)
        crsp = pd.concat([crsp, extra], ignore_index=True)        # delisting day with no price row
        crsp = crsp.merge(dl[["permno", "date", "dlret"]], on=["permno", "date"], how="left")
        has = crsp["dlret"].notna()
        crsp.loc[has, "ret"] = (1 + crsp.loc[has, "ret"].fillna(0)) * (1 + crsp.loc[has, "dlret"]) - 1
        crsp = crsp.drop(columns="dlret")
        delisted_day = dict(zip(dl["permno"], np.searchsorted(days, dl["date"].values)))

    # Arrays aligned to the trading calendar: one row per stock, one column per trading day
    permnos = np.sort(panel["permno"].dropna().unique())
    pidx = {p: i for i, p in enumerate(permnos)}
    crsp = crsp[crsp["permno"].isin(pidx)]
    r = crsp["permno"].map(pidx).values
    c = np.searchsorted(days, crsp["date"].values)

    def grid(values):
        a = np.full((len(permnos), nd), np.nan)
        a[r, c] = values
        return a

    ret = grid(crsp["ret"].values)
    ar = ret - spx["sprtrn"].values[None, :]
    mcap = grid(crsp["prc"].abs().values * crsp["shrout"].values / 1000)        # $ millions
    price = grid(crsp["prc"].abs().values)                                      # unadjusted price
    cfac = pd.DataFrame(grid(crsp["cfacpr"].values)).ffill(axis=1).bfill(axis=1).values
    del crsp
    gc.collect()

    def csum(a):
        z = np.nan_to_num(a)
        return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)

    ok = ~np.isnan(ar)
    cs_ar, cs_ar2, cs_n = csum(ar), csum(ar ** 2), csum(ok.astype(float))
    cs_lr = csum(np.log1p(np.where(np.isnan(ret), 0, ret)))
    cs_rn = csum((~np.isnan(ret)).astype(float))

    # Day 0: same day if the call starts before 4:00 pm New York time on a trading day, else next trading day
    t = pd.to_datetime(panel["call_date"].dt.strftime("%Y-%m-%d") + " " + panel["call_time_utc"].astype(str),
                       errors="coerce", utc=True)
    after_close = (t.dt.tz_convert("America/New_York").dt.hour >= 16).fillna(False).values
    cd = panel["call_date"].dt.normalize().values
    i = np.searchsorted(days, cd)
    trading = (i < nd) & (days[np.minimum(i, nd - 1)] == cd)
    d0 = i + (trading & after_close).astype(int)
    row = panel["permno"].map(pidx).fillna(-1).astype(int).values
    valid = (row >= 0) & (d0 < nd)
    panel["day0"] = pd.NaT
    panel.loc[valid, "day0"] = days[d0[valid]]
    panel["day0"] = pd.to_datetime(panel["day0"]).astype("datetime64[ns]")
    rr = np.where(valid, row, 0)

    def window_sum(cs, a, b):
        lo, hi = d0 + a, d0 + b + 1
        good = valid & (lo >= 0) & (hi <= nd)
        out = np.full(len(panel), np.nan)
        out[good] = cs[rr[good], hi[good]] - cs[rr[good], lo[good]]
        return out

    dlday = panel["permno"].map(delisted_day).fillna(-1).astype(int).values
    for name, (a, b) in {"car_0_1": (0, 1), "car_2_5": (2, 5), "car_2_20": (2, 20), "car_2_60": (2, 60)}.items():
        s, n = window_sum(cs_ar, a, b), window_sum(cs_n, a, b)
        delisted_in = (dlday >= d0 + a) & (dlday < d0 + b)              # stock stopped trading inside the window
        lo, hi = d0 + a, np.minimum(d0 + b, dlday) + 1
        g = valid & delisted_in & (lo >= 0) & (hi <= nd)
        s[g] = cs_ar[rr[g], hi[g]] - cs_ar[rr[g], lo[g]]
        panel[name] = np.where((n >= 0.8 * (b - a + 1)) | g, s, np.nan)

    def window_vol(a, b, need):
        s, s2, n = window_sum(cs_ar, a, b), window_sum(cs_ar2, a, b), window_sum(cs_n, a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (s2 - s ** 2 / n) / (n - 1)
        return np.where(n >= need, np.sqrt(np.maximum(var, 0)) * np.sqrt(252), np.nan)

    panel["vol_post_2_20"] = window_vol(2, 20, 15)
    panel["vol_pre"] = window_vol(-60, -5, 40)

    # Size and momentum, measured at the end of the month before the call
    ms = panel["call_date"].dt.to_period("M").dt.to_timestamp().values
    last_before = np.searchsorted(days, ms) - 1                                   # last trading day before month start
    mc_ff = pd.DataFrame(mcap).ffill(axis=1, limit=10).values
    ok_m = (row >= 0) & (last_before >= 0)
    panel["mcap_m"] = np.where(ok_m, mc_ff[rr, np.clip(last_before, 0, nd - 1)], np.nan)
    panel["size"] = np.log(panel["mcap_m"] * 1e6)
    start12 = np.searchsorted(days, (pd.DatetimeIndex(ms) - pd.DateOffset(months=12)).values)
    end1 = np.searchsorted(days, (pd.DatetimeIndex(ms) - pd.DateOffset(months=1)).values)   # exclusive
    good = (row >= 0) & (start12 < end1)
    lr = np.full(len(panel), np.nan)
    nn = np.zeros(len(panel))
    lr[good] = cs_lr[rr[good], end1[good]] - cs_lr[rr[good], start12[good]]
    nn[good] = cs_rn[rr[good], end1[good]] - cs_rn[rr[good], start12[good]]
    panel["mom"] = np.where(nn >= 150, np.expm1(lr), np.nan)

    # Book-to-market: latest book equity PUBLISHED before Day 0
    fq = to_dt(pd.read_parquet(path("compustat_fundq")), "rdq").dropna(subset=["rdq", "ceqq"])
    fq = fq.sort_values("rdq")[["gvkey", "rdq", "ceqq"]]
    left = panel[["keydevid", "gvkey", "day0"]].dropna(subset=["day0"]).sort_values("day0")
    bm = pd.merge_asof(left, fq, left_on="day0", right_on="rdq", by="gvkey",
                       allow_exact_matches=False, direction="backward", tolerance=pd.Timedelta(days=200))
    panel = panel.merge(bm[["keydevid", "ceqq"]], on="keydevid", how="left")
    panel["bm"] = np.where(panel["ceqq"] > 0, panel["ceqq"] / panel["mcap_m"], np.nan)
    panel = panel.drop(columns="ceqq")

    # SUE from UNADJUSTED I/B/E/S: actual EPS vs. the last consensus before the announcement
    act = to_dt(read_many("ibesu_actuals.parquet"), "pends", "anndats")
    cons = to_dt(read_many("ibesu_consensus_*.parquet"), "statpers", "fpedats")
    if len(act) and len(cons):
        act = act[act["pdicity"] == "QTR"].dropna(subset=["anndats", "actual_eps"])
        m = panel[["keydevid", "ticker", "call_date", "permno"]].dropna(subset=["ticker"]).merge(act, on="ticker")
        m = m[(m["anndats"] - m["call_date"]).abs() <= pd.Timedelta(days=5)]
        m["gap"] = (m["anndats"] - m["call_date"]).abs()
        m = m.sort_values(["keydevid", "gap"]).drop_duplicates("keydevid")
        cons = cons.dropna(subset=["medest"]).sort_values("statpers")
        m = pd.merge_asof(m.sort_values("anndats"), cons.rename(columns={"fpedats": "pends"}),
                          left_on="anndats", right_on="statpers", by=["ticker", "pends"],
                          allow_exact_matches=False, direction="backward", tolerance=pd.Timedelta(days=120))
        m = m.dropna(subset=["medest"])
        # A split between the consensus date and the announcement: put the consensus on the new share basis
        mrow = m["permno"].map(pidx).fillna(-1).astype(int).values
        si = np.clip(np.searchsorted(days, m["statpers"].values, side="right") - 1, 0, nd - 1)
        ai = np.clip(np.searchsorted(days, m["anndats"].values, side="right") - 1, 0, nd - 1)
        k = np.ones(len(m))
        okr = mrow >= 0
        with np.errstate(invalid="ignore", divide="ignore"):
            k[okr] = cfac[mrow[okr], ai[okr]] / cfac[mrow[okr], si[okr]]
        k = np.where(np.isfinite(k) & (k > 0), k, 1.0)
        med, sd = m["medest"].values * k, m["stdev"].values * k
        pre_price = np.full(len(m), np.nan)
        pre_price[okr] = price[mrow[okr], np.clip(ai[okr] - 1, 0, nd - 1)]
        surprise = m["actual_eps"].values - med
        with np.errstate(invalid="ignore", divide="ignore"):
            sue = pd.DataFrame({
                "keydevid": m["keydevid"].values, "fiscal_period_end": m["pends"].values,
                "actual_eps": m["actual_eps"].values, "consensus_eps": med, "eps_surprise": surprise,
                "sue": np.where(sd > 0, surprise / sd, np.nan),
                "sue_price": np.where(pre_price > 0, surprise / pre_price, np.nan),
                "surprise_pct": np.where(med != 0, surprise / np.abs(med), np.nan),
                "n_analysts": m["numest"].values,
                "dispersion": np.where(med != 0, sd / np.abs(med), np.nan),
                "split_adjusted_consensus": k != 1.0})
        panel = panel.merge(sue, on="keydevid", how="left")

    panel.to_parquet(f"{BASE}/events_numeric.parquet", index=False)
    log(f"targets/controls: {panel['car_2_20'].notna().sum():,} calls with CAR[+2,+20], "
        f"{panel.get('sue', pd.Series(dtype=float)).notna().sum():,} with SUE; "
        f"returns end {pd.Timestamp(days[-1]).date()}")


# 4. TEXT AND LANGUAGE FEATURES (one year at a time)
WORD = re.compile(r"[A-Za-z]+")
NUMERIC_SECTION = {1: "operator", 2: "prepared", 3: "question", 4: "answer"}


def section_map():
    """Map S&P's component type codes to operator / prepared / question / answer, using the official names."""
    if not os.path.exists(path("component_types")):
        return NUMERIC_SECTION
    t = pd.read_parquet(path("component_types"))
    idc = [c for c in t.columns if c.endswith("typeid")][0]
    namec = [c for c in t.columns if c.endswith("typename")][0]
    out = {}
    for code, name in zip(t[idc], t[namec]):
        n = str(name).lower()
        out[int(code)] = ("operator" if "operator" in n else "other" if "unknown" in n
                          else "question" if "question" in n
                          else "answer" if "answer" in n
                          else "prepared" if ("presenter" in n or "presentation" in n) else "other")
    return out


def load_lm():
    files = (glob.glob(f"{BASE}/*MasterDictionary*.csv") + glob.glob(f"{ROOT}/*MasterDictionary*.csv")
             + glob.glob("*MasterDictionary*.csv"))
    if not files:
        return None
    lm = pd.read_csv(files[0])
    lm["Word"] = lm["Word"].astype(str).str.upper()
    lists = {c.lower(): set(lm.loc[lm[c] != 0, "Word"])
             for c in ["Positive", "Negative", "Uncertainty", "Weak_Modal"] if c in lm.columns}
    lists["hedge"] = lists.get("uncertainty", set()) | lists.get("weak_modal", set())
    return lists


def stats(texts, lm):
    words = [w.upper() for t in texts for w in WORD.findall(t or "")]
    out = {"words": len(words)}
    if lm:
        cnt = Counter(words)
        for cat, vocab in lm.items():
            out[cat] = sum(n for w, n in cnt.items() if w in vocab)
    return out


def tone(s):
    p, n = s.get("positive", 0), s.get("negative", 0)
    return (p - n) / (p + n) if p + n else np.nan


def per1000(s, cat):
    return 1000 * s.get(cat, 0) / s["words"] if s["words"] else np.nan


def stage_text():
    keep_ids = set(pd.read_parquet(f"{BASE}/event_panel.parquet", columns=["transcriptid"])["transcriptid"])
    smap, lm = section_map(), load_lm()
    log("Loughran-McDonald dictionary " + ("loaded" if lm else "NOT found: tone/hedging skipped"))
    years = sorted({int(re.search(r"transcript_text_(\d{4})_part", f).group(1))
                    for f in glob.glob(f"{RAW}/transcript_text_*_part*.parquet")})
    for y in years:
        feat_file = f"{FEAT_DIR}/features_{y}{'_lm' if lm else ''}.parquet"
        if os.path.exists(feat_file) and os.path.exists(f"{TEXT_DIR}/text_{y}.parquet"):
            continue
        txt = read_many(f"transcript_text_{y}_part*.parquet")
        txt = txt[txt["transcriptid"].isin(keep_ids)]
        txt["section"] = txt["transcriptcomponenttypeid"].map(smap).fillna("other")
        ppl_file = path(f"transcript_people_{y}")
        if os.path.exists(ppl_file):
            txt = txt.merge(pd.read_parquet(ppl_file)[["transcriptpersonid", "role"]],
                            on="transcriptpersonid", how="left")
        else:
            txt["role"] = np.nan
        txt = txt.sort_values(["transcriptid", "componentorder"])
        texts, feats = [], []
        for tid, g in txt.groupby("transcriptid", sort=False):
            prep = g.loc[g["section"] == "prepared", "componenttext"].fillna("").tolist()
            ques = g.loc[g["section"] == "question", "componenttext"].fillna("").tolist()
            ans_rows = g[g["section"] == "answer"]
            ans = ans_rows["componenttext"].fillna("").tolist()
            qa = g[g["section"].isin(["question", "answer"])]
            texts.append({"transcriptid": tid, "prepared_text": "\n\n".join(prep),
                          "qa_text": "\n\n".join(qa["section"].str.upper() + ": " + qa["componenttext"].fillna(""))})
            sp, sa, sq = stats(prep, lm), stats(ans, lm), stats(ques, lm)
            role = ans_rows["role"].astype(str).str.lower()
            f = {"transcriptid": tid, "n_questions": len(ques), "n_answers": len(ans),
                 "words_prepared": sp["words"], "words_answers": sa["words"], "words_questions": sq["words"],
                 "avg_answer_words": sa["words"] / len(ans) if ans else np.nan,
                 "share_other_sections": float((g["section"] == "other").mean()),
                 "share_answers_by_execs": float(role.str.contains("exec").mean()) if len(role) else np.nan}
            if lm:
                f.update({"tone_prepared": tone(sp), "tone_answers": tone(sa),
                          "hedging_prepared": per1000(sp, "hedge"), "hedging_answers": per1000(sa, "hedge"),
                          "uncertainty_answers": per1000(sa, "uncertainty")})
                f["tone_gap"] = f["tone_answers"] - f["tone_prepared"]
            feats.append(f)
        pd.DataFrame(texts).to_parquet(f"{TEXT_DIR}/text_{y}.parquet", index=False)
        pd.DataFrame(feats).to_parquet(feat_file, index=False)
        log(f"text {y}: {len(feats):,} calls")
        del txt, texts, feats
        gc.collect()


# 5. FINAL TABLE
def stage_final():
    ev = pd.read_parquet(f"{BASE}/events_numeric.parquet")
    lm_files = sorted(glob.glob(f"{FEAT_DIR}/features_*_lm.parquet"))
    plain = sorted(f for f in glob.glob(f"{FEAT_DIR}/features_*.parquet") if not f.endswith("_lm.parquet"))
    use = lm_files if lm_files else plain
    feats = pd.concat([pd.read_parquet(f) for f in use], ignore_index=True) if use else pd.DataFrame(
        columns=["transcriptid"])
    model = ev.merge(feats, on="transcriptid", how="left")

    # F4: change vs. the same company's previous four calls (uses only past calls)
    model = model.sort_values(["permno", "call_date"])
    for col in ["tone_prepared", "tone_answers", "hedging_answers"]:
        if col in model.columns:
            prev = model.groupby("permno")[col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
            model[f"{col}_chg"] = model[col] - prev
    model = model.sort_values(["call_date", "permno"]).reset_index(drop=True)
    model = assign_samples(model)

    model.to_parquet(f"{BASE}/model_table.parquet", index=False)
    model.to_csv(f"{BASE}/model_table.csv", index=False)
    text = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{TEXT_DIR}/text_*.parquet"))],
                     ignore_index=True) if glob.glob(f"{TEXT_DIR}/text_*.parquet") else pd.DataFrame(
        columns=["transcriptid", "prepared_text", "qa_text"])
    full = model.merge(text, on="transcriptid", how="left")
    full.to_parquet(f"{BASE}/model_table_with_text.parquet", index=False)
    if WRITE_TEXT_CSV:
        full.to_csv(f"{BASE}/model_table_with_text.csv", index=False)
    report(model, full)


def assign_samples(model):
    """Label each call train / holdout by date, and keep a holdout call only if its company also
    appears in the training data (at least MIN_HISTORY_CALLS training calls with a target).

    Companies that appear only in training are kept, so training still includes firms that later
    left the index (no survivorship bias)."""
    d = model["call_date"]
    model["period"] = np.where(d < pd.Timestamp(HOLDOUT_START), "train", "holdout")
    model["sample"] = np.where(model["car_2_20"].notna(), model["period"], "no_target")
    counts = model[model["sample"] == "train"].groupby("permno").size()
    hist = model["permno"].map(counts).fillna(0)
    model.loc[(model["sample"] == "holdout") & (hist < MIN_HISTORY_CALLS), "sample"] = "holdout_no_history"
    model["prior_calls_with_target"] = (model.groupby("permno")["car_2_20"]
                                       .transform(lambda s: s.notna().cumsum().shift(1).fillna(0)).astype(int))
    return model


def report(model, full):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    lines = [f"MODEL TABLE  ({BASE})",
             f"calls: {len(model):,}   companies: {model['permno'].nunique():,}   columns: {model.shape[1]}",
             f"duplicate calls: {model['keydevid'].duplicated().sum()}",
             f"calls with transcript text: {full['prepared_text'].fillna('').str.len().gt(0).sum():,}"]
    lines.append("\nsamples (calls / companies / date range):")
    for smp in ["train", "holdout", "holdout_no_history", "no_target"]:
        g = model[model["sample"] == smp]
        if len(g):
            lines.append(f"  {smp:22s} {len(g):7,} calls  {g['permno'].nunique():5,} companies  "
                         f"{g['call_date'].min().date()} to {g['call_date'].max().date()}")
    tr = set(model.loc[model["sample"] == "train", "permno"])
    ho = set(model.loc[model["sample"] == "holdout", "permno"])
    lines.append(f"  check: every holdout company is in train: {ho <= tr}")
    usable = model[model["sample"].isin(["train", "holdout"])].dropna(
        subset=[c for c in ["car_2_20", "sue", "size", "bm", "mom", "car_0_1"] if c in model.columns])
    lines.append("usable with target + core controls: " +
                 ", ".join(f"{k} {v:,}" for k, v in usable["sample"].value_counts().items()))
    if "split_adjusted_consensus" in model.columns:
        lines.append(f"calls with a split between consensus and announcement: "
                     f"{int(model['split_adjusted_consensus'].fillna(False).sum())}")
    if "share_other_sections" in model.columns:
        lines.append(f"avg share of transcript pieces with unmapped section: {model['share_other_sections'].mean():.3%}")
    lines.append("\nmissing values by column (share):")
    miss = model.isna().mean().sort_values(ascending=False)
    lines.append(miss[miss > 0].round(3).to_string())
    lines.append("\ncalls per year:")
    lines.append(model.groupby(model["call_date"].dt.year).size().to_string())
    if "sector" in model.columns:
        lines.append("\ncalls per sector:")
        lines.append(model["sector"].value_counts(dropna=False).to_string())
    num = [c for c in ["car_0_1", "car_2_20", "sue", "size", "bm", "mom", "tone_answers", "hedging_answers"]
           if c in model.columns]
    lines.append("\nkey variables:")
    lines.append(model[num].describe(percentiles=[.01, .5, .99]).T.round(4).to_string())
    text = "\n".join(lines)
    open(f"{BASE}/data_report.txt", "w").write(text)
    print(text)
    log(f"saved model_table.parquet / .csv, model_table_with_text.parquet, data_report.txt in {BASE}")


if __name__ == "__main__":
    log(f"project folder: {BASE}  |  stages: {', '.join(STAGES)}")
    for s in STAGES:
        log(f"--- stage: {s}")
        {"pull": stage_pull, "panel": stage_panel, "targets": stage_targets,
         "text": stage_text, "final": stage_final}[s]()
    log("done")
