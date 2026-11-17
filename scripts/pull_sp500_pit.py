"""Pull raw WRDS tables for every S&P 500 member since 2008 (point-in-time universe).

Universe = every PERMNO in ``crspq.msp500list_v2`` whose membership ends on or after
START (2008-01-01). Every table is restricted to dates on or after START (team decision,
2026-11-17): there is no earlier lookback, so history-based features (momentum, prior-quarter
SUE) are missing for the first events of 2008. Rows are NOT trimmed to membership windows;
apply point-in-time membership downstream (mbrstartdt <= event date <= mbrenddt).

Choices follow ``docs/data_audit.md`` and ``docs/wrds_chatbot_answers.md`` (Qn below):
  * CRSP from the quarterly CIZ update (``crspq``). ``dlyret`` already includes delisting
    returns: never add ``stkdelists.delret`` again; ``dlyretmissflg`` DM/DP marks unpriced
    delistings (Q3). Main benchmark = S&P 500 total return (dsp500_v2.vwretd, point in time); CRSP VW market (indno 1000200, Q4) is a robustness check.
  * I/B/E/S adjusted AND unadjusted tables, plus split factors ``adj``/``adjsum`` (Q6);
    non-EPS measures SAL, CPS, EBT, DPS (Q9, Q13); surprise history (Q12); company
    guidance from ``det_guidance`` (Q10, Q11; reliable from 2007, so all of our sample).
  * Capital IQ: every transcript version + speakers + text; Key Developments for earnings
    releases (type 28) and expected release dates (55, 144) (Q18); GVKEY links from
    ``ciq_common.wrds_gvkey`` (Q20). Every version is pulled; downstream picks the latest
    version created by the Day 0 close (docs/pipeline.md, "Transcript version and timing").
  * Compustat ``fundq`` keyed on (gvkey, datadate, fyr) (Q21).

Table locations are resolved at run time from information_schema (WRDS schema names vary
between the data dictionary and PostgreSQL), and every filter column is checked before any
data is pulled. One connection is used for the whole run; if login fails the script stops
without retrying, to avoid locking the account (Q23).

Usage (needs ~/.pgpass):
    python scripts/pull_sp500_pit.py --check            # resolve tables + columns, pull nothing
    python scripts/pull_sp500_pit.py --groups core transcripts_meta
    python scripts/pull_sp500_pit.py --groups transcripts_text
Each file is skipped if it exists, so an interrupted run resumes. Row counts go to
``manifest.csv`` in the output folder.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

START = "2008-01-01"
START_YEAR = 2008
END_YEAR = 2026
XMEASURES = "('SAL','CPS','EBT','DPS')"   # revenue, cash flow/share, EBITDA, dividends/share

# Candidate schemas per table, first match wins (data-dictionary vs. PostgreSQL names)
WHERE = {
    "msp500list_v2": ["crspq"], "ccmxpf_lnkhist": ["crspq", "crsp"],
    "stksecurityinfohist": ["crspq"], "dsp500_v2": ["crspq"], "inddlyseriesdata": ["crspq"],
    "dsf_v2": ["crspq"], "stkdelists": ["crspq"], "factors_daily": ["ff"],
    "ibcrsphist": ["wrdsapps", "wrdsapps_link_crsp_ibes"],
    "fundq": ["comp"], "co_hgic": ["comp"],
    "act_epsus": ["ibes", "tr_ibes"], "actu_epsus": ["ibes", "tr_ibes"],
    "statsum_epsus": ["ibes", "tr_ibes"], "statsumu_epsus": ["ibes", "tr_ibes"],
    "act_xepsus": ["ibes", "tr_ibes"], "actu_xepsus": ["ibes", "tr_ibes"],
    "statsum_xepsus": ["ibes", "tr_ibes"], "statsumu_xepsus": ["ibes", "tr_ibes"],
    "surpsum": ["ibes", "tr_ibes"], "surpsumu": ["ibes", "tr_ibes"],
    "adj": ["ibes", "tr_ibes"], "adjsum": ["ibes", "tr_ibes"],
    "det_guidance": ["ibes_guidance", "tr_ibes_guidance", "ibes"],
    "wrds_gvkey": ["ciq_common", "ciq"],
    "wrds_keydev": ["ciq_keydev", "ciq"],
    "wrds_transcript_detail": ["ciq_transcripts", "ciq"],
    "wrds_transcript_person": ["ciq_transcripts", "ciq"],
    "ciqtranscriptcomponent": ["ciq_transcripts", "ciq"],
}

# name -> (group, table, sql template, shard date column or None, columns that must exist)
# {T} = resolved table, {P}/{G}/{I}/{C} = universe subqueries, %(a)s/%(b)s = shard dates.
QUERIES: dict[str, tuple[str, str, str, str | None, list[str]]] = {
    # Universe and identifier links
    "crsp_msp500list_v2": ("core", "msp500list_v2", """select permno, indno, mbrstartdt,
        mbrenddt, mbrflg from {T} where mbrenddt >= '{S}'""", None, ["mbrenddt"]),
    "crsp_ccmxpf_lnkhist": ("core", "ccmxpf_lnkhist", """select gvkey, lpermno as permno,
        lpermco as permco, linktype, linkprim, linkdt, linkenddt from {T}
        where lpermno in ({P}) and linktype in ('LU','LC')""", None, ["lpermno", "linktype"]),
    "wrdsapps_ibcrsphist": ("core", "ibcrsphist", """select ticker, permno, ncusip, sdate,
        edate, score from {T} where permno in ({P})""", None, ["score", "sdate", "edate"]),
    "ciq_wrds_gvkey": ("core", "wrds_gvkey", """select companyid, gvkey, companyname,
        startdate, enddate, primaryflag from {T} where gvkey in ({G})""", None,
        ["gvkey", "primaryflag"]),
    "crsp_stksecurityinfohist": ("core", "stksecurityinfohist", """select * from {T}
        where permno in ({P}) and secinfoenddt >= '{S}'""", None, ["secinfoenddt"]),
    # CRSP market data
    "crsp_dsp500_v2": ("core", "dsp500_v2", """select caldt, spindx, sprtrn, vwretd, vwretx
        from {T} where caldt >= '{S}'""", None, ["caldt", "vwretd"]),
    "crsp_vwmkt_daily": ("core", "inddlyseriesdata", """select dlycaldt, dlytotret as vwretd,
        dlyprcret as vwretx from {T} where indno = 1000200 and dlycaldt >= '{S}'""", None,
        ["indno", "dlytotret", "dlyprcret"]),
    "crsp_dsf_v2": ("core", "dsf_v2", """select permno, permco, dlycaldt, dlyret, dlyretx,
        dlyprc, dlyvol, dlycap, shrout, dlyfacprc, dlycumfacpr, dlycumfacshr, dlydelflg,
        dlyretmissflg, sharetype, securitytype, primaryexch, siccd, ticker, cusip from {T}
        where permno in ({P}) and dlycaldt between %(a)s and %(b)s""", "dlycaldt",
        ["dlyretmissflg", "shrout"]),
    "crsp_stkdelists": ("core", "stkdelists", """select permno, delistingdt, delret,
        delretmisstype, delactiontype, delstatustype, delreasontype from {T}
        where permno in ({P}) and delistingdt >= '{S}'""", None, ["delretmisstype"]),
    "ff_factors_daily": ("core", "factors_daily", """select date, mktrf, smb, hml, umd, rf
        from {T} where date >= '{S}'""", None, ["umd"]),
    # Compustat (Q21: fyr is part of the key)
    "comp_fundq": ("core", "fundq", """select gvkey, datadate, fyr, fyearq, fqtr, datafqtr,
        datacqtr, rdq, epspxq, saleq, ceqq, seqq, txditcq, pstkq, cshoq, prccq from {T}
        where gvkey in ({G}) and datadate >= '{S}' and indfmt = 'INDL' and datafmt = 'STD'
        and consol = 'C' and popsrc = 'D'""", None, ["fyr", "datafqtr", "datacqtr", "rdq"]),
    "comp_co_hgic": ("core", "co_hgic", """select gvkey, gsector, ggroup, gind, gsubind,
        indfrom, indthru from {T} where gvkey in ({G})""", None, ["indfrom"]),
    # I/B/E/S EPS: actuals and monthly consensus, adjusted and unadjusted
    "ibes_act_epsus": ("core", "act_epsus", """select ticker, measure, pends, pdicity, value,
        anndats, anntims, actdats, acttims from {T} where ticker in ({I})
        and pdicity = 'QTR' and anndats >= '{S}'""", None, ["actdats", "acttims"]),
    "ibes_actu_epsus": ("core", "actu_epsus", """select ticker, measure, pends, pdicity, value,
        anndats, anntims, actdats, acttims, curr_act from {T} where ticker in ({I})
        and pdicity = 'QTR' and anndats >= '{S}'""", None, ["curr_act"]),
    "ibes_statsum_epsus": ("core", "statsum_epsus", """select ticker, measure, fiscalp,
        statpers, fpedats, fpi, medest, meanest, stdev, highest, lowest, numest, numup,
        numdown from {T} where ticker in ({I}) and fpi in ('6','7')
        and statpers between %(a)s and %(b)s""", "statpers",
        ["highest", "lowest", "numup", "numdown"]),
    "ibes_statsumu_epsus": ("core", "statsumu_epsus", """select ticker, measure, fiscalp,
        statpers, fpedats, fpi, medest, meanest, stdev, highest, lowest, numest, numup,
        numdown, curcode from {T} where ticker in ({I}) and fpi in ('6','7')
        and statpers between %(a)s and %(b)s""", "statpers", ["curcode"]),
    # I/B/E/S non-EPS measures (Q9, Q13)
    "ibes_act_xepsus": ("core", "act_xepsus", """select * from {T} where ticker in ({I})
        and measure in {X} and pdicity = 'QTR' and anndats >= '{S}'""", None,
        ["measure", "pdicity", "anndats"]),
    "ibes_actu_xepsus": ("core", "actu_xepsus", """select * from {T} where ticker in ({I})
        and measure in {X} and pdicity = 'QTR' and anndats >= '{S}'""", None,
        ["measure", "pdicity", "anndats"]),
    "ibes_statsum_xepsus": ("core", "statsum_xepsus", """select * from {T}
        where ticker in ({I}) and measure in {X} and fpi in ('6','7')
        and statpers between %(a)s and %(b)s""", "statpers", ["measure", "fpi", "statpers"]),
    "ibes_statsumu_xepsus": ("core", "statsumu_xepsus", """select * from {T}
        where ticker in ({I}) and measure in {X} and fpi in ('6','7')
        and statpers between %(a)s and %(b)s""", "statpers", ["measure", "fpi", "statpers"]),
    # I/B/E/S surprise history (Q12): quarterly EPS and non-EPS
    "ibes_surpsum": ("core", "surpsum", """select * from {T} where ticker in ({I})
        and measure in ('EPS','SAL','CPS','EBT','DPS') and fiscalp = 'QTR'
        and anndats >= '{S}'""", None, ["measure", "fiscalp", "anndats"]),
    "ibes_surpsumu": ("core", "surpsumu", """select * from {T} where ticker in ({I})
        and measure in ('EPS','SAL','CPS','EBT','DPS') and fiscalp = 'QTR'
        and anndats >= '{S}'""", None, ["measure", "fiscalp", "anndats"]),
    # I/B/E/S split factors (Q6): small, no date filter so splits before 2008 are covered
    "ibes_adj": ("core", "adj", "select * from {T} where ticker in ({I})", None, ["ticker"]),
    "ibes_adjsum": ("core", "adjsum", "select * from {T} where ticker in ({I})", None,
        ["ticker"]),
    # I/B/E/S company guidance (Q10, Q11): all measures and periods
    "ibes_det_guidance": ("core", "det_guidance", """select * from {T} where ticker in ({I})
        and anndats >= '{S}'""", None,
        ["val_1", "val_2", "measure", "pdicity", "anndats", "anntims", "actdats"]),
    # Capital IQ Key Developments (Q18): 28 = earnings release, 55/144 = expected dates
    "ciq_keydev_earnings": ("core", "wrds_keydev", """select * from {T}
        where companyid in ({C}) and keydeveventtypeid in (28, 55, 144)
        and announcedate >= '{S}'""", None,
        ["keydeveventtypeid", "announcedate", "announcetime", "announceddateutc"]),
    # Capital IQ transcripts: every version + speakers (Q15, Q16)
    "ciq_transcript_detail": ("transcripts_meta", "wrds_transcript_detail", """select *
        from {T} where keydeveventtypeid = 48 and companyid in ({C})
        and mostimportantdateutc >= '{S}'""", None,
        ["transcriptcreationdate_utc", "transcriptcreationtime_utc"]),
    "ciq_transcript_person": ("transcripts_meta", "wrds_transcript_person", """select
        p.transcriptid, p.transcriptcomponentid, p.componentorder,
        p.transcriptcomponenttypeid, p.transcriptcomponenttypename, p.transcriptpersonid,
        p.transcriptpersonname, p.proid, p.companyofperson, p.speakertypeid,
        p.speakertypename from {T} p join {D} d on d.transcriptid = p.transcriptid
        where d.keydeveventtypeid = 48 and d.companyid in ({C})
        and d.mostimportantdateutc between %(a)s and %(b)s""", "mostimportantdateutc",
        ["speakertypename"]),
}

TEXT_SQL = """select transcriptid, transcriptcomponentid, componentorder,
    transcriptcomponenttypeid, transcriptpersonid, componenttext
    from {T} where transcriptid = any(%(ids)s)"""


def resolve(db) -> dict[str, tuple[str, set[str]]]:
    """table -> (schema.table, its columns), using the first candidate schema that exists."""
    names = "', '".join(WHERE)
    cols = db.raw_sql(f"""select table_schema s, table_name t, column_name c
        from information_schema.columns where table_name in ('{names}')""")
    found = {}
    for t, schemas in WHERE.items():
        for s in schemas:
            c = set(cols[(cols.s == s) & (cols.t == t)].c)
            if c:
                found[t] = (f"{s}.{t}", c)
                break
    return found


def build(found) -> dict[str, tuple[str, str, str | None]]:
    """Fill table names and universe subqueries; fail before pulling if anything is missing."""
    T = {t: v[0] for t, v in found.items()}
    missing = [t for t in WHERE if t not in T]
    for name, (_, table, _, _, need) in QUERIES.items():
        if table in found:
            missing += [f"{table}.{c}" for c in need if c not in found[table][1]]
    if missing:
        sys.exit("Missing tables/columns, nothing pulled: " + ", ".join(missing))
    P = f"select distinct permno from {T['msp500list_v2']} where mbrenddt >= '{START}'"
    G = f"""select distinct gvkey from {T['ccmxpf_lnkhist']}
            where lpermno in ({P}) and linktype in ('LU','LC')"""
    I = f"select distinct ticker from {T['ibcrsphist']} where permno in ({P})"
    C = f"select distinct companyid from {T['wrds_gvkey']} where gvkey in ({G})"
    out = {}
    for name, (group, table, sql, shard, _) in QUERIES.items():
        out[name] = (group, sql.format(T=T[table], D=T["wrds_transcript_detail"], P=P, G=G,
                                       I=I, C=C, S=START, X=XMEASURES), shard)
    out["_text"] = ("transcripts_text", TEXT_SQL.format(T=T["ciqtranscriptcomponent"]), None)
    return out


def log(out: Path, name: str, rows: int, secs: float) -> None:
    new = not (out / "manifest.csv").exists()
    with open(out / "manifest.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "rows", "seconds", "pulled_utc"])
        w.writerow([name, rows, round(secs, 1), datetime.now(timezone.utc).isoformat()])


def save(db, out: Path, name: str, sql: str, params: dict | None = None) -> None:
    path = out / f"{name}.parquet"
    if path.exists():
        print(f"skip  {name}", flush=True)
        return
    t = time.time()
    df = db.raw_sql(sql, params=params)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(path)
    log(out, name, len(df), time.time() - t)
    print(f"saved {name}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)


def pull_text(db, out: Path, sql: str) -> None:
    """Text of every transcript version, one file per call year."""
    import pandas as pd
    meta = pd.read_parquet(out / "ciq_transcript_detail.parquet",
                           columns=["transcriptid", "mostimportantdateutc"])
    meta["year"] = pd.to_datetime(meta["mostimportantdateutc"]).dt.year
    for y, grp in meta.groupby("year"):
        ids = [int(t) for t in grp["transcriptid"].unique()]
        save(db, out, f"ciq_transcript_component_{y}", sql, {"ids": ids})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["core", "transcripts_meta"],
                    choices=["core", "transcripts_meta", "transcripts_text"])
    ap.add_argument("--check", action="store_true", help="resolve tables/columns only")
    ap.add_argument("--username", default="lf2883")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "raw" / "sp500_pit_2008"))
    a = ap.parse_args()

    import wrds
    try:   # one attempt only: repeated failed logins can lock the account (Q23)
        db = wrds.Connection(wrds_username=a.username)
    except BaseException as e:
        sys.exit(f"WRDS login failed ({type(e).__name__}); not retrying. Refresh ~/.pgpass "
                 "with wrds.Connection(...).create_pgpass_file() in a terminal.")
    try:
        found = resolve(db)
        queries = build(found)
        for t, (full, _) in sorted(found.items()):
            print(f"  {t:<24} -> {full}")
        if a.check:
            print("check passed: all tables and filter columns found")
            return
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, (group, sql, shard) in queries.items():
            if group not in a.groups or name == "_text":
                continue
            if shard is None:
                save(db, out, name, sql)
            else:
                for y in range(START_YEAR, END_YEAR + 1):
                    save(db, out, f"{name}_{y}", sql, {"a": f"{y}-01-01", "b": f"{y}-12-31"})
        if "transcripts_text" in a.groups:
            pull_text(db, out, queries["_text"][1])
    finally:
        db.close()


if __name__ == "__main__":
    main()
