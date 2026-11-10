"""Optional live pull from WRDS into ``data/raw`` (requires ``pip install -e '.[wrds]'``).

File-first design: every downstream stage reads the parquet files this module writes, so the
same files can instead be exported from the WRDS web UI. Each query is pulled once and skipped
if its file already exists, so an interrupted run resumes where it stopped.

Changes vs. the legacy notebook (see docs/data_audit.md):
  * I/B/E/S unadjusted tables (statsumu_epsus / actu_epsus) + CRSP cfacshr for adjustment.
  * No float32 downcast (EPS like 0.0058 and returns lose precision).
  * Link tables keep their validity dates and link quality fields; filtering happens in
    ``data.linking``, not in SQL, so it is auditable.
  * Transcript version metadata and speaker types are pulled for version selection and roles.
  * Delisting returns, GICS history and the value-weighted index are pulled.

Before the full-universe run, check table names with ``db.list_tables(library=...)`` and
``db.describe_table(...)``: CRSP is migrating to the CIZ format (crsp.stkdlysecuritydata).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from linguistic_alpha.config import Config

# Server-side universe filters, composed into the queries below.
SP500_PERMNOS = "select distinct permno from crsp.msp500list"
SP500_GVKEYS = f"""select distinct gvkey from crsp.ccmxpf_lnkhist
                   where lpermno in ({SP500_PERMNOS}) and linktype in ('LU','LC')"""
SP500_IBES = f"select distinct ticker from wrdsapps.ibcrsphist where permno in ({SP500_PERMNOS})"
SP500_CIQ = f"select distinct companyid from ciq.wrds_gvkey where gvkey in ({SP500_GVKEYS})"


@dataclass(frozen=True)
class Query:
    name: str          # raw file name
    sql: str
    by_year: str | None = None   # date column to shard on, for large tables
    group: str = "core"          # core | transcripts | optional


QUERIES: list[Query] = [
    # Universe and links
    Query("crsp_msp500list", "select permno, start, ending from crsp.msp500list"),
    Query("crsp_ccmxpf_lnkhist", f"""select gvkey, lpermno as permno, lpermco as permco, linktype,
                 linkprim, linkdt, linkenddt from crsp.ccmxpf_lnkhist
                 where lpermno in ({SP500_PERMNOS})"""),
    Query("wrdsapps_ibcrsphist", f"""select ticker, permno, ncusip, sdate, edate, score
                 from wrdsapps.ibcrsphist where permno in ({SP500_PERMNOS})"""),
    Query("ciq_wrds_gvkey", f"""select companyid, gvkey, startdate, enddate, primaryflag
                 from ciq.wrds_gvkey where gvkey in ({SP500_GVKEYS})"""),
    Query("ciq_company", f"""select companyid, companyname from ciq.ciqcompany
                 where companyid in ({SP500_CIQ})""", group="transcripts"),
    Query("comp_co_hgic", f"""select gvkey, gsector, ggroup, gind, gsubind, indfrom, indthru
                 from comp.co_hgic where gvkey in ({SP500_GVKEYS})"""),
    Query("comp_company", f"""select gvkey, cik from comp.company
                 where gvkey in ({SP500_GVKEYS})"""),
    # CRSP market data
    Query("crsp_dsi", "select date, sprtrn, vwretd from crsp.dsi where date >= %(start)s"),
    Query("crsp_dsf", f"""select permno, date, ret, retx, prc, vol, shrout, cfacpr, cfacshr
                 from crsp.dsf where permno in ({SP500_PERMNOS})
                 and date between %(a)s and %(b)s""", by_year="date"),
    Query("crsp_dsedelist", f"""select permno, dlstdt, dlstcd, dlret from crsp.dsedelist
                 where permno in ({SP500_PERMNOS})"""),
    Query("ff_factors_daily", """select date, mktrf, smb, hml, umd, rf from ff.factors_daily
                 where date >= %(start)s"""),
    # Compustat fundamentals
    Query("comp_fundq", f"""select gvkey, datadate, fyearq, fqtr, rdq, ceqq, seqq, txditcq,
                 pstkq, cshoq, prccq, epspxq from comp.fundq
                 where gvkey in ({SP500_GVKEYS}) and datadate >= %(start)s
                 and indfmt = 'INDL' and datafmt = 'STD' and consol = 'C' and popsrc = 'D'"""),
    # I/B/E/S (unadjusted)
    Query("ibes_actu_epsus", f"""select ticker, pends, pdicity, value, anndats, anntims, curr_act
                 from ibes.actu_epsus where ticker in ({SP500_IBES})
                 and pdicity = 'QTR' and anndats >= %(start)s"""),
    Query("ibes_statsumu_epsus", f"""select ticker, statpers, fpedats, fpi, medest, meanest,
                 stdev, numest, curcode from ibes.statsumu_epsus
                 where ticker in ({SP500_IBES}) and fpi in ('6','7')
                 and statpers between %(a)s and %(b)s""", by_year="statpers"),
    Query("ibes_surpsum", f"""select ticker, pends, pdicity, anndats, actual, surpmean, surpstdev
                 from ibes.surpsum where ticker in ({SP500_IBES})
                 and pdicity = 'Q' and anndats >= %(start)s""", group="optional"),
    # Capital IQ transcripts
    Query("ciq_transcript_detail", f"""select companyid, keydevid, transcriptid, headline,
                 mostimportantdateutc, mostimportanttimeutc, keydeveventtypeid,
                 transcriptcollectiontypeid, transcriptpresentationtypeid,
                 transcriptcreationdate_utc, transcriptcreationtime_utc
                 from ciq.wrds_transcript_detail
                 where keydeveventtypeid = 48 and companyid in ({SP500_CIQ})
                 and mostimportantdateutc >= %(start)s""", group="transcripts"),
    Query("ciq_transcript_person", f"""select distinct p.transcriptid, p.transcriptpersonid,
                 p.transcriptpersonname, p.speakertypename
                 from ciq.wrds_transcript_person p
                 join ciq.wrds_transcript_detail d on d.transcriptid = p.transcriptid
                 where d.keydeveventtypeid = 48 and d.companyid in ({SP500_CIQ})
                 and d.mostimportantdateutc between %(a)s and %(b)s""",
          by_year="mostimportantdateutc", group="transcripts"),
    Query("ciq_transcript_component", f"""select c.transcriptid, c.componentorder,
                 c.transcriptcomponenttypeid, c.transcriptpersonid, c.componenttext
                 from ciq.ciqtranscriptcomponent c
                 join ciq.wrds_transcript_detail d on d.transcriptid = c.transcriptid
                 where d.keydeveventtypeid = 48 and d.companyid in ({SP500_CIQ})
                 and d.mostimportantdateutc between %(a)s and %(b)s""",
          by_year="mostimportantdateutc", group="transcripts"),
]


def run_pull(cfg: Config, groups: tuple[str, ...] = ("core", "transcripts")) -> None:
    """Pull every query in ``groups`` into ``cfg.paths.raw``, skipping files that exist."""
    import wrds  # optional dependency

    username = os.environ.get(cfg["wrds"]["username_env"])
    start_year, end_year = cfg["wrds"]["start_year"], cfg["wrds"]["end_year"]
    out = cfg.paths.raw
    out.mkdir(parents=True, exist_ok=True)

    db = wrds.Connection(wrds_username=username)
    try:
        for q in (q for q in QUERIES if q.group in groups):
            if q.by_year is None:
                _pull_one(db, q.sql, out / f"{q.name}.parquet", {"start": f"{start_year - 2}-01-01"})
            else:
                for y in range(start_year - 1, end_year + 1):
                    _pull_one(db, q.sql, out / f"{q.name}_{y}.parquet",
                              {"a": f"{y}-01-01", "b": f"{y}-12-31"})
        _pull_sec_8k(db, out, start_year - 1, end_year)
    finally:
        db.close()


def _pull_sec_8k(db, out: Path, start_year: int, end_year: int) -> None:
    """Pull WRDS 8-K/Exhibit 99.1 index metadata; text is fetched from SEC archives later."""
    try:
        if "wrdssec" not in set(db.list_libraries()):
            print("skip SEC press releases: wrdssec library is not subscribed")
            return
        db.raw_sql("select accession from wrdssec.wrds_forms limit 0")
        db.raw_sql("select accession from wrdssec.exhibits limit 0")
        companies = db.raw_sql(f"select distinct cik from comp.company where gvkey in ({SP500_GVKEYS})")
    except Exception as exc:  # subscription/schema availability varies by WRDS account
        print(f"skip SEC press releases: unable to inspect WRDS SEC tables ({type(exc).__name__})")
        return
    ciks = pd.to_numeric(companies["cik"], errors="coerce").dropna().astype("int64").unique()
    if not len(ciks):
        print("skip SEC press releases: no numeric CIKs for linked S&P 500 firms")
        return
    cik_sql = ",".join(f"'{int(x)}'" for x in ciks)
    print("Pulling SEC 8-K Exhibit 99.1 filing references from WRDS")
    for year in range(start_year, end_year + 1):
        path = out / f"sec_8k_exhibits_{year}.parquet"
        if path.exists():
            print(f"skip  {path.name}")
            continue
        sql = f"""select f.cik, f.fdate as filing_date, f.form, f.accession,
                          e.filename, e.type as exhibit_type, e.description, f.wrdsfname
                   from wrdssec.wrds_forms f
                   join wrdssec.exhibits e on e.accession = f.accession
                   where ltrim(f.cik, chr(48)) in ({cik_sql})
                     and f.form like '8-K%%' and e.type = 'EX-99.1'
                     and f.fdate between %(a)s and %(b)s"""
        try:
            df = db.raw_sql(sql, params={"a": f"{year}-01-01", "b": f"{year}-12-31"})
            df.to_parquet(path, index=False)
            print(f"saved {path.name}: {len(df):,} earnings-release exhibit references")
        except Exception as exc:
            print(f"skip SEC {year}: {type(exc).__name__}: {str(exc)[:160]}")
            break


def _pull_one(db, sql: str, path: Path, params: dict) -> None:
    if path.exists():
        print(f"skip  {path.name}")
        return
    df = db.raw_sql(sql, params=params)
    df.to_parquet(path, index=False)
    print(f"saved {path.name}: {len(df):,} rows")
