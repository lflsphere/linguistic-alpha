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
    Query("ciq_wrds_gvkey", f"""select companyid, gvkey, startdate, enddate
                 from ciq.wrds_gvkey where gvkey in ({SP500_GVKEYS})"""),
    Query("comp_co_hgic", f"""select gvkey, gsector, ggroup, gind, gsubind, indfrom, indthru
                 from comp.co_hgic where gvkey in ({SP500_GVKEYS})"""),
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
    Query("ciq_transcript_person", f"""select p.transcriptid, p.transcriptpersonid,
                 p.transcriptpersonname, p.companyid, p.speakertypeid, p.speakertypename
                 from ciq.wrds_transcript_person p
                 join ciq.wrds_transcript_detail d on d.transcriptid = p.transcriptid
                 where d.keydeveventtypeid = 48 and d.companyid in ({SP500_CIQ})
                 and d.mostimportantdateutc >= %(start)s""", group="transcripts"),
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
    finally:
        db.close()


def _pull_one(db, sql: str, path: Path, params: dict) -> None:
    if path.exists():
        print(f"skip  {path.name}")
        return
    df = db.raw_sql(sql, params=params)
    df.to_parquet(path, index=False)
    print(f"saved {path.name}: {len(df):,} rows")
