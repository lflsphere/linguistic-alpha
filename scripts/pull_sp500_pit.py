"""Pull raw WRDS tables for every S&P 500 member since 2005 (point-in-time universe).

Universe = every PERMNO in ``crspq.msp500list_v2`` whose membership ends on or after
2005-01-01. Rows are NOT trimmed to membership windows: lookbacks (momentum, pre-event
volatility) need history outside them. Apply point-in-time membership downstream with
``crsp_msp500list_v2`` (mbrstartdt <= event date <= mbrenddt).

Tables follow the columns of ``Linguistic_Alpha.ipynb``, with the fixes listed in
``docs/data_audit.md``:
  * CRSP from the QUARTERLY CIZ update (schema ``crspq``, through 2026-06-30) rather than
    the annual ``crsp`` schema (through 2025-12-31); values agree on the overlap.
    ``crspq.dsf_v2`` keeps ``shrout``. CIZ ``dlyret`` already includes the delisting
    return, so do not splice ``stkdelists.delret`` in again.
    CRSP VW market return = ``crspq.inddlyseriesdata`` indno 1000200.
  * I/B/E/S adjusted (as in the notebook) AND unadjusted tables (audit finding 1).
  * Transcript metadata keeps every version + speaker roles (findings 9, 12).
  * No float32 downcast (finding 4).

Usage (needs ``~/.pgpass`` or WRDS_USERNAME):
    python scripts/pull_sp500_pit.py --groups core transcripts_meta
    python scripts/pull_sp500_pit.py --groups transcripts_text --text-versions all|one
Each file is skipped if it exists, so an interrupted run resumes. Row counts go to
``manifest.csv`` in the output folder.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

START = "2005-01-01"
LOOKBACK_START = {"crsp": 2004, "comp": 2002, "ibes": 2004}   # history before START
END_YEAR = 2026

P = f"select distinct permno from crspq.msp500list_v2 where mbrenddt >= '{START}'"
G = f"""select distinct gvkey from crspq.ccmxpf_lnkhist
        where lpermno in ({P}) and linktype in ('LU','LC')"""
I = f"select distinct ticker from wrdsapps.ibcrsphist where permno in ({P})"
C = f"select distinct companyid from ciq.wrds_gvkey where gvkey in ({G})"
CALLS = f"""select transcriptid from ciq.wrds_transcript_detail
            where keydeveventtypeid = 48 and companyid in ({C})"""

# name -> (group, sql, shard) ; shard = (date column, first year) or None
QUERIES: dict[str, tuple[str, str, tuple[str, int] | None]] = {
    # Universe and identifier links
    "crsp_msp500list_v2": ("core", f"""select permno, indno, mbrstartdt, mbrenddt, mbrflg
        from crspq.msp500list_v2 where mbrenddt >= '{START}'""", None),
    "crsp_ccmxpf_lnkhist": ("core", f"""select gvkey, lpermno as permno, lpermco as permco,
        linktype, linkprim, linkdt, linkenddt from crspq.ccmxpf_lnkhist
        where lpermno in ({P}) and linktype in ('LU','LC')""", None),
    "wrdsapps_ibcrsphist": ("core", f"""select ticker, permno, ncusip, sdate, edate, score
        from wrdsapps.ibcrsphist where permno in ({P})""", None),
    "ciq_wrds_gvkey": ("core", f"""select companyid, gvkey, companyname, startdate, enddate,
        primaryflag from ciq.wrds_gvkey where gvkey in ({G})""", None),
    "crsp_stksecurityinfohist": ("core", f"""select * from crspq.stksecurityinfohist
        where permno in ({P})""", None),
    # CRSP market data (CIZ)
    "crsp_dsp500_v2": ("core", """select caldt, spindx, sprtrn, vwretd, vwretx
        from crspq.dsp500_v2 where caldt >= '2004-01-01'""", None),
    "crsp_vwmkt_daily": ("core", """select dlycaldt, dlytotret as vwretd, dlyprcret as vwretx
        from crspq.inddlyseriesdata where indno = 1000200 and dlycaldt >= '2004-01-01'""", None),
    "crsp_dsf_v2": ("core", f"""select permno, permco, dlycaldt, dlyret, dlyretx, dlyprc,
        dlyvol, dlycap, shrout, dlyfacprc, dlycumfacpr, dlycumfacshr, dlydelflg,
        dlyretmissflg, sharetype, securitytype, primaryexch, siccd, ticker, cusip
        from crspq.dsf_v2
        where permno in ({P}) and dlycaldt between %(a)s and %(b)s""",
        ("dlycaldt", LOOKBACK_START["crsp"])),
    "crsp_stkdelists": ("core", f"""select permno, delistingdt, delret, delretmisstype,
        delactiontype, delstatustype, delreasontype from crspq.stkdelists
        where permno in ({P})""", None),
    "ff_factors_daily": ("core", """select date, mktrf, smb, hml, umd, rf
        from ff.factors_daily where date >= '2004-01-01'""", None),
    # Compustat
    "comp_fundq": ("core", f"""select gvkey, datadate, fyearq, fqtr, rdq, epspxq, ceqq, seqq,
        txditcq, pstkq, cshoq, prccq from comp.fundq
        where gvkey in ({G}) and datadate >= '{LOOKBACK_START["comp"]}-01-01'
        and indfmt = 'INDL' and datafmt = 'STD' and consol = 'C' and popsrc = 'D'""", None),
    "comp_co_hgic": ("core", f"""select gvkey, gsector, ggroup, gind, gsubind, indfrom, indthru
        from comp.co_hgic where gvkey in ({G})""", None),
    # I/B/E/S: adjusted (notebook) and unadjusted (audit finding 1)
    "ibes_act_epsus": ("core", f"""select ticker, pends, pdicity, value, anndats, anntims
        from ibes.act_epsus where ticker in ({I}) and pdicity = 'QTR'
        and anndats >= '{LOOKBACK_START["ibes"]}-01-01'""", None),
    "ibes_actu_epsus": ("core", f"""select ticker, pends, pdicity, value, anndats, anntims,
        curr_act from ibes.actu_epsus where ticker in ({I}) and pdicity = 'QTR'
        and anndats >= '{LOOKBACK_START["ibes"]}-01-01'""", None),
    "ibes_statsum_epsus": ("core", f"""select ticker, statpers, fpedats, fpi, medest, meanest,
        stdev, numest from ibes.statsum_epsus where ticker in ({I}) and fpi in ('6','7')
        and statpers between %(a)s and %(b)s""", ("statpers", LOOKBACK_START["ibes"])),
    "ibes_statsumu_epsus": ("core", f"""select ticker, statpers, fpedats, fpi, medest, meanest,
        stdev, numest, curcode from ibes.statsumu_epsus where ticker in ({I})
        and fpi in ('6','7') and statpers between %(a)s and %(b)s""",
        ("statpers", LOOKBACK_START["ibes"])),
    # Capital IQ transcripts: metadata for every version, speakers without text
    "ciq_transcript_detail": ("transcripts_meta", f"""select * from ciq.wrds_transcript_detail
        where keydeveventtypeid = 48 and companyid in ({C})
        and mostimportantdateutc >= '{START}'""", None),
    "ciq_transcript_person": ("transcripts_meta", f"""select p.transcriptid,
        p.transcriptcomponentid, p.componentorder, p.transcriptcomponenttypeid,
        p.transcriptcomponenttypename, p.transcriptpersonid, p.transcriptpersonname,
        p.proid, p.companyofperson, p.speakertypeid, p.speakertypename
        from ciq.wrds_transcript_person p join ciq.wrds_transcript_detail d
          on d.transcriptid = p.transcriptid
        where d.keydeveventtypeid = 48 and d.companyid in ({C})
        and d.mostimportantdateutc between %(a)s and %(b)s""",
        ("mostimportantdateutc", 2005)),
}

TEXT_SQL = """select c.transcriptid, c.transcriptcomponentid, c.componentorder,
    c.transcriptcomponenttypeid, c.transcriptpersonid, c.componenttext
    from ciq.ciqtranscriptcomponent c where c.transcriptid = any(%(ids)s)"""


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


def pull_text(db, out: Path, versions: str) -> None:
    import pandas as pd
    meta = pd.read_parquet(out / "ciq_transcript_detail.parquet")
    meta["year"] = pd.to_datetime(meta["mostimportantdateutc"]).dt.year
    if versions == "one":
        # Latest-created version per call (not max transcriptid: audit finding 9)
        meta["created"] = pd.to_datetime(meta["transcriptcreationdate_utc"].astype(str) + " "
                                          + meta["transcriptcreationtime_utc"].astype(str),
                                          errors="coerce")
        meta = meta.sort_values(["keydevid", "created", "transcriptid"]).groupby("keydevid").tail(1)
    for y, grp in meta.groupby("year"):
        ids = [int(t) for t in grp["transcriptid"].unique()]
        save(db, out, f"ciq_transcript_component_{versions}_{y}", TEXT_SQL, {"ids": ids})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["core", "transcripts_meta"])
    ap.add_argument("--text-versions", choices=["all", "one"], default="one")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "raw" / "sp500_pit"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import wrds
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    try:
        for name, (group, sql, shard) in QUERIES.items():
            if group not in a.groups:
                continue
            if shard is None:
                save(db, out, name, sql)
            else:
                for y in range(shard[1], END_YEAR + 1):
                    save(db, out, f"{name}_{y}", sql, {"a": f"{y}-01-01", "b": f"{y}-12-31"})
        if "transcripts_text" in a.groups:
            pull_text(db, out, a.text_versions)
    finally:
        db.close()


if __name__ == "__main__":
    main()
