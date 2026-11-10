#!/usr/bin/env python3
"""Pull WRDS inputs and build the licensed, event-level S&P 500 modeling dataset.

Run ``uv run --extra wrds --extra models python scripts/build_wrds_dataset.py``.
The run resumes from yearly Parquet pulls in data/raw and leaves all licensed data local.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from linguistic_alpha.config import load_config
from linguistic_alpha.controls.earnings import earnings_surprise
from linguistic_alpha.controls.firm import firm_controls
from linguistic_alpha.data import linking, transcripts
from linguistic_alpha.data.wrds_pull import run_pull
from linguistic_alpha.events.timing import assign_day0, call_timestamp_utc
from linguistic_alpha.io import read_table, write_table
from linguistic_alpha.targets.car import abnormal_returns, event_car, post_call_volatility
from linguistic_alpha.data.universe import sp500_member_on


ROOT = Path(__file__).resolve().parents[1]
_SEC_NETWORK_UNAVAILABLE = False


def cik_key(value) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits.lstrip("0") or "0"


def press_release_text(cik, filing_date, text) -> str:
    """Prefer 8-K Exhibit 99.1 (the earnings release), then clean it for text features."""
    raw = str(text or "")
    match = re.search(r"(?is)(?:EXHIBIT|EX-?)\s*99\.1\b(.*?)(?=(?:EXHIBIT|EX-?)\s*99\.2\b|$)", raw)
    if match:
        raw = match.group(1)
    raw = html.unescape(re.sub(r"(?is)<script\b.*?</script>|<style\b.*?</style>", " ", raw))
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _fetch_exhibit(url: str, cache_path: Path, contact_email: str) -> bool:
    global _SEC_NETWORK_UNAVAILABLE
    if cache_path.exists() and cache_path.stat().st_size:
        return True
    if _SEC_NETWORK_UNAVAILABLE:
        return False
    request = urllib.request.Request(
        url, headers={"User-Agent": f"linguistic-alpha research contact {contact_email}",
                      "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read()
            if not body:
                return False
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(body)
            time.sleep(0.13)  # keep request rate below the SEC's 10 req/sec ceiling
            return True
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404, 429, 500, 502, 503, 504) or attempt == 3:
                print(f"SEC exhibit unavailable ({exc.code}): {cache_path.name}")
                return False
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            # A DNS/socket failure is an environment-wide network problem, not
            # an individual missing filing. Avoid retrying thousands of URLs.
            if isinstance(exc, urllib.error.URLError):
                _SEC_NETWORK_UNAVAILABLE = True
                print(f"SEC exhibit fetch unavailable ({type(exc).__name__}); skipping SEC downloads")
                return False
            if attempt == 3:
                print(f"SEC exhibit fetch failed ({type(exc).__name__}): {cache_path.name}")
                return False
            time.sleep(min(2 ** attempt, 8))
    return False


def attach_press_releases(events: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Attach same-day WRDS-indexed Exhibit 99.1 text from SEC archives."""
    events = events.copy()
    events["press_release_text"] = ""
    try:
        ciks = read_table(raw, "comp_company")
        ciks["gvkey"] = ciks["gvkey"].astype("string").str.zfill(6)
        ciks["cik_key"] = ciks["cik"].map(cik_key)
        events = events.merge(ciks[["gvkey", "cik_key"]].drop_duplicates("gvkey"),
                              on="gvkey", how="left")
    except FileNotFoundError:
        events["cik_key"] = pd.NA
    ciks = set(events["cik_key"].dropna().astype(str))
    if not ciks:
        return events
    filing_parts = []
    for path in sorted(raw.glob("sec_8k_exhibits_[0-9][0-9][0-9][0-9].parquet")):
        filings = pd.read_parquet(path)
        filings["filing_date"] = _date(filings["filing_date"])
        filings["cik_key"] = filings["cik"].map(cik_key)
        filings = filings[filings["cik_key"].isin(ciks)]
        filing_parts.append(filings)
    if not filing_parts:
        print("No WRDS SEC 8-K exhibit references found; F5 press-release features will be missing")
        return events
    filings = pd.concat(filing_parts, ignore_index=True)
    calls = events[["keydevid", "cik_key", "call_date_et"]].copy()
    candidates = calls.merge(filings, left_on=["cik_key", "call_date_et"],
                             right_on=["cik_key", "filing_date"], how="left")
    candidates = candidates[candidates["filename"].notna()].drop_duplicates("keydevid")
    contact_email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not contact_email:
        print("SEC_CONTACT_EMAIL is unset; F5 press-release texts will be missing")
        return events
    cache = raw / "sec_exhibit_html"
    release_lookup = {}
    for row in candidates.itertuples(index=False):
        cik = cik_key(row.cik_key)
        accession = str(row.accession).replace("-", "")
        filename = Path(str(row.filename)).name
        cache_path = cache / f"{accession}_{filename}"
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
        if not _fetch_exhibit(url, cache_path, contact_email):
            continue
        release_lookup[row.keydevid] = press_release_text(cik, row.call_date_et,
                                                          cache_path.read_text(errors="replace"))
    events["press_release_text"] = events["keydevid"].map(release_lookup).fillna("")
    print(f"Calls with same-day SEC release text: {sum(events['press_release_text'].str.len() > 0):,}")
    return events


def read_transcript_shards(folder: Path, name: str, ids: set[int]) -> pd.DataFrame:
    """Read only the selected transcript ids from each year shard to limit peak memory."""
    paths = sorted(folder.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, filters=[[('transcriptid', 'in', list(ids))]])
        if len(frame):
            frames.append(frame)
        print(f"transcript components {path.name}: {len(frame):,} selected rows")
        del frame
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def build_panel(cfg, run_features: bool = True) -> pd.DataFrame:
    raw = cfg.paths.raw
    calls = read_table(raw, "ciq_transcript_detail")
    calls = transcripts.select_latest_version(calls)
    calls["call_ts_utc"] = call_timestamp_utc(calls["mostimportantdateutc"],
                                               calls["mostimportanttimeutc"])
    dsi = read_table(raw, "crsp_dsi")
    dsi["date"] = _date(dsi["date"])
    time = assign_day0(calls["call_ts_utc"], pd.DatetimeIndex(dsi["date"].dropna().unique()),
                       call_date_fallback=calls["mostimportantdateutc"],
                       market_close_et=cfg["events"]["market_close_et"],
                       missing_call_time=cfg["events"]["missing_call_time"])
    calls = pd.concat([calls.reset_index(drop=True), time.reset_index(drop=True)], axis=1)
    calls["call_date_et"] = calls["call_date_et"].where(
        calls["call_date_et"].notna(), _date(calls["mostimportantdateutc"]))
    start, end = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
    # Keep one year before the model sample for point-in-time history features.
    calls = calls[calls["call_date_et"].between(start - pd.DateOffset(years=1), end)].copy()
    calls["companyname"] = calls.get("companyname", calls["companyid"].astype("string"))
    try:
        names = read_table(raw, "ciq_company")
        calls = calls.drop(columns="companyname", errors="ignore").merge(names, on="companyid", how="left")
        calls["companyname"] = calls["companyname"].fillna(calls["companyid"].astype("string"))
    except FileNotFoundError:
        pass
    print(f"Latest transcript versions in configured date range: {len(calls):,}")

    linked = linking.build_event_links(
        calls, read_table(raw, "ciq_wrds_gvkey"), read_table(raw, "crsp_ccmxpf_lnkhist"),
        read_table(raw, "wrdsapps_ibcrsphist"))
    linked = linked[linked["permno"].notna()].copy()
    linked["permno"] = pd.to_numeric(linked["permno"], errors="coerce").astype("Int64")
    membership = read_table(raw, "crsp_msp500list")
    linked["in_sp500"] = sp500_member_on(linked, membership)

    # Attach the date-valid GICS sub-industry and keep candidates auditable.
    gics = read_table(raw, "comp_co_hgic")
    gics["gvkey"] = gics["gvkey"].astype("string").str.zfill(6)
    for c in ("indfrom", "indthru"):
        gics[c] = _date(gics[c])
    gx = linked[["keydevid", "gvkey", "call_date_et"]].merge(gics, on="gvkey", how="left")
    valid = (gx["indfrom"].isna() | (gx["indfrom"] <= gx["call_date_et"])) & \
            (gx["indthru"].isna() | (gx["indthru"] >= gx["call_date_et"]))
    gx = gx[valid].sort_values(["keydevid", "indfrom"]).drop_duplicates("keydevid", keep="last")
    linked = linked.merge(gx[["keydevid", "gsector", "ggroup", "gind", "gsubind"]],
                          on="keydevid", how="left")

    # Keep an audit copy for the entire linked call universe, including excluded events.
    dsf = read_table(raw, "crsp_dsf")
    dsf["date"] = _date(dsf["date"])
    delist = read_table(raw, "crsp_dsedelist")
    bench = dsi
    ar = abnormal_returns(dsf, bench, delist, cfg["targets"]["benchmark"])
    linked = linked.merge(event_car(linked, ar, cfg["targets"]["car_windows"],
                                    cfg["targets"]["min_window_coverage"]), on="keydevid", how="left")
    linked = linked.merge(post_call_volatility(linked, ar, cfg["targets"]["vol_post_window"],
                                               cfg["targets"]["vol_pre_window"]),
                          on="keydevid", how="left")
    linked = linked.merge(firm_controls(linked, dsf, read_table(raw, "comp_fundq"), ar),
                          on="keydevid", how="left")

    actuals = read_table(raw, "ibes_actu_epsus")
    consensus = read_table(raw, "ibes_statsumu_epsus")
    sue = earnings_surprise(linked, actuals, consensus, dsf, dsf)
    linked = linked.merge(sue, on="keydevid", how="left")
    linked["surprise_pct"] = linked["eps_surprise"] / linked["consensus_eps"].abs().replace(0, np.nan)
    linked["consensus"] = linked["consensus_eps"]
    linked = attach_press_releases(linked, raw)

    # Canonical transcript sections from the selected version IDs.
    tids = set(linked["transcriptid"].dropna().astype(int))
    comp_raw = read_transcript_shards(raw, "ciq_transcript_component", tids)
    try:
        persons = read_table(raw, "ciq_transcript_person")
    except FileNotFoundError:
        persons = None
    if len(comp_raw):
        components = transcripts.build_components(comp_raw, persons)
        sections = transcripts.call_sections(components)
        qa_parts = components[components["section"].isin(["question", "answer"])].copy()
        qa_parts["label"] = qa_parts["section"].str.upper() + ": " + qa_parts["text"].astype(str)
        qa_text = qa_parts.groupby("transcriptid")["label"].agg("\n\n".join).rename("qa_text")
        sections = sections.merge(qa_text, on="transcriptid", how="left")
        sections["qa_text"] = sections["qa_text"].fillna("")
        linked = linked.merge(sections, on="transcriptid", how="left")
        linked["has_transcript"] = (linked["prepared_text"].fillna("").str.strip().ne("") |
                                     linked["qa_text"].fillna("").str.strip().ne(""))
        linked["prepared_text"] = linked["prepared_text"].fillna("")
        linked["qa_text"] = linked["qa_text"].fillna("")
    else:
        components = pd.DataFrame()
        linked["has_transcript"] = False
        linked["prepared_text"] = ""
        linked["qa_text"] = ""

    # Count available prior stock-return sessions for the universe exclusion rule.
    ar_by = {p: pd.DatetimeIndex(g.loc[g["ar"].notna(), "date"]) for p, g in ar.groupby("permno")}
    linked["prior_return_days"] = [int((ar_by.get(r.permno, pd.DatetimeIndex([])) < r.day0).sum())
                                    if pd.notna(r.day0) else 0 for r in
                                    linked[["permno", "day0"]].itertuples(index=False)]
    from linguistic_alpha.data.universe import apply_exclusions
    linked = apply_exclusions(linked, cfg.raw)
    linked = linked.sort_values(["call_date_et", "permno"]).reset_index(drop=True)
    linked["firm"] = linked["gvkey"].astype("string").str.zfill(6)
    write_table(linked, cfg.paths.processed, "event_panel_all")

    feature_events = linked.copy()
    # Scripts/01 is the repository's implementation of F1-F4 and Q&A interaction features.
    panel_path = cfg.paths.processed / "feature_input.csv"
    feature_path = cfg.paths.processed / "language_features.csv"
    pairs_path = cfg.paths.processed / "qa_pairs.parquet"
    meta_path = cfg.paths.processed / "language_features_meta.json"
    needed = ["transcriptid", "ticker", "firm", "companyname", "call_date", "call_date_et", "call_time_utc", "call_ts_utc", "day0",
              "car_0_1", "car_2_5", "car_2_20", "car_2_60", "vol_post_2_20", "eps_surprise",
              "surprise_pct", "sue", "consensus_eps", "dispersion", "n_analysts", "size", "bm",
              "mom", "vol_pre", "fiscal_period_end", "prepared_text", "qa_text"]
    feature_events["call_date"] = feature_events["call_date_et"]
    feature_events["call_time_utc"] = feature_events["mostimportanttimeutc"]
    feature_events["ticker"] = feature_events["ticker"].fillna(
        feature_events["permno"].astype("string"))
    feature_events["firm"] = feature_events["gvkey"].astype("string").str.zfill(6)
    for c in needed:
        if c not in feature_events:
            feature_events[c] = np.nan
    feature_events[needed].to_csv(panel_path, index=False)
    if run_features:
        subprocess.run([sys.executable, str(ROOT / "scripts/01_build_features.py"),
                        "--panel", str(panel_path), "--features-out", str(feature_path),
                        "--pairs-out", str(cfg.paths.processed / "qa_pairs.csv"),
                        "--meta-out", str(meta_path), "--embed", "tfidf"], check=True)
        features = pd.read_csv(feature_path)
        if "transcriptid" in features:
            linked = linked.merge(features.drop(columns=[c for c in needed if c in features and
                                                          c not in ("transcriptid",)]),
                                  on="transcriptid", how="left", suffixes=("", "_feature"))
        pairs_csv = cfg.paths.processed / "qa_pairs.csv"
        if pairs_csv.exists():
            pairs = pd.read_csv(pairs_csv)
            for col in pairs.select_dtypes(include="object"):
                pairs[col] = pairs[col].astype("string")
            write_table(pairs, cfg.paths.processed, "qa_pairs")
    model_table = linked[linked["included"] & (linked["call_date_et"] >= start)].copy()
    write_table(model_table, cfg.paths.processed, "model_table")
    attrition = linked["exclusion_reason"].fillna("included").value_counts().rename_axis(
        "reason").reset_index(name="events")
    attrition.to_csv(cfg.paths.processed / "sample_attrition.csv", index=False)
    print(f"Final event panel: {len(model_table):,} included calls / {model_table['permno'].nunique():,} PERMNOs")
    print(f"Targets available: {model_table['car_2_20'].notna().sum():,} CAR[+2,+20] labels")
    print(f"Wrote {cfg.paths.processed / 'model_table.parquet'}")
    if run_features:
        subprocess.run([sys.executable, str(ROOT / "scripts/finalize_dataset.py")], check=True)
    return model_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-pull", action="store_true", help="build using data/raw files already present")
    ap.add_argument("--skip-features", action="store_true", help="join targets/controls/text only")
    ap.add_argument("--groups", nargs="+", default=["core", "transcripts"])
    args = ap.parse_args()
    cfg = load_config()
    cfg.paths.ensure()
    if not args.skip_pull:
        run_pull(cfg, groups=tuple(args.groups))
    build_panel(cfg, run_features=not args.skip_features)


if __name__ == "__main__":
    main()
