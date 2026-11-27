"""``la events``: calls -> links -> I/B/E/S release -> Day 0 -> universe -> ``processed/events``.

Order (docs/pipeline.md, Universe and Event timing):
  1. One row per call (keydevid) from ``interim/transcript_versions`` (parse pass).
  2. Call time and the call's own session (``events.placeholder_time_last``: CIQ's 00:00:00 UTC
     placeholder counts as an unknown time: the call date, ``missing_call_time``).
  3. Links, keeping every candidate permno (``data/linking.py``).
  4. Release match per candidate ticker: the I/B/E/S quarterly EPS ``anndats`` within
     ``events.release_match_days`` calendar days of the call's New York date, preferring the latest
     release on or before the call date. Several periods announced on one day (catch-up filings):
     the latest ``pends``.
  5. Day 0 from the release time (New York), ``missing_call_time`` when unknown.
  6. Membership on Day 0 and one permno per event (``sample.share_class_rule``) among the members.
  7. Exclusions (``data/universe.py``), each kept with an ``exclusion_reason``.

The circular dependency (membership on Day 0 needs the release, the release needs a ticker, the
ticker needs a permno) is broken by carrying every candidate permno through steps 3-5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from linguistic_alpha.data import linking, transcripts, universe
from linguistic_alpha.events import timing
from linguistic_alpha.schemas import EVENTS, validate

CALL_COLS = ["keydevid", "companyid", "companyname", "headline", "mostimportantdateutc",
             "mostimportanttimeutc", "keydeveventtypename"]


def call_table(versions: pd.DataFrame, trading_days: pd.DatetimeIndex, ecfg: dict) -> pd.DataFrame:
    """One row per (keydevid, companyid) with the call time and the call's own session
    (``call_day0``). Joint calls (one keydevid, two companies) keep both companies."""
    c = versions[CALL_COLS].drop_duplicates(["keydevid", "companyid"]).copy()
    for col in ("keydevid", "companyid"):
        c[col] = pd.to_numeric(c[col]).astype("int64")
    c["call_date"] = pd.to_datetime(c["mostimportantdateutc"].astype("string"),
                                    errors="coerce").dt.normalize()
    c["call_time_utc"] = c["mostimportanttimeutc"].astype("string").str.strip()
    c["call_time_placeholder"] = c["call_time_utc"].eq(universe.PLACEHOLDER_TIME).fillna(False)
    time = c["call_time_utc"]
    if ecfg.get("placeholder_time_last", False):
        time = time.mask(c["call_time_placeholder"])
    c["call_ts_utc"] = timing.call_timestamp_utc(c["call_date"], time)
    d0 = timing.assign_day0(c["call_ts_utc"], trading_days, call_date_fallback=c["call_date"],
                            market_close_et=ecfg["market_close_et"],
                            missing_call_time=ecfg["missing_call_time"])
    c["call_date_et"] = d0["call_date_et"]
    c["call_day0"] = d0["day0"]
    c["fiscal_label"] = universe.fiscal_label(c["headline"])
    return c.reset_index(drop=True)


def release_table(actuals: pd.DataFrame) -> pd.DataFrame:
    """I/B/E/S quarterly EPS actuals, one row per (ticker, anndats): the latest ``pends``."""
    a = actuals.copy()
    if "measure" in a:
        a = a[a["measure"].eq("EPS")]
    if "pdicity" in a:
        a = a[a["pdicity"].eq("QTR")]
    a["anndats"] = pd.to_datetime(a["anndats"].astype("string"), errors="coerce").dt.normalize()
    a = a.dropna(subset=["ticker", "anndats"]).sort_values(["ticker", "anndats", "pends"])
    a = a.drop_duplicates(["ticker", "anndats"], keep="last")
    return a[["ticker", "anndats", "anntims", "pends"]].reset_index(drop=True)


def match_releases(cands: pd.DataFrame, releases: pd.DataFrame,
                   window: tuple[int, int] = (-3, 1)) -> pd.DataFrame:
    """Per (keydevid, permno) candidate with a ticker: the matched release.

    ``window``: release ``anndats`` minus the call's New York date, in calendar days, inclusive.
    The latest release on or before the call date wins; else the earliest after it.
    Returns ``keydevid``, ``permno``, ``release_anndats``, ``release_anntims``, ``release_pends``,
    ``release_gap_days``.
    """
    lo, hi = window
    m = cands.dropna(subset=["ibes_ticker"])[["keydevid", "permno", "ibes_ticker", "call_date_et"]]
    m = m.merge(releases, left_on="ibes_ticker", right_on="ticker")
    m["release_gap_days"] = (m["anndats"] - m["call_date_et"]).dt.days.astype("Int64")
    m = m[m["release_gap_days"].between(lo, hi)].copy()
    m["_after"] = m["release_gap_days"] > 0
    m["_abs"] = m["release_gap_days"].abs()
    m = m.sort_values(["keydevid", "permno", "_after", "_abs"]).drop_duplicates(["keydevid", "permno"])
    return (m.rename(columns={"anndats": "release_anndats", "anntims": "release_anntims",
                              "pends": "release_pends"})
             [["keydevid", "permno", "release_anndats", "release_anntims", "release_pends",
               "release_gap_days"]].reset_index(drop=True))


def daily_history(dsf: pd.DataFrame) -> pd.DataFrame:
    """CRSP daily rows sorted for as-of joins, with the running count of valid returns."""
    d = dsf[["permno", "dlycaldt", "dlyret", "dlycap"]].dropna(subset=["permno"]).copy()
    d["permno"] = d["permno"].astype("int64")
    d["dlycaldt"] = pd.to_datetime(d["dlycaldt"].astype("string")).dt.normalize()
    d = d.sort_values(["permno", "dlycaldt"])
    d["n_valid"] = d["dlyret"].notna().groupby(d["permno"]).cumsum()
    return d.sort_values("dlycaldt").reset_index(drop=True)


def asof_before(rows: pd.DataFrame, daily: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Last CRSP session strictly before ``date_col`` per row: ``mktcap_m1`` (Day -1 market cap)
    and ``n_prior_returns`` (valid returns before that date). Index aligned to ``rows``."""
    r = rows[["permno", date_col]].reset_index(names="_row").dropna()
    r["permno"] = r["permno"].astype("int64")
    r[date_col] = pd.to_datetime(r[date_col])
    r = r.sort_values(date_col)
    m = pd.merge_asof(r, daily[["permno", "dlycaldt", "dlycap", "n_valid"]], left_on=date_col,
                      right_on="dlycaldt", by="permno", allow_exact_matches=False)
    m = m.set_index("_row").reindex(rows.index)
    return pd.DataFrame({"mktcap_m1": m["dlycap"].to_numpy(),
                         "n_prior_returns": m["n_valid"].to_numpy()}, index=rows.index)


def gics_on(events: pd.DataFrame, hgic: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """GICS codes valid on ``date_col`` per keydevid (the latest ``indfrom`` if several)."""
    h = hgic.copy()
    for c in ("indfrom", "indthru"):
        h[c] = pd.to_datetime(h[c].astype("string"), errors="coerce")
    m = events[["keydevid", "gvkey", date_col]].dropna().merge(
        h[["gvkey", "gsector", "ggroup", "gind", "gsubind", "indfrom", "indthru"]], on="gvkey")
    m = m[linking.valid_on(m, date_col, "indfrom", "indthru")].sort_values("indfrom")
    return m.drop_duplicates("keydevid", keep="last")[
        ["keydevid", "gsector", "ggroup", "gind", "gsubind"]]


def build_events(versions: pd.DataFrame, raw: dict[str, pd.DataFrame], cfg: dict
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(events, excluded, log)``.

    ``raw``: ``ciq_wrds_gvkey``, ``crsp_ccmxpf_lnkhist``, ``wrdsapps_ibcrsphist``,
    ``crsp_msp500list_v2``, ``comp_co_hgic``, ``ibes_actu_epsus``, ``crsp_dsf_v2`` (``permno``,
    ``dlycaldt``, ``dlyret``, ``dlycap``). ``log`` holds the attrition and diagnostic counts.
    """
    ecfg, scfg = cfg["events"], cfg["sample"]
    daily = daily_history(raw["crsp_dsf_v2"])
    days = pd.DatetimeIndex(daily["dlycaldt"].unique()).sort_values()
    log: dict = {}

    calls_kc = call_table(versions, days, ecfg)
    n_all = int(calls_kc["keydevid"].nunique())
    is_ec = calls_kc["keydeveventtypename"].astype("string").str.lower().str.contains(
        "earnings call", na=False)
    calls_kc = calls_kc[is_ec].reset_index(drop=True)
    calls = calls_kc.drop_duplicates("keydevid").reset_index(drop=True)
    log["calls_all"], log["calls_earnings"] = n_all, len(calls)
    log["joint_calls"] = int(calls_kc["keydevid"].duplicated().sum())

    cands, link_steps = linking.build_event_links(
        calls_kc, raw["ciq_wrds_gvkey"], raw["crsp_ccmxpf_lnkhist"], raw["wrdsapps_ibcrsphist"],
        "call_date_et", max_score=ecfg.get("ibes_max_score", 2),
        open_end=ecfg.get("iclink_open_end", "auto"))
    log["link_steps"] = link_steps
    cands = cands.merge(calls[["keydevid", "call_date_et", "call_day0"]], on="keydevid",
                        how="left", suffixes=("", "_c"))
    if "call_date_et_c" in cands:
        cands = cands.drop(columns="call_date_et_c")

    rel = match_releases(cands, release_table(raw["ibes_actu_epsus"]),
                         tuple(ecfg.get("release_match_days", (-3, 1))))
    cands = cands.merge(rel, on=["keydevid", "permno"], how="left")
    cands["release_ts_utc"] = timing.release_timestamp_utc(
        cands["release_anndats"], cands["release_anntims"].astype("string"),
        ecfg.get("release_time_tz", timing.NY))
    r0 = timing.assign_day0(cands["release_ts_utc"], days,
                            call_date_fallback=cands["release_anndats"],
                            market_close_et=ecfg["market_close_et"],
                            missing_call_time=ecfg["missing_call_time"])
    has_rel = cands["release_anndats"].notna()
    cands["release_date_et"] = r0["call_date_et"].where(has_rel)
    cands["release_time_missing"] = has_rel & cands["release_ts_utc"].isna()
    cands["day0"] = r0["day0"].where(has_rel)
    if ecfg.get("anchor", "release") == "call":
        cands["day0"] = cands["call_day0"]
    elif ecfg.get("anchor", "release") != "release":
        raise ValueError(ecfg["anchor"])

    cands["ref_date"] = cands["day0"].fillna(cands["call_date_et"])
    cands["sp500_member"] = universe.sp500_member_on(cands, raw["crsp_msp500list_v2"], "ref_date")
    cands = cands.join(asof_before(cands, daily, "ref_date"))
    cands["has_ticker"] = cands["ibes_ticker"].notna()
    cands["eligible"] = cands["sp500_member"] & cands["day0"].notna()
    cands["n_member_candidates"] = (cands["eligible"].astype(int)
                                    .groupby(cands["keydevid"]).transform("sum"))
    # best tier per keydevid, so the exclusion reason is the furthest step any candidate reached
    tier = np.select([cands["eligible"], cands["sp500_member"] & cands["has_ticker"],
                      cands["sp500_member"], cands["has_ticker"], cands["permno"].notna()],
                     [0, 1, 2, 3, 4], 5)
    cands["_tier"] = tier
    best = cands["_tier"].eq(cands.groupby("keydevid")["_tier"].transform("min"))
    ev = universe.choose_share_class(cands[best].drop(columns="eligible").assign(eligible=True),
                                     scfg["share_class_rule"])
    ev = ev.drop(columns=["eligible", "_tier", "has_ticker", "ref_date"])
    log["calls_multi_member_permno"] = int((ev["n_member_candidates"] > 1).sum())
    log["share_class_choice"] = ev.loc[ev["n_member_candidates"] > 1,
                                       "share_class_choice"].value_counts().to_dict()

    ev = calls.drop(columns=["call_date_et", "call_day0", "companyid", "companyname"]).merge(
        ev, on="keydevid", how="left")
    ev = ev.merge(calls_kc[["keydevid", "companyid", "companyname"]],
                  on=["keydevid", "companyid"], how="left")
    ev["event_date"] = ev["day0"].fillna(ev["call_date_et"])
    ev["call_lag_sessions"] = timing.call_lag_sessions(ev["call_day0"], ev["day0"], days)
    ev = ev.merge(gics_on(ev, raw["comp_co_hgic"], "event_date"), on="keydevid", how="left")

    day0 = ev.set_index("keydevid")["day0"]
    sel = transcripts.select_versions(versions, days, cfg, release_day0=day0)
    sel = sel.assign(keydevid=pd.to_numeric(sel["keydevid"]).astype("int64"),
                     transcriptid=pd.to_numeric(sel["transcriptid"]).astype("int64"))
    keep = [c for c in ("keydevid", "transcriptid", "version_created_utc", "version_type",
                        "version_final", "version_before_cutoff") if c in sel]
    ev = ev.merge(sel[keep], on="keydevid", how="left")
    ev["transcriptid"] = ev["transcriptid"].astype("Int64")

    ev = universe.apply_exclusions(ev, cfg)
    log["attrition"] = attrition(ev, log["calls_all"])
    ev = ev.sort_values(["event_date", "gvkey", "keydevid"]).reset_index(drop=True)

    kept = ev[ev["exclusion_reason"].isna()].drop(columns="exclusion_reason").copy()
    kept = kept.astype({"keydevid": "int64", "transcriptid": "int64", "companyid": "int64",
                        "permno": "int64", "gvkey": "string", "ibes_ticker": "string",
                        "call_lag_sessions": "Int64"})
    cols = list(EVENTS) + [c for c in kept.columns if c not in EVENTS]
    kept = kept[cols].reset_index(drop=True)
    validate(kept, EVENTS, "events", key="keydevid")
    excluded = ev[ev["exclusion_reason"].notna()].reset_index(drop=True)
    excluded = excluded[["exclusion_reason"] + [c for c in cols if c in excluded]]
    return kept, excluded, log


def attrition(ev: pd.DataFrame, n_all: int) -> list[tuple[str, int, int]]:
    """(step, calls before, calls after), following ``universe.REASONS``."""
    rows = [("Earnings calls (event type)", n_all, len(ev))]
    left = len(ev)
    counts = ev["exclusion_reason"].value_counts()
    for name, label in universe.REASONS:
        after = left - int(counts.get(name, 0))
        rows.append((label, left, after))
        left = after
    return rows


def _md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                              for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_log(path, events: pd.DataFrame, excluded: pd.DataFrame, log: dict, cfg: dict):
    """Attrition and diagnostics for ``la events`` (outputs/validation/events_log.md)."""
    from datetime import datetime, timezone
    from pathlib import Path

    ecfg, scfg = cfg["events"], cfg["sample"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    att = pd.DataFrame(log["attrition"], columns=["Step", "Calls before", "Calls after"])
    att["Dropped"] = att["Calls before"] - att["Calls after"]
    links = pd.DataFrame(log["link_steps"], columns=["Link (all calls, 2008 on)", "Calls before",
                                                     "Calls after"])
    ex_in = excluded[excluded["exclusion_reason"].ne("outside_sample_period")]
    insample = pd.concat([events.assign(exclusion_reason=pd.NA).astype({"exclusion_reason": "string"}),
                          ex_in], ignore_index=True)
    lag = pd.to_numeric(insample["call_lag_sessions"], errors="coerce")
    by_year = events.groupby(events["day0"].dt.year.astype(str)).size().rename("events").to_frame()
    by_year.index.name = "year (Day 0)"
    by_sector = events.groupby(events["gsector"].fillna("missing")).size().rename("events").to_frame()
    by_sector.index.name = "gsector"
    reasons = excluded["exclusion_reason"].value_counts().rename("events").to_frame()
    reasons.index.name = "exclusion_reason"
    gap = events["release_gap_days"].value_counts().sort_index().rename("events").to_frame()
    gap.index.name = "release anndats - call date (days)"
    lagd = pd.to_numeric(events["call_lag_sessions"]).value_counts().sort_index() \
        .rename("events").to_frame()
    lagd.index.name = "call_lag_sessions"
    dups = excluded[excluded["exclusion_reason"].isin(["duplicate_call", "duplicate_release"])]
    dups = dups[["keydevid", "exclusion_reason", "duplicate_of_keydevid",
                 "duplicate_release_of_keydevid", "gvkey", "companyname", "headline", "call_date",
                 "call_time_utc", "fiscal_label"]].copy()
    dups["call_date"] = dups["call_date"].dt.date
    lines = [
        "# la events: attrition and checks",
        "",
        f"Built by `la events` (`src/linguistic_alpha/events/build.py`) from "
        f"`data/interim/transcript_versions` and `data/raw/sp500_pit_2008`. "
        f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        f"Switches: anchor `{ecfg.get('anchor')}`, placeholder_time_last "
        f"`{ecfg.get('placeholder_time_last')}`, duplicate_window_days "
        f"`{ecfg.get('duplicate_window_days')}`, release_match_days "
        f"`{ecfg.get('release_match_days')}`, max_call_lag_sessions "
        f"`{ecfg.get('max_call_lag_sessions')}`, sample {scfg['start']} to {scfg['end']} (Day 0).",
        "",
        "## Attrition (each call gets the first reason it fails, in this order)",
        "",
        _md(att.set_index("Step")),
        "",
        "## Links on all calls (before the sample-period filter)",
        "",
        _md(links.set_index("Link (all calls, 2008 on)")),
        "",
        "## Checks (calls inside the sample period)",
        "",
        f"- Release time missing (anntims blank or 00:00:00, Day 0 = next session): "
        f"{int(insample['release_time_missing'].fillna(False).sum()):,} matched calls, "
        f"{int(events['release_time_missing'].sum()):,} kept events.",
        f"- Call lag of 2+ sessions after the release: {int((lag >= 2).sum()):,}; "
        f"call before the release's Day 0 (lag < 0): {int((lag < 0).sum()):,}.",
        f"- No I/B/E/S ticker: {int((insample['exclusion_reason'] == 'no_ibes_ticker').sum()):,} "
        f"S&P 500 calls (exclusion reason); no release in the match window: "
        f"{int((insample['exclusion_reason'] == 'no_ibes_release').sum()):,}.",
        f"- Calls with more than one member permno on Day 0: "
        f"{log['calls_multi_member_permno']:,} (choice: {log['share_class_choice']}).",
        f"- CIQ 00:00:00 UTC placeholder call times among kept events: "
        f"{int(events['call_time_placeholder'].sum()):,}.",
        f"- Kept events with no GICS sector on Day 0: {int(events['gsector'].isna().sum()):,}.",
        f"- Joint calls (one keydevid, two CIQ companies; both are candidates): "
        f"{log['joint_calls']:,}.",
        "",
        "## Kept events by call lag",
        "",
        _md(lagd),
        "",
        "## Kept events by release date minus call date",
        "",
        _md(gap),
        "",
        "## Events per year",
        "",
        _md(by_year),
        "",
        "## Events per GICS sector",
        "",
        _md(by_sector),
        "",
        "## Excluded calls by reason (all years)",
        "",
        _md(reasons),
        "",
        f"## Duplicate calls dropped ({len(dups):,}, all years)",
        "",
        "<details><summary>All dropped duplicates</summary>",
        "",
        _md(dups.set_index("keydevid")),
        "",
        "</details>",
        "",
    ]
    path.write_text("\n".join(lines))
    return path
