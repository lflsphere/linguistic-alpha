"""Sample universe and event exclusions.

Rules (project scope §3, meeting 10/23, docs/pipeline.md "Universe"):
  * Historical S&P 500 membership on Day 0 (crsp.msp500list), never today's list.
    ``mbrenddt`` = 2026-09-30 (the pull's end) means still a member.
  * One permno per event (``sample.share_class_rule: ccm_primary_then_mktcap``), chosen among
    the candidates that are members on Day 0: the CCM linkprim P link if exactly one member has
    it, else the largest market cap on Day -1.
  * Several calls for one release (``events.duplicate_calls: keep_first``): keep the first call
    by call time; log the others. Ported from ``scripts/full_source.py`` (agent C), with two
    switches pending team approval:
      - ``events.placeholder_time_last``: CIQ's 00:00:00 UTC placeholder sorts last within its
        date (and counts as an unknown call time for the call's session).
      - ``events.duplicate_window_days``: two calls are duplicates only for the same gvkey and
        fiscal label within this many calendar days (null: any distance).
    Calls still mapped to the same I/B/E/S release after that are ``duplicate_release``.
  * Optional GICS exclusions (``sample.exclude_gics_industries``, empty: biotech is kept since
    2026-11-17), using comp.co_hgic history valid on Day 0.
  * Calls with no transcript version by the cutoff (checked before keep_first, so the call kept
    for a release has usable text), or fewer than
    ``sample.min_prior_return_days`` valid CRSP returns before Day 0, are excluded. A missing
    I/B/E/S consensus is a controls-stage exclusion (``la controls``), not done here.
  * Keep an ``exclusion_reason`` column instead of dropping rows silently.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from linguistic_alpha.data.linking import valid_on

FISCAL_RE = re.compile(r"\b((?:Q[1-4]|H[12]|FY)\s*,?\s*(?:19|20)\d{2})\b", re.I)
ANNUAL_RE = re.compile(r",\s*((?:19|20)\d{2})\s+Earnings Call\b", re.I)
PLACEHOLDER_TIME = "00:00:00"

# exclusion reasons, in the order they are applied (the attrition table follows this order)
REASONS = [
    ("outside_sample_period", "Day 0 in sample.start .. sample.end (call date if no release)"),
    ("no_gvkey", "CIQ companyid -> gvkey (date-valid)"),
    ("no_permno", "gvkey -> permno (CCM LU/LC, date-valid)"),
    ("not_sp500_member", "S&P 500 member on Day 0 (call date if no release)"),
    ("no_ibes_ticker", "permno -> I/B/E/S ticker (score <= 2, date-valid)"),
    ("no_ibes_release", "I/B/E/S EPS release matched to the call (events.release_match_days)"),
    ("no_transcript_version", "Transcript version by the cutoff (no_version_before_cutoff: drop)"),
    ("duplicate_call", "One call per firm and fiscal period (duplicate_calls: keep_first)"),
    ("duplicate_release", "One call per I/B/E/S release (keep_first)"),
    ("call_lag", "Call in Day 0 or Day +1 (events.max_call_lag_sessions)"),
    ("gics_excluded", "Not in sample.exclude_gics_industries"),
    ("short_return_history", "At least sample.min_prior_return_days CRSP returns before Day 0"),
]


def sp500_member_on(events: pd.DataFrame, membership: pd.DataFrame,
                    date_col: str = "day0") -> pd.Series:
    """Boolean mask aligned to ``events``: ``permno`` was an S&P 500 member on ``date_col``."""
    sp = membership[["permno", "mbrstartdt", "mbrenddt"]].dropna(subset=["permno"]).copy()
    sp["permno"] = sp["permno"].astype("int64")
    for c in ("mbrstartdt", "mbrenddt"):
        sp[c] = pd.to_datetime(sp[c].astype("string"), errors="coerce")
    e = events[["permno", date_col]].reset_index(names="_row").dropna()
    e["permno"] = e["permno"].astype("int64")
    m = e.merge(sp, on="permno")
    hit = m.loc[valid_on(m, date_col, "mbrstartdt", "mbrenddt"), "_row"].unique()
    return pd.Series(events.index.isin(hit), index=events.index)


def fiscal_label(headline: pd.Series) -> pd.Series:
    """'Q4 2008' style label from the CIQ headline (agent C, ``scripts/full_source.py``).
    Annual-results calls ('..., 2011 Earnings Call') and FY labels map to Q4 of that year."""
    h = headline.astype("string")
    q = h.str.extract(FISCAL_RE, expand=False).str.upper().str.replace(r"[\s,]+", " ", regex=True)
    q = q.str.replace(r"^FY ", "Q4 ", regex=True)
    a = "Q4 " + h.str.extract(ANNUAL_RE, expand=False)
    return q.fillna(a)


def duplicate_calls(calls: pd.DataFrame, window_days: int | None = 3,
                    placeholder_last: bool = True) -> pd.Series:
    """keep_first: the keydevid each duplicate call is a duplicate of (NA for kept calls).

    ``calls``: ``keydevid``, ``transcriptid`` (unused), ``gvkey``, ``headline``, ``call_date``
    (UTC date, normalised) and ``call_time_utc`` ("HH:MM:SS"). Calls are ordered by call date and
    time within gvkey; with ``placeholder_last`` the 00:00:00 placeholder sorts as 23:59:59.
    Two calls with the same gvkey and fiscal label are one release when the later one is within
    ``window_days`` calendar days of the first of its run (None: any distance); beyond the window
    a new run starts (same label reused after a fiscal-year change). Calls with no fiscal label
    are never duplicates. Index aligned to ``calls``.
    """
    c = calls[["keydevid", "gvkey", "headline", "call_date", "call_time_utc"]].copy()
    t = c["call_time_utc"].astype("string")
    if placeholder_last:
        t = t.where(t.ne(PLACEHOLDER_TIME))
    t = t.fillna("23:59:59")
    c["_ts"] = pd.to_datetime(c["call_date"].dt.strftime("%Y-%m-%d") + " " + t, errors="coerce")
    c["_label"] = fiscal_label(c["headline"])
    c["_label"] = c["_label"].fillna("kd" + c["keydevid"].astype(str))
    c = c.sort_values(["gvkey", "_ts", "keydevid"])
    dup_of = pd.Series(pd.NA, index=c.index, dtype="Int64")
    dates = c["call_date"]
    for _, idx in c.groupby(["gvkey", "_label"], sort=False).groups.items():
        if len(idx) < 2:
            continue
        idx = list(idx)
        first = idx[0]
        for i in idx[1:]:
            if window_days is None or (dates[i] - dates[first]).days <= window_days:
                dup_of[i] = c.at[first, "keydevid"]
            else:
                first = i
    return dup_of.reindex(calls.index)


def choose_share_class(cands: pd.DataFrame, rule: str = "ccm_primary_then_mktcap") -> pd.DataFrame:
    """One row per keydevid from candidate rows (``keydevid``, ``permno``, ``linkprim``,
    ``mktcap_m1``, and an optional boolean ``eligible``: S&P 500 member on Day 0 with a release).

    Eligible candidates come first. Among them: a single candidate is kept; else the one linkprim P
    candidate; else (none or several P) the largest Day -1 market cap, then the lowest permno.
    Adds ``share_class_choice``: single | primary | mktcap.
    """
    if rule != "ccm_primary_then_mktcap":
        raise ValueError(rule)
    c = cands.copy()
    if "eligible" not in c:
        c["eligible"] = True
    c["_elig"] = c["eligible"].fillna(False).astype(bool)
    c["_n"] = c["_elig"].groupby(c["keydevid"]).transform("sum")
    c["_np"] = (c["_elig"] & c["linkprim"].eq("P")).groupby(c["keydevid"]).transform("sum")
    c["_prim"] = c["linkprim"].eq("P").fillna(False) & c["_np"].eq(1)
    c["_cap"] = pd.to_numeric(c["mktcap_m1"], errors="coerce").fillna(-np.inf)
    c = c.sort_values(["keydevid", "_elig", "_prim", "_cap", "permno"],
                      ascending=[True, False, False, False, True], na_position="last")
    out = c.drop_duplicates("keydevid").copy()
    out["share_class_choice"] = np.select(
        [out["_n"] <= 1, out["_prim"]], ["single", "primary"], "mktcap")
    return out.drop(columns=["_elig", "_n", "_np", "_prim", "_cap"]).reset_index(drop=True)


def apply_exclusions(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``exclusion_reason`` (NA when the event stays in the sample), plus
    ``duplicate_of_keydevid`` and ``duplicate_release_of_keydevid``.

    ``events``: one row per keydevid with ``event_date`` (Day 0, or the call date when there is no
    release), ``gvkey``, ``permno``, ``sp500_member``, ``ibes_ticker``, ``release_anndats``,
    ``release_pends``, ``headline``, ``call_date``, ``call_time_utc``, ``call_lag_sessions``,
    ``gind``, ``transcriptid`` (NA when no version by the cutoff) and ``n_prior_returns``.
    ``cfg``: the full config (``sample``, ``events``). Reasons are applied in ``REASONS`` order;
    each event gets the first one it fails.
    """
    s, ev = cfg["sample"], cfg["events"]
    e = events.copy()
    reason = pd.Series(pd.NA, index=e.index, dtype="string")

    def mark(name, bad):
        nonlocal reason
        bad = pd.Series(bad, index=e.index).fillna(False).astype(bool) & reason.isna()
        reason = reason.mask(bad, name)

    d = pd.to_datetime(e["event_date"])
    mark("outside_sample_period", d.isna() | (d < pd.Timestamp(s["start"]))
         | (d > pd.Timestamp(s["end"])))
    mark("no_gvkey", e["gvkey"].isna())
    mark("no_permno", e["permno"].isna())
    mark("not_sp500_member", ~e["sp500_member"].fillna(False).astype(bool))
    mark("no_ibes_ticker", e["ibes_ticker"].isna())
    mark("no_ibes_release", e["release_anndats"].isna())
    # before keep_first, so the first call kept for a release is one with usable text
    mark("no_transcript_version", e["transcriptid"].isna())

    e["duplicate_of_keydevid"] = pd.Series(pd.NA, index=e.index, dtype="Int64")
    if ev.get("duplicate_calls", "keep_first") == "keep_first":
        ok = reason.isna()
        dup = duplicate_calls(e[ok], ev.get("duplicate_window_days", 3),
                              ev.get("placeholder_time_last", True))
        e.loc[ok, "duplicate_of_keydevid"] = dup
        mark("duplicate_call", e["duplicate_of_keydevid"].notna())
        # several calls still on one I/B/E/S release: keep the first by call time
        ok = reason.isna()
        r = e[ok].copy()
        t = r["call_time_utc"].astype("string")
        if ev.get("placeholder_time_last", True):
            t = t.where(t.ne(PLACEHOLDER_TIME))
        r["_ts"] = pd.to_datetime(r["call_date"].dt.strftime("%Y-%m-%d") + " "
                                  + t.fillna("23:59:59"), errors="coerce")
        r = r.sort_values(["_ts", "keydevid"])
        key = ["ibes_ticker", "release_anndats", "release_pends"]
        first = r.groupby(key, dropna=False)["keydevid"].transform("first")
        e["duplicate_release_of_keydevid"] = pd.Series(pd.NA, index=e.index, dtype="Int64")
        e.loc[r.index, "duplicate_release_of_keydevid"] = first.where(first.ne(r["keydevid"]))
        mark("duplicate_release", e["duplicate_release_of_keydevid"].notna())
    else:
        raise ValueError(ev.get("duplicate_calls"))

    lag = pd.to_numeric(e["call_lag_sessions"], errors="coerce")
    mark("call_lag", lag.isna() | (lag < 0) | (lag > ev["max_call_lag_sessions"]))
    excl = {str(x) for x in s.get("exclude_gics_industries") or []}
    mark("gics_excluded", e["gind"].astype("string").isin(excl) if excl else False)
    mark("short_return_history",
         pd.to_numeric(e["n_prior_returns"], errors="coerce").fillna(0)
         < s.get("min_prior_return_days", 0))
    e["exclusion_reason"] = reason
    return e
