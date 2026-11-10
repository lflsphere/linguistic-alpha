"""Sample universe and event exclusions.

Status: specified, not implemented.

Rules (project scope §3, meeting 10/23):
  * Historical S&P 500 membership on the call date (crsp.msp500list), never today's list.
  * Exclude GICS Biotechnology (``sample.exclude_gics_industries``), using comp.co_hgic
    history valid on the call date.
  * Exclude calls with no transcript, no I/B/E/S consensus, or fewer than
    ``sample.min_prior_return_days`` days of CRSP returns before Day 0.
  * Keep an ``exclusion_reason`` column instead of dropping rows silently.
"""

from __future__ import annotations

import pandas as pd


def _day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def sp500_member_on(events: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Boolean mask: permno was an S&P 500 member on ``call_date_et``."""
    e = events.reset_index(drop=True)
    m = membership.copy()
    date_col = next((c for c in ("call_date_et", "call_date", "mostimportantdateutc") if c in e), None)
    if date_col is None:
        raise ValueError("events require a call date column")
    d = _day(e[date_col])
    start, end = _day(m["start"]), _day(m["ending"])
    by_permno = {}
    for permno, idx in e.groupby("permno", dropna=False).groups.items():
        candidates = m[m["permno"].eq(permno)]
        if candidates.empty:
            by_permno.update({i: False for i in idx})
            continue
        for i in idx:
            by_permno[i] = bool(((start.loc[candidates.index] <= d.iloc[i]) &
                                 (end.loc[candidates.index].isna() |
                                  (end.loc[candidates.index] >= d.iloc[i]))).any())
    return pd.Series(by_permno).reindex(range(len(e)), fill_value=False).astype(bool).set_axis(events.index)


def apply_exclusions(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``exclusion_reason`` (NA when the event stays in the sample)."""
    e = events.copy()
    reasons = pd.Series("", index=e.index, dtype="string")

    def exclude(mask: pd.Series, label: str) -> None:
        nonlocal reasons
        mask = mask.reindex(e.index).fillna(False).astype(bool)
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask].eq(""),
                                                    reasons.loc[mask] + ";") + label

    if "in_sp500" in e:
        exclude(~e["in_sp500"].fillna(False).astype(bool), "not_sp500_on_call_date")
    if "has_transcript" in e:
        exclude(~e["has_transcript"].fillna(False).astype(bool), "no_transcript")
    elif "transcriptid" in e:
        exclude(e["transcriptid"].isna(), "no_transcript")
    if "consensus" in e:
        exclude(e["consensus"].isna(), "no_ibes_consensus")
    if "prior_return_days" in e:
        minimum = int(cfg.get("sample", cfg).get("min_prior_return_days", 60))
        exclude(e["prior_return_days"].fillna(0).lt(minimum), "insufficient_prior_returns")
    if "gsubind" in e:
        biotech = cfg.get("sample", cfg).get("exclude_gics_industries", ["352010"])
        exclude(e["gsubind"].astype("string").str.replace(r"\.0$", "", regex=True)
                .isin([str(x) for x in biotech]), "excluded_gics_industry")
    e["exclusion_reason"] = reasons.mask(reasons.eq(""), pd.NA)
    e["included"] = e["exclusion_reason"].isna()
    return e
