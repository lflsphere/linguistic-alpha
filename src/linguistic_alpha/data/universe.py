"""Sample universe and event exclusions.

Status: specified, not implemented.

Rules (project scope §3, meeting 10/23):
  * Historical S&P 500 membership on Day 0 (crsp.msp500list), never today's list. Index
    changes fall mostly in late December, outside earnings season, so Day 0 vs. call date
    rarely matters.
  * Several calls for one release (``events.duplicate_calls: keep_first``): keep the first
    call by call time; log the others.
  * Optional GICS exclusions (``sample.exclude_gics_industries``, empty: biotech is kept since
    2026-11-17), using comp.co_hgic history valid on the call date.
  * One permno per event (``sample.share_class_rule``): CCM linkprim P, else the share class
    with the largest market cap on Day -1.
  * Exclude calls with no transcript, no I/B/E/S consensus, or fewer than
    ``sample.min_prior_return_days`` days of CRSP returns before Day 0.
  * Keep an ``exclusion_reason`` column instead of dropping rows silently.
"""

from __future__ import annotations

import pandas as pd


def sp500_member_on(events: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Boolean mask: permno was an S&P 500 member on ``call_date_et``."""
    raise NotImplementedError


def apply_exclusions(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``exclusion_reason`` (NA when the event stays in the sample)."""
    raise NotImplementedError
