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


def sp500_member_on(events: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Boolean mask: permno was an S&P 500 member on ``call_date_et``."""
    raise NotImplementedError


def apply_exclusions(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``exclusion_reason`` (NA when the event stays in the sample)."""
    raise NotImplementedError
