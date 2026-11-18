"""Earnings surprise and analyst controls from I/B/E/S (unadjusted).

Status: specified, not implemented.

Definitions:
  consensus  = last ``medest`` with statpers < anndats for fpedats == pends (fpi 6).
  sue        = SUE_TS (primary, ``controls.sue_scaler: ts``): (actual - consensus) / SD of the
               firm's prior surprises, rolling over the previous 8 quarters (at least 4),
               shifted so the current surprise never enters its own scale. 2008-10 feed the
               scale only. Same definition as ``scripts/common.SUE_VARIANT``.
  sue_std    = (actual - consensus) / analyst stdev                 (robustness; NaN when stdev 0)
  sue_price  = (actual - consensus) / price before Day 0            (robustness, scope doc)
  dispersion = stdev / |consensus| ; n_analysts = numest

Fixes vs. the legacy notebook (docs/data_audit.md §SUE):
  * Unadjusted tables + put actual and consensus on the same share basis with CRSP cfacshr
    between statpers and anndats (split-adjusted IBES rounds old EPS to 2 decimals).
  * Match actual to the event on anndats vs. Day 0, not +-5 calendar days around call_date.
  * stdev == 0 (single analyst or identical forecasts) is common; sue_std is NaN there,
    which is why the analyst-SD version is only a robustness run.
"""

from __future__ import annotations

import pandas as pd


def earnings_surprise(events: pd.DataFrame, actuals: pd.DataFrame, consensus: pd.DataFrame,
                      prices: pd.DataFrame, cfacshr: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
