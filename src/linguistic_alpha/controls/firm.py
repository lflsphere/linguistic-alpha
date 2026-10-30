"""Firm characteristics: size, book-to-market, momentum, pre-event volatility, CAR[0,+1].

Status: specified, not implemented.

Definitions:
  size    = ln(|prc| * shrout * 1000) at the last trading day of the month before the call
  bm      = book equity / market cap, book equity from the latest fundq with rdq < Day 0;
            BE = seqq + txditcq - pstkq (fallback ceqq); negative BE -> NaN
  mom     = prod(1+ret) over months -12..-2 relative to the call month
  vol_pre = std(AR, -60..-5) * sqrt(252)
  car_0_1 = sum AR over Day 0 and Day +1 (control, not target)

Fixes vs. the legacy notebook (docs/data_audit.md §Controls): missing returns inside the
momentum window are not filled with 0; fundq rows with NULL rdq use datadate + 90 days lag.
"""

from __future__ import annotations

import pandas as pd


def firm_controls(events: pd.DataFrame, dsf: pd.DataFrame, fundq: pd.DataFrame,
                  ar: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
