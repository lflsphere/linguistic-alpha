"""Event-driven long-short backtest on predicted CAR[+2,+20].

Status: specified, not implemented.
  * Enter at close of Day +1, hold to Day +20; quintile sort within each calendar week.
  * Portfolios A (Model A), L (language only), B (Model B); gross and net of spreads and
    borrow fees (sources open: ``backtest.costs``); capacity curve vs. AUM; decay by day and year.
  * Trading costs enter only here, the last stage. RavenPack counterfactual: future work.
"""

from __future__ import annotations

import pandas as pd


def long_short(preds: pd.DataFrame, returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    raise NotImplementedError
