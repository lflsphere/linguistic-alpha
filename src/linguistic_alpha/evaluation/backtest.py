"""Event-driven long-short backtest on predicted CAR[+2,+20].

Status: specified, not implemented.
  * Enter at close of Day +1, hold to Day +20; quintile sort within each calendar week.
  * Portfolios A (Model A), L (language only), B (Model B); gross and net of spreads
    (taqmsec) and borrow fees (markit); capacity curve vs. AUM; decay by day and year.
  * Counterfactual slice vs. RavenPack sentiment (prof 10/26): where RP and language disagree.
"""

from __future__ import annotations

import pandas as pd


def long_short(preds: pd.DataFrame, returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    raise NotImplementedError
