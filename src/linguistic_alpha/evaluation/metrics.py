"""Prediction-quality metrics.

Status: specified, not implemented.
  * Information coefficient: Spearman(pred, realized CAR) per period, then mean, t-stat.
  * ``delta_ic``: IC(Model B) - IC(Model A) on identical events (paired test).
"""

from __future__ import annotations

import pandas as pd


def information_coefficient(pred: pd.Series, actual: pd.Series, period: pd.Series) -> pd.Series:
    raise NotImplementedError


def compare_models(preds: pd.DataFrame, actual: pd.Series, period: pd.Series) -> pd.DataFrame:
    raise NotImplementedError
