"""Pooled OLS with FE + clustered SEs, Fama-MacBeth by quarter (needs the ``models`` extra).

Status: specified, not implemented.
  * Cross-sectionally winsorize / rank-normalize regressors per quarter before fitting.
  * Report t-stats with Newey-West on the Fama-MacBeth slope series, and BH-FDR across the
    language features (success criterion: |t| > 3 or FDR-significant).
"""

from __future__ import annotations

import pandas as pd

from linguistic_alpha.models.specs import ModelSpec


def fama_macbeth(panel: pd.DataFrame, spec: ModelSpec, period: str = "quarter") -> pd.DataFrame:
    raise NotImplementedError


def pooled_ols(panel: pd.DataFrame, spec: ModelSpec, cluster: tuple[str, ...] = ("permno",)
               ) -> pd.DataFrame:
    raise NotImplementedError


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    """Out-of-sample predictions of ``spec.target`` for ``test``."""
    raise NotImplementedError
