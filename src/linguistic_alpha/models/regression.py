"""Pooled OLS with FE + clustered SEs, Fama-MacBeth by quarter (needs the ``models`` extra).

Status: specified, not implemented.
  * Penalties (lasso / ridge / enet) are chosen by time-ordered validation inside each training
    window (``splits.inner_cv``), never on the test block. Only language is penalised
    (``models.unpenalized``); fixed effects are removed by demeaning (``models.fe_method``).
  * ``*_resid`` interactions (tone on release metrics) are fitted here, on the training window,
    and applied to the test window: they are model-stage features.
  * Cross-sectionally winsorize / rank-normalize regressors per quarter before fitting.
  * Penalise only language (``models.unpenalized``): controls, release metrics and the reaction
    window get zero penalty in every model, so Model A is plain OLS under every estimator.
    With scikit-learn, partial them out first (Frisch-Waugh: residualise the target and the
    language measures on them within the window) and fit the penalised model on the residuals.
    For squared loss this is exact; the control coefficients then come from OLS of
    (target - language fit) on the controls.
  * Fixed effects by demeaning (``models.fe_method: demean``): demean target and regressors by
    ``spec.fixed_effects`` inside the training window, then fit with no dummies, so penalised
    estimators (lasso, ridge, enet) never shrink the fixed effects. Test rows subtract the
    training industry means; the test quarter's effect drops out.
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
