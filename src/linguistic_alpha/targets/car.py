"""Cumulative abnormal returns and post-call volatility.

Status: specified, not implemented (waits on crsp_dsf / crsp_dsi / crsp_dsedelist).

Definitions (project scope §1, §4):
  AR(i,t)          = R(i,t) - R(bench,t); bench = S&P 500 (crsp.dsi), FF4 for robustness.
  CAR[a,b]         = sum of AR over sessions Day0+a .. Day0+b (offsets via events.timing).
  vol_post_2_20    = std(AR, +2..+20) * sqrt(252)
  vol_ratio_2_20   = vol_post_2_20 / std(AR, -60..-5) * sqrt(252)   (scope definition)

Fixes vs. the legacy notebook (docs/data_audit.md §Targets):
  * Merge delisting returns: R = (1+ret)(1+dlret)-1 on the delisting date.
  * Benchmark consistency: ``sprtrn`` is a price return while CRSP ``ret`` includes dividends;
    use ``vwretd`` or a total-return S&P series, or document the drift (~2%/yr).
  * Coverage rule (``targets.min_window_coverage``) is reported per event, not only applied.
"""

from __future__ import annotations

import pandas as pd


def abnormal_returns(dsf: pd.DataFrame, bench: pd.DataFrame, delist: pd.DataFrame,
                     model: str = "sp500") -> pd.DataFrame:
    """Daily ``permno, date, ret, ar`` with delisting returns folded in."""
    raise NotImplementedError


def event_car(events: pd.DataFrame, ar: pd.DataFrame, windows: dict[str, list[int]],
              min_coverage: float) -> pd.DataFrame:
    """One row per keydevid with every configured CAR window."""
    raise NotImplementedError


def post_call_volatility(events: pd.DataFrame, ar: pd.DataFrame, post: list[int],
                         pre: list[int]) -> pd.DataFrame:
    raise NotImplementedError
