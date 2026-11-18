"""Cumulative abnormal returns and post-call volatility.

Status: specified, not implemented (inputs: crsp_dsf_v2, crsp_dsp500_v2).

Definitions (project scope §1, §4):
  AR(i,t)          = R(i,t) - R(bench,t); bench = S&P 500 total return, point in time
                     (crsp_dsp500_v2.vwretd: the index's constituents on each day, dividends
                     included like dlyret). Robustness: the market model,
                     AR = R(i) - (alpha + beta * R(S&P 500 total return)), alpha and beta by OLS on days -250..-30 before Day 0 (at least 80% valid days).
                     CRSP VW market = index check only; FF4 is left for a future risk-management leg.
  R(i,t)           = dlyret, which already includes delisting returns (no splice).
  CAR[a,b]         = prod(1 + R(i,t)) - prod(1 + R(bench,t)) over sessions Day0+a .. Day0+b:
                     the stock and the index are compounded separately (buy-and-hold), then
                     differenced. Null stock days are skipped in the stock product (never filled
                     with 0); the index is compounded over every day; the window is missing when
                     valid days < ``targets.min_window_coverage``. Market model: the benchmark
                     leg is alpha + beta * R(bench,t), compounded the same way.
  vol_post_2_20    = std(AR, +2..+20) * sqrt(252)
  vol_ratio_2_20   = vol_post_2_20 / (std(AR, -60..-5) * sqrt(252))   (scope definition)

Fixes vs. the legacy notebook (docs/data_audit.md §Targets):
  * Delisting: the legacy file needed R = (1+ret)(1+dlret)-1; CIZ ``dlyret`` already includes
    it, so splicing ``delret`` again would count it twice.
  * Benchmark consistency: the legacy notebook used ``sprtrn`` (price only) against a return
    that includes dividends (AR drifted up ~2%/yr); ``vwretd`` (total return) fixes it.
  * Coverage rule (``targets.min_window_coverage``) is reported per event, not only applied.
"""

from __future__ import annotations

import pandas as pd


def abnormal_returns(dsf: pd.DataFrame, bench: pd.DataFrame,
                     model: str = "sp500") -> pd.DataFrame:
    """Daily ``permno, date, ret, bench_ret, ar``; CIZ ``dlyret`` already holds delisting returns."""
    raise NotImplementedError


def event_car(events: pd.DataFrame, ar: pd.DataFrame, windows: dict[str, list[int]],
              min_coverage: float) -> pd.DataFrame:
    """One row per keydevid with every configured CAR window (stock and index compounded
    separately) and its valid-day coverage."""
    raise NotImplementedError


def post_call_volatility(events: pd.DataFrame, ar: pd.DataFrame, post: list[int],
                         pre: list[int]) -> pd.DataFrame:
    raise NotImplementedError
