"""Walk-forward splits by call quarter, with an embargo and an untouched holdout.

Status: specified, not implemented.
  * Train on ``split.walk_forward.train_quarters``, test on the next ``test_quarters``, roll.
  * Embargo: drop training events whose target window overlaps the first test Day 0
    (``embargo_trading_days`` = length of the target window).
  * Events after ``split.holdout_start`` are never returned until ``holdout=True``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def walk_forward(events: pd.DataFrame, cfg: dict, holdout: bool = False
                 ) -> Iterator[tuple[pd.Index, pd.Index]]:
    raise NotImplementedError
