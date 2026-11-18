"""Walk-forward splits by Day 0 quarter, with a per-target embargo and an untouched holdout.

  * Rolling windows (``split.walk_forward``): train on the ``train_quarters`` before a test
    block of ``test_quarters``, then roll forward by ``test_quarters``. Rolling is the current
    choice; the train / validation / test design is still under discussion (docs/pipeline.md).
  * Embargo per target: a training event is dropped when its target window ends on or after the
    first test Day 0. The window end is Day 0 + the target's last offset (20 sessions for
    CAR[+2,+20], 60 for CAR[+2,+60]), so every target gets its own embargo.
  * Validation (``split.inner_cv``): time-ordered folds inside each training window. Fold k
    validates on one of the window's last quarters and trains on the quarters before it, with the
    same embargo. Hyperparameters (penalties) are chosen there, never on the test block.
  * Events from ``split.holdout_start`` on are never returned unless ``holdout=True``, which
    yields one split: the ``train_quarters`` before the holdout (embargoed) vs. the whole holdout.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def target_end_offset(cfg: dict, target: str) -> int:
    """Last session offset of a target window (e.g. 20 for ``car_2_20``)."""
    windows = cfg["targets"]["car_windows"]
    if target in windows:
        return int(windows[target][1])
    vol = cfg["targets"].get("vol_post_window")
    if target.startswith("vol_") and vol:
        return int(vol[1])
    raise KeyError(f"unknown target {target!r}")


def session_offset(day0: pd.Series, offset: int,
                   sessions: pd.DatetimeIndex | None = None) -> pd.Series:
    """Date ``offset`` trading sessions after each Day 0 (business days if no calendar is given)."""
    day0 = pd.to_datetime(day0)
    if sessions is None:
        return day0 + pd.offsets.BDay(offset)
    sessions = pd.DatetimeIndex(sessions).sort_values()
    pos = sessions.searchsorted(day0.to_numpy())
    pos = np.minimum(pos + offset, len(sessions) - 1)
    out = pd.Series(sessions[pos], index=day0.index)
    beyond = (sessions.searchsorted(day0.to_numpy()) + offset) >= len(sessions)
    return out.where(~beyond, pd.NaT)   # window runs past the calendar: no complete target


def _quarters(day0: pd.Series) -> pd.Series:
    return pd.to_datetime(day0).dt.to_period("Q")


def embargo_mask(train_day0: pd.Series, first_test_day0: pd.Timestamp, end_offset: int,
                 sessions: pd.DatetimeIndex | None = None) -> pd.Series:
    """True for training events whose target window ends before the first test Day 0."""
    end = session_offset(train_day0, end_offset, sessions)
    return end.notna() & end.lt(first_test_day0)


def walk_forward(events: pd.DataFrame, cfg: dict, holdout: bool = False, target: str = "car_2_20",
                 sessions: pd.DatetimeIndex | None = None, day0_col: str = "day0"
                 ) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield ``(train_index, test_index)`` pairs of ``events.index``."""
    wf = cfg["split"]["walk_forward"]
    n_train, n_test = int(wf["train_quarters"]), int(wf["test_quarters"])
    end_offset = target_end_offset(cfg, target)
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    day0 = pd.to_datetime(events[day0_col])
    q = _quarters(day0)
    in_holdout = day0.ge(holdout_start)

    if holdout:
        test = events.index[in_holdout]
        first = q[in_holdout].min() if len(test) else None
        if first is None:
            return
        train_q = pd.period_range(first - n_train, first - 1, freq="Q")
        cand = q.isin(train_q) & ~in_holdout
        keep = embargo_mask(day0[cand], day0[in_holdout].min(), end_offset, sessions)
        yield events.index[cand][keep.to_numpy()], test
        return

    dev_q = pd.period_range(q[~in_holdout].min(), q[~in_holdout].max(), freq="Q")
    start = n_train
    while start < len(dev_q):
        test_q = dev_q[start:start + n_test]
        train_q = dev_q[start - n_train:start]
        test_mask = q.isin(test_q) & ~in_holdout
        if not test_mask.any():
            start += n_test
            continue
        cand = q.isin(train_q) & ~in_holdout
        keep = embargo_mask(day0[cand], day0[test_mask].min(), end_offset, sessions)
        yield events.index[cand][keep.to_numpy()], events.index[test_mask]
        start += n_test


def inner_cv(events: pd.DataFrame, train_index: pd.Index, cfg: dict, target: str = "car_2_20",
             sessions: pd.DatetimeIndex | None = None, day0_col: str = "day0"
             ) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Time-ordered validation folds inside one training window (forward chaining by quarter)."""
    n_folds = int(cfg["split"].get("inner_cv", {}).get("folds", 3))
    end_offset = target_end_offset(cfg, target)
    sub = events.loc[train_index]
    day0 = pd.to_datetime(sub[day0_col])
    q = _quarters(day0)
    quarters = sorted(q.unique())
    for val_q in quarters[-n_folds:]:
        val = q.eq(val_q)
        cand = q.lt(val_q)
        if not cand.any() or not val.any():
            continue
        keep = embargo_mask(day0[cand], day0[val].min(), end_offset, sessions)
        yield sub.index[cand][keep.to_numpy()], sub.index[val]
