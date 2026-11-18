"""Leakage checks for walk-forward splits: embargo per target, holdout, inner validation."""

import pandas as pd
import pytest

from linguistic_alpha.config import load_config
from linguistic_alpha.models.splits import inner_cv, session_offset, target_end_offset, walk_forward


@pytest.fixture
def cfg():
    return load_config().raw


@pytest.fixture
def events():
    # 5 events per week, 2011-01 to 2026-08: dense enough to have events near every boundary
    days = pd.bdate_range("2011-01-03", "2026-08-31")[::1]
    return pd.DataFrame({"day0": days}, index=pd.RangeIndex(len(days), name="event"))


@pytest.mark.parametrize("target", ["car_2_20", "car_2_60", "car_2_5"])
def test_no_training_target_window_reaches_the_test_block(cfg, events, target):
    end = target_end_offset(cfg, target)
    n = 0
    for train, test in walk_forward(events, cfg, target=target):
        n += 1
        first_test = events.loc[test, "day0"].min()
        train_end = session_offset(events.loc[train, "day0"], end)
        assert train_end.max() < first_test
        assert events.loc[train, "day0"].max() < first_test
        assert not set(train) & set(test)
    assert n > 20


def test_embargo_grows_with_the_target_window(cfg, events):
    short = [len(tr) for tr, _ in walk_forward(events, cfg, target="car_2_20")]
    long = [len(tr) for tr, _ in walk_forward(events, cfg, target="car_2_60")]
    assert len(short) == len(long)
    assert all(lo < sh for sh, lo in zip(short, long))


def test_dev_splits_never_touch_the_holdout(cfg, events):
    start = pd.Timestamp(cfg["split"]["holdout_start"])
    for train, test in walk_forward(events, cfg):
        assert events.loc[test, "day0"].max() < start
        assert events.loc[train, "day0"].max() < start


def test_holdout_is_one_split_with_embargo(cfg, events):
    splits = list(walk_forward(events, cfg, holdout=True, target="car_2_60"))
    assert len(splits) == 1
    train, test = splits[0]
    start = pd.Timestamp(cfg["split"]["holdout_start"])
    assert events.loc[test, "day0"].min() >= start
    assert session_offset(events.loc[train, "day0"], 60).max() < events.loc[test, "day0"].min()


def test_rolling_window_has_fixed_length(cfg, events):
    n_train = cfg["split"]["walk_forward"]["train_quarters"]
    for train, _ in walk_forward(events, cfg):
        assert events.loc[train, "day0"].dt.to_period("Q").nunique() == n_train


def test_inner_cv_is_time_ordered_and_embargoed(cfg, events):
    train, _ = next(walk_forward(events, cfg))
    folds = list(inner_cv(events, train, cfg, target="car_2_20"))
    assert len(folds) == cfg["split"]["inner_cv"]["folds"]
    for fit, val in folds:
        assert set(fit) <= set(train) and set(val) <= set(train)
        assert session_offset(events.loc[fit, "day0"], 20).max() < events.loc[val, "day0"].min()
