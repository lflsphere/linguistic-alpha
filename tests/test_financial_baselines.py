"""Synthetic regression tests for the new local financial-baseline path."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linguistic_alpha.config import load_config, REPO_ROOT
from linguistic_alpha.controls.financial_baselines import CORE_COLUMNS, compound_pre_announcement
from linguistic_alpha.events.financial_timing import align_market_day, nyse_hours
from linguistic_alpha.models.financial_baselines import archived_split
from linguistic_alpha.targets.financial_car import event_car, financial_abnormal_returns


def test_market_timing_handles_early_close_holiday_and_missing_clock():
    days = pd.DatetimeIndex(["2024-07-02","2024-07-03","2024-07-05"])
    stamp = pd.to_datetime(pd.Series(["2024-07-02T13:00Z","2024-07-02T14:00Z",
                                     "2024-07-03T17:00Z","2024-07-04T14:00Z",None]),utc=True)
    out = align_market_day(stamp,days,schedule=nyse_hours(days))
    assert out.call_session.tolist() == ["PRE","INTRADAY","POST","NON_TRADING","unknown"]
    assert out.day0.iloc[:4].tolist() == [days[0],days[1],days[2],days[2]]
    assert pd.isna(out.day0.iloc[4])


def test_short_momentum_uses_complete_market_sessions_before_release():
    days = pd.bdate_range("2023-01-02",periods=40)
    ret = pd.Series(.01,index=days)
    release = days[35]
    ret.loc[release:] = .99
    value,end = compound_pre_announcement(ret,days,release)
    assert value == pytest.approx(1.01**30-1)
    assert end == days[34]
    assert np.isnan(compound_pre_announcement(ret.drop(days[30]),days,release)[0])


def test_missing_return_is_not_zero_and_does_not_shift_target():
    days = pd.bdate_range("2024-01-02",periods=7)
    stocks = pd.DataFrame({"permno":1,"date":days,"ret":[0.,0.,.03,np.nan,.05,.06,0.]})
    dsi = pd.DataFrame({"date":days,"vwretd":.01})
    delist = pd.DataFrame({"permno":[1],"dlstdt":[days[4]],"dlret":[-.1]})
    ar = financial_abnormal_returns(stocks,dsi,delist)
    events = pd.DataFrame({"keydevid":[9],"permno":[1],"day0":[days[0]]})
    result = event_car(events,ar,{"car_2_5":[2,5]},.8,days).iloc[0]
    assert result.car_2_5_valid_days == 3 and pd.isna(result.car_2_5)
    assert ar.loc[ar.date.eq(days[4]),"ret_total"].item() == pytest.approx(1.05*.9-1)


def test_archived_split_preserves_the_chronological_embargo():
    days = pd.bdate_range("2024-01-02",periods=12)
    panel = pd.DataFrame({"call_date":days,"original_day0":days,"car_2_3":0.})
    split,cutoff = archived_split(panel,days,{"car_2_3":[2,3]},.25)
    assert cutoff == days[8]
    assert split.value_counts().to_dict() == {"train":5,"test":4,"purged":3}


def test_legacy_factory_defaults_and_new_specifications(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_model_a_dashboard import CONTROLS, pipeline

    cfg = load_config(REPO_ROOT/"configs/model_a_financial.yaml")["financial_baselines"]
    columns = [*CONTROLS,*CORE_COLUMNS]
    train = pd.DataFrame(columns=columns)
    legacy = pipeline("extra_trees",train,7)
    current = pipeline("extra_trees",train,7,controls=CORE_COLUMNS)
    assert legacy.named_steps["prep"].transformers[0][2] == CONTROLS
    assert current.named_steps["prep"].transformers[0][2] == CORE_COLUMNS
    assert "car_0_1" not in CORE_COLUMNS
    assert not cfg["specifications"]["a2"]["add_short_momentum"]
    assert cfg["specifications"]["a2_short_momentum"]["add_short_momentum"]
