"""Immediate financial controls and the audited announcement-date momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd

from linguistic_alpha.events.financial_timing import session_close_utc
from linguistic_alpha.events.timing import NY
from linguistic_alpha.targets.financial_car import numeric


CORE = ["sue", "size", "bm", "mom", "vol_pre", "n_analysts", "dispersion"]
CORE_COLUMNS = ["pit_" + name for name in CORE]
MOMENTUM_COLUMN = "valid_mom_pre_30"


def compound_pre_announcement(ret, calendar, release_date):
    """Compound the 30 market sessions strictly before ANNDATS, with full coverage."""
    if pd.isna(release_date):
        return np.nan, pd.NaT
    index = calendar.searchsorted(release_date)
    if index < 30:
        return np.nan, pd.NaT
    window = calendar[index-30:index]
    values = ret.reindex(window)
    if values.count() != 30:
        return np.nan, window[-1]
    return (1 + values).prod() - 1, window[-1]


def screen_financial_controls(panel, stocks, calendar, closes):
    """Apply the archived A2 feature gates at the recorded CIQ clock."""
    panel = panel.copy()
    source = stocks.assign(ret=numeric(stocks.ret).where(lambda x: x.ge(-1)))
    stock_returns = {k:g.set_index("date").ret.sort_index() for k,g in source.groupby("permno")}
    dates = []
    for row in panel.itertuples(index=False):
        ret = stock_returns.get(row.permno)
        days = ret.index if ret is not None else pd.DatetimeIndex([])
        position = days.searchsorted(row.original_day0)
        exact = position < len(days) and days[position] == row.original_day0
        price = days[position-1] if position else pd.NaT
        vol_end = days[position-5] if exact and position >= 60 else pd.NaT
        momentum, momentum_end = np.nan, pd.NaT
        if ret is not None and row.actual_anndats < row.original_day1:
            momentum, momentum_end = compound_pre_announcement(ret,calendar,row.actual_anndats)
        dates.append((price,vol_end,momentum,momentum_end))
    evidence = pd.DataFrame(dates,index=panel.index,columns=["price_date","vol_end","short_mom","short_end"])
    signal = panel.signal_immediate_utc
    local_date = signal.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    close = lambda day: session_close_utc(day,closes)
    prior_consensus = panel.consensus_statpers.lt(panel.actual_anndats) & panel.consensus_statpers.lt(local_date)
    mom_end = panel.call_date.dt.to_period("M").dt.start_time - pd.DateOffset(months=2)
    position = calendar.searchsorted(mom_end,side="right")-1
    mom_date = pd.Series(calendar.take(position.clip(min=0)),index=panel.index).where(position>=0)
    market_cap_before = close(panel.mcap_date).lt(signal)
    usable = {
        "sue": close(evidence.price_date).lt(signal) & panel.actual_anndats.lt(local_date) & prior_consensus,
        "size": market_cap_before,
        "bm": market_cap_before & panel.book_equity_date.lt(local_date),
        "mom": close(mom_date).lt(signal),
        "vol_pre": close(evidence.vol_end).lt(signal),
        "n_analysts": prior_consensus,
        "dispersion": prior_consensus,
    }
    for name,allowed in usable.items():
        panel["pit_"+name] = numeric(panel[name]).where(allowed)
    panel["momentum_observation_end"] = evidence.short_end
    panel[MOMENTUM_COLUMN] = evidence.short_mom.where(close(evidence.short_end).lt(signal))
    return panel
