"""Prepare the common A2 panel from authorized local WRDS extracts and archived Model A."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from linguistic_alpha.controls.financial_baselines import CORE, CORE_COLUMNS, MOMENTUM_COLUMN, screen_financial_controls
from linguistic_alpha.events.financial_timing import align_market_day, nyse_hours, session_close_utc, session_hours
from linguistic_alpha.io import read_table
from linguistic_alpha.targets.financial_car import event_car, financial_abnormal_returns, market_date_at, market_days


def archived_split(panel, calendar, windows, test_share):
    """Freeze the original common-label chronological cohort and its calendar embargo."""
    common = panel[list(windows)].notna().all(axis=1) & panel.call_date.notna()
    if not common.any():
        raise ValueError("No archived events with all original labels")
    counts = panel.loc[common,"call_date"].value_counts().sort_index()
    cutoff = counts.cumsum().ge(np.ceil((1-test_share)*int(common.sum()))).idxmax()
    horizon = max(end for _,end in windows.values())
    embargo_start = calendar[max(0,calendar.searchsorted(cutoff)-horizon)]
    before = panel.call_date.lt(cutoff)
    split = pd.Series("outside_archive",index=panel.index)
    split.loc[common] = "purged"
    split.loc[common & before & panel.original_day0.lt(embargo_start)] = "train"
    split.loc[common & ~before] = "test"
    return split,cutoff


def prepare_panel(raw: Path, baseline: Path, settings: dict, schedule_path: Path | None = None):
    windows = settings["target_windows"]
    columns = ["keydevid","permno","ticker","day0","call_date_et","call_ts_utc","actual_anndats",
               "consensus_statpers","mcap_date","book_equity_date",*CORE,*windows]
    panel = read_table(baseline.parent,baseline.stem,columns=columns)
    if panel.keydevid.isna().any() or panel.keydevid.duplicated().any():
        raise ValueError("Archived Model A must have unique nonmissing event IDs")
    panel = panel.rename(columns={"day0":"original_day0","call_date_et":"call_date"})
    for name in ("original_day0","call_date","actual_anndats","consensus_statpers","mcap_date","book_equity_date"):
        panel[name] = pd.to_datetime(panel[name],errors="coerce").dt.normalize()
    panel["signal_immediate_utc"] = pd.to_datetime(panel.call_ts_utc,utc=True,errors="coerce")

    dsi = read_table(raw,"crsp_dsi",columns=["date","vwretd"])
    calendar = market_days(dsi.date)
    schedule = pd.read_csv(schedule_path) if schedule_path is not None else nyse_hours(calendar)
    _,closes = session_hours(calendar,schedule)
    aligned = align_market_day(panel.signal_immediate_utc,calendar,schedule=schedule)
    panel["day0"] = aligned.day0
    panel["call_session"] = aligned.call_session
    panel["split"],cutoff = archived_split(panel,calendar,windows,settings["test_share"])
    panel["original_day1"] = market_date_at(panel.original_day0,calendar,1)

    stocks = read_table(raw,"crsp_dsf",columns=["permno","date","ret"])
    stocks["date"] = pd.to_datetime(stocks.date,errors="coerce").dt.normalize()
    delist = read_table(raw,"crsp_dsedelist",columns=["permno","dlstdt","dlret"])
    ar = financial_abnormal_returns(stocks,dsi,delist)
    panel = screen_financial_controls(panel,stocks,calendar,closes)
    horizon = max(end for _,end in windows.values())
    paired_eligible = panel.split.isin(["train","test"])
    for prefix,anchor in (("original_", "original_day0"),("", "day0")):
        event_keys = panel[["keydevid","permno",anchor]].rename(columns={anchor:"day0"})
        measured = event_car(event_keys,ar,windows,settings["minimum_coverage"],calendar)
        if not measured.keydevid.eq(panel.keydevid).all():
            raise AssertionError("Target rows changed order")
        for target in windows:
            panel[prefix+"measured_"+target] = measured[target]
            if not prefix:
                panel[target+"_coverage"] = measured[target+"_coverage"]
        day1 = market_date_at(panel[anchor],calendar,1)
        end = market_date_at(panel[anchor],calendar,horizon)
        paired_eligible &= panel.signal_immediate_utc.lt(session_close_utc(day1,closes))
        paired_eligible &= ~panel.split.eq("train") | end.lt(cutoff)
    for target in windows:
        panel["eligible_"+target] = (paired_eligible & panel["measured_"+target].notna()
                                      & panel["original_measured_"+target].notna())
        panel["archive_"+target] = panel[target]
        panel[target] = panel["measured_"+target]
    output = ["keydevid","permno","ticker","call_date","original_day0","day0","call_session",
              "signal_immediate_utc","actual_anndats","momentum_observation_end","split",
              *CORE_COLUMNS,MOMENTUM_COLUMN]
    for target in windows:
        output.extend([target,"archive_"+target,"eligible_"+target,target+"_coverage"])
    return panel[output]
