"""Earnings surprise and analyst controls from I/B/E/S (unadjusted).

Implemented with point-in-time unadjusted I/B/E/S actuals and summary estimates.

Definitions:
  consensus  = last ``medest`` with statpers < anndats for fpedats == pends (fpi 6).
  sue        = (actual - consensus) / price before Day 0          (scope doc, primary)
  sue_std    = (actual - consensus) / stdev                         (legacy, robustness)
  dispersion = stdev / |consensus| ; n_analysts = numest

Fixes vs. the legacy notebook (docs/data_audit.md §SUE):
  * Unadjusted tables + put actual and consensus on the same share basis with CRSP cfacshr
    between statpers and anndats (split-adjusted IBES rounds old EPS to 2 decimals).
  * Match actual to the event on anndats vs. Day 0, not +-5 calendar days around call_date.
  * stdev == 0 (single analyst or identical forecasts) is common; sue_std is NaN there,
    which is why the scope doc scales by price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()


def earnings_surprise(events: pd.DataFrame, actuals: pd.DataFrame, consensus: pd.DataFrame,
                      prices: pd.DataFrame, cfacshr: pd.DataFrame) -> pd.DataFrame:
    """Match point-in-time quarterly actuals/consensus and compute price-scaled and IBES SUE."""
    e = events.copy()
    a = actuals.copy()
    c = consensus.copy()
    p = prices[["permno", "date", "prc", "cfacshr"]].copy()
    _dates(a, ["pends", "anndats"])
    _dates(c, ["statpers", "fpedats"])
    _dates(p, ["date"])
    for df, col in ((a, "value"), (c, "medest"), (c, "stdev"), (c, "numest"),
                    (p, "prc"), (p, "cfacshr")):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    a["ticker"] = a["ticker"].astype("string").str.upper()
    c["ticker"] = c["ticker"].astype("string").str.upper()
    actual_by_ticker = {k: g.sort_values("anndats") for k, g in a.groupby("ticker")}
    cons_by_ticker = {k: g.sort_values("statpers") for k, g in c.groupby("ticker")}
    prices_by_permno = {k: g.sort_values("date") for k, g in p.groupby("permno")}
    ticker_col = "ibes_ticker" if "ibes_ticker" in e else "ticker"
    rows = []
    for r in e.itertuples(index=False):
        if pd.isna(r.day0):
            rows.append({"keydevid": r.keydevid})
            continue
        call_date = pd.Timestamp(getattr(r, "call_date_et", getattr(r, "call_date", r.day0))).tz_localize(None).normalize()
        call_ts = getattr(r, "call_ts_utc", pd.NaT)
        if pd.notna(call_ts):
            call_ts = pd.Timestamp(call_ts)
            if call_ts.tzinfo is None:
                call_ts = call_ts.tz_localize("UTC")
            call_ts = call_ts.tz_convert("America/New_York").tz_localize(None)
        ticker = getattr(r, ticker_col, None)
        aa = actual_by_ticker.get(str(ticker).upper())
        cc = cons_by_ticker.get(str(ticker).upper())
        if aa is None or cc is None:
            rows.append({"keydevid": r.keydevid})
            continue
        # Associate the earnings release with the call and require the actual to be
        # public by call time when both dates are the same.
        cand = aa[(aa["anndats"] <= call_date) & (aa["anndats"] >= call_date - pd.Timedelta(days=10))]
        if call_ts is not None and pd.notna(call_ts) and "anntims" in cand:
            same = cand["anndats"].eq(call_date)
            raw_time = cand["anntims"].astype("string").str.strip()
            # WRDS commonly returns HH:MM:SS. Removing punctuation and then
            # parsing everything after HH accidentally treats seconds as minutes.
            parts = raw_time.str.extract(r"^(\d{1,2}):(\d{2})")
            compact = raw_time.str.replace(r"\D", "", regex=True).str.zfill(4).str[:4]
            hour = parts[0].fillna(compact.str[:2])
            minute = parts[1].fillna(compact.str[2:4])
            minutes = pd.to_numeric(hour, errors="coerce") * 60 + pd.to_numeric(minute, errors="coerce")
            call_minutes = call_ts.hour * 60 + call_ts.minute
            cand = cand[~same | minutes.le(call_minutes)]
        elif call_ts is None or pd.isna(call_ts):
            # Without an earnings-call clock time, a same-day actual might have been
            # released after the call; require a strictly earlier announcement date.
            cand = cand[cand["anndats"] < call_date]
        if cand.empty:
            rows.append({"keydevid": r.keydevid})
            continue
        act = cand.iloc[(call_date - cand["anndats"]).dt.days.argmin()]
        fc = cc[(cc["fpedats"] == act["pends"]) & (cc["statpers"] < act["anndats"])]
        if "fpi" in fc:
            fc = fc[fc["fpi"].astype("string").isin(["6", "7"])]
        if fc.empty:
            rows.append({"keydevid": r.keydevid})
            continue
        con = fc.iloc[-1]
        # IBES occasionally contains a fiscal-period row with null actual or
        # summary statistics. Keep the event in the panel, but leave its SUE
        # fields missing instead of failing the full-universe build.
        if pd.isna(act["value"]) or pd.isna(con["medest"]):
            rows.append({"keydevid": r.keydevid})
            continue
        stock = prices_by_permno.get(r.permno)
        if stock is None:
            rows.append({"keydevid": r.keydevid})
            continue
        price_rows = stock[stock["date"] < pd.Timestamp(r.day0).normalize()]
        if price_rows.empty:
            rows.append({"keydevid": r.keydevid})
            continue
        pr = price_rows.iloc[-1]
        price = abs(float(pr["prc"])) if pd.notna(pr["prc"]) else np.nan
        actual, median, sd = float(act["value"]), float(con["medest"]), con["stdev"]
        # Put the pre-announcement consensus on the actual's share basis if CRSP
        # split factors show a stock split between the consensus and release dates.
        cf = stock[(stock["date"] >= con["statpers"]) & (stock["date"] <= act["anndats"])]
        if len(cf) and pd.notna(cf.iloc[0]["cfacshr"]) and pd.notna(cf.iloc[-1]["cfacshr"]):
            ratio = cf.iloc[-1]["cfacshr"] / cf.iloc[0]["cfacshr"]
            if pd.notna(ratio) and ratio > 0:
                median *= ratio
                sd = sd * ratio if pd.notna(sd) else np.nan
        surprise = actual - median
        rows.append({"keydevid": r.keydevid, "fiscal_period_end": act["pends"],
                     "actual_eps": actual, "consensus_eps": median,
                     "eps_surprise": surprise,
                     "sue": surprise / price if pd.notna(price) and price > 0 else np.nan,
                     "sue_std": surprise / sd if pd.notna(sd) and sd > 0 else np.nan,
                     "n_analysts": con["numest"],
                     "dispersion": sd / abs(median) if pd.notna(sd) and median != 0 else np.nan,
                     "consensus_statpers": con["statpers"], "actual_anndats": act["anndats"],
                     "price_pre_day0": price})
    return pd.DataFrame(rows)
