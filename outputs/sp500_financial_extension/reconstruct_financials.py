"""Reconstruct the archived financial fields from local WRDS sources."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
from io import StringIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "outputs/sp500_car_extension"))
from extend_car import NY, RAW, TEAM, call_timestamp_utc, locate_day0, verify_mapping

INPUT = ROOT / "outputs/sp500_car_extension/sp500_car_extended.csv"
MARKET = ["mcap_m", "size", "mom", "vol_pre", "vol_post_2_20"]
EPS = ["fiscal_period_end", "actual_eps", "consensus_eps", "eps_surprise", "sue", "sue_price",
       "surprise_pct", "n_analysts", "dispersion", "split_adjusted_consensus"]
FEATURES = MARKET + ["bm"] + EPS
DATE = "fiscal_period_end"
BOOL = "split_adjusted_consensus"
OUTPUTS = ["sp500_model_extended.csv", "feature_validation.csv", "feature_coverage.csv",
           "feature_provenance.csv", "feature_lineage.md", "summary_zh.md"]
AGGREGATE_COVERAGE = "feature_coverage_aggregate.csv"
RECON = "reconstructed"
OLD_YEAR = 2024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv():
    with INPUT.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        names, data = next(reader), list(reader)
    need(len(names) == len(set(names)) and len(data) == 29698, "Unexpected input shape")
    need(all(len(x) == len(names) for x in data), "Malformed input CSV")
    cols = ["keydevid", "transcriptid", "permno", "gvkey", "ticker", "sector",
            "call_date", "call_time_utc", "day0", *FEATURES]
    ix = [names.index(n) for n in cols]
    e = pd.DataFrame([[r[i] for i in ix] for r in data], columns=cols)
    for c in ["keydevid", "transcriptid", "permno", "gvkey", *[x for x in FEATURES if x not in {DATE, BOOL}]]:
        e[c] = pd.to_numeric(e[c], errors="raise")
    for c in ["call_date", "day0", DATE]:
        e[c] = pd.to_datetime(e[c], errors="raise")
    e[BOOL] = e[BOOL].map({"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0})
    for c in ["keydevid", "transcriptid"]:
        need(e[c].is_unique and e[c].notna().all(), f"Nonunique {c}")
    e["clock"] = call_timestamp_utc(e.call_date, e.call_time_utc)
    need(e.clock.notna().all(), "Unparseable archived call clock")
    e["et_date"] = e.clock.dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    e["year"] = e.call_date.dt.year
    return names, data, e


def date_columns(d, names):
    for n in names:
        d[n] = pd.to_datetime(d[n], errors="raise").dt.normalize()
    return d


def num_columns(d, names):
    for n in names:
        d[n] = pd.to_numeric(d[n], errors="raise")
    return d


def dstr(d):
    return str(pd.Timestamp(d).date()) if pd.notna(d) else ""


def read_benchmark(path, date):
    d = date_columns(num_columns(pd.read_parquet(path, columns=[date, "sprtrn"]), ["sprtrn"]), [date])
    need(d[date].is_unique and np.isfinite(d.sprtrn).all(), f"Invalid benchmark in {path.name}")
    return d.set_index(date).sprtrn.astype(float).sort_index()


class Market:
    def __init__(self, events, inventory):
        self.legacy_bench = read_benchmark(RAW / "crsp_dsi.parquet", "date")
        self.ciz_bench = read_benchmark(TEAM / "crsp_dsp500_v2.parquet", "caldt")
        need(self.legacy_bench.index.isin(self.ciz_bench.index).all(), "CRSP calendars disagree")
        self.cut = self.legacy_bench.index[-1]
        self.bench = pd.concat([self.legacy_bench, self.ciz_bench.loc[self.ciz_bench.index > self.cut]])
        self.calendar = self.bench.index
        ff = pd.read_parquet(TEAM / "ff_factors_daily.parquet", columns=["date"])
        ff_dates = pd.DatetimeIndex(pd.to_datetime(ff.date)).sort_values()
        existing = ff_dates[(ff_dates >= self.ciz_bench.index[0]) & (ff_dates <= self.calendar[-1])]
        need(ff_dates.is_unique and existing.equals(self.ciz_bench.index),
             "Auxiliary date calendar does not reproduce all CIZ trading dates")
        # Dates only. This permits a known Day 0 after the last return without
        # permitting any missing stock or benchmark observation to be filled.
        self.event_calendar = self.calendar.union(ff_dates[ff_dates > self.calendar[-1]])
        inventory["ff_factors_daily.parquet (calendar dates only)"] = (
            ff_dates.min(), ff_dates.max(), len(ff_dates))
        self.sid = pd.Index(sorted(events.permno.unique()))
        self.legacy = self.empty()
        self.modern = self.empty()
        for y in (2020, 2021, 2023, 2024):
            self.read(RAW / f"crsp_dsf_{y}.parquet", self.legacy, False, inventory)
        for y in (2024, 2025, 2026):
            self.read(TEAM / f"crsp_dsf_v2_{y}.parquet", self.modern, True, inventory)
        inventory["crsp_dsp500_v2.parquet SPRTRN"] = (self.ciz_bench.index.min(), self.ciz_bench.index.max(),
                                                     len(self.ciz_bench))
        last = inventory["crsp_dsf_v2_2026.parquet"][1]
        need(last == self.calendar[-1], "Latest stock and benchmark dates disagree")
        self.legacy_delist()
        end = self.calendar.searchsorted(self.cut, side="right")
        self.combined = {key: np.column_stack((self.legacy[key][:, :end], self.modern[key][:, end:]))
                         for key in ("ret", "cap", "price")}
        # The original archive's factors are anchored to its own terminal share basis.
        # For every production EPS pair use the CIZ factors at both ends of the ratio.
        self.combined["factor"] = self.modern["factor"]
        self.combined["factor_day"] = self.modern["factor_day"]

    def empty(self):
        return {k: np.full((len(self.sid), len(self.calendar)), np.nan)
                for k in ("ret", "cap", "price", "factor", "factor_day")}

    def read(self, path, dest, ciz, inventory):
        cols = ({"dlycaldt": "date", "dlyret": "ret", "dlyprc": "prc",
                 "shrout": "shares", "dlycumfacpr": "factor", "dlyfacprc": "factor_day"} if ciz else
                {"date": "date", "ret": "ret", "prc": "prc", "shrout": "shares", "cfacpr": "factor"})
        d = pd.read_parquet(path, columns=["permno", *cols]).rename(columns=cols)
        date_columns(d, ["date"])
        num_columns(d, [k for k in d if k != "date"])
        need(not d.duplicated(["permno", "date"]).any(), f"Duplicate stock sessions in {path.name}")
        inventory[path.name] = (d.date.min(), d.date.max(), len(d))
        i, j = self.sid.get_indexer(d.permno), self.calendar.get_indexer(d.date)
        need((j >= 0).all(), f"Unknown CRSP sessions in {path.name}")
        use = i >= 0
        d["price"] = d.prc.abs().where(d.prc.ne(0))
        d["cap"] = d.price * d.shares / 1000
        for k in dest.keys() & set(d.columns):
            dest[k][i[use], j[use]] = d.loc[use, k].to_numpy(dtype=float, na_value=np.nan)

    def legacy_delist(self):
        d = pd.read_parquet(RAW / "crsp_dsedelist.parquet", columns=["permno", "dlstdt", "dlret"]).dropna()
        date_columns(d, ["dlstdt"])
        num_columns(d, ["permno", "dlret"])
        r = self.sid.get_indexer(d.permno)
        t = self.calendar.searchsorted(d.dlstdt)
        use = (r >= 0) & d.dlstdt.le(self.cut).to_numpy() & (t < len(self.calendar))
        r, t = r[use], t[use]
        need(not pd.MultiIndex.from_arrays([r, t]).has_duplicates, "Duplicate legacy delist security sessions")
        # The historical validation grid deliberately includes only loaded years.
        within = self.calendar[t].year.isin([2020, 2021, 2023, 2024])
        r, t = r[within], t[within]
        ret = self.legacy["ret"][r, t]
        dl = d.loc[use, "dlret"].to_numpy(dtype=float)[within]
        self.legacy["ret"][r, t] = np.where(np.isnan(ret), dl, (1 + ret) * (1 + dl) - 1)

    def position(self, date, side="left"):
        return self.calendar.searchsorted(date, side=side)

    def asof(self, row, col, index, tolerance):
        if index < 0:
            return np.nan, -1
        start = max(0, index - tolerance)
        vals = row[col][start:index+1]
        hit = np.flatnonzero(np.isfinite(vals) & (vals > 0))
        return (vals[hit[-1]], start + hit[-1]) if len(hit) else (np.nan, -1)

    def window(self, record, timing, production, source):
        security = self.sid.get_loc(record.permno)
        slot = self.bench if production else self.legacy_bench
        cal = slot.index
        d0 = timing
        data = {k: v[security] for k, v in source.items()}
        month = record.call_date.replace(day=1)
        prev_close = self.position(month) - 1
        max_boundary = cal[-1] + pd.Timedelta(days=1)
        values, reasons, stamps = {}, {}, {}
        if month <= max_boundary:
            cap, day = self.asof(data, "cap", prev_close, 10)
            if pd.notna(cap):
                values.update(mcap_m=cap, size=np.log(cap * 1e6))
                stamps["cap_observation"] = dstr(self.calendar[day])
            else:
                reasons.update(mcap_m="no_same_security_cap_within_10_sessions", size="missing_market_cap")
        else:
            reasons.update(mcap_m="prior_month_outside_crsp", size="missing_market_cap")
        start, end = month - pd.DateOffset(months=12), month - pd.DateOffset(months=1)
        if end > max_boundary:
            reasons["mom"] = "momentum_end_outside_crsp"
        else:
            a, b = self.position(start), self.position(end)
            rr = data["ret"][a:b]
            observed = rr[np.isfinite(rr)]
            if len(observed) >= 150 and (observed >= -1).all():
                values["mom"] = -1. if (observed == -1).any() else np.expm1(np.log1p(observed).sum())
                stamps["momentum_last_session"] = dstr(self.calendar[b-1])
            else:
                reasons["mom"] = "fewer_than_150_valid_returns"
        for feature, lo, hi, minimum in (("vol_pre", -60, -5, 40), ("vol_post_2_20", 2, 20, 15)):
            if d0 < 0:
                reasons[feature] = "day0_outside_crsp"
            elif d0+lo < 0 or d0+hi >= len(cal):
                reasons[feature] = "volatility_window_outside_crsp"
            else:
                ar = data["ret"][d0+lo:d0+hi+1] - slot.iloc[d0+lo:d0+hi+1].to_numpy()
                ar = ar[np.isfinite(ar)]
                if len(ar) >= minimum:
                    values[feature] = np.std(ar, ddof=1) * np.sqrt(252)
                    stamps[feature + "_window_end"] = dstr(cal[d0+hi])
                else:
                    reasons[feature] = f"fewer_than_{minimum}_paired_returns"
        return values, reasons, stamps


class Fundamentals:
    def __init__(self, path, inventory):
        f = pd.read_parquet(path, columns=["gvkey", "datadate", "rdq", "ceqq"])
        date_columns(f, ["datadate", "rdq"])
        num_columns(f, ["ceqq", "gvkey"])
        inventory[f"{path.parent.name}/{path.name} datadate"] = (f.datadate.min(), f.datadate.max(), len(f))
        inventory[f"{path.parent.name}/{path.name} RDQ"] = (f.rdq.min(), f.rdq.max(), f.rdq.count())
        usable = f.dropna(subset=["rdq", "ceqq", "datadate"])
        self.source = {g: d.sort_values("rdq", kind="stable") for g, d in usable.groupby("gvkey")}

    def select(self, e, day0, cap, production):
        if day0 is None:
            return np.nan, "day0_outside_crsp", None
        if not pd.notna(cap) or cap <= 0:
            return np.nan, "missing_market_cap", None
        candidates = self.source.get(e.gvkey)
        if candidates is None:
            return np.nan, "no_reported_ceqq", None
        # RDQ carries a date, so a release dated the call day cannot be verified as known.
        cutoff = min(day0, e.et_date) if production else day0
        f = candidates.loc[candidates.rdq.lt(cutoff)]
        f = f.loc[f.rdq.ge(day0 - pd.Timedelta(days=200))]
        if production:
            f = f.loc[f.datadate.le(f.rdq)]
        if f.empty:
            return np.nan, "no_eligible_rdq_within_200_days", None
        same = f.loc[f.rdq.eq(f.rdq.iloc[-1])]
        if production and same[["datadate", "ceqq"]].drop_duplicates().shape[0] > 1:
            return np.nan, "ambiguous_latest_ceqq", None
        row = same.iloc[-1]
        if row.ceqq <= 0:
            return np.nan, "nonpositive_ceqq", row.rdq
        return row.ceqq / cap, "", row.rdq


class Earnings:
    def __init__(self, folder, years, inventory):
        actual = pd.read_parquet(folder / "ibes_actu_epsus.parquet")
        date_columns(actual, ["pends", "anndats"])
        num_columns(actual, ["value"])
        inventory[f"{folder.name} unadjusted actual ANNDATS"] = (actual.anndats.min(), actual.anndats.max(), len(actual))
        actual = actual.loc[actual.pdicity.eq("QTR")].dropna(subset=["anndats", "pends", "value"])
        clock = actual.anntims.astype("string").str.strip().fillna("")
        valid_clock = clock.str.fullmatch(r"\d{1,2}:\d\d:\d\d") & clock.ne("00:00:00")
        actual["release"] = (actual.anndats + pd.to_timedelta(clock.where(valid_clock, ""), errors="coerce")
                             ).dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        self.actual = {t: d.reset_index(drop=True) for t, d in actual.groupby("ticker", sort=False)}
        forecasts = []
        for y in years:
            f = pd.read_parquet(folder / f"ibes_statsumu_epsus_{y}.parquet")
            date_columns(f, ["statpers", "fpedats"])
            num_columns(f, ["medest", "stdev", "numest"])
            inventory[f"{folder.name}/{y} unadjusted consensus STATPERS"] = (f.statpers.min(), f.statpers.max(), len(f))
            forecasts.append(f.loc[f.fpi.astype(str).eq("6")].dropna(subset=["statpers", "fpedats", "medest"]))
        c = pd.concat(forecasts, ignore_index=True).sort_values("statpers", kind="stable")
        self.forecasts = {k: g for k, g in c.groupby(["ticker", "fpedats"], sort=False)}
        links = pd.read_parquet(folder / "wrdsapps_ibcrsphist.parquet")
        num_columns(links, ["permno", "score"])
        date_columns(links, ["sdate", "edate"])
        self.links = {p: d for p, d in links.dropna(subset=["ticker", "score", "sdate"]).groupby("permno")}
        self.link_end = links.edate.max()
        self.security = {}
        if folder == TEAM:
            security = pd.read_parquet(folder / "crsp_stksecurityinfohist.parquet", columns=[
                "permno", "secinfostartdt", "secinfoenddt", "cusip"])
            date_columns(security, ["secinfostartdt", "secinfoenddt"])
            self.security = {p: g for p, g in security.dropna(subset=["cusip"]).groupby("permno")}

    def ticker(self, e):
        links = self.links.get(e.permno)
        if links is None:
            return None, "no_ibes_security_link", ""
        d = links.loc[links.sdate.le(e.call_date) & (links.edate.ge(e.call_date) | links.edate.isna())]
        if d.empty:
            # Extend only a common extract-end link whose CUSIP is independently
            # observed in a single security-history spell across the gap.
            current = self.security.get(e.permno)
            if e.call_date <= self.link_end or current is None:
                return None, "no_date_valid_ibes_security_link", ""
            known = current.loc[current.secinfostartdt.le(self.link_end)
                                & current.secinfoenddt.ge(e.call_date), "cusip"].unique()
            interval = current.loc[current.secinfostartdt.le(e.call_date)
                                   & current.secinfoenddt.ge(self.link_end), "cusip"]
            if len(known) != 1 or interval.nunique() != 1:
                return None, "ibes_link_end_without_continuous_cusip", ""
            d = links.loc[links.edate.eq(self.link_end) & links.ncusip.eq(known[0])]
            if d.empty:
                return None, "ibes_link_cusip_disagreement", ""
            basis = "continuous_cusip_across_ibes_link_extract_end"
        else:
            basis = "date_valid_ibes_link"
        ticker = d.loc[d.score.eq(d.score.min()), "ticker"].unique()
        if len(ticker) != 1:
            return None, "ambiguous_ibes_security_link", ""
        if basis.startswith("continuous") and e.ticker and e.ticker != ticker[0]:
            return None, "archived_ibes_ticker_disagrees_with_cusip_bridge", ""
        return ticker[0], "", basis

    def match(self, e, production):
        ticker, reason, basis = self.ticker(e) if production else (e.ticker, "", "archive_ticker_replay")
        if reason:
            return None, None, reason
        a = self.actual.get(ticker)
        if a is None:
            return None, None, "no_ibes_actual_for_ticker"
        gap = (a.anndats - e.call_date).abs()
        found = a.loc[gap.le(pd.Timedelta(days=5))].copy()
        if found.empty:
            return None, None, "no_quarterly_actual_within_5_days"
        found["gap"] = gap
        # First identify the earnings event as in the archive. Do not substitute
        # a different release merely because the nearest one was unavailable.
        a = found.loc[found.gap.eq(found.gap.min())]
        if production:
            a = a.loc[a.anndats.lt(e.et_date) | (a.anndats.eq(e.et_date) & a.release.le(e.clock))]
            if a.empty:
                return None, None, "actual_after_call_or_same_day_clock_unverified"
            if a[["anndats", "pends", "value", "curr_act"]].drop_duplicates().shape[0] != 1:
                return None, None, "ambiguous_quarterly_actual"
            a = a.sort_values("release", kind="stable", na_position="first")
        a = a.iloc[-1] if production else a.iloc[0]
        if production and a.pends > a.anndats:
            return None, None, "actual_fiscal_period_after_release"
        c = self.forecasts.get((ticker, a.pends))
        if c is None:
            return None, None, "no_consensus_for_matching_fiscal_period"
        c = c.loc[c.statpers.lt(a.anndats) & c.statpers.ge(a.anndats-pd.Timedelta(days=120))]
        if c.empty:
            return None, None, "no_pre_release_consensus_within_120_days"
        c = c.loc[c.statpers.eq(c.statpers.iloc[-1])]
        if production:
            if c[["medest", "stdev", "numest", "curcode"]].drop_duplicates().shape[0] != 1:
                return None, None, "ambiguous_latest_consensus"
            if pd.isna(a.curr_act) or c.curcode.iloc[-1] != a.curr_act:
                return None, None, "actual_consensus_currency_unverified"
        a = a.copy()
        a["link_basis"] = basis
        return a, c.iloc[-1], ""

    def values(self, e, market, a, c, production):
        values = {DATE: a.pends, "actual_eps": a.value}
        why = {}
        if pd.notna(c.numest) and c.numest > 0:
            values["n_analysts"] = c.numest
        else:
            why["n_analysts"] = "missing_analyst_count"
        if pd.notna(c.stdev) and c.stdev >= 0 and c.medest != 0:
            values["dispersion"] = c.stdev / abs(c.medest)
        else:
            why["dispersion"] = "zero_consensus_or_missing_dispersion"
        cal = market.calendar if production else market.legacy_bench.index
        ia, ic = cal.searchsorted(a.anndats, side="right")-1, cal.searchsorted(c.statpers, side="right")-1
        p = market.sid.get_loc(e.permno)
        source = market.combined if production else market.legacy
        factor = source["factor"][p]
        k = np.nan
        reason = "split_factor_or_crsp_release_coverage_missing"
        if ia >= 0 and ic >= 0 and (not production or a.anndats <= cal[-1]):
            if production:
                f = factor[[ic, ia]]
                change_at_release = (cal[ia] == a.anndats and
                                     np.isfinite(source["factor_day"][p, ia]) and
                                     not np.isclose(source["factor_day"][p, ia], 1., atol=1e-7, rtol=0))
                if change_at_release:
                    reason = "same_release_day_corporate_action_clock_unverified"
                elif (np.isfinite(f) & (f > 0)).all():
                    k = f[1] / f[0]
                    # CIZ accumulated factors are rounded; verified daily factors preserve exact 1.
                    daily = source["factor_day"][p, ic+1:ia+1]
                    if np.isfinite(daily).all() and np.all(daily > 0):
                        exact = 1 / np.prod(daily)
                        if np.isclose(k, exact, atol=2e-6, rtol=2e-6):
                            k = exact
                        else:
                            k = np.nan
                            reason = "ciz_daily_and_cumulative_factors_disagree"
            else:
                before, _ = market.asof({"factor": factor}, "factor", ic, len(cal))
                after, _ = market.asof({"factor": factor}, "factor", ia, len(cal))
                k = after / before if pd.notna(before) and pd.notna(after) else 1.
        if not np.isfinite(k) or k <= 0:
            why.update({f: reason for f in ["consensus_eps", "eps_surprise", "sue",
                                           "surprise_pct", BOOL]})
            why["sue_price"] = ("pre_announcement_price_outside_crsp" if a.anndats > cal[-1] else reason)
            return values, why
        # The old WRDS pull stored numeric columns as float32 before the
        # post-join arithmetic. Replay that precision only on historical checks.
        am, cm, sd0 = a.value, c.medest, c.stdev if pd.notna(c.stdev) else np.nan
        if not production:
            am, cm, sd0 = map(float, np.float32([am, cm, sd0]))
        med, sd = cm * k, sd0 * k
        surprise = am - med
        if not production and med != 0 and pd.notna(sd):
            values["dispersion"] = sd / abs(med)
        values.update(consensus_eps=med, eps_surprise=surprise, split_adjusted_consensus=float(k != 1))
        if med != 0:
            values["surprise_pct"] = surprise / abs(med)
        else:
            why["surprise_pct"] = "zero_consensus"
        if pd.notna(sd) and sd > 0:
            values["sue"] = surprise / sd
        else:
            why["sue"] = "zero_or_missing_forecast_stdev"
        q = source["price"][p, ia-1] if ia > 0 else np.nan
        if np.isfinite(q) and q > 0 and (not production or cal[ia-1] < e.et_date):
            values["sue_price"] = surprise / q
        else:
            why["sue_price"] = "missing_or_ineligible_pre_release_price"
        return values, why


def measure(events, market, fund, earnings, production, mapped):
    event_calendar = market.event_calendar if production else market.legacy_bench.index
    day0 = locate_day0(events, event_calendar, legacy=True)
    by_event, audit = [], []
    for pos, e in enumerate(events.itertuples()):
        row = {}
        stamp = {"keydevid": int(e.keydevid), "transcriptid": int(e.transcriptid),
                 "call_date": dstr(e.call_date), "verified_mapping": bool(mapped[pos])}
        if not mapped[pos]:
            reason = dict.fromkeys(FEATURES, "unverified_event_security_link")
        else:
            i = day0[pos]
            day = event_calendar[i] if i >= 0 else None
            stamp["feature_day0"] = dstr(day)
            row, reason, clocks = market.window(e, i, production, market.combined if production else market.legacy)
            stamp.update(clocks)
            bm, why, rdq = fund.select(e, day, row.get("mcap_m", np.nan), production)
            if why:
                reason["bm"] = why
            else:
                row["bm"] = bm
                stamp["book_equity_rdq"] = dstr(rdq)
            actual, cons, why = earnings.match(e, production)
            if why:
                reason.update(dict.fromkeys(EPS, why))
            else:
                block, why = earnings.values(e, market, actual, cons, production)
                row.update(block)
                reason.update(why)
                stamp.update(earnings_release_date=dstr(actual.anndats),
                             earnings_release_clock_utc=str(actual.release) if pd.notna(actual.release) else "",
                             consensus_statpers=dstr(cons.statpers), ibes_link_basis=actual.link_basis)
                if production:
                    need(cons.statpers < actual.anndats
                         and (actual.anndats < e.et_date or actual.release <= e.clock),
                         "EPS timing violation")
            if production:
                if rdq is not None and "bm" in row:
                    need(rdq < day and rdq < e.et_date, "Compustat RDQ timing violation")
                for key in ("cap_observation", "momentum_last_session", "vol_pre_window_end"):
                    if key in stamp:
                        observed = pd.Timestamp(stamp[key])
                        same_day_closed = observed == e.et_date and e.clock.tz_convert(NY).hour >= 16
                        need(observed < e.et_date or same_day_closed, f"{key} observable after event")
        for name in FEATURES:
            v = row.get(name)
            if pd.notna(v):
                need(name == DATE or np.isfinite(v), f"Nonfinite calculated {name}")
                stamp[name + "_status"] = RECON
            else:
                need(name in reason, f"Unexpected missing {name}")
                stamp[name + "_status"] = reason[name]
        by_event.append(row)
        audit.append(stamp)
    return pd.DataFrame(by_event, index=events.index).reindex(columns=FEATURES), pd.DataFrame(audit, index=events.index)


def error_arrays(reference, values, feature):
    keep = reference.notna() & values.notna()
    if feature == DATE:
        error = (pd.to_datetime(reference[keep])-pd.to_datetime(values[keep])).dt.days.abs().to_numpy(dtype=float)
        good = error == 0
    else:
        expected, rebuilt = reference[keep].to_numpy(dtype=float), values[keep].to_numpy(dtype=float)
        error = abs(expected - rebuilt)
        atol = 5e-8 if feature not in {"n_analysts", BOOL} else 0
        rtol = 2e-7 if atol else 0
        good = np.isclose(expected, rebuilt, atol=atol, rtol=rtol)
    return keep, error, good


def validation(events, candidate):
    rows = []
    for year, events_y in events.groupby("year"):
        c = candidate.loc[events_y.index]
        for f in FEATURES:
            a = events_y[f]
            keep, err, good = error_arrays(a, c[f], f)
            rows.append(dict(feature=f, reference="original_legacy", year=year, total_events=len(a),
                             archived_observed=int(a.notna().sum()), reproduced_observed=int(c[f].notna().sum()),
                             overlap=int(keep.sum()), mean_abs_error=err.mean() if len(err) else np.nan,
                             median_abs_error=np.median(err) if len(err) else np.nan,
                             max_abs_error=err.max() if len(err) else np.nan,
                             agreements=int(good.sum()),
                             agreement_rate=good.mean() if len(good) else np.nan,
                             agreement_at_1e_5=(err <= 1e-5).mean() if len(err) else np.nan,
                             availability_disagreements=int(a.notna().ne(c[f].notna()).sum())))
    return pd.DataFrame(rows)


def stale_proof(e, market):
    future = e.loc[e.year.ge(2025)]
    cut = market.legacy_bench.index[-1]
    cap = market.legacy["cap"][0, market.position(cut)]
    expected = {}
    for month in future.call_date.dt.to_period("M").unique():
        day = month.start_time
        a = market.position(day-pd.DateOffset(months=12))
        b = min(len(market.legacy_bench), market.position(day-pd.DateOffset(months=1)))
        rr = market.legacy["ret"][0, a:b]
        keep = rr[np.isfinite(rr)]
        expected[month] = np.expm1(np.log1p(keep).sum()) if len(keep) >= 150 else np.nan
    mom = future.call_date.dt.to_period("M").map(expected)
    return {"cap_matches": int(np.isclose(future.mcap_m, cap, atol=1e-6, rtol=1e-9).sum()),
            "size_matches": int(np.isclose(future["size"], np.log(cap*1e6), atol=1e-8, rtol=1e-9).sum()),
            "mom_matches": int((np.isclose(future.mom, mom, atol=1e-8, rtol=1e-7)
                               & future.mom.notna()).sum()), "old_mom_observed": future.mom.count()}


def prepare():
    names, original, events = read_csv()
    inventory = {}
    market = Market(events, inventory)
    # All four historical inputs are present for these event cohorts and their windows.
    earlier = events.year.eq(2021) & events.call_date.lt("2021-11-01")
    historical = events.loc[earlier | events.year.eq(2024)]
    future = events.loc[events.year.ge(2025)]
    mapped, outside, reviewed = verify_mapping(pd.concat([historical, future], ignore_index=True))
    check = pd.Series(mapped, index=historical.index.append(future.index))
    fund = Fundamentals(RAW / "comp_fundq.parquet", inventory)
    ibes = Earnings(RAW, (2020, 2021, 2023, 2024), inventory)
    estimates, evidence = measure(historical, market, fund, ibes, False, check.loc[historical.index].to_numpy())
    return names, original, events, market, inventory, historical, future, check, outside, reviewed, estimates, evidence


def cell(value, feature):
    if pd.isna(value):
        return ""
    if feature == DATE:
        return dstr(value)
    if feature == BOOL:
        return "True" if value else "False"
    if feature == "n_analysts":
        return str(int(value))
    return format(float(value), ".17g")


def assemble(names, archived, events, proposed, audit, approved):
    result = [r.copy() for r in archived]
    col = {f: names.index(f) for f in FEATURES}
    for i, trial in proposed.iterrows():
        for f, c in col.items():
            if f in approved:
                result[i][c] = cell(trial[f], f)
            else:
                audit.at[i, f + "_status"] = ("archived_unchanged_validation_failed" if archived[i][c]
                                              else "unavailable_validation_failed")
            need((audit.at[i, f+"_status"] == RECON) == (f in approved and bool(result[i][c])),
                 f"Feature status mismatch for {f}")
    make_coverage = []
    for name, indices in [("sp500_all_sectors", events.index),
                          ("healthcare", events.index[events.sector.eq("Health Care")])]:
        for year, sample in events.loc[indices].groupby("year"):
            ids = sample.index
            recent = year >= 2025
            for f, c in col.items():
                orig = [archived[i][c] for i in ids]
                vals = [result[i][c] for i in ids]
                old_count, new_count = sum(map(bool, orig)), sum(map(bool, vals))
                keys = [float(v) if f not in [DATE, BOOL] else v for v in vals if v]
                freq = Counter(keys)
                mode, mode_n = freq.most_common(1)[0] if freq else ("", 0)
                status = Counter(audit.loc[ids, f+"_status"]) if recent else Counter()
                changed = sum(bool(a) and bool(b) and a != b and (
                    a != b if f in (DATE, BOOL) else not np.isclose(float(a), float(b), atol=5e-8, rtol=2e-7))
                    for a, b in zip(orig, vals))
                missing_reasons = "|".join(f"{k}={n}" for k, n in sorted(status.items())
                                           if k != RECON and k != "archived_unchanged_validation_failed")
                make_coverage.append({
                    "scope": name, "year": year, "feature": f, "events": len(ids),
                    "archived_nonmissing": old_count, "nonmissing": new_count,
                    "nonmissing_pct": 100 * new_count / len(ids), "distinct": len(freq),
                    "most_frequent_value": mode, "most_frequent_n": mode_n,
                    "reconstructed": status[RECON], "corrected_nonmissing": changed,
                    "newly_nonmissing": sum(not a and bool(b) for a, b in zip(orig, vals)),
                    "cleared_archived_to_missing": sum(bool(a) and not b for a, b in zip(orig, vals)),
                    "unresolved_missing": len(ids)-new_count, "missing_reasons": missing_reasons,
                })
    return result, pd.DataFrame(make_coverage)


def save_model(path, names, old, output, events, approved):
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(output)
    approved_positions = {names.index(f) for f in approved}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        need(next(reader) == names, "A model-table column changed")
        for i, saved in enumerate(old):
            new = next(reader, None)
            need(new == output[i], f"Output roundtrip mismatch at row {i}")
            allow = approved_positions if events.year.iloc[i] >= 2025 else set()
            need(all(c in allow or saved[c] == new[c] for c in range(len(names))),
                 f"An archived protected value changed at row {i}")
        need(next(reader, None) is None, "Output added records")


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join(["---"]*len(head)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def readiness(events, output, names):
    field = {name: names.index(name) for name in [*FEATURES, "car_2_20"]}
    group = events.loc[events.year.ge(2025) & events.sector.eq("Health Care")]
    rows = []
    for year, d in group.groupby("year"):
        def complete(features, ids):
            return sum(all(output[i][field[f]] for f in features) for i in ids)
        label = [i for i in d.index if output[i][field["car_2_20"]]]
        rows.append([year, len(d), len(label),
                     complete(["mcap_m", "size", "mom", "vol_pre"], label),
                     complete(["sue", "size", "bm", "mom", "vol_pre", "n_analysts", "dispersion"], label)])
    return rows


def verify_inventory(inventory):
    """Check the local cache against the tracked description without contacting WRDS."""
    args = ["git", "--no-optional-locks", "-c", f"safe.directory={ROOT.as_posix()}"]
    path = "06273f1:data/raw/sp500_pit/manifest.csv"
    x = subprocess.check_output([*args, "show", path], cwd=ROOT, encoding="utf-8")
    manifest = {row["file"]: int(row["rows"]) for row in csv.DictReader(StringIO(x))}
    common = ["crsp_msp500list_v2", "crsp_ccmxpf_lnkhist", "ciq_wrds_gvkey",
              "crsp_stksecurityinfohist", "comp_co_hgic", "wrdsapps_ibcrsphist"]
    for name in common:
        path = TEAM / (name + ".parquet")
        d = pd.read_parquet(path)
        need(len(d) == manifest[name], f"Local row count differs from manifest for {name}")
        fields = {"crsp_msp500list_v2": ("mbrstartdt", "mbrenddt"),
                  "crsp_ccmxpf_lnkhist": ("linkdt", "linkenddt"),
                  "ciq_wrds_gvkey": ("startdate", "enddate"),
                  "crsp_stksecurityinfohist": ("secinfostartdt", "secinfoenddt"),
                  "comp_co_hgic": ("indfrom", "indthru"),
                  "wrdsapps_ibcrsphist": ("sdate", "edate")}
        lo, hi = fields[name]
        inventory[name + " (nonempty spell dates)"] = (
            pd.to_datetime(d[lo]).min(), pd.to_datetime(d[hi]).max(), len(d))
    for name, label, field in [("ibes_act_epsus", "adjusted actual ANNDATS", "anndats"),
                                ("ibes_statsum_epsus_2025", "adjusted 2025 STATPERS", "statpers"),
                                ("ibes_statsum_epsus_2026", "adjusted 2026 STATPERS", "statpers")]:
        d = pd.read_parquet(TEAM / (name + ".parquet"), columns=[field])
        need(len(d) == manifest[name], f"Local row count differs from manifest for {name}")
        values = pd.to_datetime(d[field])
        inventory[label + " (inventory only)"] = (values.min(), values.max(), len(d))
    for name, recorded in [
        *( (f"crsp_dsf_v2_{y}", f"crsp_dsf_v2_{y}.parquet") for y in (2024, 2025, 2026)),
        ("crsp_dsp500_v2", "crsp_dsp500_v2.parquet SPRTRN"),
        ("ff_factors_daily", "ff_factors_daily.parquet (calendar dates only)"),
        ("comp_fundq", "sp500_pit/comp_fundq.parquet datadate"),
        ("ibes_actu_epsus", "sp500_pit unadjusted actual ANNDATS"),
        *((f"ibes_statsumu_epsus_{y}", f"sp500_pit/{y} unadjusted consensus STATPERS") for y in (2024, 2025, 2026)),
    ]:
        need(inventory[recorded][2] == manifest[name], f"Local row count differs from manifest for {name}")


def docs(out, events, inventory, coverage, validations, approved, stain, readiness_rows,
         digest, mapped, outside, reviewed, audit):
    input_table = table(["Source", "First date", "Last date", "Rows"], [
        [k, dstr(a), dstr(b), f"{n:,}"] for k, (a, b, n) in inventory.items()])
    eps_rows = [
        ["fiscal_period_end, actual_eps", "Matching I/B/E/S QTR PENDS and unadjusted actual VALUE"],
        ["consensus_eps", "Unadjusted median on the actual announcement share basis, MED_EST × k"],
        ["eps_surprise", "Actual EPS minus share-basis consensus"],
        ["sue", "Surprise / (STDEV × k), only when the adjusted standard deviation is positive"],
        ["sue_price", "Surprise / absolute unadjusted price on the trading session preceding the index at or before ANNDATS"],
        ["surprise_pct", "Surprise / abs(adjusted consensus), provided consensus is nonzero"],
        ["n_analysts", "NUMEST from the same eligible forecast snapshot"],
        ["dispersion", "(STDEV × k) / abs(MED_EST × k), identical to the unadjusted ratio for positive k"],
        ["split_adjusted_consensus", "Whether the verified split factor k differs from one"],
    ]
    withheld = [f for f in FEATURES if f not in approved]
    bridge_n = audit.ibes_link_basis.eq("continuous_cusip_across_ibes_link_extract_end").sum()
    method = f"""# S&P 500 financial-feature reconstruction

The production CSV retains its input header, all {len(events):,} archive events, row order, event identifiers, linguistic columns, CAR values, CAR provenance and all 2010–2024 values. Only the approved 2025–2026 financial fields are replaced. The SHA-256 of the input CAR extension table is `{digest}`.

## Traced historical definitions

The archived CSV uses Git `2a3a90a:sp500_pipeline.py`, particularly `stage_targets()` at lines 286–445 and source queries at 175–197. The financial helpers in `src/linguistic_alpha/controls/firm.py:36–49,61–90` and `controls/earnings.py:115–131` are later alternatives. In particular, the former uses a CEQQ fallback book-equity method, and the latter puts price-scaled SUE in the field named `sue`. Neither naming choice is adopted here.

| Output | Source and convention |
| --- | --- |
| `mcap_m`, `size` | Lines 325, 380–386. Use `abs(PRC) × SHROUT / 1000` in millions of dollars on the last market day before the archive call month. On that exact PERMNO only, allow up to 10 market sessions since the last observed capitalization. Use `ln(mcap_m × 1e6)` for size. CIZ `dlyprc` and `shrout` preserve the unit conversion. |
| `mom` | Lines 337–338, 387–394. Compound observed total stock returns from the start of call month minus 12 months to strictly before the start of call month minus one month. Require at least 150 observed returns and a fully bounded calendar window, and exclude months without a verified endpoint. |
| `vol_pre`, `vol_post_2_20` | Lines 340–352, 371–378. Sample standard deviation with ddof=1 of daily stock return minus S&P 500 `SPRTRN`, times √252. Inclusive session windows [−60,−5] and [+2,+20] with at least 40 and 15 paired observations. The full calendar window must exist. **Post-event volatility is a realized outcome.** |
| `bm` | Lines 396–404. Positive Compustat quarterly `CEQQ` divided by `mcap_m`. Latest RDQ strictly before Day 0, within 200 calendar days. Production also requires RDQ before the New York call date and DATADATE ≤ RDQ, and declines conflicting records for the latest RDQ. There is no DATADATE-based availability fallback. |

Day 0 for these reconstructions uses the original UTC archive date and 16:00 Eastern clock from lines 340–352. Timing uses original event-specific calendar positions and retains the previously published Day 0 string as-is. A post-cutoff event never falls back to the first stock row or to a calendar-edge price. The stock side uses legacy daily `RET` through 2024 and CIZ daily `dlyret` thereafter. Legacy same-session `DLRET` is compounded with regular return, or used alone when that is the observed return. CIZ already includes the delisting return. The `SPRTRN` price-index benchmark also switches to CIZ from 2025. Missing prices and returns remain missing except the specified same-security market-cap carry-forward.

The local `ff_factors_daily.parquet` date index matches every CIZ index session in the common range. Its dates extend Day 0 anchoring through 2026-09-30 for features whose underlying inputs already exist, such as some July pre-windows and book-to-market. The file supplies session dates only. There is no Fama–French return substitution, and the stock and benchmark windows still end at the actual CIZ return cutoff of 2026-06-30.

### Earnings surprise

The original pulls unadjusted `actu_epsus` and `statsumu_epsus` (lines 189–197). The dated PERMNO-to-I/B/E/S link selects the lowest score (265–270). Lines 406–443 match the nearest quarterly actual within ±5 calendar days of the archived call date, then the latest FPI=6 forecast snapshot strictly before ANNDATS, at most 120 days before it, for the same ticker and fiscal period. This run additionally verifies the actual predates the call or has an observed non-midnight same-day announcement time at or before the call in Eastern time. A later release is not used. Conflicting same-key observations, mixed currencies and a fiscal period-end after release are rejected.

{table(["Field", "Definition"], eps_rows)}

`k` is the announcement-session CRSP price-event factor divided by the consensus-session factor, following the original `CFACPR` ratio. During production the two CIZ `dlycumfacpr` endpoints use the same CIZ reference share basis. Where daily `dlyfacprc` is complete, its inverse product across the interval must agree with the endpoint ratio. A same-date non-unit event factor is left unresolved if the split timing relative to release cannot be established. Validation replays the original float32 source precision for EPS values used in arithmetic. Production keeps raw numeric precision. The adjusted I/B/E/S tables are inventoried and never used for the surprise calculation.

The local I/B/E/S–CRSP link extract has a common terminal date of 2025-12-31. For a call beyond that boundary, the pipeline requires the prior link to end exactly at the extract boundary and an independently observed CRSP security-history spell with the **same eight-character CUSIP** spanning the boundary and the call. The archived date-valid PERMNO–GVKEY and exact CIQ transcript-clock links must also pass. A nonempty archived I/B/E/S ticker must agree. The sidecar records the resulting bridge for {bridge_n:,} matched EPS blocks. This does not verify subsequent changes after the CUSIP data end. No security link is extended beyond its independent coverage.

## Sources and localized defects

Actual Parquet contents were checked against the directory and the tracked README and manifest at Git `06273f1:data/raw/sp500_pit/`. The corresponding local team copies of those metadata files are absent, so only selected tracked manifest entries were checked against the local Parquet row counts. The adjusted snapshots below are inventory only.

{input_table}

The linkage audit matched {int(mapped.sum()):,} of {len(mapped):,} attempted historical-validation and production event mappings. It uses exact CIQ transcript clocks and dated CIQ-to-GVKEY / CCM links. CIZ membership is checked against real spells, with {outside:,} of {reviewed:,} archive events through the CIZ limit outside their saved membership spell. The user-supplied archived universe defines the output, including those rows.

The cutoff-session capitalisation and size for the archive's first sorted security match all {stain['cap_matches']:,} archived future cap rows and {stain['size_matches']:,} size rows. Lines 348–353 mapped invalid late events to stock row zero, and lines 382–394 used that row for cap and momentum even though their guard only checked whether the original PERMNO was valid. The end-of-calendar clip froze cap at the last 2024 date. The first-stock momentum readback matches {stain['mom_matches']:,} of {stain['old_mom_observed']:,} nonmissing archived entries. Post-2024 volatility was empty because Day 0 was outside the old calendar. The old EPS-price denominator also clipped at the old calendar edge, so existing post-2024 price-scaled SUE is replaced from dated prices instead of copied.

## Validation and output

The historical comparisons use the 2021 events through October and all 2024 archive calls, with original dates and source definitions. The earlier slice keeps validation windows in the available legacy stock files. `feature_validation.csv` reports overlap, availability, mean, median, maximum absolute error and tolerance pass rates. The production checks separately enforce book equity and earnings cutoffs. Approved features: {', '.join(sorted(approved))}. Unapproved fields: {', '.join(withheld) if withheld else 'none'}.

The local `feature_provenance.csv` has one row per 2025–2026 archive event, with dates and per-field status. `reconstructed` means sourced, including genuine unchanged values. Other status strings give the unavailability reason or identify a validation-gated field kept unchanged. The model table adds no provenance columns. The tracked `feature_coverage_aggregate.csv` includes original and final coverage, distinct counts, modal frequencies, changed and cleared archived values, new values and unresolved reasons for the full universe and Healthcare subset. Exact modal values are present only in the local `feature_coverage.csv`. Historical coverage rows are based on preserved cells.

Replay from the root with Python, pandas, NumPy and PyArrow. The existing `outputs/sp500_car_extension/extend_car.py` supplies the event mapping and Day 0 helpers. The historical input and existing outputs are kept read-only. Run with a fresh destination.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python outputs/sp500_financial_extension/reconstruct_financials.py --out outputs/sp500_financial_extension/reproduction
```

No models, feature selection, performance metrics or strategies were evaluated. `vol_post_2_20`, `car_0_1` and `car_-1_1` must remain outside call-time feature lists. Static vendor snapshots do not establish original release vintages. Transcript delivery is a separate gate for future models.
"""
    (out / "feature_lineage.md").write_text(method, encoding="utf-8")
    counts = coverage.loc[coverage.year.ge(2025)].set_index(["scope", "year", "feature"])
    rows = [[f, *[f"{counts.loc[(scope,y,f),'reconstructed']:,}"
                   for scope in ("sp500_all_sectors", "healthcare") for y in (2025, 2026)]]
            for f in FEATURES]
    v = validations.loc[validations.year.eq(2024)]
    vr = [[d.feature, d.overlap, f"{d.mean_abs_error:.3g}", f"{d.median_abs_error:.3g}",
           f"{d.max_abs_error:.3g}", f"{d.agreement_rate:.2%}"] for d in v.itertuples()]
    zh = f"""# 标普 500 财务特征重建

保留 {len(events):,} 场电话会及全部原始列。2010–2024、语言特征、CAR 及其来源字段逐单元格核对。重建批准 {len(approved)}/{len(FEATURES)} 个字段，仅替换 2025–2026 财务字段。未批准字段为 {', '.join(withheld) if withheld else '无'}。

## 数据与原口径

CRSP CIZ 个股收益、价格、流通股数及标普 500 `SPRTRN` 覆盖到 2026-06-30。本地 Fama–French 文件仅提供经重合期全量验证的交易日期，锚定到 2026-09-30，不补收益。Compustat 季度期末最晚为 2026-09-30，库内最晚 RDQ 为 2026-11-06。未调整 I/B/E/S 实际公告和 2026 年共识量均覆盖到 2026-08-20。I/B/E/S 链接原表截至 2025-12-31，部分 2026 记录可由连续相同 CUSIP 和有效 CCM / CIQ 键复核。超过独立身份覆盖的记录不外推。

按原始 `sp500_pipeline.py` 取同证券月末市值，计算 `size = ln(mcap_m × 1e6)`。动量按原 12 至 1 月区间累计，前后波动率是标普 500 调整日收益的样本标准差乘 √252。B/M 是公告日期在截止前的正 `CEQQ / mcap_m`。EPS 使用未调整实际值和同财期公告前最后共识，按 CRSP 拆股因子统一股本口径。原 `sue` 为惊喜除共识标准差，`sue_price` 为惊喜除公告前价格。

已确认重复市值和规模来自旧脚本越过行情尾部后误用的首个 PERMNO。首行回放也解释了全部 {stain['mom_matches']:,} 个旧非空动量。新表使用对应证券的原始记录，不能验证的留空。

## 2024 历史复验

财期差以天计，其余按字段原单位。更早的 2021 复验、验证阈值及覆盖差异保存在 CSV。

{table(["字段", "可比较事件", "平均绝对误差", "绝对误差中位数", "最大绝对误差", "一致率"], vr)}

## 逐年重建数量

{table(["字段", "全表 2025", "全表 2026", "医疗保健 2025", "医疗保健 2026"], rows)}

## Healthcare 样本外准备度

{table(["年度", "事件", "CAR[+2,+20] 可用", "其中四个价格控制齐全", "其中七个财务控制齐全"], readiness_rows)}

四个价格控制为市值、规模、动量和事前波动率。七个财务控制为 SUE、规模、B/M、动量、事前波动率、分析师数和分歧度。完整性只用于数据诊断，未训练或比较模型。未覆盖的行情窗口、不明公告时点、零标准差及无法验证的链接均有标记。FinBERT/LM 的实际到达时点及未来样本的资格还需要单独核验。`vol_post_2_20` 与两个事件反应 CAR 不能作通话时预测特征。

[汇总覆盖](feature_coverage_aggregate.csv) · [历史复验](feature_validation.csv) · [字段公式与复跑](feature_lineage.md)
"""
    (out / "summary_zh.md").write_text(zh, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    need(not any((out / n).exists() for n in [*OUTPUTS, AGGREGATE_COVERAGE]), "Run with a fresh destination")
    digest = sha256(INPUT.read_bytes()).hexdigest()
    names, old, events, market, inventory, hist, future, mapped, outside, reviewed, estimate, evidence = prepare()
    v = validation(hist, estimate)
    # Each field must independently reproduce both local historical cohorts.
    passed = v.overlap.ge(1000) & v.agreement_rate.ge(.999) & v.availability_disagreements.eq(0)
    v["cohort_passed"] = passed
    approved = {name for name, d in v.groupby("feature") if d.cohort_passed.all()}
    v["approved_for_reconstruction"] = v.feature.isin(approved)
    v["abs_tolerance"] = np.where(v.feature.isin([DATE, BOOL, "n_analysts"]), 0, 5e-8)
    v["relative_tolerance"] = np.where(v.feature.isin([DATE, BOOL, "n_analysts"]), 0, 2e-7)
    print(v[["feature", "year", "overlap", "max_abs_error", "agreement_rate", "cohort_passed"]].to_string(index=False),
          flush=True)
    fund = Fundamentals(TEAM / "comp_fundq.parquet", inventory)
    ibes = Earnings(TEAM, (2024, 2025, 2026), inventory)
    verify_inventory(inventory)
    reconstructed, audit = measure(future, market, fund, ibes, True, mapped.loc[future.index].to_numpy())
    output, coverage = assemble(names, old, events, reconstructed, audit, approved)
    sample_readiness = readiness(events, output, names)
    proof = stale_proof(events, market)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".financial-publish-", dir=out) as tmp:
        tmp = Path(tmp)
        save_model(tmp / OUTPUTS[0], names, old, output, events, approved)
        v.to_csv(tmp / OUTPUTS[1], index=False, float_format="%.12g")
        coverage.to_csv(tmp / OUTPUTS[2], index=False, float_format="%.12g")
        coverage.drop(columns="most_frequent_value").to_csv(
            tmp / AGGREGATE_COVERAGE, index=False, float_format="%.12g")
        audit.to_csv(tmp / OUTPUTS[3], index=False)
        docs(tmp, events, inventory, coverage, v, approved, proof, sample_readiness,
             digest, mapped, outside, reviewed, audit)
        need(sha256(INPUT.read_bytes()).hexdigest() == digest, "Source CAR table changed")
        for name in [*OUTPUTS[1:], AGGREGATE_COVERAGE, OUTPUTS[0]]:
            (tmp / name).rename(out / name)
    print(f"Published {len(events):,} rows, {len(approved)} validated features and {len(audit):,} event audits.",
          flush=True)
    print(f"Healthcare readiness: {sample_readiness}", flush=True)


if __name__ == "__main__":
    main()
