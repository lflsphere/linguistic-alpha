"""Firm characteristics: size, book-to-market, momentum, pre-event volatility, CAR[-1,+1] and CAR[0,+1].

Definitions (sessions = benchmark trading days, ``crsp_dsp500_v2.caldt``):
  size    = ln(market cap in $) at the last session of the month before Day 0's month. Market cap
            is the firm's (``permco``) total ``dlycap`` over its share classes, so GOOGL and
            GOOG both get Alphabet's size; missing when the event's permno has no cap that day.
  bm      = book equity / the same market cap, book equity from the latest fundq available
            before Day 0 (rdq < Day 0; NULL rdq -> datadate + 90 days) with datadate at most a
            year before Day 0; BE = seqq + txditcq - pstkq (seqq missing -> ceqq + txditcq;
            missing txditcq, pstkq -> 0); BE <= 0 -> NaN
  mom     = prod(1 + ret) - 1 over months -12..-2 relative to Day 0's month (sessions from the
            first of month -12 to the last of month -2); missing returns are skipped, never
            filled with 0, and mom is missing below ``targets.min_window_coverage`` valid days
  vol_pre = std(AR, -60..-5) * sqrt(252), ``targets.vol_pre_window`` (same coverage rule)
  price_m1 = |dlyprc| at Day -1 (scales sue, guidance and the forecast range)
  car_m1_1 = CAR over Day -1 .. Day +1 (``targets.reaction_window``; control, not target).
             Day -1 is the last close before the release, so the window holds the whole reaction
             whether the release came before the open or after the close.
  car_0_1  = CAR over Day 0 .. Day +1, the comparison window. Both always computed; Model A
             uses the one named by the leg (``controls.reaction_legs``).
  Reaction windows use ``targets.car.abnormal_returns`` / ``event_car`` (agent E's code: stock
  and S&P 500 total return compounded separately, then differenced). Any null day in a
  reaction window sets it to missing (docs/pipeline.md, Missing daily returns).

Fixes vs. the legacy notebook (docs/data_audit.md §Controls): missing returns inside the
momentum window are not filled with 0; fundq rows with NULL rdq use datadate + 90 days lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from linguistic_alpha.targets.car import event_car, post_call_volatility

RDQ_FALLBACK_DAYS = 90
MAX_FUNDQ_AGE_DAYS = 366
_EPS = 1e-9


class _Daily:
    """Session x permno matrices of return, |price| and firm (permco) market cap."""

    def __init__(self, dsf: pd.DataFrame, sessions: pd.DatetimeIndex):
        d = pd.DataFrame({
            "permno": dsf["permno"].astype("int64").to_numpy(),
            "permco": pd.to_numeric(dsf["permco"], errors="coerce").to_numpy(float)
            if "permco" in dsf.columns else dsf["permno"].astype("int64").to_numpy(float),
            "date": pd.to_datetime(dsf["dlycaldt"]).to_numpy(),
            "ret": pd.to_numeric(dsf["dlyret"], errors="coerce").to_numpy(float),
            "prc": np.abs(pd.to_numeric(dsf["dlyprc"], errors="coerce").to_numpy(float)),
            "cap": pd.to_numeric(dsf["dlycap"], errors="coerce").to_numpy(float),
        }).drop_duplicates(["permno", "date"])
        d["firm_cap"] = d.groupby(["permco", "date"])["cap"].transform("sum", min_count=1)
        d.loc[d["cap"].isna(), "firm_cap"] = np.nan
        self.sessions = sessions
        self.permnos = pd.Index(pd.unique(d["permno"]))
        shape = (len(sessions), len(self.permnos))
        ri = sessions.get_indexer(d["date"])
        ci = self.permnos.get_indexer(d["permno"])
        ok = ri >= 0
        self.R, self.P, self.C = (np.full(shape, np.nan) for _ in range(3))
        for M, c in ((self.R, "ret"), (self.P, "prc"), (self.C, "firm_cap")):
            M[ri[ok], ci[ok]] = d[c].to_numpy(float)[ok]

    def at(self, M: np.ndarray, col: np.ndarray, pos: np.ndarray) -> np.ndarray:
        ok = (col >= 0) & (pos >= 0) & (pos < len(self.sessions))
        out = np.full(len(col), np.nan)
        out[ok] = M[pos[ok], col[ok]]
        return out


def _month_start(d: pd.Series) -> pd.Series:
    return d.dt.to_period("M").dt.to_timestamp()


def _book_equity(fundq: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"gvkey": fundq["gvkey"].astype("string").to_numpy(),
                      "datadate": pd.to_datetime(fundq["datadate"], errors="coerce").to_numpy(),
                      "rdq": pd.to_datetime(fundq["rdq"], errors="coerce").to_numpy()})
    num = {c: pd.to_numeric(fundq[c], errors="coerce").to_numpy(float)
           for c in ("seqq", "ceqq", "txditcq", "pstkq") if c in fundq.columns}
    txd = np.nan_to_num(num.get("txditcq", np.zeros(len(f))))
    pst = np.nan_to_num(num.get("pstkq", np.zeros(len(f))))
    seq = num.get("seqq", np.full(len(f), np.nan))
    ceq = num.get("ceqq", np.full(len(f), np.nan))
    be = np.where(np.isnan(seq), ceq + txd, seq + txd - pst)
    f["be"] = np.where(be > 0, be, np.nan)
    f["avail"] = f["rdq"].fillna(f["datadate"] + pd.Timedelta(days=RDQ_FALLBACK_DAYS))
    return f.dropna(subset=["gvkey", "datadate", "avail"])


def book_to_market(events: pd.DataFrame, fundq: pd.DataFrame, me_thousands: np.ndarray):
    """bm per event and the fundq availability date used (latest with avail < Day 0)."""
    f = _book_equity(fundq).sort_values(["avail", "datadate"])
    e = pd.DataFrame({"_i": np.arange(len(events)),
                      "gvkey": events["gvkey"].astype("string").to_numpy(),
                      "day0": pd.to_datetime(events["day0"]).to_numpy()})
    e["_key"] = e["day0"] - pd.Timedelta(1, "ns")           # strictly before Day 0
    e = e.dropna(subset=["gvkey"]).sort_values("_key")
    m = pd.merge_asof(e, f, left_on="_key", right_on="avail", by="gvkey", direction="backward")
    stale = (m["day0"] - m["datadate"]).dt.days > MAX_FUNDQ_AGE_DAYS
    m.loc[stale, ["be", "avail"]] = np.nan
    m = m.set_index("_i").reindex(np.arange(len(events)))
    me_mil = me_thousands / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = m["be"].to_numpy(float) / np.where(me_mil > 0, me_mil, np.nan)
    return bm, pd.to_datetime(m["avail"]).to_numpy()


def firm_controls(events: pd.DataFrame, dsf: pd.DataFrame, fundq: pd.DataFrame,
                  ar: pd.DataFrame, bench: pd.Series, tcfg: dict,
                  mom_months: list[int] = (-12, -2)) -> pd.DataFrame:
    """One row per event: size, bm, mom, vol_pre, price_m1, car_m1_1, car_0_1 and the dates
    of their latest inputs (``src_*``, for the look-ahead check).

    ``events``: keydevid, permno, gvkey, day0. ``dsf``: crsp_dsf_v2 (permno, permco, dlycaldt,
    dlyret, dlyprc, dlycap). ``ar``: ``targets.car.abnormal_returns``. ``bench``: benchmark
    return series by session (``targets.car.benchmark_series``). ``tcfg``: the ``targets`` config.
    """
    sessions = pd.DatetimeIndex(bench.index)
    daily = _Daily(dsf, sessions)
    day0 = pd.to_datetime(events["day0"]).reset_index(drop=True)
    pos0 = sessions.searchsorted(day0.to_numpy(), side="left")
    col = daily.permnos.get_indexer(events["permno"].astype("int64"))
    cov = tcfg.get("min_window_coverage", 0.8)

    def sess(pos):
        p = np.clip(pos, 0, len(sessions) - 1)
        return np.where((pos >= 0) & (pos < len(sessions)), sessions[p].to_numpy(),
                        np.datetime64("NaT", "ns"))

    # Day -1 price
    price_m1 = daily.at(daily.P, col, pos0 - 1)

    # size and the market cap for B/M: last session before Day 0's month
    m0 = _month_start(day0)
    pos_me = sessions.searchsorted(m0.to_numpy(), side="left") - 1
    me = daily.at(daily.C, col, pos_me)
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.log(np.where(me > 0, me * 1000.0, np.nan))
    bm, fq_date = book_to_market(events.reset_index(drop=True), fundq, me)

    # momentum: months mom_months[0] .. mom_months[1] relative to Day 0's month
    a, b = mom_months
    lo = sessions.searchsorted((m0 + pd.DateOffset(months=a)).to_numpy(), side="left")
    hi = sessions.searchsorted((m0 + pd.DateOffset(months=b + 1)).to_numpy(), side="left") - 1
    R = daily.R
    logr = np.where(np.isnan(R), 0.0, np.log1p(np.where(np.isnan(R), 0.0, R)))
    cs = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(logr, 0)])
    cn = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(~np.isnan(R), 0)])
    ok = (col >= 0) & (lo >= 0) & (hi >= lo) & (hi < len(sessions))
    c = np.clip(col, 0, None)
    lo_c, hi_c = np.clip(lo, 0, len(sessions) - 1), np.clip(hi, 0, len(sessions) - 1)
    n_valid = cn[hi_c + 1, c] - cn[lo_c, c]
    length = hi_c - lo_c + 1
    mom = np.exp(cs[hi_c + 1, c] - cs[lo_c, c]) - 1.0
    mom = np.where(ok & (n_valid >= cov * length - _EPS), mom, np.nan)

    ev = pd.DataFrame({"keydevid": events["keydevid"].astype("int64").to_numpy(),
                       "permno": events["permno"].astype("int64").to_numpy(),
                       "day0": day0.to_numpy()})
    pre = list(tcfg["vol_pre_window"])
    vol = post_call_volatility(ev, ar, pre, pre, cov, bench=bench)
    vol_pre = vol.iloc[:, 1].to_numpy(float)
    # reaction windows: any null day -> missing (coverage 1)
    cars = event_car(ev, ar, dict(tcfg["reaction_window"]), 1.0, bench=bench)

    out = pd.DataFrame({
        "keydevid": ev["keydevid"],
        "size": size, "bm": bm, "mom": mom, "vol_pre": vol_pre, "price_m1": price_m1,
        "mktcap_firm_m": me / 1000.0,
    })
    for name in tcfg["reaction_window"]:
        out[name] = cars[name].to_numpy(float)
    rw_end = max(w[1] for w in tcfg["reaction_window"].values())
    out["src_price_date"] = sess(pos0 - 1)
    out["src_size_date"] = sess(pos_me)
    out["src_fundq_avail"] = fq_date
    out["src_mom_end"] = np.where(ok, sess(hi), np.datetime64("NaT", "ns"))
    out["src_vol_pre_end"] = sess(pos0 + pre[1])
    out["src_reaction_end"] = sess(pos0 + rw_end)
    out["reaction_end_rel"] = rw_end
    return out
