"""Earnings surprise, analyst controls and release metrics from I/B/E/S.

Definitions:
  consensus  = last ``medest`` with statpers < anndats, fpedats == pends, fpi 6 (unadjusted
               ``statsumu_epsus``). The actual is the ``actu_epsus`` row the event's release
               was taken from (same ticker, pends and anndats).
  Share basis: unadjusted I/B/E/S values are on the share basis of their own date. The
               cumulative split factor ``ibes_adj.adj`` (in force from ``spdates``) converts
               them: value on today's basis = value / adj(date). The consensus is put on the
               actual's basis with adj(anndats) / adj(statpers), so a split between the two
               dates does not create a surprise (docs/data_audit.md, SUE row 1). CRSP share
               factors are never applied to EPS.
  sue_ts     = SUE_TS (primary, ``controls.sue_scaler: ts``): surprise / SD of the firm's prior
               surprises, previous 8 quarters (at least 4), shifted so the current surprise is
               not in its own scale. Surprises are compared on today's share basis, so the scale
               is split-invariant. 2008-10 releases feed the scale only. Same definition as
               ``scripts/common.SUE_VARIANT``.
  sue        = surprise / price at Day -1 (scope doc; robustness)
  sue_std    = surprise / analyst stdev (legacy; NaN when stdev 0)
  dispersion = stdev / |consensus| (NaN when consensus 0); n_analysts = numest

Release metrics (``controls.release_metrics``, docs/pipeline.md, Release metrics):
  rev_sue               (SAL actual - SAL consensus) / |SAL consensus|; adjusted ``act_xepsus``
                        and ``statsum_xepsus`` (amounts are not split-affected)
  guidance_given        1 when any guidance row (any measure) is dated within +-1 day of the
                        release and on or before Day 0
  guidance_vs_consensus (EPS guidance midpoint - mean_at_date) / price at Day -1 (missing when
                        above ``MAX_GUIDANCE_GAP`` in absolute value: a bad mean_at_date)
  guidance_width        (val_2 - val_1) / price at Day -1; point guidance -> 0
  guidance_change       sign of the EPS guidance midpoint minus the firm's previous guidance for
                        the same period (before the release window): +1 raised, 0 kept, -1 cut
  net_revisions         (numup - numdown) / numest of the consensus row
  forecast_range        (highest - lowest) / price at Day -1
  EPS guidance row: the nearest fiscal period ending after the reported quarter (quarterly before
  annual on ties), latest announcement in the window. ``det_guidance`` values are on today's
  share basis, so they are multiplied by adj(guidance date) before dividing by the price.
  ``flag_and_zero`` (``controls.release_missing``) is applied in :func:`flag_and_zero`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUE_HIST = (8, 4)          # (window, min periods) of the SUE_TS scale
GUIDANCE_DAYS = 1          # guidance within +-1 calendar day of the release
CHANGE_TOL = 0.005         # guidance midpoints within half a cent count as kept
MAX_GUIDANCE_GAP = 0.5     # |guidance - mean_at_date| / price above this -> mean_at_date bad


def _date(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), errors="coerce").astype("datetime64[ns]")


def _num(s) -> np.ndarray:
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(float)


def split_factor(adj: pd.DataFrame, tickers, dates) -> np.ndarray:
    """Cumulative I/B/E/S split factor in force on each date (last ``spdates`` <= date).

    Dates before a ticker's first row take its first factor; tickers with no row take 1.
    """
    q = pd.DataFrame({"ticker": pd.Series(tickers).astype("string").to_numpy(),
                      "date": _date(dates).to_numpy(), "_i": np.arange(len(tickers))})
    out = np.ones(len(q))
    if adj is None or not len(adj):
        return out
    a = pd.DataFrame({"ticker": adj["ticker"].astype("string"), "spdates": _date(adj["spdates"]),
                      "adj": _num(adj["adj"])}).dropna(subset=["spdates", "adj"])
    a = a[a["adj"] > 0].sort_values("spdates")
    q["ticker"] = q["ticker"].astype("string")
    ok = q["date"].notna() & q["ticker"].notna()
    qq = q[ok].sort_values("date")
    m = pd.merge_asof(qq, a, left_on="date", right_on="spdates", by="ticker",
                      direction="backward")
    first = a.groupby("ticker")["adj"].first()
    fill = m["ticker"].map(first)
    vals = m["adj"].fillna(fill).fillna(1.0).to_numpy(float)
    out[m["_i"].to_numpy()] = vals
    return out


def _consensus(actuals: pd.DataFrame, summary: pd.DataFrame, measure: str | None = None,
               cols=("medest", "stdev", "numest", "numup", "numdown", "highest",
                     "lowest")) -> pd.DataFrame:
    """For each actual row (ticker, pends, anndats): the last fpi-6 summary with
    fpedats == pends and statpers < anndats."""
    s = summary
    if measure is not None and "measure" in s.columns:
        s = s[s["measure"] == measure]
    if "fpi" in s.columns:
        s = s[s["fpi"].astype("string") == "6"]
    s = pd.DataFrame({"ticker": s["ticker"].astype("string"), "fpedats": _date(s["fpedats"]),
                      "statpers": _date(s["statpers"]),
                      **{c: _num(s[c]) for c in cols if c in s.columns}})
    s = s.dropna(subset=["ticker", "fpedats", "statpers", "medest"])
    s = s.sort_values("statpers").drop_duplicates(["ticker", "fpedats", "statpers"], keep="last")
    a = actuals[["ticker", "pends", "anndats"]].copy()
    a["ticker"] = a["ticker"].astype("string")
    a["_i"] = np.arange(len(a))
    a = a.dropna(subset=["anndats", "pends"])
    # statpers < anndats: search on anndats - 1 ns, backward
    a["_key"] = a["anndats"] - pd.Timedelta(1, "ns")
    a = a.sort_values("_key")
    m = pd.merge_asof(a, s.rename(columns={"fpedats": "pends"}), left_on="_key",
                      right_on="statpers", by=["ticker", "pends"], direction="backward")
    keep = ["statpers", *[c for c in cols if c in s.columns]]
    out = m.set_index("_i")[keep].reindex(np.arange(len(actuals)))
    return out.reset_index(drop=True)
    return out


def eps_surprise_history(actuals: pd.DataFrame, consensus: pd.DataFrame,
                         adj: pd.DataFrame) -> pd.DataFrame:
    """Every quarterly EPS actual with its consensus and surprise (2008 on), plus SUE_TS.

    Columns: ticker, pends, anndats, actual, statpers, medest, stdev, numest, numup, numdown,
    highest, lowest (consensus on the actual's share basis), surprise, surprise_today
    (today's share basis), sue_ts.
    """
    a = actuals
    if "measure" in a.columns:
        a = a[a["measure"] == "EPS"]
    if "pdicity" in a.columns:
        a = a[a["pdicity"] == "QTR"]
    a = pd.DataFrame({"ticker": a["ticker"].astype("string").to_numpy(),
                      "pends": _date(a["pends"]).to_numpy(),
                      "anndats": _date(a["anndats"]).to_numpy(),
                      "actual": _num(a["value"])})
    a = a.dropna(subset=["ticker", "pends", "anndats"])
    a = a.sort_values("anndats").drop_duplicates(["ticker", "pends"], keep="first")
    a = a.reset_index(drop=True)
    c = _consensus(a, consensus)
    a = pd.concat([a, c], axis=1)
    f_ann = split_factor(adj, a["ticker"], a["anndats"])
    f_stat = split_factor(adj, a["ticker"], a["statpers"])
    to_actual = np.where(a["statpers"].notna(), f_ann / f_stat, np.nan)
    for col in ("medest", "stdev", "highest", "lowest"):
        a[col] = a[col] * to_actual
    a["surprise"] = a["actual"] - a["medest"]
    a["surprise_today"] = a["surprise"] / f_ann
    a = a.sort_values(["ticker", "anndats", "pends"]).reset_index(drop=True)
    sd, last = _prior_scale(a)
    a["sue_ts_scale_date"] = last
    a["sue_ts"] = a["surprise_today"] / pd.Series(sd, index=a.index).replace(0.0, np.nan)
    return a


def _prior_scale(a: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """SD of the last ``SUE_HIST[0]`` surprises announced strictly before each row's anndats
    (at least ``SUE_HIST[1]``), and the latest anndats among them. Rows announced on the same
    day never enter each other's scale. ``a`` is sorted by ticker, anndats."""
    w, mp = SUE_HIST
    sd = np.full(len(a), np.nan)
    last = np.full(len(a), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    surprise = a["surprise_today"].to_numpy(float)
    ann = a["anndats"].to_numpy("datetime64[ns]")
    for _, idx in a.groupby("ticker", sort=False).indices.items():
        d, x = ann[idx], surprise[idx]
        for k, i in enumerate(idx):
            j = np.searchsorted(d, d[k], side="left")      # rows dated before this one
            prior = x[max(0, j - w):j]
            prior = prior[~np.isnan(prior)] if j else prior
            if j:
                last[i] = d[j - 1]
            if len(prior) >= mp:
                sd[i] = np.std(prior, ddof=1)
    return sd, last


def earnings_surprise(events: pd.DataFrame, history: pd.DataFrame,
                      price_m1: pd.Series) -> pd.DataFrame:
    """One row per event: SUE variants, analyst controls, net revisions and forecast range.

    ``events``: keydevid, ibes_ticker, release_pends, release_anndats, day0.
    ``history``: :func:`eps_surprise_history`. ``price_m1``: |price| at Day -1 by keydevid.
    """
    e = pd.DataFrame({"keydevid": events["keydevid"].astype("int64").to_numpy(),
                      "ticker": events["ibes_ticker"].astype("string").to_numpy(),
                      "pends": _date(events["release_pends"]).to_numpy(),
                      "anndats": _date(events["release_anndats"]).to_numpy()})
    h = history.drop_duplicates(["ticker", "pends"]).rename(columns={"anndats": "anndats_act"})
    m = e.merge(h, on=["ticker", "pends"], how="left")
    # the actual must be the event's own release
    other = m["anndats_act"].notna() & (m["anndats_act"] != m["anndats"])
    m.loc[other, h.columns.difference(["ticker", "pends"])] = np.nan
    price = m["keydevid"].map(price_m1).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "keydevid": m["keydevid"],
            "eps_actual": m["actual"], "eps_consensus": m["medest"], "eps_surprise": m["surprise"],
            "sue_ts": m["sue_ts"],
            "sue": m["surprise"] / np.where(price > 0, price, np.nan),
            "sue_std": m["surprise"] / m["stdev"].where(m["stdev"] > 0),
            "n_analysts": m["numest"],
            "dispersion": m["stdev"] / m["medest"].abs().where(m["medest"] != 0),
            "net_revisions": (m["numup"] - m["numdown"]) / m["numest"].where(m["numest"] > 0),
            "forecast_range": (m["highest"] - m["lowest"]) / np.where(price > 0, price, np.nan),
            "src_eps_anndats": pd.to_datetime(m["anndats_act"]),
            "src_consensus_statpers": pd.to_datetime(m["statpers"]),
            "src_sue_ts_scale_date": pd.to_datetime(m["sue_ts_scale_date"]),
        })
    return out.reset_index(drop=True)


def revenue_surprise(events: pd.DataFrame, act_x: pd.DataFrame,
                     statsum_x: pd.DataFrame) -> pd.DataFrame:
    """``keydevid, rev_sue`` (SAL, percent of consensus) and the dates used."""
    a = act_x
    a = a[(a["measure"] == "SAL") & (a.get("pdicity", "QTR") == "QTR")]
    a = pd.DataFrame({"ticker": a["ticker"].astype("string").to_numpy(),
                      "pends": _date(a["pends"]).to_numpy(),
                      "anndats": _date(a["anndats"]).to_numpy(), "actual": _num(a["value"])})
    a = a.dropna(subset=["ticker", "pends", "anndats", "actual"])
    a = a.sort_values("anndats").drop_duplicates(["ticker", "pends"]).reset_index(drop=True)
    c = _consensus(a, statsum_x, measure="SAL", cols=("medest",))
    a = pd.concat([a, c], axis=1)
    e = pd.DataFrame({"keydevid": events["keydevid"].astype("int64").to_numpy(),
                      "ticker": events["ibes_ticker"].astype("string").to_numpy(),
                      "pends": _date(events["release_pends"]).to_numpy(),
                      "day0": _date(events["day0"]).to_numpy()})
    m = e.merge(a, on=["ticker", "pends"], how="left")
    late = m["anndats"] > m["day0"]                   # reported after Day 0: not usable
    with np.errstate(invalid="ignore", divide="ignore"):
        rev = (m["actual"] - m["medest"]) / m["medest"].abs().where(m["medest"] != 0)
    rev[late] = np.nan
    return pd.DataFrame({"keydevid": m["keydevid"], "rev_sue": rev.to_numpy(float),
                         "src_rev_anndats": m["anndats"].where(~late),
                         "src_rev_statpers": m["statpers"].where(~late)})


def _period_end(yr, mon) -> pd.Series:
    yr = pd.to_numeric(pd.Series(yr), errors="coerce")
    mon = pd.to_numeric(pd.Series(mon), errors="coerce")
    ok = yr.notna() & mon.between(1, 12)
    out = pd.Series(pd.NaT, index=yr.index, dtype="datetime64[ns]")
    s = pd.to_datetime(dict(year=yr[ok].astype(int), month=mon[ok].astype(int), day=1))
    out[ok] = (s + pd.offsets.MonthEnd(0)).to_numpy()
    return out


def release_guidance(events: pd.DataFrame, guidance: pd.DataFrame, adj: pd.DataFrame,
                     price_m1: pd.Series) -> pd.DataFrame:
    """``keydevid, guidance_given, guidance_vs_consensus, guidance_width, guidance_change``
    (NaN where the metric is undefined; :func:`flag_and_zero` fills them)."""
    e = pd.DataFrame({"keydevid": events["keydevid"].astype("int64").to_numpy(),
                      "ticker": events["ibes_ticker"].astype("string").to_numpy(),
                      "pends": _date(events["release_pends"]).to_numpy(),
                      "rel": _date(events["release_anndats"]).to_numpy(),
                      "day0": _date(events["day0"]).to_numpy()})
    g = pd.DataFrame({"ticker": guidance["ticker"].astype("string").to_numpy(),
                      "measure": guidance["measure"].astype("string").to_numpy(),
                      "pdicity": guidance["pdicity"].astype("string").to_numpy(),
                      "anndats": _date(guidance["anndats"]).to_numpy(),
                      "anntims": guidance["anntims"].astype("string").fillna("").to_numpy(),
                      "period_end": _period_end(guidance["prd_yr"], guidance["prd_mon"]).to_numpy(),
                      "val_1": _num(guidance["val_1"]), "val_2": _num(guidance["val_2"]),
                      "mean_at_date": _num(guidance["mean_at_date"])})
    g = g.dropna(subset=["ticker", "anndats"])
    g["mid"] = np.where(np.isnan(g["val_2"]), g["val_1"], (g["val_1"] + g["val_2"]) / 2)
    g["width"] = np.where(np.isnan(g["val_2"]), 0.0, g["val_2"] - g["val_1"])

    # every guidance row of the event's ticker inside the window and on or before Day 0
    j = e.merge(g, on="ticker", how="inner")
    lo = j["rel"] - pd.Timedelta(days=GUIDANCE_DAYS)
    hi = j["rel"] + pd.Timedelta(days=GUIDANCE_DAYS)
    j = j[(j["anndats"] >= lo) & (j["anndats"] <= hi) & (j["anndats"] <= j["day0"])]
    given = j.groupby("keydevid").size()
    latest = j.groupby("keydevid")["anndats"].max()

    eps = j[(j["measure"] == "EPS") & (j["period_end"] > j["pends"]) & j["mid"].notna()].copy()
    eps["qtr_first"] = (eps["pdicity"] != "QTR").astype(int)
    eps = eps.sort_values(["keydevid", "period_end", "qtr_first", "anndats", "anntims"],
                          ascending=[True, True, True, False, False])
    eps = eps.drop_duplicates("keydevid")

    # previous EPS guidance for the same period, announced before the release window
    prev = g[g["measure"] == "EPS"][["ticker", "pdicity", "period_end", "anndats", "mid"]]
    prev = prev.dropna(subset=["period_end", "mid"]).sort_values("anndats")
    q = eps[["keydevid", "ticker", "pdicity", "period_end", "rel"]].copy()
    q[["ticker", "pdicity"]] = q[["ticker", "pdicity"]].astype("string")
    prev = prev.astype({"ticker": "string", "pdicity": "string"})
    q["_key"] = q["rel"] - pd.Timedelta(days=GUIDANCE_DAYS) - pd.Timedelta(1, "ns")
    q = q.sort_values("_key")
    pm = pd.merge_asof(q, prev.rename(columns={"anndats": "prev_anndats", "mid": "prev_mid"}),
                       left_on="_key", right_on="prev_anndats",
                       by=["ticker", "pdicity", "period_end"], direction="backward")
    eps = eps.merge(pm[["keydevid", "prev_mid"]], on="keydevid", how="left")
    d = eps["mid"] - eps["prev_mid"]
    eps["change"] = np.where(d.isna(), np.nan, np.where(d > CHANGE_TOL, 1.0,
                                                        np.where(d < -CHANGE_TOL, -1.0, 0.0)))
    f = split_factor(adj, eps["ticker"], eps["anndats"])
    price = eps["keydevid"].map(price_m1).to_numpy(float)
    price = np.where(price > 0, price, np.nan)
    vs = (eps["mid"] - eps["mean_at_date"]) * f / price
    # data-error guard: a guidance-consensus gap above half the share price is a bad
    # mean_at_date (e.g. PFE 2025-11-04: 23,625 against guidance of $3), not news
    eps["vs_cons"] = vs.where(vs.abs() <= MAX_GUIDANCE_GAP)
    eps["width_p"] = eps["width"] * f / price
    eps = eps.set_index("keydevid")

    k = e["keydevid"]
    return pd.DataFrame({
        "keydevid": k,
        "guidance_given": k.map(given).notna().astype(float).to_numpy(),
        "guidance_vs_consensus": k.map(eps["vs_cons"]).to_numpy(float),
        "guidance_width": k.map(eps["width_p"]).to_numpy(float),
        "guidance_change": k.map(eps["change"]).to_numpy(float),
        "src_guidance_anndats": pd.to_datetime(k.map(latest)).to_numpy(),
    })


def flag_and_zero(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """``controls.release_missing: flag_and_zero``: each missing metric -> 0 plus
    ``<metric>_missing`` (1 = was missing). ``guidance_given`` is never missing."""
    out = df.copy()
    for c in metrics:
        if c == "guidance_given":
            out[c] = out[c].fillna(0.0)
            continue
        miss = out[c].isna()
        out[c + "_missing"] = miss.astype("int64")
        out[c] = out[c].fillna(0.0)
    return out
