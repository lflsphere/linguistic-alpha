"""Cumulative abnormal returns and post-call volatility.

Inputs: crsp_dsf_v2 (dlyret), crsp_dsp500_v2 (vwretd), crsp_stkdelists; Day 0 from
processed/events.

Definitions (project scope §1, §4):
  AR(i,t)          = R(i,t) - R(bench,t); bench = S&P 500 total return, point in time
                     (crsp_dsp500_v2.vwretd: the index's constituents on each day, dividends
                     included like dlyret). Robustness: the market model,
                     AR = R(i) - (alpha + beta * R(S&P 500 total return)), alpha and beta by OLS on days -250..-30 before Day 0 (at least 80% valid days).
                     CRSP VW market = index check only; FF4 is left for a future risk-management leg.
  R(i,t)           = dlyret, which already includes delisting returns (no splice).
  CAR[a,b]         = prod(1 + R(i,t)) - prod(1 + R(bench,t)) over sessions Day0+a .. Day0+b:
                     the stock and the index are compounded separately (buy-and-hold), then
                     differenced. Null stock days are skipped in the stock product (never filled
                     with 0); the index is compounded over every day; the window is missing when
                     valid days < ``targets.min_window_coverage``. Market model: the benchmark
                     leg is alpha + beta * R(bench,t), compounded the same way.
  vol_post_2_20    = std(AR, +2..+20) * sqrt(252)
  vol_ratio_2_20   = vol_post_2_20 / (std(AR, -60..-5) * sqrt(252))   (scope definition)

Edge rules (not in the scope doc):
  * Delisting inside a window (approved by the user 2026-11-27): the delisting date's dlyret
    holds the delisting return; on later sessions the proceeds are held in the benchmark leg,
    so AR = 0 and the day counts as valid. Dropping these events instead would remove mostly
    cash mergers from the targets (33 events on the 2026-11-27 run, 6 inside [+2, +20]).
    For volatility, post-delisting days are missing (there is no trading to measure).
  * A CAR window that runs past the last session in the data is missing, whatever its coverage
    (CRSP ends 2026-09-30, so CAR[+2,+60] is missing for Day 0 from July 2026).
  * Sessions are the benchmark's trading calendar. A Day 0 that is not a session rolls forward.
  * Volatility uses the same coverage minimum as the CARs, std with ddof = 1.

Fixes vs. the legacy notebook (docs/data_audit.md §Targets):
  * Delisting: the legacy file needed R = (1+ret)(1+dlret)-1; CIZ ``dlyret`` already includes
    it, so splicing ``delret`` again would count it twice.
  * Benchmark consistency: the legacy notebook used ``sprtrn`` (price only) against a return
    that includes dividends (AR drifted up ~2%/yr); ``vwretd`` (total return) fixes it.
  * Coverage rule (``targets.min_window_coverage``) is reported per event, not only applied.
  * The legacy reaction_windows.csv summed daily ARs (and needed every day); CARs here are
    compounded per leg, so they differ from it by the compounding cross terms.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

ANNUALIZE = np.sqrt(252.0)
_EPS = 1e-9


def _col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def benchmark_series(bench: pd.DataFrame, column: str = "vwretd") -> pd.Series:
    """Benchmark daily return indexed by session date (sorted, unique)."""
    date = _col(bench, "date", "caldt", "dlycaldt")
    s = pd.Series(pd.to_numeric(bench[column], errors="coerce").to_numpy(float),
                  index=pd.to_datetime(bench[date]))
    s = s[~s.index.duplicated()].sort_index()
    return s.rename("bench_ret")


def abnormal_returns(dsf: pd.DataFrame, bench: pd.DataFrame,
                     model: str = "sp500", bench_column: str = "vwretd") -> pd.DataFrame:
    """Daily ``permno, date, ret, bench_ret, ar``; CIZ ``dlyret`` already holds delisting returns.

    One row per permno and benchmark session from the permno's first to its last CRSP row, so
    a session CRSP leaves out, or a null dlyret, shows up as ``ret`` NaN (never 0).
    ``model``: "sp500" or "vwmkt" (AR = R - R(bench), beta = 1; pass the matching ``bench``
    frame). The market model is estimated per event: use ``event_car(market_model=...)``.
    """
    if model == "market_model":
        raise ValueError("market-model ARs depend on the event: use event_car(market_model=...)")
    if model not in ("sp500", "vwmkt"):
        raise ValueError(f"unknown model {model!r}")
    b = benchmark_series(bench, bench_column)
    sessions = b.index
    date = _col(dsf, "date", "dlycaldt")
    ret = _col(dsf, "ret", "dlyret")
    d = pd.DataFrame({"permno": dsf["permno"].astype("int64").to_numpy(),
                      "date": pd.to_datetime(dsf[date]).to_numpy(),
                      "ret": pd.to_numeric(dsf[ret], errors="coerce").to_numpy(float)})
    d = d.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"])
    span = d.groupby("permno")["date"].agg(["min", "max"])
    lo = sessions.searchsorted(span["min"].to_numpy(), side="left")
    hi = sessions.searchsorted(span["max"].to_numpy(), side="right")
    n = hi - lo
    grid = pd.DataFrame({"permno": np.repeat(span.index.to_numpy(), n),
                         "date": sessions[np.concatenate([np.arange(a, z) for a, z in zip(lo, hi)])
                                          if len(n) else np.array([], dtype=int)]})
    out = grid.merge(d, on=["permno", "date"], how="left")
    out["bench_ret"] = b.reindex(out["date"]).to_numpy()
    out["ar"] = out["ret"] - out["bench_ret"]
    return out[["permno", "date", "ret", "bench_ret", "ar"]]


class _Panel:
    """Dense session x permno return matrix plus each event's Day 0 position."""

    def __init__(self, events: pd.DataFrame, ar: pd.DataFrame, bench: pd.Series | None,
                 delist_dates: pd.Series | None):
        if bench is None:
            bench = ar.drop_duplicates("date").set_index("date")["bench_ret"].sort_index()
        self.sessions = pd.DatetimeIndex(bench.index)
        self.b = bench.to_numpy(float)
        permnos = pd.Index(pd.unique(ar["permno"]))
        self.R = np.full((len(self.sessions), len(permnos)), np.nan)
        ri = self.sessions.get_indexer(pd.to_datetime(ar["date"]))
        ci = permnos.get_indexer(ar["permno"])
        ok = ri >= 0
        self.R[ri[ok], ci[ok]] = ar["ret"].to_numpy(float)[ok]
        # session index of the delisting date per permno (n_sessions = never delisted)
        self.delist_pos = np.full(len(permnos), len(self.sessions) + 10**6)
        if delist_dates is not None and len(delist_dates):
            dd = pd.to_datetime(delist_dates).groupby(level=0).max()
            j = permnos.get_indexer(dd.index)
            keep = j >= 0
            self.delist_pos[j[keep]] = self.sessions.searchsorted(dd.to_numpy()[keep], side="left")
        self.col = permnos.get_indexer(events["permno"])
        d0 = pd.to_datetime(events["day0"]).to_numpy()
        self.pos0 = self.sessions.searchsorted(d0, side="left")
        self.pos0 = np.where(self.col >= 0, self.pos0, -10**7)  # no CRSP data -> all invalid

    def window(self, a: int, b: int):
        """(stock returns, bench returns, in-calendar mask, post-delisting mask), each n x L."""
        idx = self.pos0[:, None] + np.arange(a, b + 1)[None, :]
        inside = (idx >= 0) & (idx < len(self.sessions))
        safe = np.clip(idx, 0, len(self.sessions) - 1)
        cols = np.clip(self.col, 0, None)[:, None]
        r = np.where(inside, self.R[safe, cols], np.nan)
        bb = np.where(inside, self.b[safe], np.nan)
        post = inside & (idx > self.delist_pos[np.clip(self.col, 0, None)][:, None]) \
            & (self.col >= 0)[:, None]
        return r, bb, inside, post


def _enough(n_valid: np.ndarray, length: int, min_coverage: float) -> np.ndarray:
    return n_valid >= min_coverage * length - _EPS


def _market_model(p: _Panel, est: list[int], min_coverage: float):
    r, bb, inside, _ = p.window(est[0], est[1])
    valid = ~np.isnan(r) & ~np.isnan(bb)
    n = valid.sum(1)
    x = np.where(valid, bb, np.nan)
    y = np.where(valid, r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx, my = np.nanmean(x, 1), np.nanmean(y, 1)
        beta = np.nansum((x - mx[:, None]) * (y - my[:, None]), 1) / \
            np.nansum((x - mx[:, None]) ** 2, 1)
    alpha = my - beta * mx
    bad = ~_enough(n, est[1] - est[0] + 1, min_coverage)
    alpha[bad] = np.nan
    beta[bad] = np.nan
    return alpha, beta


def event_car(events: pd.DataFrame, ar: pd.DataFrame, windows: dict[str, list[int]],
              min_coverage: float, *, bench: pd.Series | None = None,
              delist_dates: pd.Series | None = None,
              market_model: dict | None = None) -> pd.DataFrame:
    """One row per keydevid with every configured CAR window (stock and index compounded
    separately) and its valid-day coverage.

    ``events``: keydevid, permno, day0. ``ar``: output of :func:`abnormal_returns`.
    ``bench``: benchmark series over every session (default: taken from ``ar``; pass it when
    windows can run past a permno's last row, e.g. after a delisting).
    ``delist_dates``: delisting date per permno (index = permno); later sessions get AR = 0.
    ``market_model``: ``{"estimation_window": [-250, -30], "min_coverage": 0.8}`` adds
    ``<name>_mm`` columns and ``mm_alpha``, ``mm_beta``.
    Columns: ``<name>``, ``<name>_cov`` (valid days / window length), ``delist_rel`` (session
    of the delisting relative to Day 0 when it falls inside the widest window).
    """
    p = _Panel(events, ar, bench, delist_dates)
    out = pd.DataFrame({"keydevid": events["keydevid"].to_numpy()})
    if market_model is not None:
        alpha, beta = _market_model(p, market_model["estimation_window"],
                                    market_model.get("min_coverage", 0.8))
        out["mm_alpha"], out["mm_beta"] = alpha, beta
    for name, (a, b) in windows.items():
        r, bb, inside, post = p.window(a, b)
        length = b - a + 1
        complete = inside.all(1)                      # the window lies inside the data
        legs = {"": bb}
        if market_model is not None:
            legs["_mm"] = alpha[:, None] + beta[:, None] * bb
        for suffix, leg in legs.items():
            rs = np.where(post, leg, r)               # after delisting: held in the benchmark leg
            valid = ~np.isnan(rs)
            stock = np.prod(np.where(valid, 1.0 + rs, 1.0), 1)
            index = np.prod(1.0 + leg, 1)             # every day; NaN if the leg is undefined
            car = stock - index
            ok = complete & _enough(valid.sum(1), length, min_coverage)
            out[name + suffix] = np.where(ok, car, np.nan)
            if suffix == "":
                out[name + "_cov"] = valid.sum(1) / length
    lo = min(w[0] for w in windows.values())
    hi = max(w[1] for w in windows.values())
    rel = p.delist_pos[np.clip(p.col, 0, None)] - p.pos0
    out["delist_rel"] = pd.array(np.where((p.col >= 0) & (rel >= lo) & (rel <= hi), rel, -1),
                                 dtype="Int64")
    out.loc[out["delist_rel"] == -1, "delist_rel"] = pd.NA
    return out


def post_call_volatility(events: pd.DataFrame, ar: pd.DataFrame, post: list[int],
                         pre: list[int], min_coverage: float = 0.8, *,
                         bench: pd.Series | None = None,
                         delist_dates: pd.Series | None = None) -> pd.DataFrame:
    """``keydevid, vol_post_<a>_<b>, vol_ratio_<a>_<b>``: annualized std of daily AR (ddof 1)
    over ``post``, and its ratio to the same over ``pre``. Missing below ``min_coverage``;
    post-delisting sessions are missing."""
    p = _Panel(events, ar, bench, delist_dates)

    def vol(w):
        r, bb, _, after = p.window(w[0], w[1])
        x = np.where(after, np.nan, r - bb)
        n = (~np.isnan(x)).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.nanstd(np.where(n[:, None] > 1, x, 0.0), axis=1, ddof=1) * ANNUALIZE
        return np.where(_enough(n, w[1] - w[0] + 1, min_coverage) & (n > 1), s, np.nan)

    v_post, v_pre = vol(post), vol(pre)
    tag = f"{post[0]}_{post[1]}".replace("-", "m")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(v_pre > 0, v_post / v_pre, np.nan)
    return pd.DataFrame({"keydevid": events["keydevid"].to_numpy(),
                         f"vol_post_{tag}": v_post, f"vol_ratio_{tag}": ratio})


def build_targets(events: pd.DataFrame, dsf: pd.DataFrame, bench: pd.DataFrame,
                  delists: pd.DataFrame | None, tcfg: dict) -> pd.DataFrame:
    """processed/targets from events, CRSP daily, the benchmark frame and delistings."""
    model = tcfg.get("benchmark", "sp500")
    col = tcfg.get("benchmark_column", "vwretd")
    ar = abnormal_returns(dsf, bench, "vwmkt" if model == "vwmkt" else "sp500", col)
    b = benchmark_series(bench, col)
    dd = None
    if delists is not None and len(delists):
        dl = delists.dropna(subset=["permno", "delistingdt"])
        dd = pd.Series(pd.to_datetime(dl["delistingdt"]).to_numpy(),
                       index=dl["permno"].astype("int64").to_numpy())
    cars = event_car(events, ar, tcfg["car_windows"], tcfg["min_window_coverage"], bench=b,
                     delist_dates=dd, market_model=tcfg.get("market_model"))
    if model == "market_model":   # main columns follow targets.benchmark
        for name in tcfg["car_windows"]:
            cars[name + "_sp500"] = cars[name]
            cars[name] = cars[name + "_mm"]
    vol = post_call_volatility(events, ar, tcfg["vol_post_window"], tcfg["vol_pre_window"],
                               tcfg["min_window_coverage"], bench=b, delist_dates=dd)
    out = cars.merge(vol, on="keydevid", how="left")
    out.insert(1, "permno", events["permno"].astype("int64").to_numpy())
    out.insert(2, "day0", pd.to_datetime(events["day0"]).to_numpy())
    return out
