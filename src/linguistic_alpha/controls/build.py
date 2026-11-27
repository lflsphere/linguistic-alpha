"""Assemble processed/controls: one row per keydevid, raw (not standardised).

Columns: ``schemas.CONTROLS``, ``sue_ts`` (primary SUE, ``controls.sue_scaler: ts``), the release
metrics with ``<metric>_missing`` flags (``controls.release_missing: flag_and_zero``), and
``src_*`` dates of the latest input behind each control, used by :func:`lookahead_check`.
Winsorising and z-scoring happen inside each training window (docs/pipeline.md,
Standardisation), never here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from linguistic_alpha import schemas
from linguistic_alpha.controls.earnings import (earnings_surprise, eps_surprise_history,
                                                flag_and_zero, release_guidance, revenue_surprise)
from linguistic_alpha.controls.firm import firm_controls
from linguistic_alpha.targets.car import abnormal_returns, benchmark_series

SUE_COLUMN = {"ts": "sue_ts", "price": "sue", "stdev": "sue_std"}

# input date -> rule against Day 0: "lt" strictly before Day 0, "le" on or before Day 0
# (released with the news), "entry" on or before Day +1 (the trade entry)
LOOKAHEAD = {
    "src_consensus_statpers": "lt", "src_sue_ts_scale_date": "lt", "src_price_date": "lt",
    "src_size_date": "lt", "src_fundq_avail": "lt", "src_mom_end": "lt",
    "src_vol_pre_end": "lt", "src_rev_statpers": "lt",
    "src_eps_anndats": "le", "src_rev_anndats": "le", "src_guidance_anndats": "le",
    "src_reaction_end": "entry",
}


def lookahead_check(out: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Violations per input date column (0 everywhere = no control uses data after its time)."""
    day0 = pd.to_datetime(out["day0"])
    pos = sessions.searchsorted(day0.to_numpy(), side="left")
    entry = pd.Series(sessions[np.clip(pos + 1, 0, len(sessions) - 1)], index=out.index)
    rows = []
    for c, rule in LOOKAHEAD.items():
        d = pd.to_datetime(out[c])
        has = d.notna()
        if rule == "lt":
            bad = has & (d >= day0)
        elif rule == "le":
            bad = has & (d > day0)
        else:
            bad = has & (d > entry)
        rows.append({"input": c, "rule": {"lt": "< Day 0", "le": "<= Day 0",
                                          "entry": "<= Day +1 (entry)"}[rule],
                     "checked": int(has.sum()), "violations": int(bad.sum())})
    return pd.DataFrame(rows)


def build_controls(events: pd.DataFrame, dsf: pd.DataFrame, tables: dict,
                   calls: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    tcfg, ccfg = cfg["targets"], cfg["controls"]
    col = tcfg.get("benchmark_column", "vwretd")
    bench = tables["crsp_dsp500_v2"]
    b = benchmark_series(bench, col)
    ar = abnormal_returns(dsf[dsf["permno"].isin(events["permno"])], bench, "sp500", col)

    firm = firm_controls(events, dsf, tables["comp_fundq"], ar, b, tcfg,
                         ccfg.get("mom_window_months", [-12, -2]))
    price = firm.set_index("keydevid")["price_m1"]
    hist = eps_surprise_history(tables["ibes_actu_epsus"], tables["ibes_statsumu_epsus"],
                                tables["ibes_adj"])
    eps = earnings_surprise(events, hist, price)
    rev = revenue_surprise(events, tables["ibes_act_xepsus"], tables["ibes_statsum_xepsus"])
    guid = release_guidance(events, tables["ibes_det_guidance"], tables["ibes_adj"], price)

    vf = calls.dropna(subset=["keydevid"]).astype({"keydevid": "int64"})
    vf = vf.drop_duplicates("keydevid").set_index("keydevid")["version_final"]

    out = events[["keydevid", "permno", "gvkey", "ibes_ticker", "day0"]].copy()
    out["keydevid"] = out["keydevid"].astype("int64")
    for part in (eps, firm, rev, guid):
        out = out.merge(part, on="keydevid", how="left")
    out["version_final"] = out["keydevid"].map(vf)
    raw_metrics = out[ccfg["release_metrics"]].copy()
    out = flag_and_zero(out, ccfg["release_metrics"])
    out["sue_primary"] = SUE_COLUMN[ccfg.get("sue_scaler", "ts")]

    missing_vf = int(out["version_final"].isna().sum())
    out["version_final"] = out["version_final"].fillna(-1).astype("int64")
    schemas.validate(out, schemas.CONTROLS, "controls", key="keydevid")
    for c in ccfg["release_metrics"]:
        if c not in out.columns:
            raise ValueError(f"controls: release metric {c} not built")

    sessions = pd.DatetimeIndex(b.index)
    la = lookahead_check(out, sessions)
    report = _report(out, raw_metrics, la, missing_vf, ccfg)
    return out, report


def _report(out, raw_metrics, la, missing_vf, ccfg) -> str:
    cols = ["sue_ts", "sue", "sue_std", "size", "bm", "mom", "vol_pre", "n_analysts",
            "dispersion", "car_m1_1", "car_0_1"]
    year = pd.to_datetime(out["day0"]).dt.year
    rm = raw_metrics.drop(columns=["guidance_given"], errors="ignore")
    rm.columns = [c + " (raw)" for c in rm.columns]
    na = pd.concat([out[cols].isna(), rm.isna()], axis=1)
    t = na.groupby(year).mean()
    t.loc["all"] = na.mean()
    t.insert(0, "events", list(year.value_counts().sort_index()) + [len(out)])
    pct = t.copy()
    for c in pct.columns[1:]:
        pct[c] = (100 * pct[c]).map(lambda v: f"{v:.1f}")
    gg = out.groupby(year)["guidance_given"].mean()
    lines = [
        "# la controls log", "",
        f"{len(out):,} events, Day 0 {pd.to_datetime(out['day0']).min():%Y-%m-%d} to "
        f"{pd.to_datetime(out['day0']).max():%Y-%m-%d}. Primary SUE column: "
        f"`{out['sue_primary'].iloc[0]}` (`controls.sue_scaler: {ccfg.get('sue_scaler')}`).",
        f"Release metrics: `flag_and_zero`; `(raw)` columns below are the missing rates before "
        f"the zero fill. version_final missing (no transcript_calls row, stored as -1): "
        f"{missing_vf}.", "",
        "## Missing rate (%) by year of Day 0", "", _md(pct.reset_index(names="year")), "",
        "## guidance_given share by year", "",
        _md(gg.map(lambda v: f"{100 * v:.1f}").rename("guidance_given %").rename_axis("year").reset_index(
            )), "",
        "## Look-ahead check (latest input date per control vs. Day 0)", "", _md(la), "",
        f"Total violations: **{int(la['violations'].sum())}**", "",
    ]
    return "\n".join(lines)


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(rows)
