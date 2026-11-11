#!/usr/bin/env python3
"""Train several language-free baselines on a chronological 80/20 split.

The same date cutoff and common labeled sample are used for CAR[+2,+5],
CAR[+2,+20], and CAR[+2,+60]. The test period is evaluated once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm

from linguistic_alpha.config import load_config
from linguistic_alpha.io import read_table


CONTROLS = ["sue", "size", "bm", "mom", "car_0_1", "vol_pre", "n_analysts", "dispersion"]
TARGETS = {"CAR[+2,+5]": "car_2_5", "CAR[+2,+20]": "car_2_20", "CAR[+2,+60]": "car_2_60"}
MODELS = ["zero", "train_mean", "ols_controls", "ols_industry_year_fe", "ridge_industry_year_fe",
          "random_forest", "extra_trees"]
NET_COST = 0.0020  # illustrative 5 bps per side at entry/exit, both long and short legs


def pipeline(model: str, train: pd.DataFrame, seed: int, *, controls=CONTROLS) -> Pipeline:
    numeric = [c for c in controls if c in train]
    categorical = [c for c in ("gsubind", "year") if c in train] if "industry_year_fe" in model else []
    steps_num = [("impute", SimpleImputer(strategy="median"))]
    if model.startswith(("ols", "ridge")):
        steps_num.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(steps_num), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers, sparse_threshold=1.0)
    if model.startswith("ols"):
        estimator = LinearRegression()
    elif model.startswith("ridge"):
        estimator = Ridge(alpha=10.0, solver="lsqr")
    elif model == "random_forest":
        estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                                          max_features=0.8, n_jobs=-1, random_state=seed)
    elif model == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=10,
                                        max_features=0.8, n_jobs=-1, random_state=seed)
    else:
        raise ValueError(model)
    return Pipeline([("prep", prep), ("model", estimator)])


def clean_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable scalars to NumPy NaN for sklearn imputers."""
    out = frame.copy()
    for col in CONTROLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in ("gsubind", "year"):
        if col in out:
            out[col] = out[col].astype(object)
            out.loc[pd.isna(out[col]), col] = np.nan
    return out


def qspread(pred: pd.DataFrame, pred_col: str, actual_col: str) -> tuple[float, float, int]:
    spreads = []
    for _, g in pred.groupby(pred["call_date"].dt.to_period("Q")):
        g = g[[pred_col, actual_col]].dropna()
        if len(g) < 20:
            continue
        low, high = g[pred_col].quantile([0.2, 0.8])
        long = g.loc[g[pred_col] >= high, actual_col].mean()
        short = g.loc[g[pred_col] <= low, actual_col].mean()
        spreads.append(long - short)
    gross = float(np.mean(spreads)) if spreads else np.nan
    return gross, gross - NET_COST if np.isfinite(gross) else np.nan, len(spreads)


def html_dashboard(metrics: pd.DataFrame, eda: dict, split: dict) -> str:
    metric_html = metrics.sort_values(["target", "rank_ic"], ascending=[True, False]).to_html(
        index=False, float_format=lambda x: f"{x:.4f}", classes="metrics")
    yearly_html = eda["events_by_year"].to_html(index=False, classes="metrics")
    missing_html = eda["missing_controls"].to_html(index=False, float_format=lambda x: f"{x:.1%}",
                                                    classes="metrics")
    target_html = eda["target_summary"].to_html(index=False, float_format=lambda x: f"{x:.4f}",
                                                 classes="metrics")
    corr_html = eda["control_target_corr"].to_html(index=False, float_format=lambda x: f"{x:.3f}",
                                                     classes="metrics")
    ols_html = eda["ols_controls"].to_html(index=False, float_format=lambda x: f"{x:.4g}",
                                            classes="metrics")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Model A baseline dashboard</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:36px auto;padding:0 20px;color:#18212f}}
h1,h2{{color:#17365d}}.card{{display:inline-block;background:#f1f5f9;padding:12px 20px;margin:5px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;margin:14px 0 30px;font-size:13px}}th,td{{padding:8px;border:1px solid #d5dce5;text-align:right}}
th{{background:#e8eef5}}td:first-child,th:first-child{{text-align:left}}.note{{color:#536273}}
</style></head><body><h1>Model A: language-free return forecasts</h1>
<p>Chronological split on event date; controls only. All events on the cutoff date stay in one split.</p>
<div class="card">Train: {split['train_n']:,} events ({split['train_start']}–{split['train_end']})</div>
<div class="card">Test: {split['test_n']:,} events ({split['test_start']}–{split['test_end']})</div>
<div class="card">Cutoff: {split['cutoff_date']} | Purge boundary: {split['purge_day0_before']}</div>
<h2>Out-of-sample metrics by horizon and model</h2>{metric_html}
<p class="note">IC = Pearson correlation; rank IC = Spearman correlation. Net Q5–Q1 subtracts an illustrative
20 bp round-trip cost (5 bp per trade, each long/short leg). It excludes borrow, market impact, and overlapping-position effects.</p>
<h2>Event counts by year</h2>{yearly_html}
<h2>Missingness in baseline controls</h2>{missing_html}
<h2>Target distribution (common labeled sample)</h2>{target_html}
<h2>Control–target Pearson correlations (EDA)</h2>{corr_html}
<h2>Training-period pooled OLS on controls with PERMNO-clustered standard errors</h2>{ols_html}
<h2>Model definitions</h2><ul><li>Zero and training-mean prediction benchmarks.</li>
<li>OLS on controls; OLS with GICS sub-industry and year fixed effects.</li>
<li>Ridge with sub-industry/year effects (alpha=10, fixed in advance).</li>
<li>Random forest and extremely randomized trees using the same standard controls.</li></ul>
<p class="note">The OLS coefficient table is fit on training data only; standard errors are clustered by PERMNO.
They do not account for common quarter shocks, so coefficient p-values are exploratory. No transcript text or
language features enter Model A. This is a research prototype, not a capacity-aware execution backtest.</p>
</body></html>"""


def main() -> None:
    cfg = load_config()
    df = read_table(cfg.paths.processed, "model_table_baseline")
    df["call_date"] = pd.to_datetime(df["call_date_et"], errors="coerce")
    df["year"] = df["call_date"].dt.year.astype("Int64").astype("string")
    targets = list(TARGETS.values())
    common = df.dropna(subset=targets + ["call_date"]).sort_values("call_date").copy()
    date_counts = common.groupby("call_date").size().sort_index()
    boundary = int(np.ceil(0.8 * len(common)))
    cutoff = date_counts.cumsum().ge(boundary).idxmax()
    # Purge training events whose 60-session target window could overlap the
    # first test call. This is the longest horizon and therefore applies to all.
    trading = read_table(cfg.paths.raw, "crsp_dsi", columns=["date"])["date"]
    trading = pd.DatetimeIndex(pd.to_datetime(trading, errors="coerce").dropna().unique()).sort_values()
    cutoff_pos = int(trading.searchsorted(pd.Timestamp(cutoff), side="left"))
    purge_pos = max(0, cutoff_pos - 60)
    purge_cutoff = trading[purge_pos]
    common["day0"] = pd.to_datetime(common["day0"], errors="coerce")
    train = common[(common["call_date"] < cutoff) & (common["day0"] < purge_cutoff)].copy()
    test = common[common["call_date"] >= cutoff].copy()
    purged = common[(common["call_date"] < cutoff) & (common["day0"] >= purge_cutoff)].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Chronological 80/20 split produced an empty side")

    seed = int(cfg["models"].get("seed", 7))
    metrics, prediction_rows = [], []
    ols_rows = []
    for display_target, target in TARGETS.items():
        y_train = train[target].astype(float)
        y_test = test[target].astype(float)
        ols_x = clean_model_frame(train)[CONTROLS].reset_index(drop=True).copy()
        for col in CONTROLS:
            ols_x[col] = ols_x[col].fillna(ols_x[col].median())
        ols_fit = sm.OLS(y_train.reset_index(drop=True), sm.add_constant(ols_x)).fit(
            cov_type="cluster", cov_kwds={"groups": train["permno"].astype(int).to_numpy()})
        for control in CONTROLS:
            ols_rows.append({"target": display_target, "control": control,
                             "coefficient": ols_fit.params[control],
                             "clustered_se": ols_fit.bse[control],
                             "t_stat": ols_fit.tvalues[control],
                             "p_value": ols_fit.pvalues[control], "n_train": len(train)})
        for model in MODELS:
            if model == "zero":
                pred = np.zeros(len(test))
            elif model == "train_mean":
                pred = np.full(len(test), y_train.mean())
            else:
                estimator = pipeline(model, train, seed)
                estimator.fit(clean_model_frame(train), y_train)
                pred = estimator.predict(clean_model_frame(test))
            row = pd.DataFrame({"keydevid": test["keydevid"].to_numpy(),
                                "permno": test["permno"].to_numpy(),
                                "call_date": test["call_date"].to_numpy(),
                                "target": display_target, "model": model,
                                "actual": y_test.to_numpy(), "prediction": pred})
            ic = row["actual"].corr(row["prediction"]) if row["prediction"].nunique() > 1 else np.nan
            ric = (spearmanr(row["actual"], row["prediction"], nan_policy="omit").statistic
                   if row["prediction"].nunique() > 1 else np.nan)
            sign_acc = (np.mean(np.sign(y_test) == np.sign(pred))
                        if np.any(np.asarray(pred) != 0) else np.nan)
            gross, net, n_q = qspread(row, "prediction", "actual")
            metrics.append({"target": display_target, "model": model, "n_train": len(train),
                            "n_test": len(test), "ic": ic, "rank_ic": ric,
                            "rmse": mean_squared_error(y_test, pred) ** 0.5,
                            "mae": mean_absolute_error(y_test, pred),
                            "r2": r2_score(y_test, pred), "sign_accuracy": sign_acc,
                            "q5_q1_gross": gross, "q5_q1_net_20bp": net, "test_quarters": n_q})
            prediction_rows.append(row)
            print(f"{display_target} | {model}: IC={ic:.3f}, rankIC={ric:.3f}, RMSE={metrics[-1]['rmse']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    out = Path("outputs/model_a_80_20")
    out.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    # Row-level forecasts derive from licensed WRDS inputs; keep them local.
    predictions.to_parquet(cfg.paths.processed / "model_a_predictions.parquet", index=False)
    split = {"split": "chronological 80/20 with 60-trading-day label purge",
             "cutoff_date": cutoff.date().isoformat(), "purge_day0_before": purge_cutoff.date().isoformat(),
             "train_n": len(train), "test_n": len(test), "purged_n": len(purged), "common_n": len(common),
             "train_start": train.call_date.min().date().isoformat(),
             "train_end": train.call_date.max().date().isoformat(),
             "test_start": test.call_date.min().date().isoformat(),
             "test_end": test.call_date.max().date().isoformat(),
             "targets": TARGETS, "controls": CONTROLS, "models": MODELS,
             "qspread_cost_assumption": NET_COST}
    (out / "split.json").write_text(json.dumps(split, indent=2))
    events_by_year = (common.groupby(common.call_date.dt.year).size().rename("events")
                      .rename_axis("year").reset_index())
    missing_controls = (common[CONTROLS].isna().mean().rename("missing_share")
                        .rename_axis("control").reset_index())
    target_summary = pd.DataFrame([{"target": label, "mean": common[col].mean(),
                                    "std": common[col].std(), "p05": common[col].quantile(.05),
                                    "median": common[col].median(), "p95": common[col].quantile(.95)}
                                   for label, col in TARGETS.items()])
    corr_rows = []
    for label, target in TARGETS.items():
        for control in CONTROLS:
            corr_rows.append({"target": label, "control": control,
                              "pearson_r": common[control].corr(common[target])})
    control_target_corr = pd.DataFrame(corr_rows)
    eda = {"events_by_year": events_by_year, "missing_controls": missing_controls,
           "target_summary": target_summary, "control_target_corr": control_target_corr}
    events_by_year.to_csv(out / "events_by_year.csv", index=False)
    missing_controls.to_csv(out / "control_missingness.csv", index=False)
    target_summary.to_csv(out / "target_summary.csv", index=False)
    control_target_corr.to_csv(out / "control_target_correlations.csv", index=False)
    ols_controls = pd.DataFrame(ols_rows)
    eda["ols_controls"] = ols_controls
    ols_controls.to_csv(out / "ols_controls_clustered.csv", index=False)
    (out / "dashboard.html").write_text(html_dashboard(metrics_df, eda, split))
    print(f"Wrote dashboard and results to {out}")


if __name__ == "__main__":
    main()
