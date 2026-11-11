"""
Model A Baseline: Feature selection and model comparison on numeric controls only.
Predicts CAR[+2,+20] (post-earnings drift) using only financial controls — no linguistic features.
Run on the GCP VM where model_table.parquet lives.

Usage:
    python3 model_a_baseline.py [--folder ~/earnings_sp500]
"""
import argparse, os, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRIMARY_TARGET = "car_2_20"
SECONDARY_TARGETS = ["car_0_1", "car_2_5", "car_2_60"]

CONTROL_FEATURES = [
    "sue", "sue_price", "size", "bm", "mom", "vol_pre",
    "n_analysts", "dispersion", "car_0_1",
]

SECTOR_COL = "sector"
CV_FOLDS = 5
RANDOM_STATE = 42

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
def load_data(base):
    df = pd.read_parquet(f"{base}/model_table.parquet")
    log(f"loaded {len(df):,} rows, {df.shape[1]} columns")
    train = df[df["sample"] == "train"].copy()
    log(f"train sample: {len(train):,} rows ({train['call_date'].min().date()} to {train['call_date'].max().date()})")
    return df, train


def prepare_features(train):
    """Prepare feature matrix X and target y from training data."""
    # Only keep rows with primary target and all core controls
    cols_needed = CONTROL_FEATURES + [PRIMARY_TARGET, SECTOR_COL]
    mask = train[PRIMARY_TARGET].notna()
    for c in CONTROL_FEATURES:
        if c in train.columns:
            mask &= train[c].notna()
    clean = train[mask].copy()
    log(f"clean rows (no missing in controls + target): {len(clean):,} / {len(train):,}")

    X = clean[CONTROL_FEATURES].copy()
    y = clean[PRIMARY_TARGET].values
    sector = clean[SECTOR_COL].values
    meta = clean[["keydevid", "call_date", "permno", SECTOR_COL]].copy()

    return X, y, sector, meta, clean


# ---------------------------------------------------------------------------
# 2. Feature analysis
# ---------------------------------------------------------------------------
def feature_analysis(X, y, sector):
    """Correlation, rank correlation, and per-sector breakdown."""
    results = {}

    # Pearson and Spearman correlations with target
    corrs = []
    for col in X.columns:
        r_p, p_p = stats.pearsonr(X[col], y)
        r_s, p_s = stats.spearmanr(X[col], y)
        corrs.append({
            "feature": col,
            "pearson_r": r_p, "pearson_p": p_p,
            "spearman_r": r_s, "spearman_p": p_s,
            "abs_spearman": abs(r_s),
        })
    corr_df = pd.DataFrame(corrs).sort_values("abs_spearman", ascending=False)
    results["correlations"] = corr_df
    log("feature correlations with CAR[2,20]:")
    for _, row in corr_df.iterrows():
        sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
        log(f"  {row['feature']:20s}  pearson={row['pearson_r']:+.4f}  spearman={row['spearman_r']:+.4f} {sig}")

    # Feature-feature correlation matrix
    feat_corr = X.corr(method="spearman")
    results["feature_corr_matrix"] = feat_corr

    # Per-sector mean target
    sector_means = pd.DataFrame({"sector": sector, "target": y}).groupby("sector")["target"].agg(["mean", "std", "count"])
    results["sector_means"] = sector_means
    log("\ntarget by sector:")
    for s, row in sector_means.sort_values("mean", ascending=False).iterrows():
        log(f"  {s:30s}  mean={row['mean']:+.4f}  std={row['std']:.4f}  n={int(row['count']):,}")

    return results


# ---------------------------------------------------------------------------
# 3. Model training with time-series cross-validation
# ---------------------------------------------------------------------------
def time_series_cv_splits(dates, n_splits=CV_FOLDS):
    """Expanding window CV: train on earlier data, test on next chunk."""
    sorted_dates = np.sort(dates)
    n = len(sorted_dates)
    chunk = n // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = chunk * (i + 2)
        test_start = train_end
        test_end = min(train_end + chunk, n)
        if test_end <= test_start:
            continue
        cutoff = sorted_dates[train_end - 1]
        test_cutoff = sorted_dates[min(test_end - 1, n - 1)]
        train_idx = np.where(dates <= cutoff)[0]
        test_idx = np.where((dates > cutoff) & (dates <= test_cutoff))[0]
        if len(test_idx) >= 100:
            splits.append((train_idx, test_idx))
    return splits


def evaluate_predictions(y_true, y_pred):
    """Compute regression and classification metrics."""
    # Regression
    ic = stats.pearsonr(y_pred, y_true)[0] if len(y_true) > 2 else np.nan
    rank_ic = stats.spearmanr(y_pred, y_true)[0] if len(y_true) > 2 else np.nan
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Classification (direction)
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    accuracy = np.mean(direction_true == direction_pred)

    # Quintile spread
    q = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    quintile_means = pd.Series(y_true).groupby(q).mean()
    top_bottom = quintile_means.iloc[-1] - quintile_means.iloc[0] if len(quintile_means) >= 2 else np.nan

    return {
        "ic": ic, "rank_ic": rank_ic, "r2": r2, "rmse": rmse,
        "direction_accuracy": accuracy, "quintile_spread": top_bottom,
    }


def run_models(X, y, dates, sector):
    """Train and evaluate multiple models using time-series CV."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False
        log("xgboost not installed, skipping XGBRegressor")

    # Add sector dummies
    sector_dummies = pd.get_dummies(pd.Series(sector), prefix="sec", drop_first=True).values
    X_with_sector = np.hstack([X.values, sector_dummies])
    feature_names = list(X.columns) + [f"sec_{i}" for i in range(sector_dummies.shape[1])]

    splits = time_series_cv_splits(dates, CV_FOLDS)
    log(f"\ntime-series CV: {len(splits)} folds")
    for i, (tr, te) in enumerate(splits):
        log(f"  fold {i+1}: train={len(tr):,} (to {pd.Timestamp(dates[tr[-1]]).date()}), test={len(te):,} (to {pd.Timestamp(dates[te[-1]]).date()})")

    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=50, subsample=0.8, random_state=RANDOM_STATE),
    }
    if has_xgb:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    all_results = {}
    all_predictions = {}

    for name, model in models.items():
        log(f"\n{'='*60}")
        log(f"Training: {name}")
        fold_metrics = []
        oof_preds = np.full(len(y), np.nan)

        for i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X_with_sector[train_idx], X_with_sector[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model_clone = _clone_model(model)
            model_clone.fit(X_tr_s, y_tr)
            preds = model_clone.predict(X_te_s)

            metrics = evaluate_predictions(y_te, preds)
            fold_metrics.append(metrics)
            oof_preds[test_idx] = preds

            log(f"  fold {i+1}: IC={metrics['ic']:.4f}  rank_IC={metrics['rank_ic']:.4f}  "
                f"R²={metrics['r2']:.4f}  dir_acc={metrics['direction_accuracy']:.3f}")

        # Average across folds
        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}

        log(f"\n  AVERAGE: IC={avg['ic']:.4f}±{std['ic']:.4f}  rank_IC={avg['rank_ic']:.4f}±{std['rank_ic']:.4f}  "
            f"R²={avg['r2']:.4f}  dir_acc={avg['direction_accuracy']:.3f}  Q5-Q1={avg['quintile_spread']:.4f}")

        all_results[name] = {"avg": avg, "std": std, "folds": fold_metrics}
        all_predictions[name] = oof_preds

        # Feature importance for tree models
        if hasattr(model_clone, "feature_importances_"):
            imp = pd.Series(model_clone.feature_importances_, index=feature_names)
            imp = imp.sort_values(ascending=False).head(15)
            log(f"  top features: {', '.join(f'{n}={v:.3f}' for n, v in imp.items())}")

    return all_results, all_predictions, feature_names


def _clone_model(model):
    """Clone a sklearn model with same params."""
    from sklearn.base import clone
    return clone(model)


# ---------------------------------------------------------------------------
# 4. Final model on full train, predict holdout (report only, don't optimize on it)
# ---------------------------------------------------------------------------
def final_train_and_holdout_report(df, best_model_name, X_train, y_train, sector_train, dates_train):
    """Train best model on full train set, report holdout metrics once."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    model_map = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=50, subsample=0.8, random_state=RANDOM_STATE),
    }
    if has_xgb:
        model_map["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    model = model_map[best_model_name]

    # Prepare holdout
    holdout = df[df["sample"] == "holdout"].copy()
    mask = holdout[PRIMARY_TARGET].notna()
    for c in CONTROL_FEATURES:
        if c in holdout.columns:
            mask &= holdout[c].notna()
    holdout = holdout[mask]
    log(f"\nholdout: {len(holdout):,} rows")

    # Build sector dummies consistent with train
    all_sectors = sorted(df[SECTOR_COL].dropna().unique())
    sector_to_idx = {s: i for i, s in enumerate(all_sectors[1:])}  # drop_first

    def make_sector_dummies(sectors):
        d = np.zeros((len(sectors), len(sector_to_idx)))
        for i, s in enumerate(sectors):
            if s in sector_to_idx:
                d[i, sector_to_idx[s]] = 1
        return d

    X_tr = np.hstack([X_train.values, make_sector_dummies(sector_train)])
    X_ho = np.hstack([holdout[CONTROL_FEATURES].values, make_sector_dummies(holdout[SECTOR_COL].values)])
    y_ho = holdout[PRIMARY_TARGET].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ho_s = scaler.transform(X_ho)

    model.fit(X_tr_s, y_train)
    preds = model.predict(X_ho_s)

    metrics = evaluate_predictions(y_ho, preds)
    log(f"\n{'='*60}")
    log(f"HOLDOUT REPORT ({best_model_name} trained on full train set):")
    log(f"  IC:         {metrics['ic']:.4f}")
    log(f"  Rank IC:    {metrics['rank_ic']:.4f}")
    log(f"  R²:         {metrics['r2']:.4f}")
    log(f"  RMSE:       {metrics['rmse']:.4f}")
    log(f"  Dir Acc:    {metrics['direction_accuracy']:.3f}")
    log(f"  Q5-Q1:      {metrics['quintile_spread']:.4f}")

    return metrics, preds


# ---------------------------------------------------------------------------
# 5. Summary report
# ---------------------------------------------------------------------------
def write_report(base, feat_results, model_results, holdout_metrics, best_model):
    """Write a comprehensive text report."""
    out = Path(base) / "model_a_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL A BASELINE REPORT — Numeric Controls Only")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Target: {PRIMARY_TARGET}")
    lines.append(f"Features: {', '.join(CONTROL_FEATURES)} + sector dummies")
    lines.append("=" * 70)

    lines.append("\n\n--- FEATURE CORRELATIONS WITH TARGET ---")
    corr_df = feat_results["correlations"]
    lines.append(f"{'Feature':20s} {'Pearson r':>10s} {'Spearman r':>11s} {'p-value':>10s}")
    for _, row in corr_df.iterrows():
        lines.append(f"{row['feature']:20s} {row['pearson_r']:+10.4f} {row['spearman_r']:+11.4f} {row['spearman_p']:10.2e}")

    lines.append("\n\n--- CROSS-VALIDATION RESULTS (time-series expanding window) ---")
    lines.append(f"{'Model':20s} {'IC':>8s} {'±':>6s} {'Rank IC':>8s} {'±':>6s} {'R²':>8s} {'Dir Acc':>8s} {'Q5-Q1':>8s}")
    ranked = sorted(model_results.items(), key=lambda x: x[1]["avg"]["rank_ic"], reverse=True)
    for name, res in ranked:
        a, s = res["avg"], res["std"]
        lines.append(f"{name:20s} {a['ic']:+8.4f} {s['ic']:6.4f} {a['rank_ic']:+8.4f} {s['rank_ic']:6.4f} "
                      f"{a['r2']:+8.4f} {a['direction_accuracy']:8.3f} {a['quintile_spread']:+8.4f}")

    lines.append(f"\nBest model by avg Rank IC: {best_model}")

    if holdout_metrics:
        lines.append(f"\n\n--- HOLDOUT RESULTS ({best_model}) ---")
        for k, v in holdout_metrics.items():
            lines.append(f"  {k:25s}: {v:+.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    lines.append("\n\n--- NOTES ---")
    lines.append("- car_0_1 (announcement return) is included as a feature since it is known before the [+2,+20] window")
    lines.append("- Sector is encoded as one-hot dummies (drop first)")
    lines.append("- All features standardized (zero mean, unit variance) per fold")
    lines.append("- Time-series CV: expanding window, no future data leaks into training")
    lines.append("- Holdout (2023-2024) evaluated once, not used for model selection")
    lines.append("- This is Model A (controls only). Model B will add linguistic features.")

    report_text = "\n".join(lines)
    out.write_text(report_text)
    log(f"\nreport saved to {out}")
    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = parser.parse_args()
    base = args.folder

    log("=" * 60)
    log("MODEL A BASELINE — Feature Selection & Model Comparison")
    log("=" * 60)

    # Load
    df, train = load_data(base)

    # Prepare
    X, y, sector, meta, clean = prepare_features(train)
    dates = clean["call_date"].values

    # Feature analysis
    log("\n--- FEATURE ANALYSIS ---")
    feat_results = feature_analysis(X, y, sector)

    # Model comparison
    log("\n--- MODEL COMPARISON (time-series CV) ---")
    model_results, predictions, feature_names = run_models(X, y, dates, sector)

    # Pick best by average rank IC
    best_model = max(model_results, key=lambda k: model_results[k]["avg"]["rank_ic"])
    log(f"\n>>> Best model by Rank IC: {best_model} "
        f"(IC={model_results[best_model]['avg']['ic']:.4f}, "
        f"Rank IC={model_results[best_model]['avg']['rank_ic']:.4f})")

    # Holdout report (one-time look)
    holdout_metrics, holdout_preds = final_train_and_holdout_report(
        df, best_model, X, y, sector, dates)

    # Save report
    report = write_report(base, feat_results, model_results, holdout_metrics, best_model)
    print("\n" + report)

    # Save results as JSON
    json_out = Path(base) / "model_a_results.json"
    save = {
        "best_model": best_model,
        "features": CONTROL_FEATURES,
        "target": PRIMARY_TARGET,
        "cv_results": {name: {
            "avg": {k: float(v) for k, v in r["avg"].items()},
            "std": {k: float(v) for k, v in r["std"].items()},
        } for name, r in model_results.items()},
        "holdout": {k: float(v) for k, v in holdout_metrics.items()},
        "timestamp": datetime.now().isoformat(),
    }
    json_out.write_text(json.dumps(save, indent=2))
    log(f"results JSON saved to {json_out}")
    log("done")


if __name__ == "__main__":
    main()
