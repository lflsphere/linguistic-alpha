"""
Model A Baseline v2: Proper preprocessing, full non-linguistic feature set, feature selection.
Predicts CAR[+2,+20] using financial controls + call structure — no linguistic/sentiment features.

Preprocessing pipeline:
  1. Winsorize outliers (1st/99th percentile)
  2. Log-transform skewed features
  3. Impute missing with median (per fold, no leakage)
  4. Multicollinearity check (VIF)
  5. Feature selection (mutual information, univariate, RF importance)
  6. StandardScaler per fold

Usage:
    python3 model_a_baseline.py [--folder ~/earnings_sp500]
"""
import argparse, os, json, warnings, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRIMARY_TARGET = "car_2_20"

# All non-linguistic numeric features available in model_table
FINANCIAL_CONTROLS = [
    "sue", "sue_price", "surprise_pct",
    "size", "bm", "mom", "vol_pre",
    "n_analysts", "dispersion",
    "car_0_1",
]

CALL_STRUCTURE = [
    "n_questions", "n_answers",
    "words_prepared", "words_answers", "words_questions",
    "avg_answer_words", "share_answers_by_execs",
    "prior_calls_with_target",
]

ALL_FEATURES = FINANCIAL_CONTROLS + CALL_STRUCTURE
SECTOR_COL = "sector"
INDUSTRY_COL = "gind"

# Features to log-transform (right-skewed, strictly positive)
LOG_TRANSFORM = ["words_prepared", "words_answers", "words_questions", "avg_answer_words", "n_analysts"]

# Winsorize bounds
WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99

CV_FOLDS = 5
MIN_FOLD_SIZE = 100
RANDOM_STATE = 42
VIF_THRESHOLD = 10.0

def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(base):
    df = pd.read_parquet(f"{base}/model_table.parquet")
    log_msg(f"loaded {len(df):,} rows, {df.shape[1]} columns")
    log_msg(f"columns: {', '.join(df.columns)}")
    train = df[df["sample"] == "train"].copy()
    log_msg(f"train: {len(train):,} rows ({train['call_date'].min().date()} to {train['call_date'].max().date()})")
    return df, train


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
def winsorize(series, low=WINSOR_LOW, high=WINSOR_HIGH):
    """Clip values to percentile bounds."""
    lo_val = series.quantile(low)
    hi_val = series.quantile(high)
    return series.clip(lo_val, hi_val)


def preprocess_df(df, features, fit_stats=None):
    """
    Winsorize, log-transform, return processed df and fit statistics.
    If fit_stats is provided, use those bounds (for test set).
    """
    out = df[features].copy()
    new_stats = {}

    for col in features:
        if col not in out.columns:
            continue

        # Winsorize
        if fit_stats is None:
            lo = out[col].quantile(WINSOR_LOW)
            hi = out[col].quantile(WINSOR_HIGH)
            new_stats[f"{col}_lo"] = lo
            new_stats[f"{col}_hi"] = hi
        else:
            lo = fit_stats[f"{col}_lo"]
            hi = fit_stats[f"{col}_hi"]
        out[col] = out[col].clip(lo, hi)

        # Log-transform right-skewed features
        if col in LOG_TRANSFORM:
            out[col] = np.log1p(out[col].clip(lower=0))

        # Median impute
        if fit_stats is None:
            med = out[col].median()
            new_stats[f"{col}_median"] = med
        else:
            med = fit_stats[f"{col}_median"]
        out[col] = out[col].fillna(med)

    return out, (new_stats if fit_stats is None else fit_stats)


def check_multicollinearity(X, names):
    """Compute VIF for each feature. Flag pairs with high collinearity."""
    from numpy.linalg import lstsq
    log_msg("\n--- MULTICOLLINEARITY CHECK (VIF) ---")
    vifs = []
    X_arr = X.copy()
    for i in range(X_arr.shape[1]):
        others = np.delete(X_arr, i, axis=1)
        y_i = X_arr[:, i]
        coef, _, _, _ = lstsq(others, y_i, rcond=None)
        y_pred = others @ coef
        ss_res = np.sum((y_i - y_pred) ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vifs.append(vif)
        flag = " *** HIGH" if vif > VIF_THRESHOLD else ""
        log_msg(f"  {names[i]:25s}  VIF={vif:8.2f}{flag}")

    # Also show top correlated pairs
    corr = np.corrcoef(X_arr.T)
    log_msg("\n  Top correlated feature pairs:")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((abs(corr[i, j]), names[i], names[j], corr[i, j]))
    pairs.sort(reverse=True)
    for absv, n1, n2, v in pairs[:10]:
        log_msg(f"    {n1:25s} x {n2:25s}  r={v:+.3f}")

    return vifs


# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------
def feature_selection(X, y, feature_names):
    """
    Three-method feature selection:
    1. Univariate (Spearman correlation significance)
    2. Mutual information regression
    3. Random Forest importance
    Returns ranked features and selected subset.
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor

    log_msg("\n--- FEATURE SELECTION ---")
    n_features = X.shape[1]
    scores = pd.DataFrame({"feature": feature_names})

    # 1. Univariate Spearman
    spearman_r = []
    spearman_p = []
    for i in range(n_features):
        r, p = stats.spearmanr(X[:, i], y)
        spearman_r.append(abs(r))
        spearman_p.append(p)
    scores["spearman_abs"] = spearman_r
    scores["spearman_p"] = spearman_p
    scores["spearman_rank"] = scores["spearman_abs"].rank(ascending=False)

    # 2. Mutual information
    mi = mutual_info_regression(X, y, random_state=RANDOM_STATE, n_neighbors=10)
    scores["mi"] = mi
    scores["mi_rank"] = scores["mi"].rank(ascending=False)

    # 3. Random Forest importance
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    scores["rf_importance"] = rf.feature_importances_
    scores["rf_rank"] = scores["rf_importance"].rank(ascending=False)

    # Average rank across methods
    scores["avg_rank"] = (scores["spearman_rank"] + scores["mi_rank"] + scores["rf_rank"]) / 3
    scores = scores.sort_values("avg_rank")

    log_msg(f"{'Feature':25s} {'Spearman |r|':>13s} {'p-value':>10s} {'MI':>8s} {'RF imp':>8s} {'Avg rank':>9s}")
    for _, row in scores.iterrows():
        sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
        log_msg(f"  {row['feature']:25s} {row['spearman_abs']:10.4f} {sig:3s} {row['spearman_p']:10.2e} "
                f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:9.1f}")

    # Select features: keep those in top half by avg rank, OR significant spearman (p < 0.05)
    n_keep = max(n_features // 2, 5)
    top_by_rank = set(scores.head(n_keep)["feature"])
    significant = set(scores[scores["spearman_p"] < 0.05]["feature"])
    selected = sorted(top_by_rank | significant, key=lambda f: scores.set_index("feature").loc[f, "avg_rank"])

    log_msg(f"\nSelected {len(selected)} features (top {n_keep} by avg rank + significant univariates):")
    log_msg(f"  {', '.join(selected)}")

    return scores, selected


# ---------------------------------------------------------------------------
# 4. Time-series CV and model evaluation
# ---------------------------------------------------------------------------
def time_series_cv_splits(dates, n_splits=CV_FOLDS):
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
        if len(test_idx) >= MIN_FOLD_SIZE:
            splits.append((train_idx, test_idx))
    return splits


def evaluate_predictions(y_true, y_pred):
    ic = stats.pearsonr(y_pred, y_true)[0] if len(y_true) > 2 else np.nan
    rank_ic = stats.spearmanr(y_pred, y_true)[0] if len(y_true) > 2 else np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    accuracy = np.mean(direction_true == direction_pred)

    q = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    quintile_means = pd.Series(y_true).groupby(q).mean()
    top_bottom = quintile_means.iloc[-1] - quintile_means.iloc[0] if len(quintile_means) >= 2 else np.nan

    return {
        "ic": ic, "rank_ic": rank_ic, "r2": r2, "rmse": rmse,
        "direction_accuracy": accuracy, "quintile_spread": top_bottom,
    }


def _clone_model(model):
    from sklearn.base import clone
    return clone(model)


def build_sector_dummies(sector_values, all_sectors):
    """One-hot encode sector, dropping first for dummy trap."""
    sector_to_idx = {s: i for i, s in enumerate(all_sectors[1:])}
    d = np.zeros((len(sector_values), len(sector_to_idx)))
    for i, s in enumerate(sector_values):
        if s in sector_to_idx:
            d[i, sector_to_idx[s]] = 1
    return d, [f"sec_{s[:8]}" for s in all_sectors[1:]]


def run_models(X_numeric, feature_names, sector, dates, y, all_sectors):
    """Train and evaluate models with per-fold preprocessing."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    splits = time_series_cv_splits(dates, CV_FOLDS)
    log_msg(f"\n--- MODEL COMPARISON (time-series CV, {len(splits)} folds) ---")
    for i, (tr, te) in enumerate(splits):
        log_msg(f"  fold {i+1}: train={len(tr):,} (to {pd.Timestamp(dates[tr[-1]]).date()}), "
                f"test={len(te):,} (to {pd.Timestamp(dates[te[-1]]).date()})")

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

    for name, model in models.items():
        log_msg(f"\n{'='*60}")
        log_msg(f"Training: {name}")
        fold_metrics = []

        for i, (train_idx, test_idx) in enumerate(splits):
            # Per-fold preprocessing (fit on train, transform test)
            X_tr_raw = pd.DataFrame(X_numeric[train_idx], columns=feature_names)
            X_te_raw = pd.DataFrame(X_numeric[test_idx], columns=feature_names)
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_proc, fit_stats = preprocess_df(X_tr_raw, feature_names)
            X_te_proc, _ = preprocess_df(X_te_raw, feature_names, fit_stats=fit_stats)

            # Add sector dummies
            sec_tr, sec_names = build_sector_dummies(sector[train_idx], all_sectors)
            sec_te, _ = build_sector_dummies(sector[test_idx], all_sectors)

            X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
            X_te_full = np.hstack([X_te_proc.values, sec_te])

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_full)
            X_te_s = scaler.transform(X_te_full)

            model_clone = _clone_model(model)
            model_clone.fit(X_tr_s, y_tr)
            preds = model_clone.predict(X_te_s)

            metrics = evaluate_predictions(y_te, preds)
            fold_metrics.append(metrics)
            log_msg(f"  fold {i+1}: IC={metrics['ic']:.4f}  rank_IC={metrics['rank_ic']:.4f}  "
                    f"R²={metrics['r2']:.4f}  dir_acc={metrics['direction_accuracy']:.3f}  "
                    f"Q5-Q1={metrics['quintile_spread']:.4f}")

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        log_msg(f"\n  AVERAGE: IC={avg['ic']:.4f}±{std['ic']:.4f}  rank_IC={avg['rank_ic']:.4f}±{std['rank_ic']:.4f}  "
                f"R²={avg['r2']:.4f}  dir_acc={avg['direction_accuracy']:.3f}  Q5-Q1={avg['quintile_spread']:.4f}")

        all_results[name] = {"avg": avg, "std": std, "folds": fold_metrics}

        # Feature importance for last fold's tree model
        if hasattr(model_clone, "feature_importances_"):
            all_names = feature_names + sec_names
            imp = pd.Series(model_clone.feature_importances_, index=all_names)
            imp = imp.sort_values(ascending=False).head(15)
            log_msg(f"  top features: {', '.join(f'{n}={v:.3f}' for n, v in imp.items())}")

    return all_results


def holdout_report(df, best_model_name, X_raw, y_train, sector_train, feature_names, all_sectors):
    """Train best model on full train, evaluate on holdout once."""
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
    for c in feature_names:
        if c in holdout.columns:
            pass  # we'll impute missing
    holdout = holdout[mask]
    log_msg(f"\nholdout: {len(holdout):,} rows")

    # Preprocess train
    X_tr_df = pd.DataFrame(X_raw, columns=feature_names)
    X_tr_proc, fit_stats = preprocess_df(X_tr_df, feature_names)

    # Preprocess holdout with train stats
    X_ho_df = holdout[feature_names].copy()
    X_ho_proc, _ = preprocess_df(X_ho_df, feature_names, fit_stats=fit_stats)
    y_ho = holdout[PRIMARY_TARGET].values

    # Sector dummies
    sec_tr, sec_names = build_sector_dummies(sector_train, all_sectors)
    sec_ho, _ = build_sector_dummies(holdout[SECTOR_COL].values, all_sectors)

    X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
    X_ho_full = np.hstack([X_ho_proc.values, sec_ho])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_ho_s = scaler.transform(X_ho_full)

    model.fit(X_tr_s, y_train)
    preds = model.predict(X_ho_s)

    metrics = evaluate_predictions(y_ho, preds)
    log_msg(f"\n{'='*60}")
    log_msg(f"HOLDOUT REPORT ({best_model_name} trained on full train set):")
    for k, v in metrics.items():
        log_msg(f"  {k:25s}: {v:+.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def write_report(base, feat_scores, selected_features, model_results, holdout_metrics, best_model, vifs, feature_names):
    out = Path(base) / "model_a_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL A BASELINE REPORT v2 — Full Non-Linguistic Features + Preprocessing")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Target: {PRIMARY_TARGET}")
    lines.append(f"All candidate features ({len(feature_names)}): {', '.join(feature_names)}")
    lines.append(f"Selected features ({len(selected_features)}): {', '.join(selected_features)}")
    lines.append(f"Preprocessing: winsorize 1%/99%, log1p({', '.join(LOG_TRANSFORM)}), median impute, StandardScaler")
    lines.append("=" * 70)

    lines.append("\n\n--- FEATURE SELECTION RANKING ---")
    lines.append(f"{'Feature':25s} {'|Spearman|':>11s} {'p-value':>10s} {'MI':>8s} {'RF imp':>8s} {'Avg rank':>9s} {'Selected':>9s}")
    for _, row in feat_scores.iterrows():
        sel = "YES" if row["feature"] in selected_features else ""
        lines.append(f"{row['feature']:25s} {row['spearman_abs']:11.4f} {row['spearman_p']:10.2e} "
                     f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:9.1f} {sel:>9s}")

    lines.append("\n\n--- CROSS-VALIDATION RESULTS ---")
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

    lines.append("\n\n--- PREPROCESSING NOTES ---")
    lines.append("- Winsorization at 1st/99th percentile (fit on train per fold)")
    lines.append(f"- Log1p transform on: {', '.join(LOG_TRANSFORM)}")
    lines.append("- Missing values imputed with train-fold median")
    lines.append("- Features standardized (zero mean, unit variance) per fold")
    lines.append("- Sector encoded as one-hot dummies (drop first)")
    lines.append("- Feature selection: top half by avg rank of (Spearman, MI, RF importance) + significant univariates")
    lines.append("- Time-series expanding window CV, no future leakage")
    lines.append("- Holdout (2023-2024) evaluated once, not used for model selection")
    lines.append("- This is Model A (no linguistic features). Model B adds tone, hedging, etc.")

    report_text = "\n".join(lines)
    out.write_text(report_text)
    log_msg(f"\nreport saved to {out}")
    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = parser.parse_args()
    base = args.folder

    log_msg("=" * 60)
    log_msg("MODEL A BASELINE v2 — Preprocessing + Feature Selection")
    log_msg("=" * 60)

    # Load
    df, train = load_data(base)

    # Check which features exist
    available = [f for f in ALL_FEATURES if f in train.columns]
    missing_cols = [f for f in ALL_FEATURES if f not in train.columns]
    if missing_cols:
        log_msg(f"features not in data (skipping): {missing_cols}")
    log_msg(f"candidate features ({len(available)}): {', '.join(available)}")

    # Keep rows with target (allow missing features — we'll impute)
    train_with_target = train[train[PRIMARY_TARGET].notna()].copy()
    log_msg(f"rows with target: {len(train_with_target):,}")

    X_raw = train_with_target[available].values
    y = train_with_target[PRIMARY_TARGET].values
    sector = train_with_target[SECTOR_COL].values
    dates = train_with_target["call_date"].values
    all_sectors = sorted(df[SECTOR_COL].dropna().unique())

    # Step 1: Preprocess for feature analysis (full train)
    log_msg("\n--- PREPROCESSING ---")
    X_df = pd.DataFrame(X_raw, columns=available)
    X_proc, fit_stats = preprocess_df(X_df, available)
    log_msg(f"after preprocessing: {X_proc.shape[0]:,} rows, {X_proc.shape[1]} features")

    # Show distributions before/after
    log_msg("\nfeature distributions (after winsorize + log):")
    for col in available:
        s = X_proc[col]
        log_msg(f"  {col:25s}  mean={s.mean():+8.4f}  std={s.std():8.4f}  "
                f"skew={s.skew():+6.2f}  missing={X_df[col].isna().sum()}")

    # Step 2: Multicollinearity check
    vifs = check_multicollinearity(X_proc.values, available)

    # Step 3: Feature selection
    feat_scores, selected_features = feature_selection(X_proc.values, y, available)

    # Step 4: Run models with SELECTED features only
    selected_idx = [available.index(f) for f in selected_features]
    X_selected = X_raw[:, selected_idx]

    model_results = run_models(X_selected, selected_features, sector, dates, y, all_sectors)

    # Pick best
    best_model = max(model_results, key=lambda k: model_results[k]["avg"]["rank_ic"])
    best_res = model_results[best_model]
    log_msg(f"\n>>> Best model: {best_model} "
            f"(IC={best_res['avg']['ic']:.4f}, Rank IC={best_res['avg']['rank_ic']:.4f})")

    # Step 5: Holdout
    holdout_metrics = holdout_report(df, best_model, X_selected, y, sector, selected_features, all_sectors)

    # Step 6: Report
    report = write_report(base, feat_scores, selected_features, model_results,
                         holdout_metrics, best_model, vifs, available)
    print("\n" + report)

    # Save JSON
    json_out = Path(base) / "model_a_results.json"
    save = {
        "version": "v2",
        "best_model": best_model,
        "all_candidate_features": available,
        "selected_features": selected_features,
        "preprocessing": {
            "winsorize": f"{WINSOR_LOW}/{WINSOR_HIGH}",
            "log_transform": LOG_TRANSFORM,
            "impute": "median (per fold)",
            "scale": "StandardScaler (per fold)",
        },
        "target": PRIMARY_TARGET,
        "cv_results": {name: {
            "avg": {k: float(v) for k, v in r["avg"].items()},
            "std": {k: float(v) for k, v in r["std"].items()},
        } for name, r in model_results.items()},
        "holdout": {k: float(v) for k, v in holdout_metrics.items()},
        "feature_selection": feat_scores[["feature", "spearman_abs", "mi", "rf_importance", "avg_rank"]].to_dict("records"),
        "timestamp": datetime.now().isoformat(),
    }
    json_out.write_text(json.dumps(save, indent=2))
    log_msg(f"results JSON saved to {json_out}")
    log_msg("done")


if __name__ == "__main__":
    main()
