"""
Model B: Model A controls + Loughran-McDonald linguistic features.
Predicts CAR[+2,+20] using financial controls, call structure, AND tone/hedging features.

Train: 2010–2021 (expanding-window CV inside for model selection)
Test:  2022–2024 (one-time evaluation of best CV model)

Usage:
    python3 model_b_linguistic.py [--folder ~/earnings_sp500]
"""
import argparse, os, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"

# Model A's selected controls (11 features from feature selection)
MODEL_A_FEATURES = [
    "dispersion", "mom", "prior_calls_with_target", "vol_pre",
    "car_0_1", "size", "sue_price", "sue", "bm",
    "words_answers", "n_answers",
]

# New linguistic features from LM dictionary
LINGUISTIC_FEATURES = [
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap",
    "tone_prepared_chg", "tone_answers_chg", "hedging_answers_chg",
]

ALL_CANDIDATE_FEATURES = MODEL_A_FEATURES + LINGUISTIC_FEATURES

SECTOR_COL = "sector"

# Features to log-transform (same as Model A, only applies to controls)
LOG_TRANSFORM = ["words_answers"]

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99
TRAIN_END = "2021-12-31"
CV_FOLDS = 5
MIN_FOLD_SIZE = 100
RANDOM_STATE = 42

def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def load_data(base):
    df = pd.read_parquet(f"{base}/model_table.parquet")
    log_msg(f"loaded {len(df):,} rows, {df.shape[1]} columns")
    ling_present = [c for c in LINGUISTIC_FEATURES if c in df.columns]
    ling_missing = [c for c in LINGUISTIC_FEATURES if c not in df.columns]
    log_msg(f"linguistic columns present: {ling_present}")
    if ling_missing:
        log_msg(f"WARNING — linguistic columns missing: {ling_missing}")
    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy()
    test = has_target[has_target["call_date"] > cutoff].copy()
    log_msg(f"train: {len(train):,} rows ({train['call_date'].min().date()} to {train['call_date'].max().date()})")
    log_msg(f"test:  {len(test):,} rows ({test['call_date'].min().date()} to {test['call_date'].max().date()})")
    return df, train, test


def preprocess_df(df, features, fit_stats=None):
    out = df[features].copy()
    new_stats = {}
    for col in features:
        if col not in out.columns:
            continue
        if fit_stats is None:
            lo = out[col].quantile(WINSOR_LOW)
            hi = out[col].quantile(WINSOR_HIGH)
            new_stats[f"{col}_lo"] = lo
            new_stats[f"{col}_hi"] = hi
        else:
            lo = fit_stats[f"{col}_lo"]
            hi = fit_stats[f"{col}_hi"]
        out[col] = out[col].clip(lo, hi)
        if col in LOG_TRANSFORM:
            out[col] = np.log1p(out[col].clip(lower=0))
        if fit_stats is None:
            med = out[col].median()
            new_stats[f"{col}_median"] = med
        else:
            med = fit_stats[f"{col}_median"]
        out[col] = out[col].fillna(med)
    return out, (new_stats if fit_stats is None else fit_stats)


def feature_selection(X, y, feature_names):
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor

    log_msg("\n--- FEATURE SELECTION (Model A controls + linguistic) ---")
    n_features = X.shape[1]
    scores = pd.DataFrame({"feature": feature_names})

    spearman_r, spearman_p = [], []
    for i in range(n_features):
        r, p = stats.spearmanr(X[:, i], y)
        spearman_r.append(abs(r))
        spearman_p.append(p)
    scores["spearman_abs"] = spearman_r
    scores["spearman_p"] = spearman_p
    scores["spearman_rank"] = scores["spearman_abs"].rank(ascending=False)

    mi = mutual_info_regression(X, y, random_state=RANDOM_STATE, n_neighbors=10)
    scores["mi"] = mi
    scores["mi_rank"] = scores["mi"].rank(ascending=False)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    scores["rf_importance"] = rf.feature_importances_
    scores["rf_rank"] = scores["rf_importance"].rank(ascending=False)

    scores["avg_rank"] = (scores["spearman_rank"] + scores["mi_rank"] + scores["rf_rank"]) / 3
    scores["is_linguistic"] = scores["feature"].isin(LINGUISTIC_FEATURES)
    scores = scores.sort_values("avg_rank")

    log_msg(f"{'Feature':25s} {'|Spearman|':>11s} {'p-value':>10s} {'MI':>8s} {'RF imp':>8s} {'Avg rank':>9s} {'Type':>8s}")
    for _, row in scores.iterrows():
        ftype = "LING" if row["is_linguistic"] else "ctrl"
        sig = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
        log_msg(f"  {row['feature']:25s} {row['spearman_abs']:10.4f} {sig:3s} {row['spearman_p']:10.2e} "
                f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:9.1f} {ftype:>8s}")

    # Select: keep top 60% by avg rank + all significant (p < 0.05) features
    n_keep = max(int(n_features * 0.6), 8)
    top_by_rank = set(scores.head(n_keep)["feature"])
    significant = set(scores[scores["spearman_p"] < 0.05]["feature"])
    selected = sorted(top_by_rank | significant, key=lambda f: scores.set_index("feature").loc[f, "avg_rank"])

    n_ling_selected = sum(1 for f in selected if f in LINGUISTIC_FEATURES)
    log_msg(f"\nSelected {len(selected)} features ({n_ling_selected} linguistic):")
    log_msg(f"  {', '.join(selected)}")

    return scores, selected


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


def build_sector_dummies(sector_values, all_sectors):
    sector_to_idx = {s: i for i, s in enumerate(all_sectors[1:])}
    d = np.zeros((len(sector_values), len(sector_to_idx)))
    for i, s in enumerate(sector_values):
        if s in sector_to_idx:
            d[i, sector_to_idx[s]] = 1
    return d, [f"sec_{s[:8]}" for s in all_sectors[1:]]


def _clone_model(model):
    from sklearn.base import clone
    return clone(model)


def run_models(X_numeric, feature_names, sector, dates, y, all_sectors):
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
            X_tr_raw = pd.DataFrame(X_numeric[train_idx], columns=feature_names)
            X_te_raw = pd.DataFrame(X_numeric[test_idx], columns=feature_names)
            y_tr, y_te = y[train_idx], y[test_idx]
            X_tr_proc, fit_stats = preprocess_df(X_tr_raw, feature_names)
            X_te_proc, _ = preprocess_df(X_te_raw, feature_names, fit_stats=fit_stats)
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
        if hasattr(model_clone, "feature_importances_"):
            all_names = feature_names + sec_names
            imp = pd.Series(model_clone.feature_importances_, index=all_names)
            imp = imp.sort_values(ascending=False).head(15)
            log_msg(f"  top features: {', '.join(f'{n}={v:.3f}' for n, v in imp.items())}")
    return all_results


def test_report(test_df, best_model_name, X_raw, y_train, sector_train, feature_names, all_sectors):
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
    log_msg(f"\ntest set: {len(test_df):,} rows")

    X_tr_df = pd.DataFrame(X_raw, columns=feature_names)
    X_tr_proc, fit_stats = preprocess_df(X_tr_df, feature_names)
    X_te_df = test_df[feature_names].copy()
    X_te_proc, _ = preprocess_df(X_te_df, feature_names, fit_stats=fit_stats)
    y_te = test_df[PRIMARY_TARGET].values

    sec_tr, sec_names = build_sector_dummies(sector_train, all_sectors)
    sec_te, _ = build_sector_dummies(test_df[SECTOR_COL].values, all_sectors)
    X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
    X_te_full = np.hstack([X_te_proc.values, sec_te])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_te_s = scaler.transform(X_te_full)
    model.fit(X_tr_s, y_train)
    preds = model.predict(X_te_s)

    metrics = evaluate_predictions(y_te, preds)
    log_msg(f"\n{'='*60}")
    log_msg(f"TEST REPORT ({best_model_name} trained on full train 2010-2021):")
    for k, v in metrics.items():
        log_msg(f"  {k:25s}: {v:+.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    return metrics


def write_report(base, feat_scores, selected, model_results, test_metrics, best_model, model_a_results):
    out = Path(base) / "model_b_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL B REPORT — Controls + Loughran-McDonald Linguistic Features")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Target: {PRIMARY_TARGET}")
    lines.append(f"Train: 2010–2021 | Test: 2022–2024")
    lines.append(f"Model A controls (11): {', '.join(MODEL_A_FEATURES)}")
    lines.append(f"Linguistic features (9): {', '.join(LINGUISTIC_FEATURES)}")
    lines.append(f"Selected features ({len(selected)}): {', '.join(selected)}")
    lines.append("=" * 70)

    lines.append("\n\n--- FEATURE SELECTION RANKING ---")
    lines.append(f"{'Feature':25s} {'|Spearman|':>11s} {'p-value':>10s} {'MI':>8s} {'RF imp':>8s} {'Avg rank':>9s} {'Type':>8s} {'Selected':>9s}")
    for _, row in feat_scores.iterrows():
        ftype = "LING" if row["is_linguistic"] else "ctrl"
        sel = "YES" if row["feature"] in selected else ""
        lines.append(f"{row['feature']:25s} {row['spearman_abs']:11.4f} {row['spearman_p']:10.2e} "
                     f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:9.1f} {ftype:>8s} {sel:>9s}")

    lines.append("\n\n--- CROSS-VALIDATION RESULTS (train 2010-2021) ---")
    lines.append(f"{'Model':20s} {'IC':>8s} {'±':>6s} {'Rank IC':>8s} {'±':>6s} {'R²':>8s} {'Dir Acc':>8s} {'Q5-Q1':>8s}")
    ranked = sorted(model_results.items(), key=lambda x: x[1]["avg"]["rank_ic"], reverse=True)
    for name, res in ranked:
        a, s = res["avg"], res["std"]
        lines.append(f"{name:20s} {a['ic']:+8.4f} {s['ic']:6.4f} {a['rank_ic']:+8.4f} {s['rank_ic']:6.4f} "
                     f"{a['r2']:+8.4f} {a['direction_accuracy']:8.3f} {a['quintile_spread']:+8.4f}")

    lines.append(f"\nBest model by avg Rank IC: {best_model}")

    lines.append("\n\n--- MODEL A vs MODEL B COMPARISON ---")
    a_rank_ic = model_a_results["cv_rank_ic"]
    a_q5q1 = model_a_results["cv_q5q1"]
    lines.append(f"Model A best ({model_a_results['best_model']}, controls only): "
                 f"Rank IC {a_rank_ic:+.3f}, Q5-Q1 {a_q5q1:+.2%}")
    best_b = model_results[best_model]["avg"]
    lines.append(f"Model B best ({best_model}, controls + linguistic): "
                 f"Rank IC {best_b['rank_ic']:+.3f}, Q5-Q1 {best_b['quintile_spread']:+.2%}")
    diff = best_b["rank_ic"] - a_rank_ic
    lines.append(f"Rank IC improvement: {diff:+.3f} ({'better' if diff > 0 else 'worse' if diff < 0 else 'same'})")

    if test_metrics:
        lines.append(f"\n\n--- TEST RESULTS 2022-2024 ({best_model}) ---")
        for k, v in test_metrics.items():
            lines.append(f"  {k:25s}: {v:+.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        lines.append(f"\nModel A test: Rank IC {model_a_results['test_rank_ic']:+.4f}, "
                     f"Q5-Q1 {model_a_results['test_q5q1']:+.2%}")
        lines.append(f"Model B test: Rank IC {test_metrics['rank_ic']:+.4f}, "
                     f"Q5-Q1 {test_metrics['quintile_spread']:+.2%}")

    lines.append("\n\n--- NOTES ---")
    lines.append("- Same preprocessing as Model A: winsorize 1%/99%, log1p, median impute, StandardScaler")
    lines.append("- Linguistic features: LM dictionary tone (net positive), hedging (weak modal), uncertainty")
    lines.append("- _chg features = change vs. rolling 4-call average for same firm")
    lines.append("- tone_gap = tone_answers - tone_prepared (shift from script to Q&A)")
    lines.append("- Time-series expanding window CV on train (2010-2021), no future leakage")
    lines.append("- Test (2022-2024) evaluated once after model selection")

    report_text = "\n".join(lines)
    out.write_text(report_text)
    log_msg(f"\nreport saved to {out}")
    return report_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = parser.parse_args()
    base = args.folder

    log_msg("=" * 60)
    log_msg("MODEL B — Controls + Linguistic Features (Train/Test)")
    log_msg("=" * 60)

    df, train, test = load_data(base)

    # Load Model A results for comparison
    model_a_json = Path(base) / "model_a_results.json"
    if model_a_json.exists():
        ma = json.loads(model_a_json.read_text())
        model_a_results = {
            "best_model": ma["best_model"],
            "cv_rank_ic": ma["cv_results"][ma["best_model"]]["avg"]["rank_ic"],
            "cv_q5q1": ma["cv_results"][ma["best_model"]]["avg"]["quintile_spread"],
            "test_rank_ic": ma["test"]["rank_ic"],
            "test_q5q1": ma["test"]["quintile_spread"],
        }
        log_msg(f"Model A reference: {model_a_results['best_model']}, "
                f"CV Rank IC {model_a_results['cv_rank_ic']:+.3f}")
    else:
        log_msg("WARNING: model_a_results.json not found — run Model A first")
        model_a_results = {"best_model": "?", "cv_rank_ic": 0, "cv_q5q1": 0,
                           "test_rank_ic": 0, "test_q5q1": 0}

    available = [f for f in ALL_CANDIDATE_FEATURES if f in train.columns]
    missing_cols = [f for f in ALL_CANDIDATE_FEATURES if f not in train.columns]
    if missing_cols:
        log_msg(f"features not in data (skipping): {missing_cols}")
    log_msg(f"candidate features ({len(available)}): {', '.join(available)}")

    train_with_target = train.copy()
    log_msg(f"train rows: {len(train_with_target):,}")

    X_raw = train_with_target[available].values
    y = train_with_target[PRIMARY_TARGET].values
    sector = train_with_target[SECTOR_COL].values
    dates = train_with_target["call_date"].values
    all_sectors = sorted(df[SECTOR_COL].dropna().unique())

    log_msg("\n--- PREPROCESSING ---")
    X_df = pd.DataFrame(X_raw, columns=available)
    X_proc, fit_stats = preprocess_df(X_df, available)
    log_msg(f"after preprocessing: {X_proc.shape[0]:,} rows, {X_proc.shape[1]} features")

    feat_scores, selected_features = feature_selection(X_proc.values, y, available)

    selected_idx = [available.index(f) for f in selected_features]
    X_selected = X_raw[:, selected_idx]

    model_results = run_models(X_selected, selected_features, sector, dates, y, all_sectors)

    best_model = max(model_results, key=lambda k: model_results[k]["avg"]["rank_ic"])
    best_res = model_results[best_model]
    log_msg(f"\n>>> Best model: {best_model} "
            f"(IC={best_res['avg']['ic']:.4f}, Rank IC={best_res['avg']['rank_ic']:.4f})")

    # Test evaluation
    test_metrics = test_report(test, best_model, X_selected, y, sector, selected_features, all_sectors)

    # Report
    report = write_report(base, feat_scores, selected_features, model_results,
                         test_metrics, best_model, model_a_results)
    print("\n" + report)

    # Save JSON
    json_out = Path(base) / "model_b_results.json"
    save = {
        "version": "v2",
        "best_model": best_model,
        "split": {"train": "2010-2021", "test": "2022-2024"},
        "model_a_features": MODEL_A_FEATURES,
        "linguistic_features": LINGUISTIC_FEATURES,
        "all_candidate_features": available,
        "selected_features": selected_features,
        "target": PRIMARY_TARGET,
        "cv_results": {name: {
            "avg": {k: float(v) for k, v in r["avg"].items()},
            "std": {k: float(v) for k, v in r["std"].items()},
        } for name, r in model_results.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
        "model_a_comparison": {
            "model_a_cv_rank_ic": model_a_results["cv_rank_ic"],
            "model_a_cv_q5q1": model_a_results["cv_q5q1"],
            "model_b_cv_rank_ic": float(best_res["avg"]["rank_ic"]),
            "model_b_cv_q5q1": float(best_res["avg"]["quintile_spread"]),
            "rank_ic_diff": float(best_res["avg"]["rank_ic"] - model_a_results["cv_rank_ic"]),
        },
        "feature_selection": feat_scores[["feature", "spearman_abs", "mi", "rf_importance", "avg_rank", "is_linguistic"]].to_dict("records"),
        "timestamp": datetime.now().isoformat(),
    }
    json_out.write_text(json.dumps(save, indent=2))
    log_msg(f"results JSON saved to {json_out}")
    log_msg("done")


if __name__ == "__main__":
    main()
