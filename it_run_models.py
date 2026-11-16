"""
IT Sector Models A & B — runs on both Option 1 (market bench) and Option 2 (IT bench).

Reads:  ~/earnings_it/model_table_opt1.parquet, model_table_opt2.parquet
Writes: ~/earnings_it/it_model_{a,b}_{opt1,opt2}_report.txt
        ~/earnings_it/it_model_{a,b}_{opt1,opt2}_results.json

Usage:
    python3 it_run_models.py [--folder ~/earnings_it]
"""
import argparse, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"
TRAIN_END = "2021-12-31"
CV_FOLDS = 4
MIN_FOLD_SIZE = 50
RANDOM_STATE = 42
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

FINANCIAL_CONTROLS = [
    "sue", "sue_price", "surprise_pct",
    "size", "bm", "mom", "vol_pre",
    "n_analysts", "dispersion", "car_0_1",
]
CALL_STRUCTURE = [
    "n_questions", "n_answers",
    "words_prepared", "words_answers", "words_questions",
    "avg_answer_words", "share_answers_by_execs",
    "prior_calls_with_target",
]
MODEL_A_CANDIDATES = FINANCIAL_CONTROLS + CALL_STRUCTURE

LINGUISTIC_FEATURES = [
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap",
    "tone_prepared_chg", "tone_answers_chg", "hedging_answers_chg",
]

LOG_TRANSFORM_A = ["words_prepared", "words_answers", "words_questions", "avg_answer_words", "n_analysts"]
LOG_TRANSFORM_B = ["words_answers"]


def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def preprocess_df(df, features, log_cols, fit_stats=None):
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
        if col in log_cols:
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

    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=30,
                               max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    scores["rf_importance"] = rf.feature_importances_
    scores["rf_rank"] = scores["rf_importance"].rank(ascending=False)

    scores["avg_rank"] = (scores["spearman_rank"] + scores["mi_rank"] + scores["rf_rank"]) / 3
    scores = scores.sort_values("avg_rank")

    log_msg(f"{'Feature':25s} {'|Spearman|':>11s} {'p-value':>10s} {'MI':>8s} {'RF imp':>8s} {'Avg rank':>9s}")
    for _, row in scores.iterrows():
        log_msg(f"  {row['feature']:25s} {row['spearman_abs']:10.4f}   {row['spearman_p']:10.2e} "
                f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:9.1f}")

    n_keep = max(n_features // 2, 5)
    top_by_rank = set(scores.head(n_keep)["feature"])
    significant = set(scores[scores["spearman_p"] < 0.10]["feature"])  # relaxed for smaller sample
    selected = sorted(top_by_rank | significant,
                      key=lambda f: scores.set_index("feature").loc[f, "avg_rank"])

    log_msg(f"\nSelected {len(selected)} features: {', '.join(selected)}")
    return scores, selected


def evaluate_predictions(y_true, y_pred):
    if len(y_true) < 10:
        return {k: np.nan for k in ["ic", "rank_ic", "r2", "rmse", "direction_accuracy", "quintile_spread"]}
    ic = stats.pearsonr(y_pred, y_true)[0]
    rank_ic = stats.spearmanr(y_pred, y_true)[0]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    accuracy = np.mean(direction_true == direction_pred)
    q = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    qm = pd.Series(y_true).groupby(q).mean()
    spread = qm.iloc[-1] - qm.iloc[0] if len(qm) >= 2 else np.nan
    return {"ic": ic, "rank_ic": rank_ic, "r2": r2, "rmse": rmse,
            "direction_accuracy": accuracy, "quintile_spread": spread}


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


def run_models(X_numeric, feature_names, dates, y, log_cols):
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    splits = time_series_cv_splits(dates, CV_FOLDS)
    log_msg(f"\nTime-series CV: {len(splits)} folds")
    for i, (tr, te) in enumerate(splits):
        log_msg(f"  fold {i+1}: train={len(tr):,}, test={len(te):,}")

    # Smaller sample -> adjusted hyperparameters
    models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_samples_leaf=30, subsample=0.8, random_state=RANDOM_STATE),
    }
    if has_xgb:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    all_results = {}
    for name, model in models.items():
        log_msg(f"\n  {name}:")
        fold_metrics = []
        for i, (train_idx, test_idx) in enumerate(splits):
            X_tr_raw = pd.DataFrame(X_numeric[train_idx], columns=feature_names)
            X_te_raw = pd.DataFrame(X_numeric[test_idx], columns=feature_names)
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_proc, fit_stats = preprocess_df(X_tr_raw, feature_names, log_cols)
            X_te_proc, _ = preprocess_df(X_te_raw, feature_names, log_cols, fit_stats=fit_stats)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_proc.values)
            X_te_s = scaler.transform(X_te_proc.values)

            m = clone(model)
            m.fit(X_tr_s, y_tr)
            preds = m.predict(X_te_s)
            metrics = evaluate_predictions(y_te, preds)
            fold_metrics.append(metrics)

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        log_msg(f"    IC={avg['ic']:+.4f}±{std['ic']:.4f}  Rank IC={avg['rank_ic']:+.4f}±{std['rank_ic']:.4f}  "
                f"R²={avg['r2']:+.4f}  Dir={avg['direction_accuracy']:.3f}  Q5-Q1={avg['quintile_spread']:+.4f}")
        all_results[name] = {"avg": avg, "std": std}

    return all_results


def test_evaluation(train_df, test_df, best_name, features, log_cols):
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

    model_map = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_samples_leaf=30, subsample=0.8, random_state=RANDOM_STATE),
    }
    if has_xgb:
        model_map["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    X_tr_raw = train_df[features]
    X_te_raw = test_df[features]
    y_tr = train_df[PRIMARY_TARGET].values
    y_te = test_df[PRIMARY_TARGET].values

    X_tr_proc, fit_stats = preprocess_df(X_tr_raw, features, log_cols)
    X_te_proc, _ = preprocess_df(X_te_raw, features, log_cols, fit_stats=fit_stats)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_proc.values)
    X_te_s = scaler.transform(X_te_proc.values)

    model = model_map[best_name]
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)
    metrics = evaluate_predictions(y_te, preds)

    log_msg(f"\n  TEST ({best_name}):")
    for k, v in metrics.items():
        log_msg(f"    {k:25s}: {v:+.4f}")
    return metrics


def run_one(df, model_name, option_name, features, log_cols, out_dir):
    """Run feature selection, CV, and test for one model+option combo."""
    log_msg(f"\n{'='*70}")
    log_msg(f"{model_name} — {option_name}")
    log_msg(f"{'='*70}")

    df = df.copy()
    df["call_date"] = pd.to_datetime(df["call_date"])
    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy()
    test = has_target[has_target["call_date"] > cutoff].copy()
    log_msg(f"train: {len(train):,} ({train['call_date'].min().date()} to {train['call_date'].max().date()})")
    log_msg(f"test:  {len(test):,} ({test['call_date'].min().date()} to {test['call_date'].max().date()})")

    available = [f for f in features if f in train.columns]
    log_msg(f"candidate features ({len(available)}): {', '.join(available)}")

    X_raw = train[available].values
    y = train[PRIMARY_TARGET].values
    dates = train["call_date"].values

    X_df = pd.DataFrame(X_raw, columns=available)
    X_proc, _ = preprocess_df(X_df, available, log_cols)

    # Feature selection
    feat_scores, selected = feature_selection(X_proc.values, y, available)

    # CV with selected features
    sel_idx = [available.index(f) for f in selected]
    X_sel = X_raw[:, sel_idx]
    cv_results = run_models(X_sel, selected, dates, y, log_cols)

    best = max(cv_results, key=lambda k: cv_results[k]["avg"]["rank_ic"])
    best_res = cv_results[best]
    log_msg(f"\n>>> Best: {best} (Rank IC={best_res['avg']['rank_ic']:+.4f})")

    # Test
    test_metrics = test_evaluation(train, test, best, selected, log_cols)

    # Save report
    tag = f"{model_name.lower().replace(' ', '_')}_{option_name.lower().replace(' ', '_').replace('(','').replace(')','')}"
    lines = [
        "=" * 70,
        f"{model_name} — {option_name} — IT Sector Only",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Target: {PRIMARY_TARGET}",
        f"Train: 2010–2021 ({len(train):,} calls) | Test: 2022–2024 ({len(test):,} calls)",
        f"Companies: {df['permno'].nunique()}",
        f"Selected features ({len(selected)}): {', '.join(selected)}",
        "=" * 70,
        "",
        "--- FEATURE SELECTION ---",
    ]
    lines.append(f"{'Feature':25s} {'|ρ|':>6s} {'p':>10s} {'MI':>8s} {'RF':>8s} {'AvgRk':>7s} {'Type':>6s} {'Sel':>4s}")
    for _, row in feat_scores.iterrows():
        ftype = "LING" if row["feature"] in LINGUISTIC_FEATURES else "ctrl"
        sel = "YES" if row["feature"] in selected else ""
        lines.append(f"{row['feature']:25s} {row['spearman_abs']:6.4f} {row['spearman_p']:10.2e} "
                     f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:7.1f} {ftype:>6s} {sel:>4s}")

    lines.append("")
    lines.append("--- CROSS-VALIDATION ---")
    lines.append(f"{'Model':20s} {'IC':>8s} {'±':>6s} {'Rank IC':>8s} {'±':>6s} {'R²':>8s} {'Dir':>6s} {'Q5-Q1':>8s}")
    ranked = sorted(cv_results.items(), key=lambda x: x[1]["avg"]["rank_ic"], reverse=True)
    for name, res in ranked:
        a, s = res["avg"], res["std"]
        lines.append(f"{name:20s} {a['ic']:+8.4f} {s['ic']:6.4f} {a['rank_ic']:+8.4f} {s['rank_ic']:6.4f} "
                     f"{a['r2']:+8.4f} {a['direction_accuracy']:6.3f} {a['quintile_spread']:+8.4f}")
    lines.append(f"\nBest: {best}")

    lines.append("")
    lines.append(f"--- TEST 2022-2024 ({best}) ---")
    for k, v in test_metrics.items():
        lines.append(f"  {k:25s}: {v:+.4f}")

    report_path = Path(out_dir) / f"it_{tag}_report.txt"
    report_path.write_text("\n".join(lines))
    log_msg(f"report: {report_path}")

    # Save JSON
    json_path = Path(out_dir) / f"it_{tag}_results.json"
    json_path.write_text(json.dumps({
        "model": model_name, "option": option_name, "best_algo": best,
        "selected_features": selected,
        "cv_rank_ic": best_res["avg"]["rank_ic"],
        "cv_quintile_spread": best_res["avg"]["quintile_spread"],
        "test": {k: round(v, 6) for k, v in test_metrics.items()},
        "train_size": len(train), "test_size": len(test),
        "n_companies": int(df["permno"].nunique()),
    }, indent=2))

    return {"best": best, "cv": best_res, "test": test_metrics, "selected": selected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_it"))
    args = ap.parse_args()
    OUT = args.folder

    results = {}

    for opt_label, opt_file in [("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
                                 ("Option 2 (IT sector bench)", "model_table_opt2.parquet")]:
        fpath = f"{OUT}/{opt_file}"
        if not os.path.exists(fpath):
            log_msg(f"SKIP {opt_label}: {fpath} not found")
            continue
        df = pd.read_parquet(fpath)

        # Model A: controls only, no sector dummies (single sector)
        res_a = run_one(df, "Model A", opt_label, MODEL_A_CANDIDATES, LOG_TRANSFORM_A, OUT)
        results[f"A_{opt_label}"] = res_a

        # Model B: controls + linguistic
        model_b_candidates = [f for f in (MODEL_A_CANDIDATES + LINGUISTIC_FEATURES) if f in df.columns]
        res_b = run_one(df, "Model B", opt_label, model_b_candidates, LOG_TRANSFORM_B, OUT)
        results[f"B_{opt_label}"] = res_b

    # Summary comparison
    log_msg(f"\n{'='*70}")
    log_msg("SUMMARY — IT Sector Analysis")
    log_msg(f"{'='*70}")
    log_msg(f"{'Config':40s} {'Best Algo':15s} {'CV Rank IC':>11s} {'Test Rank IC':>13s} {'Test Q5-Q1':>11s}")
    for key, res in results.items():
        log_msg(f"{key:40s} {res['best']:15s} {res['cv']['avg']['rank_ic']:+11.4f} "
                f"{res['test']['rank_ic']:+13.4f} {res['test']['quintile_spread']:+11.4f}")

    # Save summary
    summary_path = Path(OUT) / "it_summary.json"
    summary = {}
    for key, res in results.items():
        summary[key] = {
            "best_algo": res["best"],
            "selected_features": res["selected"],
            "cv_rank_ic": round(res["cv"]["avg"]["rank_ic"], 6),
            "test_rank_ic": round(res["test"]["rank_ic"], 6),
            "test_quintile_spread": round(res["test"]["quintile_spread"], 6),
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    log_msg(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
