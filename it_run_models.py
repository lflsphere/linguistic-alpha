"""
IT Sector Models A & B — runs on both Option 1 (market bench) and Option 2 (IT bench).
Simple train/test split: train 2010-2021, test 2022-2024. No CV.

Reads:  ~/earnings_it/model_table_opt1.parquet, model_table_opt2.parquet
Writes: ~/earnings_it/it_{model}_{option}_report.txt + _results.json

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
    significant = set(scores[scores["spearman_p"] < 0.10]["feature"])
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


def run_one(df, model_name, option_name, features, log_cols, out_dir):
    """Feature selection on train, train all algos on 2010-2021, evaluate on 2022-2024."""
    log_msg(f"\n{'='*70}")
    log_msg(f"{model_name} — {option_name}")
    log_msg(f"{'='*70}")

    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

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

    # Preprocess train for feature selection
    X_tr_raw = train[available]
    y_tr = train[PRIMARY_TARGET].values
    X_tr_proc, fit_stats = preprocess_df(X_tr_raw, available, log_cols)

    # Feature selection on train
    feat_scores, selected = feature_selection(X_tr_proc.values, y_tr, available)

    # Preprocess with selected features
    X_tr_sel = train[selected]
    X_te_sel = test[selected]
    X_tr_p, fit_stats_sel = preprocess_df(X_tr_sel, selected, log_cols)
    X_te_p, _ = preprocess_df(X_te_sel, selected, log_cols, fit_stats=fit_stats_sel)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_p.values)
    X_te_s = scaler.transform(X_te_p.values)
    y_te = test[PRIMARY_TARGET].values

    # Train all algos, evaluate on test
    algos = {
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
        algos["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    all_test_results = {}
    log_msg(f"\n--- TEST RESULTS (train 2010-2021 → test 2022-2024) ---")
    log_msg(f"{'Algo':20s} {'IC':>8s} {'Rank IC':>9s} {'R²':>8s} {'Dir Acc':>8s} {'Q5-Q1':>8s}")

    for name, algo in algos.items():
        algo.fit(X_tr_s, y_tr)
        preds = algo.predict(X_te_s)
        metrics = evaluate_predictions(y_te, preds)
        all_test_results[name] = metrics
        log_msg(f"  {name:20s} {metrics['ic']:+8.4f} {metrics['rank_ic']:+9.4f} "
                f"{metrics['r2']:+8.4f} {metrics['direction_accuracy']:8.3f} {metrics['quintile_spread']:+8.4f}")

    best = max(all_test_results, key=lambda k: all_test_results[k]["rank_ic"])
    best_metrics = all_test_results[best]
    log_msg(f"\n>>> Best: {best} (Rank IC={best_metrics['rank_ic']:+.4f}, Q5-Q1={best_metrics['quintile_spread']:+.4f})")

    # Save report
    tag = f"{model_name.lower().replace(' ', '_')}_{option_name.lower().replace(' ', '_').replace('(','').replace(')','').replace(' ','_')}"
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
        "--- FEATURE SELECTION (on train) ---",
    ]
    lines.append(f"{'Feature':25s} {'|ρ|':>6s} {'p':>10s} {'MI':>8s} {'RF':>8s} {'AvgRk':>7s} {'Type':>6s} {'Sel':>4s}")
    for _, row in feat_scores.iterrows():
        ftype = "LING" if row["feature"] in LINGUISTIC_FEATURES else "ctrl"
        sel = "YES" if row["feature"] in selected else ""
        lines.append(f"{row['feature']:25s} {row['spearman_abs']:6.4f} {row['spearman_p']:10.2e} "
                     f"{row['mi']:8.4f} {row['rf_importance']:8.4f} {row['avg_rank']:7.1f} {ftype:>6s} {sel:>4s}")

    lines.append("")
    lines.append("--- TEST RESULTS (all algos) ---")
    lines.append(f"{'Algo':20s} {'IC':>8s} {'Rank IC':>9s} {'R²':>8s} {'Dir Acc':>8s} {'Q5-Q1':>8s}")
    ranked = sorted(all_test_results.items(), key=lambda x: x[1]["rank_ic"], reverse=True)
    for name, m in ranked:
        lines.append(f"{name:20s} {m['ic']:+8.4f} {m['rank_ic']:+9.4f} "
                     f"{m['r2']:+8.4f} {m['direction_accuracy']:8.3f} {m['quintile_spread']:+8.4f}")
    lines.append(f"\nBest: {best}")

    report_path = Path(out_dir) / f"it_{tag}_report.txt"
    report_path.write_text("\n".join(lines))
    log_msg(f"report: {report_path}")

    json_path = Path(out_dir) / f"it_{tag}_results.json"
    json_path.write_text(json.dumps({
        "model": model_name, "option": option_name, "best_algo": best,
        "selected_features": selected,
        "all_test_results": {k: {mk: round(mv, 6) for mk, mv in v.items()} for k, v in all_test_results.items()},
        "test": {k: round(v, 6) for k, v in best_metrics.items()},
        "train_size": len(train), "test_size": len(test),
        "n_companies": int(df["permno"].nunique()),
    }, indent=2))

    return {"best": best, "test": best_metrics, "selected": selected, "all_results": all_test_results}


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

        # Model A: controls only
        res_a = run_one(df, "Model A", opt_label, MODEL_A_CANDIDATES, LOG_TRANSFORM_A, OUT)
        results[f"A_{opt_label}"] = res_a

        # Model B: controls + linguistic
        model_b_candidates = [f for f in (MODEL_A_CANDIDATES + LINGUISTIC_FEATURES) if f in df.columns]
        res_b = run_one(df, "Model B", opt_label, model_b_candidates, LOG_TRANSFORM_B, OUT)
        results[f"B_{opt_label}"] = res_b

    # Summary
    log_msg(f"\n{'='*70}")
    log_msg("SUMMARY — IT Sector Analysis")
    log_msg(f"{'='*70}")
    log_msg(f"{'Config':40s} {'Best Algo':15s} {'Test Rank IC':>13s} {'Test Q5-Q1':>11s}")
    for key, res in results.items():
        log_msg(f"{key:40s} {res['best']:15s} {res['test']['rank_ic']:+13.4f} "
                f"{res['test']['quintile_spread']:+11.4f}")

    summary_path = Path(OUT) / "it_summary.json"
    summary = {}
    for key, res in results.items():
        summary[key] = {
            "best_algo": res["best"],
            "selected_features": res["selected"],
            "test_rank_ic": round(res["test"]["rank_ic"], 6),
            "test_quintile_spread": round(res["test"]["quintile_spread"], 6),
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    log_msg(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
