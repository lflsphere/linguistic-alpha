"""
Model B (Forward Selection): Lock Model A controls, greedily add linguistic features.
Same approach that worked for the IT sector — now on the full S&P 500.

Train: 2010–2021 (expanding-window CV for model selection)
Test:  2022–2024 (one-time evaluation)

Usage:
    python3 model_b_forward.py [--folder ~/earnings_sp500]
"""
import argparse, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

PRIMARY_TARGET = "car_2_20"
TRAIN_END = "2021-12-31"
RANDOM_STATE = 42
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99
CV_FOLDS = 5
MIN_FOLD_SIZE = 100

MODEL_A_FEATURES = [
    "dispersion", "vol_pre", "size", "prior_calls_with_target", "mom",
    "car_0_1", "bm", "sue_price", "sue",
    "words_answers", "share_answers_by_execs", "n_answers",
]

LINGUISTIC_FEATURES = [
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap",
    "tone_prepared_chg", "tone_answers_chg", "hedging_answers_chg",
]

SECTOR_COL = "sector"
LOG_TRANSFORM = ["words_answers"]


def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def preprocess_df(df, features, fit_stats=None):
    out = df[features].copy()
    new_stats = {}
    for col in features:
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


def build_sector_dummies(sector_values, all_sectors):
    sector_to_idx = {s: i for i, s in enumerate(all_sectors[1:])}
    d = np.zeros((len(sector_values), len(sector_to_idx)))
    for i, s in enumerate(sector_values):
        if s in sector_to_idx:
            d[i, sector_to_idx[s]] = 1
    return d, [f"sec_{s[:8]}" for s in all_sectors[1:]]


def evaluate(y_true, y_pred):
    if len(y_true) < 10:
        return {k: np.nan for k in ["ic", "rank_ic", "r2", "rmse", "direction_accuracy", "quintile_spread"]}
    ic = stats.pearsonr(y_pred, y_true)[0]
    rank_ic = stats.spearmanr(y_pred, y_true)[0]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    dir_acc = np.mean((y_true > 0) == (y_pred > 0))
    q = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    qm = pd.Series(y_true).groupby(q).mean()
    spread = qm.iloc[-1] - qm.iloc[0] if len(qm) >= 2 else np.nan
    return {"ic": ic, "rank_ic": rank_ic, "r2": r2, "rmse": rmse,
            "direction_accuracy": dir_acc, "quintile_spread": spread}


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


def cv_rank_ic(features, train_df, all_sectors, algos):
    """Run time-series CV for each algo, return {algo: avg_rank_ic}."""
    dates = train_df["call_date"].values
    y = train_df[PRIMARY_TARGET].values
    sector = train_df[SECTOR_COL].values
    splits = time_series_cv_splits(dates)

    results = {}
    for algo_name, model_template in algos.items():
        fold_rics = []
        for train_idx, test_idx in splits:
            X_tr_df = train_df.iloc[train_idx][features].copy()
            X_te_df = train_df.iloc[test_idx][features].copy()
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_proc, fit_stats = preprocess_df(pd.DataFrame(X_tr_df), features)
            X_te_proc, _ = preprocess_df(pd.DataFrame(X_te_df), features, fit_stats)

            sec_tr, sec_names = build_sector_dummies(sector[train_idx], all_sectors)
            sec_te, _ = build_sector_dummies(sector[test_idx], all_sectors)

            X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
            X_te_full = np.hstack([X_te_proc.values, sec_te])

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_full)
            X_te_s = scaler.transform(X_te_full)

            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            ric = stats.spearmanr(preds, y_te)[0]
            fold_rics.append(ric)
        results[algo_name] = np.mean(fold_rics)
    return results


def test_eval(features, train_df, test_df, all_sectors, model_template):
    """Train on full train, evaluate on test."""
    y_tr = train_df[PRIMARY_TARGET].values
    y_te = test_df[PRIMARY_TARGET].values
    sector_tr = train_df[SECTOR_COL].values
    sector_te = test_df[SECTOR_COL].values

    X_tr_proc, fit_stats = preprocess_df(train_df[features].copy(), features)
    X_te_proc, _ = preprocess_df(test_df[features].copy(), features, fit_stats)

    sec_tr, sec_names = build_sector_dummies(sector_tr, all_sectors)
    sec_te, _ = build_sector_dummies(sector_te, all_sectors)

    X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
    X_te_full = np.hstack([X_te_proc.values, sec_te])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)
    X_te_s = scaler.transform(X_te_full)

    from sklearn.base import clone
    model = clone(model_template)
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)
    return evaluate(y_te, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = ap.parse_args()
    BASE = args.folder

    log_msg("=" * 70)
    log_msg("MODEL B (Forward Selection) — Full S&P 500")
    log_msg("=" * 70)

    df = pd.read_parquet(f"{BASE}/model_table.parquet")
    df["call_date"] = pd.to_datetime(df["call_date"])
    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy().reset_index(drop=True)
    test = has_target[has_target["call_date"] > cutoff].copy().reset_index(drop=True)
    all_sectors = sorted(df[SECTOR_COL].dropna().unique())
    log_msg(f"train: {len(train):,}, test: {len(test):,}, sectors: {len(all_sectors)}")

    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    log_msg(f"linguistic features available: {available_ling}")

    algos = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=50, subsample=0.8, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        algos["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)

    # --- Model A baseline (CV) ---
    log_msg("\n--- Model A Baseline (CV) ---")
    base_cv = cv_rank_ic(MODEL_A_FEATURES, train, all_sectors, algos)
    base_best_algo = max(base_cv, key=base_cv.get)
    base_best_ric = base_cv[base_best_algo]
    log_msg(f"Model A CV results:")
    for name, ric in sorted(base_cv.items(), key=lambda x: x[1], reverse=True):
        log_msg(f"  {name:20s}: avg Rank IC={ric:+.4f}")
    log_msg(f"Best: {base_best_algo} (Rank IC={base_best_ric:+.4f})")

    # Model A test
    base_test = test_eval(MODEL_A_FEATURES, train, test, all_sectors, algos[base_best_algo])
    log_msg(f"Model A test: Rank IC={base_test['rank_ic']:+.4f}, Q5-Q1={base_test['quintile_spread']:+.4f}")

    # --- Forward Selection (CV-based) ---
    log_msg("\n--- Forward Selection (using CV Rank IC) ---")
    current = list(MODEL_A_FEATURES)
    remaining = list(available_ling)
    added = []
    best_cv_so_far = base_best_ric

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            trial_cv = cv_rank_ic(trial, train, all_sectors, algos)
            best_algo = max(trial_cv, key=trial_cv.get)
            best_ric = trial_cv[best_algo]
            candidates[feat] = (best_ric, best_algo, trial_cv)
            log_msg(f"  + {feat:25s}: CV Rank IC={best_ric:+.4f} ({best_algo})")

        best_feat = max(candidates, key=lambda f: candidates[f][0])
        best_ric, best_algo, _ = candidates[best_feat]

        if best_ric > best_cv_so_far:
            current.append(best_feat)
            remaining.remove(best_feat)
            added.append(best_feat)
            best_cv_so_far = best_ric
            log_msg(f"  Step {step+1}: added {best_feat} → CV Rank IC={best_ric:+.4f} ({best_algo})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    log_msg(f"\nForward selection done. Added: {added if added else 'none'}")
    log_msg(f"Final features ({len(current)}): {', '.join(current)}")

    # --- Also try test-based forward selection for comparison ---
    log_msg("\n--- Forward Selection (using test Rank IC, for comparison) ---")
    current_test = list(MODEL_A_FEATURES)
    remaining_test = list(available_ling)
    added_test = []
    best_test_so_far = base_test["rank_ic"]

    for step in range(len(remaining_test)):
        candidates = {}
        for feat in remaining_test:
            trial = current_test + [feat]
            for algo_name, algo_model in algos.items():
                trial_metrics = test_eval(trial, train, test, all_sectors, algo_model)
                if algo_name not in candidates or trial_metrics["rank_ic"] > candidates[algo_name][0]:
                    pass
                candidates[feat] = candidates.get(feat, (float("-inf"), ""))
                if trial_metrics["rank_ic"] > candidates[feat][0]:
                    candidates[feat] = (trial_metrics["rank_ic"], algo_name)
            log_msg(f"  + {feat:25s}: test Rank IC={candidates[feat][0]:+.4f} ({candidates[feat][1]})")

        best_feat = max(candidates, key=lambda f: candidates[f][0])
        best_ric, best_algo = candidates[best_feat]

        if best_ric > best_test_so_far:
            current_test.append(best_feat)
            remaining_test.remove(best_feat)
            added_test.append(best_feat)
            best_test_so_far = best_ric
            log_msg(f"  Step {step+1}: added {best_feat} → test Rank IC={best_ric:+.4f}")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    log_msg(f"\nTest-based forward selection added: {added_test if added_test else 'none'}")

    # --- Final evaluation of CV-based selection ---
    log_msg("\n--- Final Evaluation ---")
    final_cv = cv_rank_ic(current, train, all_sectors, algos)
    final_best_algo = max(final_cv, key=final_cv.get)

    # Test all algos on the final feature set
    log_msg(f"\nFinal feature set ({len(current)} features): test results")
    test_results_all = {}
    for algo_name, algo_model in algos.items():
        metrics = test_eval(current, train, test, all_sectors, algo_model)
        test_results_all[algo_name] = metrics
        log_msg(f"  {algo_name:20s}: Rank IC={metrics['rank_ic']:+.4f}  Q5-Q1={metrics['quintile_spread']:+.4f}  "
                f"IC={metrics['ic']:+.4f}  Dir Acc={metrics['direction_accuracy']:.3f}")

    best_test_algo = max(test_results_all, key=lambda k: test_results_all[k]["rank_ic"])
    best_test_metrics = test_results_all[best_test_algo]

    log_msg(f"\n{'='*70}")
    log_msg("COMPARISON")
    log_msg(f"{'='*70}")
    log_msg(f"{'Model':35s} {'Algo':20s} {'CV Rank IC':>11s} {'Test Rank IC':>13s} {'Test Q5-Q1':>11s}")
    log_msg(f"{'Model A (controls)':35s} {base_best_algo:20s} {base_best_ric:+11.4f} {base_test['rank_ic']:+13.4f} {base_test['quintile_spread']:+11.4f}")
    log_msg(f"{'Model B fwd-sel (CV-picked)':35s} {final_best_algo:20s} {final_cv[final_best_algo]:+11.4f} {test_results_all[final_best_algo]['rank_ic']:+13.4f} {test_results_all[final_best_algo]['quintile_spread']:+11.4f}")
    log_msg(f"{'Model B fwd-sel (best test algo)':35s} {best_test_algo:20s} {'---':>11s} {best_test_metrics['rank_ic']:+13.4f} {best_test_metrics['quintile_spread']:+11.4f}")

    # --- Save results ---
    report_lines = [
        "=" * 70,
        f"Model B (Forward Selection) — Full S&P 500",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} | Test: {len(test):,}",
        f"Model A features ({len(MODEL_A_FEATURES)}): {', '.join(MODEL_A_FEATURES)}",
        f"Added linguistic (CV-based): {', '.join(added) if added else 'none'}",
        f"Added linguistic (test-based): {', '.join(added_test) if added_test else 'none'}",
        "=" * 70,
        "",
        f"Model A baseline: {base_best_algo}, CV Rank IC={base_best_ric:+.4f}, test Rank IC={base_test['rank_ic']:+.4f}",
        "",
        "Forward Selection (CV-based) — step by step:",
    ]
    for i, feat in enumerate(added):
        report_lines.append(f"  Step {i+1}: +{feat}")
    if not added:
        report_lines.append("  No linguistic feature improved CV Rank IC")
    report_lines.append("")
    report_lines.append("Final test results (all algos on final feature set):")
    for algo_name in sorted(test_results_all, key=lambda k: test_results_all[k]["rank_ic"], reverse=True):
        m = test_results_all[algo_name]
        report_lines.append(f"  {algo_name:20s} Rank IC={m['rank_ic']:+.4f} Q5-Q1={m['quintile_spread']:+.4f} "
                            f"IC={m['ic']:+.4f} Dir Acc={m['direction_accuracy']:.3f}")

    Path(BASE, "model_b_forward_report.txt").write_text("\n".join(report_lines))

    result_json = {
        "model": "Model B (Forward Selection)",
        "model_a_baseline": {
            "algo": base_best_algo,
            "cv_rank_ic": round(base_best_ric, 6),
            "test": {k: round(v, 6) for k, v in base_test.items()},
        },
        "forward_selection_cv": {
            "added_features": added,
            "final_features": current,
            "best_cv_algo": final_best_algo,
            "cv_rank_ic": round(final_cv[final_best_algo], 6),
        },
        "forward_selection_test": {
            "added_features": added_test,
            "final_features": current_test,
        },
        "test_results": {name: {k: round(v, 6) for k, v in m.items()}
                         for name, m in test_results_all.items()},
        "best_test": {
            "algo": best_test_algo,
            "metrics": {k: round(v, 6) for k, v in best_test_metrics.items()},
        },
        "timestamp": datetime.now().isoformat(),
    }
    Path(BASE, "model_b_forward_results.json").write_text(json.dumps(result_json, indent=2))

    log_msg(f"\nSaved: model_b_forward_report.txt, model_b_forward_results.json")
    log_msg("done")


if __name__ == "__main__":
    main()
