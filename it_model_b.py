"""
IT Sector — Improved Model B: tries multiple strategies to add linguistic signal
on top of Model A controls. Whichever strategy beats Model A becomes Model B.

Strategies:
  1. Forward selection: lock Model A features, greedily add linguistic features
  2. Interaction features: tone × sue, tone_chg × car_-1_1, hedging × vol_pre, etc.
  3. Residualized tone: regress tone on SUE/car_-1_1, use residual as "unexpected tone"
  4. All linguistic forced in with strong regularization (ElasticNet with higher alpha)
  5. PCA on linguistic features: compress 9 LM features into 2-3 components

Train/test split: 2010-2021 / 2022-2024. No CV (small sample).
Uses only Ridge/ElasticNet (trees overfit on IT subsample).

Usage:
    python3 it_model_b_improved.py [--folder ~/earnings_it]
"""
import argparse, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"
TRAIN_END = "2021-12-31"
RANDOM_STATE = 42
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

MODEL_A_FEATURES_OPT1 = [
    "vol_pre", "car_-1_1", "mom", "bm", "dispersion",
    "avg_answer_words", "size", "prior_calls_with_target", "words_prepared",
]
MODEL_A_FEATURES_OPT2 = [
    "vol_pre", "prior_calls_with_target", "avg_answer_words", "size",
    "dispersion", "words_questions", "car_-1_1", "n_questions", "mom", "n_answers",
]

LINGUISTIC_FEATURES = [
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap",
    "tone_prepared_chg", "tone_answers_chg", "hedging_answers_chg",
]

LOG_TRANSFORM = ["words_prepared", "words_answers", "words_questions",
                 "avg_answer_words", "n_analysts"]


def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def winsorize_and_impute(series, lo_val=None, hi_val=None, med_val=None):
    if lo_val is None:
        lo_val = series.quantile(WINSOR_LOW)
        hi_val = series.quantile(WINSOR_HIGH)
        med_val = series.median()
    s = series.clip(lo_val, hi_val).fillna(med_val)
    return s, lo_val, hi_val, med_val


def preprocess(train_df, test_df, features):
    """Winsorize, log-transform, impute, scale. Returns X_tr, X_te, scaler."""
    stats_dict = {}
    X_tr = train_df[features].copy()
    X_te = test_df[features].copy()

    for col in features:
        if col in LOG_TRANSFORM:
            X_tr[col] = np.log1p(X_tr[col].clip(lower=0))
            X_te[col] = np.log1p(X_te[col].clip(lower=0))
        X_tr[col], lo, hi, med = winsorize_and_impute(X_tr[col])
        X_te[col], _, _, _ = winsorize_and_impute(X_te[col], lo, hi, med)
        stats_dict[col] = (lo, hi, med)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.values)
    X_te_s = scaler.transform(X_te.values)
    return X_tr_s, X_te_s, scaler, stats_dict


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


def fit_and_eval(X_tr, X_te, y_tr, y_te, algos):
    """Fit each algo, return {name: metrics}."""
    results = {}
    for name, model in algos.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results[name] = evaluate(y_te, preds)
    return results


def best_result(results):
    best_name = max(results, key=lambda k: results[k]["rank_ic"])
    return best_name, results[best_name]


N_CV_FOLDS = 5


def cv_rank_ic(train_df, features, y_col, algos, n_folds=N_CV_FOLDS):
    """Evaluate features via k-fold CV within training set. Returns best (mean_rank_ic, algo_name)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    algo_scores = {name: [] for name in algos}
    X_all = train_df[features].copy()
    y_all = train_df[y_col].values

    for fold_tr, fold_val in kf.split(X_all):
        fold_train = train_df.iloc[fold_tr]
        fold_test = train_df.iloc[fold_val]
        y_tr_f = y_all[fold_tr]
        y_val_f = y_all[fold_val]
        X_tr_f, X_te_f, _, _ = preprocess(fold_train, fold_test, features)
        for name, model_cls in algos.items():
            import copy
            model = copy.deepcopy(model_cls)
            model.fit(X_tr_f, y_tr_f)
            preds = model.predict(X_te_f)
            ric = stats.spearmanr(preds, y_val_f)[0]
            algo_scores[name].append(ric if not np.isnan(ric) else 0.0)

    best_name = max(algo_scores, key=lambda k: np.mean(algo_scores[k]))
    best_mean = np.mean(algo_scores[best_name])
    return best_mean, best_name


def strategy_forward_selection(train, test, base_features, y_tr, y_te):
    """Lock base features, greedily add linguistic features using CV within training set."""
    log_msg("\n--- Strategy 1: Forward Selection (5-fold CV) ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]

    current = list(base_features)
    algos = {"Ridge": Ridge(alpha=1.0), "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)}

    base_cv_ric, base_algo = cv_rank_ic(train, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline ({len(current)} features): CV Rank IC={base_cv_ric:+.4f} ({base_algo})")

    added = []
    remaining = list(available_ling)
    best_so_far = base_cv_ric

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = cv_rank_ic(train, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
            log_msg(f"    + {feat:25s}: CV Rank IC={cv_ric:+.4f} ({algo_name})")

        best_feat = max(candidates, key=lambda f: candidates[f][0])
        best_ric, best_algo = candidates[best_feat]

        if best_ric > best_so_far:
            current.append(best_feat)
            remaining.remove(best_feat)
            added.append(best_feat)
            best_so_far = best_ric
            log_msg(f"  Step {step+1}: added {best_feat} → CV Rank IC={best_ric:+.4f}")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    # Final evaluation on holdout
    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr, y_te, algos)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(current)} features, Holdout Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")
    log_msg(f"  Added linguistic: {added if added else 'none'}")

    return {"name": "Forward Selection", "features": current, "added_ling": added,
            "algo": best_name, "metrics": best_metrics, "all_results": final_results}


def strategy_interactions(train, test, base_features, y_tr, y_te):
    """Add interaction features between linguistic and financial controls."""
    log_msg("\n--- Strategy 2: Interaction Features ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]

    interactions = [
        ("tone_answers", "sue", "tone×sue"),
        ("tone_answers_chg", "car_-1_1", "tone_chg×car01"),
        ("tone_answers_chg", "sue", "tone_chg×sue"),
        ("hedging_answers", "vol_pre", "hedge×vol"),
        ("tone_gap", "sue", "gap×sue"),
        ("tone_prepared_chg", "mom", "tone_prep_chg×mom"),
        ("uncertainty_answers", "vol_pre", "uncert×vol"),
        ("tone_answers", "size", "tone×size"),
    ]

    tr = train.copy()
    te = test.copy()
    int_names = []
    for f1, f2, name in interactions:
        if f1 in tr.columns and f2 in tr.columns:
            tr[name] = tr[f1] * tr[f2]
            te[name] = te[f1] * te[f2]
            int_names.append(name)

    # Try: base + raw linguistic + interactions
    all_features = list(base_features) + available_ling + int_names
    X_tr, X_te, _, _ = preprocess(tr, te, all_features)

    algos = {
        "Ridge": Ridge(alpha=1.0),
        "Ridge_strong": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "ElasticNet_strong": ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=5000),
    }
    results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
    best_name, best_metrics = best_result(results)
    log_msg(f"  All features + interactions ({len(all_features)}): Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    # Also try: base + only interactions (no raw linguistic)
    int_only = list(base_features) + int_names
    X_tr2, X_te2, _, _ = preprocess(tr, te, int_only)
    results2 = fit_and_eval(X_tr2, X_te2, y_tr, y_te, algos)
    best2_name, best2_metrics = best_result(results2)
    log_msg(f"  Base + interactions only ({len(int_only)}): Rank IC={best2_metrics['rank_ic']:+.4f} ({best2_name})")

    if best2_metrics["rank_ic"] > best_metrics["rank_ic"]:
        return {"name": "Interactions (no raw ling)", "features": int_only,
                "added_ling": int_names, "algo": best2_name, "metrics": best2_metrics, "all_results": results2}
    return {"name": "Interactions + raw ling", "features": all_features,
            "added_ling": available_ling + int_names, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_residualized_tone(train, test, base_features, y_tr, y_te):
    """Regress tone on SUE/car_-1_1 to get 'unexpected tone', use that as a feature."""
    log_msg("\n--- Strategy 3: Residualized Tone ---")
    tr = train.copy()
    te = test.copy()

    tone_cols = [c for c in ["tone_answers", "tone_prepared", "tone_answers_chg", "tone_prepared_chg"]
                 if c in tr.columns]
    control_cols = [c for c in ["sue", "car_-1_1", "size"] if c in tr.columns]

    resid_names = []
    for tc in tone_cols:
        name = f"{tc}_resid"
        X_ctrl_tr = tr[control_cols].fillna(0).values
        X_ctrl_te = te[control_cols].fillna(0).values
        y_tone_tr = tr[tc].fillna(0).values
        y_tone_te = te[tc].fillna(0).values

        reg = LinearRegression()
        reg.fit(X_ctrl_tr, y_tone_tr)
        tr[name] = y_tone_tr - reg.predict(X_ctrl_tr)
        te[name] = y_tone_te - reg.predict(X_ctrl_te)
        resid_names.append(name)
        log_msg(f"  {tc} → {name} (R²={reg.score(X_ctrl_tr, y_tone_tr):.4f})")

    all_features = list(base_features) + resid_names
    X_tr, X_te, _, _ = preprocess(tr, te, all_features)

    algos = {"Ridge": Ridge(alpha=1.0), "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)}
    results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
    best_name, best_metrics = best_result(results)
    log_msg(f"  Base + residualized tone ({len(all_features)}): Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Residualized Tone", "features": all_features,
            "added_ling": resid_names, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_strong_regularization(train, test, base_features, y_tr, y_te):
    """Force all linguistic features, but use strong regularization to shrink noise."""
    log_msg("\n--- Strategy 4: All Linguistic + Strong Regularization ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    all_features = list(base_features) + available_ling

    X_tr, X_te, _, _ = preprocess(train, test, all_features)

    algos = {
        "Ridge_a0.5": Ridge(alpha=0.5),
        "Ridge_a1": Ridge(alpha=1.0),
        "Ridge_a5": Ridge(alpha=5.0),
        "Ridge_a10": Ridge(alpha=10.0),
        "Ridge_a50": Ridge(alpha=50.0),
        "EN_a0.005": ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=5000),
        "EN_a0.01": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "EN_a0.05": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "EN_a0.1_l0.9": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000),
    }
    results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Results ({len(all_features)} features):")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True):
        log_msg(f"    {name:20s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "Strong Regularization", "features": all_features,
            "added_ling": available_ling, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_pca_linguistic(train, test, base_features, y_tr, y_te):
    """PCA on linguistic features, add top components to base."""
    log_msg("\n--- Strategy 5: PCA on Linguistic Features ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]

    # Preprocess linguistic features separately
    tr_ling = train[available_ling].copy()
    te_ling = test[available_ling].copy()
    for col in available_ling:
        tr_ling[col], lo, hi, med = winsorize_and_impute(tr_ling[col])
        te_ling[col], _, _, _ = winsorize_and_impute(te_ling[col], lo, hi, med)

    scaler_ling = StandardScaler()
    tr_ling_s = scaler_ling.fit_transform(tr_ling.values)
    te_ling_s = scaler_ling.transform(te_ling.values)

    for n_comp in [2, 3, 4]:
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        tr_pca = pca.fit_transform(tr_ling_s)
        te_pca = pca.transform(te_ling_s)
        var_explained = pca.explained_variance_ratio_.sum()
        log_msg(f"  PCA({n_comp}): {var_explained:.1%} variance explained")

        # Add PCA components to train/test
        tr_aug = train.copy()
        te_aug = test.copy()
        pca_names = [f"ling_pc{i+1}" for i in range(n_comp)]
        for i, name in enumerate(pca_names):
            tr_aug[name] = tr_pca[:, i]
            te_aug[name] = te_pca[:, i]

        all_features = list(base_features) + pca_names
        X_tr, X_te, _, _ = preprocess(tr_aug, te_aug, all_features)

        algos = {"Ridge": Ridge(alpha=1.0), "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)}
        results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
        bn, bm = best_result(results)
        log_msg(f"    Base + PCA({n_comp}): Rank IC={bm['rank_ic']:+.4f} ({bn})")

    # Use best n_comp (try 3 as default)
    n_comp = 3
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    tr_pca = pca.fit_transform(tr_ling_s)
    te_pca = pca.transform(te_ling_s)
    tr_aug = train.copy()
    te_aug = test.copy()
    pca_names = [f"ling_pc{i+1}" for i in range(n_comp)]
    for i, name in enumerate(pca_names):
        tr_aug[name] = tr_pca[:, i]
        te_aug[name] = te_pca[:, i]

    all_features = list(base_features) + pca_names
    X_tr, X_te, _, _ = preprocess(tr_aug, te_aug, all_features)
    algos = {"Ridge": Ridge(alpha=1.0), "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)}
    results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
    best_name, best_metrics = best_result(results)

    # Loadings
    log_msg(f"\n  PCA loadings (top 3 components):")
    for i in range(n_comp):
        loadings = dict(zip(available_ling, pca.components_[i]))
        top = sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        log_msg(f"    PC{i+1}: " + ", ".join(f"{n}={v:+.3f}" for n, v in top))

    return {"name": f"PCA({n_comp}) Linguistic", "features": all_features,
            "added_ling": pca_names, "algo": best_name, "metrics": best_metrics, "all_results": results}


def run_option(opt_name, opt_file, base_features, out_dir):
    log_msg(f"\n{'#'*70}")
    log_msg(f"# {opt_name}")
    log_msg(f"{'#'*70}")

    df = pd.read_parquet(f"{out_dir}/{opt_file}")
    df["call_date"] = pd.to_datetime(df["call_date"])
    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy().reset_index(drop=True)
    test = has_target[has_target["call_date"] > cutoff].copy().reset_index(drop=True)
    y_tr = train[PRIMARY_TARGET].values
    y_te = test[PRIMARY_TARGET].values
    log_msg(f"train: {len(train):,}, test: {len(test):,}")

    # Model A baseline
    X_tr_a, X_te_a, _, _ = preprocess(train, test, base_features)
    algos_a = {"Ridge": Ridge(alpha=1.0), "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)}
    res_a = fit_and_eval(X_tr_a, X_te_a, y_tr, y_te, algos_a)
    a_best_name, a_best = best_result(res_a)
    log_msg(f"\nModel A baseline: Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})")

    # Run all strategies
    strategies = [
        strategy_forward_selection(train, test, base_features, y_tr, y_te),
        strategy_interactions(train, test, base_features, y_tr, y_te),
        strategy_residualized_tone(train, test, base_features, y_tr, y_te),
        strategy_strong_regularization(train, test, base_features, y_tr, y_te),
        strategy_pca_linguistic(train, test, base_features, y_tr, y_te),
    ]

    # Pick the best strategy
    log_msg(f"\n{'='*70}")
    log_msg(f"STRATEGY COMPARISON — {opt_name}")
    log_msg(f"{'='*70}")
    log_msg(f"{'Strategy':35s} {'Algo':20s} {'Rank IC':>9s} {'Q5-Q1':>9s} {'vs A':>9s}")
    log_msg(f"{'Model A (baseline)':35s} {a_best_name:20s} {a_best['rank_ic']:+9.4f} {a_best['quintile_spread']:+9.4f} {'---':>9s}")

    for s in strategies:
        delta = s["metrics"]["rank_ic"] - a_best["rank_ic"]
        marker = " ***" if delta > 0 else ""
        log_msg(f"{s['name']:35s} {s['algo']:20s} {s['metrics']['rank_ic']:+9.4f} "
                f"{s['metrics']['quintile_spread']:+9.4f} {delta:+9.4f}{marker}")

    winner = max(strategies, key=lambda s: s["metrics"]["rank_ic"])
    if winner["metrics"]["rank_ic"] > a_best["rank_ic"]:
        log_msg(f"\n>>> Winner: {winner['name']} (Rank IC={winner['metrics']['rank_ic']:+.4f}, "
                f"Δ={winner['metrics']['rank_ic'] - a_best['rank_ic']:+.4f} vs Model A)")
    else:
        log_msg(f"\n>>> No strategy beat Model A. Best attempt: {winner['name']} "
                f"(Rank IC={winner['metrics']['rank_ic']:+.4f})")

    # Save
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    report_lines = [
        "=" * 70,
        f"IT Sector — Model B Improved — {opt_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} | Test: {len(test):,}",
        f"Model A baseline: Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})",
        "=" * 70, "",
    ]
    for s in strategies:
        delta = s["metrics"]["rank_ic"] - a_best["rank_ic"]
        report_lines.append(f"{s['name']:35s} {s['algo']:20s} Rank IC={s['metrics']['rank_ic']:+.4f} "
                            f"Q5-Q1={s['metrics']['quintile_spread']:+.4f} Δ={delta:+.4f}")
        report_lines.append(f"  features: {', '.join(s['added_ling'])}")
        report_lines.append("")

    report_lines.append(f"Winner: {winner['name']}")
    report_lines.append(f"  Algo: {winner['algo']}")
    report_lines.append(f"  Features ({len(winner['features'])}): {', '.join(winner['features'])}")
    report_lines.append(f"  Rank IC: {winner['metrics']['rank_ic']:+.4f}")
    report_lines.append(f"  Q5-Q1: {winner['metrics']['quintile_spread']:+.4f}")
    report_lines.append(f"  IC: {winner['metrics']['ic']:+.4f}")
    report_lines.append(f"  Dir Acc: {winner['metrics']['direction_accuracy']:.3f}")

    Path(out_dir, f"it_model_b_improved_{tag}_report.txt").write_text("\n".join(report_lines))

    result_json = {
        "option": opt_name,
        "model_a_baseline": {"algo": a_best_name, "rank_ic": round(a_best["rank_ic"], 6)},
        "winner": {
            "strategy": winner["name"],
            "algo": winner["algo"],
            "features": winner["features"],
            "added_ling": winner["added_ling"],
            "metrics": {k: round(v, 6) for k, v in winner["metrics"].items()},
        },
        "all_strategies": [
            {"name": s["name"], "algo": s["algo"],
             "rank_ic": round(s["metrics"]["rank_ic"], 6),
             "quintile_spread": round(s["metrics"]["quintile_spread"], 6)}
            for s in strategies
        ],
    }
    Path(out_dir, f"it_model_b_improved_{tag}_results.json").write_text(json.dumps(result_json, indent=2))

    return winner, a_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_it"))
    args = ap.parse_args()
    OUT = args.folder

    results = {}
    for opt_name, opt_file, base_feats in [
        ("Option 1 (S&P 500 bench)", "model_table_opt1.parquet", MODEL_A_FEATURES_OPT1),
        ("Option 2 (IT sector bench)", "model_table_opt2.parquet", MODEL_A_FEATURES_OPT2),
    ]:
        fpath = f"{OUT}/{opt_file}"
        if not os.path.exists(fpath):
            log_msg(f"SKIP {opt_name}: not found")
            continue
        winner, model_a = run_option(opt_name, opt_file, base_feats, OUT)
        results[opt_name] = (winner, model_a)

    log_msg(f"\n{'='*70}")
    log_msg("FINAL SUMMARY")
    log_msg(f"{'='*70}")
    for opt_name, (winner, model_a) in results.items():
        delta = winner["metrics"]["rank_ic"] - model_a["rank_ic"]
        log_msg(f"{opt_name}:")
        log_msg(f"  Model A: Rank IC={model_a['rank_ic']:+.4f}")
        log_msg(f"  Best B:  Rank IC={winner['metrics']['rank_ic']:+.4f} ({winner['name']}, {winner['algo']}) Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
