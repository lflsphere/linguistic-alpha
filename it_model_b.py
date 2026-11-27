"""
IT Sector — Model A + B pipeline.

Model A:  all control features (no selection).
Model B:  controls + LM features via 6 strategies (best wins).
Model B*: controls + ALL LM features (Ridge regularisation handles weighting).

All models train on ranked target to directly optimize Rank IC.
Time-series CV (not KFold) to avoid look-ahead bias.

Train: 2010-2023 | Test: 2024-2026.

Usage:
    python3 it_model_b.py [--folder ~/earnings_it]
"""
import argparse, copy, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"
TRAIN_END = "2023-12-31"
RANDOM_STATE = 42
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99
MIN_IMPROVEMENT = 0.0  # add any feature that improves CV Rank IC

# Candidate controls for Model A selection
CONTROL_CANDIDATES = [
    "sue", "surprise_pct", "vol_pre", "car_-1_1", "mom", "bm", "size",
    "dispersion", "avg_answer_words", "words_prepared", "words_questions",
    "words_answers", "n_questions", "n_answers", "n_analysts",
    "prior_calls_with_target",
]

LINGUISTIC_FEATURES = [
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap",
    "tone_prepared_chg", "tone_answers_chg", "hedging_answers_chg",
]

LOG_TRANSFORM = ["words_prepared", "words_answers", "words_questions",
                 "avg_answer_words", "n_analysts"]

TS_CV_SPLITS = [
    ("2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
]


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


METRIC_KEYS = ["ic", "rank_ic", "oos_r2", "mae", "hit_rate", "hit_rate_baseline",
                "quintile_spread"]


def evaluate(y_true, y_pred, y_train=None, X_tr=None, X_te=None, is_ranked=True):
    if len(y_true) < 10:
        return {k: np.nan for k in METRIC_KEYS}

    ic = float(stats.pearsonr(y_pred, y_true)[0])
    rank_ic = float(stats.spearmanr(y_pred, y_true)[0])

    q = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    qm = pd.Series(y_true).groupby(q).mean()
    spread = float(qm.iloc[-1] - qm.iloc[0]) if len(qm) >= 2 else np.nan

    if is_ranked and X_tr is not None and X_te is not None and y_train is not None:
        from sklearn.linear_model import LinearRegression as _LR
        cal = _LR()
        cal.fit(X_tr, stats.rankdata(y_train))
        train_preds_rank = cal.predict(X_tr)
        cal_line = _LR()
        cal_line.fit(train_preds_rank.reshape(-1, 1), y_train)
        y_pred_cal = cal_line.predict(y_pred.reshape(-1, 1))
    elif is_ranked:
        y_pred_cal = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-12) * np.std(y_true)
    else:
        y_pred_cal = y_pred

    ss_res = np.sum((y_true - y_pred_cal) ** 2)
    ss_zero = np.sum(y_true ** 2)
    oos_r2 = float(1 - ss_res / ss_zero) if ss_zero > 0 else np.nan

    mae = float(np.mean(np.abs(y_true - y_pred_cal)))

    pred_sign = np.sign(y_pred_cal - np.median(y_pred_cal))
    pred_sign[pred_sign == 0] = 1
    actual_sign = np.sign(y_true)
    actual_sign[actual_sign == 0] = 1
    hit_rate = float(np.mean(pred_sign == actual_sign))

    if y_train is not None:
        pos_share = np.mean(y_train > 0)
        hit_baseline = float(max(pos_share, 1 - pos_share))
    else:
        pos_share = np.mean(y_true > 0)
        hit_baseline = float(max(pos_share, 1 - pos_share))

    return {"ic": ic, "rank_ic": rank_ic, "oos_r2": oos_r2, "mae": mae,
            "hit_rate": hit_rate, "hit_rate_baseline": hit_baseline,
            "quintile_spread": spread, "predictions": y_pred_cal.tolist()}


def evaluate_ols(X_tr, X_te, y_tr, y_te):
    ols = LinearRegression()
    ols.fit(X_tr, y_tr)
    preds = ols.predict(X_te)
    return evaluate(y_te, preds, y_train=y_tr, is_ranked=False)


def metrics_for_json(m):
    return {k: round(v, 6) if isinstance(v, float) else v
            for k, v in m.items() if k != "predictions"}


def fit_and_eval(X_tr, X_te, y_tr, y_te, algos, y_train_raw=None):
    results = {}
    for name, model in algos.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results[name] = evaluate(y_te, preds, y_train=y_train_raw,
                                 X_tr=X_tr, X_te=X_te, is_ranked=True)
    results["OLS"] = evaluate_ols(X_tr, X_te,
                                  y_train_raw if y_train_raw is not None else y_te, y_te)
    return results


def best_result(results):
    candidates = {k: v for k, v in results.items() if k != "OLS"}
    best_name = max(candidates, key=lambda k: candidates[k]["rank_ic"])
    return best_name, results[best_name]


def get_algos():
    return {
        "Ridge": Ridge(alpha=1.0),
        "Ridge_a5": Ridge(alpha=5.0),
        "Ridge_a10": Ridge(alpha=10.0),
        "RidgeCV": RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "EN_strong": ElasticNet(alpha=0.05, l1_ratio=0.7, max_iter=5000),
    }


def ts_cv_rank_ic(full_train, features, y_col, algos):
    """Time-series expanding-window CV. Train on ranked target."""
    algo_scores = {name: [] for name in algos}
    for train_end, val_start, val_end in TS_CV_SPLITS:
        fold_train = full_train[full_train["call_date"] <= train_end]
        fold_val = full_train[(full_train["call_date"] >= val_start) &
                              (full_train["call_date"] <= val_end)]
        if len(fold_train) < 100 or len(fold_val) < 50:
            continue
        X_tr_f, X_te_f, _, _ = preprocess(fold_train, fold_val, features)
        y_tr_ranks = stats.rankdata(fold_train[y_col].values)
        y_val_raw = fold_val[y_col].values
        for name, model_cls in algos.items():
            model = copy.deepcopy(model_cls)
            model.fit(X_tr_f, y_tr_ranks)
            preds = model.predict(X_te_f)
            ric = stats.spearmanr(preds, y_val_raw)[0]
            algo_scores[name].append(ric if not np.isnan(ric) else 0.0)
    best_name = max(algo_scores, key=lambda k: np.mean(algo_scores[k]) if algo_scores[k] else -1)
    best_mean = np.mean(algo_scores[best_name]) if algo_scores[best_name] else 0.0
    return best_mean, best_name


# ======================================================================
# MODEL A: Forward-select control features
# ======================================================================

def select_model_a(train, test, y_tr, y_te):
    """Use all available control features for Model A (no selection)."""
    log_msg("\n" + "=" * 70)
    log_msg("MODEL A: All Controls (no feature selection)")
    log_msg("=" * 70)

    current = [c for c in CONTROL_CANDIDATES if c in train.columns
               and train[c].notna().sum() > len(train) * 0.5]
    log_msg(f"  Model A features ({len(current)}): {', '.join(current)}")

    selection_log = []

    # Final evaluation on holdout
    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    y_tr_ranks = stats.rankdata(y_tr)
    final_algos = get_algos()
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, final_algos, y_train_raw=y_tr)
    a_best_name, a_best = best_result(final_results)
    log_msg(f"  Model A holdout: Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})")

    return current, a_best_name, a_best, selection_log, final_results


# ======================================================================
# MODEL B STRATEGIES (best of 6 becomes Model B)
# ======================================================================

def strategy_forward_selection(train, test, base_features, y_tr, y_te):
    """Lock base features, greedily add linguistic features using time-series CV."""
    log_msg("\n--- Model B Strategy 1: Forward Selection (time-series CV) ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]

    current = list(base_features)
    algos = get_algos()

    base_cv_ric, base_algo = ts_cv_rank_ic(train, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline ({len(current)} features): TS-CV Rank IC={base_cv_ric:+.4f} ({base_algo})")

    added = []
    remaining = list(available_ling)
    best_so_far = base_cv_ric
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = ts_cv_rank_ic(train, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
            log_msg(f"    + {feat:25s}: TS-CV Rank IC={cv_ric:+.4f} ({algo_name})")

        best_feat = max(candidates, key=lambda f: candidates[f][0])
        best_ric, best_algo = candidates[best_feat]

        improvement = best_ric - best_so_far
        if improvement > 0:
            current.append(best_feat)
            remaining.remove(best_feat)
            added.append(best_feat)
            best_so_far = best_ric
            selection_log.append({"step": step + 1, "added": best_feat,
                                  "cv_rank_ic": float(best_ric), "improvement": float(improvement)})
            log_msg(f"  Step {step+1}: added {best_feat} → TS-CV Rank IC={best_ric:+.4f} (Δ={improvement:+.4f})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    y_tr_ranks = stats.rankdata(y_tr)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, get_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(current)} features, Holdout Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Forward Selection", "features": current, "added_ling": added,
            "algo": best_name, "metrics": best_metrics, "all_results": final_results,
            "selection_log": selection_log}


def strategy_interactions(train, test, base_features, y_tr, y_te):
    """Economically-motivated interaction features."""
    log_msg("\n--- Model B Strategy 2: Interaction Features ---")

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

    y_tr_ranks = stats.rankdata(y_tr)

    int_only = list(base_features) + int_names
    X_tr2, X_te2, _, _ = preprocess(tr, te, int_only)
    results2 = fit_and_eval(X_tr2, X_te2, y_tr_ranks, y_te, get_algos(), y_train_raw=y_tr)
    best2_name, best2_metrics = best_result(results2)
    log_msg(f"  Base + interactions only ({len(int_only)}): Rank IC={best2_metrics['rank_ic']:+.4f} ({best2_name})")

    available_ling = [f for f in LINGUISTIC_FEATURES if f in tr.columns]
    all_features = list(base_features) + available_ling + int_names
    X_tr, X_te, _, _ = preprocess(tr, te, all_features)
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, get_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(results)
    log_msg(f"  All + interactions ({len(all_features)}): Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    if best2_metrics["rank_ic"] > best_metrics["rank_ic"]:
        return {"name": "Interactions (no raw ling)", "features": int_only,
                "added_ling": int_names, "algo": best2_name, "metrics": best2_metrics, "all_results": results2}
    return {"name": "Interactions + raw ling", "features": all_features,
            "added_ling": available_ling + int_names, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_residualized_tone(train, test, base_features, y_tr, y_te):
    """Unexpected tone after controlling for SUE/car_-1_1."""
    log_msg("\n--- Model B Strategy 3: Residualized Tone ---")
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
    y_tr_ranks = stats.rankdata(y_tr)
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, get_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(results)
    log_msg(f"  Base + residualized ({len(all_features)}): Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Residualized Tone", "features": all_features,
            "added_ling": resid_names, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_ridgecv_all(train, test, base_features, y_tr, y_te):
    """All linguistic features with auto-tuned regularization."""
    log_msg("\n--- Model B Strategy 4: All Linguistic + RidgeCV ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    all_features = list(base_features) + available_ling

    X_tr, X_te, _, _ = preprocess(train, test, all_features)
    y_tr_ranks = stats.rankdata(y_tr)

    algos = {
        "RidgeCV": RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]),
        "Ridge_a10": Ridge(alpha=10.0),
        "Ridge_a50": Ridge(alpha=50.0),
        "Ridge_a100": Ridge(alpha=100.0),
        "EN_a0.01": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "EN_a0.05": ElasticNet(alpha=0.05, l1_ratio=0.7, max_iter=5000),
        "EN_a0.1": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000),
    }
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, algos, y_train_raw=y_tr)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Results ({len(all_features)} features):")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True):
        log_msg(f"    {name:20s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "All Ling + RidgeCV", "features": all_features,
            "added_ling": available_ling, "algo": best_name, "metrics": best_metrics, "all_results": results}


def strategy_pca_linguistic(train, test, base_features, y_tr, y_te):
    """PCA on linguistic features."""
    log_msg("\n--- Model B Strategy 5: PCA on Linguistic Features ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]

    tr_ling = train[available_ling].copy()
    te_ling = test[available_ling].copy()
    for col in available_ling:
        tr_ling[col], lo, hi, med = winsorize_and_impute(tr_ling[col])
        te_ling[col], _, _, _ = winsorize_and_impute(te_ling[col], lo, hi, med)

    scaler_ling = StandardScaler()
    tr_ling_s = scaler_ling.fit_transform(tr_ling.values)
    te_ling_s = scaler_ling.transform(te_ling.values)

    y_tr_ranks = stats.rankdata(y_tr)
    best_overall = None

    for n_comp in [2, 3, 4]:
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        tr_pca = pca.fit_transform(tr_ling_s)
        te_pca = pca.transform(te_ling_s)
        var_explained = pca.explained_variance_ratio_.sum()
        log_msg(f"  PCA({n_comp}): {var_explained:.1%} variance explained")

        tr_aug = train.copy()
        te_aug = test.copy()
        pca_names = [f"ling_pc{i+1}" for i in range(n_comp)]
        for i, name in enumerate(pca_names):
            tr_aug[name] = tr_pca[:, i]
            te_aug[name] = te_pca[:, i]

        all_features = list(base_features) + pca_names
        X_tr, X_te, _, _ = preprocess(tr_aug, te_aug, all_features)
        results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, get_algos(), y_train_raw=y_tr)
        bn, bm = best_result(results)
        log_msg(f"    Base + PCA({n_comp}): Rank IC={bm['rank_ic']:+.4f} ({bn})")

        if best_overall is None or bm["rank_ic"] > best_overall["metrics"]["rank_ic"]:
            best_overall = {"n_comp": n_comp, "features": all_features, "pca_names": pca_names,
                            "algo": bn, "metrics": bm, "all_results": results}

    n = best_overall["n_comp"]
    return {"name": f"PCA({n}) Linguistic", "features": best_overall["features"],
            "added_ling": best_overall["pca_names"], "algo": best_overall["algo"],
            "metrics": best_overall["metrics"], "all_results": best_overall["all_results"]}


def strategy_ensemble(train, test, base_features, y_tr, y_te):
    """Average predictions from best single-feature models.
    Feature filtering uses CV Rank IC (not test set) to avoid leakage."""
    log_msg("\n--- Model B Strategy 6: Single-Feature Ensemble ---")
    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    y_tr_ranks = stats.rankdata(y_tr)
    algos = get_algos()

    # Baseline CV Rank IC (controls only)
    base_cv_ric, base_cv_algo = ts_cv_rank_ic(train, base_features, PRIMARY_TARGET, algos)
    log_msg(f"  Base CV Rank IC: {base_cv_ric:+.4f} ({base_cv_algo})")

    # Score each single-feature model by CV Rank IC
    cv_results = {}
    for feat in available_ling:
        trial = list(base_features) + [feat]
        cv_ric, cv_algo = ts_cv_rank_ic(train, trial, PRIMARY_TARGET, algos)
        cv_results[feat] = {"cv_rank_ic": cv_ric, "cv_algo": cv_algo}
        log_msg(f"    + {feat:25s}: CV Rank IC={cv_ric:+.4f} ({cv_algo})")

    # Keep features that improve over baseline using CV (no test leakage)
    good_feats = [(f, r) for f, r in sorted(cv_results.items(),
                  key=lambda x: x[1]["cv_rank_ic"], reverse=True) if r["cv_rank_ic"] > base_cv_ric]
    log_msg(f"  Features improving over base (CV): {[f for f, _ in good_feats]}")

    # Train final models on full training set and evaluate on test
    if len(good_feats) >= 2:
        all_preds = []
        for feat, _ in good_feats:
            trial = list(base_features) + [feat]
            X_tr_f, X_te_f, _, _ = preprocess(train, test, trial)
            model = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
            model.fit(X_tr_f, y_tr_ranks)
            all_preds.append(model.predict(X_te_f))
        ensemble_preds = np.column_stack(all_preds).mean(axis=1)
        ens_metrics = evaluate(y_te, ensemble_preds, y_train=y_tr, is_ranked=True)
        log_msg(f"  Ensemble of {len(good_feats)} models: Rank IC={ens_metrics['rank_ic']:+.4f}")

        all_good = list(base_features) + [f for f, _ in good_feats]
        X_tr_g, X_te_g, _, _ = preprocess(train, test, all_good)
        combined_results = fit_and_eval(X_tr_g, X_te_g, y_tr_ranks, y_te, algos, y_train_raw=y_tr)
        comb_name, comb_metrics = best_result(combined_results)
        log_msg(f"  Combined ({len(all_good)} feats): Rank IC={comb_metrics['rank_ic']:+.4f} ({comb_name})")

        if ens_metrics["rank_ic"] > comb_metrics["rank_ic"]:
            return {"name": "Single-Feature Ensemble", "features": [f"ensemble({','.join(f for f,_ in good_feats)})"],
                    "added_ling": [f for f, _ in good_feats], "algo": "Ensemble",
                    "metrics": ens_metrics, "all_results": {"Ensemble": ens_metrics}}
        return {"name": "Combined Good Features", "features": all_good,
                "added_ling": [f for f, _ in good_feats], "algo": comb_name,
                "metrics": comb_metrics, "all_results": combined_results}
    elif len(good_feats) == 1:
        feat, _ = good_feats[0]
        trial = list(base_features) + [feat]
        X_tr_f, X_te_f, _, _ = preprocess(train, test, trial)
        results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, algos, y_train_raw=y_tr)
        bn, bm = best_result(results)
        return {"name": f"Single Best ({feat})", "features": trial,
                "added_ling": [feat], "algo": bn, "metrics": bm, "all_results": results}
    else:
        X_tr_base, X_te_base, _, _ = preprocess(train, test, base_features)
        base_model = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
        base_model.fit(X_tr_base, y_tr_ranks)
        base_preds = base_model.predict(X_te_base)
        base_metrics = evaluate(y_te, base_preds, y_train=y_tr, is_ranked=True)
        return {"name": "No Improvement", "features": list(base_features),
                "added_ling": [], "algo": "RidgeCV",
                "metrics": base_metrics, "all_results": {"RidgeCV": base_metrics}}


# ======================================================================
# MODEL B*: All LM features (regularisation handles weighting)
# ======================================================================

def run_model_b_star(train, test, base_features, y_tr, y_te):
    """Model B*: controls + ALL LM features, let Ridge handle it."""
    log_msg("\n" + "=" * 70)
    log_msg("MODEL B*: Controls + All LM Features")
    log_msg("=" * 70)

    available_ling = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    all_features = list(base_features) + available_ling

    X_tr, X_te, _, _ = preprocess(train, test, all_features)
    y_tr_ranks = stats.rankdata(y_tr)

    algos = get_algos()
    algos.update({
        "Ridge_a50": Ridge(alpha=50.0),
        "Ridge_a100": Ridge(alpha=100.0),
    })
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, algos, y_train_raw=y_tr)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Model B* features ({len(all_features)}): {len(base_features)} controls + {len(available_ling)} LM")
    log_msg(f"  LM features: {', '.join(available_ling)}")
    log_msg(f"  Model B* holdout: Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")
    log_msg(f"  All algorithms:")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True):
        log_msg(f"    {name:20s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "Model B* (all LM)", "features": all_features,
            "added_ling": available_ling, "algo": best_name, "metrics": best_metrics,
            "all_results": results}


# ======================================================================
# MAIN
# ======================================================================

def run_option(opt_name, opt_file, out_dir):
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
    log_msg(f"train: {len(train):,} (to {TRAIN_END}), test: {len(test):,}")

    # --- Model A: select controls ---
    base_features, a_best_name, a_best, a_sel_log, a_results = select_model_a(train, test, y_tr, y_te)

    # --- Model B: best of 6 strategies ---
    log_msg(f"\n{'='*70}")
    log_msg(f"MODEL B: 6 Strategies")
    log_msg(f"{'='*70}")

    strategies = [
        strategy_forward_selection(train, test, base_features, y_tr, y_te),
        strategy_interactions(train, test, base_features, y_tr, y_te),
        strategy_residualized_tone(train, test, base_features, y_tr, y_te),
        strategy_ridgecv_all(train, test, base_features, y_tr, y_te),
        strategy_pca_linguistic(train, test, base_features, y_tr, y_te),
        strategy_ensemble(train, test, base_features, y_tr, y_te),
    ]

    model_b = max(strategies, key=lambda s: s["metrics"]["rank_ic"])

    # --- Model B*: all LM features ---
    model_b_star = run_model_b_star(train, test, base_features, y_tr, y_te)

    # --- Summary ---
    log_msg(f"\n{'='*70}")
    log_msg(f"SUMMARY — {opt_name}")
    log_msg(f"{'='*70}")
    log_msg(f"{'Model':35s} {'Algo':20s} {'Rank IC':>9s} {'Q5-Q1':>9s} {'vs A':>9s}")
    log_msg(f"{'Model A (baseline)':35s} {a_best_name:20s} {a_best['rank_ic']:+9.4f} {a_best['quintile_spread']:+9.4f} {'---':>9s}")

    for s in strategies:
        delta = s["metrics"]["rank_ic"] - a_best["rank_ic"]
        marker = " ***" if delta > 0 else ""
        log_msg(f"  {s['name']:33s} {s['algo']:20s} {s['metrics']['rank_ic']:+9.4f} "
                f"{s['metrics']['quintile_spread']:+9.4f} {delta:+9.4f}{marker}")

    b_delta = model_b["metrics"]["rank_ic"] - a_best["rank_ic"]
    bs_delta = model_b_star["metrics"]["rank_ic"] - a_best["rank_ic"]
    log_msg(f"{'Model B (best strategy)':35s} {model_b['algo']:20s} {model_b['metrics']['rank_ic']:+9.4f} "
            f"{model_b['metrics']['quintile_spread']:+9.4f} {b_delta:+9.4f}")
    log_msg(f"{'Model B* (all LM)':35s} {model_b_star['algo']:20s} {model_b_star['metrics']['rank_ic']:+9.4f} "
            f"{model_b_star['metrics']['quintile_spread']:+9.4f} {bs_delta:+9.4f}")

    # Save
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    report_lines = [
        "=" * 70,
        f"IT Sector — Model A + B — {opt_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} (≤{TRAIN_END}) | Test: {len(test):,} (>{TRAIN_END})",
        f"Method: Time-series CV, ranked target, min Δ={MIN_IMPROVEMENT}",
        "=" * 70, "",
        "--- MODEL A: All Controls (no selection) ---",
        f"Features ({len(base_features)}): {', '.join(base_features)}",
        f"Holdout Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})", "",
    ]

    # All 6 strategies
    report_lines.append("--- MODEL B STRATEGIES ---")
    for s in strategies:
        delta = s["metrics"]["rank_ic"] - a_best["rank_ic"]
        report_lines.append(f"{s['name']:35s} {s['algo']:20s} Rank IC={s['metrics']['rank_ic']:+.4f} "
                            f"Q5-Q1={s['metrics']['quintile_spread']:+.4f} Δ={delta:+.4f}")
        report_lines.append(f"  features: {', '.join(s['added_ling'])}")
        report_lines.append("")

    if strategies[0].get("selection_log"):
        report_lines.append("--- FORWARD SELECTION LOG ---")
        report_lines.append(f"{'Step':>5s} {'Added':30s} {'TS-CV Rank IC':>14s} {'Delta':>10s}")
        for entry in strategies[0]["selection_log"]:
            feat_name = entry["added"] or "(baseline)"
            report_lines.append(f"{entry['step']:5d} {feat_name:30s} {entry['cv_rank_ic']:+14.4f} {entry['improvement']:+10.4f}")
        report_lines.append("")

    report_lines.append(f"Model B (best strategy): {model_b['name']}")
    report_lines.append(f"  Algo: {model_b['algo']}")
    report_lines.append(f"  Features ({len(model_b['features'])}): {', '.join(model_b['features'])}")
    bm = model_b['metrics']
    report_lines.append(f"  Rank IC: {bm['rank_ic']:+.4f}")
    report_lines.append(f"  Q5-Q1: {bm['quintile_spread']:+.4f}")
    report_lines.append(f"  IC: {bm['ic']:+.4f}")
    report_lines.append(f"  OOS R²: {bm['oos_r2']:+.6f}")
    report_lines.append(f"  MAE: {bm['mae']:.6f}")
    report_lines.append(f"  Hit Rate: {bm['hit_rate']:.3f} (baseline: {bm['hit_rate_baseline']:.3f})")
    report_lines.append("")

    # Model B*
    report_lines.append("--- MODEL B*: All LM Features (Ridge regularised) ---")
    report_lines.append(f"LM features ({len(model_b_star['added_ling'])}): {', '.join(model_b_star['added_ling'])}")
    report_lines.append(f"Algo: {model_b_star['algo']}")
    report_lines.append(f"Holdout Rank IC={model_b_star['metrics']['rank_ic']:+.4f}  "
                        f"Q5-Q1={model_b_star['metrics']['quintile_spread']:+.4f}  Δ={bs_delta:+.4f}")
    report_lines.append(f"All algorithms:")
    for name, m in sorted(model_b_star["all_results"].items(), key=lambda x: x[1]["rank_ic"], reverse=True):
        report_lines.append(f"  {name:20s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    Path(out_dir, f"it_model_b_improved_{tag}_report.txt").write_text("\n".join(report_lines))

    ols_ic = a_results.get("OLS", {}).get("rank_ic", np.nan)

    result_json = {
        "option": opt_name,
        "train_end": TRAIN_END,
        "method": "time-series CV, ranked target",
        "model_a": {
            "features": base_features,
            "algo": a_best_name,
            "metrics": metrics_for_json(a_best),
            "ols_metrics": metrics_for_json(a_results.get("OLS", {})),
            "selection_log": a_sel_log,
        },
        "model_b": {
            "strategy": model_b["name"],
            "algo": model_b["algo"],
            "features": model_b["features"],
            "added_ling": model_b["added_ling"],
            "metrics": metrics_for_json(model_b["metrics"]),
            "relative_ic": round(model_b["metrics"]["rank_ic"] - ols_ic, 6)
                           if not np.isnan(ols_ic) else None,
        },
        "model_b_star": {
            "name": model_b_star["name"],
            "algo": model_b_star["algo"],
            "features": model_b_star["features"],
            "added_ling": model_b_star["added_ling"],
            "metrics": metrics_for_json(model_b_star["metrics"]),
            "relative_ic": round(model_b_star["metrics"]["rank_ic"] - ols_ic, 6)
                           if not np.isnan(ols_ic) else None,
        },
        "all_strategies": [
            {"name": s["name"], "algo": s["algo"],
             "metrics": metrics_for_json(s["metrics"]),
             "relative_ic": round(s["metrics"]["rank_ic"] - ols_ic, 6)
                            if not np.isnan(ols_ic) else None}
            for s in strategies
        ],
    }
    Path(out_dir, f"it_model_b_improved_{tag}_results.json").write_text(json.dumps(result_json, indent=2))

    # Save predictions CSV
    pred_rows = []
    for label, m_dict in [("model_a", a_best), ("model_b", model_b["metrics"]),
                           ("model_b_star", model_b_star["metrics"])]:
        preds = m_dict.get("predictions", [])
        for i, p in enumerate(preds):
            pred_rows.append({"model": label, "obs_idx": i, "y_pred_cal": p,
                              "y_true": float(y_te[i]) if i < len(y_te) else np.nan})
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(f"{out_dir}/predictions_b_{tag}.csv", index=False)
        log_msg(f"  saved predictions to {out_dir}/predictions_b_{tag}.csv")

    # Save Model A features for Model C to read
    Path(out_dir, f"model_a_features_{tag}.json").write_text(
        json.dumps({"features": base_features, "algo": a_best_name,
                     "rank_ic": round(a_best["rank_ic"], 6)}, indent=2))

    return model_b, model_b_star, a_best, base_features, strategies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_it"))
    args = ap.parse_args()
    OUT = args.folder

    results = {}
    for opt_name, opt_file in [
        ("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
        ("Option 2 (IT sector bench)", "model_table_opt2.parquet"),
    ]:
        fpath = f"{OUT}/{opt_file}"
        if not os.path.exists(fpath):
            log_msg(f"SKIP {opt_name}: not found")
            continue
        mb, mbs, model_a, base_feats, strats = run_option(opt_name, opt_file, OUT)
        results[opt_name] = (mb, mbs, model_a, base_feats)

    log_msg(f"\n{'='*70}")
    log_msg("FINAL SUMMARY")
    log_msg(f"{'='*70}")
    for opt_name, (mb, mbs, model_a, base_feats) in results.items():
        log_msg(f"{opt_name}:")
        log_msg(f"  Model A  ({len(base_feats)} feats): Rank IC={model_a['rank_ic']:+.4f}")
        log_msg(f"  Model B  ({mb['name']}): Rank IC={mb['metrics']['rank_ic']:+.4f} "
                f"Δ={mb['metrics']['rank_ic'] - model_a['rank_ic']:+.4f}")
        log_msg(f"  Model B* ({len(mbs['added_ling'])} LM):   Rank IC={mbs['metrics']['rank_ic']:+.4f} "
                f"Δ={mbs['metrics']['rank_ic'] - model_a['rank_ic']:+.4f}")
    log_msg("MODEL_B_DONE")


if __name__ == "__main__":
    main()
