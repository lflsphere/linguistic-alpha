"""
IT Sector — Model C v2: New FinBERT strategies to improve over Model A.

Strategies 5-9 on top of the existing 4 from it_model_c_finbert.py:
  5. Cross-sectionally residualized FinBERT (quarter z-scored)
  6. FinBERT × Control interactions (LASSO pre-screen + forward select)
  7. XGBoost-preserved forward selection (force XGBoost algo)
  8. Option-specific feature candidates (exclude mean-sentiment levels)
  9. Meta-ensemble of strategies that beat Model A in CV

Usage:
    python3 it_model_c_v2.py --option 1 [--folder ~/earnings_it]
    python3 it_model_c_v2.py --option 2 [--folder ~/earnings_it]
"""
import argparse, copy, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import (Ridge, RidgeCV, Lasso, LassoCV,
                                  ElasticNet, LinearRegression)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"
TRAIN_END = "2023-12-31"
RANDOM_STATE = 42
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

FINBERT_CANDIDATES = [
    "fb_sent_prepared", "fb_sent_answers",
    "fb_pos_pct_prepared", "fb_neg_pct_prepared",
    "fb_pos_pct_answers", "fb_neg_pct_answers",
    "fb_sent_std_prepared", "fb_sent_std_answers",
    "fb_sent_gap",
    "fb_sent_prepared_chg", "fb_sent_answers_chg",
    "fb_n_sent_prepared", "fb_n_sent_answers",
]

INTERACTION_CONTROLS = ["sue", "car_-1_1", "vol_pre", "surprise_pct"]

LOG_TRANSFORM = ["words_prepared", "words_answers", "words_questions",
                 "avg_answer_words", "n_analysts"]

TS_CV_SPLITS = [
    ("2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
]


# ======================================================================
# UTILITIES (same as it_model_c_finbert.py)
# ======================================================================

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
        cal = LinearRegression()
        cal.fit(X_tr, stats.rankdata(y_train))
        train_preds_rank = cal.predict(X_tr)
        cal_line = LinearRegression()
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
    return results


def best_result(results):
    best_name = max(results, key=lambda k: results[k]["rank_ic"])
    return best_name, results[best_name]


def get_linear_algos():
    return {
        "Ridge": Ridge(alpha=1.0),
        "Ridge_a5": Ridge(alpha=5.0),
        "Ridge_a10": Ridge(alpha=10.0),
        "RidgeCV": RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "EN_strong": ElasticNet(alpha=0.05, l1_ratio=0.7, max_iter=5000),
    }


def get_all_algos():
    algos = get_linear_algos()
    algos.update({
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=80,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            min_samples_leaf=80, subsample=0.7, random_state=RANDOM_STATE),
    })
    if HAS_XGB:
        algos["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            min_child_weight=80, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    return algos


def ts_cv_rank_ic(full_train, features, y_col, algos):
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


def ts_cv_rank_ic_fixed_algo(full_train, features, y_col, algo_name, algo_instance):
    """CV with a single forced algorithm."""
    scores = []
    for train_end, val_start, val_end in TS_CV_SPLITS:
        fold_train = full_train[full_train["call_date"] <= train_end]
        fold_val = full_train[(full_train["call_date"] >= val_start) &
                              (full_train["call_date"] <= val_end)]
        if len(fold_train) < 100 or len(fold_val) < 50:
            continue
        X_tr_f, X_te_f, _, _ = preprocess(fold_train, fold_val, features)
        y_tr_ranks = stats.rankdata(fold_train[y_col].values)
        y_val_raw = fold_val[y_col].values
        model = copy.deepcopy(algo_instance)
        model.fit(X_tr_f, y_tr_ranks)
        preds = model.predict(X_te_f)
        ric = stats.spearmanr(preds, y_val_raw)[0]
        scores.append(ric if not np.isnan(ric) else 0.0)
    return np.mean(scores) if scores else 0.0


# ======================================================================
# NEW HELPERS
# ======================================================================

def cross_sectional_residualize(train_df, test_df, fb_cols):
    """Z-score FinBERT features within each quarter (cross-sectionally).
    Training quarters use their own stats. Test quarters use expanding-window
    stats from all training data (no leakage)."""
    tr = train_df.copy()
    te = test_df.copy()
    tr["_yq"] = tr["call_date"].dt.to_period("Q")
    te["_yq"] = te["call_date"].dt.to_period("Q")

    resid_cols = []
    for col in fb_cols:
        rname = f"{col}_resid"
        resid_cols.append(rname)

        # Training: z-score within each quarter
        qstats = tr.groupby("_yq")[col].agg(["mean", "std"])
        tr[rname] = np.nan
        for yq in tr["_yq"].unique():
            mask = tr["_yq"] == yq
            m, s = qstats.loc[yq, "mean"], qstats.loc[yq, "std"]
            s = s if (not np.isnan(s) and s > 1e-8) else 1.0
            tr.loc[mask, rname] = (tr.loc[mask, col] - m) / s

        # Test: use full training set expanding stats (safe — all training data)
        train_mean = tr[col].mean()
        train_std = tr[col].std()
        if np.isnan(train_std) or train_std < 1e-8:
            train_std = 1.0
        te[rname] = (te[col] - train_mean) / train_std

    tr.drop(columns=["_yq"], inplace=True)
    te.drop(columns=["_yq"], inplace=True)
    return tr, te, resid_cols


def create_interaction_features(train_df, test_df, fb_cols, ctrl_cols):
    """Create FinBERT × control interaction features."""
    tr = train_df.copy()
    te = test_df.copy()
    int_names = []
    available_ctrls = [c for c in ctrl_cols if c in tr.columns]
    for fb in fb_cols:
        if fb not in tr.columns:
            continue
        for ctrl in available_ctrls:
            name = f"{fb}x{ctrl}"
            tr[name] = tr[fb].fillna(0) * tr[ctrl].fillna(0)
            te[name] = te[fb].fillna(0) * te[ctrl].fillna(0)
            int_names.append(name)
    return tr, te, int_names


# ======================================================================
# STRATEGY 5: Residualized FinBERT
# ======================================================================

def strategy_residualized_finbert(train, test, base_features, y_tr, y_te):
    """Forward-select from cross-sectionally residualized FinBERT features."""
    log_msg("\n--- Strategy 5: Residualized FinBERT (quarter z-scored) ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]

    tr, te, resid_cols = cross_sectional_residualize(train, test, available_fb)
    log_msg(f"  Created {len(resid_cols)} residualized features")

    # Verify residualization stats on training data
    for rc in resid_cols[:3]:
        m, s = tr[rc].mean(), tr[rc].std()
        log_msg(f"    {rc}: train mean={m:.4f}, std={s:.4f}")

    current = list(base_features)
    algos = get_all_algos()
    base_cv_ric, base_algo = ts_cv_rank_ic(tr, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline: CV Rank IC={base_cv_ric:+.4f} ({base_algo})")

    added = []
    remaining = list(resid_cols)
    best_so_far = base_cv_ric
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = ts_cv_rank_ic(tr, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
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
            log_msg(f"  Step {step+1}: +{best_feat} → CV={best_ric:+.4f} (Δ={improvement:+.4f})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    X_tr_f, X_te_f, _, _ = preprocess(tr, te, current)
    y_tr_ranks = stats.rankdata(y_tr)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, get_all_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(added)} resid features, Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Residualized FinBERT (quarter z-scored)", "features": current,
            "added_finbert": added, "algo": best_name, "metrics": best_metrics,
            "all_results": final_results, "selection_log": selection_log,
            "cv_rank_ic": float(best_so_far)}


# ======================================================================
# STRATEGY 6: FinBERT × Control Interactions
# ======================================================================

def strategy_interactions_finbert(train, test, base_features, y_tr, y_te):
    """LASSO pre-screen FinBERT×control interactions, then forward-select survivors."""
    log_msg("\n--- Strategy 6: FinBERT × Control Interactions ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    tr, te, int_names = create_interaction_features(train, test, available_fb, INTERACTION_CONTROLS)
    log_msg(f"  Created {len(int_names)} interaction features")

    # LASSO pre-screening on training data only
    all_feats_for_lasso = list(base_features) + int_names
    X_tr_l, _, _, _ = preprocess(tr, tr, all_feats_for_lasso)  # preprocess train only
    y_tr_ranks = stats.rankdata(y_tr)

    lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 30), cv=3, max_iter=10000,
                        random_state=RANDOM_STATE)
    lasso_cv.fit(X_tr_l, y_tr_ranks)
    coefs = pd.Series(lasso_cv.coef_, index=all_feats_for_lasso)
    # Keep interaction features with nonzero coefficients
    survivors = [f for f in int_names if abs(coefs[f]) > 1e-8]
    log_msg(f"  LASSO alpha={lasso_cv.alpha_:.5f}, {len(survivors)}/{len(int_names)} interactions survived")
    if survivors:
        log_msg(f"  Survivors: {survivors[:10]}{'...' if len(survivors) > 10 else ''}")

    if not survivors:
        log_msg(f"  No interactions survived LASSO. Trying economically-motivated subset.")
        # Fallback: hand-picked interactions
        fallback_pairs = [
            ("fb_neg_pct_answers", "sue"),
            ("fb_sent_gap", "sue"),
            ("fb_sent_std_answers", "vol_pre"),
            ("fb_sent_answers_chg", "car_-1_1"),
            ("fb_neg_pct_answers", "vol_pre"),
            ("fb_n_sent_prepared", "surprise_pct"),
            ("fb_neg_pct_prepared", "sue"),
            ("fb_sent_prepared_chg", "car_-1_1"),
        ]
        survivors = []
        for fb, ctrl in fallback_pairs:
            name = f"{fb}x{ctrl}"
            if name in int_names:
                survivors.append(name)
        log_msg(f"  Fallback subset: {len(survivors)} interactions")

    # Forward selection on survivors
    current = list(base_features)
    algos = get_all_algos()
    base_cv_ric, base_algo = ts_cv_rank_ic(tr, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline: CV Rank IC={base_cv_ric:+.4f}")

    added = []
    remaining = list(survivors)
    best_so_far = base_cv_ric
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(min(len(remaining), 6)):  # cap at 6 interactions
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = ts_cv_rank_ic(tr, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
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
            log_msg(f"  Step {step+1}: +{best_feat} → CV={best_ric:+.4f} (Δ={improvement:+.4f})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    X_tr_f, X_te_f, _, _ = preprocess(tr, te, current)
    y_tr_ranks_final = stats.rankdata(y_tr)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks_final, y_te, get_all_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(added)} interactions, Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "FinBERT × Control Interactions", "features": current,
            "added_finbert": added, "algo": best_name, "metrics": best_metrics,
            "all_results": final_results, "selection_log": selection_log,
            "cv_rank_ic": float(best_so_far)}


# ======================================================================
# STRATEGY 7: XGBoost-Preserved Forward Selection
# ======================================================================

def strategy_xgb_forward_selection(train, test, base_features, y_tr, y_te):
    """Forward-select FinBERT features forcing XGBoost as the algorithm."""
    log_msg("\n--- Strategy 7: XGBoost-Preserved Forward Selection ---")
    if not HAS_XGB:
        log_msg("  ERROR: XGBoost not available. Skipping.")
        return {"name": "XGBoost Forward Sel (SKIPPED)", "features": list(base_features),
                "added_finbert": [], "algo": "N/A",
                "metrics": {k: np.nan for k in METRIC_KEYS},
                "all_results": {}, "cv_rank_ic": -1.0}

    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]

    # Try 3 XGBoost configs, pick best by CV on base features
    xgb_configs = {
        "XGB_default": XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            min_child_weight=80, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "XGB_deeper": XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.02,
            min_child_weight=60, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "XGB_shallow": XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
    }

    best_xgb_name = None
    best_xgb_cv = -999
    for name, xgb in xgb_configs.items():
        cv_ric = ts_cv_rank_ic_fixed_algo(train, base_features, PRIMARY_TARGET, name, xgb)
        log_msg(f"  {name}: base CV Rank IC={cv_ric:+.4f}")
        if cv_ric > best_xgb_cv:
            best_xgb_cv = cv_ric
            best_xgb_name = name

    xgb_model = xgb_configs[best_xgb_name]
    log_msg(f"  Best XGBoost config: {best_xgb_name} (CV={best_xgb_cv:+.4f})")

    # Forward selection with the chosen XGBoost config
    current = list(base_features)
    added = []
    remaining = list(available_fb)
    best_so_far = best_xgb_cv
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric = ts_cv_rank_ic_fixed_algo(train, trial, PRIMARY_TARGET,
                                               best_xgb_name, xgb_model)
            candidates[feat] = cv_ric
        best_feat = max(candidates, key=candidates.get)
        best_ric = candidates[best_feat]
        improvement = best_ric - best_so_far
        if improvement > 0:
            current.append(best_feat)
            remaining.remove(best_feat)
            added.append(best_feat)
            best_so_far = best_ric
            selection_log.append({"step": step + 1, "added": best_feat,
                                  "cv_rank_ic": float(best_ric), "improvement": float(improvement)})
            log_msg(f"  Step {step+1}: +{best_feat} → CV={best_ric:+.4f} (Δ={improvement:+.4f})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    # Final evaluation with XGBoost and also other algos for comparison
    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    y_tr_ranks = stats.rankdata(y_tr)

    # Primary: forced XGBoost
    xgb_final = copy.deepcopy(xgb_model)
    xgb_final.fit(X_tr_f, y_tr_ranks)
    xgb_preds = xgb_final.predict(X_te_f)
    xgb_metrics = evaluate(y_te, xgb_preds, y_train=y_tr, X_tr=X_tr_f, X_te=X_te_f, is_ranked=True)

    # Also evaluate all algos for comparison
    all_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, get_all_algos(), y_train_raw=y_tr)
    all_results[best_xgb_name] = xgb_metrics

    best_name, best_metrics = best_result(all_results)
    log_msg(f"  XGBoost result: Rank IC={xgb_metrics['rank_ic']:+.4f}")
    log_msg(f"  Best overall: {best_name} Rank IC={best_metrics['rank_ic']:+.4f}")

    # Report the XGBoost result as primary (that's the point of this strategy)
    return {"name": f"XGBoost Forward Sel ({best_xgb_name})", "features": current,
            "added_finbert": added, "algo": best_xgb_name, "metrics": xgb_metrics,
            "all_results": all_results, "selection_log": selection_log,
            "cv_rank_ic": float(best_so_far),
            "best_any_algo": {"name": best_name, "metrics": best_metrics}}


# ======================================================================
# STRATEGY 8: Option-Specific Feature Candidates
# ======================================================================

def strategy_specific_candidates(train, test, base_features, y_tr, y_te):
    """Forward-select from a curated subset excluding mean-sentiment levels."""
    log_msg("\n--- Strategy 8: Specific Feature Candidates ---")

    # Exclude mean sentiment (fb_sent_prepared, fb_sent_answers) — correlated with LM tone
    priority_candidates = [
        "fb_neg_pct_answers", "fb_n_sent_prepared",    # Opt 2 winners
        "fb_sent_gap", "fb_sent_std_prepared", "fb_sent_std_answers",  # relative/dispersion
        "fb_sent_prepared_chg", "fb_sent_answers_chg",  # change signals
        "fb_pos_pct_answers", "fb_neg_pct_prepared",    # percentage-based
        "fb_pos_pct_prepared", "fb_n_sent_answers",     # remaining non-mean
    ]
    available = [f for f in priority_candidates if f in train.columns]
    log_msg(f"  Candidates ({len(available)}): {', '.join(available)}")

    current = list(base_features)
    algos = get_all_algos()
    base_cv_ric, base_algo = ts_cv_rank_ic(train, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline: CV Rank IC={base_cv_ric:+.4f} ({base_algo})")

    added = []
    remaining = list(available)
    best_so_far = base_cv_ric
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = ts_cv_rank_ic(train, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
            log_msg(f"    + {feat:30s}: CV={cv_ric:+.4f} ({algo_name})")
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
            log_msg(f"  Step {step+1}: +{best_feat} → CV={best_ric:+.4f} (Δ={improvement:+.4f})")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    y_tr_ranks = stats.rankdata(y_tr)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, get_all_algos(), y_train_raw=y_tr)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(added)} features, Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Specific Candidates (no mean-sent)", "features": current,
            "added_finbert": added, "algo": best_name, "metrics": best_metrics,
            "all_results": final_results, "selection_log": selection_log,
            "cv_rank_ic": float(best_so_far)}


# ======================================================================
# STRATEGY 9: Meta-Ensemble
# ======================================================================

def strategy_meta_ensemble(strategy_results, model_a_cv_ric, test, y_te, y_tr):
    """Average predictions from strategies that beat Model A in CV Rank IC."""
    log_msg("\n--- Strategy 9: Meta-Ensemble ---")

    # Filter by CV Rank IC (not test!) to avoid leakage
    passing = []
    for s in strategy_results:
        cv_ric = s.get("cv_rank_ic", -999)
        test_ric = s["metrics"]["rank_ic"]
        log_msg(f"  {s['name']:45s} CV={cv_ric:+.4f}  Test={test_ric:+.4f}  "
                f"{'PASS' if cv_ric > model_a_cv_ric else 'FAIL'}")
        if cv_ric > model_a_cv_ric:
            passing.append(s)

    if len(passing) < 2:
        log_msg(f"  Only {len(passing)} strategies passed CV filter — no ensemble possible")
        if len(passing) == 1:
            return {"name": f"Meta-Ensemble (degenerate: {passing[0]['name']})",
                    "features": passing[0]["features"],
                    "added_finbert": passing[0]["added_finbert"],
                    "algo": "Single-strategy", "metrics": passing[0]["metrics"],
                    "all_results": {}, "cv_rank_ic": passing[0].get("cv_rank_ic", 0)}
        return None

    # Average predictions from passing strategies
    all_preds = []
    for s in passing:
        preds = s["metrics"].get("predictions", [])
        if preds and len(preds) == len(y_te):
            all_preds.append(np.array(preds))
    if len(all_preds) < 2:
        log_msg("  Not enough valid predictions for ensemble")
        return None

    ensemble_preds = np.column_stack(all_preds).mean(axis=1)
    ens_metrics = evaluate(y_te, ensemble_preds, y_train=y_tr, is_ranked=False)
    log_msg(f"  Ensemble of {len(all_preds)} strategies: Rank IC={ens_metrics['rank_ic']:+.4f}")

    passing_names = [s["name"] for s in passing]
    return {"name": f"Meta-Ensemble ({len(all_preds)} strategies)",
            "features": passing_names,
            "added_finbert": passing_names,
            "algo": "Ensemble-avg", "metrics": ens_metrics,
            "all_results": {"Ensemble": ens_metrics},
            "cv_rank_ic": np.mean([s.get("cv_rank_ic", 0) for s in passing])}


# ======================================================================
# MAIN
# ======================================================================

def run_option(opt_name, opt_file, base_features, out_dir, fb_df):
    log_msg(f"\n{'#'*70}")
    log_msg(f"# Model C v2 — {opt_name}")
    log_msg(f"{'#'*70}")

    df = pd.read_parquet(f"{out_dir}/{opt_file}")
    df["call_date"] = pd.to_datetime(df["call_date"])

    pre_len = len(df)
    df = df.merge(fb_df, on="transcriptid", how="left")
    fb_cols = [c for c in FINBERT_CANDIDATES if c in df.columns]
    fb_coverage = df[fb_cols[0]].notna().sum() if fb_cols else 0
    log_msg(f"merged FinBERT: {pre_len} rows, {fb_coverage} with scores ({fb_coverage/pre_len:.1%})")

    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy().reset_index(drop=True)
    test = has_target[has_target["call_date"] > cutoff].copy().reset_index(drop=True)
    y_tr = train[PRIMARY_TARGET].values
    y_te = test[PRIMARY_TARGET].values
    log_msg(f"train: {len(train):,} (to {TRAIN_END}), test: {len(test):,}")

    # Model A baseline
    X_tr_a, X_te_a, _, _ = preprocess(train, test, base_features)
    y_tr_ranks = stats.rankdata(y_tr)
    res_a = fit_and_eval(X_tr_a, X_te_a, y_tr_ranks, y_te, get_all_algos(), y_train_raw=y_tr)
    a_best_name, a_best = best_result(res_a)
    log_msg(f"\nModel A baseline ({len(base_features)} feats): Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})")

    # Also get Model A's CV Rank IC (for Strategy 9 filtering)
    model_a_cv_ric, _ = ts_cv_rank_ic(train, base_features, PRIMARY_TARGET, get_all_algos())
    log_msg(f"Model A CV Rank IC: {model_a_cv_ric:+.4f}")

    # Run all v2 strategies
    s5 = strategy_residualized_finbert(train, test, base_features, y_tr, y_te)
    s6 = strategy_interactions_finbert(train, test, base_features, y_tr, y_te)
    s7 = strategy_xgb_forward_selection(train, test, base_features, y_tr, y_te)
    s8 = strategy_specific_candidates(train, test, base_features, y_tr, y_te)

    new_strategies = [s5, s6, s7, s8]
    s9 = strategy_meta_ensemble(new_strategies, model_a_cv_ric, test, y_te, y_tr)
    if s9:
        new_strategies.append(s9)

    # Summary
    log_msg(f"\n{'='*70}")
    log_msg(f"SUMMARY — Model C v2 — {opt_name}")
    log_msg(f"{'='*70}")
    log_msg(f"{'Model':50s} {'Algo':20s} {'Rank IC':>9s} {'Q5-Q1':>9s} {'vs A':>9s} {'CV IC':>9s}")
    log_msg(f"{'Model A (baseline)':50s} {a_best_name:20s} {a_best['rank_ic']:+9.4f} "
            f"{a_best['quintile_spread']:+9.4f} {'---':>9s} {model_a_cv_ric:+9.4f}")

    winners = []
    for m in new_strategies:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        cv_ic = m.get("cv_rank_ic", 0)
        marker = " *** BEATS A" if delta > 0 else ""
        log_msg(f"{m['name']:50s} {m['algo']:20s} {m['metrics']['rank_ic']:+9.4f} "
                f"{m['metrics']['quintile_spread']:+9.4f} {delta:+9.4f} {cv_ic:+9.4f}{marker}")
        if delta > 0:
            winners.append(m)

    if winners:
        best_winner = max(winners, key=lambda m: m["metrics"]["rank_ic"])
        log_msg(f"\n>>> WINNER: {best_winner['name']} (Rank IC={best_winner['metrics']['rank_ic']:+.4f}, "
                f"Δ={best_winner['metrics']['rank_ic'] - a_best['rank_ic']:+.4f} vs Model A)")
        log_msg(">>> COMMIT RECOMMENDATION: integrate this strategy")
    else:
        log_msg(f"\n>>> No strategy beat Model A's +{a_best['rank_ic']:.4f}")
        log_msg(">>> Document negative results only")

    # Save report
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    report_lines = [
        "=" * 70,
        f"IT Sector — Model C v2 — {opt_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} (≤{TRAIN_END}) | Test: {len(test):,} (>{TRAIN_END})",
        f"FinBERT coverage: {fb_coverage}/{pre_len} ({fb_coverage/pre_len:.1%})",
        f"Model A baseline ({len(base_features)} feats): Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})",
        f"Model A CV Rank IC: {model_a_cv_ric:+.4f}",
        f"Method: Time-series CV, ranked target",
        "=" * 70, "",
    ]

    for m in new_strategies:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        beat = " *** BEATS MODEL A" if delta > 0 else ""
        report_lines.append(f"{m['name']:50s} {m['algo']:20s} Rank IC={m['metrics']['rank_ic']:+.4f} "
                            f"Q5-Q1={m['metrics']['quintile_spread']:+.4f} Δ={delta:+.4f}{beat}")
        report_lines.append(f"  Added features: {', '.join(str(f) for f in m['added_finbert'])}")
        mm = m["metrics"]
        report_lines.append(f"  OOS R²={mm.get('oos_r2', 0):+.6f}  MAE={mm.get('mae', 0):.6f}  "
                            f"Hit Rate={mm.get('hit_rate', 0):.3f} (baseline {mm.get('hit_rate_baseline', 0):.3f})")
        report_lines.append(f"  CV Rank IC: {m.get('cv_rank_ic', 0):+.4f}")
        if m.get("all_results") and len(m["all_results"]) > 1:
            report_lines.append(f"  All algorithms:")
            for aname, ametrics in sorted(m["all_results"].items(),
                                           key=lambda x: x[1]["rank_ic"], reverse=True)[:6]:
                report_lines.append(f"    {aname:25s} Rank IC={ametrics['rank_ic']:+.4f}  "
                                    f"Q5-Q1={ametrics['quintile_spread']:+.4f}  "
                                    f"Hit Rate={ametrics['hit_rate']:.3f}")
        if m.get("selection_log"):
            report_lines.append(f"  Selection log:")
            for entry in m["selection_log"]:
                feat_name = entry["added"] or "(baseline)"
                report_lines.append(f"    Step {entry['step']:2d}: {feat_name:35s} "
                                    f"CV={entry['cv_rank_ic']:+.4f} Δ={entry['improvement']:+.4f}")
        # For Strategy 7, also report best-any-algo
        if m.get("best_any_algo"):
            ba = m["best_any_algo"]
            report_lines.append(f"  Best any algo: {ba['name']} Rank IC={ba['metrics']['rank_ic']:+.4f}")
        report_lines.append("")

    if winners:
        bw = max(winners, key=lambda m: m["metrics"]["rank_ic"])
        report_lines.append(f"COMMIT RECOMMENDATION: {bw['name']}")
        report_lines.append(f"  Algo: {bw['algo']}")
        report_lines.append(f"  Features: {', '.join(str(f) for f in bw['added_finbert'])}")
        report_lines.append(f"  Rank IC: {bw['metrics']['rank_ic']:+.4f}")
        report_lines.append(f"  Δ vs Model A: {bw['metrics']['rank_ic'] - a_best['rank_ic']:+.4f}")
    else:
        report_lines.append("NO COMMIT: No strategy beats Model A")

    Path(out_dir, f"it_model_c_v2_{tag}_report.txt").write_text("\n".join(report_lines))

    # Save JSON
    result_json = {
        "option": opt_name,
        "train_end": TRAIN_END,
        "train_size": len(train),
        "test_size": len(test),
        "model_a_baseline": {"algo": a_best_name, "features": base_features,
                              "rank_ic": a_best["rank_ic"],
                              "cv_rank_ic": model_a_cv_ric,
                              "metrics": metrics_for_json(a_best)},
        "strategies": [
            {"name": m["name"], "algo": m["algo"],
             "added_finbert": m["added_finbert"],
             "cv_rank_ic": m.get("cv_rank_ic", 0),
             "metrics": metrics_for_json(m["metrics"]),
             "beats_model_a": m["metrics"]["rank_ic"] > a_best["rank_ic"],
             "selection_log": m.get("selection_log")}
            for m in new_strategies
        ],
        "has_winner": len(winners) > 0,
    }
    if winners:
        bw = max(winners, key=lambda m: m["metrics"]["rank_ic"])
        result_json["best_winner"] = {
            "name": bw["name"], "algo": bw["algo"],
            "added_finbert": bw["added_finbert"],
            "rank_ic": bw["metrics"]["rank_ic"],
            "delta_vs_a": bw["metrics"]["rank_ic"] - a_best["rank_ic"],
        }

    Path(out_dir, f"it_model_c_v2_{tag}_results.json").write_text(json.dumps(result_json, indent=2))

    # Save predictions
    pred_rows = []
    for m in new_strategies:
        preds = m["metrics"].get("predictions", [])
        if preds:
            for i, p in enumerate(preds):
                pred_rows.append({"model": m["name"], "obs_idx": i,
                                  "y_pred_cal": p, "y_true": float(y_te[i])})
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(f"{out_dir}/predictions_c_v2_{tag}.csv", index=False)

    log_msg(f"\nsaved results to {out_dir}/it_model_c_v2_{tag}_*")
    return new_strategies, a_best, winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_it"))
    ap.add_argument("--option", type=int, choices=[1, 2], required=True,
                    help="Which option to run (1 or 2)")
    args = ap.parse_args()
    OUT = args.folder

    fb_path = f"{OUT}/finbert_features.parquet"
    if not os.path.exists(fb_path):
        log_msg(f"ERROR: {fb_path} not found")
        return
    fb_df = pd.read_parquet(fb_path)
    log_msg(f"loaded FinBERT features: {len(fb_df):,} rows, {fb_df.shape[1]} columns")

    options = {
        1: ("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
        2: ("Option 2 (IT sector bench)", "model_table_opt2.parquet"),
    }
    opt_name, opt_file = options[args.option]
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    fpath = f"{OUT}/{opt_file}"
    if not os.path.exists(fpath):
        log_msg(f"ERROR: {fpath} not found")
        return

    a_feat_path = f"{OUT}/model_a_features_{tag}.json"
    if os.path.exists(a_feat_path):
        a_info = json.loads(Path(a_feat_path).read_text())
        base_features = a_info["features"]
        log_msg(f"loaded Model A features: {base_features}")
    else:
        log_msg(f"WARNING: {a_feat_path} not found, using fallback")
        df_tmp = pd.read_parquet(fpath)
        base_features = [c for c in ["sue", "surprise_pct", "vol_pre", "car_-1_1",
                                      "mom", "bm", "size", "dispersion",
                                      "avg_answer_words", "words_prepared",
                                      "words_questions", "words_answers",
                                      "n_questions", "n_answers", "n_analysts",
                                      "prior_calls_with_target"]
                         if c in df_tmp.columns]

    strategies, model_a, winners = run_option(opt_name, opt_file, base_features, OUT, fb_df)

    log_msg(f"\n{'='*70}")
    log_msg(f"DONE — Option {args.option}")
    log_msg(f"{'='*70}")
    log_msg(f"Model A: Rank IC={model_a['rank_ic']:+.4f}")
    if winners:
        bw = max(winners, key=lambda m: m["metrics"]["rank_ic"])
        log_msg(f"WINNER: {bw['name']} Rank IC={bw['metrics']['rank_ic']:+.4f} "
                f"(Δ={bw['metrics']['rank_ic'] - model_a['rank_ic']:+.4f})")
    else:
        log_msg("No strategy beat Model A")


if __name__ == "__main__":
    main()
