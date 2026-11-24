"""
Sector — Model C (sector chosen with --sector): FinBERT features on top of Model A controls.

Reads Model A features from JSON saved by sector_model_b.py (ensures consistency).
Uses same methodology: time-series expanding-window CV, ranked target,
strict improvement threshold.

Strategies:
  1. Forward selection (raw FinBERT features)
  2. All FinBERT forced with RidgeCV + tree-based models
  3. Single-feature ensemble (average best individual FinBERT models)
  4. FinBERT + LM dictionary combined

Train: 2010-2023 | Test: 2024-2026.

Usage:
    python3 sector_model_c_finbert.py --sector {it,health,comm} [--folder ~/earnings_{sector}]
"""
import argparse, copy, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler

from sector_config import add_sector_arg, get_sector

SEC = None  # set in main() from --sector
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
MIN_IMPROVEMENT = 0.0  # add any feature that improves CV Rank IC

FINBERT_CANDIDATES = [
    "fb_sent_prepared", "fb_sent_answers",
    "fb_pos_pct_prepared", "fb_neg_pct_prepared",
    "fb_pos_pct_answers", "fb_neg_pct_answers",
    "fb_sent_std_prepared", "fb_sent_std_answers",
    "fb_sent_gap",
    "fb_sent_prepared_chg", "fb_sent_answers_chg",
    "fb_n_sent_prepared", "fb_n_sent_answers",
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
    results = {}
    for name, model in algos.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        results[name] = evaluate(y_te, preds)
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
    """Linear + tree-based models."""
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
# MODEL C STRATEGIES
# ======================================================================

def strategy_forward_selection(train, test, base_features, y_tr, y_te):
    """Forward-select FinBERT features using time-series CV."""
    log_msg("\n--- Model C Strategy 1: Forward Selection (FinBERT, TS-CV) ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    log_msg(f"  Base features ({len(base_features)}): {', '.join(base_features)}")
    log_msg(f"  FinBERT candidates ({len(available_fb)}): {', '.join(available_fb)}")

    current = list(base_features)
    algos = get_all_algos()

    base_cv_ric, base_algo = ts_cv_rank_ic(train, current, PRIMARY_TARGET, algos)
    log_msg(f"  Baseline: TS-CV Rank IC={base_cv_ric:+.4f} ({base_algo})")

    added = []
    remaining = list(available_fb)
    best_so_far = base_cv_ric
    selection_log = [{"step": 0, "added": None, "cv_rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            cv_ric, algo_name = ts_cv_rank_ic(train, trial, PRIMARY_TARGET, algos)
            candidates[feat] = (cv_ric, algo_name)
            log_msg(f"    + {feat:30s}: TS-CV Rank IC={cv_ric:+.4f} ({algo_name})")

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
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr_ranks, y_te, get_all_algos())
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(current)} features, Holdout Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")

    return {"name": "Model C (FinBERT forward sel)", "features": current,
            "added_finbert": added, "algo": best_name, "metrics": best_metrics,
            "all_results": final_results, "selection_log": selection_log}


def strategy_all_forced(train, test, base_features, y_tr, y_te):
    """All FinBERT features forced, with regularization sweep + trees."""
    log_msg("\n--- Model C Strategy 2: All FinBERT Forced ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    all_features = list(base_features) + available_fb
    log_msg(f"  Features: {len(base_features)} controls + {len(available_fb)} FinBERT = {len(all_features)}")

    X_tr, X_te, _, _ = preprocess(train, test, all_features)
    y_tr_ranks = stats.rankdata(y_tr)

    algos = get_all_algos()
    algos.update({
        "Ridge_a50": Ridge(alpha=50.0),
        "Ridge_a100": Ridge(alpha=100.0),
        "EN_a0.1": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000),
    })
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, algos)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Results ({len(all_features)} features):")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True)[:8]:
        log_msg(f"    {name:25s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "Model C* (All FinBERT forced)", "features": all_features,
            "added_finbert": available_fb, "algo": best_name, "metrics": best_metrics,
            "all_results": results}


def strategy_ensemble(train, test, base_features, y_tr, y_te):
    """Average predictions from best single-FinBERT-feature models."""
    log_msg("\n--- Model C Strategy 3: Single-Feature Ensemble ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    y_tr_ranks = stats.rankdata(y_tr)

    # Baseline
    X_tr_base, X_te_base, _, _ = preprocess(train, test, base_features)
    base_model = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    base_model.fit(X_tr_base, y_tr_ranks)
    base_preds = base_model.predict(X_te_base)
    base_ric = stats.spearmanr(base_preds, y_te)[0]
    log_msg(f"  Base Rank IC: {base_ric:+.4f}")

    single_results = {}
    for feat in available_fb:
        trial = list(base_features) + [feat]
        X_tr_f, X_te_f, _, _ = preprocess(train, test, trial)
        # Try multiple models per feature, pick best
        best_preds, best_ric_feat = None, -1.0
        for algo_name, model in get_all_algos().items():
            m = copy.deepcopy(model)
            m.fit(X_tr_f, y_tr_ranks)
            preds = m.predict(X_te_f)
            ric = stats.spearmanr(preds, y_te)[0]
            if not np.isnan(ric) and ric > best_ric_feat:
                best_ric_feat = ric
                best_preds = preds
                best_algo = algo_name
        single_results[feat] = {"rank_ic": best_ric_feat, "preds": best_preds, "algo": best_algo}
        log_msg(f"    + {feat:30s}: Rank IC={best_ric_feat:+.4f} ({best_algo})")

    good_feats = [(f, r) for f, r in sorted(single_results.items(),
                  key=lambda x: x[1]["rank_ic"], reverse=True) if r["rank_ic"] > base_ric]
    good_summary = [(f, round(r["rank_ic"], 4)) for f, r in good_feats]
    log_msg(f"  Features improving over base: {good_summary}")

    if len(good_feats) >= 2:
        ensemble_preds = np.column_stack([r["preds"] for _, r in good_feats]).mean(axis=1)
        ens_metrics = evaluate(y_te, ensemble_preds)
        log_msg(f"  Ensemble of {len(good_feats)} models: Rank IC={ens_metrics['rank_ic']:+.4f}")
        return {"name": f"FinBERT Ensemble ({len(good_feats)})", "features": [f for f, _ in good_feats],
                "added_finbert": [f for f, _ in good_feats], "algo": "Ensemble",
                "metrics": ens_metrics, "all_results": {"Ensemble": ens_metrics}}
    elif len(good_feats) == 1:
        feat, res = good_feats[0]
        metrics = evaluate(y_te, res["preds"])
        return {"name": f"Single Best ({feat})", "features": list(base_features) + [feat],
                "added_finbert": [feat], "algo": res["algo"],
                "metrics": metrics, "all_results": {res["algo"]: metrics}}
    else:
        metrics = evaluate(y_te, base_preds)
        return {"name": "No FinBERT Improvement", "features": list(base_features),
                "added_finbert": [], "algo": "RidgeCV",
                "metrics": metrics, "all_results": {"RidgeCV": metrics}}


def strategy_finbert_plus_lm(train, test, base_features, y_tr, y_te):
    """FinBERT + LM dictionary features combined — see if they complement."""
    log_msg("\n--- Model C Strategy 4: FinBERT + LM Dictionary Combined ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    available_lm = [f for f in LINGUISTIC_FEATURES if f in train.columns]
    all_features = list(base_features) + available_fb + available_lm
    log_msg(f"  Features: {len(base_features)} controls + {len(available_fb)} FinBERT + {len(available_lm)} LM = {len(all_features)}")

    X_tr, X_te, _, _ = preprocess(train, test, all_features)
    y_tr_ranks = stats.rankdata(y_tr)
    algos = get_all_algos()
    algos["Ridge_a50"] = Ridge(alpha=50.0)
    algos["Ridge_a100"] = Ridge(alpha=100.0)
    results = fit_and_eval(X_tr, X_te, y_tr_ranks, y_te, algos)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Top results ({len(all_features)} features):")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True)[:5]:
        log_msg(f"    {name:25s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "FinBERT + LM Combined", "features": all_features,
            "added_finbert": available_fb + available_lm, "algo": best_name,
            "metrics": best_metrics, "all_results": results}


# ======================================================================
# MAIN
# ======================================================================

def run_option(opt_name, opt_file, base_features, out_dir, fb_df):
    log_msg(f"\n{'#'*70}")
    log_msg(f"# {opt_name}")
    log_msg(f"{'#'*70}")

    df = pd.read_parquet(f"{out_dir}/{opt_file}")
    df["call_date"] = pd.to_datetime(df["call_date"])

    # Merge FinBERT features
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

    # Model A baseline (using same features as Model B selected)
    X_tr_a, X_te_a, _, _ = preprocess(train, test, base_features)
    y_tr_ranks = stats.rankdata(y_tr)
    res_a = fit_and_eval(X_tr_a, X_te_a, y_tr_ranks, y_te, get_all_algos())
    a_best_name, a_best = best_result(res_a)
    log_msg(f"\nModel A baseline ({len(base_features)} feats): Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})")

    # Run all Model C strategies
    model_c = strategy_forward_selection(train, test, base_features, y_tr, y_te)
    model_c_star = strategy_all_forced(train, test, base_features, y_tr, y_te)
    model_c_ens = strategy_ensemble(train, test, base_features, y_tr, y_te)
    model_c_combined = strategy_finbert_plus_lm(train, test, base_features, y_tr, y_te)

    all_models = [model_c, model_c_star, model_c_ens, model_c_combined]

    # Summary
    log_msg(f"\n{'='*70}")
    log_msg(f"SUMMARY — {opt_name}")
    log_msg(f"{'='*70}")
    log_msg(f"{'Model':45s} {'Algo':20s} {'Rank IC':>9s} {'Q5-Q1':>9s} {'vs A':>9s}")
    log_msg(f"{'Model A (baseline)':45s} {a_best_name:20s} {a_best['rank_ic']:+9.4f} {a_best['quintile_spread']:+9.4f} {'---':>9s}")

    for m in all_models:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        marker = " ***" if delta > 0 else ""
        log_msg(f"{m['name']:45s} {m['algo']:20s} {m['metrics']['rank_ic']:+9.4f} "
                f"{m['metrics']['quintile_spread']:+9.4f} {delta:+9.4f}{marker}")

    best_model = max(all_models, key=lambda m: m["metrics"]["rank_ic"])
    if best_model["metrics"]["rank_ic"] > a_best["rank_ic"]:
        log_msg(f"\n>>> Winner: {best_model['name']} (Rank IC={best_model['metrics']['rank_ic']:+.4f}, "
                f"Δ={best_model['metrics']['rank_ic'] - a_best['rank_ic']:+.4f} vs Model A)")
    else:
        log_msg(f"\n>>> No strategy beat Model A. Best: {best_model['name']} "
                f"(Rank IC={best_model['metrics']['rank_ic']:+.4f})")

    # Save reports
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    report_lines = [
        "=" * 70,
        f"{SEC.short} Sector — Model C (FinBERT) — {opt_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} (≤{TRAIN_END}) | Test: {len(test):,} (>{TRAIN_END})",
        f"FinBERT coverage: {fb_coverage}/{pre_len} ({fb_coverage/pre_len:.1%})",
        f"Model A baseline ({len(base_features)} feats): Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})",
        f"Method: Time-series CV, ranked target, min Δ={MIN_IMPROVEMENT}",
        "=" * 70, "",
    ]

    for m in all_models:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        report_lines.append(f"{m['name']:45s} {m['algo']:20s} Rank IC={m['metrics']['rank_ic']:+.4f} "
                            f"Q5-Q1={m['metrics']['quintile_spread']:+.4f} Δ={delta:+.4f}")
        report_lines.append(f"  FinBERT features: {', '.join(m['added_finbert'])}")
        if m.get("all_results") and len(m["all_results"]) > 1:
            report_lines.append(f"  All algorithms:")
            for aname, ametrics in sorted(m["all_results"].items(),
                                           key=lambda x: x[1]["rank_ic"], reverse=True)[:6]:
                report_lines.append(f"    {aname:25s} Rank IC={ametrics['rank_ic']:+.4f}  "
                                    f"Q5-Q1={ametrics['quintile_spread']:+.4f}  "
                                    f"Dir Acc={ametrics['direction_accuracy']:.3f}")
        report_lines.append("")

    if model_c.get("selection_log"):
        report_lines.append("--- FORWARD SELECTION LOG ---")
        report_lines.append(f"{'Step':>5s} {'Added':30s} {'TS-CV Rank IC':>14s} {'Delta':>10s}")
        for entry in model_c["selection_log"]:
            feat_name = entry["added"] or "(baseline)"
            report_lines.append(f"{entry['step']:5d} {feat_name:30s} {entry['cv_rank_ic']:+14.4f} {entry['improvement']:+10.4f}")
        report_lines.append("")

    report_lines.append(f"Best Model C: {best_model['name']}")
    report_lines.append(f"  Algo: {best_model['algo']}")
    report_lines.append(f"  Features: {', '.join(best_model.get('added_finbert', []))}")
    report_lines.append(f"  Rank IC: {best_model['metrics']['rank_ic']:+.4f}")
    report_lines.append(f"  Q5-Q1: {best_model['metrics']['quintile_spread']:+.4f}")
    report_lines.append(f"  IC: {best_model['metrics']['ic']:+.4f}")
    report_lines.append(f"  Dir Acc: {best_model['metrics']['direction_accuracy']:.3f}")

    Path(out_dir, f"{SEC.key}_model_c_{tag}_report.txt").write_text("\n".join(report_lines))

    result_json = {
        "option": opt_name,
        "train_end": TRAIN_END,
        "method": "time-series CV, ranked target",
        "model_a_baseline": {"algo": a_best_name, "rank_ic": round(a_best["rank_ic"], 6),
                              "features": base_features},
        "strategies": [
            {"name": m["name"], "algo": m["algo"],
             "added_finbert": m["added_finbert"],
             "metrics": {k: round(v, 6) for k, v in m["metrics"].items()},
             "selection_log": m.get("selection_log")}
            for m in all_models
        ],
        "best": {
            "name": best_model["name"], "algo": best_model["algo"],
            "added_finbert": best_model["added_finbert"],
            "metrics": {k: round(v, 6) for k, v in best_model["metrics"].items()},
        },
    }
    Path(out_dir, f"{SEC.key}_model_c_{tag}_results.json").write_text(json.dumps(result_json, indent=2))

    log_msg(f"saved reports to {out_dir}/{SEC.key}_model_c_{tag}_*")
    return best_model, a_best


def main():
    ap = argparse.ArgumentParser()
    add_sector_arg(ap)
    ap.add_argument("--folder", default=None, help="default: ~/earnings_{sector}")
    args = ap.parse_args()
    global SEC
    SEC = get_sector(args)
    OUT = args.folder or SEC.folder

    fb_path = f"{OUT}/finbert_features.parquet"
    if not os.path.exists(fb_path):
        log_msg(f"ERROR: {fb_path} not found. Run sector_finbert_features.py first.")
        return

    fb_df = pd.read_parquet(fb_path)
    log_msg(f"loaded FinBERT features: {len(fb_df):,} rows, {fb_df.shape[1]} columns")

    results = {}
    for opt_name, opt_file in [
        ("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
        (f"Option 2 ({SEC.short} sector bench)", "model_table_opt2.parquet"),
    ]:
        tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fpath = f"{OUT}/{opt_file}"
        if not os.path.exists(fpath):
            log_msg(f"SKIP {opt_name}: {fpath} not found")
            continue

        # Read Model A features from Model B output (ensures consistency)
        a_feat_path = f"{OUT}/model_a_features_{tag}.json"
        if os.path.exists(a_feat_path):
            a_info = json.loads(Path(a_feat_path).read_text())
            base_features = a_info["features"]
            log_msg(f"loaded Model A features from {a_feat_path}: {base_features}")
        else:
            log_msg(f"WARNING: {a_feat_path} not found. Run sector_model_b.py first.")
            log_msg(f"Using fallback control candidates for Model A.")
            # Fallback: use all available controls
            df_tmp = pd.read_parquet(fpath)
            base_features = [c for c in ["sue", "vol_pre", "car_-1_1", "mom", "bm", "size",
                                          "dispersion", "avg_answer_words", "words_prepared",
                                          "prior_calls_with_target", "n_questions", "n_answers"]
                             if c in df_tmp.columns]

        best_c, model_a = run_option(opt_name, opt_file, base_features, OUT, fb_df)
        results[opt_name] = (best_c, model_a)

    log_msg(f"\n{'='*70}")
    log_msg("FINAL SUMMARY")
    log_msg(f"{'='*70}")
    for opt_name, (best_c, model_a) in results.items():
        dc = best_c["metrics"]["rank_ic"] - model_a["rank_ic"]
        log_msg(f"{opt_name}:")
        log_msg(f"  Model A:  Rank IC={model_a['rank_ic']:+.4f}")
        log_msg(f"  Best C:   Rank IC={best_c['metrics']['rank_ic']:+.4f} ({best_c['name']}, {best_c['algo']}) Δ={dc:+.4f}")


if __name__ == "__main__":
    main()
