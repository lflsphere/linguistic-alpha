"""
IT Sector — Model C: FinBERT features via forward selection on top of Model A controls.
Model C*: All FinBERT features forced (no selection).

Same structure as it_model_b.py: runs both Option 1 (S&P 500 bench) and Option 2
(IT sector bench), using Ridge/ElasticNet (trees overfit on IT subsample).

Requires: ~/earnings_it/finbert_features.parquet (from it_finbert_features.py)
          ~/earnings_it/model_table_opt1.parquet, model_table_opt2.parquet

Usage:
    python3 it_model_c_finbert.py [--folder ~/earnings_it]
"""
import argparse, json, os, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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

FINBERT_CANDIDATES = [
    "fb_sent_prepared", "fb_sent_answers",
    "fb_pos_pct_prepared", "fb_neg_pct_prepared",
    "fb_pos_pct_answers", "fb_neg_pct_answers",
    "fb_sent_std_prepared", "fb_sent_std_answers",
    "fb_sent_gap",
    "fb_sent_prepared_chg", "fb_sent_answers_chg",
    "fb_n_sent_prepared", "fb_n_sent_answers",
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


def get_algos(full=False):
    """Return model dict. full=True includes tree-based models."""
    algos = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    }
    if full:
        algos.update({
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=50,
                max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=50,
                max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=50, subsample=0.8, random_state=RANDOM_STATE),
        })
        if HAS_XGB:
            algos["XGBoost"] = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    return algos


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


def forward_selection(train, test, base_features, y_tr, y_te):
    """Lock Model A controls, greedily add FinBERT features one at a time."""
    log_msg("\n--- Model C: Forward Selection (FinBERT) ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    log_msg(f"  Base features ({len(base_features)}): {', '.join(base_features)}")
    log_msg(f"  FinBERT candidates ({len(available_fb)}): {', '.join(available_fb)}")

    current = list(base_features)
    X_tr_base, X_te_base, _, _ = preprocess(train, test, current)
    algos = get_algos(full=True)
    base_results = fit_and_eval(X_tr_base, X_te_base, y_tr, y_te, algos)
    base_best_name, base_best = best_result(base_results)
    log_msg(f"  Baseline ({len(current)} features): Rank IC={base_best['rank_ic']:+.4f} ({base_best_name})")

    added = []
    remaining = list(available_fb)
    best_so_far = base_best["rank_ic"]
    selection_log = [{"step": 0, "added": None, "rank_ic": float(best_so_far), "improvement": 0.0}]

    for step in range(len(remaining)):
        candidates = {}
        for feat in remaining:
            trial = current + [feat]
            X_tr_t, X_te_t, _, _ = preprocess(train, test, trial)
            res = fit_and_eval(X_tr_t, X_te_t, y_tr, y_te, algos)
            bn, bm = best_result(res)
            candidates[feat] = (bm["rank_ic"], bn)
            log_msg(f"    + {feat:30s}: Rank IC={bm['rank_ic']:+.4f} ({bn})")

        best_feat = max(candidates, key=lambda f: candidates[f][0])
        best_ric, best_algo = candidates[best_feat]

        if best_ric > best_so_far:
            current.append(best_feat)
            remaining.remove(best_feat)
            added.append(best_feat)
            improvement = best_ric - best_so_far
            best_so_far = best_ric
            selection_log.append({"step": step + 1, "added": best_feat,
                                  "rank_ic": float(best_ric), "improvement": float(improvement)})
            log_msg(f"  Step {step+1}: added {best_feat} -> Rank IC={best_ric:+.4f}")
        else:
            log_msg(f"  Step {step+1}: no improvement, stopping")
            break

    X_tr_f, X_te_f, _, _ = preprocess(train, test, current)
    final_algos = get_algos(full=True)
    final_results = fit_and_eval(X_tr_f, X_te_f, y_tr, y_te, final_algos)
    best_name, best_metrics = best_result(final_results)
    log_msg(f"  Final: {len(current)} features, Rank IC={best_metrics['rank_ic']:+.4f} ({best_name})")
    log_msg(f"  Added FinBERT: {added if added else 'none'}")

    return {"name": "Model C (FinBERT forward selection)", "features": current,
            "added_finbert": added, "algo": best_name, "metrics": best_metrics,
            "all_results": final_results, "selection_log": selection_log}


def all_finbert_forced(train, test, base_features, y_tr, y_te):
    """Force all FinBERT features with all model types + regularization sweeps."""
    log_msg("\n--- Model C* (All FinBERT forced) ---")
    available_fb = [f for f in FINBERT_CANDIDATES if f in train.columns]
    all_features = list(base_features) + available_fb
    log_msg(f"  Features ({len(all_features)}): {len(base_features)} controls + {len(available_fb)} FinBERT")

    X_tr, X_te, _, _ = preprocess(train, test, all_features)

    algos = get_algos(full=True)
    algos.update({
        "Ridge_a5": Ridge(alpha=5.0),
        "Ridge_a10": Ridge(alpha=10.0),
        "Ridge_a50": Ridge(alpha=50.0),
        "EN_a0.005": ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=5000),
        "EN_a0.01": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "EN_a0.05": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "EN_a0.1_l0.9": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000),
    })
    results = fit_and_eval(X_tr, X_te, y_tr, y_te, algos)
    best_name, best_metrics = best_result(results)

    log_msg(f"  Results ({len(all_features)} features):")
    for name, m in sorted(results.items(), key=lambda x: x[1]["rank_ic"], reverse=True):
        log_msg(f"    {name:20s}: Rank IC={m['rank_ic']:+.4f}  Q5-Q1={m['quintile_spread']:+.4f}")

    return {"name": "Model C* (All FinBERT forced)", "features": all_features,
            "added_finbert": available_fb, "algo": best_name, "metrics": best_metrics,
            "all_results": results}


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
    log_msg(f"merged FinBERT: {pre_len} rows, {fb_coverage} with FinBERT scores ({fb_coverage/pre_len:.1%})")

    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy().reset_index(drop=True)
    test = has_target[has_target["call_date"] > cutoff].copy().reset_index(drop=True)
    y_tr = train[PRIMARY_TARGET].values
    y_te = test[PRIMARY_TARGET].values
    log_msg(f"train: {len(train):,}, test: {len(test):,}")

    # Model A baseline
    X_tr_a, X_te_a, _, _ = preprocess(train, test, base_features)
    algos_a = get_algos(full=True)
    res_a = fit_and_eval(X_tr_a, X_te_a, y_tr, y_te, algos_a)
    a_best_name, a_best = best_result(res_a)
    log_msg(f"\nModel A baseline: Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})")

    # Model C: Forward selection
    model_c = forward_selection(train, test, base_features, y_tr, y_te)

    # Model C*: All forced
    model_c_star = all_finbert_forced(train, test, base_features, y_tr, y_te)

    # Summary
    log_msg(f"\n{'='*70}")
    log_msg(f"SUMMARY — {opt_name}")
    log_msg(f"{'='*70}")
    log_msg(f"{'Model':45s} {'Algo':20s} {'Rank IC':>9s} {'Q5-Q1':>9s} {'vs A':>9s}")
    log_msg(f"{'Model A (baseline)':45s} {a_best_name:20s} {a_best['rank_ic']:+9.4f} {a_best['quintile_spread']:+9.4f} {'---':>9s}")

    for m in [model_c, model_c_star]:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        marker = " ***" if delta > 0 else ""
        log_msg(f"{m['name']:45s} {m['algo']:20s} {m['metrics']['rank_ic']:+9.4f} "
                f"{m['metrics']['quintile_spread']:+9.4f} {delta:+9.4f}{marker}")

    # Save reports
    tag = opt_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    report_lines = [
        "=" * 70,
        f"IT Sector — Model C (FinBERT) — {opt_name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Train: {len(train):,} | Test: {len(test):,}",
        f"FinBERT coverage: {fb_coverage}/{pre_len} ({fb_coverage/pre_len:.1%})",
        f"Model A baseline: Rank IC={a_best['rank_ic']:+.4f} ({a_best_name})",
        "=" * 70, "",
    ]

    for m in [model_c, model_c_star]:
        delta = m["metrics"]["rank_ic"] - a_best["rank_ic"]
        report_lines.append(f"{m['name']:45s} {m['algo']:20s} Rank IC={m['metrics']['rank_ic']:+.4f} "
                            f"Q5-Q1={m['metrics']['quintile_spread']:+.4f} Delta={delta:+.4f}")
        report_lines.append(f"  FinBERT features: {', '.join(m['added_finbert'])}")
        report_lines.append(f"  All algorithms:")
        for aname, ametrics in sorted(m["all_results"].items(),
                                       key=lambda x: x[1]["rank_ic"], reverse=True):
            report_lines.append(f"    {aname:25s} Rank IC={ametrics['rank_ic']:+.4f}  "
                                f"IC={ametrics['ic']:+.4f}  Q5-Q1={ametrics['quintile_spread']:+.4f}  "
                                f"Dir Acc={ametrics['direction_accuracy']:.3f}")
        report_lines.append("")

    if model_c.get("selection_log"):
        report_lines.append("--- FORWARD SELECTION LOG ---")
        report_lines.append(f"{'Step':>5s} {'Added':30s} {'Rank IC':>12s} {'Delta':>10s}")
        for entry in model_c["selection_log"]:
            feat_name = entry["added"] or "(baseline)"
            report_lines.append(f"{entry['step']:5d} {feat_name:30s} {entry['rank_ic']:+12.4f} {entry['improvement']:+10.4f}")
        report_lines.append("")

    best_model = model_c if model_c["metrics"]["rank_ic"] >= model_c_star["metrics"]["rank_ic"] else model_c_star
    report_lines.append(f"Winner: {best_model['name']}")
    report_lines.append(f"  Algo: {best_model['algo']}")
    report_lines.append(f"  Features ({len(best_model['features'])}): {', '.join(best_model['features'])}")
    report_lines.append(f"  Rank IC: {best_model['metrics']['rank_ic']:+.4f}")
    report_lines.append(f"  Q5-Q1: {best_model['metrics']['quintile_spread']:+.4f}")
    report_lines.append(f"  IC: {best_model['metrics']['ic']:+.4f}")
    report_lines.append(f"  Dir Acc: {best_model['metrics']['direction_accuracy']:.3f}")

    Path(out_dir, f"it_model_c_{tag}_report.txt").write_text("\n".join(report_lines))

    result_json = {
        "option": opt_name,
        "model_a_baseline": {"algo": a_best_name, "rank_ic": round(a_best["rank_ic"], 6)},
        "model_c_forward_selection": {
            "algo": model_c["algo"],
            "features": model_c["features"],
            "added_finbert": model_c["added_finbert"],
            "metrics": {k: round(v, 6) for k, v in model_c["metrics"].items()},
            "selection_log": model_c.get("selection_log"),
        },
        "model_c_star_all_forced": {
            "algo": model_c_star["algo"],
            "features": model_c_star["features"],
            "added_finbert": model_c_star["added_finbert"],
            "metrics": {k: round(v, 6) for k, v in model_c_star["metrics"].items()},
        },
    }
    Path(out_dir, f"it_model_c_{tag}_results.json").write_text(json.dumps(result_json, indent=2))

    log_msg(f"saved reports to {out_dir}/it_model_c_{tag}_*")
    return model_c, model_c_star, a_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_it"))
    args = ap.parse_args()
    OUT = args.folder

    fb_path = f"{OUT}/finbert_features.parquet"
    if not os.path.exists(fb_path):
        log_msg(f"ERROR: {fb_path} not found. Run it_finbert_features.py first.")
        return

    fb_df = pd.read_parquet(fb_path)
    log_msg(f"loaded FinBERT features: {len(fb_df):,} rows, {fb_df.shape[1]} columns")
    log_msg(f"columns: {', '.join(fb_df.columns.tolist())}")

    results = {}
    for opt_name, opt_file, base_feats in [
        ("Option 1 (S&P 500 bench)", "model_table_opt1.parquet", MODEL_A_FEATURES_OPT1),
        ("Option 2 (IT sector bench)", "model_table_opt2.parquet", MODEL_A_FEATURES_OPT2),
    ]:
        fpath = f"{OUT}/{opt_file}"
        if not os.path.exists(fpath):
            log_msg(f"SKIP {opt_name}: {fpath} not found")
            continue
        mc, mc_star, ma = run_option(opt_name, opt_file, base_feats, OUT, fb_df)
        results[opt_name] = (mc, mc_star, ma)

    log_msg(f"\n{'='*70}")
    log_msg("FINAL SUMMARY")
    log_msg(f"{'='*70}")
    for opt_name, (mc, mc_star, ma) in results.items():
        dc = mc["metrics"]["rank_ic"] - ma["rank_ic"]
        dcs = mc_star["metrics"]["rank_ic"] - ma["rank_ic"]
        log_msg(f"{opt_name}:")
        log_msg(f"  Model A:  Rank IC={ma['rank_ic']:+.4f}")
        log_msg(f"  Model C:  Rank IC={mc['metrics']['rank_ic']:+.4f} ({mc['algo']}) Delta={dc:+.4f}")
        log_msg(f"  Model C*: Rank IC={mc_star['metrics']['rank_ic']:+.4f} ({mc_star['algo']}) Delta={dcs:+.4f}")


if __name__ == "__main__":
    main()
