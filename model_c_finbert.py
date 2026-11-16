"""
Model C: FinBERT features via forward selection on top of Model A's 12 controls.
Model C*: All FinBERT features forced (no selection).

Forward selection starts with Model A's 12 selected controls as base,
then iteratively adds FinBERT features one at a time — keeping the one
that most improves CV Rank IC — until no addition helps.

Train: 2010–2021 (expanding-window CV inside for selection + model eval)
Test:  2022–2024 (one-time evaluation)

Usage:
    python3 model_c_finbert.py [--folder ~/earnings_sp500]
"""
import argparse, os, json, warnings, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PRIMARY_TARGET = "car_2_20"
SECTOR_COL = "sector"

MODEL_A_CONTROLS = [
    "dispersion", "vol_pre", "size", "prior_calls_with_target",
    "mom", "car_0_1", "bm", "sue_price", "sue",
    "words_answers", "share_answers_by_execs", "n_answers",
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

LOG_TRANSFORM = ["words_answers"]
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99
TRAIN_END = "2021-12-31"
CV_FOLDS = 5
MIN_FOLD_SIZE = 100
RANDOM_STATE = 42


def log_msg(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# --- data ---
def load_data(base):
    mt = pd.read_parquet(f"{base}/model_table.parquet")
    fb = pd.read_parquet(f"{base}/finbert_features.parquet")
    log_msg(f"model_table: {len(mt):,} rows, finbert: {len(fb):,} rows")
    df = mt.merge(fb, on="transcriptid", how="inner")
    log_msg(f"merged: {len(df):,} rows")

    df["call_date"] = pd.to_datetime(df["call_date"])
    has_target = df[df[PRIMARY_TARGET].notna()].copy()
    cutoff = pd.Timestamp(TRAIN_END)
    train = has_target[has_target["call_date"] <= cutoff].copy()
    test = has_target[has_target["call_date"] > cutoff].copy()
    log_msg(f"train: {len(train):,} ({train['call_date'].min().date()} to {train['call_date'].max().date()})")
    log_msg(f"test:  {len(test):,} ({test['call_date'].min().date()} to {test['call_date'].max().date()})")
    return df, train, test


# --- preprocessing ---
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


def build_sector_dummies(sector_values, all_sectors):
    sector_to_idx = {s: i for i, s in enumerate(all_sectors[1:])}
    d = np.zeros((len(sector_values), len(sector_to_idx)))
    for i, s in enumerate(sector_values):
        if s in sector_to_idx:
            d[i, sector_to_idx[s]] = 1.0
    names = [f"sec_{s}" for s in all_sectors[1:]]
    return d, names


# --- CV ---
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
    return {"ic": ic, "rank_ic": rank_ic, "r2": r2, "rmse": rmse,
            "direction_accuracy": accuracy, "quintile_spread": top_bottom}


def cv_rank_ic(feature_names, X_numeric, y, sector, all_sectors, dates, splits):
    """Run GradientBoosting CV and return average Rank IC."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    fold_rank_ics = []
    for train_idx, test_idx in splits:
        X_tr_raw = pd.DataFrame(X_numeric[train_idx], columns=feature_names)
        X_te_raw = pd.DataFrame(X_numeric[test_idx], columns=feature_names)
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_proc, fit_stats = preprocess_df(X_tr_raw, feature_names)
        X_te_proc, _ = preprocess_df(X_te_raw, feature_names, fit_stats=fit_stats)

        sec_tr, _ = build_sector_dummies(sector[train_idx], all_sectors)
        sec_te, _ = build_sector_dummies(sector[test_idx], all_sectors)

        X_tr_full = np.hstack([X_tr_proc.values, sec_tr])
        X_te_full = np.hstack([X_te_proc.values, sec_te])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_full)
        X_te_s = scaler.transform(X_te_full)

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=50, subsample=0.8, random_state=RANDOM_STATE)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
        rank_ic = stats.spearmanr(preds, y_te)[0] if len(y_te) > 2 else np.nan
        fold_rank_ics.append(rank_ic)

    return np.mean(fold_rank_ics)


def run_full_cv(feature_names, X_numeric, y, sector, all_sectors, dates, splits):
    """Run all 8 models through CV, return results dict."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        has_xgb = False

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

            m = clone(model)
            m.fit(X_tr_s, y_tr)
            preds = m.predict(X_te_s)
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

        if hasattr(m, "feature_importances_"):
            all_names = feature_names + sec_names
            imp = pd.Series(m.feature_importances_, index=all_names)
            imp = imp.sort_values(ascending=False).head(15)
            log_msg(f"  top features: {', '.join(f'{n}={v:.3f}' for n, v in imp.items())}")

    return all_results


def test_report(test_df, best_model_name, X_train, y_train, sector_train, feature_names, all_sectors):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor)
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

    X_tr_df = pd.DataFrame(X_train, columns=feature_names)
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


# --- forward selection ---
def forward_selection(train_df, all_sectors):
    """Add FinBERT features one at a time to Model A's controls, keeping those that improve CV Rank IC."""
    base_features = list(MODEL_A_CONTROLS)
    candidates = [f for f in FINBERT_CANDIDATES if f in train_df.columns]
    log_msg(f"\n{'='*60}")
    log_msg(f"FORWARD FEATURE SELECTION")
    log_msg(f"Base features ({len(base_features)}): {', '.join(base_features)}")
    log_msg(f"FinBERT candidates ({len(candidates)}): {', '.join(candidates)}")

    y = train_df[PRIMARY_TARGET].values
    sector = train_df[SECTOR_COL].values
    dates = train_df["call_date"].values

    splits = time_series_cv_splits(dates)
    log_msg(f"CV splits: {len(splits)} folds")

    # Baseline: Model A controls only
    X_base = train_df[base_features].values
    base_rank_ic = cv_rank_ic(base_features, X_base, y, sector, all_sectors, dates, splits)
    log_msg(f"\nBaseline CV Rank IC (controls only): {base_rank_ic:+.4f}")

    selected_finbert = []
    selection_log = [{"step": 0, "added": None, "rank_ic": base_rank_ic, "improvement": 0.0}]
    current_features = list(base_features)
    current_rank_ic = base_rank_ic
    remaining = list(candidates)

    step = 0
    while remaining:
        step += 1
        log_msg(f"\n--- Step {step}: testing {len(remaining)} candidates ---")
        best_feat = None
        best_ic = current_rank_ic
        results = []

        for feat in remaining:
            trial_features = current_features + [feat]
            X_trial = train_df[trial_features].values
            trial_ic = cv_rank_ic(trial_features, X_trial, y, sector, all_sectors, dates, splits)
            results.append((feat, trial_ic))
            log_msg(f"  + {feat:30s} → CV Rank IC = {trial_ic:+.4f}  (Δ = {trial_ic - current_rank_ic:+.4f})")

        results.sort(key=lambda x: x[1], reverse=True)
        best_feat, best_ic = results[0]

        if best_ic > current_rank_ic:
            improvement = best_ic - current_rank_ic
            log_msg(f"\n  ✓ Adding {best_feat} (Rank IC {current_rank_ic:+.4f} → {best_ic:+.4f}, Δ={improvement:+.4f})")
            selected_finbert.append(best_feat)
            current_features.append(best_feat)
            current_rank_ic = best_ic
            remaining.remove(best_feat)
            selection_log.append({"step": step, "added": best_feat,
                                  "rank_ic": best_ic, "improvement": improvement})
        else:
            log_msg(f"\n  ✗ No candidate improves Rank IC. Stopping.")
            break

    log_msg(f"\n{'='*60}")
    log_msg(f"Forward selection complete.")
    log_msg(f"Selected FinBERT features ({len(selected_finbert)}): {', '.join(selected_finbert) if selected_finbert else '(none)'}")
    log_msg(f"Final feature set ({len(current_features)}): {', '.join(current_features)}")
    log_msg(f"Final CV Rank IC: {current_rank_ic:+.4f} (baseline was {base_rank_ic:+.4f})")

    return current_features, selected_finbert, selection_log


def write_report(base, variant, features, n_controls, n_finbert, finbert_names,
                 cv_results, test_metrics, best_model, selection_log=None):
    tag = "c" if variant == "C" else "c_all"
    out = Path(base) / f"model_{tag}_report.txt"
    lines = []
    lines.append("=" * 70)
    if variant == "C":
        lines.append("MODEL C (FinBERT, FORWARD SELECTION) REPORT")
    else:
        lines.append("MODEL C* (ALL FinBERT) REPORT — No Feature Selection")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Target: {PRIMARY_TARGET}")
    lines.append(f"Train: 2010–2021 | Test: 2022–2024")
    lines.append(f"Features ({len(features)}): {', '.join(features)}")
    lines.append(f"  Controls ({n_controls}): {', '.join(features[:n_controls])}")
    lines.append(f"  FinBERT ({n_finbert}): {', '.join(finbert_names)}")
    if variant == "C":
        lines.append(f"  Forward selection on CV Rank IC (GradientBoosting)")
    else:
        lines.append(f"  NO feature selection — all FinBERT features forced")
    lines.append("=" * 70)

    if selection_log:
        lines.append("\n\n--- FORWARD SELECTION LOG ---")
        lines.append(f"{'Step':>5s} {'Added':30s} {'CV Rank IC':>12s} {'Δ':>10s}")
        for entry in selection_log:
            feat_name = entry["added"] or "(baseline)"
            lines.append(f"{entry['step']:5d} {feat_name:30s} {entry['rank_ic']:+12.4f} {entry['improvement']:+10.4f}")

    lines.append("\n\n--- CROSS-VALIDATION RESULTS (train 2010-2021) ---")
    lines.append(f"{'Model':20s} {'IC':>8s} {'±':>6s} {'Rank IC':>8s} {'±':>6s} {'R²':>8s} {'Dir Acc':>8s} {'Q5-Q1':>8s}")
    ranked = sorted(cv_results.items(), key=lambda x: x[1]["avg"]["rank_ic"], reverse=True)
    for name, res in ranked:
        a, s = res["avg"], res["std"]
        lines.append(f"{name:20s} {a['ic']:+8.4f} {s['ic']:6.4f} {a['rank_ic']:+8.4f} {s['rank_ic']:6.4f} "
                     f"{a['r2']:+8.4f} {a['direction_accuracy']:8.3f} {a['quintile_spread']:+8.4f}")
    lines.append(f"\nBest model by avg Rank IC: {best_model}")

    if test_metrics:
        lines.append(f"\n\n--- TEST RESULTS 2022-2024 ({best_model}) ---")
        for k, v in test_metrics.items():
            lines.append(f"  {k:25s}: {v:+.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    lines.append("\n\n--- NOTES ---")
    lines.append("- Same preprocessing as Model A/B: winsorize 1%/99%, log1p, median impute, StandardScaler")
    lines.append("- FinBERT features from ProsusAI/finbert sentence-level scoring")
    if variant == "C":
        lines.append("- Forward selection: start with Model A's 12 controls, add FinBERT features one at a time")
        lines.append("- Keep feature if it improves CV Rank IC (GradientBoosting, expanding window)")
    else:
        lines.append("- All 13 FinBERT features forced (no selection)")
    lines.append("- Time-series expanding window CV on train (2010-2021)")
    lines.append("- Test (2022-2024) evaluated once after model selection")

    report_text = "\n".join(lines)
    out.write_text(report_text)
    log_msg(f"\nreport saved to {out}")
    return report_text


def save_results_json(base, variant, features, n_controls, finbert_names,
                      cv_results, test_metrics, best_model, selection_log=None):
    tag = "c" if variant == "C" else "c_all"
    out = Path(base) / f"model_{tag}_results.json"
    cv_clean = {}
    for name, res in cv_results.items():
        cv_clean[name] = {"avg": res["avg"], "std": res["std"]}
    obj = {
        "variant": f"Model {variant}" if variant == "C" else "Model C*",
        "best_model": best_model,
        "features_used": features,
        "n_controls": n_controls,
        "n_finbert": len(finbert_names),
        "finbert_features": finbert_names,
        "model_a_controls": list(MODEL_A_CONTROLS),
        "cv_results": cv_clean,
        "test": test_metrics,
    }
    if selection_log:
        obj["forward_selection_log"] = selection_log
    with open(out, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log_msg(f"results JSON saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = ap.parse_args()
    base = args.folder

    df, train, test = load_data(base)
    all_sectors = sorted(train[SECTOR_COL].dropna().unique())

    # ===== MODEL C: Forward selection =====
    log_msg("\n" + "=" * 70)
    log_msg("MODEL C — FinBERT features, forward selection")
    log_msg("=" * 70)

    selected_features, selected_finbert, selection_log = forward_selection(train, all_sectors)
    n_controls = len(MODEL_A_CONTROLS)
    n_finbert = len(selected_finbert)

    y_train = train[PRIMARY_TARGET].values
    sector_train = train[SECTOR_COL].values
    dates_train = train["call_date"].values
    X_train = train[selected_features].values
    splits = time_series_cv_splits(dates_train)

    log_msg("\nRunning full CV with selected features...")
    cv_results = run_full_cv(selected_features, X_train, y_train, sector_train,
                             all_sectors, dates_train, splits)
    best_model = max(cv_results, key=lambda k: cv_results[k]["avg"]["rank_ic"])
    log_msg(f"\nBest model: {best_model}")

    test_metrics = test_report(test, best_model, X_train, y_train, sector_train,
                               selected_features, all_sectors)

    write_report(base, "C", selected_features, n_controls, n_finbert, selected_finbert,
                 cv_results, test_metrics, best_model, selection_log)
    save_results_json(base, "C", selected_features, n_controls, selected_finbert,
                      cv_results, test_metrics, best_model, selection_log)

    # ===== MODEL C*: All FinBERT forced =====
    log_msg("\n" + "=" * 70)
    log_msg("MODEL C* — ALL FinBERT features forced (no selection)")
    log_msg("=" * 70)

    all_finbert = [f for f in FINBERT_CANDIDATES if f in train.columns]
    all_features = list(MODEL_A_CONTROLS) + all_finbert
    n_fb_all = len(all_finbert)
    log_msg(f"Features ({len(all_features)}): {n_controls} controls + {n_fb_all} FinBERT")

    X_train_all = train[all_features].values
    splits_all = time_series_cv_splits(dates_train)

    cv_results_all = run_full_cv(all_features, X_train_all, y_train, sector_train,
                                 all_sectors, dates_train, splits_all)
    best_model_all = max(cv_results_all, key=lambda k: cv_results_all[k]["avg"]["rank_ic"])
    log_msg(f"\nBest model: {best_model_all}")

    test_metrics_all = test_report(test, best_model_all, X_train_all, y_train, sector_train,
                                   all_features, all_sectors)

    write_report(base, "C*", all_features, n_controls, n_fb_all, all_finbert,
                 cv_results_all, test_metrics_all, best_model_all)
    save_results_json(base, "C*", all_features, n_controls, all_finbert,
                      cv_results_all, test_metrics_all, best_model_all)

    log_msg("\n" + "=" * 70)
    log_msg("COMPARISON")
    log_msg("=" * 70)
    c_best_cv = cv_results[best_model]["avg"]
    cs_best_cv = cv_results_all[best_model_all]["avg"]
    log_msg(f"Model C  ({n_controls} ctrl + {n_finbert} FinBERT, selected):  "
            f"CV Rank IC {c_best_cv['rank_ic']:+.4f}, Test {test_metrics['rank_ic']:+.4f}")
    log_msg(f"Model C* ({n_controls} ctrl + {n_fb_all} FinBERT, ALL forced): "
            f"CV Rank IC {cs_best_cv['rank_ic']:+.4f}, Test {test_metrics_all['rank_ic']:+.4f}")
    log_msg("\ndone")


if __name__ == "__main__":
    main()
