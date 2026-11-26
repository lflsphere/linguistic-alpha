"""
Robustness checks and feature importance for a sector results folder.

The it_model_* protocol picks algorithm, strategy and ensemble members by HOLDOUT Rank IC.
Here the algorithm is picked by time-series CV inside train only, so holdout numbers are
not selected on the holdout. Adds bootstrap CIs (clustered by firm) and a Model A report.

Usage: PYTHONPATH=src python cd_diagnostics.py --folder cd_results --prefix cd --sector-label CD
"""
import argparse, copy, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import it_model_b as B


def cv_pick(train, feats):
    _, algo = B.ts_cv_rank_ic(train, feats, B.PRIMARY_TARGET, B.get_algos())
    return algo


def fit_predict(train, test, feats, algo):
    Xtr, Xte, _, _ = B.preprocess(train, test, feats)
    m = copy.deepcopy(B.get_algos()[algo])
    m.fit(Xtr, stats.rankdata(train[B.PRIMARY_TARGET].values))
    return m.predict(Xte), m


def boot(y, preds, firms, n=5000, seed=42):
    """Firm-clustered percentile CIs for each model's Rank IC and its difference vs. the first."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(firms == f) for f in np.unique(firms)]
    out = {k: [] for k in preds}
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k, p in preds.items():
            out[k].append(stats.spearmanr(p[idx], y[idx])[0])
    base = np.array(out[next(iter(preds))])
    q = lambda a: [round(float(v), 4) for v in np.percentile(a, [2.5, 97.5])]
    return {k: {"ci95": q(v), "delta_vs_A_ci95": q(np.array(v) - base)} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="cd_results")
    ap.add_argument("--prefix", default="cd")
    ap.add_argument("--sector-label", default="CD")
    a = ap.parse_args()
    out, imp = {}, {}
    fb_path = f"{a.folder}/finbert_features.parquet"
    fb = pd.read_parquet(fb_path) if os.path.exists(fb_path) else None
    for opt, fn in [("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
                    (f"Option 2 ({a.sector_label} sector bench)", "model_table_opt2.parquet")]:
        tag = opt.lower().replace(" ", "_").replace("(", "").replace(")", "")
        df = pd.read_parquet(f"{a.folder}/{fn}")
        df["call_date"] = pd.to_datetime(df["call_date"])
        if fb is not None:
            df = df.merge(fb, on="transcriptid", how="left")
        df = df[df[B.PRIMARY_TARGET].notna()]
        cut = pd.Timestamp(B.TRAIN_END)
        train = df[df["call_date"] <= cut].reset_index(drop=True)
        test = df[df["call_date"] > cut].reset_index(drop=True)
        y = test[B.PRIMARY_TARGET].values
        base = [c for c in B.CONTROL_CANDIDATES if c in df]
        sets = {"A (16 controls)": base, "B* (controls + 9 LM)": base + B.LINGUISTIC_FEATURES}
        if fb is not None:
            sets["C* (controls + 13 FinBERT)"] = base + [c for c in fb.columns if c.startswith("fb_")]
        res, preds = {}, {}
        for k, feats in sets.items():
            algo = cv_pick(train, feats)
            p, m = fit_predict(train, test, feats, algo)
            preds[k] = p
            res[k] = {"cv_algo": algo, "rank_ic": round(float(stats.spearmanr(p, y)[0]), 4),
                      "p_value": round(float(stats.spearmanr(p, y)[1]), 4),
                      "quintile_spread": round(float(B.evaluate(y, p)["quintile_spread"]), 4)}
            coef = pd.Series(m.coef_, index=feats)
            tot = coef.abs().sum()
            imp[f"{k} - {opt} ({algo})"] = [
                {"feature": f, "std_coef": float(coef[f]), "abs_std_coef": float(abs(coef[f])),
                 "rel_importance_pct": float(abs(coef[f]) / tot * 100) if tot else 0.0}
                for f in coef.abs().sort_values(ascending=False).index]
        for k, v in boot(y, preds, test["permno"].values).items():
            res[k].update(v)
        out[opt] = {"n_train": len(train), "n_test": len(test), "test_firms": int(test["permno"].nunique()),
                    "test_dates": [str(test["call_date"].min().date()), str(test["call_date"].max().date())],
                    "share_test_target_positive": round(float((y > 0).mean()), 4), "models": res}

        # Model A report in the it_model_a_* format (current protocol: all 16 controls)
        Xtr, Xte, _, _ = B.preprocess(train, test, base)
        allr = B.fit_and_eval(Xtr, Xte, stats.rankdata(train[B.PRIMARY_TARGET].values), y, B.get_algos())
        best, bm = B.best_result(allr)
        lines = ["=" * 70, f"{a.sector_label} Sector — Model A — {opt}",
                 f"Target: {B.PRIMARY_TARGET}; trained on ranks (R² not meaningful)",
                 f"Train: {len(train):,} (≤{B.TRAIN_END}) | Test: {len(test):,} (>{B.TRAIN_END})",
                 f"Companies: {df['permno'].nunique()}", f"Features ({len(base)}): {', '.join(base)}", "=" * 70, "",
                 "Algo                       IC   Rank IC    Q5-Q1"]
        lines += [f"{n:20s} {m['ic']:+8.4f}  {m['rank_ic']:+8.4f} {m['quintile_spread']:+8.4f}"
                  for n, m in sorted(allr.items(), key=lambda x: -x[1]["rank_ic"])]
        lines += ["", f"Best (by holdout, IT protocol): {best}", f"Picked by train CV: {res['A (16 controls)']['cv_algo']}"]
        Path(a.folder, f"{a.prefix}_model_a_{tag}_report.txt").write_text("\n".join(lines))
        Path(a.folder, f"{a.prefix}_model_a_{tag}_results.json").write_text(json.dumps(
            {"option": opt, "features": base, "best_algo": best,
             "results": {n: {k: round(float(v), 6) for k, v in m.items()} for n, m in allr.items()}}, indent=2))
    Path(a.folder, f"{a.prefix}_diagnostics.json").write_text(json.dumps(out, indent=2))
    Path(a.folder, "feature_importance.json").write_text(json.dumps(imp, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
