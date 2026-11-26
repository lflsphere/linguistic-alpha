"""
Models A / B / B* / C / C* with every choice made on train/validation only.

Differences from it_model_b.py / it_model_c_finbert.py (which pick algorithm, strategy, PCA size,
interaction variant and ensemble members by HOLDOUT Rank IC):
  * Every candidate (feature set x algorithm) is scored by expanding-window time-series CV inside
    train (validation years 2021, 2022, 2023). The algorithm, the strategy and the ensemble members
    are the CV argmax. The holdout (2024) is scored once, after all choices are frozen.
  * Purge: CAR[+2,+20] ends ~30 calendar days after the call, so training rows within PURGE_DAYS of a
    validation/test start are dropped (their labels overlap the evaluation period).
  * Every fitted transform (winsor/impute/scale, residualisation, PCA) is fitted on the fold's train.
  * One algorithm pool for all models (linear + RF + GB + XGBoost if installed), so A, B and C compete
    on equal terms.
Forward selection and ensemble-member choice reuse the same CV folds, so their CV score is optimistic
(selection on validation); the holdout number is not.

Usage: PYTHONPATH=src python cv_protocol.py --folder cd_results --prefix cd --sector-label CD
"""
import argparse, copy, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

import it_model_c_finbert as C

warnings.filterwarnings("ignore")
TARGET, TRAIN_END, PURGE_DAYS = C.PRIMARY_TARGET, pd.Timestamp(C.TRAIN_END), 35
FOLDS = [(pd.Timestamp(a), pd.Timestamp(b), pd.Timestamp(c)) for a, b, c in C.TS_CV_SPLITS]
CONTROLS = ["sue", "surprise_pct", "vol_pre", "car_-1_1", "mom", "bm", "size", "dispersion",
            "avg_answer_words", "words_prepared", "words_questions", "words_answers", "n_questions",
            "n_answers", "n_analysts", "prior_calls_with_target"]
LM, FB = C.LINGUISTIC_FEATURES, C.FINBERT_CANDIDATES
INTERACTIONS = [("tone_answers", "sue"), ("tone_answers_chg", "car_-1_1"), ("tone_answers_chg", "sue"),
                ("hedging_answers", "vol_pre"), ("tone_gap", "sue"), ("tone_prepared_chg", "mom"),
                ("uncertainty_answers", "vol_pre"), ("tone_answers", "size")]


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def algos():
    a = C.get_all_algos()
    a.update({"Ridge_a50": Ridge(alpha=50.0), "Ridge_a100": Ridge(alpha=100.0),
              "EN_a0.1": ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000)})
    return a


def ric(p, y):
    r = stats.spearmanr(p, y)[0]
    return 0.0 if np.isnan(r) else r


# ---- feature builders: (train_df, eval_df) -> (train_df, eval_df, features), fitted on train_df only
def fixed(feats):
    return lambda tr, te: (tr, te, list(feats))


def with_interactions(base, raw_ling):
    def f(tr, te):
        tr, te, names = tr.copy(), te.copy(), []
        for a, b in INTERACTIONS:
            n = f"{a}x{b}"
            tr[n], te[n] = tr[a] * tr[b], te[a] * te[b]
            names.append(n)
        return tr, te, base + (LM if raw_ling else []) + names
    return f


def residualised(base):
    def f(tr, te):
        tr, te, names = tr.copy(), te.copy(), []
        ctl = ["sue", "car_-1_1", "size"]
        for c in ["tone_answers", "tone_prepared", "tone_answers_chg", "tone_prepared_chg"]:
            reg = LinearRegression().fit(tr[ctl].fillna(0), tr[c].fillna(0))
            tr[c + "_resid"] = tr[c].fillna(0) - reg.predict(tr[ctl].fillna(0))
            te[c + "_resid"] = te[c].fillna(0) - reg.predict(te[ctl].fillna(0))
            names.append(c + "_resid")
        return tr, te, base + names
    return f


def pca_ling(base, k):
    def f(tr, te):
        tr, te = tr.copy(), te.copy()
        a, b = tr[LM].copy(), te[LM].copy()
        for c in LM:
            a[c], lo, hi, med = C.winsorize_and_impute(a[c])
            b[c], *_ = C.winsorize_and_impute(b[c], lo, hi, med)
        sc = StandardScaler().fit(a.values)
        p = PCA(n_components=k, random_state=C.RANDOM_STATE).fit(sc.transform(a.values))
        za, zb = p.transform(sc.transform(a.values)), p.transform(sc.transform(b.values))
        names = [f"ling_pc{i+1}" for i in range(k)]
        for i, n in enumerate(names):
            tr[n], te[n] = za[:, i], zb[:, i]
        return tr, te, base + names
    return f


# ---- CV machinery
def splits(train):
    for tr_end, v0, v1 in FOLDS:
        ftr = train[train["call_date"] <= tr_end - pd.Timedelta(days=PURGE_DAYS)]
        fva = train[train["call_date"].between(v0, v1)]
        if len(ftr) >= 100 and len(fva) >= 50:
            yield ftr, fva


def fit_pred(tr, te, builder, algo):
    tr2, te2, feats = builder(tr, te)
    Xtr, Xte, _, _ = C.preprocess(tr2, te2, feats)
    m = copy.deepcopy(algo).fit(Xtr, stats.rankdata(tr2[TARGET].values))
    return m.predict(Xte)


def cv_scores(train, builder, pool):
    """Per-algo list of fold Rank ICs, plus fold predictions for later ensembling."""
    sc, preds = {k: [] for k in pool}, {k: [] for k in pool}
    for ftr, fva in splits(train):
        tr2, te2, feats = builder(ftr, fva)
        Xtr, Xte, _, _ = C.preprocess(tr2, te2, feats)
        yr = stats.rankdata(tr2[TARGET].values)
        for k, a in pool.items():
            p = copy.deepcopy(a).fit(Xtr, yr).predict(Xte)
            sc[k].append(ric(p, fva[TARGET].values))
            preds[k].append(p)
    return sc, preds


def best_algo(train, builder, pool):
    sc, preds = cv_scores(train, builder, pool)
    k = max(sc, key=lambda a: np.mean(sc[a]))
    return k, float(np.mean(sc[k])), [round(float(v), 4) for v in sc[k]], preds[k]


def candidate(name, train, builder, pool):
    algo, cv, folds, preds = best_algo(train, builder, pool)
    log(f"    {name:40s} CV={cv:+.4f} {folds} ({algo})")
    return {"name": name, "builder": builder, "algo": algo, "cv": cv, "cv_folds": folds, "cv_preds": preds}


def forward_select(train, base, pool_feats, pool, label):
    cur, remaining = list(base), list(pool_feats)
    algo, best, _, _ = best_algo(train, fixed(cur), pool)
    steps = [{"added": None, "cv": round(best, 4), "algo": algo}]
    while remaining:
        trials = {f: best_algo(train, fixed(cur + [f]), pool)[:2] for f in remaining}
        f = max(trials, key=lambda k: trials[k][1])
        if trials[f][1] <= best:
            break
        cur.append(f); remaining.remove(f); best = trials[f][1]
        steps.append({"added": f, "cv": round(best, 4), "algo": trials[f][0]})
        log(f"    {label} forward: + {f} -> CV {best:+.4f} ({trials[f][0]})")
    return cur, steps


def ensemble(train, base, feats, pool, base_cv, label):
    """Members = single-feature models whose CV beats the base model's CV (chosen on CV, not holdout)."""
    singles = {f: candidate(f"{label} single +{f}", train, fixed(base + [f]), pool) for f in feats}
    members = [s for s in singles.values() if s["cv"] > base_cv]
    if len(members) < 2:
        return None
    nf = len(members[0]["cv_preds"])
    fold_ics = []
    for i, (_, fva) in enumerate(splits(train)):
        if i >= nf:
            break
        fold_ics.append(ric(np.mean([m["cv_preds"][i] for m in members], axis=0), fva[TARGET].values))
    return {"name": f"{label} ensemble ({len(members)})", "members": members, "algo": "Ensemble",
            "cv": float(np.mean(fold_ics)), "cv_folds": [round(float(v), 4) for v in fold_ics],
            "added": [m["name"].split("+", 1)[1] for m in members]}


# ---- holdout scoring (once, after all choices are frozen)
def holdout_pred(c, train, test, pool):
    if c["algo"] == "Ensemble":
        return np.mean([holdout_pred(m, train, test, pool) for m in c["members"]], axis=0)
    return fit_pred(train, test, c["builder"], pool[c["algo"]])


def metrics(y, p):
    q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
    qm = pd.Series(y).groupby(q.values).mean()
    r, pv = stats.spearmanr(p, y)
    return {"rank_ic": round(float(r), 4), "p_value": round(float(pv), 4),
            "ic": round(float(stats.pearsonr(p, y)[0]), 4), "q5_q1": round(float(qm.iloc[-1] - qm.iloc[0]), 4)}


def boot(y, preds, firms, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(firms == f) for f in np.unique(firms)]
    out = {k: [] for k in preds}
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k, p in preds.items():
            out[k].append(stats.spearmanr(p[idx], y[idx])[0])
    base = np.array(out["A"])
    q = lambda a: [round(float(v), 4) for v in np.nanpercentile(a, [2.5, 97.5])]
    return {k: {"ci95": q(v), "delta_vs_A_ci95": q(np.array(v) - base)} for k, v in out.items()}


def run(df, opt):
    df = df[df[TARGET].notna()].copy()
    train = df[df["call_date"] <= TRAIN_END - pd.Timedelta(days=PURGE_DAYS)].reset_index(drop=True)
    test = df[df["call_date"] > TRAIN_END].reset_index(drop=True)
    purged = int(((df["call_date"] <= TRAIN_END) & (df["call_date"] > TRAIN_END - pd.Timedelta(days=PURGE_DAYS))).sum())
    log(f"### {opt}: train {len(train):,} (purged {purged}), test {len(test):,}")
    pool, base = algos(), [c for c in CONTROLS if c in df]
    has_fb = all(f in df for f in FB)

    log("  Model A")
    A = candidate("A (16 controls)", train, fixed(base), pool)

    log("  Model B candidates")
    fs_feats, fs_log = forward_select(train, base, LM, pool, "B")
    B = [candidate("B forward selection", train, fixed(fs_feats), pool),
         candidate("B interactions only", train, with_interactions(base, False), pool),
         candidate("B interactions + raw LM", train, with_interactions(base, True), pool),
         candidate("B residualised tone", train, residualised(base), pool),
         candidate("B* all 9 LM", train, fixed(base + LM), pool)]
    B += [candidate(f"B PCA({k}) LM", train, pca_ling(base, k), pool) for k in (2, 3, 4)]
    ens = ensemble(train, base, LM, pool, A["cv"], "B")
    if ens:
        log(f"    {ens['name']:40s} CV={ens['cv']:+.4f} {ens['cv_folds']}")
        B.append(ens)
        B.append(candidate("B combined good LM", train, fixed(base + ens["added"]), pool))
    B[0]["added"] = fs_feats[len(base):]
    groups = {"A": [A], "B": B}

    if has_fb:
        log("  Model C candidates")
        fc_feats, fc_log = forward_select(train, base, FB, pool, "C")
        Cc = [candidate("C forward selection", train, fixed(fc_feats), pool),
              candidate("C* all 13 FinBERT", train, fixed(base + FB), pool),
              candidate("C FinBERT + LM", train, fixed(base + FB + LM), pool)]
        Cc[0]["added"] = fc_feats[len(base):]
        ens = ensemble(train, base, FB, pool, A["cv"], "C")
        if ens:
            log(f"    {ens['name']:40s} CV={ens['cv']:+.4f} {ens['cv_folds']}")
            Cc.append(ens)
        groups["C"] = Cc

    y, out, preds = test[TARGET].values, {}, {}
    for g, cands in groups.items():
        win = max(cands, key=lambda c: c["cv"])           # strategy picked on CV
        for c in cands:
            c["holdout"] = metrics(y, holdout_pred(c, train, test, pool))
        preds[g] = holdout_pred(win, train, test, pool)
        out[g] = {"selected": win["name"], "algo": win["algo"], "cv": round(win["cv"], 4),
                  "holdout": win["holdout"],
                  "candidates": [{k: c.get(k) for k in ["name", "algo", "cv", "cv_folds", "added", "holdout"]}
                                 for c in sorted(cands, key=lambda c: -c["cv"])]}
    for star, name in [("B*", "B* all 9 LM"), ("C*", "C* all 13 FinBERT")]:
        c = next((c for c in groups.get(star[0], []) if c["name"] == name), None)
        if c:
            preds[star] = holdout_pred(c, train, test, pool)
            out[star] = {"selected": name, "algo": c["algo"], "cv": round(c["cv"], 4), "holdout": c["holdout"]}
    for k, v in boot(y, preds, test["permno"].values).items():
        out[k]["holdout"].update(v)
    out["forward_selection"] = {"B": fs_log, **({"C": fc_log} if has_fb else {})}
    out["meta"] = {"n_train": len(train), "n_purged": purged, "n_test": len(test),
                   "test_firms": int(test["permno"].nunique()), "purge_days": PURGE_DAYS,
                   "algos": list(pool), "folds": [[str(d.date()) for d in f] for f in FOLDS]}
    return out


def report(res, label):
    L = [f"{label} — every choice made by train-only time-series CV (purge {PURGE_DAYS}d); holdout scored once", ""]
    for opt, r in res.items():
        m = r["meta"]
        L += ["=" * 92, f"{opt}: train {m['n_train']:,} (purged {m['n_purged']}), test {m['n_test']} ({m['test_firms']} firms)",
              "=" * 92, f"{'Model':6s} {'selected on CV':34s} {'algo':17s} {'CV':>8s} {'Holdout':>8s} {'p':>6s} "
              f"{'Q5-Q1':>7s}  {'95% CI':17s} {'Δ vs A 95% CI':17s}"]
        for g in ["A", "B", "B*", "C", "C*"]:
            if g not in r:
                continue
            h = r[g]["holdout"]
            L.append(f"{g:6s} {r[g]['selected']:34s} {r[g]['algo']:17s} {r[g]['cv']:+8.4f} {h['rank_ic']:+8.4f} "
                     f"{h['p_value']:6.3f} {h['q5_q1']:+7.4f}  [{h['ci95'][0]:+.3f},{h['ci95'][1]:+.3f}]  "
                     + ("" if g == "A" else f"[{h['delta_vs_A_ci95'][0]:+.3f},{h['delta_vs_A_ci95'][1]:+.3f}]"))
        for g in ["B", "C"]:
            if g not in r:
                continue
            L += ["", f"{g} candidates (sorted by CV; holdout shown for information only, never used to choose):"]
            for c in r[g]["candidates"]:
                L.append(f"  {c['name']:38s} {c['algo']:17s} CV={c['cv']:+.4f} {c['cv_folds']}  holdout={c['holdout']['rank_ic']:+.4f}")
            L.append(f"  forward selection: " + " -> ".join(
                f"{s['added'] or 'base'} {s['cv']:+.4f}" for s in r["forward_selection"][g]))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="cd_results")
    ap.add_argument("--prefix", default="cd")
    ap.add_argument("--sector-label", default="CD")
    a = ap.parse_args()
    log(f"XGBoost available: {C.HAS_XGB}")
    fb_path = f"{a.folder}/finbert_features.parquet"
    fb = pd.read_parquet(fb_path) if os.path.exists(fb_path) else None
    res = {}
    for opt, fn in [("Option 1 (S&P 500 bench)", "model_table_opt1.parquet"),
                    (f"Option 2 ({a.sector_label} sector bench)", "model_table_opt2.parquet")]:
        df = pd.read_parquet(f"{a.folder}/{fn}")
        df["call_date"] = pd.to_datetime(df["call_date"])
        if fb is not None:
            df = df.merge(fb, on="transcriptid", how="left")
        res[opt] = run(df, opt)
    Path(a.folder, f"{a.prefix}_cv_protocol_results.json").write_text(json.dumps(res, indent=2))
    Path(a.folder, f"{a.prefix}_cv_protocol_report.txt").write_text(report(res, a.sector_label))
    print(report(res, a.sector_label))


if __name__ == "__main__":
    main()
