#!/usr/bin/env python
"""
04: Step 4 of the framework. Out-of-sample prediction, Models A / B / C / C_alone, development sample only.

Models (common.spec_features; configs/default.yaml models.measures)
  A        controls
  B        A + dictionary measures (NLTK + Loughran-McDonald)
  C        B + FinBERT sentence tone                     -> C - B: what FinBERT adds to the dictionary
  C_alone  A + FinBERT only                              -> C_alone - B: dictionary vs. FinBERT head to head
  C and C_alone need the FinBERT columns (01_build_features.py --finbert); without them only A and B run.

Reaction legs (--legs, default both): every model runs once per leg, with the announcement reaction added
to Model A's controls: m1_1 = CAR[-1,+1] (AR_M11, main) and 0_1 = CAR[0,+1] (AR01, comparison). Outputs go
to outputs/oos/<tag>_<leg>/. A leg whose column is missing is skipped with a warning (the legacy panel has
CAR[0,+1] only). --freeze selects on the main leg when it ran, and freezes both legs.

Design
  * Expanding windows: train 2012..y-1, test y, for y = FIRST_TEST_YEAR..DEV_END (Section 6).
  * Purging: a training event is used only if its target window closed before the first test event of the
    fold (a 60-day CAR from a November call is not known in January).
  * All preprocessing (winsorize, impute, z-score) is fitted on the training fold only. Firm dummies stand in
    for industry FE; year FE cannot be estimated for an unseen year.
  * Estimators: OLS (full-rank subsets), Elastic Net (every measure, inner TimeSeriesSplit CV), optional
    XGBoost; plus single-signal OLS (A vs A + s) for every linguistic factor, embeddings and LLM ones included.
  * Metrics: pooled Rank IC, annual IC and ICIR over test years, OOS R^2 (vs training mean and vs zero),
    RMSE, MAE; increments Delta IC / Delta R^2 with quarter-block bootstrap p-values for every pair in
    common.COMPARISONS; Clark-West only for nested pairs (not for C_alone vs B).

  --freeze writes outputs/frozen_spec.json (target, model, features, tuning procedure, data hash), the only
  input 05_backtest_holdout.py --stage holdout accepts.

Options: --targets, --models ols,enet[,xgb], --legs m1_1,0_1, --sample all|highprec, --sue, --freeze,
         --freeze-model
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

log = C.get_logger("oos")
L1_GRID = [0.1, 0.5, 0.9, 1.0]


# ============================================================================ models
def enet_alphas(X: np.ndarray, y: np.ndarray, n: int = 40) -> np.ndarray:
    a_max = np.abs((X - X.mean(0)).T @ (y - y.mean())).max() / (len(y) * min(L1_GRID))
    return np.logspace(np.log10(a_max), np.log10(a_max * 1e-3), n)


def fit_predict(kind: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame):
    """Returns (test predictions, coefficient Series or None, tuning info)."""
    if kind == "ols":
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(Xtr, ytr)
        return m.predict(Xte), pd.Series(m.coef_, index=Xtr.columns), {}
    if kind == "enet":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        m = ElasticNetCV(l1_ratio=L1_GRID, alphas=enet_alphas(Xtr.to_numpy(), ytr), cv=TimeSeriesSplit(3),
                         max_iter=50000, random_state=C.SEED).fit(Xtr, ytr)
        return m.predict(Xte), pd.Series(m.coef_, index=Xtr.columns), {"alpha": m.alpha_, "l1_ratio": m.l1_ratio_}
    if kind == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, reg_lambda=5.0, random_state=C.SEED, n_jobs=1).fit(Xtr, ytr)
        return m.predict(Xte), pd.Series(m.feature_importances_, index=Xtr.columns), {}
    raise ValueError(kind)


def design(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], firms: list[str]):
    pp = C.Preprocessor(feats).fit(train)
    cats = {"firm": firms}
    Xtr = pp.transform(train).join(C.fe_dummies(train, ["firm"], cats))
    Xte = pp.transform(test).join(C.fe_dummies(test, ["firm"], cats))
    return Xtr, Xte


def folds(ev: pd.DataFrame, target: str, first: int = C.FIRST_TEST_YEAR, last: int = C.DEV_END):
    for yr in range(first, last + 1):
        test = ev[(ev["year"] == yr) & ev[target].notna()]
        if test.empty:
            continue
        train = C.purged_train(ev, target, test["day0"].min())
        yield yr, train[train[target].notna()], test


def single_signals(fb: bool) -> list[str]:
    return C.LING_ALL + (C.LING_FB if fb else [])


def specs(models: list[str], a_cols: list[str], fb: bool) -> list[tuple[str, str, list[str]]]:
    out = []
    for k in models:
        out += [(k, name, feats) for name, feats in C.spec_features(a_cols, k, fb).items()]
    out += [("ols", f"B:{s}", a_cols + [s]) for s in single_signals(fb)]
    return out


def run_oos(ev: pd.DataFrame, targets: list[str], models: list[str], a_cols: list[str], fb: bool):
    firms = sorted(ev["firm"].astype(str).unique())
    preds, coefs, tuning = [], [], []
    for target in targets:
        for yr, train, test in folds(ev, target):
            ytr = train[target].to_numpy()
            for kind, spec, feats in specs(models, a_cols, fb):
                Xtr, Xte = design(train, test, feats, firms)
                p, coef, info = fit_predict(kind, Xtr, ytr, Xte)
                preds.append(test[["transcriptid", "ticker", "day0", "year", "event_q", target]]
                             .rename(columns={target: "y"})
                             .assign(target=target, model=kind, spec=spec, pred=p, bench=ytr.mean(),
                                     n_train=len(train)))
                if (spec == "A" or spec in C.MAIN_SPECS) and coef is not None:
                    coefs.append(coef.rename("value").to_frame().assign(target=target, model=kind, spec=spec,
                                                                          test_year=yr).reset_index(names="feature"))
                if info:
                    tuning.append(dict(target=target, model=kind, spec=spec, test_year=yr, **info))
            log.info("%s | test %d | train n=%d (purged) | test n=%d", target, yr, len(train), len(test))
    return pd.concat(preds, ignore_index=True), pd.concat(coefs, ignore_index=True), pd.DataFrame(tuning)


# ============================================================================ evaluation
def metrics(d: pd.DataFrame) -> dict:
    y, f, b = d["y"].to_numpy(), d["pred"].to_numpy(), d["bench"].to_numpy()
    ann = C.ic_stats(C.ic_series(d, "pred", "y", by="year"))
    is_ret = d["target"].iloc[0] in C.RETURN_TARGETS        # a zero forecast is only meaningful for returns
    return dict(n=len(d), rank_ic=C.spearman(f, y), ic_mean=ann["ic_mean"], icir=ann["icir"],
                oos_r2=C.oos_r2(y, f, b), oos_r2_zero=C.oos_r2(y, f, np.zeros_like(y)) if is_ret else np.nan,
                rmse=float(np.sqrt(np.mean((y - f) ** 2))), mae=float(np.mean(np.abs(y - f))))


def increment(g: pd.DataFrame, base: str, alt: str, reps: int, nested: bool) -> dict:
    """alt minus base on the same test events; Clark-West only when base is nested in alt."""
    A = g[g.spec == base].set_index("transcriptid")
    B = g[g.spec == alt].set_index("transcriptid").loc[A.index]
    d = A[["y", "pred", "bench", "event_q", "year"]].rename(columns={"pred": "fA"}).assign(fB=B["pred"])
    d = d.reset_index(drop=True)
    y, fa, fb, bm = (d[c].to_numpy(float) for c in ("y", "fA", "fB", "bench"))

    def d_ic(i):
        return C.spearman(fb[i], y[i]) - C.spearman(fa[i], y[i])

    def d_r2(i):
        return C.oos_r2(y[i], fb[i], bm[i]) - C.oos_r2(y[i], fa[i], bm[i])

    draws = C.block_indices(d, reps=reps)
    b_ic, b_r2 = C.bootstrap(d_ic, draws), C.bootstrap(d_r2, draws)
    cw_t, cw_p = C.clark_west(y, fa, fb) if nested else (np.nan, np.nan)
    ia = C.ic_stats(C.ic_series(d.rename(columns={"fA": "f"}), "f", "y"))
    ib = C.ic_stats(C.ic_series(d.rename(columns={"fB": "f"}), "f", "y"))
    return dict(base=base, spec=alt, nested=nested,
                signal=alt.split(":", 1)[1] if ":" in alt else "ALL",
                d_ic=d_ic(np.arange(len(d))), d_ic_p=C.boot_p_two_sided(b_ic), d_icir=ib["icir"] - ia["icir"],
                d_oos_r2=d_r2(np.arange(len(d))), d_r2_p=C.boot_p_two_sided(b_r2), cw_t=cw_t, cw_p=cw_p, n=len(d))


def compare(preds: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [dict(target=t, model=m, spec=s, **metrics(g)) for (t, m, s), g in preds.groupby(["target", "model", "spec"])]
    met = pd.DataFrame(rows)
    inc = []
    for (t, m), g in preds.groupby(["target", "model"]):
        have = set(g.spec)
        pairs = [p for p in C.COMPARISONS if {p[0], p[1]} <= have]
        pairs += [("A", s, True) for s in sorted(have) if s.startswith("B:")]
        inc += [dict(target=t, model=m, **increment(g, b, a, reps, n)) for b, a, n in pairs]
    inc = pd.DataFrame(inc)
    single = inc["spec"].str.startswith("B:")
    inc.loc[single, "cw_q_fdr"] = C.bh_qvalues(inc.loc[single, "cw_p"])
    return met, inc


def plots(preds: pd.DataFrame, coefs: pd.DataFrame, out_dir) -> None:
    keys = [(t, m) for t, m in preds[preds.spec == "B"][["target", "model"]].drop_duplicates().itertuples(index=False)]
    mains = [s for s in C.MAIN_SPECS if s in set(preds.spec)]
    fig, axes = plt.subplots(1, len({t for t, _ in keys}), figsize=(4.5 * len({t for t, _ in keys}), 3.6), squeeze=False)
    for ax, t in zip(axes.ravel(), sorted({t for t, _ in keys})):
        for _, m in [k for k in keys if k[0] == t]:
            g = preds[(preds.target == t) & (preds.model == m)]
            A = g[g.spec == "A"].sort_values("day0").set_index("transcriptid")
            for s in mains:
                B = g[g.spec == s].set_index("transcriptid").loc[A.index]
                ax.plot(A["day0"], np.cumsum((A.y - A.pred) ** 2 - (A.y - B.pred) ** 2), label=f"{m} {s}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{t}: cumulative SSE(A) - SSE(model)", fontsize=9)
        ax.legend(fontsize=7)
    C.savefig(fig, out_dir / "cumulative_sse_diff.png")

    widest = "C" if "C" in mains else "B"                  # dictionary + FinBERT when available
    ling = C.LING_DICT + (C.LING_FB if widest == "C" else [])
    en = coefs[(coefs.model == "enet") & (coefs.spec == widest) & coefs.feature.isin(ling)]
    if len(en):
        sel = en.assign(nz=en.value.abs() > 1e-10).pivot_table(index="feature", columns="target", values="nz",
                                                               aggfunc="mean").reindex(ling)
        sel.to_csv(out_dir / "enet_selection_frequency.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        sel.plot.barh(ax=ax)
        ax.set_xlabel("share of folds with non-zero coefficient")
        C.savefig(fig, out_dir / "enet_selection_frequency.png")


def freeze(met: pd.DataFrame, inc: pd.DataFrame, args, leg: str, legs: list[str], out_dir, fb: bool) -> dict:
    """Freeze A and every language model for one target and estimator; the holdout scores all of them once.
    The estimator is the one whose best language model (B, C or C_alone) beats A most on dev."""
    target = args.freeze_target
    cand = inc[(inc.target == target) & (inc.base == "A") & inc.spec.isin(C.MAIN_SPECS)].copy()
    if args.freeze_model:
        cand = cand[cand.model == args.freeze_model]
    if cand.empty:
        raise SystemExit(f"no language-model results for {target} to freeze")
    best = cand.sort_values(["d_oos_r2", "d_ic"], ascending=False).iloc[0]
    k = best.model
    by_leg = {g: C.spec_features(C.model_a_cols(leg=g), k, fb) for g in legs}
    feats = by_leg[leg]
    mets = {s: met[(met.target == target) & (met.model == k) & (met.spec == s)].iloc[0].to_dict() for s in feats}
    incs = inc[(inc.target == target) & (inc.model == k) & ~inc.spec.str.startswith("B:")]
    cols = ("d_ic", "d_ic_p", "d_oos_r2", "d_r2_p", "cw_t", "cw_p")
    spec = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        features_sha256=C.file_sha256(C.FEATURES_PATH), features_meta=C.load_meta(),
        reaction_sha256=C.file_sha256(C.REACTION_PATH) if C.REACTION_PATH.exists() else None,
        target=target, model=k, sue_variant=C.SUE_VARIANT, sample=args.sample,
        firm_fe=True, selected_leg=leg, legs=by_leg, features=feats, selected_spec=best.spec,
        features_A=feats["A"], features_B=feats["B"],              # kept for older readers
        tuning=("ElasticNetCV, inner TimeSeriesSplit(3), l1_ratio grid " + str(L1_GRID) + ", 40 alphas; "
                "re-tuned on the full purged dev sample") if k == "enet" else "none",
        selection_rule=("forced by --freeze-model" if args.freeze_model else
                        f"estimator with the max dev OOS Delta R^2 vs A over B / C / C_alone on leg {leg}, "
                        "ties by Delta IC"),
        source_run=out_dir.name,
        validation=dict(metrics=mets, increments=[dict(base=r.base, spec=r.spec, **{c: r[c] for c in cols})
                                                  for _, r in incs.iterrows()]),
        verdict=(f"{best.spec} beats A in validation" if best.d_oos_r2 > 0 and best.cw_p < 0.10 else
                 f"no language model beats A in validation (best: {best.spec}); "
                 "holdout acts as a confirmation of the negative"))
    C.FROZEN_SPEC_PATH.write_text(json.dumps(spec, indent=2, default=float))
    log.info("frozen spec: %s / %s -> %s (%s)", target, k, C.FROZEN_SPEC_PATH, spec["verdict"])
    return spec


def run_leg(ev, leg: str, targets, models, fb: bool, args, tag: str):
    a_cols = C.model_a_cols(leg=leg)
    out_dir = C.stage_dir(f"oos/{tag}_{leg}")
    preds, coefs, tuning = run_oos(ev, targets, models, a_cols, fb)
    met, inc = compare(preds, args.reps)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    coefs.to_csv(out_dir / "coefficients.csv", index=False)
    tuning.to_csv(out_dir / "tuning.csv", index=False)
    met.to_csv(out_dir / "metrics.csv", index=False)
    inc.to_csv(out_dir / "increments.csv", index=False)
    plots(preds, coefs, out_dir)

    main_tab = met[~met.spec.str.startswith("B:")].sort_values(["target", "model", "spec"])
    inc_main = inc[~inc.spec.str.startswith("B:")].drop(columns=["signal", "cw_q_fdr"], errors="ignore")
    inc_single = inc[inc.spec.str.startswith("B:")].sort_values(["target", "cw_p"])
    md = [f"# Step 4: out-of-sample Models A / B / C / C_alone (tag `{tag}`, leg `{leg}`)", "",
          f"Reaction control in every model: {C.REACTION_LEGS[leg]}. Purged expanding windows, test years "
          f"{C.FIRST_TEST_YEAR}-{C.DEV_END}; OOS R^2 vs training-mean forecast. {len(ev)} dev events.", "",
          "## Model metrics", C.md_table(main_tab, index=False), "",
          "## Increments (spec minus base): B, C, C_alone vs A; C vs B (FinBERT added); "
          "C_alone vs B (FinBERT vs dictionary, not nested: no Clark-West)",
          C.md_table(inc_main, index=False), "",
          "## Single-signal increments (OLS, A + s vs A)",
          C.md_table(inc_single[["target", "signal", "d_ic", "d_oos_r2", "cw_t", "cw_p", "cw_q_fdr"]], index=False), ""]
    pd.set_option("display.width", 220)
    print(f"--- leg {leg}")
    print(main_tab.round(4).to_string(index=False))
    print(inc_main.round(4).to_string(index=False))
    return met, inc, out_dir, md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(C.OOS_TARGETS))
    ap.add_argument("--models", default="ols,enet")
    ap.add_argument("--legs", default=",".join(C.REACTION_LEGS))
    ap.add_argument("--sample", default="all", choices=["all", "highprec"])
    ap.add_argument("--sue", default=C.SUE_VARIANT, choices=["SUE_TS", "SUE_PCT", "SUE_IBES"])
    ap.add_argument("--reps", type=int, default=C.BOOT_REPS)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--freeze-target", default=C.PRIMARY_TARGET)
    ap.add_argument("--freeze-model", default=None)
    args = ap.parse_args()

    C.SUE_VARIANT = args.sue
    models = [m for m in args.models.split(",") if m]
    if "xgb" in models:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            log.warning("xgboost not installed: skipping")
            models.remove("xgb")
    tag = "_".join(p for p, d in ((args.sample, args.sample != "all"),
                                  (args.sue, args.sue != "SUE_TS")) if d) or "base"

    ev = C.load_events("dev", args.sample)
    C.assert_no_holdout(ev)
    wanted = [g for g in args.legs.split(",") if g]
    legs = C.available_legs(ev, wanted)
    for g in sorted(set(wanted) - set(legs)):
        log.warning("leg %s skipped: column %s not in data/features.csv", g, C.REACTION_LEGS[g])
    if not legs:
        raise SystemExit("no reaction leg available")
    fb = C.has_finbert(ev)
    if not fb:
        log.warning("no FinBERT columns (%s): running A and B only; build them with "
                    "01_build_features.py --finbert to add C and C_alone", ", ".join(C.LING_FB))
    targets = [t for t in args.targets.split(",") if t]
    runs = {g: run_leg(ev, g, targets, models, fb, args, tag) for g in legs}

    if args.freeze:
        leg = C.MAIN_LEG if C.MAIN_LEG in runs else legs[0]
        if leg != C.MAIN_LEG:
            log.warning("main leg %s did not run: selecting on leg %s", C.MAIN_LEG, leg)
        met, inc, out_dir, md = runs[leg]
        spec = freeze(met, inc, args, leg, legs, out_dir, fb)
        md += ["## Frozen specification", f"`{spec['target']}` / `{spec['model']}` (selected on leg `{leg}`, "
               f"frozen legs {legs}): {spec['verdict']}", ""]
    for g, (_, _, out_dir, md) in runs.items():
        (out_dir / "summary.md").write_text("\n".join(md))
        log.info("outputs in %s", out_dir)


if __name__ == "__main__":
    main()
