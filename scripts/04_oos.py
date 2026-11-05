#!/usr/bin/env python
"""
04: Step 4 of the framework. Out-of-sample prediction, Model A vs Model B, development sample only.

Design
  * Expanding windows: train 2012..y-1, test y, for y = FIRST_TEST_YEAR..DEV_END (Section 6).
  * Purging: a training event is used only if its target window closed before the first test event of the
    fold (a 60-day CAR from a November call is not known in January).
  * All preprocessing (winsorize, impute, z-score) is fitted on the training fold only. Firm dummies stand in
    for industry FE; year FE cannot be estimated for an unseen year.
  * Models: OLS (A vs A + LING_CORE), Elastic Net (A vs A + LING_ALL, inner TimeSeriesSplit CV),
    optional XGBoost; plus single-signal OLS (A vs A + s) for every linguistic factor.
  * Metrics: pooled Rank IC, annual IC and ICIR over test years, OOS R^2 (vs training mean and vs zero),
    RMSE, MAE; increments Delta IC / Delta R^2 with quarter-block bootstrap p-values; Clark-West for nesting.

  --freeze writes outputs/frozen_spec.json (target, model, features, tuning procedure, data hash), the only
  input 05_backtest_holdout.py --stage holdout accepts.

Options: --targets, --models ols,enet[,xgb], --delayed, --sample all|highprec, --sue, --freeze, --freeze-model
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


def specs(models: list[str], a_cols: list[str]) -> list[tuple[str, str, list[str]]]:
    out = []
    for k in models:
        out.append((k, "A", a_cols))
        out.append((k, "B", a_cols + (C.LING_CORE if k == "ols" else C.LING_ALL)))
    out += [("ols", f"B:{s}", a_cols + [s]) for s in C.LING_ALL]
    return out


def run_oos(ev: pd.DataFrame, targets: list[str], models: list[str], a_cols: list[str]):
    firms = sorted(ev["firm"].astype(str).unique())
    preds, coefs, tuning = [], [], []
    for target in targets:
        for yr, train, test in folds(ev, target):
            ytr = train[target].to_numpy()
            for kind, spec, feats in specs(models, a_cols):
                Xtr, Xte = design(train, test, feats, firms)
                p, coef, info = fit_predict(kind, Xtr, ytr, Xte)
                preds.append(test[["transcriptid", "ticker", "day0", "year", "event_q", target]]
                             .rename(columns={target: "y"})
                             .assign(target=target, model=kind, spec=spec, pred=p, bench=ytr.mean(),
                                     n_train=len(train)))
                if spec in ("A", "B") and coef is not None:
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


def compare(preds: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [dict(target=t, model=m, spec=s, **metrics(g)) for (t, m, s), g in preds.groupby(["target", "model", "spec"])]
    met = pd.DataFrame(rows)
    inc = []
    for (t, m), g in preds.groupby(["target", "model"]):
        A = g[g.spec == "A"].set_index("transcriptid")
        for s in sorted(set(g.spec) - {"A"}):
            B = g[g.spec == s].set_index("transcriptid").loc[A.index]
            d = A[["y", "pred", "bench", "event_q", "year"]].rename(columns={"pred": "fA"}).assign(fB=B["pred"])
            d = d.reset_index(drop=True)
            y, fa, fb, bm = (d[c].to_numpy(float) for c in ("y", "fA", "fB", "bench"))

            def d_ic(i):
                return C.spearman(fb[i], y[i]) - C.spearman(fa[i], y[i])

            def d_r2(i):
                return C.oos_r2(y[i], fb[i], bm[i]) - C.oos_r2(y[i], fa[i], bm[i])

            draws = C.block_indices(d, reps=reps)
            b_ic, b_r2 = C.bootstrap(d_ic, draws), C.bootstrap(d_r2, draws)
            cw_t, cw_p = C.clark_west(y, fa, fb)
            ia = C.ic_stats(C.ic_series(d.rename(columns={"fA": "f"}), "f", "y"))
            ib = C.ic_stats(C.ic_series(d.rename(columns={"fB": "f"}), "f", "y"))
            inc.append(dict(target=t, model=m, spec=s, signal=s.split(":", 1)[1] if ":" in s else "ALL",
                            d_ic=d_ic(np.arange(len(d))), d_ic_p=C.boot_p_two_sided(b_ic),
                            d_icir=ib["icir"] - ia["icir"], d_oos_r2=d_r2(np.arange(len(d))), d_r2_p=C.boot_p_two_sided(b_r2), cw_t=cw_t, cw_p=cw_p, n=len(d)))
    inc = pd.DataFrame(inc)
    single = inc["spec"].str.startswith("B:")
    inc.loc[single, "cw_q_fdr"] = C.bh_qvalues(inc.loc[single, "cw_p"])
    return met, inc


def plots(preds: pd.DataFrame, coefs: pd.DataFrame, out_dir) -> None:
    keys = [(t, m) for t, m in preds[preds.spec == "B"][["target", "model"]].drop_duplicates().itertuples(index=False)]
    fig, axes = plt.subplots(1, len({t for t, _ in keys}), figsize=(4.5 * len({t for t, _ in keys}), 3.6), squeeze=False)
    for ax, t in zip(axes.ravel(), sorted({t for t, _ in keys})):
        for _, m in [k for k in keys if k[0] == t]:
            g = preds[(preds.target == t) & (preds.model == m)]
            A = g[g.spec == "A"].sort_values("day0").set_index("transcriptid")
            B = g[g.spec == "B"].set_index("transcriptid").loc[A.index]
            ax.plot(A["day0"], np.cumsum((A.y - A.pred) ** 2 - (A.y - B.pred) ** 2), label=m)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{t}: cumulative SSE(A) - SSE(B)", fontsize=9)
        ax.legend(fontsize=7)
    C.savefig(fig, out_dir / "cumulative_sse_diff.png")

    en = coefs[(coefs.model == "enet") & (coefs.spec == "B") & coefs.feature.isin(C.LING_ALL)]
    if len(en):
        sel = en.assign(nz=en.value.abs() > 1e-10).pivot_table(index="feature", columns="target", values="nz",
                                                               aggfunc="mean").reindex(C.LING_ALL)
        sel.to_csv(out_dir / "enet_selection_frequency.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        sel.plot.barh(ax=ax)
        ax.set_xlabel("share of folds with non-zero coefficient")
        C.savefig(fig, out_dir / "enet_selection_frequency.png")


def freeze(met: pd.DataFrame, inc: pd.DataFrame, args, a_cols: list[str], out_dir) -> dict:
    target = args.freeze_target
    cand = inc[(inc.target == target) & (inc.spec == "B")].copy()
    if args.freeze_model:
        cand = cand[cand.model == args.freeze_model]
    if cand.empty:
        raise SystemExit(f"no Model B results for {target} to freeze")
    best = cand.sort_values(["d_oos_r2", "d_ic"], ascending=False).iloc[0]
    k = best.model
    feats_b = a_cols + (C.LING_CORE if k == "ols" else C.LING_ALL)
    mA = met[(met.target == target) & (met.model == k) & (met.spec == "A")].iloc[0].to_dict()
    mB = met[(met.target == target) & (met.model == k) & (met.spec == "B")].iloc[0].to_dict()
    spec = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        features_sha256=C.file_sha256(C.FEATURES_PATH), features_meta=C.load_meta(),
        target=target, model=k, sue_variant=C.SUE_VARIANT, sample=args.sample, delayed=bool(args.delayed),
        firm_fe=True, features_A=a_cols, features_B=feats_b,
        tuning=("ElasticNetCV, inner TimeSeriesSplit(3), l1_ratio grid " + str(L1_GRID) + ", 40 alphas; "
                "re-tuned on the full purged dev sample") if k == "enet" else "none",
        selection_rule=("forced by --freeze-model" if args.freeze_model else
                        "max dev OOS Delta R^2 (B - A) for the target, ties by Delta IC"),
        source_run=out_dir.name,
        validation=dict(model_A=mA, model_B=mB, increment={c: best[c] for c in
                                                           ("d_ic", "d_ic_p", "d_oos_r2", "d_r2_p", "cw_t", "cw_p")}),
        verdict=("B beats A in validation" if best.d_oos_r2 > 0 and best.cw_p < 0.10 else
                 "B does NOT beat A in validation; holdout acts as a confirmation of the negative"))
    C.FROZEN_SPEC_PATH.write_text(json.dumps(spec, indent=2, default=float))
    log.info("frozen spec: %s / %s -> %s (%s)", target, k, C.FROZEN_SPEC_PATH, spec["verdict"])
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=",".join(C.OOS_TARGETS))
    ap.add_argument("--models", default="ols,enet")
    ap.add_argument("--delayed", action="store_true")
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
    tag = "_".join(p for p, d in (("delayed", args.delayed), (args.sample, args.sample != "all"),
                                  (args.sue, args.sue != "SUE_TS")) if d) or "base"
    out_dir = C.stage_dir(f"oos/{tag}")

    ev = C.load_events("dev", args.sample)
    C.assert_no_holdout(ev)
    a_cols = C.model_a_cols(args.delayed)
    targets = [t for t in args.targets.split(",") if t]
    preds, coefs, tuning = run_oos(ev, targets, models, a_cols)
    met, inc = compare(preds, args.reps)

    preds.to_csv(out_dir / "predictions.csv", index=False)
    coefs.to_csv(out_dir / "coefficients.csv", index=False)
    tuning.to_csv(out_dir / "tuning.csv", index=False)
    met.to_csv(out_dir / "metrics.csv", index=False)
    inc.to_csv(out_dir / "increments.csv", index=False)
    plots(preds, coefs, out_dir)

    main_tab = met[~met.spec.str.startswith("B:")].sort_values(["target", "model", "spec"])
    inc_main = inc[inc.spec == "B"].drop(columns=["signal", "cw_q_fdr"], errors="ignore")
    inc_single = inc[inc.spec.str.startswith("B:")].sort_values(["target", "cw_p"])
    md = [f"# Step 4: out-of-sample Model A vs Model B (tag `{tag}`)", "",
          f"Purged expanding windows, test years {C.FIRST_TEST_YEAR}-{C.DEV_END}; OOS R^2 vs training-mean "
          f"forecast. {len(ev)} dev events.", "",
          "## Model metrics", C.md_table(main_tab, index=False), "",
          "## Increment of Model B over Model A", C.md_table(inc_main, index=False), "",
          "## Single-signal increments (OLS, A + s vs A)",
          C.md_table(inc_single[["target", "signal", "d_ic", "d_oos_r2", "cw_t", "cw_p", "cw_q_fdr"]], index=False), ""]
    if args.freeze:
        spec = freeze(met, inc, args, a_cols, out_dir)
        md += ["## Frozen specification", f"`{spec['target']}` / `{spec['model']}`: {spec['verdict']}", ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    pd.set_option("display.width", 220)
    print(main_tab.round(4).to_string(index=False))
    print(inc_main.round(4).to_string(index=False))
    log.info("outputs in %s", out_dir)


if __name__ == "__main__":
    main()
