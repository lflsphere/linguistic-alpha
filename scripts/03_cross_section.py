#!/usr/bin/env python
"""
03: Steps 2 and 3 of the framework on the development sample (2012-2023). Never reads the holdout.

  Step 2  Model A baseline; single-signal, family and joint Model B regressions with FE and clustered SEs;
          annual Fama-MacBeth; pooled, partial and annual Rank IC (ICIR); point-in-time quantile sorts;
          Benjamini-Hochberg FDR across signals x horizons.
  Step 3  Alpha decay: beta_h and IC_h for cumulative windows [2,5], [2,20], [2,60] and the
          non-overlapping buckets [2,5], [6,20], [21,60].

Options (each non-default combination writes to its own folder outputs/cross_section/<tag>/)
  --fe firm,year | firm | year | none     fixed effects (industry FE = firm FE in this panel)
  --cluster event_q | year | hc3          SE clustering (calendar quarter of day 0 by default)
  --sue SUE_TS | SUE_PCT | SUE_IBES       earnings-surprise definition in Model A
  --sample all | highprec                 highprec drops events with cent-rounded consensus (NVDA pre-2020)
  --delayed                               add CAR[0,+1] to Model A (delayed-signal robustness spec)
  --reps N                                block-bootstrap draws for IC inference
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402

log = C.get_logger("xsec")


def full_rank(feats: list[str]) -> list[str]:
    """Drop exact linear combinations before joint OLS."""
    f = list(feats)
    if {"TONE_P", "TONE_Q", "TONE_GAP"} <= set(f):
        f.remove("TONE_Q")
    if {"dTONE_P", "dTONE_Q", "dTONE_GAP"} <= set(f):      # dTONE_GAP = dTONE_Q - dTONE_P exactly
        f.remove("dTONE_Q")
    return f


class Design:
    """In-sample design: winsorize / impute / z-score on the full dev sample (in-sample inference only)."""

    def __init__(self, ev: pd.DataFrame, a_cols: list[str], fe, cluster: str):
        self.ev, self.a_cols, self.cluster = ev, a_cols, cluster
        self.Z = C.Preprocessor(a_cols + C.LING_ALL).fit(ev).transform(ev)
        self.FE = C.fe_dummies(ev, fe)

    def fit(self, y: str, extra: list[str]):
        m = self.ev[y].notna()
        X = pd.concat([self.Z.loc[m, self.a_cols + extra], self.FE.loc[m]], axis=1)
        groups = None if self.cluster == "hc3" else self.ev.loc[m, self.cluster]
        return C.ols_fit(self.ev.loc[m, y], X, groups), m


# ============================================================================ Step 2 blocks
def model_a_baseline(D: Design, targets: list[str]) -> pd.DataFrame:
    rows = []
    for y in targets:
        res, m = D.fit(y, [])
        row = dict(target=y, n=int(m.sum()), r2=res.rsquared, adj_r2=res.rsquared_adj)
        for a in D.a_cols:
            row[f"b_{a}"], row[f"t_{a}"] = res.params[a], res.tvalues[a]
        rows.append(row)
    return pd.DataFrame(rows)


def single_signal(D: Design, targets: list[str]) -> pd.DataFrame:
    rows = []
    for y in targets:
        base, _ = D.fit(y, [])
        for s in C.LING_ALL:
            res, m = D.fit(y, [s])
            rows.append(dict(target=y, signal=s, family=C.FAMILY_OF[s], beta=res.params[s], se=res.bse[s],
                             t=res.tvalues[s], p=res.pvalues[s], d_r2=res.rsquared - base.rsquared,
                             d_adj_r2=res.rsquared_adj - base.rsquared_adj, n=int(m.sum())))
    out = pd.DataFrame(rows)
    out["q_fdr"] = C.bh_qvalues(out["p"])
    return out


def joint_tests(D: Design, targets: list[str]) -> pd.DataFrame:
    rows = []
    sets = {**{f"family:{k}": full_rank(v) for k, v in C.LING_FAMILIES.items()}, "core": C.LING_CORE}
    for y in targets:
        base, _ = D.fit(y, [])
        for name, feats in sets.items():
            res, m = D.fit(y, feats)
            F, p = C.joint_wald(res, feats)
            rows.append(dict(target=y, block=name, k=len(feats), wald_F=F, p=p,
                             d_r2=res.rsquared - base.rsquared, d_adj_r2=res.rsquared_adj - base.rsquared_adj,
                             n=int(m.sum())))
    out = pd.DataFrame(rows)
    out["q_fdr"] = C.bh_qvalues(out["p"])
    return out


def fama_macbeth(ev: pd.DataFrame, Z: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Annual cross-sections (~12 events each): y ~ 1 + signal + SUE. Quarterly sections have N = 3."""
    rows = []
    for y in targets:
        for s in C.LING_ALL:
            coefs = []
            for _, g in ev.groupby("year"):
                g = g[g[y].notna()]
                if len(g) < 8:
                    continue
                X = np.column_stack([np.ones(len(g)), Z.loc[g.index, s], Z.loc[g.index, "SUE"]])
                coefs.append(np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0][1])
            coefs = np.asarray(coefs)
            se = C.nw_se_mean(coefs, lags=1)
            rows.append(dict(target=y, signal=s, fm_beta=coefs.mean(), fm_t=coefs.mean() / se if se else np.nan,
                             fm_pos_share=(coefs > 0).mean(), n_years=len(coefs)))
    return pd.DataFrame(rows)


def rank_ic(D: Design, targets: list[str], reps: int) -> pd.DataFrame:
    ev, rows = D.ev, []
    for y in targets:
        m = ev[y].notna()
        XA = pd.concat([D.Z.loc[m, D.a_cols], D.FE.loc[m]], axis=1)
        XA = np.column_stack([np.ones(m.sum()), XA.to_numpy()])
        proj = XA @ np.linalg.pinv(XA)
        ry = ev.loc[m, y].to_numpy() - proj @ ev.loc[m, y].to_numpy()
        for s in C.LING_ALL:
            sub = ev.loc[m, ["event_q", "year", s, y]].copy()
            sub["_rs"] = D.Z.loc[m, s].to_numpy() - proj @ D.Z.loc[m, s].to_numpy()
            sub["_ry"] = ry
            sub = sub.reset_index(drop=True)
            a, b, ra, rb = (sub[c].to_numpy(float) for c in (s, y, "_rs", "_ry"))
            ic, pic = C.spearman(a, b), C.spearman(ra, rb)
            draws = C.block_indices(sub, reps=reps)
            b_ic = C.bootstrap(lambda i: C.spearman(a[i], b[i]), draws)
            b_pic = C.bootstrap(lambda i: C.spearman(ra[i], rb[i]), draws)
            ann = C.ic_stats(C.ic_series(sub, s, y, by="year"))
            rows.append(dict(target=y, signal=s, ic=ic, ic_lo=np.nanpercentile(b_ic, 2.5),
                             ic_hi=np.nanpercentile(b_ic, 97.5), ic_p=C.boot_p_two_sided(b_ic),
                             partial_ic=pic, partial_ic_p=C.boot_p_two_sided(b_pic), **ann))
    out = pd.DataFrame(rows)
    out["ic_q_fdr"] = C.bh_qvalues(out["ic_p"])
    out["partial_ic_q_fdr"] = C.bh_qvalues(out["partial_ic_p"])
    return out


def quantile_sorts(ev_dev: pd.DataFrame, ev_hist: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Point-in-time buckets: each event ranked against signals of strictly earlier calls (any firm)."""
    rows = []
    hist = ev_hist.sort_values("day0").reset_index(drop=True)
    for s in C.LING_ALL:
        b = pd.Series(C.pit_bucket(hist["day0"], hist[s]), index=hist["transcriptid"])
        bucket = ev_dev["transcriptid"].map(b)
        for y in targets:
            d = pd.DataFrame({"b": bucket, "y": ev_dev[y]}).dropna()
            means = d.groupby("b")["y"].mean()
            if len(means) < C.N_QUANTILES:
                continue
            top, bot = d.loc[d.b == C.N_QUANTILES, "y"], d.loc[d.b == 1, "y"]
            t, p = stats.ttest_ind(top, bot, equal_var=False)
            diffs = np.diff(means.to_numpy())
            row = dict(target=y, signal=s, spread=top.mean() - bot.mean(), spread_t=t, spread_p=p,
                       monotonic=bool((diffs > 0).all() or (diffs < 0).all()), n=len(d))
            row.update({f"q{int(k)}": v for k, v in means.items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    out["spread_q_fdr"] = C.bh_qvalues(out["spread_p"])
    return out


# ============================================================================ Step 3: decay
def decay_plots(single: pd.DataFrame, ic: pd.DataFrame, out_dir) -> pd.DataFrame:
    h = {"CAR_2_5": 5, "CAR_2_20": 20, "CAR_2_60": 60}
    cum = single[single.target.isin(h)].assign(h=lambda d: d.target.map(h))
    cum = cum.merge(ic[["target", "signal", "ic", "partial_ic"]], on=["target", "signal"])
    n = len(C.LING_ALL)
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(16, 3.2 * np.ceil(n / 4)))
    for ax, s in zip(axes.ravel(), C.LING_ALL):
        d = cum[cum.signal == s].sort_values("h")
        ax.errorbar(d.h, d.beta * 1e4, yerr=1.96 * d.se * 1e4, marker="o", capsize=3, label="beta (bps per SD)")
        ax.axhline(0, color="grey", lw=0.8)
        ax2 = ax.twinx()
        ax2.plot(d.h, d.ic, "s--", color="tab:orange", label="Rank IC")
        ax2.plot(d.h, d.partial_ic, "^:", color="tab:green", label="partial IC")
        ax2.set_ylim(-0.4, 0.4)
        ax.set_title(s, fontsize=10)
        ax.set_xticks([5, 20, 60])
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    C.savefig(fig, out_dir / "decay_cumulative.png")

    bk = single[single.target.isin(C.BUCKET_TARGETS)]
    piv_b = bk.pivot(index="signal", columns="target", values="beta")[C.BUCKET_TARGETS] * 1e4
    piv_t = bk.pivot(index="signal", columns="target", values="t")[C.BUCKET_TARGETS]
    fig, ax = plt.subplots(figsize=(12, 4))
    piv_b.loc[C.LING_ALL].plot.bar(ax=ax)
    ax.set_ylabel("beta, bps of CAR per 1 SD")
    ax.set_title("Non-overlapping decay buckets")
    C.savefig(fig, out_dir / "decay_buckets.png")
    tab = piv_b.add_prefix("bps_").join(piv_t.add_prefix("t_")).loc[C.LING_ALL]
    tab.to_csv(out_dir / "decay_buckets.csv")
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fe", default=",".join(C.FE_DEFAULT))
    ap.add_argument("--cluster", default=C.CLUSTER_DEFAULT, choices=["event_q", "year", "hc3"])
    ap.add_argument("--sue", default=C.SUE_VARIANT, choices=["SUE_TS", "SUE_PCT", "SUE_IBES"])
    ap.add_argument("--sample", default="all", choices=["all", "highprec"])
    ap.add_argument("--delayed", action="store_true")
    ap.add_argument("--reps", type=int, default=C.BOOT_REPS)
    args = ap.parse_args()

    C.SUE_VARIANT = args.sue
    fe = tuple(f for f in args.fe.split(",") if f and f != "none")
    tag_parts = [p for p, d in ((f"fe-{'-'.join(fe) or 'none'}", fe != C.FE_DEFAULT),
                                (f"cl-{args.cluster}", args.cluster != C.CLUSTER_DEFAULT),
                                (args.sue, args.sue != "SUE_TS"), (args.sample, args.sample != "all"),
                                ("delayed", args.delayed)) if d]
    out_dir = C.stage_dir(f"cross_section/{'_'.join(tag_parts) or 'base'}")

    ev = C.load_events("dev", args.sample)
    C.assert_no_holdout(ev)
    ev_hist = C.load_events("all", args.sample)
    ev_hist = ev_hist[ev_hist["year"] <= C.DEV_END]            # history for PIT buckets: burn-in + dev
    a_cols = C.model_a_cols(args.delayed)
    D = Design(ev, a_cols, fe, args.cluster)
    log.info("dev sample: %d events | FE=%s | cluster=%s | SUE=%s | delayed=%s", len(ev), fe, args.cluster,
             args.sue, args.delayed)

    targets = C.TARGETS
    base = model_a_baseline(D, targets)
    single = single_signal(D, targets)
    joint = joint_tests(D, targets)
    fm = fama_macbeth(ev, D.Z, targets)
    log.info("rank IC with %d block-bootstrap draws per test", args.reps)
    ic = rank_ic(D, targets, args.reps)
    sorts = quantile_sorts(ev, ev_hist, targets)
    decay = decay_plots(single, ic, out_dir)

    master = single.merge(fm, on=["target", "signal"]).merge(
        ic.drop(columns=["ic_lo", "ic_hi"]), on=["target", "signal"]).merge(
        sorts[["target", "signal", "spread", "spread_t", "monotonic", "spread_q_fdr"]], on=["target", "signal"],
        how="left")
    for name, df in (("model_a_baseline", base), ("single_signal", single), ("joint_tests", joint),
                     ("fama_macbeth", fm), ("rank_ic", ic), ("quantile_sorts", sorts), ("factor_scorecard", master)):
        df.to_csv(out_dir / f"{name}.csv", index=False)

    pt = C.PRIMARY_TARGET
    card = master[master.target == pt].set_index("signal")[
        ["family", "beta", "t", "p", "q_fdr", "d_adj_r2", "fm_t", "ic", "partial_ic", "partial_ic_q_fdr",
         "ic_mean", "icir", "spread", "monotonic"]]
    card["beta"] *= 1e4
    md = [f"# Steps 2-3: cross-sectional evidence ({len(ev)} dev events, tag `{out_dir.name}`)", "",
          f"FE: {fe or 'none'} | SE: {args.cluster} | SUE: {args.sue} | sample: {args.sample} | "
          f"delayed: {args.delayed}. Signals are z-scored; beta in bps of CAR per 1 SD.", "",
          "## Model A baseline", C.md_table(base[["target", "n", "r2", "adj_r2"] + [f"t_{a}" for a in a_cols]],
                                             index=False), "",
          f"## Factor scorecard, primary target {pt}", C.md_table(card), "",
          "## Joint tests (Wald F on Model B blocks)", C.md_table(joint, index=False), "",
          "## Decay (non-overlapping buckets, bps per SD and t)", C.md_table(decay), "",
          "## Signals with FDR q < 0.10 (regression, any target)",
          C.md_table(single[single.q_fdr < 0.10][["target", "signal", "beta", "t", "q_fdr"]], index=False)
          if (single.q_fdr < 0.10).any() else "none", ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    pd.set_option("display.width", 200)
    print(base[["target", "n", "r2", "adj_r2"]].round(3).to_string(index=False))
    print(card.round(3).to_string())
    print(joint.round(3).to_string(index=False))
    log.info("outputs in %s", out_dir)


if __name__ == "__main__":
    main()
