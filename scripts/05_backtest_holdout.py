#!/usr/bin/env python
"""
05: economic evaluation (Section 7) and the final untouched holdout (Section 6).

  --stage dev      backtest the development-period OOS predictions written by 04_oos.py
                   (outputs/oos/<tag>/predictions.csv), every return horizon, Model A vs Model B
  --stage holdout  load outputs/frozen_spec.json, check that data/features.csv is unchanged since freezing,
                   refit A and B once on the purged 2012-2023 sample, predict the 2024 calls, evaluate
                   statistically and economically. Each run is appended to outputs/holdout/holdout_log.json;
                   a second run is refused without --force because repeated looks void the holdout.

Trading rule: enter at the close of day +1 (after the excluded [0,+1] reaction), exit at the close of day +h.
  directional  w = sign(prediction)                        inherits the sample's unconditional drift
  balanced     w = sign(prediction - training-mean target)  bets only on the model's conditional view
  long         w = +1                                       benchmark: post-call drift of the universe
Each trade is long/short the stock against an equal notional S&P 500 hedge, so its return is w x CAR[+2,+h].
Cost per round trip: 2 sides x (stock + hedge) bps. Quarterly strategy return: equal-weight mean of the
trades entered in the calendar quarter; Sharpe annualised with sqrt(4).
Limits from the panel: no daily returns, so no daily P&L, intra-trade drawdown, factor or beta exposure;
exposure is reported by name and long/short share.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

log = C.get_logger("backtest")
oos = importlib.import_module("04_oos")
TRADE_TARGETS = ["CAR_2_5", "CAR_2_20", "CAR_2_60"]


# ============================================================================ strategy
def weights(d: pd.DataFrame, rule: str) -> np.ndarray:
    if rule == "long":
        return np.ones(len(d))
    if rule == "directional":
        return np.sign(d["pred"].to_numpy())
    if rule == "balanced":
        return np.sign((d["pred"] - d["bench"]).to_numpy())
    raise ValueError(rule)


def backtest(d: pd.DataFrame, rule: str, cost_bps: float) -> tuple[dict, pd.DataFrame, pd.Series]:
    d = d.sort_values("day0").copy()
    d["w"] = weights(d, rule)
    d["gross"] = d["w"] * d["y"]
    d["net"] = d["gross"] - d["w"].abs() * 2 * cost_bps / 1e4
    q = d.groupby("event_q")["net"].mean()
    cum = q.cumsum()
    act = d[d["w"] != 0]
    years = max((d["day0"].max() - d["day0"].min()).days / 365.25, 1.0)
    se = C.nw_se_mean(act["net"], lags=2)
    res = dict(n_trades=len(act), hit_rate=(act["net"] > 0).mean(), long_share=(act["w"] > 0).mean(),
               mean_gross_bps=act["gross"].mean() * 1e4, mean_net_bps=act["net"].mean() * 1e4,
               t_net=act["net"].mean() / se if se else np.nan,
               sharpe_ann=q.mean() / q.std() * 2 if q.std() > 0 else np.nan,
               max_drawdown=float((cum - cum.cummax()).min()), total_net=float(cum.iloc[-1]) if len(cum) else np.nan,
               round_trips_per_year=len(act) / years,
               breakeven_bps_per_side=act["gross"].mean() / 2 * 1e4)
    res.update({f"w_{t}": g["w"].mean() for t, g in d.groupby("ticker")})
    return res, d, cum


def evaluate_block(preds: pd.DataFrame, cost_bps: float) -> tuple[pd.DataFrame, dict]:
    """Backtests for every (target, model) with A/B x rule, plus the long benchmark."""
    rows, curves = [], {}
    for (t, m), g in preds[preds["spec"].isin(["A", "B"])].groupby(["target", "model"]):
        if t not in TRADE_TARGETS:
            continue
        A = g[g.spec == "A"]
        r, _, cum = backtest(A, "long", cost_bps)
        rows.append(dict(target=t, model=m, strategy="long", **r))
        curves[(t, m, "long")] = cum
        for spec in ("A", "B"):
            for rule in ("directional", "balanced"):
                r, _, cum = backtest(g[g.spec == spec], rule, cost_bps)
                rows.append(dict(target=t, model=m, strategy=f"{spec}-{rule}", **r))
                curves[(t, m, f"{spec}-{rule}")] = cum
    return pd.DataFrame(rows), curves


def plot_curves(curves: dict, out_path, title: str) -> None:
    keys = sorted({(t, m) for t, m, _ in curves})
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6), squeeze=False)
    for ax, (t, m) in zip(axes.ravel(), keys):
        for s in ("long", "A-balanced", "B-balanced", "B-directional"):
            if (t, m, s) in curves:
                c = curves[(t, m, s)]
                ax.plot(pd.PeriodIndex(c.index, freq="Q").to_timestamp(), c.values, label=s,
                        lw=2 if s == "B-balanced" else 1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{t} / {m}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    C.savefig(fig, out_path)


# ============================================================================ stages
def stage_dev(args, cost_bps: float) -> None:
    src = C.OUT_DIR / "oos" / args.tag / "predictions.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found: run 04_oos.py first")
    preds = pd.read_csv(src, parse_dates=["day0"])
    C.assert_no_holdout(preds)
    out_dir = C.stage_dir(f"backtest/{args.tag}")
    res, curves = evaluate_block(preds, cost_bps)
    res.to_csv(out_dir / "backtest_dev.csv", index=False)
    plot_curves(curves, out_dir / "cumulative_net_dev.png",
                f"Dev OOS ({C.FIRST_TEST_YEAR}-{C.DEV_END}), net of {cost_bps:.0f} bps per side")
    decay = res[res.strategy.isin(["long", "A-balanced", "B-balanced"])].pivot_table(
        index=["model", "strategy"], columns="target", values="sharpe_ann")
    decay.to_csv(out_dir / "sharpe_by_horizon.csv")
    cols = ["target", "model", "strategy", "n_trades", "hit_rate", "long_share", "mean_gross_bps", "mean_net_bps",
            "t_net", "sharpe_ann", "max_drawdown", "breakeven_bps_per_side"]
    md = [f"# Section 7: economic evaluation on dev OOS predictions (tag `{args.tag}`)", "",
          f"Costs: {cost_bps:.1f} bps per side (stock + hedge). Trades enter at the close of day +1.", "",
          C.md_table(res[cols], index=False), "", "## Net Sharpe by horizon (decay vs tradability)",
          C.md_table(decay), "",
          "Read B-balanced against A-balanced (the incremental value of language) and against long "
          "(the universe's post-call drift, large for these three names over 2018-2023)."]
    (out_dir / "summary.md").write_text("\n".join(md))
    pd.set_option("display.width", 220)
    print(res[cols].round(3).to_string(index=False))
    log.info("outputs in %s", out_dir)


def stage_holdout(args, cost_bps: float) -> None:
    if not C.FROZEN_SPEC_PATH.exists():
        raise SystemExit("outputs/frozen_spec.json not found: run 04_oos.py --freeze first")
    spec = json.loads(C.FROZEN_SPEC_PATH.read_text())
    out_dir = C.stage_dir("holdout")
    log_path = out_dir / "holdout_log.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    if history and not args.force:
        raise SystemExit(f"holdout already evaluated {len(history)} time(s) (see {log_path}); "
                         "pass --force only if you accept that the holdout is no longer untouched")
    sha = C.file_sha256(C.FEATURES_PATH)
    if sha != spec["features_sha256"] and not args.force:
        raise SystemExit("data/features.csv changed since the spec was frozen; re-freeze or pass --force")

    C.SUE_VARIANT = spec["sue_variant"]
    ev = C.load_events("dev+holdout", spec["sample"])
    t = spec["target"]
    test = ev[(ev["year"] >= C.HOLDOUT_START) & ev[t].notna()]
    dev = ev[ev["year"] <= C.DEV_END]
    train = C.purged_train(dev, t, test["day0"].min())
    train = train[train[t].notna()]
    firms = sorted(ev["firm"].astype(str).unique())
    log.info("holdout: target %s, model %s | train n=%d (purged, %d-%d) | test n=%d (%s to %s)", t, spec["model"],
             len(train), train["year"].min(), train["year"].max(), len(test), test["day0"].min().date(),
             test["day0"].max().date())

    preds = []
    for name, feats in (("A", spec["features_A"]), ("B", spec["features_B"])):
        Xtr, Xte = oos.design(train, test, feats, firms)
        p, coef, info = oos.fit_predict(spec["model"], Xtr, train[t].to_numpy(), Xte)
        preds.append(test[["transcriptid", "ticker", "day0", "year", "event_q", t]].rename(columns={t: "y"})
                     .assign(target=t, model=spec["model"], spec=name, pred=p, bench=train[t].mean()))
        if coef is not None:
            coef.rename("value").to_csv(out_dir / f"coefficients_{name}.csv")
    preds = pd.concat(preds, ignore_index=True)
    met = pd.DataFrame([dict(spec=s, **oos.metrics(g)) for s, g in preds.groupby("spec")])
    A, B = (preds[preds.spec == s].set_index("transcriptid") for s in ("A", "B"))
    B = B.loc[A.index]
    cw_t, cw_p = C.clark_west(A["y"], A["pred"], B["pred"])
    inc = dict(d_ic=C.spearman(B["pred"], A["y"]) - C.spearman(A["pred"], A["y"]),
               d_oos_r2=C.oos_r2(A["y"], B["pred"], A["bench"]) - C.oos_r2(A["y"], A["pred"], A["bench"]),
               cw_t=cw_t, cw_p=cw_p, n=len(A))
    bt, curves = (evaluate_block(preds, cost_bps) if t in TRADE_TARGETS else (pd.DataFrame(), {}))

    preds.to_csv(out_dir / "holdout_predictions.csv", index=False)
    met.to_csv(out_dir / "holdout_metrics.csv", index=False)
    if len(bt):
        bt.to_csv(out_dir / "holdout_backtest.csv", index=False)
        plot_curves(curves, out_dir / "cumulative_net_holdout.png", f"Holdout {C.HOLDOUT_START}, net")
    record = dict(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), spec_frozen_at=spec["frozen_at"],
                  features_sha256=sha, target=t, model=spec["model"], n_test=len(test), increment=inc,
                  metrics=met.to_dict(orient="records"), forced=bool(args.force))
    history.append(record)
    log_path.write_text(json.dumps(history, indent=2, default=float))

    md = [f"# Final holdout ({C.HOLDOUT_START}): frozen spec `{t}` / `{spec['model']}`", "",
          f"Spec frozen {spec['frozen_at']} ({spec['verdict']}). Run #{len(history)}"
          f"{' (FORCED: holdout no longer untouched)' if len(history) > 1 else ''}.", "",
          f"Train: {len(train)} purged dev events; test: {len(test)} calls. With N = {len(test)}, read these as "
          "a sign check, not as inference.", "",
          "## Statistical", C.md_table(met, index=False), "",
          C.md_table(pd.Series(inc, name="value").to_frame()), ""]
    if len(bt):
        md += ["## Economic (net of costs)", C.md_table(bt[["strategy", "n_trades", "hit_rate", "mean_net_bps",
                                                            "sharpe_ann", "total_net"]], index=False)]
    (out_dir / "summary.md").write_text("\n".join(md))
    print(met.round(4).to_string(index=False))
    print(pd.Series(inc).round(4).to_string())
    if len(bt):
        print(bt[["strategy", "n_trades", "hit_rate", "mean_net_bps", "total_net"]].round(3).to_string(index=False))
    log.info("outputs in %s", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--tag", default="base", help="04_oos.py output folder to backtest (dev stage)")
    ap.add_argument("--cost-stock", type=float, default=C.COST_BPS["stock"])
    ap.add_argument("--cost-hedge", type=float, default=C.COST_BPS["hedge"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cost = args.cost_stock + args.cost_hedge
    if args.stage == "dev":
        stage_dev(args, cost)
    else:
        stage_holdout(args, cost)


if __name__ == "__main__":
    main()
