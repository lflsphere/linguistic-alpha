"""Score the cached LLM-judge answers on the 414 labelled pairs (no fitting, so no CV needed) and combine
them with the existing out-of-fold scores by equal-weight rank mean. Combinations are post hoc.

    .venv/bin/python scripts/f3/eval_judge.py [--model claude-haiku-5-5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402
import llm_judge as J  # noqa: E402


def rank_mean(*cols):
    return pd.concat([c.rank(pct=True) for c in cols], axis=1).mean(axis=1)


def main(model):
    d = pd.read_pickle(G.CACHE / "labelled.pkl")[["pair_id", "transcriptid", *G.TARGETS]]
    j = J.load(model)
    oof = pd.read_csv(G.OUT / "oof_scores.csv")
    m = d.merge(j, on="pair_id", how="left").merge(oof, on="pair_id", how="left")
    n_fail = int(m["p_evasive"].isna().sum())
    col = lambda name, t: m[f"{name}|{t}"]  # noqa: E731
    cands = {
        "evasive": {
            "J judge p_evasive": m["p_evasive"],
            "J judge p_non_answer": m["p_non_answer"],
            "rank-mean(J, E1)": rank_mean(m["p_evasive"], col("E1 rank-mean(C1, T3)", "evasive")),
            "rank-mean(J, C1, T3)": rank_mean(m["p_evasive"], col("C1 rich pair features", "evasive"),
                                              col("T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit", "evasive")),
        },
        "non_answer": {
            "J judge p_non_answer": m["p_non_answer"],
            "J judge p_evasive": m["p_evasive"],
            "rank-mean(J, C1)": rank_mean(m["p_non_answer"], col("C1 rich pair features", "non_answer")),
            "rank-mean(J, C1+C2)": rank_mean(m["p_non_answer"], col("C1+C2 combined", "non_answer")),
        },
    }
    rows = []
    for t, cs in cands.items():
        for name, s in cs.items():
            r = G.auc_ci(m[t], s, m["transcriptid"])
            rows.append(dict(model=model, candidate=name, target=t, **{k: r[k] for k in
                             ("auc", "auc_lo", "auc_hi", "ap", "n", "n_pos")}))
    res = pd.DataFrame(rows)
    res.to_csv(G.OUT / f"llm_judge_results_{model}.csv", index=False)
    print(res.round(3).to_string(index=False))
    ok = j[j["p_evasive"].notna()]
    per = dict(in_tok=ok["in_tok"].mean(), out_tok=ok["out_tok"].mean(), cost=ok["cost_usd"].sum())
    pin, pout = J.PRICES[model]
    full = J.N_ALL_PAIRS * (per["in_tok"] * pin + per["out_tok"] * pout) / 1e6
    print(f"\nfailed/unparsed: {n_fail}; mean in {per['in_tok']:.0f} tok, out {per['out_tok']:.0f} tok; "
          f"414-pair cost ${per['cost']:.3f}; list-price full sample ~${full:,.0f} (${full/2:,.0f} batch)")
    # positive-rate calibration check
    for t, p in (("evasive", "p_evasive"), ("non_answer", "p_non_answer")):
        print(f"{t}: base {m[t].mean():.3f}, judge mean p {m[p].mean():.3f}, share p>=0.5 {(m[p] >= .5).mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=J.DEFAULT_MODEL)
    main(ap.parse_args().model)
