"""Score the LLM judge and the OOF feature scores against labeller B (90 pairs shared with A).

B is a second LLM labeller (docs/labeling_handoff_teammate.md), so this measures agreement with another
model rater, not with a human. A's own label is scored as a reference (a binary score, so its AUC is
(TPR + TNR) / 2).

    .venv/bin/python scripts/f3/eval_vs_b.py [--model claude-haiku-5-5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402
import llm_judge as J  # noqa: E402

B_LABELS = G.ROOT / "data" / "qa_labels_b.csv"


def main(model):
    d = pd.read_pickle(G.CACHE / "labelled.pkl")[["pair_id", "transcriptid", *G.TARGETS]]
    b = pd.read_csv(B_LABELS)[["pair_id", *G.TARGETS]].rename(columns={t: f"{t}_b" for t in G.TARGETS})
    oof = pd.read_csv(G.OUT / "oof_scores.csv")
    m = d.merge(b, on="pair_id").merge(J.load(model), on="pair_id", how="left").merge(oof, on="pair_id", how="left")
    scores = {
        "evasive": {"A label": m["evasive"], "J judge p_evasive": m["p_evasive"],
                    "E1 rank-mean(C1, T3)": m["E1 rank-mean(C1, T3)|evasive"],
                    "C1 rich pair features": m["C1 rich pair features|evasive"]},
        "non_answer": {"A label": m["non_answer"], "J judge p_non_answer": m["p_non_answer"],
                       "C1 rich pair features": m["C1 rich pair features|non_answer"],
                       "C1+C2 combined": m["C1+C2 combined|non_answer"]},
    }
    rows = []
    for t, cs in scores.items():
        for truth in ("B", "A"):
            y = m[f"{t}_b"] if truth == "B" else m[t]
            for name, s in cs.items():
                if name == "A label" and truth == "A":
                    continue
                r = G.auc_ci(y, s, m["transcriptid"])
                rows.append(dict(target=t, truth=truth, score=name, **{k: r[k] for k in
                                 ("auc", "auc_lo", "auc_hi", "ap", "n", "n_pos")}))
    res = pd.DataFrame(rows)
    res.to_csv(G.OUT / f"vs_labeller_b_{model}.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=J.DEFAULT_MODEL)
    main(ap.parse_args().model)
