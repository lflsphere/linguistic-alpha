"""Evaluate every F3 candidate on A's 414 build-period labels (docs/handoff_f3_relevance.md).

Writes outputs/validation/f3_relevance/:
  results.csv          one row per candidate x target: AUC, AP, call-bootstrap 95% CIs, mean/min/max AUC
                       over 10 fold assignments (learned models only)
  raw_scores.csv       every unfitted single score (sign fixed a priori: high = positive)
  oof_scores.csv       pair_id + the OOF score of each candidate (no text)
  by_year.csv / by_sector.csv   the selected candidates, as a sanity check only (cells of 30-39 pairs)
  coefficients.csv     full-fit standardised coefficients of the combined model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402
import pair_features as P  # noqa: E402
import text_models as T  # noqa: E402

LEAN = ["nli_decline_mean", "nli_decline_max", "nli_number_mean", "nli_number_max"]   # 2 NLI hypotheses only
BASE = ["REL", "REL_JAC", "log_a_words", "log_q_words", "len_ratio", "HEDGE_A", "NONANS_RULE"]   # PAIR_CLF_FEATS
SEEDS = range(10)


def load():
    src = G.CACHE / "labelled.pkl"
    d = pd.read_pickle(src) if src.exists() else G.load_labelled()
    d = pd.concat([d, P.frame(d)], axis=1)
    cs = G.CACHE / "cross_scores_labelled.csv"
    cross = []
    if cs.exists():
        c = pd.read_csv(cs, dtype={"pair_id": str})
        cross = [x for x in c.columns if x != "pair_id"]
        d = d.merge(c, on="pair_id", how="left", validate="1:1")
    return d, cross


def raw_table(d, cross):
    """Unfitted scores. Relevance-type scores are negated (low relevance = evasive)."""
    neg = {"REL", "REL_JAC", "q_cover", "q_cover_head", "q_bigram_cover", "a_spec_p100", "a_num_p100",
           "a_any_number", "a_direct", "log_a_words", "log_ratio"}
    neg |= {c for c in cross if c.startswith(("ce_", "nli_number", "nli_direct", "nliqa_answers"))}
    cols = ["EVADE_CLF", "NONANS_CLF", "NONANS_RULE"] + [c for c in BASE if c != "NONANS_RULE"] + P.RICH + cross
    rows = []
    for c in dict.fromkeys(cols):
        s = d[c].astype(float) * (-1 if c in neg else 1)
        for t in G.TARGETS:
            r = G.auc_ci(d[t], s, d["transcriptid"], n_boot=500)
            rows.append(dict(score=("-" if c in neg else "") + c, target=t, **r))
    return pd.DataFrame(rows)


COMBOS = {   # fixed before looking at results of the combination: equal-weight mean of OOF ranks, nothing fitted
    "E1 rank-mean(C1, T3)": ["C1 rich pair features", "T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit"],
    "E2 rank-mean(C1+C2, T3)": ["C1+C2 combined", "T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit"],
    "E3 rank-mean(C1, T1, T3)": ["C1 rich pair features", "T1 TF-IDF answer logit",
                                 "T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit"],
    "E4 rank-mean(L1, T3) (post hoc)": ["L1 rich + NLI decline/number (post hoc)",
                                        "T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit"],
}


def _rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def learned(d, cross):
    E = None
    try:
        E = T.emb_features(d["pair_id"].tolist())
    except Exception as e:                      # noqa: BLE001 - cache missing in a test checkout
        print("embedding cache:", e)
    sets = {
        "C0 base logit (PAIR_CLF_FEATS, inner-CV C)": BASE,
        "C1 rich pair features": BASE + P.RICH,
        "C2 cross-encoder + NLI scores": BASE + cross if cross else None,
        "C1+C2 combined": BASE + P.RICH + cross if cross else None,
        # added after the second run (post hoc): the two NLI hypotheses that carried C2, to cut its cost ~4x
        "L1 rich + NLI decline/number (post hoc)": BASE + P.RICH + LEAN if cross else None,
    }
    out, oof_cols, chosen = [], {}, {}
    g = d["transcriptid"].to_numpy()
    for t in G.TARGETS:
        y = d[t].astype(int).to_numpy()
        allo = {}
        for name, cols in sets.items():
            if cols:
                X = d[cols].astype(float).to_numpy()
                allo[name] = {s: G.oof_model(X, y, g, seed=s) for s in SEEDS}
        txt = {"T1 TF-IDF answer logit": d["a_anon"], "T2 TF-IDF question||answer logit":
               d["q_anon"] + " || " + d["a_anon"]}
        for name, texts in txt.items():
            allo[name] = {s: T.tfidf_oof(texts, y, g, seed=s) for s in SEEDS}
        if E is not None:
            allo["T3 MiniLM vectors [Q,A,Q*A,|Q-A|] logit"] = {s: T.emb_oof(E, y, g, seed=s) for s in SEEDS}
        for name, parts in COMBOS.items():
            if all(p in allo for p in parts):
                allo[name] = {s: (np.mean([_rank(allo[p][s][0]) for p in parts], axis=0), None) for s in SEEDS}
        for name, oofs in allo.items():
            out.append(_row(name, t, y, g, oofs))
            oof_cols[f"{name}|{t}"] = oofs[0][0]
            chosen[f"{name}|{t}"] = oofs[0][1]
    return pd.DataFrame(out), pd.DataFrame(oof_cols, index=d["pair_id"]), chosen


def _row(name, t, y, g, oofs):
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score(y, o) for o, _ in oofs.values()]
    first = next(iter(oofs.values()))[0]
    r = G.auc_ci(y, first, g)
    return dict(candidate=name, target=t, **r, auc_mean10=np.mean(aucs), auc_min10=np.min(aucs),
                auc_max10=np.max(aucs))


def main():
    G.OUT.mkdir(parents=True, exist_ok=True)
    d, cross = load()
    raw = raw_table(d, cross)
    raw.to_csv(G.OUT / "raw_scores.csv", index=False)
    res, oof, chosen = learned(d, cross)
    # stored agent C scores (already OOF) as the reference rows
    ref = []
    for c, t in (("EVADE_CLF", "evasive"), ("NONANS_CLF", "non_answer"), ("NONANS_CLF", "evasive"),
                 ("EVADE_CLF", "non_answer")):
        ref.append(dict(candidate=f"ref {c} (agent C OOF)", target=t, **G.auc_ci(d[t], d[c], d["transcriptid"])))
    res = pd.concat([pd.DataFrame(ref), res], ignore_index=True)
    res.to_csv(G.OUT / "results.csv", index=False)
    oof.reset_index().to_csv(G.OUT / "oof_scores.csv", index=False)
    pd.Series({k: str(v) for k, v in chosen.items()}).to_csv(G.OUT / "chosen_C.csv", header=["C_by_outer_fold"])
    # sanity cells for the best learned candidate per target
    cells = []
    for t in G.TARGETS:
        best = res[(res.target == t) & ~res.candidate.str.startswith("ref")].sort_values("auc_mean10").iloc[-1]
        s = oof[f"{best.candidate}|{t}"].to_numpy()
        for key in ("year", "gsector"):
            for v, idx in d.groupby(key).groups.items():
                yy = d.loc[idx, t]
                if yy.nunique() < 2:
                    cells.append(dict(by=key, cell=v, target=t, candidate=best.candidate, n=len(idx),
                                      n_pos=int(yy.sum()), auc=np.nan))
                    continue
                from sklearn.metrics import roc_auc_score
                cells.append(dict(by=key, cell=v, target=t, candidate=best.candidate, n=len(idx),
                                  n_pos=int(yy.sum()), auc=roc_auc_score(yy, s[d.index.get_indexer(idx)])))
    cells = pd.DataFrame(cells)
    cells[cells.by == "year"].to_csv(G.OUT / "by_year.csv", index=False)
    cells[cells.by == "gsector"].to_csv(G.OUT / "by_sector.csv", index=False)
    coefs = []
    for t in G.TARGETS:
        for name, cols in (("C1", BASE + P.RICH), ("L1", BASE + P.RICH + LEAN)):
            if not set(cols) <= set(d.columns):
                continue
            m = G._logit(1.0).fit(d[cols].astype(float), d[t].astype(int))
            coefs += [dict(model=name, target=t, feature=c, coef_std_C1=v) for c, v in zip(cols, m[-1].coef_[0])]
    pd.DataFrame(coefs).to_csv(G.OUT / "coefficients.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["candidate", "target", "auc", "auc_lo", "auc_hi", "ap", "auc_mean10", "auc_min10", "auc_max10"]]
          .round(3).to_string(index=False))
    top = raw.sort_values("auc", ascending=False).groupby("target").head(8)
    print(top[["score", "target", "auc", "auc_lo", "auc_hi", "ap"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
