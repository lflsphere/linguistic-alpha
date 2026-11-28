"""Candidate 2: cross-encoder and NLI relevance scores, cached in data/interim/f3_cache/.

Per pair (anonymised text, as for the MiniLM cosine), the answer is cut into word windows and each window is
scored against the question; the pair gets the max / mean / first-window score.
- ce_marco:  cross-encoder/ms-marco-MiniLM-L6-v2 relevance logit (query = question, passage = answer window)
- ce_bge:    BAAI/bge-reranker-base relevance logit
- ce_qnli:   cross-encoder/qnli-electra-base P(the window answers the question)
- nli_*:     cross-encoder/nli-deberta-v3-base P(entailment), premise = answer window, fixed hypotheses
             (decline / too early / defer / gives a number / answers directly)
- nliqa_*:   same model, premise = "Question: ... Answer: ..." (first 512 tokens), hypotheses on the pair
Nothing is fitted. Throughput (windows/s, pairs/s) is written to f3_cache/throughput.json for the cost estimate.

    .venv/bin/python scripts/f3/cross_scores.py            # the 414 labelled pairs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402

ANON_VERSION = "own-firm-name+year-v1"
RANK_WORDS, NLI_WORDS = 200, 80
RANKERS = {"ce_marco": "cross-encoder/ms-marco-MiniLM-L6-v2",
           "ce_bge": "BAAI/bge-reranker-base",
           "ce_qnli": "cross-encoder/qnli-electra-base"}
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
HYP_A = {"decline": "The speaker declines to give the requested information.",
         "early": "The speaker says it is too early to tell.",
         "defer": "The speaker will provide this information later.",
         "number": "The speaker gives a specific number.",
         "direct": "The speaker answers the question directly."}
HYP_QA = {"evades": "The answer avoids the question.",
          "answers": "The answer gives the information the question asked for.",
          "refuses": "The answer refuses to answer the question."}


def windows(text: str, size: int) -> list[str]:
    w = (text or "").split()
    if not w:
        return [""]
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(_device()).eval()
    return tok, m, torch


def _score(tok, m, torch, a, b, batch=32, max_len=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(a), batch):
            enc = tok(a[i:i + batch], b[i:i + batch], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(m.device)
            out.append(m(**enc).logits.float().cpu().numpy())
    return np.vstack(out)


def _agg(pid_of_window, vals, prefix):
    s = pd.DataFrame({"pair_id": pid_of_window, "v": vals})
    g = s.groupby("pair_id", sort=False)["v"]
    return pd.DataFrame({f"{prefix}_max": g.max(), f"{prefix}_mean": g.mean(), f"{prefix}_first": g.first(),
                         f"{prefix}_min": g.min()})


def score_rankers(d, which=RANKERS):
    res, thr = [], {}
    for key, name in which.items():
        tok, m, torch = _load(name)
        pid, qs, ws = [], [], []
        for p, q, a in zip(d["pair_id"], d["q_anon"], d["a_anon"]):
            for w in windows(a, RANK_WORDS):
                pid.append(p), qs.append(q), ws.append(w)
        t0 = time.time()
        lg = _score(tok, m, torch, qs, ws)
        dt = time.time() - t0
        if lg.shape[1] == 1:
            v = lg[:, 0]
        else:                                   # qnli-electra: 2 logits? take P(entailment) via softmax
            e = np.exp(lg - lg.max(1, keepdims=True))
            v = (e / e.sum(1, keepdims=True))[:, 0]
        if key == "ce_qnli" and lg.shape[1] == 1:
            v = 1 / (1 + np.exp(-v))
        res.append(_agg(pid, v, key))
        thr[key] = dict(model=name, windows=len(ws), pairs=len(d), seconds=dt, windows_per_pair=len(ws) / len(d))
        print(f"{key}: {len(ws)} windows in {dt:.1f}s", flush=True)
    return pd.concat(res, axis=1), thr


def score_nli(d):
    tok, m, torch = _load(NLI_MODEL)
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    ent = lab["entailment"]
    res, thr = [], {}
    pid, ws = [], []
    for p, a in zip(d["pair_id"], d["a_anon"]):
        for w in windows(a, NLI_WORDS):
            pid.append(p), ws.append(w)
    t0 = time.time()
    for h, text in HYP_A.items():
        lg = _score(tok, m, torch, ws, [text] * len(ws))
        e = np.exp(lg - lg.max(1, keepdims=True))
        res.append(_agg(pid, (e / e.sum(1, keepdims=True))[:, ent], f"nli_{h}"))
    thr["nli_answer"] = dict(model=NLI_MODEL, windows=len(ws), pairs=len(d), hypotheses=len(HYP_A),
                             seconds=time.time() - t0, windows_per_pair=len(ws) / len(d))
    prem = [f"Question: {q} Answer: {a}" for q, a in zip(d["q_anon"], d["a_anon"])]
    t0 = time.time()
    for h, text in HYP_QA.items():
        lg = _score(tok, m, torch, prem, [text] * len(prem), batch=16)
        e = np.exp(lg - lg.max(1, keepdims=True))
        res.append(pd.DataFrame({f"nliqa_{h}": (e / e.sum(1, keepdims=True))[:, ent]},
                                index=d["pair_id"].to_numpy()))
    thr["nli_pair"] = dict(model=NLI_MODEL, windows=len(prem), pairs=len(d), hypotheses=len(HYP_QA),
                           seconds=time.time() - t0, windows_per_pair=1.0)
    print("nli done", flush=True)
    return pd.concat(res, axis=1), thr


def main():
    G.CACHE.mkdir(parents=True, exist_ok=True)
    src = G.CACHE / "labelled.pkl"
    d = pd.read_pickle(src) if src.exists() else G.load_labelled()
    a, t1 = score_rankers(d)
    b, t2 = score_nli(d)
    out = a.join(b)
    out.index.name = "pair_id"
    out.reset_index().to_csv(G.CACHE / "cross_scores_labelled.csv", index=False)
    meta = dict(anonymization=ANON_VERSION, rank_words=RANK_WORDS, nli_words=NLI_WORDS, device=_device(),
                hyp_answer=HYP_A, hyp_pair=HYP_QA, throughput={**t1, **t2})
    (G.CACHE / "throughput.json").write_text(json.dumps(meta, indent=2))
    print("wrote", G.CACHE / "cross_scores_labelled.csv")


if __name__ == "__main__":
    main()
