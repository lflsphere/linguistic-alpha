"""Supervised text models on the labelled pairs (fitted only inside the outer training folds).

- emb_features(): the cached MiniLM vectors (data/interim/emb_cache/, read-only) of the labelled pairs:
  [Q, A, Q*A, |Q-A|] (1,536 columns) for an L2 logistic regression.
- tfidf_oof(): word 1-2 gram TF-IDF of the answer (and of "question || answer") with an L2 logistic
  regression; the vocabulary and IDF are fitted on the training folds only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402


def emb_features(pair_ids, folder=G.EMB_CACHE, cache=G.CACHE / "emb_labelled.npz"):
    if cache is not None and Path(cache).exists():
        with np.load(cache, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
            if set(pair_ids) <= set(ids):
                pos = {p: i for i, p in enumerate(ids)}
                return z["X"][[pos[p] for p in pair_ids]]
    want = set(pair_ids)
    Q, A = {}, {}
    for f in sorted(Path(folder).glob("pairs_*.npz")) or sorted(Path(folder).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            if "Q" not in z.files:
                continue
            ids = z["ids"].astype(str)
            hit = np.flatnonzero(np.isin(ids, list(want)))
            for i in hit:
                Q[ids[i]], A[ids[i]] = z["Q"][i].astype(np.float32), z["A"][i].astype(np.float32)
    missing = [p for p in pair_ids if p not in Q]
    if missing:
        raise KeyError(f"{len(missing)} labelled pairs not in the embedding cache, e.g. {missing[:3]}")
    q = np.vstack([Q[p] for p in pair_ids])
    a = np.vstack([A[p] for p in pair_ids])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    X = np.hstack([q, a, q * a, np.abs(q - a)])
    if cache is not None:
        np.savez(cache, ids=np.asarray(pair_ids, dtype=str), X=X)
    return X


def _tfidf_logit(C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def tfidf_oof(texts, y, groups, seed=G.SEED, grid=(0.3, 1.0, 3.0, 10.0)):
    texts = np.asarray(texts, dtype=object)
    return G.oof_model(texts, y, groups, make=_tfidf_logit, grid=grid, seed=seed)


def _emb_logit(C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def emb_oof(X, y, groups, seed=G.SEED, grid=(0.0003, 0.001, 0.003, 0.01)):
    return G.oof_model(X, y, groups, make=_emb_logit, grid=grid, seed=seed)
