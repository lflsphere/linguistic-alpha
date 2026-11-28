"""Shared helpers for the F3 relevance search (docs/handoff_f3_relevance.md, agent G).

- load_labelled(): A's 414 build-period labels joined to the pair features of data/qa_pairs_full.csv,
  with the texts anonymised as in scripts/full_source.py (own firm name and calendar years masked).
- oof_*(): out-of-fold scores, GroupKFold by call (transcriptid); any hyper-parameter is chosen in inner
  GroupKFold folds of the training part only.
- auc_ci(): AUC and average precision with a bootstrap CI that resamples calls.
Nothing here reads holdout calls: rows after split.build_end are refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

LABELS = ROOT / "data" / "qa_labels.csv"
PAIRS_FULL = ROOT / "data" / "qa_pairs_full.csv"
CALLS = ROOT / "data" / "interim" / "transcript_calls.parquet"
EMB_CACHE = ROOT / "data" / "interim" / "emb_cache"
CACHE = ROOT / "data" / "interim" / "f3_cache"
OUT = ROOT / "outputs" / "validation" / "f3_relevance"
BUILD_END = pd.Timestamp("2023-12-31")
TARGETS = ("evasive", "non_answer")
SEED = 42
N_SPLITS = 5
N_BOOT = 2000


def _fs():
    import full_source as F            # anonymizer, the 01 module (norm, NONANSWER_RE)
    return F


def anonymise(texts: pd.Series, names: pd.Series) -> list[str]:
    F = _fs()
    B = F._builder()
    cache = {}
    out = []
    for t, n in zip(texts, names):
        if n not in cache:
            cache[n] = F.anonymizer(n)
        out.append(cache[n](B.norm(t or "")))
    return out


def load_labelled(labels=LABELS, pairs_full=PAIRS_FULL, calls=CALLS) -> pd.DataFrame:
    lab = pd.read_csv(labels, dtype={"pair_id": str, "gvkey": str})
    lab = lab[pd.to_datetime(lab["call_date"]) <= BUILD_END].copy()
    feats = ["pair_id", "q_words", "a_words", "HEDGE_A", "NONANS_RULE", "REL_JAC", "REL",
             "log_a_words", "log_q_words", "len_ratio", "EVADE_CLF", "NONANS_CLF"]
    pf = pd.read_csv(pairs_full, usecols=feats, dtype={"pair_id": str})
    d = lab.merge(pf, on="pair_id", how="left", validate="1:1")
    names = pd.read_parquet(calls, columns=["transcriptid", "companyname"]).drop_duplicates("transcriptid")
    d = d.merge(names, on="transcriptid", how="left")
    d["q_anon"] = anonymise(d["question"], d["companyname"])
    d["a_anon"] = anonymise(d["answer"], d["companyname"])
    d["year"] = pd.to_datetime(d["call_date"]).dt.year
    return d.reset_index(drop=True)


# ============================================================================ evaluation
def folds(groups, n_splits=N_SPLITS, seed=SEED):
    from sklearn.model_selection import GroupKFold
    g = np.asarray(groups)
    k = min(n_splits, len(np.unique(g)))
    return list(GroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(g)), groups=g))


def auc_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    """AUC, AP and call-bootstrap 95% CIs. `s` high = positive."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s, g = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    ok = ~np.isnan(s)
    y, s, g = y[ok], s[ok], g[ok]
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    A, P = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        if y[idx].min() == y[idx].max():
            continue
        A.append(roc_auc_score(y[idx], s[idx]))
        P.append(average_precision_score(y[idx], s[idx]))
    lo, hi = np.percentile(A, [2.5, 97.5])
    plo, phi = np.percentile(P, [2.5, 97.5])
    return dict(auc=auc, auc_lo=lo, auc_hi=hi, ap=ap, ap_lo=plo, ap_hi=phi, base=y.mean(), n=len(y),
                n_pos=int(y.sum()))


def _logit(C):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


def oof_model(X, y, groups, make=_logit, grid=(0.01, 0.03, 0.1, 0.3, 1.0), seed=SEED):
    """Out-of-fold P(y=1). For each outer fold, the parameter in `grid` is picked by inner GroupKFold AUC
    on the training part only. Returns (oof, chosen params)."""
    from sklearn.metrics import roc_auc_score
    X = X if isinstance(X, np.ndarray) else np.asarray(X, float)
    y, g = np.asarray(y), np.asarray(groups)
    oof, chosen = np.full(len(y), np.nan), []
    for tr, te in folds(g, seed=seed):
        best, best_auc = grid[0], -1
        if len(grid) > 1:
            for c in grid:
                inner = np.full(len(tr), np.nan)
                for itr, ite in folds(g[tr], seed=seed + 1):
                    m = make(c).fit(X[tr][itr], y[tr][itr])
                    inner[ite] = m.predict_proba(X[tr][ite])[:, 1]
                a = roc_auc_score(y[tr], inner)
                if a > best_auc:
                    best, best_auc = c, a
        m = make(best).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        chosen.append(best)
    return oof, chosen


def repeat_auc(fn, y, seeds=range(10)):
    """AUC of fn(seed) -> oof over several fold assignments (spread from the split alone)."""
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(y, fn(s)) for s in seeds]
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))
