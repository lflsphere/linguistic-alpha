#!/usr/bin/env python
"""
01: build the event-level research table from data/panel.csv (framework Sections 2-4).

Outputs
  data/features.csv        one row per call: ids, timing, targets, Model A controls, linguistic features
  data/qa_pairs.csv        one row per analyst-question / management-answer pair (texts kept for labeling)
  data/features_meta.json  audit metadata: lexicon source, embedding backend, fit corpus, truncation stats

Options
  --embed {auto,st,tfidf}  st = sentence-transformers (needs a local cache or Hugging Face access);
                           tfidf = TF-IDF / LSA fitted on development-period text only (offline fallback)
  --finbert                add FinBERT tone (TONE_P_FB, TONE_Q_FB) as an alternative tone measurement

Lexicons: drop the Loughran-McDonald Master Dictionary CSV (any file matching *MasterDictionary*.csv) in
data/ to use the official word lists. Without it, compact embedded seed lists approximating the LM
categories are used and the run is flagged in features_meta.json.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import common as C

log = C.get_logger("build")

# ============================================================================ lexicons
SEED_LEXICON = {
    "positive": """able abundance accomplish accomplished accomplishment accomplishments achieve achieved
        achievement achievements achieves achieving advancement advancements advances advancing advantage
        advantageous advantages attain attained attractive beneficial benefit benefited benefiting best better
        boom boost boosted breakthrough breakthroughs collaborate collaboration confident constructive creative
        delight delighted efficiencies efficiency efficient empower enable enabled enables enabling encouraged
        encouraging enhance enhanced enhancement enhancements enhances enjoy enjoyed enthusiasm enthusiastic
        excellence excellent exceptional excited exciting exclusive favorable favorably gain gained gaining
        gains good great greater greatest happy highest ideal impressive improve improved improvement
        improvements improves improving incredible innovate innovation innovations innovative leadership
        leading lucrative opportunities opportunity optimistic outperform outperformed outstanding perfect
        pleased pleasure popular positive positively profitability profitable progress progressing
        prosperity rebound rewarding robust satisfied strength strengthen strengthened strengths strong
        stronger strongest succeed succeeded success successes successful successfully superior surpass
        surpassed tremendous upturn valuable winner winning""",
    "negative": """abandon adverse adversely bad breach burden challenge challenged challenges challenging
        closure concern concerned concerns crisis critical damage damaged decline declined declines declining
        decrease decreased decreases decreasing deficit delay delayed delays deteriorate deteriorated
        deterioration difficult difficulties difficulty disappoint disappointed disappointing disruption
        disruptions downturn erosion error fail failed failure failures headwind headwinds impair impaired
        impairment inability ineffective layoff layoffs litigation lose losing loss losses lost negative
        negatively obsolete penalty poor problem problems recall recession restructuring severe shortage
        shortages shortfall slow slowdown slowed slower slowing slump suffer termination threat tough unable
        unfavorable unfortunately weak weaken weakened weaker weakness weaknesses worse worsen worst writedown""",
    "uncertainty": """anticipate anticipated anticipates anticipating approximate approximately assume assumed
        assumes assuming assumption assumptions believe believed believes contingency contingent doubt
        doubtful exposure exposures fluctuate fluctuated fluctuates fluctuating fluctuation fluctuations
        imprecise indefinite likelihood pending predict predicted predicting prediction predictions
        preliminary presumably probable probably random risk risks risky roughly seem seemed seems speculate
        speculation speculative sudden suddenly tentative turbulence uncertainties uncertainty unclear
        unexpected unforeseen unknown unknowns unpredictable unpredictability unproven unusual variability
        variable variance variation variations varies vary varying volatile volatility""",
    "weak_modal": """almost apparently appeared appearing appears conceivable could depend depended depending
        depends may maybe might nearly occasionally perhaps possible possibly seldom sometimes somewhat
        suggest suggests uncertain uncertainly""",
}
NEGATORS = {"no", "not", "none", "neither", "never", "nobody", "nor", "without"}
HEDGE_PHRASES = ["i think", "we think", "i believe", "we believe", "i guess", "sort of", "kind of",
                 "to some extent", "to some degree", "more or less", "it depends", "hard to say",
                 "difficult to say", "hard to predict", "difficult to predict", "too early to tell",
                 "not sure", "we'll see", "we will see", "i would say", "i'd say"]
HEDGE_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, HEDGE_PHRASES)) + r")\b")

# Heuristic non-answer rules (refuse / unable / defer), in the spirit of Gow, Larcker and Zakolyukina (2021)
_VERBS = (r"(?:break (?:that|it|this|those) out|disclose|comment|provide|give|get into|go into|talk about|"
          r"speculate|share|guide|forecast|quantify|split (?:that|it) out|detail|parse)")
NONANSWER_RE = re.compile("|".join([
    r"\b(?:we|i)\s+(?:don't|do not|won't|will not|wouldn't|would not|can't|cannot|can not)\s+(?:\w+\s+){0,2}" + _VERBS,
    r"\bnot (?:going|gonna|able|prepared|in a position) to (?:\w+\s+){0,2}" + _VERBS,
    r"\b(?:too early|premature) to\b",
    r"\bno comment\b",
    r"\bwe(?:'re| are) not (?:providing|giving|disclosing|guiding|breaking)\b",
    r"\b(?:don't|do not) have (?:that|the|those) (?:number|numbers|data|figure|figures|breakdown|details?)\b",
    r"\b(?:take|follow up on) (?:that|it) offline\b",
    r"\bas a (?:matter of )?(?:policy|practice),? we\b",
    r"\bwe(?:'ve| have) never (?:disclosed|broken out|provided)\b",
    r"\bnot something (?:that )?we(?:'re| are)? ?(?:going to )?(?:disclose|break out|discuss|talk about|comment on)",
]))
BOILER_RE = re.compile(
    r"forward-looking statements?|safe harbor|prior written consent|reproduced or transcribed|webcast|"
    r"\breplay\b|reconciliation of (?:these|the|our) non-gaap|subject to (?:a number of )?(?:significant )?"
    r"risks and uncertainties|most recent (?:annual report on )?form 10-k", re.I)
NOISE_RE = re.compile(r"\[[^\]]{0,40}\]")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
EXCEL_CELL_LIMIT = 32767


def load_lexicons() -> tuple[dict, str]:
    files = sorted(C.DATA_DIR.glob("*MasterDictionary*.csv"))
    if files:
        lm = pd.read_csv(files[-1])
        cols = {c.lower(): c for c in lm.columns}
        words = lm[cols["word"]].astype(str).str.lower()

        def cat(name):
            return set(words[lm[cols[name]].fillna(0).astype(float) > 0])

        lex = dict(positive=cat("positive"), negative=cat("negative"),
                   uncertainty=cat("uncertainty"), weak_modal=cat("weak_modal"))
        src = files[-1].name
    else:
        lex = {k: set(v.split()) for k, v in SEED_LEXICON.items()}
        src = "embedded seed lists (approximate LM categories)"
        log.warning("LM Master Dictionary not found in data/: using embedded seed lists")
    lex["uncertainty"] = lex["uncertainty"] - lex["weak_modal"]   # UNC and HEDGE kept conceptually distinct
    return lex, src


# ============================================================================ parsing
def norm(t: str) -> str:
    return NOISE_RE.sub(" ", str(t).replace("\u2019", "'").replace("\u2018", "'"))


def clean_prepared(text: str) -> tuple[str, float]:
    paras = [p.strip() for p in norm(text).split("\n") if p.strip()]
    keep = [p for p in paras if not BOILER_RE.search(p)]
    total = sum(len(p) for p in paras)
    return "\n".join(keep), (1 - sum(len(p) for p in keep) / total) if total else 0.0


def parse_qa(text: str, truncated: bool) -> tuple[list[dict], list[str]]:
    """Pairs = a question block (consecutive questions merged) + all answer blocks until the next question."""
    parts = re.split(r"(?m)^(QUESTION|ANSWER):\s*", norm(text))
    blocks = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    if truncated and blocks:
        blocks = blocks[:-1]          # last block is cut mid-sentence by the cell limit
    pairs, answers, cur = [], [], None
    for role, txt in blocks:
        if not txt:
            continue
        if role == "QUESTION":
            if cur is not None and not cur["a"]:
                cur["q"] += " " + txt
                continue
            if cur is not None:
                pairs.append(cur)
            cur = {"q": txt, "a": []}
        else:
            answers.append(txt)
            if cur is not None:
                cur["a"].append(txt)
    if cur is not None and cur["a"]:
        pairs.append(cur)
    return [{"q": p["q"], "a": " ".join(p["a"])} for p in pairs], answers


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def lex_counts(text: str, lex: dict) -> dict:
    tk = tokens(text)
    pos = neg = unc = weak = 0
    for i, w in enumerate(tk):
        if w in lex["positive"]:
            window = tk[max(0, i - 3):i]
            if not any(t in NEGATORS or t.endswith("n't") for t in window):
                pos += 1
        if w in lex["negative"]:
            neg += 1
        if w in lex["uncertainty"]:
            unc += 1
        if w in lex["weak_modal"]:
            weak += 1
    return dict(n=len(tk), pos=pos, neg=neg, unc=unc, weak=weak, phr=len(HEDGE_RE.findall(text.lower())))


def net_tone(c: dict) -> float:
    return (c["pos"] - c["neg"]) / (c["pos"] + c["neg"]) if (c["pos"] + c["neg"]) > 0 else np.nan


def per_k(x, n) -> float:
    return 1000.0 * x / n if n > 0 else np.nan


# ============================================================================ embeddings
def _unit(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(n > 0, n, 1.0)


def _chunks(text: str, size: int = 180) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w), 1), size)] or [""]


def rowcos(A, B) -> np.ndarray:
    if hasattr(A, "multiply"):
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return (np.asarray(A) * np.asarray(B)).sum(axis=1)


class Embedder:
    """Dense vectors for short texts (Q/A pairs) and document vectors for whole sections."""

    def __init__(self, backend: str, fit_texts: list[str]):
        self.kind = None
        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer(C.ST_MODEL)
                self.kind = "st"
            except Exception as e:  # noqa: BLE001
                if backend == "st":
                    raise
                log.info("sentence-transformers unavailable (%s): TF-IDF/LSA fallback", type(e).__name__)
        if self.kind is None:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.9,
                                       ngram_range=(1, 2), max_features=60000)
            M = self.vec.fit_transform(fit_texts)
            self.svd = TruncatedSVD(min(C.LSA_DIM, M.shape[1] - 1), random_state=C.SEED).fit(M)
            self.kind = "tfidf"

    def short(self, texts: list[str]) -> np.ndarray:
        if self.kind == "st":
            return self.st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return _unit(self.svd.transform(self.vec.transform(texts)))

    def doc(self, texts: list[str]):
        if self.kind == "st":
            return np.vstack([_unit(self.st.encode(_chunks(t), normalize_embeddings=True,
                                                   show_progress_bar=False).mean(0, keepdims=True))[0]
                              for t in texts])
        return self.vec.transform(texts)   # rows are L2-normalised TF-IDF vectors


def anonymize(text: str, names: list[str]) -> str:
    """LLM/transformer look-ahead mitigation: mask firm identifiers and calendar years."""
    for n in names:
        text = re.sub(re.escape(n), "the company", text, flags=re.I)
    return re.sub(r"\b(19|20)\d{2}\b", "YEAR", text)


def finbert_tone(texts: dict, tag: str) -> pd.Series:
    """Mean P(positive) - P(negative) over sentences; cached by transcriptid."""
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.CACHE_DIR / f"finbert_{tag}.csv"
    done = pd.read_csv(cache, index_col=0).iloc[:, 0] if cache.exists() else pd.Series(dtype=float)
    todo = {k: v for k, v in texts.items() if k not in done.index}
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.FINBERT_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(C.FINBERT_MODEL).eval()
        lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
        new = {}
        for i, (k, t) in enumerate(todo.items()):
            sents = [s for s in SENT_RE.split(t) if len(s.split()) >= 4] or [t]
            probs = []
            for j in range(0, len(sents), 32):
                enc = tok(sents[j:j + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    probs.append(torch.softmax(mdl(**enc).logits, -1).numpy())
            P = np.vstack(probs)
            new[k] = float((P[:, lab["positive"]] - P[:, lab["negative"]]).mean())
            if (i + 1) % 20 == 0:
                log.info("FinBERT %s: %d/%d", tag, i + 1, len(todo))
        done = pd.concat([done, pd.Series(new)])
        done.to_csv(cache)
    return done


# ============================================================================ evasion classifier (optional)
PAIR_CLF_FEATS = ["REL", "REL_JAC", "log_a_words", "log_q_words", "len_ratio", "HEDGE_A", "NONANS_RULE"]


def label_classifiers(pairs: pd.DataFrame) -> dict:
    """Calibrate pair-level classifiers on manual labels (data/qa_labels.csv) if present.
    Labeled pairs get out-of-fold probabilities; unlabeled pairs get the full-fit probabilities."""
    out = {}
    if not C.LABELS_PATH.exists():
        return out
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lab = pd.read_csv(C.LABELS_PATH)
    for col, name in (("evasive", "EVADE_CLF"), ("non_answer", "NONANS_CLF")):
        if col not in lab:
            continue
        d = pairs.merge(lab[["pair_id", col]].dropna(), on="pair_id")
        if len(d) < 30 or d[col].nunique() < 2:
            log.warning("labels for %s: need >= 30 rows and both classes (have %d)", col, len(d))
            continue
        yv = d[col].astype(int).to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        oof = cross_val_predict(clf, d[PAIR_CLF_FEATS], yv, groups=d["transcriptid"],
                                cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
        clf.fit(d[PAIR_CLF_FEATS], yv)
        prob = pd.Series(clf.predict_proba(pairs[PAIR_CLF_FEATS])[:, 1], index=pairs["pair_id"])
        prob.loc[d["pair_id"].to_numpy()] = oof
        out[name] = (prob.to_numpy(), float(roc_auc_score(yv, oof)), int(len(d)))
        log.info("%s calibrated on %d labels, out-of-fold AUC = %.3f", name, len(d), out[name][1])
    return out


# ============================================================================ main blocks
def base_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ids, timing, targets and Model A controls."""
    et = ZoneInfo("America/New_York")
    out = pd.DataFrame(index=df.index)
    out["transcriptid"] = df["transcriptid"]
    out["ticker"] = df["ticker"]
    out["firm"] = df["ticker"]
    out["call_date"] = df["call_date"]
    out["day0"] = df["day0"]
    out["year"] = df["call_date"].dt.year
    out["event_q"] = df["day0"].dt.to_period("Q").astype(str)
    ts = pd.to_datetime(df["call_date"].dt.strftime("%Y-%m-%d") + " " + df["call_time_utc"].astype(str))
    ts_et = ts.dt.tz_localize("UTC").dt.tz_convert(et)
    hour = ts_et.dt.hour + ts_et.dt.minute / 60
    out["call_time_et"] = ts_et.dt.strftime("%Y-%m-%d %H:%M")
    out["session"] = np.select([hour >= 16, hour < 9.5], ["post", "pre"], "intraday")
    out["day0_gap_days"] = (df["day0"] - df["call_date"]).dt.days

    # targets: CAR windows start at +2, so CAR[0,+1] is excluded by construction
    out["CAR_2_5"], out["CAR_2_20"], out["CAR_2_60"] = df["car_2_5"], df["car_2_20"], df["car_2_60"]
    out["CAR_6_20"] = df["car_2_20"] - df["car_2_5"]
    out["CAR_21_60"] = df["car_2_60"] - df["car_2_20"]
    out["LOGVOL_2_20"] = np.log(df["vol_post_2_20"])

    # Model A
    sd_hist = df.groupby("ticker")["eps_surprise"].transform(
        lambda s: s.shift(1).rolling(C.SUE_HIST[0], min_periods=C.SUE_HIST[1]).std())
    out["SUE_TS"] = df["eps_surprise"] / sd_hist.replace(0.0, np.nan)
    out["SUE_PCT"] = df["surprise_pct"].clip(-1, 1)
    out["SUE_IBES"] = df["sue"]
    out["LOWPREC"] = df["consensus_eps"].abs() < 0.05     # cent rounding >= 10% of consensus
    out["DISP"] = df["dispersion"]
    out["COV"] = np.log(df["n_analysts"])
    out["SIZE"] = df["size"]
    out["BM"] = np.log(df["bm"])
    out["MOM"] = df["mom"]
    out["VOL"] = df["vol_pre"]
    out["AR01"] = df["car_0_1"]
    return out


def text_features(df: pd.DataFrame, lex: dict, embed: str, use_finbert: bool):
    rows, pair_rows, prep_clean, ans_all = [], [], {}, {}
    for r in df.itertuples(index=False):
        qa_trunc = len(str(r.qa_text)) >= EXCEL_CELL_LIMIT
        pr_trunc = len(str(r.prepared_text)) >= EXCEL_CELL_LIMIT
        prep, boiler = clean_prepared(r.prepared_text)
        pairs, answers = parse_qa(r.qa_text, qa_trunc)
        ans = " ".join(answers)
        prep_clean[r.transcriptid], ans_all[r.transcriptid] = prep, ans
        cp, ca = lex_counts(prep, lex), lex_counts(ans, lex)
        cm = {k: cp[k] + ca[k] for k in cp}
        rows.append(dict(
            transcriptid=r.transcriptid,
            TONE_P=net_tone(cp), TONE_Q=net_tone(ca), TONE_M=net_tone(cm),
            TONE_P_PW=per_k(cp["pos"] - cp["neg"], cp["n"]), TONE_Q_PW=per_k(ca["pos"] - ca["neg"], ca["n"]),
            UNC=per_k(cm["unc"], cm["n"]), UNC_P=per_k(cp["unc"], cp["n"]), UNC_Q=per_k(ca["unc"], ca["n"]),
            HEDGE=per_k(cm["weak"] + cm["phr"], cm["n"]), HEDGE_P=per_k(cp["weak"] + cp["phr"], cp["n"]),
            HEDGE_Q=per_k(ca["weak"] + ca["phr"], ca["n"]),
            words_prep_parsed=cp["n"], words_ans_parsed=ca["n"], n_pairs=len(pairs),
            n_q_blocks=len(re.findall(r"(?m)^QUESTION:", str(r.qa_text))), n_a_blocks=len(answers),
            qa_truncated=qa_trunc, prep_truncated=pr_trunc, boiler_share=boiler))
        for k, p in enumerate(pairs):
            ac = lex_counts(p["a"], lex)
            pair_rows.append(dict(
                pair_id=f"{r.transcriptid}_{k:03d}", transcriptid=r.transcriptid, ticker=r.ticker,
                call_date=r.call_date.date().isoformat(), pair_idx=k, question=p["q"], answer=p["a"],
                q_words=len(p["q"].split()), a_words=ac["n"],
                HEDGE_A=per_k(ac["weak"] + ac["phr"], ac["n"]),
                NONANS_RULE=int(bool(NONANSWER_RE.search(p["a"].lower())))))
    feats = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    pairs["log_a_words"] = np.log1p(pairs["a_words"])
    pairs["log_q_words"] = np.log1p(pairs["q_words"])
    pairs["len_ratio"] = pairs["a_words"] / pairs["q_words"].clip(lower=1)

    # lexical overlap as an embedding-free alternative relevance measure
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW

    def content(t):
        return {w for w in tokens(t) if w not in SW and len(w) > 2}

    pairs["REL_JAC"] = [len(a & b) / len(a | b) if (a | b) else np.nan
                        for a, b in ((content(q), content(x)) for q, x in zip(pairs["question"], pairs["answer"]))]

    # embeddings: unsupervised fit corpus restricted to pre-holdout text
    years = df.set_index("transcriptid")["call_date"].dt.year
    dev_ids = set(years[years <= C.DEV_END].index)
    fit_texts = (pairs.loc[pairs["transcriptid"].isin(dev_ids), "question"].tolist()
                 + pairs.loc[pairs["transcriptid"].isin(dev_ids), "answer"].tolist()
                 + [prep_clean[t] for t in dev_ids])
    emb = Embedder(embed, fit_texts)
    names = sorted(set(df["companyname"].astype(str)) | set(df["ticker"].astype(str))
                   | {n.split()[0] for n in df["companyname"].astype(str)}, key=len, reverse=True)
    prep_fn = (lambda t: anonymize(t, names)) if emb.kind == "st" else (lambda t: t)
    Q = emb.short([prep_fn(t) for t in pairs["question"]])
    A = emb.short([prep_fn(t) for t in pairs["answer"]])
    pairs["REL"] = rowcos(Q, A)

    agg = pairs.groupby("transcriptid").agg(REL=("REL", "mean"), REL_JAC=("REL_JAC", "mean"),
                                            NONANS=("NONANS_RULE", "mean"))
    feats = feats.merge(agg, on="transcriptid", how="left")
    feats["EVADE"] = 1 - feats["REL"]
    feats["TONE_GAP"] = feats["TONE_Q"] - feats["TONE_P"]

    clf_meta = {}
    for name, (prob, auc, n) in label_classifiers(pairs).items():
        pairs[name] = prob
        feats = feats.merge(pairs.groupby("transcriptid")[name].mean().rename(name), on="transcriptid", how="left")
        clf_meta[name] = {"oof_auc": auc, "n_labels": n}

    ids = df["transcriptid"].tolist()
    P = emb.doc([prep_fn(prep_clean[t]) for t in ids])
    Aa = emb.doc([prep_fn(ans_all[t]) for t in ids])
    feats = feats.set_index("transcriptid").loc[ids].reset_index()
    feats["DIV_QP"] = 1 - rowcos(P, Aa)
    # NOVEL: prepared remarks vs the same firm's previous call
    novel = np.full(len(ids), np.nan)
    order = df.reset_index(drop=True).sort_values(["ticker", "call_date"])
    for _, g in order.groupby("ticker"):
        idx = g.index.to_numpy()
        for i_prev, i_cur in zip(idx[:-1], idx[1:]):
            novel[i_cur] = 1 - float(rowcos(P[[i_cur]], P[[i_prev]])[0])
    feats["NOVEL"] = novel

    if use_finbert:
        fp = finbert_tone({t: anonymize(prep_clean[t], names) for t in ids}, "prepared")
        fq = finbert_tone({t: anonymize(ans_all[t], names) for t in ids}, "answers")
        feats["TONE_P_FB"] = feats["transcriptid"].map(fp)
        feats["TONE_Q_FB"] = feats["transcriptid"].map(fq)
    return feats, pairs, emb.kind, clf_meta, len(fit_texts)


def history_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Current value minus the mean of the firm's previous HIST_WINDOW calls (strictly prior)."""
    ev = ev.sort_values(["ticker", "call_date"]).copy()
    for new, base in (("dTONE", "TONE_M"), ("dUNC", "UNC"), ("dEVADE", "EVADE")):
        prior = ev.groupby("ticker")[base].transform(
            lambda s: s.shift(1).rolling(C.HIST_WINDOW, min_periods=C.HIST_WINDOW).mean())
        ev[new] = ev[base] - prior
    return ev.sort_values(["day0", "ticker"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed", choices=["auto", "st", "tfidf"], default="auto")
    ap.add_argument("--finbert", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(C.PANEL_PATH, parse_dates=["call_date", "day0", "fiscal_period_end"])
    df = df.sort_values(["call_date", "ticker"]).reset_index(drop=True)
    log.info("panel: %d calls, %d firms, %s to %s", len(df), df["ticker"].nunique(),
             df["call_date"].min().date(), df["call_date"].max().date())

    lex, lex_src = load_lexicons()
    base = base_table(df)
    feats, pairs, emb_kind, clf_meta, n_fit = text_features(df, lex, args.embed, args.finbert)
    ev = history_features(base.merge(feats, on="transcriptid", how="left"))
    ev["fiscal_period_end"] = ev["transcriptid"].map(df.set_index("transcriptid")["fiscal_period_end"])

    ev.to_csv(C.FEATURES_PATH, index=False)
    pairs.to_csv(C.PAIRS_PATH, index=False)
    meta = dict(
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        panel_sha256=C.file_sha256(C.PANEL_PATH), n_events=len(ev), n_pairs=len(pairs),
        lexicon_source=lex_src, embedding_backend=emb_kind,
        embedding_model=C.ST_MODEL if emb_kind == "st" else f"tfidf+lsa{C.LSA_DIM}",
        embedding_fit_corpus=f"{n_fit} texts from calls with year <= {C.DEV_END}" if emb_kind == "tfidf"
        else "pre-trained (no fit on this sample)",
        finbert=bool(args.finbert), label_classifiers=clf_meta, sue_variant_default=C.SUE_VARIANT,
        qa_truncated_calls=int(ev["qa_truncated"].sum()), prep_truncated_calls=int(ev["prep_truncated"].sum()),
        lowprec_events=int(ev["LOWPREC"].sum()))
    C.META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("wrote %s (%d rows) and %s (%d pairs)", C.FEATURES_PATH.name, len(ev), C.PAIRS_PATH.name, len(pairs))
    log.info("embedding backend: %s | lexicon: %s", emb_kind, lex_src)
    log.info("truncated Q&A texts (Excel 32,767-char cell limit): %d calls", meta["qa_truncated_calls"])


if __name__ == "__main__":
    main()
