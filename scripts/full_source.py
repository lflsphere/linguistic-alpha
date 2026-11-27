"""
Full-history source for 01 / 02 (docs/labeling_handoff_calibration.md).

`01_build_features.py --source full --embed st` calls `run()`. It reads the tables built by
`la transcripts` (data/interim/{transcript_calls,qa_pairs,call_sections}.parquet) and the raw link
tables in data/raw/sp500_pit_2008, and writes text features only (no targets / controls yet):

  data/features_text_full.csv        one row per call (2011-01-01 .. 2026-08-31), keyed by transcriptid / keydevid
  data/qa_pairs_full.csv             one row per Q&A pair of those calls: pair features + EVADE_CLF / NONANS_CLF
                                     (no texts: join on pair_id to data/interim/qa_pairs.parquet)
  data/features_text_full_meta.json  audit metadata (source, label_source, n_labels, oof_auc, ...)
  outputs/validation/full_source_log.md  call-table attrition, dropped duplicate calls, counts by year x sector

Call table: earnings calls, S&P 500 members on the call date through date-valid links
(ciq_wrds_gvkey -> crsp_ccmxpf_lnkhist LU/LC -> crsp_msp500list_v2), one permno per call
(CCM linkprim P, else the largest market cap on the session before the call), then one call per
gvkey and fiscal period (`events.duplicate_calls: keep_first`, by call time). Built from 2008 so the
history features (F4 deltas, NOVEL) are complete in 2011; features are written for sample.start-sample.end.
Firm key = gvkey.

Embeddings: sentence-transformers C.ST_MODEL only (pre-trained, nothing fitted on the sample), on text
with the call's own firm name and calendar years masked, cached by pair_id / transcriptid in
data/interim/emb_cache/ (npz shards). The label classifiers are fitted on build-period labels only
(call date <= split.build_end) and score every pair, holdout included.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

import common as C

log = C.get_logger("full")

CFG = yaml.safe_load((C.ROOT / "configs/default.yaml").read_text())
INTERIM = C.ROOT / CFG["paths"]["interim"]
RAW = C.ROOT / CFG["paths"]["raw"]
HISTORY_START = pd.Timestamp(f"{CFG['wrds']['start_year']}-01-01")   # 2008
SAMPLE_START = pd.Timestamp(CFG["sample"]["start"])
SAMPLE_END = pd.Timestamp(CFG["sample"]["end"])
BUILD_END = pd.Timestamp(CFG["split"]["build_end"])
assert CFG["events"]["duplicate_calls"] == "keep_first"

FEATURES_FULL_PATH = C.DATA_DIR / "features_text_full.csv"
PAIRS_FULL_PATH = C.DATA_DIR / "qa_pairs_full.csv"
META_FULL_PATH = C.DATA_DIR / "features_text_full_meta.json"
LOG_PATH = C.OUT_DIR / "validation" / "full_source_log.md"
EMB_CACHE = INTERIM / "emb_cache"
ANON_VERSION = "own-firm-name+year-v1"
DOC_CHUNK_WORDS = 180
DUP_WINDOW_DAYS = 3        # same gvkey + fiscal label within 3 calendar days = one release (Day 0 / Day +1,
                           # events.max_call_lag_sessions: 1, across a weekend)

CORP_SUFFIX = re.compile(
    r"[,.]?\s+(?:inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|llc|lp|l\.p|"
    r"n\.v|nv|s\.a|sa|ag|se|holdings?|group|the)\.?$", re.I)
FISCAL_RE = re.compile(r"\b((?:Q[1-4]|H[12]|FY)\s*,?\s*(?:19|20)\d{2})\b", re.I)
ANNUAL_RE = re.compile(r",\s*((?:19|20)\d{2})\s+Earnings Call\b", re.I)   # "Peabody ..., 2011 Earnings Call, Jan 24, 2012"


def _date(s):
    return pd.to_datetime(s.astype("string"), errors="coerce")


def _valid_on(df, d, lo, hi):
    return (df[lo].isna() | (df[lo] <= df[d])) & (df[hi].isna() | (df[d] <= df[hi]))


def _md(df, index=True):
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)])


def fiscal_label(headline: pd.Series) -> pd.Series:
    """'Q4 2008' style label parsed from the CIQ headline. Annual-results calls ('..., 2011 Earnings Call')
    and FY labels map to Q4 of that year (the annual release is the final-quarter release)."""
    h = headline.astype("string")
    q = h.str.extract(FISCAL_RE, expand=False).str.upper().str.replace(r"[\s,]+", " ", regex=True)
    q = q.str.replace(r"^FY ", "Q4 ", regex=True)
    a = "Q4 " + h.str.extract(ANNUAL_RE, expand=False)
    return q.fillna(a)


# ============================================================================ call table
def build_call_table(interim=INTERIM, raw=RAW, start=HISTORY_START, end=SAMPLE_END):
    """One row per keydevid: S&P 500 members on the call date, one permno, first call per fiscal period.
    Mirrors scripts/labeling/build_label_sample_full.py (labeller A) plus keep_first on duplicates.
    Returns (calls, steps, dropped_duplicates, notes)."""
    steps, notes = [], {}

    def step(name, before, after):
        steps.append((name, int(before["keydevid"].nunique()), int(after["keydevid"].nunique())))

    calls = pd.read_parquet(interim / "transcript_calls.parquet")
    for c in ("companyid", "keydevid", "transcriptid"):
        calls[c] = calls[c].astype("int64")
    calls["call_date"] = _date(calls["mostimportantdateutc"]).dt.normalize()
    calls["call_time_utc"] = calls["mostimportanttimeutc"].astype("string")
    c0 = calls
    c1 = c0[c0["keydeveventtypename"].astype("string").str.lower().str.contains("earnings call", na=False)]
    step("Earnings calls (event type)", c0, c1)
    c2 = c1[(c1["call_date"] >= start) & (c1["call_date"] <= end)]
    step(f"Call date {start.date()} to {end.date()}", c1, c2)

    g = pd.read_parquet(raw / "ciq_wrds_gvkey.parquet")
    g["companyid"] = g["companyid"].astype("int64")
    g["startdate"], g["enddate"] = _date(g["startdate"]), _date(g["enddate"])
    m = c2[["keydevid", "companyid", "call_date"]].merge(g, on="companyid", how="inner")
    m = m[_valid_on(m, "call_date", "startdate", "enddate")]
    m = m.sort_values(["keydevid", "primaryflag"], ascending=[True, False]).drop_duplicates("keydevid")
    c3 = c2.merge(m[["keydevid", "gvkey"]], on="keydevid", how="inner")
    step("CIQ companyid -> gvkey (date-valid)", c2, c3)

    ccm = pd.read_parquet(raw / "crsp_ccmxpf_lnkhist.parquet")
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"])].copy()
    ccm["linkdt"], ccm["linkenddt"] = _date(ccm["linkdt"]), _date(ccm["linkenddt"])
    lk = c3[["keydevid", "gvkey", "call_date"]].merge(ccm, on="gvkey", how="inner")
    lk = lk[_valid_on(lk, "call_date", "linkdt", "linkenddt")].copy()
    lk["permno"] = lk["permno"].astype("int64")
    c4 = c3[c3["keydevid"].isin(lk["keydevid"])]
    step("gvkey -> permno (CCM LU/LC, date-valid)", c3, c4)

    sp = pd.read_parquet(raw / "crsp_msp500list_v2.parquet")
    sp["permno"] = sp["permno"].astype("int64")
    sp["mbrstartdt"], sp["mbrenddt"] = _date(sp["mbrstartdt"]), _date(sp["mbrenddt"])
    mm = lk.merge(sp[["permno", "mbrstartdt", "mbrenddt"]], on="permno", how="inner")
    mm = mm[_valid_on(mm, "call_date", "mbrstartdt", "mbrenddt")]
    cand = mm.drop_duplicates(["keydevid", "permno"])[["keydevid", "permno", "linkprim", "call_date"]]
    c5 = c4[c4["keydevid"].isin(cand["keydevid"])]
    step("S&P 500 member on call date", c4, c5)

    # one permno per call (sample.share_class_rule): CCM linkprim P, else the largest market cap on the
    # session before the call; a call with a single member permno keeps it
    notes["calls_multi_permno"] = int((cand.groupby("keydevid")["permno"].nunique() > 1).sum())
    prim = cand[cand["linkprim"] == "P"].drop_duplicates("keydevid")
    rest = cand[~cand["keydevid"].isin(prim["keydevid"])]
    single = rest.groupby("keydevid")["permno"].transform("nunique") == 1
    rest_single, rest_multi = rest[single].drop_duplicates("keydevid"), rest[~single]
    notes["calls_no_primary_link"] = int(rest["keydevid"].nunique())
    notes["calls_mktcap_rule"] = int(rest_multi["keydevid"].nunique())
    if len(rest_multi):
        years = sorted(set(rest_multi["call_date"].dt.year) | set(rest_multi["call_date"].dt.year - 1))
        files = [f for y in years for f in glob.glob(str(raw / f"crsp_dsf_v2_{y}.parquet"))]
        permnos = [int(p) for p in rest_multi["permno"].unique()]
        dsf = ds.dataset(files).to_table(columns=["permno", "dlycaldt", "dlycap"],
                                         filter=ds.field("permno").isin(permnos)).to_pandas()
        dsf["permno"] = dsf["permno"].astype("int64")
        dsf["dlycaldt"] = _date(dsf["dlycaldt"])
        r = rest_multi.merge(dsf, on="permno")
        r = r[r["dlycaldt"] < r["call_date"]].sort_values("dlycaldt").groupby(["keydevid", "permno"]).tail(1)
        rest_multi = r.sort_values("dlycap", ascending=False).drop_duplicates("keydevid")
    one = pd.concat([prim[["keydevid", "permno"]], rest_single[["keydevid", "permno"]],
                     rest_multi[["keydevid", "permno"]]])
    c6 = c5.merge(one, on="keydevid", how="inner")
    step("One permno per call (share_class_rule)", c5, c6)

    h = pd.read_parquet(raw / "comp_co_hgic.parquet")
    h["indfrom"], h["indthru"] = _date(h["indfrom"]), _date(h["indthru"])
    s = c6[["keydevid", "gvkey", "call_date"]].merge(h[["gvkey", "gsector", "indfrom", "indthru"]], on="gvkey")
    s = s[_valid_on(s, "call_date", "indfrom", "indthru")].sort_values("indfrom") \
        .drop_duplicates("keydevid", keep="last")
    c6 = c6.merge(s[["keydevid", "gsector"]], on="keydevid", how="left")   # missing sector kept, counted
    notes["calls_missing_gsector"] = int(c6["gsector"].isna().sum())

    # one call per release (events.duplicate_calls: keep_first): same gvkey and fiscal label
    c6["fiscal_label"] = fiscal_label(c6["headline"])
    notes["calls_no_fiscal_label"] = int(c6["fiscal_label"].isna().sum())
    # 00:00:00 UTC is CIQ's placeholder (mostly pre-recorded remarks with no Q&A): ordered as an unknown time
    t_sort = c6["call_time_utc"].where(c6["call_time_utc"].ne("00:00:00")).fillna("23:59:59")
    notes["calls_time_placeholder"] = int(c6["call_time_utc"].eq("00:00:00").sum())
    c6["call_ts_utc"] = pd.to_datetime(c6["call_date"].dt.strftime("%Y-%m-%d") + " " + t_sort, errors="coerce")
    c6 = c6.sort_values(["gvkey", "call_ts_utc", "keydevid"]).reset_index(drop=True)
    key = c6["fiscal_label"].fillna("kd" + c6["keydevid"].astype(str))
    dup = np.zeros(len(c6), bool)
    kept_tid = np.full(len(c6), -1, dtype="int64")
    for _, idx in c6.groupby(["gvkey", key]).indices.items():
        if len(idx) < 2:
            continue
        first = idx[0]
        for i in idx[1:]:
            if (c6.at[i, "call_date"] - c6.at[first, "call_date"]).days <= DUP_WINDOW_DAYS:
                dup[i], kept_tid[i] = True, c6.at[first, "transcriptid"]
            else:
                first = i            # same label reused after a fiscal-year change: a different release
    notes["same_label_outside_window"] = int(c6.assign(_k=key).duplicated(["gvkey", "_k"]).sum() - dup.sum())
    dropped = c6.loc[dup, ["keydevid", "transcriptid", "gvkey", "companyname", "headline", "call_date",
                           "call_time_utc", "fiscal_label"]].copy()
    dropped["kept_transcriptid"] = kept_tid[dup]
    c7 = c6[~dup]
    step("One call per firm and fiscal period (duplicate_calls: keep_first)", c6, c7)

    sec = pd.read_parquet(interim / "call_sections.parquet")
    sec["transcriptid"] = sec["transcriptid"].astype("int64")
    c8 = c7.merge(sec[sec["transcriptid"].isin(c7["transcriptid"])], on="transcriptid", how="inner")
    step("Has call sections", c7, c8)
    cols = ["transcriptid", "keydevid", "companyid", "companyname", "gvkey", "permno", "gsector",
            "call_date", "call_time_utc", "headline", "fiscal_label", "version_final", "version_type",
            "prepared_text", "answer_text", "words_prepared", "words_question",
            "words_answer", "n_questions", "n_answers"]
    out = c8[cols].sort_values(["call_date", "gvkey", "keydevid"]).reset_index(drop=True)
    return out, steps, dropped.sort_values(["call_date", "gvkey"]).reset_index(drop=True), notes


def load_pairs(calls, interim=INTERIM):
    """pair_id = f"{transcriptid}_{pair_id:03d}" (labeller A's key); pairs with no answer turn or an empty
    answer are dropped. Returns (pairs, n_pairs_before_drop)."""
    tids = [int(t) for t in calls["transcriptid"].unique()]
    p = ds.dataset(str(interim / "qa_pairs.parquet")).to_table(
        columns=["transcriptid", "pair_id", "question", "answer", "n_answer_turns"],
        filter=ds.field("transcriptid").isin(tids)).to_pandas()
    n_all = len(p)
    p = p[(p["n_answer_turns"] > 0) & p["answer"].fillna("").str.strip().ne("")].copy()
    p["transcriptid"] = p["transcriptid"].astype("int64")
    p = p.rename(columns={"pair_id": "pair_idx"})
    p["pair_idx"] = p["pair_idx"].astype("int64")
    p["pair_id"] = p["transcriptid"].astype(str) + "_" + p["pair_idx"].astype(str).str.zfill(3)
    p = p.drop(columns=["n_answer_turns"])
    return p.sort_values(["transcriptid", "pair_idx"]).reset_index(drop=True), n_all


# ============================================================================ anonymisation
def anonymizer(companyname):
    """Mask the call's own firm name (full and without corporate suffixes) and calendar years.
    (The legacy panel code masked every firm name in the panel, which is too slow for ~800 firms.)"""
    name = str(companyname or "").strip()
    variants, short = {name}, name
    for _ in range(3):
        short = CORP_SUFFIX.sub("", short).strip(" ,.")
        variants.add(short)
    variants = sorted((v for v in variants if len(v) >= 3), key=len, reverse=True)
    rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in variants) + r")(?!\w)", re.I) \
        if variants else None

    def f(t):
        if rx is not None:
            t = rx.sub("the company", t)
        return re.sub(r"\b(19|20)\d{2}\b", "YEAR", t)
    return f


# ============================================================================ lexical features (parallel)
_LEX = None
_B = None


def _builder():
    """The 01 module, loaded by path (its file name starts with a digit)."""
    global _B
    if _B is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("build01", Path(__file__).with_name("01_build_features.py"))
        _B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_B)
    return _B


def _init(lex):
    global _LEX
    _LEX = lex


def _call_worker(batch):
    """Per call: the 01 call-level lexical features; per pair: HEDGE_A, NONANS_RULE, lengths, REL_JAC;
    plus the anonymised texts for the encoder."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    B, lex = _builder(), _LEX

    def content(t):
        return {w for w in B.tokens(t) if w not in SW and len(w) > 2}

    rows, prow, docs = [], [], []
    for tid, cname, prep_raw, ans_raw, nq, na, pairs in batch:
        anon = anonymizer(cname)
        prep, boiler = B.clean_prepared(prep_raw or "")
        ans = B.norm(ans_raw or "")
        cp, ca = B.lex_counts(prep, lex), B.lex_counts(ans, lex)
        cm = {k: cp[k] + ca[k] for k in cp}
        rows.append(dict(
            transcriptid=tid, TONE_P=B.net_tone(cp), TONE_Q=B.net_tone(ca), TONE_M=B.net_tone(cm),
            TONE_P_PW=B.per_k(cp["pos"] - cp["neg"], cp["n"]), TONE_Q_PW=B.per_k(ca["pos"] - ca["neg"], ca["n"]),
            UNC=B.per_k(cm["unc"], cm["n"]), UNC_P=B.per_k(cp["unc"], cp["n"]), UNC_Q=B.per_k(ca["unc"], ca["n"]),
            HEDGE=B.per_k(cm["weak"] + cm["phr"], cm["n"]), HEDGE_P=B.per_k(cp["weak"] + cp["phr"], cp["n"]),
            HEDGE_Q=B.per_k(ca["weak"] + ca["phr"], ca["n"]),
            words_prep_parsed=cp["n"], words_ans_parsed=ca["n"], n_pairs=len(pairs),
            n_q_blocks=int(nq), n_a_blocks=int(na), qa_truncated=False, prep_truncated=False,
            boiler_share=boiler))
        docs.append((tid, anon(prep), anon(ans)))
        for pid, q, a in pairs:
            q, a = B.norm(q or ""), B.norm(a or "")
            ac = B.lex_counts(a, lex)
            cq, cx = content(q), content(a)
            prow.append(dict(pair_id=pid, q_words=len(q.split()), a_words=ac["n"],
                             HEDGE_A=B.per_k(ac["weak"] + ac["phr"], ac["n"]),
                             NONANS_RULE=int(bool(B.NONANSWER_RE.search(a.lower()))),
                             REL_JAC=len(cq & cx) / len(cq | cx) if (cq | cx) else np.nan,
                             q_anon=anon(q), a_anon=anon(a)))
    return rows, prow, docs


def lexical_features(calls, pairs, lex, workers=None, batch_calls=200):
    grouped = {t: list(zip(g["pair_id"], g["question"], g["answer"]))
               for t, g in pairs.groupby("transcriptid", sort=False)}
    items = [(int(r.transcriptid), r.companyname, r.prepared_text, r.answer_text, r.n_questions, r.n_answers,
              grouped.pop(int(r.transcriptid), [])) for r in calls.itertuples(index=False)]
    batches = [items[i:i + batch_calls] for i in range(0, len(items), batch_calls)]
    del items, grouped
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    parallel = workers > 1 and len(batches) > 1
    rows, prow, docs = [], [], []
    t0 = time.time()
    if parallel:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(lex,))
        results = pool.imap(_call_worker, batches)
    else:
        _init(lex)
        results = map(_call_worker, batches)
    n_done = 0
    for i, (r, p, x) in enumerate(results):
        rows += r
        prow += p
        docs += x
        n_done += len(r)
        if (i + 1) % 20 == 0 or i + 1 == len(batches):
            log.info("lexical features: %d calls done (%.0fs)", n_done, time.time() - t0)
    if parallel:
        pool.close()
        pool.join()
    return (pd.DataFrame(rows), pd.DataFrame(prow),
            pd.DataFrame(docs, columns=["transcriptid", "prep_anon", "ans_anon"]))


# ============================================================================ embeddings + cache
def _chunks(text, size=DOC_CHUNK_WORDS):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w), 1), size)] or [""]


def _unit(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(n > 0, n, 1.0)


class Cache:
    """Shards <kind>_<n>.npz holding ids + float16 vectors; manifest pins the model and the anonymisation."""

    def __init__(self, folder, model, kind):
        self.folder, self.kind = Path(folder), kind
        self.folder.mkdir(parents=True, exist_ok=True)
        man = self.folder / "manifest.json"
        want = {"model": model, "anonymization": ANON_VERSION, "doc_chunk_words": DOC_CHUNK_WORDS}
        if man.exists():
            have = json.loads(man.read_text())
            if have != want:
                raise RuntimeError(f"embedding cache {folder} was built with {have}, not {want}")
        else:
            man.write_text(json.dumps(want, indent=2))
        self.files = sorted(self.folder.glob(f"{kind}_*.npz"))

    def load(self):
        parts = {}
        for f in self.files:
            with np.load(f, allow_pickle=False) as z:
                ids = z["ids"].astype(str)
                for k in z.files:
                    if k != "ids":
                        parts.setdefault(k, []).append((ids, z[k]))
        return {k: (np.concatenate([p[0] for p in v]), np.vstack([p[1] for p in v])) for k, v in parts.items()}

    def ids(self):
        s = set()
        for f in self.files:
            with np.load(f, allow_pickle=False) as z:
                s.update(z["ids"].astype(str).tolist())
        return s

    def write(self, ids, **arrays):
        f = self.folder / f"{self.kind}_{len(self.files):05d}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.asarray(ids, dtype=str), **{k: v.astype(np.float16) for k, v in arrays.items()})
        tmp.rename(f)                    # a killed run never leaves a half-written shard
        self.files.append(f)


def _lookup(ids, want):
    pos = pd.Series(np.arange(len(ids)), index=ids)
    pos = pos[~pos.index.duplicated(keep="last")]
    return pos.loc[want.to_numpy()].to_numpy()


def embed_pairs(pairs, encode, cache, batch=20000):
    """REL = cosine(question, answer) per pair."""
    have = cache.ids()
    todo = pairs[~pairs["pair_id"].isin(have)]
    log.info("pair embeddings: %d cached, %d to encode", len(pairs) - len(todo), len(todo))
    t0 = time.time()
    for i in range(0, len(todo), batch):
        b = todo.iloc[i:i + batch]
        cache.write(b["pair_id"].tolist(), Q=encode(b["q_anon"].tolist()), A=encode(b["a_anon"].tolist()))
        done = i + len(b)
        rate = done / max(time.time() - t0, 1e-9)
        log.info("pair embeddings: %d/%d (%.0f pairs/s, ETA %.0f min)", done, len(todo), rate,
                 (len(todo) - done) / max(rate, 1e-9) / 60)
    z = cache.load()
    j = _lookup(z["Q"][0], pairs["pair_id"])
    Q, A = _unit(z["Q"][1][j].astype(np.float32)), _unit(z["A"][1][j].astype(np.float32))
    return (Q * A).sum(axis=1)


def embed_docs(texts, encode, cache, batch=500):
    """Document vectors (mean of unit chunk vectors, re-normalised) of the prepared remarks (P) and of all
    answers (D), per call."""
    texts = texts.assign(tid=texts["transcriptid"].astype(str))
    have = cache.ids()
    todo = texts[~texts["tid"].isin(have)]
    log.info("document embeddings: %d cached, %d calls to encode", len(texts) - len(todo), len(todo))
    t0 = time.time()
    for i in range(0, len(todo), batch):
        b = todo.iloc[i:i + batch]
        out = {}
        for col, key in (("prep_anon", "P"), ("ans_anon", "D")):
            chunks = [_chunks(t) for t in b[col]]
            E = encode([c for cs in chunks for c in cs])
            k, vec = 0, []
            for cs in chunks:
                vec.append(_unit(E[k:k + len(cs)].mean(0, keepdims=True))[0])
                k += len(cs)
            out[key] = np.vstack(vec)
        cache.write(b["tid"].tolist(), **out)
        done = i + len(b)
        rate = done / max(time.time() - t0, 1e-9)
        log.info("document embeddings: %d/%d calls (%.1f calls/s, ETA %.0f min)", done, len(todo), rate,
                 (len(todo) - done) / max(rate, 1e-9) / 60)
    z = cache.load()
    j = _lookup(z["P"][0], texts["tid"])
    return _unit(z["P"][1][j].astype(np.float32)), _unit(z["D"][1][j].astype(np.float32))


def st_encoder(model_name=C.ST_MODEL):
    from sentence_transformers import SentenceTransformer   # --embed st only: no TF-IDF fallback
    m = SentenceTransformer(model_name)
    log.info("sentence-transformers %s on %s (max_seq_length %s)", model_name, m.device, m.max_seq_length)

    def encode(texts):
        return m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    encode.model_name = model_name
    return encode


# ============================================================================ labels
def label_classifiers_full(pairs, labels_path, build_end=BUILD_END):
    """Logistic classifiers on build-period labels: out-of-fold probabilities (GroupKFold by call) for the
    labelled pairs, full-fit probabilities for every other pair (holdout included). A median imputer
    (fitted inside each fold) covers REL_JAC / HEDGE_A NaNs."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    if not Path(labels_path).exists():
        log.warning("no labels at %s: EVADE_CLF / NONANS_CLF not built", labels_path)
        return out
    lab = pd.read_csv(labels_path, dtype={"pair_id": str})
    feats = _builder().PAIR_CLF_FEATS
    build = pairs[pd.to_datetime(pairs["call_date"]) <= build_end]
    for col, name in (("evasive", "EVADE_CLF"), ("non_answer", "NONANS_CLF")):
        if col not in lab:
            continue
        d = build.merge(lab[["pair_id", col]].dropna(), on="pair_id")
        n_unmatched = int((~lab["pair_id"].isin(build["pair_id"])).sum())
        if len(d) < 30 or d[col].nunique() < 2:
            log.warning("labels for %s: need >= 30 rows and both classes (have %d)", col, len(d))
            continue
        y = d[col].astype(int).to_numpy()
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
        k = min(5, d["transcriptid"].nunique())
        oof = cross_val_predict(clf, d[feats], y, groups=d["transcriptid"], cv=GroupKFold(n_splits=k),
                                method="predict_proba")[:, 1]
        clf.fit(d[feats], y)
        prob = pd.Series(clf.predict_proba(pairs[feats])[:, 1], index=pairs["pair_id"].to_numpy())
        prob.loc[d["pair_id"].to_numpy()] = oof
        out[name] = dict(prob=prob.to_numpy(), oof_auc=float(roc_auc_score(y, oof)), n_labels=int(len(d)),
                         n_pos=int(y.sum()), labels_not_in_pairs=n_unmatched, folds=int(k),
                         coef_std=dict(zip(feats, clf[-1].coef_[0].round(4).tolist())))
        log.info("%s: %d labels (%d ones), OOF AUC = %.3f, %d labels not matched", name, len(d), y.sum(),
                 out[name]["oof_auc"], n_unmatched)
    return out


# ============================================================================ history
def history_features(ev, key="firm"):
    """Current value minus the mean of the firm's previous HIST_WINDOW calls (strictly prior)."""
    ev = ev.sort_values([key, "call_date", "keydevid"]).copy()
    for new, base in (("dTONE", "TONE_M"), ("dUNC", "UNC"), ("dEVADE", "EVADE")):
        prior = ev.groupby(key)[base].transform(
            lambda s: s.shift(1).rolling(C.HIST_WINDOW, min_periods=C.HIST_WINDOW).mean())
        ev[new] = ev[base] - prior
    return ev


def novel(P, calls, key="firm"):
    """1 - cosine(prepared remarks, the same firm's previous call). `P` rows follow `calls` rows."""
    out = np.full(len(calls), np.nan)
    order = calls.reset_index(drop=True).sort_values([key, "call_date", "keydevid"])
    for _, g in order.groupby(key):
        idx = g.index.to_numpy()
        if len(idx) > 1:
            out[idx[1:]] = 1 - (P[idx[1:]] * P[idx[:-1]]).sum(1)
    return out


# ============================================================================ log
def write_log(path, steps, dropped, notes, ev_out, pairs_out, n_pairs_raw, n_pairs_kept, clf, run_min):
    lines = ["# Full-history source: call table and run log", "",
             "Built by `scripts/01_build_features.py --source full --embed st` (`scripts/full_source.py`) from "
             "`data/interim` (`la transcripts`) and `data/raw/sp500_pit_2008`. Firm key: gvkey.", "",
             f"Run time: {run_min:.1f} min. Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}.", "",
             "## Call attrition (calls from 2008, so the history features are complete in 2011)", "",
             "| Step | Calls before | Calls after | Dropped |", "|---|---|---|---|"]
    lines += [f"| {n} | {b:,} | {a:,} | {b - a:,} |" for n, b, a in steps]
    n_hist = steps[-1][2]
    lines += [f"| Features written for {SAMPLE_START.date()} to {SAMPLE_END.date()} "
              f"(earlier calls feed history only) | {n_hist:,} | {len(ev_out):,} | {n_hist - len(ev_out):,} |", "",
              f"- Calls with more than one member permno: {notes['calls_multi_permno']:,}; calls with no CCM "
              f"linkprim P link: {notes['calls_no_primary_link']:,}, of which {notes['calls_mktcap_rule']:,} "
              "needed the market-cap rule.",
              f"- Calls with no GICS sector valid on the call date (kept, sector blank): "
              f"{notes['calls_missing_gsector']:,}.",
              f"- Calls with no fiscal label in the headline (never treated as duplicates): "
              f"{notes['calls_no_fiscal_label']:,}.",
              f"- Calls at 00:00:00 UTC, CIQ's placeholder time (ordered last within their date for keep_first; "
              f"session = post): {notes['calls_time_placeholder']:,}.",
              f"- Same gvkey and fiscal label but more than {DUP_WINDOW_DAYS} days apart (fiscal-year changes; "
              f"kept as separate releases): {notes['same_label_outside_window']:,}.",
              f"- Q&A pairs of the kept calls (2008 on): {n_pairs_raw:,} parsed, {n_pairs_kept:,} with a non-empty "
              f"answer; {len(pairs_out):,} pairs in the written calls.", ""]
    lines += [f"## Duplicate calls dropped (same gvkey and fiscal label within {DUP_WINDOW_DAYS} days, "
              "later call time)", "",
              f"{len(dropped):,} calls dropped.", ""]
    if len(dropped):
        by_y = dropped.assign(year=dropped["call_date"].dt.year).groupby("year").size().rename("dropped")
        lines += [_md(by_y.to_frame()), "", "<details><summary>All dropped calls</summary>", ""]
        d = dropped.assign(call_date=dropped["call_date"].dt.strftime("%Y-%m-%d"),
                           headline=dropped["headline"].astype(str).str.replace("|", "/", regex=False))
        lines += [_md(d, index=False), "", "</details>", ""]
    tab = ev_out.assign(gsector=ev_out["gsector"].fillna("NA").astype(str)) \
        .pivot_table(index="year", columns="gsector", values="transcriptid", aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab.loc["total"] = tab.sum()
    lines += ["## Calls written, by year x GICS sector", "", _md(tab), "",
              f"Firms (gvkeys): {ev_out['gvkey'].nunique():,}. Pairs: {len(pairs_out):,}.", ""]
    if clf:
        lines += ["## Label classifiers (build-period labels only)", "",
                  "| Score | Labels | Ones | Labels not matched to a pair | OOF AUC |", "|---|---|---|---|---|"]
        lines += [f"| {n} | {r['n_labels']} | {r['n_pos']} | {r['labels_not_in_pairs']} | {r['oof_auc']:.3f} |"
                  for n, r in clf.items()]
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


# ============================================================================ driver
def run(embed="st", interim=INTERIM, raw=RAW, labels_path=C.LABELS_PATH, features_path=FEATURES_FULL_PATH,
        pairs_path=PAIRS_FULL_PATH, meta_path=META_FULL_PATH, log_path=LOG_PATH, cache_dir=EMB_CACHE,
        encode=None, workers=None):
    t_start = time.time()
    if embed != "st":
        raise SystemExit("--source full needs --embed st (the TF-IDF fallback depends on its fit corpus)")
    if encode is None:
        encode = st_encoder()
    model_name = getattr(encode, "model_name", "custom")
    B = _builder()

    calls, steps, dropped, notes = build_call_table(interim, raw)
    calls["firm"] = calls["gvkey"].astype(str)
    log.info("call table: %d calls, %d firms, %s to %s; %d duplicate calls dropped", len(calls),
             calls["gvkey"].nunique(), calls["call_date"].min().date(), calls["call_date"].max().date(), len(dropped))
    pairs, n_pairs_raw = load_pairs(calls, interim)
    n_pairs_kept = len(pairs)
    log.info("pairs: %d parsed, %d with an answer", n_pairs_raw, n_pairs_kept)

    lex, lex_src = B.load_lexicons()
    feats, pf, docs = lexical_features(calls, pairs, lex, workers)
    pairs = pairs.drop(columns=["question", "answer"]).merge(pf, on="pair_id", how="left", validate="one_to_one")
    del pf
    mc = calls.set_index("transcriptid")
    for c in ("keydevid", "gvkey", "gsector"):
        pairs[c] = pairs["transcriptid"].map(mc[c])
    pairs["call_date"] = pairs["transcriptid"].map(mc["call_date"]).dt.strftime("%Y-%m-%d")
    pairs["log_a_words"] = np.log1p(pairs["a_words"])
    pairs["log_q_words"] = np.log1p(pairs["q_words"])
    pairs["len_ratio"] = pairs["a_words"] / pairs["q_words"].clip(lower=1)

    pairs["REL"] = embed_pairs(pairs, encode, Cache(cache_dir, model_name, "pairs"))
    pairs = pairs.drop(columns=["q_anon", "a_anon"])
    docs = docs.set_index("transcriptid").loc[calls["transcriptid"]].reset_index()
    P, D = embed_docs(docs, encode, Cache(cache_dir, model_name, "docs"))
    del docs

    clf = label_classifiers_full(pairs, labels_path)
    for name, r in clf.items():
        pairs[name] = r["prob"]
    lab_ids = set(pd.read_csv(labels_path, usecols=["pair_id"], dtype=str)["pair_id"]) \
        if Path(labels_path).exists() else set()
    pairs["labelled_oof"] = pairs["pair_id"].isin(lab_ids) & (pd.to_datetime(pairs["call_date"]) <= BUILD_END)

    agg = {"REL": ("REL", "mean"), "REL_JAC": ("REL_JAC", "mean"), "NONANS": ("NONANS_RULE", "mean")}
    agg.update({n: (n, "mean") for n in clf})
    a = pairs.groupby("transcriptid").agg(**agg).reset_index()
    ids = calls[["transcriptid", "keydevid", "companyid", "companyname", "gvkey", "firm", "permno", "gsector",
                 "call_date", "call_time_utc", "fiscal_label", "version_final", "version_type",
                 "words_prepared", "words_question", "words_answer", "n_questions", "n_answers"]].copy()
    et = ZoneInfo(CFG["events"]["release_time_tz"])
    t_known = ids["call_time_utc"].where(ids["call_time_utc"].ne("00:00:00"))   # 00:00:00 = CIQ placeholder
    ts = pd.to_datetime(ids["call_date"].dt.strftime("%Y-%m-%d") + " " + t_known.fillna(""),
                        errors="coerce").dt.tz_localize("UTC").dt.tz_convert(et)
    hour = ts.dt.hour + ts.dt.minute / 60
    ids["call_time_et"] = ts.dt.strftime("%Y-%m-%d %H:%M")
    # missing call time -> after the close (events.missing_call_time: next_session)
    ids["session"] = np.where(ts.isna(), "post", np.select([hour >= 16, hour < 9.5], ["post", "pre"], "intraday"))
    ids["call_time_missing"] = ts.isna()
    ids["day0"] = pd.NaT            # la events not built yet
    ids["year"] = ids["call_date"].dt.year
    ev = ids.merge(feats, on="transcriptid", how="left").merge(a, on="transcriptid", how="left")
    ev["EVADE"] = 1 - ev["REL"]
    ev["TONE_GAP"] = ev["TONE_Q"] - ev["TONE_P"]
    ev["DIV_QP"] = 1 - (P * D).sum(1)
    ev["NOVEL"] = novel(P, ev)
    ev = history_features(ev, "firm")

    keep = (ev["call_date"] >= SAMPLE_START) & (ev["call_date"] <= SAMPLE_END)
    ev_out = ev[keep].sort_values(["call_date", "gvkey", "keydevid"]).reset_index(drop=True)
    ev_out["call_date"] = ev_out["call_date"].dt.strftime("%Y-%m-%d")
    pairs_out = pairs[pairs["transcriptid"].isin(ev_out["transcriptid"])].reset_index(drop=True)
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    ev_out.to_csv(features_path, index=False)
    pairs_out.to_csv(pairs_path, index=False)
    run_min = (time.time() - t_start) / 60
    meta = dict(
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), source="full",
        label_source=str(Path(labels_path).relative_to(C.ROOT) if str(labels_path).startswith(str(C.ROOT))
                         else labels_path),
        n_labels={n: r["n_labels"] for n, r in clf.items()},
        oof_auc={n: r["oof_auc"] for n, r in clf.items()},
        label_classifiers={n: {k: v for k, v in r.items() if k != "prob"} for n, r in clf.items()},
        classifier_features=B.PAIR_CLF_FEATS, classifier_fit_period=f"call_date <= {BUILD_END.date()}",
        n_events=len(ev_out), n_pairs=len(pairs_out), n_calls_history=len(ev),
        sample=[str(SAMPLE_START.date()), str(SAMPLE_END.date())], history_start=str(HISTORY_START.date()),
        firm_key="gvkey", duplicate_calls="keep_first", n_duplicate_calls_dropped=len(dropped),
        call_table_steps=[dict(step=n, before=b, after=a_) for n, b, a_ in steps], call_table_notes=notes,
        lexicon_source=lex_src, embedding_backend="st", embedding_model=model_name,
        embedding_fit_corpus="pre-trained (no fit on this sample)", anonymization=ANON_VERSION,
        qa_truncated_calls=0, prep_truncated_calls=0, run_minutes=round(run_min, 1))
    Path(meta_path).write_text(json.dumps(meta, indent=2, default=str))
    write_log(log_path, steps, dropped, notes, ev_out, pairs_out, n_pairs_raw, n_pairs_kept, clf, run_min)
    log.info("wrote %s (%d calls) and %s (%d pairs) in %.1f min", Path(features_path).name, len(ev_out),
             Path(pairs_path).name, len(pairs_out), run_min)
    return meta
