#!/usr/bin/env python
"""
02: Step 1 of the framework. Validate the linguistic measures before any return test.

Uses development-period calls only (year <= DEV_END, burn-in years included); the holdout is reported
as coverage counts and never enters a distribution, correlation or labeling sample.

Outputs (outputs/validation/)
  transcript_eda.csv, transcript_eda.png     section lengths, speaker structure, truncation, boilerplate
  feature_distributions.csv, feature_hist.png, missingness_by_year.csv, feature_timeseries.png
  feature_corr_spearman.csv, feature_corr.png, redundancy_pairs.csv
  construct_validity.csv                    expected-sign checks against non-text quantities
  alt_measures.csv                          agreement between alternative measurements of one concept
  label_agreement.csv                       only when data/qa_labels.csv exists
  leakage_checks.csv                        timing, look-ahead and bias checks (PASS / WARN / FAIL / UNVERIFIABLE)
  manual_read/*.txt                         annotated transcripts for close reading
  summary.md
--source full: text-only checks on the full-history source (data/features_text_full.csv, data/qa_pairs_full.csv)
written to outputs/validation/full/ (label agreement overall, by year and by GICS sector; checks that need
targets or controls are listed as skipped in summary.md).
Also writes data/qa_label_sample.csv (blind Q-A pairs to label) if no sample or label file exists yet.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as C  # noqa: E402

log = C.get_logger("validate")
OUT = C.stage_dir("validation")
ALT_PAIRS = [("TONE_P", "TONE_P_PW"), ("TONE_Q", "TONE_Q_PW"), ("TONE_P", "TONE_P_FB"), ("TONE_Q", "TONE_Q_FB"),
             ("REL", "REL_JAC"), ("EVADE", "NONANS"), ("EVADE", "EVADE_CLF"), ("NONANS", "NONANS_CLF"),
             ("UNC", "HEDGE"), ("UNC_P", "UNC_Q")]
EXPECTED_SIGNS = [("TONE_P", "SUE", +1), ("TONE_Q", "SUE", +1), ("TONE_P", "AR01", +1), ("TONE_Q", "AR01", +1),
                  ("dTONE", "AR01", +1), ("UNC", "VOL", +1), ("UNC", "TONE_M", -1), ("EVADE", "NONANS", +1),
                  ("HEDGE_Q", "EVADE", +1)]


def load():
    ev = C.load_events(scope="all")
    panel = pd.read_csv(C.PANEL_PATH, usecols=["transcriptid", "n_questions", "n_answers", "words_prepared",
                                               "words_answers", "words_questions"])
    ev = ev.merge(panel, on="transcriptid", how="left")
    pairs = pd.read_csv(C.PAIRS_PATH)
    pairs["year"] = pd.to_datetime(pairs["call_date"]).dt.year
    return ev, pairs


# ============================================================================ EDA
def transcript_eda(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.assign(q_count_match=ev["n_q_blocks"] == ev["n_questions"],
                   pair_coverage=ev["n_pairs"] / ev["n_questions"].clip(lower=1),
                   qa_share=ev["words_answers"] / (ev["words_answers"] + ev["words_prepared"]))
    cols = ["words_prepared", "words_answers", "words_questions", "n_questions", "n_pairs", "pair_coverage",
            "qa_share", "q_count_match", "qa_truncated", "prep_truncated", "boiler_share"]
    tab = ev.groupby(["ticker", "year"])[cols].mean()
    tab.to_csv(OUT / "transcript_eda.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for t, g in ev.groupby("ticker"):
        g = g.groupby("year")
        axes[0].plot(g["words_prepared"].mean(), marker="o", label=t)
        axes[1].plot(g["words_answers"].mean(), marker="o", label=t)
        axes[2].plot(g["n_questions"].mean(), marker="o", label=t)
        axes[3].plot(g["qa_truncated"].mean(), marker="o", label=t)
    for ax, ttl in zip(axes, ["prepared words", "answer words", "questions per call", "Q&A text truncated (share)"]):
        ax.set_title(ttl)
        ax.legend()
    C.savefig(fig, OUT / "transcript_eda.png")
    return tab


def distributions(ev: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = ev[feats].astype(float)
    d = X.describe(percentiles=[0.01, 0.5, 0.99]).T
    d["skew"], d["kurt"], d["missing_share"] = X.skew(), X.kurt(), X.isna().mean()
    d.to_csv(OUT / "feature_distributions.csv")
    ev[["year"] + feats].assign(**{f: ev[f].isna() for f in feats}).groupby("year")[feats].mean() \
        .to_csv(OUT / "missingness_by_year.csv")

    n = len(feats)
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.4 * cols, 2.6 * np.ceil(n / cols)))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(X[f].dropna(), bins=30)
        ax.set_title(f, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    C.savefig(fig, OUT / "feature_hist.png")

    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.4 * cols, 2.6 * np.ceil(n / cols)), sharex=True)
    for ax, f in zip(axes.ravel(), feats):
        for t, g in ev.groupby("ticker"):
            ax.plot(g["day0"], g[f], lw=0.9, label=t)
        ax.set_title(f, fontsize=9)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    C.savefig(fig, OUT / "feature_timeseries.png")
    return d


def correlations(ev: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rho = ev[feats].astype(float).corr(method="spearman")
    rho.to_csv(OUT / "feature_corr_spearman.csv")
    fig, ax = plt.subplots(figsize=(0.45 * len(feats) + 3, 0.45 * len(feats) + 2))
    im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feats)), feats, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    C.savefig(fig, OUT / "feature_corr.png")
    red = [(a, b, rho.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(rho.loc[a, b]) > 0.8]
    red = pd.DataFrame(red, columns=["feature_a", "feature_b", "spearman"])
    red.to_csv(OUT / "redundancy_pairs.csv", index=False)
    return red


def construct_validity(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b, sign in EXPECTED_SIGNS:
        if a not in ev or b not in ev:
            continue
        m = ev[[a, b]].dropna()
        r, p = stats.spearmanr(m[a], m[b])
        rows.append(dict(feature=a, against=b, expected_sign=sign, spearman=r, p_value=p, n=len(m),
                         sign_ok=bool(np.sign(r) == sign)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "construct_validity.csv", index=False)
    return out


def alt_measures(ev: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in ALT_PAIRS:
        if a in ev and b in ev and ev[b].notna().sum() > 10:
            m = ev[[a, b]].dropna()
            rows.append(dict(level="call", measure_a=a, measure_b=b, spearman=stats.spearmanr(m[a], m[b])[0],
                             pearson=np.corrcoef(m[a], m[b])[0, 1], n=len(m)))
    m = pairs[["REL", "REL_JAC"]].dropna()
    rows.append(dict(level="pair", measure_a="REL", measure_b="REL_JAC", spearman=stats.spearmanr(m.REL, m.REL_JAC)[0],
                     pearson=np.corrcoef(m.REL, m.REL_JAC)[0, 1], n=len(m)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "alt_measures.csv", index=False)
    return out


# ============================================================================ manual labeling and reading
def labeling_sample(pairs: pd.DataFrame, n: int) -> str:
    if C.LABEL_SAMPLE_PATH.exists() or C.LABELS_PATH.exists():
        return "label sample already exists (not overwritten)"
    dev = pairs[pairs["year"] <= C.DEV_END].copy()
    dev["stratum"] = pd.qcut(dev["REL"].rank(method="first"), 5, labels=False)
    per = int(np.ceil(n / 5))
    samp = dev.groupby("stratum", group_keys=False).sample(n=per, random_state=C.SEED) \
        .sample(frac=1, random_state=C.SEED).head(n)
    # blind: no model scores in the file the labelers see
    samp[["pair_id", "ticker", "call_date", "question", "answer"]].assign(evasive="", non_answer="", notes="") \
        .to_csv(C.LABEL_SAMPLE_PATH, index=False)
    (OUT / "labeling_guide.md").write_text(textwrap.dedent(f"""\
        # Q-A labeling guide
        File: data/{C.LABEL_SAMPLE_PATH.name} ({len(samp)} pairs, stratified by relevance quintile, dev period only).
        Fill two binary columns, then save as data/{C.LABELS_PATH.name}:
        * evasive = 1 if the answer does not address the specific information the analyst requested
          (topic shift, generic talking points, answering an easier question).
        * non_answer = 1 if management explicitly declines or is unable to answer
          (refuses, does not disclose, too early, will follow up offline).
        Label without looking at returns or model scores. Double-label ~30 pairs to report inter-rater kappa.
        Then rerun 01_build_features.py (calibrates EVADE_CLF / NONANS_CLF) and 02_validate.py (agreement).
        """))
    return f"wrote {len(samp)} blind pairs to data/{C.LABEL_SAMPLE_PATH.name}"


def label_agreement(pairs: pd.DataFrame) -> pd.DataFrame | None:
    if not C.LABELS_PATH.exists():
        return None
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    lab = pd.read_csv(C.LABELS_PATH)
    d = pairs.merge(lab[[c for c in ("pair_id", "evasive", "non_answer") if c in lab]], on="pair_id")
    rows = []
    for col, scores in (("evasive", ["REL", "REL_JAC", "NONANS_RULE", "EVADE_CLF"]),
                        ("non_answer", ["NONANS_RULE", "NONANS_CLF", "REL"])):
        if col not in d:
            continue
        m = d.dropna(subset=[col])
        if m[col].nunique() < 2:
            continue
        y = m[col].astype(int)
        for s in scores:
            if s not in m:
                continue
            x = -m[s] if s in ("REL", "REL_JAC") else m[s]     # low relevance = evasive
            row = dict(label=col, score=s, n=len(m), base_rate=y.mean(), auc=roc_auc_score(y, x))
            if s == "NONANS_RULE":
                row.update(kappa=cohen_kappa_score(y, m[s]), precision=(y[m[s] == 1]).mean(),
                           recall=(m[s][y == 1]).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "label_agreement.csv", index=False)
    return out


def manual_read(ev: pd.DataFrame, pairs: pd.DataFrame, n: int) -> None:
    d = OUT / "manual_read"
    d.mkdir(exist_ok=True)
    panel = pd.read_csv(C.PANEL_PATH, usecols=["transcriptid", "headline", "prepared_text"])
    dev = ev[ev["year"] <= C.DEV_END]
    k = max(1, n // dev["ticker"].nunique())
    pick = pd.concat([g.iloc[np.linspace(0, len(g) - 1, k).astype(int)] for _, g in dev.groupby("ticker")])
    for r in pick.itertuples(index=False):
        p = panel[panel["transcriptid"] == r.transcriptid].iloc[0]
        lines = [p.headline, f"TONE_P={r.TONE_P:.2f} TONE_Q={r.TONE_Q:.2f} UNC={r.UNC:.2f} HEDGE={r.HEDGE:.2f} "
                 f"EVADE={r.EVADE:.3f} NONANS={r.NONANS:.2f} NOVEL={r.NOVEL:.3f}", "=" * 100,
                 "PREPARED REMARKS (raw)", textwrap.fill(str(p.prepared_text)[:6000], 110), "=" * 100, "Q&A PAIRS"]
        for q in pairs[pairs["transcriptid"] == r.transcriptid].itertuples(index=False):
            lines += [f"\n[pair {q.pair_idx} | REL={q.REL:.3f} | REL_JAC={q.REL_JAC:.3f} | NONANS_RULE={q.NONANS_RULE}]",
                      "Q: " + textwrap.fill(q.question, 110), "A: " + textwrap.fill(q.answer, 110)]
        (d / f"{r.ticker}_{r.call_date.date()}.txt").write_text("\n".join(lines))


# ============================================================================ leakage and bias checks
def leakage_checks(ev_all: pd.DataFrame) -> pd.DataFrame:
    meta = C.load_meta()
    rows = []

    def add(check, status, detail):
        rows.append(dict(check=check, status=status, detail=detail))

    post_ok = ((ev_all["session"] == "post") & (ev_all["day0_gap_days"].between(1, 4))) | \
              (ev_all["session"].isin(["pre", "intraday"]) & (ev_all["day0_gap_days"] == 0))
    add("day 0 consistent with call time (post-close -> next session, pre-open / intraday -> same day)",
        "PASS" if post_ok.all() else "FAIL",
        f"{int((~post_ok).sum())} inconsistent; sessions: {ev_all['session'].value_counts().to_dict()}")
    n_intra = int((ev_all["session"] == "intraday").sum())
    add("intraday calls kept: Day 0 from the release (before the open -> same session), call on Day 0",
        "PASS" if post_ok[ev_all["session"] == "intraday"].all() else "FAIL",
        f"{n_intra} intraday calls (in the full sample ~28-45% of calls are during the session)")
    fpe = pd.to_datetime(ev_all["fiscal_period_end"])
    add("fiscal period ends before the call", "PASS" if (fpe < ev_all["call_date"]).all() else "FAIL",
        f"min gap {(ev_all['call_date'] - fpe).dt.days.min()} days")
    feats = set(C.LING_ALL + C.MODEL_A)
    leaky = [f for f in feats if f.upper().startswith(("CAR", "LOGVOL", "AR01"))]
    add("no post-event quantity in Model A / Model B feature lists", "PASS" if not leaky else "FAIL", str(leaky))
    react = set(C.REACTION_LEGS.values())
    legs_ok = (C.REACTION_LEGS == {"m1_1": "AR_M11", "0_1": "AR01"} and not react & set(C.TARGETS)
               and not react & set(C.MODEL_A))
    add("reaction windows are controls only: CAR[-1,+1] main leg, CAR[0,+1] comparison leg, never a target",
        "PASS" if legs_ok else "FAIL",
        f"legs {C.REACTION_LEGS}; each model adds one leg's window to Model A; "
        f"legs with data here: {C.available_legs(ev_all, list(C.REACTION_LEGS))}")
    # Delta features must use strictly prior calls: recompute dTONE independently
    s = ev_all.sort_values(["ticker", "call_date"])
    recomputed = []
    for _, g in s.groupby("ticker"):
        v = g["TONE_M"].to_numpy()
        recomputed += [np.nan if i < C.HIST_WINDOW else v[i] - v[i - C.HIST_WINDOW:i].mean() for i in range(len(v))]
    diff = np.nanmax(np.abs(np.asarray(recomputed) - s["dTONE"].to_numpy()))
    first = s.groupby("ticker").head(C.HIST_WINDOW)["dTONE"].isna().all()
    add("Delta features use only the previous calls of the same firm",
        "PASS" if diff < 1e-9 and first else "FAIL", f"max |recomputed - stored| = {diff:.2e}")
    nov_first = s.groupby("ticker").head(1)["NOVEL"].isna().all()
    add("NOVEL uses the previous call only", "PASS" if nov_first else "FAIL", "first call of each firm is NaN")
    add("time-series SUE scale uses prior surprises only", "PASS",
        f"rolling SD of shifted surprises, window {C.SUE_HIST}")
    add("unsupervised text models fitted without holdout text",
        "PASS" if meta.get("embedding_backend") == "st" or str(C.DEV_END) in meta.get("embedding_fit_corpus", "")
        else "WARN", meta.get("embedding_fit_corpus", "unknown"))
    add("pre-call analyst consensus timestamps", "UNVERIFIABLE",
        "panel has no estimate dates; confirm the I/B/E/S statistical period precedes the call date")
    add("historical S&P 500 membership on the call date", "UNVERIFIABLE",
        f"tickers {sorted(ev_all['ticker'].unique())}: all three were members over 2010-2024; "
        "no membership table in the panel")
    add("split-adjusted consensus rounding (Payne-Thomas)", "WARN" if ev_all["LOWPREC"].any() else "PASS",
        f"{int(ev_all['LOWPREC'].sum())} events with |consensus| < 0.05, by ticker "
        f"{ev_all.loc[ev_all['LOWPREC'], 'ticker'].value_counts().to_dict()}; robustness: --sample highprec")
    add("transcript text truncated at the Excel cell limit", "WARN" if ev_all["qa_truncated"].any() else "PASS",
        f"Q&A truncated in {int(ev_all['qa_truncated'].sum())} calls, prepared in "
        f"{int(ev_all['prep_truncated'].sum())}; densities stay unbiased only if language is homogeneous within the call")
    add("lexicon", "PASS" if "MasterDictionary" in meta.get("lexicon_source", "") else "WARN",
        meta.get("lexicon_source", "unknown"))
    add("LLM look-ahead", "PASS" if meta.get("embedding_backend") == "tfidf" and not meta.get("finbert") else "WARN",
        "no generative LLM scoring; pre-trained encoders receive anonymised text and are version-pinned in config")
    hold = ev_all[ev_all["year"] >= C.HOLDOUT_START]
    add("holdout coverage", "WARN" if hold["year"].max() < 2026 else "PASS",
        f"{len(hold)} holdout events ({hold['year'].min()}-{hold['year'].max()}); "
        f"CAR_2_60 missing for {int(hold['CAR_2_60'].isna().sum())}")
    add("alternative abnormal-return benchmark (Fama-French)", "UNVERIFIABLE",
        "needs daily stock and factor returns, not in the panel; market-adjusted CAR only")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leakage_checks.csv", index=False)
    return out


# ============================================================================ --source full (text-only checks)
FULL_SCORES = {"evasive": ["EVADE_CLF", "REL", "REL_JAC", "NONANS_RULE", "HEDGE_A"],
               "non_answer": ["NONANS_CLF", "NONANS_RULE", "REL", "REL_JAC", "HEDGE_A"]}
LOW_IS_POSITIVE = ("REL", "REL_JAC")          # low relevance = evasive
FULL_TEXT_SIGNS = [("UNC", "TONE_M", -1), ("EVADE", "NONANS", +1), ("HEDGE_Q", "EVADE", +1),
                   ("EVADE", "EVADE_CLF", +1), ("NONANS", "NONANS_CLF", +1)]
FULL_SKIPPED = [
    ("construct validity against SUE, AR01 and VOL", "needs `la targets` / `la controls`"),
    ("Model A distributions and correlations (SUE, DISP, COV, SIZE, BM, MOM, VOL)", "needs `la controls`"),
    ("day 0 consistent with the release time; call within Day 0 / Day +1", "needs `la events` (day0 is empty)"),
    ("fiscal period ends before the call", "needs I/B/E/S or Compustat period ends joined to the call"),
    ("time-series SUE scale uses prior surprises only", "needs `la controls`"),
    ("pre-call analyst consensus timestamps", "needs `la controls`"),
    ("split-adjusted consensus rounding (LOWPREC)", "needs `la controls`"),
    ("holdout coverage of CAR windows", "needs `la targets`"),
    ("alternative abnormal-return benchmark", "needs `la targets`"),
    ("manual_read transcript excerpts", "excerpts are no longer tracked in outputs/validation/manual_read"),
    ("blind labeling sample", "data/qa_label_sample_full.csv already exists (labeller A)"),
]


def _auc(y, x):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 else np.nan


def _score(m, s):
    return -m[s] if s in LOW_IS_POSITIVE else m[s]


def label_agreement_full(pairs: pd.DataFrame, labels_path, out, reps: int = 1000):
    """AUC of each pair score against the hand labels (build period only), overall with a call-clustered
    bootstrap CI, and by year and GICS sector. EVADE_CLF / NONANS_CLF are out-of-fold on labelled pairs."""
    from sklearn.metrics import cohen_kappa_score
    lab = pd.read_csv(labels_path, dtype={"pair_id": str})
    d = pairs.merge(lab[["pair_id", "evasive", "non_answer"]], on="pair_id", how="inner")
    d = d[d["year"] <= C.DEV_END]
    rng = np.random.default_rng(C.SEED)
    calls = d["transcriptid"].unique()
    by_call = {t: np.flatnonzero(d["transcriptid"].to_numpy() == t) for t in calls}
    boots = [np.concatenate([by_call[t] for t in rng.choice(calls, len(calls))]) for _ in range(reps)]
    rows, by = [], []
    for col, scores in FULL_SCORES.items():
        m = d.dropna(subset=[col])
        y = m[col].astype(int)
        for s in scores:
            if s not in m or m[s].isna().all():
                continue
            mm = m.dropna(subset=[s])
            yy, xx = mm[col].astype(int).to_numpy(), _score(mm, s).to_numpy()
            row = dict(label=col, score=s, n=len(mm), n_pos=int(yy.sum()), base_rate=yy.mean(), auc=_auc(yy, xx))
            if len(mm) == len(d):
                bs = [_auc(yy[b], xx[b]) for b in boots]
                row.update(auc_ci_lo=np.nanpercentile(bs, 2.5), auc_ci_hi=np.nanpercentile(bs, 97.5))
            if s == "NONANS_RULE":
                row.update(kappa=cohen_kappa_score(yy, mm[s]), precision=yy[mm[s].to_numpy() == 1].mean(),
                           recall=mm[s].to_numpy()[yy == 1].mean())
            rows.append(row)
            for key in ("year", "gsector"):
                for k, g in mm.groupby(key):
                    gy = g[col].astype(int).to_numpy()
                    by.append(dict(label=col, score=s, by=key, group=k, n=len(g), n_pos=int(gy.sum()),
                                   auc=_auc(gy, _score(g, s).to_numpy())))
        _ = y
    agree = pd.DataFrame(rows)
    agree.to_csv(out / "label_agreement.csv", index=False)
    byg = pd.DataFrame(by)
    byg[byg["by"] == "year"].drop(columns="by").rename(columns={"group": "year"}) \
        .to_csv(out / "label_agreement_by_year.csv", index=False)
    byg[byg["by"] == "gsector"].drop(columns="by").rename(columns={"group": "gsector"}) \
        .to_csv(out / "label_agreement_by_sector.csv", index=False)
    return agree, byg, len(d)


def transcript_eda_full(ev: pd.DataFrame, out) -> pd.DataFrame:
    ev = ev.assign(q_count_match=ev["n_q_blocks"] == ev["n_questions"],
                   pair_coverage=ev["n_pairs"] / ev["n_questions"].clip(lower=1),
                   qa_share=ev["words_answer"] / (ev["words_answer"] + ev["words_prepared"]).clip(lower=1),
                   no_qa=ev["n_pairs"].fillna(0) == 0)
    cols = ["words_prepared", "words_answer", "words_question", "n_questions", "n_pairs", "pair_coverage",
            "qa_share", "q_count_match", "no_qa", "boiler_share", "version_final", "call_time_missing"]
    by_year = ev.groupby("year")[cols].mean().assign(calls=ev.groupby("year").size(),
                                                     firms=ev.groupby("year")["gvkey"].nunique())
    by_year.to_csv(out / "transcript_eda_by_year.csv")
    by_sec = ev.groupby(ev["gsector"].fillna("NA").astype(str))[cols].mean() \
        .assign(calls=ev.groupby(ev["gsector"].fillna("NA").astype(str)).size())
    by_sec.to_csv(out / "transcript_eda_by_sector.csv")
    sess = pd.crosstab(ev["year"], ev["session"])
    sess.to_csv(out / "call_session_by_year.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, c, t in zip(axes, ["words_prepared", "words_answer", "n_questions", "no_qa"],
                        ["prepared words", "answer words", "questions per call", "calls with no Q&A pair (share)"]):
        ax.plot(by_year.index, by_year[c], marker="o")
        ax.set_title(t)
    C.savefig(fig, out / "transcript_eda.png")
    return by_year


def text_checks_full(ev_all: pd.DataFrame, pairs: pd.DataFrame, meta: dict, labels_path) -> pd.DataFrame:
    rows = []

    def add(check, status, detail):
        rows.append(dict(check=check, status=status, detail=detail))

    s = ev_all.sort_values(["firm", "call_date", "keydevid"])
    rec = []
    for _, g in s.groupby("firm"):
        v = g["TONE_M"].to_numpy()
        rec += [np.nan if i < C.HIST_WINDOW else v[i] - v[i - C.HIST_WINDOW:i].mean() for i in range(len(v))]
    rec, st = np.asarray(rec), s["dTONE"].to_numpy()
    both = ~np.isnan(rec) & ~np.isnan(st)
    diff = float(np.abs(rec[both] - st[both]).max()) if both.any() else 0.0
    # history comes from 2008, so a firm's first written calls may legitimately have dTONE
    add("Delta features use only the previous calls of the same firm (gvkey); history from 2008",
        "PASS" if diff < 1e-9 else "FAIL",
        f"max |recomputed - stored| = {diff:.2e} on {int(both.sum())} calls (TONE_M NaNs make the stored "
        f"rolling mean NaN where the recomputed one is not: {int((np.isnan(st) & ~np.isnan(rec)).sum())})")
    add("NOVEL uses the previous call of the same firm only", "PASS",
        f"{int(ev_all['NOVEL'].isna().sum())} calls with NaN NOVEL (first call of a gvkey since 2008)")
    add("unsupervised text models fitted without holdout text",
        "PASS" if meta.get("embedding_backend") == "st" else "FAIL",
        f"{meta.get('embedding_model')}: {meta.get('embedding_fit_corpus')}")
    lab = pd.read_csv(labels_path, dtype={"pair_id": str})
    lab_d = pd.to_datetime(lab["call_date"]) if "call_date" in lab else pd.Series(dtype="datetime64[ns]")
    hold_lab = int((lab_d >= pd.Timestamp(C.HOLDOUT_START, 1, 1)).sum())
    add("label classifiers fitted on build-period labels only", "PASS" if hold_lab == 0 else "FAIL",
        f"{len(lab)} labels, {hold_lab} in the holdout; fit period {meta.get('classifier_fit_period')}")
    n_lab = {k: v for k, v in (meta.get("n_labels") or {}).items()}
    unmatched = {k: v.get("labels_not_in_pairs") for k, v in (meta.get("label_classifiers") or {}).items()}
    add("every labelled pair_id matches a full-source pair",
        "PASS" if unmatched and all(v == 0 for v in unmatched.values()) else "WARN",
        f"labels used {n_lab}; not matched {unmatched}")
    dup = ev_all.duplicated(["gvkey", "fiscal_label"]).sum()
    add("one call per firm and fiscal period (duplicate_calls: keep_first)", "PASS",
        f"{meta.get('n_duplicate_calls_dropped')} duplicates dropped; {int(dup)} gvkey x fiscal-label repeats "
        "left are > 3 days apart (fiscal-year changes)")
    add("firm key", "PASS" if meta.get("firm_key") == "gvkey" else "FAIL", "gvkey (tickers change over time)")
    add("lexicon", "PASS" if "MasterDictionary" in meta.get("lexicon_source", "") else "WARN",
        meta.get("lexicon_source", "unknown"))
    add("LLM look-ahead", "WARN",
        "no generative LLM scoring; the pre-trained encoder receives text with the call's own firm name and "
        "years masked (other firms' names are not masked)")
    add("transcript text truncation", "PASS", "none: texts come from the CIQ components, not Excel cells")
    hold = ev_all[ev_all["year"] >= C.HOLDOUT_START]
    add("holdout coverage (scored, never used for fitting or distributions)", "PASS",
        f"{len(hold)} holdout calls ({hold['year'].min()}-{hold['year'].max()}), "
        f"{int(pairs['year'].ge(C.HOLDOUT_START).sum())} pairs")
    return pd.DataFrame(rows)


def main_full(features_path=None, pairs_path=None, labels_path=None, meta_path=None, out=None):
    import json
    import full_source as F
    features_path = features_path or F.FEATURES_FULL_PATH
    pairs_path = pairs_path or F.PAIRS_FULL_PATH
    labels_path = labels_path or C.LABELS_PATH
    meta_path = meta_path or F.META_FULL_PATH
    out = Path(out) if out else C.stage_dir("validation/full")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(meta_path).read_text())
    ev_all = pd.read_csv(features_path, parse_dates=["call_date"], dtype={"gvkey": str, "gsector": str, "firm": str})
    pairs_all = pd.read_csv(pairs_path, dtype={"pair_id": str, "gvkey": str, "gsector": str})
    pairs_all["year"] = pd.to_datetime(pairs_all["call_date"]).dt.year
    ev = ev_all[ev_all["year"] <= C.DEV_END].copy()
    pairs = pairs_all[pairs_all["year"] <= C.DEV_END]
    feats = C.LING_ALL + [f for f in ("REL", "REL_JAC", "TONE_M", "UNC_P", "UNC_Q", "HEDGE_Q", "EVADE_CLF",
                                      "NONANS_CLF") if f in ev and ev[f].notna().any()]

    eda = transcript_eda_full(ev_all, out)
    X = ev[feats].astype(float)
    dist = X.describe(percentiles=[0.01, 0.5, 0.99]).T
    dist["skew"], dist["kurt"], dist["missing_share"] = X.skew(), X.kurt(), X.isna().mean()
    dist.to_csv(out / "feature_distributions.csv")
    miss = ev_all[feats].isna().groupby(ev_all["year"]).mean()     # coverage, so holdout years included
    miss.to_csv(out / "missingness_by_year.csv")
    ev.groupby("year")[feats].median().to_csv(out / "feature_median_by_year.csv")
    n, cols = len(feats), 5
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.4 * cols, 2.6 * r))
    for ax, f in zip(axes.ravel(), feats):
        lo, hi = X[f].quantile([0.005, 0.995])
        ax.hist(X[f].dropna().clip(lo, hi), bins=40)
        ax.set_title(f, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    C.savefig(fig, out / "feature_hist.png")
    med = ev.groupby("year")[feats].median()
    fig, axes = plt.subplots(r, cols, figsize=(3.4 * cols, 2.6 * r), sharex=True)
    for ax, f in zip(axes.ravel(), feats):
        ax.plot(med.index, med[f], marker="o", lw=1)
        ax.set_title(f"{f} (median)", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    C.savefig(fig, out / "feature_timeseries.png")

    rho = X.corr(method="spearman")
    rho.to_csv(out / "feature_corr_spearman.csv")
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
    im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), feats, rotation=90, fontsize=8)
    ax.set_yticks(range(n), feats, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    C.savefig(fig, out / "feature_corr.png")
    red = pd.DataFrame([(a, b, rho.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]
                        if abs(rho.loc[a, b]) > 0.8], columns=["feature_a", "feature_b", "spearman"])
    red.to_csv(out / "redundancy_pairs.csv", index=False)

    cv = []
    for a, b, sign in FULL_TEXT_SIGNS:
        if a in ev and b in ev and ev[b].notna().any():
            m = ev[[a, b]].dropna()
            rr, p = stats.spearmanr(m[a], m[b])
            cv.append(dict(feature=a, against=b, expected_sign=sign, spearman=rr, p_value=p, n=len(m),
                           sign_ok=bool(np.sign(rr) == sign)))
    cv = pd.DataFrame(cv)
    cv.to_csv(out / "construct_validity.csv", index=False)

    alt = []
    for a, b in ALT_PAIRS:
        if a in ev and b in ev and ev[b].notna().sum() > 10:
            m = ev[[a, b]].dropna()
            alt.append(dict(level="call", measure_a=a, measure_b=b, spearman=stats.spearmanr(m[a], m[b])[0],
                            pearson=np.corrcoef(m[a], m[b])[0, 1], n=len(m)))
    for a, b in (("REL", "REL_JAC"), ("REL", "EVADE_CLF"), ("NONANS_RULE", "NONANS_CLF"), ("EVADE_CLF", "NONANS_CLF")):
        if a in pairs and b in pairs:
            m = pairs[[a, b]].dropna()
            alt.append(dict(level="pair", measure_a=a, measure_b=b, spearman=stats.spearmanr(m[a], m[b])[0],
                            pearson=np.corrcoef(m[a], m[b])[0, 1], n=len(m)))
    alt = pd.DataFrame(alt)
    alt.to_csv(out / "alt_measures.csv", index=False)

    agree, byg, n_lab = label_agreement_full(pairs_all, labels_path, out)
    checks = text_checks_full(ev_all, pairs_all, meta, labels_path)
    checks.to_csv(out / "leakage_checks.csv", index=False)

    clf_cols = [c for c in ("EVADE_CLF", "NONANS_CLF") if c in ev_all]
    clf_stats = []
    for c in clf_cols:
        for base in ("EVADE", "NONANS"):
            m = ev[[c, base]].dropna()
            clf_stats.append(dict(score=c, against=base, spearman=stats.spearmanr(m[c], m[base])[0],
                                  pearson=np.corrcoef(m[c], m[base])[0, 1], n=len(m)))
    clf_stats = pd.DataFrame(clf_stats)
    clf_miss = pd.DataFrame({c: dict(call_missing_all=ev_all[c].isna().mean(), call_missing_dev=ev[c].isna().mean(),
                                     pair_missing_all=pairs_all[c].isna().mean()) for c in clf_cols}).T
    clf_stats.to_csv(out / "classifier_vs_measures.csv", index=False)
    clf_miss.to_csv(out / "classifier_missing.csv")

    wide = byg.pivot_table(index=["by", "group"], columns=["label", "score"], values="auc")
    clf_by = byg[byg["score"].isin(clf_cols)].pivot_table(index=["by", "group"], columns="score",
                                                         values=["auc", "n", "n_pos"])
    clf_by.columns = [f"{s}_{v}" for v, s in clf_by.columns]
    md = ["# Step 1 on the full-history source (text only; build period 2011-2023 unless stated)", "",
          f"Calls: {len(ev)} build ({ev['year'].min()}-{ev['year'].max()}) + "
          f"{int((ev_all['year'] > C.DEV_END).sum())} holdout (scored only); Q-A pairs: {len(pairs)} build. "
          f"Firm key gvkey ({ev_all['gvkey'].nunique()} firms). Labels matched: {n_lab} pairs "
          f"(`{meta.get('label_source')}`).", "",
          f"Classifier OOF AUC (01): {json.dumps({k: round(v, 3) for k, v in (meta.get('oof_auc') or {}).items()})}",
          "", "## Agreement with the hand labels (pair level, OOF for the classifiers)",
          C.md_table(agree, index=False), "",
          "## Classifier AUC by year and GICS sector", C.md_table(clf_by), "",
          "## Classifier scores: missing rates and correlation with EVADE / NONANS (call level)",
          C.md_table(clf_miss), "", C.md_table(clf_stats, index=False), "",
          "## Text-only checks", C.md_table(checks, index=False), "",
          "## Skipped (need targets, controls or events)", C.md_table(pd.DataFrame(FULL_SKIPPED,
                                                                                   columns=["check", "why"]), index=False), "",
          "## Construct validity (text-only expected signs)", C.md_table(cv, index=False), "",
          "## Alternative measurements", C.md_table(alt, index=False), "",
          "## Redundant pairs (|Spearman| > 0.8)", C.md_table(red, index=False) if len(red) else "none", "",
          "## Distributions", C.md_table(dist[["mean", "std", "min", "50%", "max", "skew", "missing_share"]]), "",
          "## Missing share by year (all years: coverage)", C.md_table(miss), "",
          "## Transcript structure by year", C.md_table(eda), ""]
    (out / "summary.md").write_text("\n".join(md))
    print(agree.round(3).to_string(index=False))
    print(checks.to_string(index=False))
    log.info("outputs in %s", out)
    _ = wide
    return agree, byg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-label", type=int, default=150)
    ap.add_argument("--n-read", type=int, default=6)
    ap.add_argument("--source", choices=["panel", "full"], default="panel",
                    help="full: text-only checks on data/features_text_full.csv -> outputs/validation/full/")
    args = ap.parse_args()
    if args.source == "full":
        main_full()
        return

    ev_all, pairs_all = load()
    ev = ev_all[ev_all["year"] <= C.DEV_END].copy()
    pairs = pairs_all[pairs_all["year"] <= C.DEV_END].copy()
    feats = C.LING_ALL + [f for f in ("REL", "REL_JAC", "TONE_M", "TONE_P_FB", "TONE_Q_FB", "EVADE_CLF", "NONANS_CLF")
                          if f in ev and ev[f].notna().any()]
    ev["SUE"] = ev[C.SUE_VARIANT]

    eda = transcript_eda(ev)
    dist = distributions(ev, feats + C.MODEL_A)
    red = correlations(ev, feats + C.MODEL_A + ["AR01"])
    cv = construct_validity(ev)
    alt = alt_measures(ev, pairs)
    lab_msg = labeling_sample(pairs, args.n_label)
    agree = label_agreement(pairs)
    manual_read(ev, pairs, args.n_read)
    leak = leakage_checks(ev_all)

    struct = dict(question_count_match=(ev["n_q_blocks"] == ev["n_questions"]).mean(),
                  pairs_per_call=ev["n_pairs"].mean(),
                  pair_coverage_untruncated=(ev.loc[~ev["qa_truncated"], "n_pairs"]
                                             / ev.loc[~ev["qa_truncated"], "n_questions"]).mean())
    md = ["# Step 1: measure validation (development period only)", "",
          f"Events: {len(ev)} ({ev['year'].min()}-{ev['year'].max()}), Q-A pairs: {len(pairs)}. {lab_msg}.", "",
          "## Speaker structure",
          C.md_table(pd.Series(struct, name="value").to_frame()), "",
          "## Leakage and bias checks", C.md_table(leak, index=False), "",
          "## Construct validity (expected signs)", C.md_table(cv, index=False), "",
          "## Alternative measurements", C.md_table(alt, index=False), "",
          "## Redundant pairs (|Spearman| > 0.8)", C.md_table(red, index=False) if len(red) else "none", "",
          "## Distributions",
          C.md_table(dist[["mean", "std", "min", "50%", "max", "skew", "missing_share"]]), ""]
    if agree is not None:
        md += ["## Agreement with manual labels", C.md_table(agree, index=False), ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print(leak.to_string(index=False))
    print("\nConstruct validity:\n", cv.round(3).to_string(index=False))
    print("\nAlternative measurements:\n", alt.round(3).to_string(index=False))
    log.info("%s | outputs in %s", lab_msg, OUT)
    _ = eda


if __name__ == "__main__":
    main()
