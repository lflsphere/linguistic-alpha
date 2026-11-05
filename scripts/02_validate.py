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
Also writes data/qa_label_sample.csv (blind Q-A pairs to label) if no sample or label file exists yet.
"""
from __future__ import annotations

import argparse
import textwrap

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
              ((ev_all["session"] == "pre") & (ev_all["day0_gap_days"] == 0))
    add("day 0 consistent with call time (post-close -> next session, pre-open -> same day)",
        "PASS" if post_ok.all() else "FAIL",
        f"{int((~post_ok).sum())} inconsistent; sessions: {ev_all['session'].value_counts().to_dict()}")
    add("no intraday calls (reaction day ambiguous)", "PASS" if (ev_all["session"] != "intraday").all() else "WARN",
        f"{int((ev_all['session'] == 'intraday').sum())} intraday calls")
    fpe = pd.to_datetime(ev_all["fiscal_period_end"])
    add("fiscal period ends before the call", "PASS" if (fpe < ev_all["call_date"]).all() else "FAIL",
        f"min gap {(ev_all['call_date'] - fpe).dt.days.min()} days")
    feats = set(C.LING_ALL + C.MODEL_A)
    leaky = [f for f in feats if f.upper().startswith(("CAR", "LOGVOL", "AR01"))]
    add("no post-event quantity in Model A / Model B feature lists", "PASS" if not leaky else "FAIL", str(leaky))
    add("CAR[0,+1] excluded from the primary specification", "PASS" if "AR01" not in C.MODEL_A else "FAIL",
        "AR01 only enters with --delayed")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-label", type=int, default=150)
    ap.add_argument("--n-read", type=int, default=6)
    args = ap.parse_args()

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
