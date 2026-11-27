# Handoff: a stronger relevance signal for F3 (agent G)

Can start now. Read `docs/handoff_next_steps.md` (rules for every agent), agent C's brief
`docs/labeling_handoff_calibration.md` and `docs/pipeline.md` ("Q&A labels for F3", Models).

## Where F3 stands
On the 414 build-period labels, out-of-fold (GroupKFold by call):

| Score | evasive AUC [95% CI] | non_answer AUC |
|---|---|---|
| EVADE_CLF | 0.629 [0.553, 0.704] | – |
| NONANS_CLF | – | 0.674 [0.588, 0.756] |
| NONANS_RULE | 0.548 | 0.666 |
| REL_JAC | 0.576 | 0.587 |
| REL (MiniLM cosine) | 0.530 | 0.497 |

The bar is 0.7. NONANS_CLF mostly reproduces NONANS_RULE (Spearman 0.95). EVADE_CLF has
Spearman −0.25 with EVADE (1 − mean REL), so the bi-encoder cosine does not track labelled evasiveness.

## Goal
Find a pair-level score that reaches OOF AUC ≥ 0.7 on `evasive` and on `non_answer`, using only
build-period labels, and say which one goes forward.

## Inputs (read-only)
- `data/qa_labels.csv` (A's labels, as they are: B's resolutions are not signed off)
- `data/qa_pairs_full.csv`, `data/features_text_full.csv`, `data/interim/qa_pairs.parquet`
- `data/interim/emb_cache/` (MiniLM embeddings, reuse them)
- `scripts/full_source.py` (current features and classifiers, for reference)

## Candidates (test each one on its own, then the best combination)
1. **Richer pair features:** question type (yes/no, number, guidance, multi-part), number of
   sub-questions, answer specificity (numbers, units, dates, named entities), refusal and deferral
   cues, answer length relative to the question.
2. **Cross-encoder or NLI relevance** in place of the bi-encoder cosine (for example an MS MARCO
   cross-encoder, or an NLI model scoring "the answer addresses the question"). Cache the scores
   in `data/interim/f3_cache/`.
3. **LLM judge** on anonymised text (firm names and years masked, as in `scripts/full_source.py`),
   with the label rules from `docs/labeling_handoff.md` as the prompt. Run it on the 414 labelled
   pairs only, and estimate the cost of scoring all 654,447 pairs before scaling up. Do not
   scale up without the user's approval.

## Method
- Same OOF scheme as agent C: GroupKFold by call, bootstrap CIs by call. Report AUC, average
  precision, and the CI for each candidate.
- No holdout calls (2024-01-01 on), no tuning on the labels beyond the inner folds.
- Report by year and by sector only as a sanity check: the cells are small (30–39 pairs).
- If no candidate reaches 0.7, say how many more labels would narrow the CI enough to tell 0.65
  from 0.7, and draft a sample plan for them (do not label).

## Writes
- `scripts/f3/` (helper scripts), `tests/test_f3_*.py`, `outputs/validation/f3_relevance/`
  (results tables and a `summary.md`), `data/interim/f3_cache/`.
- Do not edit `scripts/full_source.py`, `scripts/01_build_features.py` or `src/`. If a candidate
  passes, describe the change to `01` in the summary; it is wired in by a later brief.

## Report back
Commit hash, the results table (one row per candidate), the recommended score, its cost to
compute on the full sample, and the label-count estimate if nothing passes.
