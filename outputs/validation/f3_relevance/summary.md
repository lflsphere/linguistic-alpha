# F3 relevance search (agent G), 2026-11-27

Labels: A's 414 build-period pairs (`data/qa_labels.csv`, as written; B's resolutions not applied), one pair per
call: 61 evasive, 57 non_answer. No holdout call (2024-01-01 on) was read. Texts anonymised as in
`scripts/full_source.py` (own firm name, calendar years). Method: OOF predictions, `GroupKFold(5, shuffle)` by
`transcriptid`; any regularisation C is picked by inner GroupKFold AUC on the training part only (grid in
`scripts/f3/f3common.py`, choices in `chosen_C.csv`). 95% CIs: 2,000 call-bootstrap draws on the seed-0 split.
`mean10 / min10 / max10`: AUC over 10 different fold assignments (the split alone moves AUC by about ±0.04).

## Results (one row per candidate; bar = 0.70)

| Candidate | evasive AUC [95% CI] | AP | mean10 (min–max) | non_answer AUC [95% CI] | AP | mean10 (min–max) |
|---|---|---|---|---|---|---|
| ref EVADE_CLF / NONANS_CLF (agent C) | 0.629 [0.546, 0.705] | 0.24 | – | 0.674 [0.588, 0.756] | 0.45 | – |
| C0 base 7 features, inner-CV C | 0.602 [0.522, 0.680] | 0.20 | 0.617 (0.59–0.64) | 0.647 [0.557, 0.734] | 0.39 | 0.655 (0.62–0.69) |
| **C1 rich pair features** (base + 33 rule features) | 0.671 [0.598, 0.743] | 0.31 | 0.665 (0.62–0.71) | **0.774 [0.694, 0.849]** | 0.46 | 0.767 (0.73–0.79) |
| C2 cross-encoder + NLI scores (base + 31) | 0.609 [0.532, 0.687] | 0.26 | 0.621 (0.60–0.66) | 0.757 [0.667, 0.841] | 0.49 | 0.765 (0.73–0.78) |
| C1+C2 combined | 0.619 [0.541, 0.698] | 0.27 | 0.647 (0.60–0.70) | **0.821 [0.750, 0.886]** | 0.49 | 0.822 (0.79–0.85) |
| **C3 LLM judge, Haiku 5.5** (zero-shot, anonymised) § | **0.847 [0.797, 0.892]** | 0.46 | – (nothing fitted) | **0.952 [0.922, 0.975]** | 0.77 | – |
| rank-mean(C3, E1) § | 0.857 [0.808, 0.901] | 0.55 | – | – | | |
| rank-mean(C3, C1) § | – | | | 0.920 [0.881, 0.951] | 0.75 | – |
| T1 TF-IDF answer logit | 0.617 [0.525, 0.700] | 0.28 | 0.618 (0.50–0.69) | 0.648 [0.567, 0.726] | 0.34 | 0.697 (0.64–0.76) |
| T2 TF-IDF question‖answer logit | 0.570 [0.488, 0.654] | 0.23 | 0.603 (0.55–0.69) | 0.709 [0.634, 0.782] | 0.33 | 0.686 (0.65–0.72) |
| T3 MiniLM vectors [Q, A, Q·A, \|Q−A\|] logit | 0.708 [0.637, 0.775] | 0.27 | 0.723 (0.67–0.77) | 0.611 [0.525, 0.693] | 0.21 | 0.630 (0.59–0.68) |
| **E1 rank-mean(C1, T3)** † | **0.735 [0.667, 0.799]** | 0.31 | 0.742 (0.72–0.78) | 0.746 [0.665, 0.818] | 0.41 | 0.750 (0.73–0.77) |
| E2 rank-mean(C1+C2, T3) † | 0.715 [0.640, 0.785] | 0.30 | 0.736 (0.70–0.78) | 0.775 [0.698, 0.843] | 0.47 | 0.788 (0.75–0.82) |
| E3 rank-mean(C1, T1, T3) † | 0.721 [0.640, 0.791] | 0.35 | 0.732 (0.67–0.78) | 0.741 [0.660, 0.811] | 0.40 | 0.769 (0.74–0.80) |
| L1 rich + NLI decline/number ‡ | 0.693 [0.622, 0.759] | 0.35 | 0.674 (0.62–0.72) | 0.798 [0.717, 0.871] | 0.52 | 0.797 (0.76–0.82) |
| E4 rank-mean(L1, T3) ‡ | 0.754 [0.689, 0.815] | 0.31 | 0.748 (0.71–0.78) | 0.763 [0.684, 0.834] | 0.43 | 0.770 (0.75–0.80) |

† Equal-weight mean of OOF percentile ranks, nothing fitted; the combinations were fixed after the first run
had shown C1 and T3 (so chosen with knowledge of the 414 labels). ‡ Added after the second run (post hoc).
§ Run later on the user's request (see "LLM judge" below); the judge is not fitted, so it needs no CV.
About 15 candidates were tried on the same 414 labels, so the best AUC is optimistic: treat 0.70–0.75 as
"probably near 0.7", not as a clear pass.

Unfitted single scores (`raw_scores.csv`, sign fixed a priori):
- Zero-shot NLI "The speaker declines to give the requested information." (mean over 80-word answer windows):
  **non_answer 0.804 [0.743, 0.855]**, with nothing fitted. The widened refusal regex `a_refuse` gives 0.749.
- Relevance does not track either label. Cross-encoders (MS MARCO MiniLM, BGE reranker, QNLI-electra), with
  low relevance taken as positive, give evasive AUC 0.49–0.56 and non_answer 0.49–0.58, no better than the
  MiniLM cosine (0.53 / 0.50). Answer specificity carries more: `-a_spec_p100` gives evasive 0.654, and NLI
  "gives a specific number" (negated) gives 0.660.

By year and by sector (`by_year.csv`, `by_sector.csv`; 30–39 pairs and 0–12 positives per cell, a sanity check
only). E1/E4 evasive: 0.60–0.90 by sector; utilities (55) is 0.49 with 4 positives. 2023 has no evasive positive.
C1+C2 non_answer: 0.72–1.00 by year except 2023 (0.08, with 3 positives). Look at that cell when the
labels grow.

## LLM judge (C3), run 2026-11-27 on the user's request
`scripts/f3/llm_judge.py run --backend cli` makes one `claude -p --model claude-haiku-5-5 --effort low` call
per pair, through the user's logged-in Claude CLI (no API key handled). The system prompt is A's rules from
`docs/labeling_handoff.md`; the judge sees only the anonymised Q and A, with tools and project context off.
Answers are in `data/interim/f3_cache/llm_judge_claude-haiku-5-5.jsonl`: 414/414 parsed, after one retry of 7
failed calls. `scripts/f3/eval_judge.py` scores them into `llm_judge_results_claude-haiku-5-5.csv`.

- evasive: `p_evasive` AUC **0.847 [0.797, 0.892]**, AP 0.46. non_answer: `p_non_answer` AUC **0.952
  [0.922, 0.975]**, AP 0.77. Both lower CI bounds clear 0.7 by a wide margin, and nothing was fitted or tuned.
- Rank-mean with E1 adds little on evasive (0.857), and adding C1 makes non_answer worse (0.920): the judge
  alone is enough.
- Calibration: `p_evasive` is far too high (mean 0.52 against a base rate of 0.147; 58% of pairs ≥ 0.5), so
  use it as a ranking or recalibrate it on the build period. `p_non_answer` is about right (mean 0.16 against
  0.138).
- **Main caveat: A's labels were written by a model** (`docs/labeling_handoff.md`, "Report back"), and the
  judge prompt is A's own rulebook. Part of the 0.85 / 0.95 is therefore model-to-model agreement, not
  agreement with a human reader. Score the judge against B's labels before F3 enters a reported model. The
  judge saw anonymised text while A saw raw text, which limits but does not remove this shared bias.
- Measured cost: $0.169 for the 414 pairs (CLI-reported list price), 1,739 input and 129 output tokens per
  pair. The input count includes the CLI's own overhead, so a direct API call should use fewer tokens.

## Recommendation (feature-based; see the update below for the LLM judge)
- **NONANS_CLF → C1** (logit on the 7 current pair features plus the 33 rule features of
  `scripts/f3/pair_features.py`, C = 0.01–0.03): 0.774 [0.694, 0.849]. It was a pre-specified candidate and
  costs about 20 single-core minutes on all 654,447 pairs (557 pairs/s, regex only). It is better than C1+C2
  (0.821) per unit of cost. Upgrade path: L1 (0.798) adds two NLI hypotheses, costing about 18 h on this Mac.
- **EVADE_CLF → E1** = mean of the build-ECDF percentile ranks of C1(evasive) and T3 (L2 logit on the cached
  MiniLM pair vectors, C ≈ 0.001–0.01): 0.735 [0.667, 0.799]; mean over 10 splits 0.742 (0.72–0.78). It needs
  no new model: T3 reads the embeddings `01` already caches. **Caveat:** the lower CI bound is below 0.7, and
  E1 was chosen after looking at the labels. Confirm it on fresh labels (plan below) before F3 enters a reported
  model. E4 (0.754) is better but post hoc and costs about 18 h of NLI.

**Update after the judge run.** If the user approves the spend, the Haiku judge is the best F3 score for
both labels (0.847 / 0.952, unfitted). The likely full-sample cost is about $87 at API list price ($43 via
the Batches API), or at most $156 / $78 at the token counts measured through the CLI. Check it against B's
labels first. Until then, C1 / E1 above remain the free fallback. The judge has not been scaled up, and doing
so needs approval.

## Cost on the full sample (654,447 pairs)
| Score | How | Time / cost |
|---|---|---|
| C1 rich features | regex, CPU | ~20 min single core (minutes with the `01` worker pool) |
| T3 | logit on the cached MiniLM vectors (`data/interim/emb_cache/`) | seconds (no encoding) |
| NLI, 1 hypothesis (DeBERTa-v3-base) | 1.78 M windows at 54.5 windows/s (MPS) | ~9 h per hypothesis; L1 needs 2 (~18 h) |
| MS MARCO MiniLM / BGE / QNLI-electra | 0.94 M windows at 178 / 28.5 / 29.9 windows/s | 1.5 h / 9.2 h / 8.8 h |
| Full C2 (5 answer + 3 pair hypotheses + 3 rankers) | | ~100 h on this Mac |
| LLM judge (estimate ~576 input / 150 output tokens per pair; measured through the CLI 1,739 / 129, CLI overhead included, so at most $156 / $78 via the Batches API for Haiku) | Haiku 5.5 / Sonnet 5.5 / Opus 5.5 | $87 / $1,736 / $3,472 ($43 / $868 / $1,736 via the Batches API); the 414 labelled pairs cost $0.05–$2.20 |

Times were measured one model at a time (`throughput_clean.json`); the first BGE timing (31 min for 677
windows) was memory pressure from other jobs. Token counts are words × 1.35; check them with
`messages.count_tokens` before any run. The LLM judge was called on the 414 labelled pairs only ($0.17); scaling up needs the user's approval.

## Label count: telling 0.65 from 0.70 (`label_count.csv`; nothing labelled)
| Target | Criterion | Total labels needed | More than now |
|---|---|---|---|
| evasive | 95% CI half-width 0.05 | 730–1,000 | +310–590 |
| evasive | 0.65 vs 0.70, one-sided 5%, 80% power | 1,170–1,600 | +750–1,190 |
| non_answer | 95% CI half-width 0.05 | 990–1,060 | +580–650 |
| non_answer | 0.65 vs 0.70, one-sided 5%, 80% power | 1,590–1,700 | +1,180–1,290 |
(The range runs from the observed call-bootstrap SE scaled by √(414/n) to the Hanley–McNeil formula at AUC 0.70.)

Draft sample plan (for whoever labels, not done here):
1. Same frame as `docs/labeling_handoff.md` step 1: build-period S&P 500 earnings calls (2011-01-01 to
   2023-12-31), drop the 414 labelled calls, drop empty answers. Seed 43.
2. About 600 new pairs (target total ~1,000, enough for a ±0.05 CI on both labels). Stratify by year (13) ×
   sector (11), about 4–5 per cell, at most 1 call per firm per year and 1 pair per call, so calls stay
   independent for the bootstrap.
3. No enrichment by model score, so the scores stay honestly evaluable without weights. If positives look too
   few (the expected count is ~90 evasive and ~80 non_answer), add an enriched stratum drawn on `a_refuse` or
   `NONANS_RULE`, and report AUCs with inverse-probability weights.
4. Same rules and labeller as A (or A and B, with kappa). Freeze the scores above (C1, E1, and L1 as a
   secondary) before labelling, then evaluate them once on the new pairs as a confirmation set, separate from
   the 414.

## Change to `01` (later brief; not done here)
In `_call_worker` / `lexical_features` of `scripts/full_source.py`: add `pair_features.pair_features(q_anon,
a_anon)` to each pair row. In `label_classifiers_full`:
(a) NONANS_CLF = logit on `PAIR_CLF_FEATS + pair_features.RICH`, with C chosen by inner GroupKFold;
(b) EVADE_CLF = mean of two percentile ranks: the C1 logit for `evasive`, and an L2 logit on the
`[Q, A, Q*A, |Q-A|]` unit vectors from the pair embedding cache. Use ranks against the build-period pairs'
ECDF, so holdout pairs do not move the scale. Store `f3_version` in the meta JSON. The labelled pairs keep OOF
values, as now.

## Files
`scripts/f3/`: `f3common.py` (data, anonymisation, CV, bootstrap), `pair_features.py`, `cross_scores.py`,
`text_models.py`, `llm_judge.py` (prompt, `estimate`, `run` via the SDK or `--backend cli`), `eval_judge.py`, `evaluate.py`,
`time_models.py`, `label_plan.py`. Outputs here: `results.csv`, `raw_scores.csv`, `oof_scores.csv`,
`by_year.csv`, `by_sector.csv`, `coefficients.csv`, `chosen_C.csv`, `label_count.csv`. Cache:
`data/interim/f3_cache/` (labelled pairs with anonymised text, cross scores, embeddings of the 414 pairs,
throughput). Tests: `tests/test_f3_relevance.py`.
Note: for non_answer the inner CV often picks C at the bottom of the grid (0.01). A lower grid might add a
little.
