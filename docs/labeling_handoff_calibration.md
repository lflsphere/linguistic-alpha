# Handoff: calibrate F3 on the labels and rerun 01 / 02 (agent C)

Starts only after labeller A (`docs/labeling_handoff.md`) and labeller B
(`docs/labeling_handoff_teammate.md`) have both landed. Working directory: `linguistic-alpha/`.
Python: `.venv/bin/python`. The label scheme is in `docs/pipeline.md` ("Q&A labels for F3").

## Step 0: gate (stop and report if any check fails)

| Check | Pass when |
|---|---|
| `data/qa_label_sample_full.csv` | about 450 rows, all calls 2011-01-01 – 2023-12-31, none in the holdout |
| `data/qa_labels.csv` (A) | same `pair_id`s as the sample; `evasive`, `non_answer` integer 0/1 with no blanks; at least 15 ones per label |
| `data/qa_labels_b.csv` (B) | 90 `pair_id`s, all in the sample; same format |
| `outputs/validation/label_interrater.csv` | exists; kappa per label |
| `outputs/validation/label_disagreements.csv` | exists |

- **Kappa below 0.4 on a label:** do not calibrate that label. Report it, so the rules are
  reworded first.
- **Kappa between 0.4 and 0.6:** go ahead, but flag the result as provisional.
- **Resolved labels:** use labeller A's labels as they are. Do not apply B's proposed
  resolutions unless the team has signed off on them. Report how many pairs that would change.

## What changes
`scripts/01_build_features.py` builds its pairs from the legacy panel (`data/panel.csv`,
179 events, 3 firms; that file is not in `data/` now either). This brief makes `01` read the
**full-history pairs** that labeller A's step 0 wrote with `la transcripts`
(`data/interim/qa_pairs.parquet`). The labelled `pair_id`s then match directly, and the
existing `label_classifiers()` merge works with no side file.

## Step 1: add `--source full` to 01 (keep `--source panel` as the default)
1. **Inputs**:
   - `data/interim/{transcript_calls,qa_pairs,call_sections}.parquet`
   - `data/raw/sp500_pit_2008/` (`ciq_wrds_gvkey`, `comp_co_hgic`, `crsp_ccmxpf_lnkhist`,
     `crsp_msp500list_v2`)
   - Config: `configs/full.yaml`
2. **Call table** (replaces `panel.csv`): one row per `keydevid`, earnings calls only, with
   - `transcriptid, keydevid, companyid, companyname, gvkey, permno, gsector`
   - `call_date` and `call_time_utc` (from `mostimportantdateutc` / `mostimportanttimeutc`)
   - `version_final`
   - `prepared_text` and `qa_text` from `call_sections`
   
   Keep only S&P 500 members on the call date, using date-valid links: `ciq_wrds_gvkey` (companyid → gvkey), then `crsp_ccmxpf_lnkhist` (gvkey → permno, `linktype` LU/LC, `linkprim` P/C, call date within `linkdt`–`linkenddt`), then `crsp_msp500list_v2` (permno a member on the call date): the same rule as labeller A's step 1. Keep biotech (team decision
   2026-11-17). Log the calls dropped at each step to `outputs/validation/full_source_log.md`.
3. **Pairs**: take them from `qa_pairs.parquet`; do not re-parse `qa_text` with `parse_qa`.
   - Use `pair_id = f"{transcriptid}_{pair_id:03d}"`, the same key labeller A used.
   - Compute the pair features (`HEDGE_A`, `NONANS_RULE`, lengths, `REL_JAC`, `REL`) with the
     existing code path.
   - There is no Excel truncation, so set `qa_truncated` / `prep_truncated` to False.
4. **Firm key**: use `gvkey` in place of `ticker` for `firm`, the F4 history and `NOVEL`.
   Tickers change over time; gvkeys don't.
5. **Look-ahead**:
   - Fit the embedder only on texts from calls up to 2023-12-31.
   - `label_classifiers` sees only build-period labels; it scores every pair, holdout included.
   - Use `--embed st` (sentence-transformers, `C.ST_MODEL`). If it is unavailable, stop:
     the TF-IDF fallback depends on which texts it was fitted on.
6. **Targets and controls**: `base_table` needs CAR windows and Model A controls, but the
   `la targets` / `la controls` stages are still outlines. In `--source full`, write the text
   features only, keyed by `transcriptid` / `keydevid`, with `day0` left empty, to
   `data/features_text_full.csv` and `data/qa_pairs_full.csv`. Do not overwrite
   `features.csv`; the scripts that read it need targets.
7. **Scale**: about 50,000 calls and 800,000 pairs. Embed in batches; cache the embeddings
   by `pair_id` in `data/interim/emb_cache/`; log progress; run it in the background.
8. **Metadata**: record `source`, `label_source`, `n_labels` and `oof_auc` in
   `data/features_text_full_meta.json`, so the legacy `features_meta.json` stays as it is.

## Step 2: 02 on the full source
- Add `--source full` to `02_validate.py`. Run the label agreement on the full pairs, with the
  AUC by year and by GICS sector, plus the checks that need only text: missing rates by year,
  feature distributions and correlations, and the transcript EDA.
- Skip checks that need targets or controls, and list them as skipped in `summary.md`.
- Write to `outputs/validation/full/` so the legacy outputs are kept.
- Add tests in `tests/`: `--source full` on a 3-call synthetic interim set gives the expected
  `pair_id`s, and the labels merge. Run `pytest`.

## Step 3: run
```bash
.venv/bin/python scripts/01_build_features.py --source full --embed st
.venv/bin/python scripts/02_validate.py --source full
```
Do not run `03`–`05` or `--freeze`, and do not touch the holdout beyond scoring it.

## Report back
- Gate results, including kappa per label and the number of pairs B's resolutions would change.
- `EVADE_CLF` / `NONANS_CLF` out-of-fold AUC, overall and by year and sector. F3 passes at
  about 0.7 or more.
- How the classifier scores compare with `REL`, `REL_JAC` and `NONANS_RULE` (from
  `outputs/validation/full/label_agreement.csv`).
- In `features_text_full.csv`: missing rates of `EVADE_CLF` and `NONANS_CLF`, and their
  correlation with `EVADE` and `NONANS`.
- Calls dropped at each step of the call table, the final counts by year and sector, and the
  run time.
- What is still needed for the full source to feed `03`–`05`: the `la events`, `targets` and
  `controls` stages.
