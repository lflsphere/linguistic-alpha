# Handoff: label Q&A pairs for evasiveness (labeller A)

Working directory: `linguistic-alpha/`. Python: `.venv/bin/python`. Do not edit any code
under `src/` or `scripts/`. Put any helper script you write in `scripts/labeling/`.
The label scheme is specified in `docs/pipeline.md` (section "Q&A labels for F3").

## Why a sample, not the entire history
The labels train and validate F3 (`evasive`, `non_answer`). The classifier then scores every
pair. The full history is about 50,000 calls × about 16 pairs (≈ 800,000 pairs), too many to
label. The sample therefore **spans the whole build period**: about 450 pairs from the
S&P 500, 2011 Q1 – 2023 Q4. The legacy 150-pair sample (`data/qa_label_sample.csv`,
3 firms) is no longer used. **Never sample from the holdout (calls after 2023-12-31).**

## Data files

| What | Path |
|---|---|
| Raw pull | `data/raw/sp500_pit_2008/` (read `README.md` and `manifest.csv` first) |
| Calls and versions | `ciq_transcript_detail.parquet`: `companyid, keydevid, transcriptid, mostimportantdateutc, mostimportanttimeutc, keydeveventtypename, transcriptpresentationtypename, transcriptcreationdate_utc, transcriptcreationtime_utc` |
| Transcript text | `ciq_transcript_component_<yyyy>.parquet`: `transcriptid, transcriptcomponentid, componentorder, transcriptcomponenttypeid, transcriptpersonid, componenttext` |
| Speakers | `ciq_transcript_person_<yyyy>.parquet`: `transcriptid, transcriptcomponentid, transcriptpersonid, speakertypename, companyofperson` |
| Firm → gvkey | `ciq_wrds_gvkey.parquet` (`companyid, gvkey, startdate, enddate, primaryflag`) |
| Sector | `comp_co_hgic.parquet` (`gvkey, gsector, gind, indfrom, indthru`) |
| Index membership | `crsp_msp500list_v2.parquet`, linked via `crsp_ccmxpf_lnkhist.parquet` |
| Parsed pairs (after step 0) | `data/interim/qa_pairs.parquet` (`transcriptid, pair_id, question_speaker_id, question, answer, n_answer_turns`), `data/interim/transcript_calls.parquet` |

## Step 1: build the sample (before any labelling)
0. **Build the full-history pairs once, with the pipeline's own stage**, so `01` later reads the
   same pairs and `pair_id`s. Do not parse the text yourself.
   - Move the legacy interim files aside first:
     `mkdir -p data/interim/legacy && mv data/interim/*.parquet data/interim/legacy/`.
   - Use `configs/full.yaml`: an override of `configs/default.yaml` that sets only
     `paths.raw: data/raw/sp500_pit_2008` (everything else comes from the default).
   - Run `la transcripts --config configs/full.yaml`. It applies the spec's version rule (the
     latest version created by the call session's close) and writes
     `data/interim/{transcript_calls,components,qa_pairs,call_sections}.parquet`.
1. From `data/interim/qa_pairs.parquet` joined to `transcript_calls`, keep earnings calls only
   (`keydeveventtypename` = earnings call), with `mostimportantdateutc` from 2011-01-01 to
   2023-12-31.
2. Keep biotech (team decision 2026-11-17). Keep only S&P 500 members on the call date, using date-valid links: `ciq_wrds_gvkey` (companyid → gvkey), then `crsp_ccmxpf_lnkhist` (gvkey → permno, `linktype` LU/LC, `linkprim` P/C, call date within `linkdt`–`linkenddt`), then `crsp_msp500list_v2` (permno a member on the call date). Agent C uses the
   same rule, so the sample is a subset of its call table. Log how many calls each step drops.
3. Drop pairs with an empty answer (`n_answer_turns` = 0).
4. Stratify by year (13) × GICS sector (11). Draw about 3 pairs per cell (≈ 450 in total),
   with at most 1 call per firm per year and at most 2 pairs per call. Use seed `42`.
5. Write `data/qa_label_sample_full.csv` with the columns `pair_id` (`<transcriptid>_<kkk>`, where `kkk` is the interim `qa_pairs.pair_id`, zero-padded to 3 digits; `01 --source full` uses the same key),
   `transcriptid, keydevid, gvkey, gsector, call_date, question, answer, evasive, non_answer, notes`,
   plus a sampling log at `outputs/validation/label_sample_full_log.md`.

## Labelling rules
Full scheme: `docs/pipeline.md`.
- `evasive = 1` when the answer does not give the specific information the analyst asked for:
  it changes the topic, falls back on generic talking points, answers an easier question, or
  covers only part of a multi-part question while skipping the main ask.
- `non_answer = 1` when management explicitly declines or cannot answer: "we don't disclose",
  "too early to say", "we don't guide to that", "we'll follow up offline".
- A refusal is usually both labels. An answer can be evasive without being a refusal.
- `evasive = 0` when the answer engages with the question, even if the news is bad or the
  answer is brief.
- Judge each pair only from its question and answer. Do not look at returns, CARs, features,
  relevance scores or any model output.
- Text may be cut off: judge only what is there and write `truncated` in `notes`.
- In `notes`, give a one-line reason for every 1 and for every borderline 0. Start a borderline
  note with `borderline:`.

## Step 2: label
1. Label one pair at a time, in `pair_id` order. Never label in bulk by keyword.
2. Save `data/qa_labels.csv` with the same columns as the sample, integer 0/1 values and no
   blanks.
3. Second pass: reread every `borderline:` row and every 1, and count the changes.
4. If either label has fewer than 15 ones, report it; do not invent positives.
5. **Do not open the teammate's files** (`data/qa_labels_b*.csv`).
6. **Do not run `scripts/01_build_features.py` or `02_validate.py`.** Agent C
   (`docs/labeling_handoff_calibration.md`) adds `--source full` to them and runs them after
   labeller B has finished.

## Report back
- Calls dropped at each sampling step, and the final counts by year and sector.
- Base rates (overall, by year, by sector), the number of borderline rows, and the number of
  second-pass changes.
- 3–5 of the hardest cases and how you decided them.
- Caveat: these are labels written by a model. Agreement is reported only after labeller B
  (`docs/labeling_handoff_teammate.md`) has finished.
