# Healthcare FinBERT research extension

## Source audit and definition

The extraction universe is the 3,551 distinct Healthcare transcript IDs in the validated financial extension. It spans 2010–2026, with 205 calls in 2024, 214 in 2025 and 193 in 2026. The latter includes panel calls outside the fully observable CAR test. All 3,551 IDs resolve to local Capital IQ component text. No compatible Healthcare FinBERT13 feature cache was present before this work. The original IT archive has sector-specific IDs. The branch's sampled implementation in `scripts/add_finbert_features.py` uses separate definitions and outputs.

The feature definitions come from `9cf2d0e6f410006b20ba9db183698ef251ac7f36:it_finbert_features.py`, in `split_sentences`, `extract_sections`, `aggregate`, `score_bulk` and the final prior-call loop. Upstream text assembly and the section map come from `origin/it-sector-analysis:sp500_pipeline.py` lines 451–468 and 515–531. The work also reviewed that branch's two saved IT Model C benchmark reports. A standalone Software comparison report was not found in the local repository history.

- Preserve existing transcript IDs, component order, and call UTC timestamp. Preferred source is `data/raw/ciq_transcript_component_<year>.parquet`. The team PIT component shards are a fallback by exact transcript ID.
- Use component 2 for prepared remarks. Reproduce the IT block parser for component 4 answers, excluding component 3 question blocks. Blank internal paragraphs in the historical representation can lead to omitted answer text. The coverage audit counts these parser boundaries.
- Split after a period, exclamation point or question mark followed by whitespace. Keep sentences at least ten characters long, cap at 512 characters and apply the original uncased BERT WordPiece tokenizer, limited to 512 tokens including special tokens. Record sentence-level character and token truncation. There is no whole-section sentence cap.
- Use `ProsusAI/finbert` revision `db38d3727cbaed87c9aed72df7b3519e2ba5cca1`. The original IT code uses only the highest-probability label. The signed sentiment per sentence is +confidence for positive, −confidence for negative and zero for neutral. It differs from subtracting negative posterior probability from positive posterior probability. Positive and negative fractions are top-label counts divided by total sentences. The standard deviation is population SD. An observed section with no retained sentences gives zero counts and null sentiment, fractions and SD. A single retained sentence gives SD zero.
- The FinBERT gap is prepared mean minus answer mean. Each change is the current section mean minus the observed average among the four immediately preceding calls with the same PERMNO. Within the current archive, the feature calculation checks strict UTC ordering. Calls with equal timestamps never serve as one another's history. Missing values among the four positions are ignored, and a missing prior mean gives a missing change.

The extractor runs FP32 on the available Intel GPU through OpenVINO. The original IT script names the model without pinning a revision. This run pins the revised model files and validates the offline implementation against PyTorch Transformers. The 17 verification examples match the reference top label, with a maximum probability difference of 1.49 × 10⁻⁶. The model publisher lists the weights under CC BY-NC-SA 4.0. [ProsusAI FinBERT model card](https://huggingface.co/ProsusAI/finbert)

## Local store and incremental operation

Everything under `data/derived/finbert/` is ignored by Git. The archive retains `call_features.parquet`, `sentence_scores.parquet`, `coverage_summary.csv`, `feature_manifest.json` and the checksummed inference ledger `inference.sqlite3`. The sentence Parquet stores transcript ID, section, sentence index, hash, top label, confidence and truncation fields. It contains no transcript text. The ledger keys sentence inference by a hash of the retained character input, and validates calls by exact input-component digest. A checkpoint commits the sentence scores and call membership together. Repeated sentences and validated calls are reused. Incremental publication recalculates prior changes over the registered calls, which also handles ingesting an earlier backfill. Local archive files are intentionally preserved.

The checkpoint, package runtime and input text remain on the machine during inference. Any public checkpoint or package retrieval is separate from the scoring process. The supplied scripts have no upload, model endpoint, sync, Git staging or external communication step. The package and model subdirectories are within the same ignored archive.

To run from the repository root with the local dependencies and checkpoint already present:

```powershell
$work = 'outputs/healthcare_unified_research'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe "$work/extract_finbert.py"
.\.venv\Scripts\python.exe "$work/extract_finbert.py" --verify-only
.\.venv\Scripts\python.exe "$work/fit_finbert.py"
.\.venv\Scripts\python.exe "$work/verify_unified.py"
.\.venv\Scripts\python.exe "$work/render_unified.py"
```

`--max-calls 0` publishes existing completed checkpoints without loading the model. `--verify-only` reloads Parquet metadata and call features, verifies digests and reconciles stored sentence counts without inference. `validate_feature_store.py` also reads the saved sentence values and independently recomputes all 13 features. For expansion beyond Healthcare, the extractor accepts a local `--events` CSV with transcript ID, PERMNO, UTC clock and sector plus one or more `--sector` flags. The SQLite archive preserves previously validated calls and coverage by source universe and sector. Use the canonical issuer panel when mixing sectors so each transcript identity has the same PERMNO and clock definitions.

## Model comparison and release checks

C/C* use the exact A/B/B* events, CAR targets, model A controls, maturity cutoffs, folds, estimator grid and fold-local winsorization, imputation and scaling. Model C uses one greedy forward path, with a strictly greater than 0.001 mean validation Rank IC improvement required at each step. Empty selection reuses the stored A prediction. C* forces all thirteen FinBERT predictors. Original B may choose a forward path or a combination from the univariate screen, as recorded by that experiment. Original A/B/B* are preserved as executed. The run refuses a partial FinBERT archive.

Annual test settings retain the original eligible event cohort for all five model scores. Native coverage and a fully observed sensitivity cohort are available separately. Long and short models are compared only on the respective annual intersection and retain their own realized targets. Performance uses raw Spearman Rank IC, Pearson IC and realized event quintiles. Paired issuer-cluster and two-month moving block intervals apply the frozen minimum observations and periods. Partial 2026 provides insufficient calendar blocks for time dependence estimates. Event timing and the overnight part of the first close-to-close return follow the earlier research restrictions.

All source A/B/B* files are checksummed around the new fit and prediction verification. The report renderer requires the full grid of new fits and independent aggregate verification. It reads aggregate data only. The local event predictions and fitted-transform Parquet are excluded from version control. Visual release checks are required on the completed report and dashboard.
