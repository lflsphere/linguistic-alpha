# linguistic-alpha

Research pipeline: do earnings-call language features predict S&P 500 post-call drift,
CAR[+2,+20] and other horizons, beyond the earnings surprise and standard controls?
Data comes from WRDS: Capital IQ transcripts, CRSP, Compustat and I/B/E/S.

- Architecture and stage contracts: [docs/pipeline.md](docs/pipeline.md)
- Legacy notebook audit and rebuild plan: [docs/data_audit.md](docs/data_audit.md)
- Run settings: [configs/default.yaml](configs/default.yaml)

## S&P 500 Pipeline (GCP)

A standalone pipeline (`sp500_pipeline.py`) pulls all WRDS source data, builds date-validated
identifier links, computes abnormal returns and controls, extracts NLP features from Capital IQ
transcripts, and produces a single modeling table — one row per earnings call.

### Dataset summary

| Metric | Value |
|--------|-------|
| Total earnings calls | 29,698 |
| Calls with CAR[+2,+20] | 26,314 |
| Calls with SUE | 27,744 |
| Date range | 2010-01-06 to 2026-10-30 |
| GICS sectors | 11 |
| Train sample | 2010–2022 |
| Holdout sample | 2023–2024 |

### Output files

| File | Description |
|------|-------------|
| [`model_table.csv`](model_table.csv) | One row per call (45 cols): IDs, sector, Day 0, CAR targets, controls, sample split (no linguistic) |
| [`model_table_with_linguistic.csv`](model_table_with_linguistic.csv) | Full table (54 cols): everything in model_table + LM tone, hedging, uncertainty, deltas |
| [`data_report.txt`](data_report.txt) | Summary statistics, coverage by year, missing values, validation checks |
| [`model_a_baseline.py`](model_a_baseline.py) | Model A script: preprocessing, feature selection, 8-model comparison |
| [`model_a_report.txt`](model_a_report.txt) | Model A baseline report: preprocessing, feature selection, CV and holdout results |
| [`model_a_results.json`](model_a_results.json) | Machine-readable Model A results (metrics, selected features, config) |
| [`model_b_linguistic.py`](model_b_linguistic.py) | Model B script: Model A controls + linguistic features, 8-model comparison |
| [`model_b_report.txt`](model_b_report.txt) | Model B report: feature selection, CV results, A vs B comparison |
| [`model_b_results.json`](model_b_results.json) | Machine-readable Model B results |
| [`sp500_pipeline.py`](sp500_pipeline.py) | Main pipeline: PULL → PANEL → TARGETS → TEXT → FINAL |

### Column reference (`model_table.csv`)

| Column(s) | Category | Description |
|-----------|----------|-------------|
| `keydevid`, `transcriptid`, `companyid` | IDs | Capital IQ event and transcript identifiers |
| `companyname`, `ticker`, `gvkey`, `permno` | IDs | Company name, ticker, Compustat and CRSP identifiers |
| `call_date`, `call_time_utc`, `headline` | Call metadata | Earnings call date, UTC time, and headline |
| `sector`, `gind`, `sic` | Industry | GICS sector name, GICS industry code, SIC code |
| `day0` | Timing | First trading day on or after the call (event day 0) |
| `car_0_1` | Target | Cumulative abnormal return, day 0–1 (announcement return) |
| `car_2_5` | Target | CAR day 2–5 (short-term drift) |
| **`car_2_20`** | **Target** | **CAR day 2–20 (primary target — post-earnings drift)** |
| `car_2_60` | Target | CAR day 2–60 (long-term drift) |
| `vol_post_2_20` | Target | Post-call realized volatility, day 2–20 |
| `vol_pre` | Control | Pre-call realized volatility (20 trading days before) |
| `mcap_m`, `size` | Control | Market cap ($M) and log market cap |
| `mom` | Control | 12-1 month momentum |
| `bm` | Control | Book-to-market ratio |
| `fiscal_period_end` | Control | Fiscal quarter end date |
| `actual_eps`, `consensus_eps`, `eps_surprise` | Earnings | Actual EPS, I/B/E/S consensus, and surprise |
| `sue`, `sue_price`, `surprise_pct` | Earnings | Standardized Unexpected Earnings, SUE/price, surprise % |
| `n_analysts`, `dispersion` | Analyst | Number of covering analysts, forecast dispersion |
| `split_adjusted_consensus` | Analyst | Whether consensus was split-adjusted |
| `n_questions`, `n_answers` | Call structure | Number of Q&A questions and answers |
| `words_prepared`, `words_answers`, `words_questions` | Call structure | Word counts by section |
| `avg_answer_words` | Call structure | Average words per executive answer |
| `share_other_sections`, `share_answers_by_execs` | Call structure | Share of non-QA text; exec participation in answers |
| `tone_prepared`, `tone_answers` | Linguistic | LM net tone: (positive − negative) / (positive + negative) per 1k words |
| `hedging_prepared`, `hedging_answers` | Linguistic | LM weak-modal (hedging) words per 1k words |
| `uncertainty_answers` | Linguistic | LM uncertainty words per 1k words in Q&A answers |
| `tone_gap` | Linguistic | `tone_answers` − `tone_prepared` (shift from script to Q&A) |
| `tone_prepared_chg`, `tone_answers_chg` | Linguistic | Change in tone vs. prior 4-call average |
| `hedging_answers_chg` | Linguistic | Change in hedging vs. prior 4-call average |
| `period` | Split | Fiscal period label |
| `sample` | Split | `train` (2010–2022), `holdout` (2023–2024), or `no_target` (2025–2026) |
| `prior_calls_with_target` | Meta | Count of prior calls for same firm with realized CAR |

### Pipeline stages

1. **PULL** — Download 14 WRDS tables (CRSP, Compustat, I/B/E/S, Capital IQ transcripts)
2. **PANEL** — Chain date-validated identity joins: companyid → gvkey → permno → IBES ticker → GICS sector
3. **TARGETS** — Compute CAR[0,1], CAR[2,5], CAR[2,20] (primary), CAR[2,60], post-call volatility
4. **TEXT** — Extract NLP features: tone (Loughran-McDonald), hedging, word counts, exec participation
5. **FINAL** — Merge numeric + language features, compute deltas vs. prior 4 calls, assign train/holdout split

### How to run

```bash
# On a GCP VM (e2-standard-8, 32GB RAM, Ubuntu 22.04)
pip install -r requirements.txt
python sp500_pipeline.py          # interactive WRDS login (Duo required)
python sp500_pipeline.py --stages panel targets text final   # rebuild from cached raw data
```

## Model A Baseline (No Linguistic Features)

Predicts CAR[+2,+20] using only financial controls and call structure — no sentiment or tone features.
Script: [`model_a_baseline.py`](model_a_baseline.py) | Report: [`model_a_report.txt`](model_a_report.txt) | JSON: [`model_a_results.json`](model_a_results.json)

### Preprocessing

- **Winsorization** at 1st/99th percentile (fit on train per CV fold, no leakage)
- **Log-transform** right-skewed features: `words_prepared`, `words_answers`, `words_questions`, `avg_answer_words`, `n_analysts`
- **Median imputation** for missing values (per fold)
- **VIF multicollinearity check** — flagged `n_answers` (15.9), `avg_answer_words` (12.3), `words_answers` (11.8)
- **StandardScaler** per fold
- **Sector one-hot encoding** (drop first)
- **Time-series expanding window CV** — 4 folds, no future leakage

### Feature selection

Three-method ensemble ranking (Spearman |r|, Mutual Information, Random Forest importance).
Selected top 11 of 18 candidate features by average rank:

| Feature | Spearman |r| | p-value | MI | RF Imp | Avg Rank | Selected |
|---------|-------------|---------|------|--------|----------|----------|
| `dispersion` | 0.032 | 1.3e-06 | 0.021 | 0.071 | **2.7** | Yes |
| `mom` | 0.009 | 0.192 | 0.030 | 0.136 | **3.7** | Yes |
| `prior_calls_with_target` | 0.006 | 0.326 | 0.011 | 0.093 | **6.3** | Yes |
| `vol_pre` | 0.003 | 0.635 | 0.063 | 0.247 | **6.7** | Yes |
| `car_0_1` | 0.018 | 0.007 | 0.003 | 0.046 | **6.7** | Yes |
| `size` | 0.015 | 0.020 | 0.008 | 0.036 | **6.7** | Yes |
| `sue_price` | 0.006 | 0.380 | 0.013 | 0.069 | **7.3** | Yes |
| `sue` | 0.014 | 0.033 | 0.006 | 0.028 | **9.0** | Yes |
| `bm` | 0.005 | 0.438 | 0.007 | 0.051 | **9.3** | Yes |
| `words_answers` | 0.017 | 0.008 | 0.000 | 0.026 | **10.5** | Yes |
| `n_answers` | 0.013 | 0.043 | 0.000 | 0.020 | **12.8** | Yes |
| `n_questions` | 0.008 | 0.213 | 0.009 | 0.025 | 9.7 | — |
| `surprise_pct` | 0.003 | 0.622 | 0.007 | 0.039 | 11.3 | — |
| `n_analysts` | 0.004 | 0.572 | 0.001 | 0.032 | 13.0 | — |
| `words_prepared` | 0.008 | 0.205 | 0.000 | 0.022 | 13.2 | — |
| `avg_answer_words` | 0.004 | 0.538 | 0.000 | 0.034 | 13.5 | — |
| `words_questions` | 0.006 | 0.340 | 0.000 | 0.023 | 14.2 | — |
| `share_answers_by_execs` | 0.008 | 0.241 | 0.000 | 0.001 | 14.5 | — |

### Cross-validation results

Time-series expanding window, 4 folds, train period 2010–2022. All 8 models compared:

| Model | Avg IC | IC ± | Avg Rank IC | Rank IC ± | R² | Dir Acc | Q5-Q1 Spread |
|-------|--------|------|-------------|-----------|-----|---------|-------------|
| **RandomForest** | **+0.052** | 0.092 | **+0.039** | 0.075 | −0.018 | 0.521 | **+1.05%** |
| XGBoost | +0.030 | 0.085 | +0.031 | 0.070 | −0.057 | 0.521 | +0.76% |
| GradientBoosting | +0.025 | 0.079 | +0.029 | 0.065 | −0.053 | 0.518 | +0.55% |
| ExtraTrees | +0.028 | 0.076 | +0.018 | 0.042 | −0.013 | 0.517 | +0.38% |
| Lasso | +0.006 | 0.092 | +0.007 | 0.015 | −0.013 | 0.533 | −0.13% |
| ElasticNet | +0.004 | 0.078 | +0.008 | 0.027 | −0.014 | 0.515 | −0.00% |
| OLS | −0.000 | 0.061 | +0.010 | 0.028 | −0.020 | 0.513 | +0.04% |
| Ridge | −0.000 | 0.061 | +0.010 | 0.028 | −0.020 | 0.513 | +0.04% |

**Best model by Rank IC: RandomForest** (avg Rank IC +0.039, Q5-Q1 spread +1.05%)

### Holdout results (2023–2024)

One-time evaluation of the best CV model (RandomForest) on unseen data:

| Metric | Value |
|--------|-------|
| IC | −0.076 |
| Rank IC | −0.005 |
| R² | −0.088 |
| RMSE | 0.063 |
| Direction accuracy | 43.5% |
| Q5-Q1 spread | −0.06% |

Numeric controls alone do not generalize to the holdout period — motivating the addition of linguistic features (tone, hedging, uncertainty) in Model B.

### Key definitions

| Metric | Definition | Range | Good value |
|--------|-----------|-------|------------|
| **IC** (Information Coefficient) | Pearson correlation between predicted and actual CAR | [−1, +1] | > +0.05 |
| **Rank IC** | Spearman rank correlation — preferred for trading signals | [−1, +1] | > +0.03 |
| **R²** | Variance explained; negative = worse than predicting zero | (−∞, 1] | > 0 |
| **Dir Accuracy** | Share of predictions with correct sign | [0, 1] | > 0.52 |
| **Q5-Q1 Spread** | Mean CAR of top quintile minus bottom quintile | any | > 0 |

## Model B: Controls + Linguistic Features

Adds Loughran-McDonald dictionary features (tone, hedging, uncertainty) on top of Model A's 11 controls.
Script: [`model_b_linguistic.py`](model_b_linguistic.py) | Report: [`model_b_report.txt`](model_b_report.txt) | JSON: [`model_b_results.json`](model_b_results.json)

### Linguistic features added

| Feature | Description |
|---------|-------------|
| `tone_prepared` | Net tone in prepared remarks: (positive − negative) / (positive + negative) per 1k words |
| `tone_answers` | Net tone in Q&A answers |
| `hedging_prepared` | Weak-modal (hedging) words per 1k words in prepared remarks |
| `hedging_answers` | Hedging words per 1k words in Q&A answers |
| `uncertainty_answers` | Uncertainty words per 1k words in Q&A answers |
| `tone_gap` | `tone_answers` − `tone_prepared` (shift from script to live Q&A) |
| `tone_prepared_chg` | Change in prepared-remarks tone vs. prior 4-call average |
| `tone_answers_chg` | Change in Q&A tone vs. prior 4-call average |
| `hedging_answers_chg` | Change in Q&A hedging vs. prior 4-call average |

### Feature selection (20 candidates → 13 selected)

| Feature | Spearman |r| | MI | RF Imp | Avg Rank | Type | Selected |
|---------|-------------|------|--------|----------|------|----------|
| `dispersion` | 0.032 | 0.021 | 0.069 | **2.7** | ctrl | Yes |
| `mom` | 0.009 | 0.030 | 0.129 | **4.3** | ctrl | Yes |
| `car_0_1` | 0.018 | 0.004 | 0.051 | **5.7** | ctrl | Yes |
| `prior_calls_with_target` | 0.006 | 0.011 | 0.085 | **6.0** | ctrl | Yes |
| `vol_pre` | 0.003 | 0.063 | 0.225 | **6.0** | ctrl | Yes |
| `size` | 0.015 | 0.008 | 0.036 | **6.3** | ctrl | Yes |
| `sue_price` | 0.006 | 0.013 | 0.066 | **7.0** | ctrl | Yes |
| `sue` | 0.014 | 0.006 | 0.024 | **8.7** | ctrl | Yes |
| `bm` | 0.005 | 0.007 | 0.051 | **9.3** | ctrl | Yes |
| `tone_prepared_chg` | 0.013 | 0.000 | 0.043 | **11.2** | LING | Yes |
| `tone_answers` | 0.006 | 0.001 | 0.030 | **12.0** | LING | Yes |
| `words_answers` | 0.017 | 0.000 | 0.022 | **12.2** | ctrl | Yes |
| `n_answers` | 0.013 | 0.000 | 0.020 | **13.5** | ctrl | Yes |
| `tone_gap` | 0.010 | 0.002 | 0.017 | 12.7 | LING | — |
| `hedging_prepared` | 0.005 | 0.000 | 0.023 | 14.3 | LING | — |
| `tone_answers_chg` | 0.001 | 0.002 | 0.028 | 14.3 | LING | — |
| `tone_prepared` | 0.004 | 0.000 | 0.025 | 15.2 | LING | — |
| `hedging_answers_chg` | 0.003 | 0.002 | 0.019 | 15.7 | LING | — |
| `uncertainty_answers` | 0.001 | 0.002 | 0.020 | 16.2 | LING | — |
| `hedging_answers` | 0.001 | 0.002 | 0.019 | 16.8 | LING | — |

Only 2 of 9 linguistic features were selected: `tone_prepared_chg` and `tone_answers`. Hedging and uncertainty features ranked in the bottom third.

### Cross-validation results

| Model | Avg IC | Avg Rank IC | Dir Accuracy | Q5-Q1 Spread |
|-------|--------|-------------|-------------|--------------|
| **RandomForest** | **+0.045** | **+0.038** | 0.522 | **+0.94%** |
| XGBoost | +0.022 | +0.031 | 0.516 | +0.57% |
| GradientBoosting | +0.016 | +0.030 | 0.515 | +0.56% |
| ExtraTrees | +0.026 | +0.018 | 0.524 | +0.25% |
| Ridge | −0.003 | +0.008 | 0.510 | +0.01% |
| OLS | −0.003 | +0.008 | 0.510 | +0.01% |
| Lasso | +0.005 | +0.007 | 0.533 | −0.14% |
| ElasticNet | +0.002 | +0.007 | 0.515 | +0.01% |

### Model A vs Model B comparison

| Metric | Model A (controls only) | Model B (+ linguistic) | Difference |
|--------|------------------------|----------------------|------------|
| **CV Rank IC** | +0.039 | +0.038 | −0.001 |
| **CV Q5-Q1** | +1.05% | +0.94% | −0.11% |
| Holdout Rank IC | −0.005 | −0.011 | −0.006 |
| Holdout Q5-Q1 | −0.06% | −0.32% | −0.26% |

**Conclusion:** LM dictionary-based linguistic features (tone, hedging, uncertainty) do not improve prediction of post-earnings drift CAR[+2,+20] beyond the financial controls in this dataset. The bag-of-words approach may be too coarse — next steps include contextual embeddings (FinBERT), interaction-quality features, or finer-grained text measures.

## Roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Pull all WRDS data and build full S&P 500 event dataset | **done** |
| 2 | Preprocessing + feature selection on 18 non-linguistic features | **done** |
| 3 | Baseline Model A — RandomForest, Rank IC +0.039 (CV) | **done** |
| 4 | Compute LM tone, uncertainty, hedging from transcript text | **done** |
| 5 | Model B — add linguistic features, compare to Model A | **done** (no improvement) |
| 6 | Try contextual embeddings (FinBERT) or interaction-quality features | next |
| 7 | Evaluation: IC comparison, long-short backtest net of costs | pending |

Done: full pipeline, GCP VM, data validation, Model A baseline, LM linguistic features, Model B comparison.
Finding: LM bag-of-words features do not improve over controls alone — finer-grained text features needed.

## Obstacles and proposed solutions

| Obstacle | Impact | Proposed solution |
|----------|--------|-------------------|
| Split-adjusted I/B/E/S rounds consensus to 2 decimals (94/179 legacy rows; 13 with consensus 0) | SUE, the main control, is wrong | Unadjusted `actu_epsus` / `statsumu_epsus` + CRSP `cfacshr`; scale SUE by price |
| `panel.xlsx` truncated text at Excel's 32,767-char cell limit (59/179 calls) | Biased language features | Never store text in xlsx; parquet only |
| CRSP ends 2024-12-31 | 2025–2026 calls have no realized CAR labels | Build through 2022; keep 2023–2024 as the untouched holdout and later calls for scoring only |
| `sprtrn` excludes dividends, CRSP `ret` includes them | AR drifts up by ~dividend yield | Use `vwretd` / total-return S&P; FF4 as robustness |
| Delisting returns pulled but not used | Survivorship bias | Merge `dlret` into returns |
| Links ignore valid dates; duplicates dropped arbitrarily | Wrong firm/ticker matches | Date-valid, ranked links with attrition log |
| No rule for transcript versions (redacted/unredacted, re-collected ids) | Wrong or mixed text per call | Rank by creation time / presentation type; agree a precedence rule |
| Industry exclusion could narrow the universe | User scope is all historical S&P 500 members | No industry exclusion in the full-universe build |
| No GCP credits for audio features | Vocal features out of reach | Defer; use cheaper text features (readability, complexity) |

## Setup

```bash
uv venv && uv pip install -e '.[dev,wrds]' pandas numpy scipy scikit-learn statsmodels matplotlib sentence-transformers xgboost transformers torch         # add ,wrds / ,models / ,nlp extras as needed
export WRDS_USERNAME=<your_wrds_user>          # only for `la pull`
source .venv/bin/activate                      # Activate virtual environment
```

## Usage

```bash
la pull                      # optional: WRDS -> data/raw (or drop web-UI exports there)
la transcripts               # raw CIQ components -> data/interim
la transcripts --legacy      # same, from ../Data/panel.xlsx (lossy, see audit)
la events | targets | controls | features | model | evaluate   # specified, not yet built
.venv/bin/python scripts/audit_legacy_panel.py
pytest
```

### Full S&P 500 dataset

The local WRDS build follows the pull-and-combine workflow in the project notebook and writes
licensed source data to ignored `data/raw/`. It creates aligned Model A and Model B files at
`data/processed/model_table_baseline.parquet` and
`data/processed/model_table_with_language.parquet` (Model B is also available at
`model_table.parquet`), plus Q&A pairs and a sample attrition report. The 2023–2024 holdout is
retained for final evaluation. Later calls appear in the linked event audit, but the subscribed CRSP
S&P membership table ends on 2024-12-31, so 2025–2026 calls are marked as membership unavailable
and excluded from the scoreable model tables until membership coverage is refreshed.

```bash
uv sync --extra wrds --extra models
WRDS_USERNAME=<your_wrds_user> SEC_CONTACT_EMAIL=<your_contact_email> \
  .venv/bin/python scripts/build_wrds_dataset.py
```

The pull resumes from completed yearly Parquet files. `SEC_CONTACT_EMAIL` is sent only in the
SEC archive request header for the same-day Exhibit 99.1 text used by the press-release divergence
features. Without SEC archive access those F5 fields stay missing. Never commit WRDS/SEC source
files or transcript text.

### Model A dashboard and FinBERT

After building the two event tables, run the language-free model comparisons and dashboard:

```bash
.venv/bin/python scripts/run_model_a_dashboard.py
```

This uses a date-ordered 80/20 cutoff, purges training events whose 60-session labels could overlap
the test period, and reports CAR[+2,+5], CAR[+2,+20], and CAR[+2,+60]. The HTML dashboard and
aggregate results are written to `outputs/model_a_80_20/`; row-level predictions remain in ignored
`data/processed/`.

#### Current Model A results

The saved comparison uses 22,233 training events (2010-01-06–2021-08-03) and 5,672 test events
(2021-10-28–2024-10-18), across 13 test quarters. The gap purges 350 events to prevent the
60-trading-day target windows from crossing the split. Across the models compared, the highest
test information coefficient (IC) by horizon was:

| Target | Model | Test IC | Rank IC | Test R² | RMSE |
|---|---|---:|---:|---:|---:|
| CAR[+2,+5] | Random Forest | 0.024 | 0.017 | -0.012 | 0.0330 |
| CAR[+2,+20] (primary) | Extra Trees | 0.041 | 0.026 | -0.0065 | 0.0631 |
| CAR[+2,+60] | Random Forest | 0.001 | 0.005 | -0.020 | 0.1215 |

These are exploratory comparisons: the models were compared on the same test period, so the
highest-IC model per horizon is not an unbiased final model-selection estimate. The primary-horizon
IC is modest, and negative R² means the prediction errors were worse than predicting zero CAR for
every event. The net top-minus-bottom quintile spread for the primary-horizon Extra Trees model was
about 0.34% per quarter under the dashboard's assumed 20 bp round-trip cost per leg; this estimate
is not yet a validated trading result. The current run is Model A only and does not establish that
linguistic features improve forecasts.

Open the [Model A dashboard](outputs/model_a_80_20/dashboard.html) for all model metrics and
diagnostics. The underlying [metrics CSV](outputs/model_a_80_20/metrics.csv), [split metadata](outputs/model_a_80_20/split.json),
and [pooled OLS control coefficients](outputs/model_a_80_20/ols_controls_clustered.csv) are also
available in the results folder. The OLS coefficient p-values use PERMNO-clustered standard errors
and do not account for common quarter shocks.

FinBERT adds sentiment columns to a separate Model B dataset:

```bash
uv sync --extra wrds --extra models --extra nlp
.venv/bin/python scripts/add_finbert_features.py
```

It scores up to 32 evenly spaced sentences per prepared-remarks or answer section and uses Apple
MPS when available. The output is `data/processed/model_table_with_language_finbert.parquet`.

### Research scripts

```bash
.venv/bin/python scripts/01_build_features.py          # add --finbert for the alternative tone measure
.venv/bin/python scripts/02_validate.py                # then label data/qa_label_sample.csv, save as data/qa_labels.csv, rerun 01 and 02
.venv/bin/python scripts/03_cross_section.py           # robustness: --sample highprec, --delayed, --cluster year, --sue SUE_IBES, --fe firm
.venv/bin/python scripts/04_oos.py --freeze            # add --models ols,enet,xgb to include XGBoost
.venv/bin/python scripts/05_backtest_holdout.py        # dev-period backtest
.venv/bin/python scripts/05_backtest_holdout.py --stage holdout
```

## Layout

```
configs/            run configuration (sample, windows, splits, feature lists)
data/{raw,interim,processed,external}/   gitignored: WRDS data is licensed
docs/               pipeline design, data audit
notebooks/          exploration only; logic lives in src/
scripts/            one-off checks
src/linguistic_alpha/
  data/             wrds_pull, transcripts, linking, universe
  events/           Day 0 timing
  targets/          CAR, post-call volatility
  controls/         SUE and analyst controls, firm characteristics
  features/         communication_state (F1-F2), interaction_quality (F3), consistency (F4-F5)
  models/           Model A / B specs, walk-forward splits, regressions
  evaluation/       IC, model comparison, long-short backtest
tests/
```

Never commit data, credentials or transcript text. The `.gitignore` blocks parquet, csv and xlsx files.
