# linguistic-alpha

Research pipeline: do earnings-call language features predict S&P 500 post-call drift,
CAR[+2,+20] and other horizons, beyond the earnings surprise and standard controls?
Data comes from WRDS: Capital IQ transcripts, CRSP, Compustat and I/B/E/S.

- Architecture and stage contracts: [docs/pipeline.md](docs/pipeline.md)
- Legacy notebook audit and rebuild plan: [docs/data_audit.md](docs/data_audit.md)
- Run settings: [configs/default.yaml](configs/default.yaml)

## Roadmap (in priority order)

| # | Task | Status | Done when |
|---|------|--------|-----------|
| 1 | Pull CRSP, Compustat, I/B/E/S and links for AAPL/MSFT/NVDA (`la pull`) | todo | raw files in `data/raw/`, table names checked (CRSP CIZ migration) |
| 2 | Linking + universe (`data/linking.py`, `data/universe.py`) | stub | one row per call, link attrition logged per step |
| 3 | Targets: CAR windows, post-call vol (`targets/car.py`) | stub | CAR[2,20] matches legacy panel within 1e-6 on the same benchmark |
| 4 | Controls: SUE on unadjusted I/B/E/S, size, B/M, mom (`controls/`) | stub | no zero consensus; AAPL Q1-2010 surprise small and positive |
| 5 | Confirm open decisions with prof/employer (see Obstacles) | todo | holdout, benchmark, version rule fixed in `configs/default.yaml` |
| 6 | Scale to the full S&P 500 2010–2024 on WRDS Cloud | todo | audit checks clean: coverage per year, no truncation, missing times reported |
| 7 | Features F1–F2: LM tone, uncertainty, hedging (`features/communication_state.py`) | interface | LM dictionary in `data/external/`, safe harbor stripped |
| 8 | Features F3–F5: evasiveness, change vs. prior calls, divergence | interface | F3 validated on a hand-labeled Q&A sample |
| 9 | Model A vs. B: walk-forward, Fama-MacBeth, FDR (`models/`) | stub | coefficient tables and out-of-sample IC for both models |
| 10 | Evaluation: IC comparison, long-short backtest net of costs (`evaluation/`) | stub | portfolios A / L / B, gross vs. net, decay, capacity |

Done: package scaffold, transcript ingestion and Q&A pairing, Day 0 timing (tested), audit of
the legacy notebook ([docs/data_audit.md](docs/data_audit.md)).

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

The [selectable financial baselines](docs/model_a_financial_baselines.md) document the local A2 and short-momentum runs.

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
