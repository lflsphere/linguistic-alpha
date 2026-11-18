# linguistic-alpha

Research pipeline: do earnings-call language features predict S&P 500 post-call drift,
CAR[+2,+20], beyond the earnings surprise and standard controls?
Data comes from WRDS: Capital IQ transcripts, CRSP, Compustat and I/B/E/S.

- Architecture and stage contracts: [docs/pipeline.md](docs/pipeline.md)
- Legacy notebook audit and rebuild plan: [docs/data_audit.md](docs/data_audit.md)
- Run settings: [configs/default.yaml](configs/default.yaml)

## Roadmap (pipeline order)

| # | Task | Status | Done when |
|---|------|--------|-----------|
| 1 | Pull the point-in-time S&P 500 from 2008 to 2026-09 (`scripts/pull_sp500_pit.py`) | done (pull + validation) | audit checks clean: coverage per year, no truncation, missing times reported |
| 2 | Settle open decisions (holdout, benchmark, version rule, SUE, universe) | done 2026-11-17 | recorded in `configs/default.yaml` and `docs/pipeline.md` |
| 3 | Transcripts pass 1: parse and clean every version (`data/transcripts.py`, `features/lexicons.py`) | parse done, cleaning todo | operator lines and safe harbor stripped, words removed logged |
| 4 | Linking + universe (`data/linking.py`, `data/universe.py`) | stub | one row per call and one permno per event, link attrition logged per step |
| 5 | Events: Day 0, call check (`events/timing.py`) | Day 0 done, table todo | `processed/events` written |
| 6 | Transcripts pass 2: one version per call, capped at entry | function done, stage todo | runs after events; `qa_pairs`, `call_sections` from selected versions |
| 7 | Targets: CAR windows, post-call vol (`targets/car.py`) | stub | CAR[2,20] matches `build_reaction_windows.py` within 1e-6 |
| 8 | Controls: SUE_TS on unadjusted I/B/E/S, firm controls, release metrics (`controls/`) | stub | no zero consensus; release metrics dated before the Day 0 close; every missing guidance metric set to 0 with `guidance_given`; SAL / EBT from the adjusted non-EPS tables |
| 9 | Features F1–F4 + interactions, standardised (`features/`) | interface | LM dictionary in `data/external/`; schemes as in `docs/pipeline.md` |
| 9a | Pin the text models: FinBERT checkpoint, embedding model (methods 3–4), hedge-phrase list source | todo | names and versions in `configs/default.yaml`; anonymised input |
| 10 | F3 labels: ~450 full-history pairs + 90-pair second labeller (`docs/pipeline.md`, Q&A labels) | todo | `data/qa_labels.csv` filled; kappa ≥ 0.6; F3 validated |
| 11 | Connect the full pull to `scripts/01–05` | todo | `common.load_events` reads `data/processed/`, not `panel.csv` |
| 12 | EDA + feature selection inside training windows | todo | selection table: each feature, each filter, reason kept or dropped |
| 13 | Models A, B, C, C_alone × 2 legs: OLS, lasso, ridge, elastic net (`models/`); rolling splits with per-target embargo and time-ordered validation | splits done, models stub | out-of-sample IC and R² for every model and leg; train / validation / test design confirmed |
| 14 | Evaluation (gross) (`evaluation/`) | stub | ΔIC, ΔOOS R², bootstrap and Clark-West tables |
| 14a | Backtest net of costs: further work, rules and cost sources to be decided | todo | portfolio rules and cost data chosen |
| 15 | Holdout 2024 Q1 – 2026 Aug, once, on frozen models | todo | hashes recorded, single run |

Done: package scaffold, transcript ingestion and Q&A pairing, Day 0 timing and version selection
(tested), audit of the legacy notebook ([docs/data_audit.md](docs/data_audit.md)), 2008 pull.
Future work: RavenPack benchmark (dropped from the core path), F5 vs. 8-K press release,
volatility target, FF4 risk leg, audio features.

## Obstacles and proposed solutions

| Obstacle | Impact | Proposed solution |
|----------|--------|-------------------|
| Split-adjusted I/B/E/S rounds consensus to 2 decimals (94/179 legacy rows; 13 with consensus 0) | SUE, the main control, is wrong | Unadjusted `actu_epsus` / `statsumu_epsus` on one split basis via I/B/E/S `adj` / `adjsum`; scale SUE by price |
| `panel.xlsx` truncated text at Excel's 32,767-char cell limit (59/179 calls) | Biased language features | Never store text in xlsx; parquet only |
| CRSP ended 2024-12-31 | Planned 2024–2026 holdout was only 2024 | **Done:** quarterly CRSP runs to 2026-09-30. Sample 2011 Q1 – 2026 Q3 (63 quarters): build 2011–2023, holdout 2024 Q1 – 2026 Q3 |
| `sprtrn` excludes dividends, CRSP `ret` includes them | AR drifts up by ~dividend yield (+0.18 pp over [+2,+20]) | Done: S&P 500 total return (`crsp_dsp500_v2.vwretd`, beta = 1); robustness: market model on the same index (beta from days −250 to −30); CRSP VW market only as an index check; FF4 left for a future risk-management leg. Windows rebuilt by `scripts/build_reaction_windows.py` |
| Delisting returns pulled but not used | Survivorship bias | **Resolved by CIZ:** `dlyret` already includes delisting returns; never add `delret` again |
| Links ignore valid dates; duplicates dropped arbitrarily | Wrong firm/ticker matches | Date-valid, ranked links with attrition log |
| No rule for transcript versions (redacted/unredacted, re-collected ids) | Wrong or mixed text per call | **Decided:** latest version created by the Day 0 close (by creation time); earliest as robustness; `version_final` control |
| Biotech exclusion (meeting 10/23) | Smaller universe | **Decided 2026-11-17:** keep biotech; check it by sector in EDA |
| No GCP credits for audio features | Vocal features out of reach | Defer; use cheaper text features (readability, complexity) |

## Setup

```bash
uv venv && uv pip install -e '.[dev,wrds]' pandas numpy scipy scikit-learn statsmodels matplotlib sentence-transformers xgboost transformers torch         # add ,wrds / ,models / ,nlp extras as needed
export WRDS_USERNAME=<your_wrds_user>          # only for `la pull`
source .venv/bin/activate                      # Activate virtual environment
```

## Usage

```bash
.venv/bin/python scripts/pull_sp500_pit.py   # WRDS -> data/raw/sp500_pit_2008 (the pull in use)
la pull                      # older generic pull (data/wrds_pull.py)
la transcripts               # raw CIQ components -> data/interim
la transcripts --legacy      # same, from ../Data/panel.xlsx (lossy, see audit)
la events | targets | controls | features | model | evaluate   # specified, not yet built
.venv/bin/python scripts/audit_legacy_panel.py
pytest
```

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
