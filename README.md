# linguistic-alpha

Research pipeline: do earnings-call language features predict S&P 500 post-call drift,
CAR[+2,+20], beyond the earnings surprise and standard controls?
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
| CRSP ends 2024-12-31 | Planned 2024–2026 holdout is only 2024 | Refresh CRSP, or move holdout to 2023–2024 (needs sign-off) |
| `sprtrn` excludes dividends, CRSP `ret` includes them | AR drifts up by ~dividend yield | Use `vwretd` / total-return S&P; FF4 as robustness |
| Delisting returns pulled but not used | Survivorship bias | Merge `dlret` into returns |
| Links ignore valid dates; duplicates dropped arbitrarily | Wrong firm/ticker matches | Date-valid, ranked links with attrition log |
| No rule for transcript versions (redacted/unredacted, re-collected ids) | Wrong or mixed text per call | Rank by creation time / presentation type; agree a precedence rule |
| Biotech exclusion needs GICS history | Universe not as agreed | Pull `comp.co_hgic`, exclude GICS 352010 on the call date |
| No GCP credits for audio features | Vocal features out of reach | Defer; use cheaper text features (readability, complexity) |

## Setup

```bash
uv venv && uv pip install -e '.[dev]'          # add ,wrds / ,models / ,nlp extras as needed
export WRDS_USERNAME=<your_wrds_user>           # only for `la pull`
```

## Usage

```bash
la pull                      # optional: WRDS -> data/raw (or drop web-UI exports there)
la transcripts               # raw CIQ components -> data/interim
la transcripts --legacy      # same, from ../Data/panel.xlsx (lossy, see audit)
la events | targets | controls | features | model | evaluate   # specified, not yet built
python scripts/audit_legacy_panel.py
pytest
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
