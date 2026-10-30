# linguistic-alpha

Research pipeline: do earnings-call language features predict S&P 500 post-call drift,
CAR[+2,+20], beyond the earnings surprise and standard controls?
Data comes from WRDS: Capital IQ transcripts, CRSP, Compustat and I/B/E/S.

- Architecture and stage contracts: [docs/pipeline.md](docs/pipeline.md)
- Legacy notebook audit and rebuild plan: [docs/data_audit.md](docs/data_audit.md)
- Run settings: [configs/default.yaml](configs/default.yaml)

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
