# Model A financial baselines

The original dashboard remains at `scripts/run_model_a_dashboard.py`. It uses the
archived CIQ event timestamp with a fixed 16:00 New York boundary, includes
CAR[0,+1] among its eight controls and fits the archived CAR labels. Its
default estimator inputs and command are preserved. The shared estimator
factory accepts an explicit feature list for the new runner.

The new runner selects either `a2` (the seven financial controls below) or
`a2_short_momentum` (the same controls plus the validated short momentum).
Configuration lives in `configs/model_a_financial.yaml`. CAR[+2,+20] is
primary. CAR[+2,+5] and CAR[+2,+60] are retained. Neither new specification
uses CAR[0,+1], announcement volume, or transcript-derived features.

## Timing and labels

The immediate-signal proxy is the archived CIQ recorded event time in UTC.
Convert it to New York time before assigning a session. Pre-open events
use the current market day. Intraday, at-close and post-close events use
the next market session. Non-trading dates also roll forward. Unknown
clocks produce missing Day 0. DSI supplies the session dates and the
2008–2024 XNYS overlay supplies the 09:30 open, regular close and early
closes, including DST through New York timezone conversion. A complete
`date,open_et,close_et` schedule can be supplied for another calendar.

CAR uses market-day offsets, CRSP value-weighted total return (`vwretd`),
and the sum of daily stock-minus-benchmark abnormal returns. Combine
`RET` and `DLRET` multiplicatively when both are observed, use the observed
component when one exists, and leave both-missing returns missing. Each
target requires a complete calendar window and at least 80% valid returns.
The signal must precede the Day +1 close at which the Day +2 return starts.

Actual call completion, vendor delivery time and point-in-time historical
vintages remain unverified. The date and close gates are conservative
bounds using the recorded CIQ clock. The panel does not require transcript
publication evidence for these financial-only experiments.

## Inputs and definitions

Prepare from an authorized archived `model_table_baseline.parquet` and
local CRSP DSI, DSF and DSEDELIST files, including supported annual shards.
The archived table supplies the original controls, dated fields, original
Day 0 and labels. The new prep screens the archived values and recalculates
the new targets and short momentum. It reads no transcript text.

| Factor | Source and frozen definition |
|---|---|
| SUE | I/B/E/S ACTU_EPSUS quarterly actual less the split-aligned STATSUMU_EPSUS median for the same fiscal period, divided by absolute CRSP price before original Day 0. Require an announcement date strictly before the local CIQ date, a consensus snapshot strictly before the announcement and the price close before the signal. |
| Size | Log of CRSP equity market capitalization at the last observed stock day before the call month, using absolute price times shares outstanding in shares. Require that day's close before the signal. |
| Book-to-market | Archived Compustat book equity divided by market capitalization. The archived code uses SEQQ + TXDITCQ − PSTKQ with a CEQQ fallback, and uses RDQ or a 90-day reporting proxy. Require archived book date strictly before the local signal date and the price close before the signal. |
| Long momentum | Archived CRSP cumulative return from month start minus 12 months through month start minus two months, inclusive, with at least 120 observations. The archived month endpoint is retained for reproduction. Require the upper market-date close before the signal. |
| Pre-event volatility | Archived annualized standard deviation of abnormal returns over original stock-row offsets −60 to −5. Require the last underlying close before the signal. |
| Analyst count | I/B/E/S NUMEST from the archived consensus snapshot, with the same prior-consensus-date gate as SUE. |
| Analyst dispersion | Archived consensus standard deviation divided by absolute split-aligned median, with the same prior-consensus-date gate. |
| Short momentum | Exactly the 30 market sessions immediately before the I/B/E/S ANNDATS insertion date, ending at the prior session. Compound DSF RET as product(1 + RET) − 1. Require all 30 observations, drop invalid returns below −1, and require the last session's close strictly before the signal. Preserve the archived upstream requirement that ANNDATS precedes original Day +1. |

## Local reproduction

Use the project's supported Python environment and install its model
dependencies. From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$run = 'outputs/local_financial_reproduction'
python scripts/run_model_a_financial.py prepare --raw data/raw --baseline data/processed/model_table_baseline.parquet --out "$run/prepared"
python scripts/run_model_a_financial.py fit --panel "$run/prepared/financial_panel.parquet" --spec a2 --out "$run/a2"
python scripts/run_model_a_financial.py fit --panel "$run/prepared/financial_panel.parquet" --spec a2_short_momentum --out "$run/a2_short_momentum"
python -m pytest -q
```

The runner requires a new directory per run. The original 80/20 cut is
2021-10-28, with the original 60-session purge. The new matched comparison
also requires both anchors' windows to finish before the cutoff for train
events and matches observed targets on both anchors. The validated counts
are 22,216 train and 5,672 test for the primary and short horizons, and
5,624 test for the long horizon. The estimators reuse training-only median
imputation, the original OLS scaling and the fixed seed 7 and tree settings.

Expected primary Extra Trees R² and rank IC are −0.480% and 0.0326 for A2,
and +0.108% and 0.0583 for the enhanced baseline. These are exploratory
results from an already examined test period. The paired uncertainty does
not establish stable improvement. With authorized prior sensitivity and
ablation artifacts, `scripts/verify_model_a_financial.py --help` describes
the optional event-level comparison.

Raw data, derived panels, event-level forecasts, transcript content and
credentials remain local. New files under `data/` and `outputs/` are ignored.
This policy does not de-track inherited files. Review legacy event-level and
transcript-excerpt files with the team before broader sharing.
