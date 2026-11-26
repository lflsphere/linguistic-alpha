# Consumer Discretionary (GICS 25): results

Same protocol as `it_results/`. Calls are those with `sector == "Consumer Discretionary"` in
`model_table_with_linguistic.csv`: 3,711 calls, 98 companies. Train ≤ 2023 (3,201), test 2024 (166 calls, 52 firms).

**Status: Models A, B, B\* done. Model C waits for FinBERT (`notebooks/cd_finbert_colab.ipynb`).**

## Holdout results (IT protocol: best algorithm/strategy picked on the holdout)

| Model | Opt 1 vs S&P 500: Rank IC | Q5−Q1 | Opt 2 vs CD index: Rank IC | Q5−Q1 |
|---|---|---|---|---|
| A (16 controls) | −0.1378 | −1.32% | +0.0211 | +1.24% |
| B (best of 6) | −0.1175 (Combined Good Features) | −1.17% | +0.0758 (All Ling + RidgeCV) | +1.86% |
| B* (all 9 LM) | −0.1277 | −1.37% | +0.0679 | +3.22% |

## Algorithm picked by train-only CV, firm-clustered bootstrap 95% CI (`cd_diagnostics.json`)

| Model | Opt 1 Rank IC [CI] | Opt 2 Rank IC [CI] |
|---|---|---|
| A | −0.141 [−0.292, +0.042] | +0.017 [−0.136, +0.176] |
| B* | −0.128 [−0.275, +0.055] | +0.068 [−0.086, +0.227] |
| B* − A | [−0.045, +0.072] | [−0.012, +0.118] |

No Rank IC and no improvement over Model A is distinguishable from zero at 95%.

## Caveats

- Test is 2024 only (CRSP cut at 2024-12-31 to match IT), although the files say "test > 2023".
- "Dir Acc" equals the share of positive test targets: models predict ranks (always > 0). Ignore it, and R²/RMSE.
- The CD index is 41% Amazon and 13% Tesla (2024-06-28), so Option 2 is largely "vs. AMZN/TSLA".
- The committed IT tables have stale `day0`, `bm`, `vol_pre` (correct CARs). CD uses the corrected CSV.
  Rebuilding IT the same way moves IT Model A from +0.0379/+0.0682 to +0.0384/+0.0699.
- The superseded Nov 16 IT runs (`it_model_b_option_*`) are not replicated. `cd_model_a_*` uses the current 16-control protocol.
- One call has no transcript text (O'Reilly, 2010-10-28).

## Reproduce

```bash
PYTHONPATH=src python sector_extract_local.py --raw <data/raw/sp500_pit_2008> --out cd_results --text-out <dir>
python it_model_b.py --folder cd_results --prefix cd --sector-label CD
# Colab: notebooks/cd_finbert_colab.ipynb, then unzip its output into cd_results/
PYTHONPATH=src python cd_diagnostics.py
```

## Placebo: read this before interpreting Model C

While testing the pipeline I ran Model C on CD with **random** FinBERT labels (sentence counts were real, no XGBoost;
nothing from it is stored here). On Option 2 the "best" Model C still showed Rank IC +0.0705, Δ +0.0287 vs. A.
The IT headline gain is +0.0261. Best-of-N selection on a 166–204 call holdout can produce such a Δ from noise,
so judge Model C by `cd_diagnostics.json` (C* with CV-picked algorithm and CI), not the best-of-4 headline.
