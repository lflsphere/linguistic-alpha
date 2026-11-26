# Consumer Discretionary (GICS 25): results

Target CAR[+2,+20] (`car_2_20`), Option 1 vs. S&P 500, Option 2 vs. a value-weighted index of the sector's
S&P 500 members. Train ≤ 2023, test 2024 (CRSP ends 2024-12-31, as for IT).

**Headline: with every choice made on train/validation only, no model beats Model A in a way that survives
its confidence interval, and FinBERT adds nothing. The one interval that excludes zero (B\* on Option 2) is
one of eight comparisons.**

## Look-ahead fixes (vs. the first CD run and the IT protocol)

| Issue | Fix |
|---|---|
| Algorithm, strategy, PCA size, interaction variant and ensemble members picked on the **holdout** | `cv_protocol.py`: all picked by expanding-window CV inside train (validation years 2021, 2022, 2023); holdout scored once |
| Sample selected with the `sector` column, which is **current** GICS (media firms moved to Communication Services in 2018, general-merchandise retailers to Staples in 2023) | Calls selected by GICS sector **on the call date** (`comp.co_hgic`): 4,246 calls, 123 firms (594 added, 59 dropped) |
| Option 2 index built from firms whose **current** GICS is CD | Index membership by GICS on each day: 148 firms ever, 72 a day (was 117 / 62) |
| CAR[+2,+20] of late-December train calls runs into the next validation/test year | Purge: train rows within 35 days of each validation/test start are dropped (8 before 2024) |
| A, B, C compared with different algorithm sets (C had trees, A/B linear only) | One pool for all: Ridge (several α), RidgeCV, ElasticNet ×3, RandomForest, GradientBoosting, XGBoost |

Forward selection already used train-only CV, so it never saw 2024. Its CV score, and that of ensemble
members, is still optimistic (chosen on the same validation folds); the holdout number is not.

## Main results: train-only selection (`cd_cv_protocol_report.txt`)

Train 3,728, test 166 calls (52 firms). Firm-clustered bootstrap 95% CIs (5,000 draws).

| Model | Opt 1 selected (algo) | Opt 1 Rank IC [CI] | Opt 2 selected (algo) | Opt 2 Rank IC [CI] |
|---|---|---|---|---|
| A (16 controls) | RandomForest | −0.088 [−0.259, +0.087] | Ridge α=10 | −0.002 [−0.159, +0.157] |
| B (best of 10 strategies on CV) | forward selection, GB | −0.020 [−0.170, +0.133] | forward selection, Ridge | +0.057 [−0.099, +0.218] |
| B\* (all 9 LM) | RandomForest | −0.074 [−0.247, +0.100] | RidgeCV | +0.090 [−0.060, +0.246] |
| C (best of 4 strategies on CV) | forward selection, GB | −0.051 [−0.213, +0.110] | forward selection, Ridge α=10 | −0.040 [−0.185, +0.109] |
| C\* (all 13 FinBERT) | GradientBoosting | −0.022 [−0.181, +0.142] | Ridge α=50 | −0.008 [−0.156, +0.135] |

Δ vs. A, 95% CI:

| | Opt 1 | Opt 2 |
|---|---|---|
| B | [−0.064, +0.186] | [−0.018, +0.138] |
| B\* | [−0.087, +0.107] | **[+0.003, +0.182]** |
| C | [−0.082, +0.152] | [−0.134, +0.056] |
| C\* | [−0.053, +0.185] | [−0.101, +0.087] |

- CV and holdout disagree. On Option 2 the CV-best C (forward selection, CV +0.045) is the worst on the
  holdout (−0.040), while "FinBERT + LM" (CV +0.019) would have won on the holdout (+0.139). Picking on the
  holdout is what produced the earlier FinBERT gains.
- Option 2 forward selection picks `fb_neg_pct_prepared`, then the sentence counts `fb_n_sent_prepared`,
  `fb_n_sent_answers` (call length), and `fb_sent_gap`.
- LM forward selection on Option 2: `tone_answers_chg`, `hedging_answers_chg`, `tone_gap`.
- Fold-level CV Rank ICs swing sign (e.g. Opt 2 A-type models: +0.07 / −0.05 / +0.03 for 2021 / 2022 / 2023).

## For comparison: IT protocol (holdout-picked) on the same corrected data

`cd_model_b_improved_*`, `cd_model_c_*` (run locally with XGBoost; `cd_run.log`, `cd_modelc_run.log`):

| | Opt 1 Rank IC (Δ vs A) | Opt 2 Rank IC (Δ vs A) |
|---|---|---|
| A (B script, linear) | −0.181 | −0.005 |
| B (best of 6) | −0.137 (+0.044) Combined Good Features | +0.085 (+0.091) All Ling + RidgeCV |
| B\* | −0.141 (+0.040) | +0.078 (+0.084) |
| A (C script, linear + trees) | −0.081 | +0.032 |
| Best C | −0.010 (+0.071) FinBERT + LM, GB | +0.097 (+0.065) FinBERT + LM, GB |

These "Δ" are inflated by selection on the 166-call holdout. A placebo with random FinBERT labels gave
Δ +0.029 on Option 2 under this protocol.

## Caveats

- Test = 166 calls, 52 firms, one year: a Rank IC needs to be about ±0.15 to be distinguishable from zero.
- "Dir Acc" in the IT-style reports equals the share of positive test targets (rank predictions are always > 0). Ignore it, and R²/RMSE.
- The CD index is dominated by Amazon (41%) and Tesla (13%) in 2024, so Option 2 is largely "vs. AMZN/TSLA".
- `*_chg` features use the firm's previous 4 calls **in the sample**; a firm entering CD has no history at entry.
- FinBERT for the 594 added calls was scored locally on Apple MPS with the same code (`finbert_score_calls.py`);
  20 calls re-scored that way match the Colab T4 run exactly. One call has no transcript text (O'Reilly, 2010-10-28).
- `cd_model_a_*` and `cd_diagnostics.json` come from `cd_diagnostics.py` (linear algorithms only, older
  check); `cd_cv_protocol_*` supersedes them.
- The committed IT tables have stale `day0`, `bm`, `vol_pre` (correct CARs) and use current GICS; `it_results/` was not re-run.

## Reproduce

```bash
PYTHONPATH=src python sector_extract_local.py --raw <data/raw/sp500_pit_2008> --out cd_results --text-out <text dir>
# FinBERT: notebooks/cd_finbert_colab.ipynb (all calls), or finbert_score_calls.py for missing calls + cd_merge_finbert.py
PYTHONPATH=src python cv_protocol.py --folder cd_results --prefix cd --sector-label CD     # main results
python it_model_b.py --folder cd_results --prefix cd --sector-label CD                       # IT protocol, comparison
python it_model_c_finbert.py --folder cd_results --prefix cd --sector-label CD
```
`--sample column --gics current` reproduces the first (current-GICS) CD tables exactly.
