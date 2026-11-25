# Healthcare Linguistic Alpha Research

Original Sector Pipeline — Annual Expanding Window Analysis

## Research design

The experiment asks whether annual coefficient updates improve overall rank performance and whether language features add ranking power relative to the updated Model A. The previously inspected Static results make the evaluation a post-hoc exploratory robustness check. The 2024 cohort is a historical diagnostic and informed original model selection. The earlier [Static evaluation](../healthcare_extended_oos/methodology.md) is the reference for the return definitions, selected strategies, rank scores, transforms, test events, and inference rules.

The baseline is Git `ffee3b4a2caaa5c450b2ffcda353be70c229e9ac`. Original model source comes from `06273f1766daeb29de4b9313dd20639439a09671`. The evaluation checks its committed Static source and aggregate inputs against Git and verifies the frozen source and input hashes before evaluating the updates. `expanding_experiment_spec.json` embeds the Static manifest in full and records the expansion rule and data hashes. Legacy predictors and labels through 2024 come from the original option-specific archived tables. For 2025–26 the evaluator verifies eligibility with the existing static-sector filter, dated S&P 500 membership and original mapping code, and compares identities, Day 0 and realized targets exactly against the saved Static event ledger. It reuses the saved sector extension and validation, the S&P 500 `SPRTRN` extended CAR, and the matching benchmark-specific Early CAR. This run performs no new benchmark construction or model selection.

## Annual refit and label-maturity rule

| Prediction period | Conceptual training years | Refit boundary, ET | Training events per benchmark | Matched events / issuers |
| --- | --- | --- | ---: | ---: |
| 2024 historical diagnostic | 2010–2023 | 2024-01-16 16:00 | 2,939 | 204 / 62 |
| 2025 | 2010–2024 | 2025-01-17 16:00 | 3,143 | 211 / 61 |
| 2026 partial | 2010–2025 | 2026-01-22 16:00 | 3,354 | 104 / 59 |
| 2025–26 pooled | Annual fits above | Two held-out cohorts combined | — | 315 / 61 |

Use the earliest Day +1 close across the matched cohort as the annual refit boundary. Day 0 stays exactly as saved in the Original Sector Pipeline and validated extension. The evaluator uses CRSP trading sessions for Day +1 and Day +20, and a nominal 16:00 Eastern close. All three cutoff dates are regular January sessions. A 16:00 timestamp is conservative for historic half-days and does not change eligibility at these boundaries. The UTC refit times are also saved. Prediction represents availability after Day +1 close. The training gate uses the close boundary itself and excludes any label ending at that same instant. Only prior-calendar-year calls with a saved target and a recoverable Day +20 strictly before refit enter training. There is no intra-year updating. The latest included label clocks were January 9, 2024, December 26, 2024, and January 6, 2026, at 16:00 Eastern. No saved target was excluded for arriving too late or having an unknown endpoint. One 2024 archive event lacks a native target within its original return vintage and stays outside the expanded history. The matched 2026 sample ends with the May 27 call, using returns through June 30.

The day-span audit preserves the original return handling. Two archived 2010–2023 labels have 18 of 19 stock/benchmark pairs and satisfy the original coverage rule. Their 19-session windows had elapsed before every relevant fit. They are distinct from immature outcomes. The validated 2025 extension labels require every paired session. The training audit summarizes the coverage and separate maturity checks, with a local per-event ledger for reproduction.

## Models and benchmark conventions

Fit all six frozen estimators separately once in each of the three years. The original 16 controls include `car_-1_1` and the archived prior-call count. Model B adds its original benchmark-specific ordered LM5. B* adds the original LM9. `hedging_answers` and `uncertainty_answers` are identical under the original dictionary and both remain. Archived predictions use `car_-1_1`. `car_0_1` and `AR01` are absent from the input lists. The predictions are unnormalized average midranks of training CAR. Each expanding fit recomputes the training target ranks, log transformations where defined, 1st/99th percentile winsorization, training median imputation and scale. The input lists and all estimator settings are recorded in the embedded manifest.

| Benchmark | A | B | B* |
| --- | --- | --- | --- |
| S&P 500-adjusted | Ridge α=1 | Ridge α=1 | Ridge α=10 |
| Healthcare sector-adjusted | Ridge α=1 | Elastic Net α=0.05, L1 ratio 0.7 | Ridge α=50 |

Both targets sum benchmark-adjusted daily returns over trading sessions +2 to +20. The industry option uses the archived same-day cap-weighted Healthcare benchmark and static GICS classification. Timing remains the original generic pipeline. There is no additional transcript clock gate. Early CAR is only available after Day +1 close. Availability of the full feature vector and an executable price at that time was not established. The historical predictive associations and overlapping event sorts therefore do not establish portfolio returns.

## Metrics and exploratory inference

- The evaluator calls the committed Static metric routine on the new predictions. It independently replays the saved Static tables against the saved scores before comparing schedules. Spearman Rank IC and Pearson correlation use the same event/target pairs. Pearson is supplementary. Rank predictions are not calibrated to CAR units. Continuous-return R² and error measures are omitted.
- Expansion results compare each LM model against updated A and each updated model against its own Static version. Pooled metrics recompute directly on the 315 raw, held-out predictions. Unnormalized midranks have different training-set sizes in 2025 and 2026, so cross-year score shifts can affect the pooled ranking. The primary analysis does not scale or recalibrate the raw scores.
- Form quintiles by predicted score within each benchmark, schedule and period. Keep the frozen minimum of 50 calls and 20 issuers. Quantile groups use `pandas.qcut` and preserve ties. Report means, medians, sample SD, n, Q5−Q1 mean spread and strict monotonicity. Report returns in decimal units in CSV, percent in the quintile report, and percentage points for spreads.
- Quarterly IC uses the frozen minimum of 30 calls and 15 issuers. Exclude the incomplete second quarter of 2026 from positive-IC share and SD. Five full quarters qualify. Six are required to report ICIR.
- Resample matched pairs using the same 2,499 draws, random seeds, issuer clusters and overlapping two-calendar-month blocks as the Static evaluator. Recompute all six score correlations for each draw. Percentile intervals require at least 30 issuers, five nominal disjoint time blocks, and at least 95% valid draws. Both intervals are pointwise. The 2024 and 2026 partial-year periods are descriptive. The 2025 and pooled periods have six and nine nominal blocks respectively.
- Exploratory pooled p-values re-center each bootstrap distribution and apply the finite-draw correction, then use the larger method-specific p-value. Adjust within two prespecified families using Holm. The LM family contains two LM contrasts for each of two benchmarks. The schedule family contains three schedules contrasts for each of two benchmarks. The exploratory screen requires adjusted p≤0.05 and both intervals excluding zero in the same direction. No result passes this screen. The paired intervals condition on the fitted scores and do not account for repeating the full model training and study-selection process.

## Reproduction

Reproduce from the repository root in PowerShell. Use the local licensed files required by the [Static evaluation](../healthcare_extended_oos/methodology.md), including its local `matched_predictions_local.csv` and `healthcare_sector_extended_car.csv`. The script reads the approved archive and verified local benchmark artifacts in place. The model run used Python 3.12.10, pandas 2.2.3, NumPy 2.5.4, SciPy 1.18.1 and scikit-learn 1.9.1. The report used Matplotlib 3.7.5. Use the same model environment for exact 2024 replay.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$dest = 'outputs/healthcare_expanding_oos/reproduction'
.\.venv\Scripts\python.exe outputs/healthcare_expanding_oos/evaluate_expanding.py --out $dest
Copy-Item outputs/healthcare_expanding_oos/methodology.md $dest
python outputs/healthcare_expanding_oos/render_expanding.py --out $dest
```

| Artifact | Contents |
| --- | --- |
| `expanding_experiment_spec.json` | Frozen original model and annual maturity rules with input hashes |
| `refit_training_audit.csv`, `evaluation_match_audit.csv` | Cutoff and training counts, late or unavailable outcomes, exact test matches |
| `matched_predictions_local.csv`, `training_maturity_local.csv` | Local derivatives containing event-link identifiers. Keep these files local. |
| `model_performance.csv`, `static_to_expanding.csv`, `incremental_comparison.csv` | Annual and pooled metrics, each model’s update effect, and LM contrasts |
| `bootstrap_inference.csv` | Method-specific intervals, p-values, replication and cluster counts |
| `quintile_returns.csv`, `quarterly_ic.csv`, `stability_summary.csv` | Sort results and temporal diagnostics |
| `dashboard.html`, `Healthcare_Static_vs_Annual_Expanding_Window.pdf`, `summary_zh.md` | Public-facing aggregate reports and Chinese summary |

The dashboard embeds only aggregate CSV data. It does not display licensed source rows or local event-level predictions. All generated files are limited to the new experiment folder and temporary QA destinations.
