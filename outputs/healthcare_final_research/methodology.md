# Healthcare Linguistic Alpha Research

## Saved predictions and test years

The report evaluates the original A/B/B* model definitions against S&P 500-adjusted and Healthcare sector-adjusted CAR[+2,+20]. Each test year, model and benchmark is scored on its own saved predictions and realized labels. For each test year, the two benchmark cohorts have identical event IDs. For each benchmark and year, the three models have identical event IDs and realized outcomes.

- **Test 2024** uses `outputs/healthcare_original_pierre_replication/original_test_predictions.parquet`. The original prediction rows, event IDs, Day 0 and realized targets agree exactly with the original test split and option-specific archive tables. The six Spearman and Pearson correlations agree at full saved precision with `native_model_metrics.csv`. The original experiment used 2024 test outcomes to compare estimators and strategies. The S&P 500 route also used these outcomes for individual feature screening. This year represents the original test and model-selection process. Its cohort and return labels predate the later CAR and financial-feature extensions.
- **Test 2025** and **Test 2026 (Partial)** use the `prediction` and `car_2_20` columns in `outputs/healthcare_expanding_oos/matched_predictions_local.csv`, filtered to their call-year rows. The saved expanding refits used 3,143 and 3,354 training calls respectively, at 16:00 ET on January 17, 2025 and January 22, 2026. New training observations needed a known Day +20 close strictly before the associated cutoff. The 2026 test ends at the May 27 earnings call, with a June 30 CRSP return cutoff. The archive for Test 2024 used 2,939 training calls through 2023-12-31.

## Models and metrics

Model A uses 16 financial and structural controls. Model B adds the selected five LM features in their original benchmark-specific order. Model B* uses all nine LM features. The manifest records the exact ordered inputs and estimator parameters. The original winner and feature choices are retained for later saved predictions. Training uses one-based average CAR ranks and original train-only preprocessing for transformations, winsorization, replacement of missing predictors and scaling. The original Q&A hedging and uncertainty variables are identical in the archived data. The model definitions include Early CAR `car_-1_1`, observable after Day +1 close. The event analysis does not establish that the full predictor set or an execution price was available at that time.

Both benchmark targets sum daily stock-minus-benchmark returns over CRSP sessions +2 to +20. The S&P 500 convention uses `SPRTRN`. The Healthcare benchmark uses the originally validated static GICS Healthcare basket, dated index membership and contemporaneous market-cap weights. The report preserves the realized targets as saved.

Rank IC is Spearman correlation between model scores and realized CAR. Pearson correlation is supplementary. The yearly increments are B−A and B*−A, each computed within a benchmark on matched events. CSV values retain floating-point precision. The report shows the six original Test 2024 Rank IC values at full archive precision. Other display values are rounded for readability.

Within each year and benchmark, quantiles divide predictions into five groups with `pandas.qcut`, preserving score ties. Q1 is the lowest group. The table reports each group's count, mean, median and sample SD of realized CAR. Q5−Q1 is the difference in mean CAR, shown in percentage points. Overlapping windows make these event-level distributions distinct from strategy returns. Quarterly Spearman IC is verified against the expanding source for 2025 and partial 2026, with no Test 2024 quarterly series in this report.

Test 2025 paired 95% intervals are retained from the saved within-year annual expanding LM contrasts and checked against independently recalculated point estimates. Each method has 2,499 valid draws. The issuer bootstrap resamples GVKEYs and the time bootstrap uses overlapping two-calendar-month blocks. The 2025 sample has 61 issuers and six nominal blocks. These are exploratory pointwise intervals conditional on the saved predictions. Test 2024 is descriptive given historical test-based selection. Partial 2026 has three nominal blocks, below the minimum of five.

## Reproduce

Run from the repository root. Use a Python environment with pandas, NumPy, SciPy and PyArrow for source verification. Use a Python environment with pandas, NumPy and Matplotlib to render the reports. The report uses Georgia headings and the Phase 1 visual palette and layout. No modeling libraries are needed by this builder.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe outputs/healthcare_final_research/build_final_report.py freeze
python outputs/healthcare_final_research/build_final_report.py render
```

Pass the same `--out` directory to both commands to rebuild in a fresh destination. The renderer verifies the hashes of all six frozen tables before rendering. `source_provenance.json` lists each source relative to the repository, its SHA-256, the frozen feature recipes and the aggregate hashes. The dashboard contains only derived aggregate tables. The licensed prediction and event ledgers remain in their original local directories.
