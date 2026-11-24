# Healthcare methodology and frozen comparison

## Original and corrected runs

| Dimension | Original Pierre run | Existing corrected run |
|---|---|---|
| Event universe | `sector == 'Health Care'` in the supplied full export. The benchmark uses static Compustat classifications, estimated locally for companies outside that export. | Effective-date GICS 35, date-valid security and membership links, audited financial archive, and the matched transcript cohort. |
| Day 0 | The pinned `assign_day0()` converts CIQ UTC to New York time and applies 16:00. | Same nominal 16:00 convention. Actual session closes set the signal and label cutoff. |
| CAR[+2,+20] | Option 1 uses SPRTRN from Pierre’s upstream market file schema. Option 2 uses active S&P members from a static Healthcare pool with same-day cap weights. The original extractor keeps its partial coverage and delisting truncation rules. | Option 1 uses CRSP broad-market VWRETD. Option 2 uses effective-date sector membership and prior-session cap weights. The corrected stock return compounds RET and DLRET, and uses a complete calendar window and paired-return coverage. |
| A controls and LM | A uses 16 original controls, including Early CAR, `prior_calls_with_target` and six transcript structure measures. B uses the selected winning LM strategy and B* includes all nine archived LM columns. The CSV ratios, missingness, and prior-four-call changes were retained exactly. | A0 selects from seven audited financial controls, A1 adds short momentum, and S1 adds eight structure measures. B uses forward-selected LM. B* uses all nine, with date-gated earlier transcript changes. |
| Signal availability | The model uses the recorded call date for splitting. It imposes no gate on the full transcript version, archived financial availability, or maturity of a predictor such as Early CAR. | The matched cohort uses creation of the exact scored transcript as the signal proxy, strictly before Day +1 close. Financial information stays frozen at call start. Creation alone cannot verify vendor delivery time. |
| Fit and imputation | Unnormalized CAR midranks, original training-only 1st/99th-percentile winsorization, median fill, count/length log transformations and scaling. Ridge, RidgeCV and Elastic Net candidates. | Continuous CAR and training-only transforms, with the estimator and feature variants selected in expanding development validation. The existing matched Healthcare models use Elastic Net `EN_003`. |
| Split and selection | Call-date training through 2023-12-31. Expanding validation in 2021–2023 informs forward feature selection. Several algorithms, the ensemble route, PCA dimension and the final strategy use holdout Rank IC. No label maturity purge. | Expanding 2021–2023 folds with label maturity purging. Refits use only matured development labels. Final selection precedes holdout scoring. |

The original main model consumed 2,939 training rows and 204 held-out 2024 calls per benchmark. The native extraction log prints an older 2021 split for its size check. That log line is not the model’s train/test rule. The corrected transcript runs use 1,890 refit calls and 175 held-out calls. The corrected call-time financial-only runs use 2,842 refit calls and 198 test calls. That financial-only cohort does not evaluate B or B*.

Original R², RMSE, directional accuracy and quintile spread are preserved in the native results. The original R² and RMSE compare unscaled ranks to decimal CAR and are not continuous-CAR error measures. The original direction statistic also compares signs across incompatible scales. Rank IC, Pearson IC and return sorts are retained with their original definitions.

[Native original metrics](original_results.csv) · [Native corrected metrics and sample sizes](native_corrected_comparison.csv) · [All six native strategies per benchmark](native_strategy_scores.csv)

## Matched test and attribution

The event key is `transcriptid + PERMNO + UTC call timestamp`. Both models’ native test forecasts are fixed. All 175 corrected transcript test events link uniquely to the original 204. Of the other 29 original test events, 23 lack a transcript version created before the target starts and 6 lack the matched audited financial archive. Equal event sets are held constant for all four matched scoring pairs per benchmark. The event ledger and outputs contain no raw transcript text.

Both forecasts are scored on the original and corrected CAR on the same 175 events. Day 0 matches exactly. The CAR rank correlations are 0.9964 for the market benchmark and 0.9991 for the Healthcare benchmark. Across all 175 events per benchmark, the observed change in CAR equals the 19-session change in benchmark return to within 1e-7. The mean corrected-minus-original CAR is −0.001154 for the market benchmark and +0.004626 for the sector benchmark. On these matched cases, the benchmark change accounts for the observed label change.

| Benchmark | Original native B−A | Shared events with original labels | Shared events with corrected labels | Corrected B−A1, same events and labels |
|---|---:|---:|---:|---:|
| Market | +0.03547 | +0.04373 | +0.04770 | 0.00000 |
| Healthcare sector | +0.03856 | +0.03848 | +0.03916 | 0.00000 |

For B, the market sample restriction changes the total Rank IC by +0.03405 and the benchmark-label swap changes it by +0.01039. The corresponding changes in sector total Rank IC are −0.00224 and −0.00107. The remaining change on the same corrected labels is −0.10554 for the market and −0.06790 for the sector. This combines the different controls and LM vintages, fitting cohorts and purging, target scale, estimators, and development-versus-test selection. The frozen comparison identifies sample and target effects along this sequence. It cannot separately identify the effects of controls and selection without additional fitting.

The original final strategy choice increases native market B by +0.00621 relative to the already computed forward route. The sector winner is itself the forward route. These comparisons hold the existing route results fixed. The original route estimators still use test selection. Their paired intervals use 2,000 time-block and issuer-cluster draws on fixed predictions and are conditional on the chosen models. They do not correct for the v7 test search. On the corrected matched forecasts, adding structure alone changes market Rank IC by +0.05531 and sector Rank IC by zero. Both corrected LM models match A1’s predictions. This does not establish that the conditional original LM increment would persist at a different time or under another selection protocol.

[Frozen same-event, same-label metrics](matched_event_comparison.csv) · [Stagewise differences](attribution.csv) · [Target check](target_reconciliation_summary.csv) · [Sample attrition](sample_attrition.csv) · [Prediction correlations](prediction_correlations.csv)
