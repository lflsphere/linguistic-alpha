# Healthcare Linguistic Alpha Research

## Frozen design

The two experiments compare annually expanding versus rolling five-year training, and separately compare CAR[+2,+5] versus CAR[+2,+20]. Models are fitted separately by test year, window, target and benchmark. The frozen research specification was saved before scoring. The outer tests are retrospective because past results for these calendar years were previously examined. The new implementation shares controls across the three models and makes all selections within historical training folds.

## Data and label validation

The original native Healthcare option tables supply 2010–24 covariates and targets, with the original 2024 data vintage. Later calls come from the validated financial feature reconstruction. New call eligibility requires verified CIQ/security links and dated S&P 500 membership. For each benchmark, the full eligible short/long counts are 205/204 in 2024, 211/211 in 2025, and 106/104 in partial 2026. Direct horizon comparisons use their intersection. Window comparisons use every eligible event for the selected horizon. The comparison code checks event and target equality before scoring.

Both labels sum stock-minus-benchmark daily returns for sessions +2 to +5 or +2 to +20. The S&P 500 price benchmark uses SPRTRN. The Healthcare return uses the original static GICS 35 basket, dated membership and same-day market capitalization weights. Native labels preserve original availability and delisting conventions. New sector short CAR uses original validated daily sector returns and the same CIZ stock return type used in the long benchmark. The short target replicates the 205 archived 2024 sector labels from legacy data and is verified with a separate four-session join on each new eligible event. The rebuilt modern Healthcare long labels match the saved series. Both modern market labels match the validated extension. See `target_validation.csv` for overlaps, errors and tolerances. `data_provenance.json` hashes source files and the analysis scripts.

## Clock and fit eligibility

Retain the original 16:00 ET Day 0 mapping and CRSP trading dates. Record the conservative Day +1 close at 16:00 ET, the Day +2 scheduled session open, and Day +5 and +20 close timestamps. A 16:00 bound intentionally waits past an early close. A usable prediction is timestamped one microsecond after that bound and precedes the Day +2 session open. This is an ordering convention for the event study. The first close-to-close daily return begins at the previous close and includes overnight movement that may be unavailable to a post-close signal. Accordingly, a live implementation could not capture the entire first return under this clock. The assumed transcript delivery, actual scoring latency and all-vintage financial data availability are not verified. The research does not apply a transcript timing screen.

Use a common annual refit boundary at the earliest eligible Day +1 close bound. Training calls must predate the test year, have observed Early CAR and a valid target, and mature strictly before refit. Each historical validation year has its own cutoff at that year's earliest eligible validation Day +1 close bound. Remove any inner-fit label mature at or after that cutoff. All inner-validation outcomes mature before the annual refit. The timing audit and fold checks exercise strict equality boundary cases. Rolling uses the previous five calendar years. The otherwise comparable expanding history starts in 2010. Training labels are required to have a known completed session window. No model uses either prediction horizon CAR, post-label volatility or the other horizon's outcome as a feature.

## Controls, LM8 and model selection

The sixteen controls are SUE, relative surprise, pre-event volatility, Early CAR, momentum, book-to-market, size, analyst dispersion, average answer length, prepared/question/answer word counts, question and answer counts, analyst count and prior calls observed. The latter counts earlier calls by archived PERMNO irrespective of whether they later have a target. This replaces the archive's target-availability count and keeps it independent of future label availability. Later financial values use the saved reconstruction and its validated source-specific definitions. Early CAR corresponds to the same benchmark as the target.

Verify exact equality, including missingness, between hedging_answers and uncertainty_answers. Keep uncertainty_answers. The other LM inputs are tone_prepared, tone_answers, hedging_prepared, tone_gap, tone_prepared_chg, tone_answers_chg and hedging_answers_chg. All source variables remain intact.

Use the last three eligible calendar years as expanding inner validation folds within each training schedule. Fit original log transforms, quantile winsorization, median replacement and scaling separately in each fold, then form average target midranks in training. Unavailable columns within a fold contribute zero after standardization. Fit four candidate estimators, Ridge at α=1, 10 and 50 and Elastic Net α=0.05, L1 ratio 0.7. Model A and B* choose estimators on the same folds. B compares the forward route and jointly fitted univariate-screened subset with A. Minimum mean IC gain is 0.001. Prespecified tie rules follow feature and estimator order, then subset size and route. B falls back to the identical A estimator and inputs if neither candidate clears the threshold. Train-only folds and scores determine every choice. The fitted estimator then scores the year's held-out predictors once. The comparison records the estimator, subset, inner IC and both accepted and rejected steps.

## Factor analysis and inference

Development-factor calculations use mature pre-2024 calls. Outer-factor analysis works separately by test year, benchmark and horizon. Factors use observed values. Spearman correlations, factor sorts, conditional groups, quarterly stability and missingness are reported separately from model predictions. Pre-volatility conditions use the applicable development median and SUE conditions use zero. Quarterly estimates require 25 calls and 15 issuers. Conditional estimates require 30 calls and 15 issuers. Correlations use pairwise observations, with a minimum of 20. No factor result from an outer year is passed back to the selector.

Spearman Rank IC is primary, with Pearson supplementary. Q1–Q5 sorts use model scores or raw factor inputs. CAR is in decimal returns in CSV and in percent in plots, while Q5−Q1 appears in percentage points. Event windows may overlap. No live portfolio result is estimated. Signed importance is a standardized-input regression coefficient divided by training n, with a supplementary absolute coefficient share. This is a description of a joint model fit.

Pointwise paired percentile intervals use 999 draws by issuer or overlapping noncircular two-month call-date blocks. Resampling preserves both predictions and any outcome pair on each sampled event. Methods address different dependence structures and neither fully addresses both at once. At least 50 events, 30 issuers and 95% valid draws are required. Time-block intervals also require five nominal disjoint blocks. Partial 2026 has three and therefore shows issuer intervals when eligible and time-block estimates without intervals. Many contrasts are exploratory. No multiplicity-adjusted confirmatory claims are made.

The local bridge methodology and its saved IT Sector reports were reviewed as references for the chronological decision design. This experiment uses the verified Healthcare inputs. The previous annual evaluation, final report and Phase 1 reference remain separate.

## Reproduction and file layout

Run from the repository root with the licensed source paths in `data_provenance.json` available locally. Fit and audit with the project's Python 3.12 environment containing pandas, NumPy, SciPy, PyArrow and scikit-learn. Render with a Python environment containing pandas, NumPy and Matplotlib. Provide the same `--out` folder to both scripts to rebuild elsewhere. The renderer reads the archived Phase 1 stylesheet in `phase1_style.css` and the saved aggregate files.

```powershell
$entry = 'outputs/healthcare_window_horizon_comparison'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe "$entry/run_comparison.py"
.\.venv\Scripts\python.exe -m unittest discover -s $entry -p test_comparison.py
python "$entry/render_comparison.py"
```

`training_cohorts.csv`, `cv_folds.csv`, `target_validation.csv` and `horizon_cohorts.csv` contain coverage and maturity evidence. `window_comparison.csv`, `horizon_comparison.csv`, `annual_model_performance.csv`, `lm_increments.csv` and `paired_inference.csv` provide the primary estimates. `model_quintiles.csv` and the factor tables distinguish model prediction sorts from raw factor sorts. `selected_features.csv`, `feature_selection_paths.csv`, `estimator_cv.csv` and `feature_importance.csv` preserve all training decisions. Event-level forecast and timing ledgers are local outputs without dashboard download links. No licensed observations are duplicated into the PDF or dashboard.
