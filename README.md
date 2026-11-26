# Healthcare Linguistic Alpha

Healthcare research branch of the broader Linguistic Alpha project.

[Final English report](outputs/healthcare_unified_research/Healthcare_Linguistic_Alpha_Unified_Research_Report.pdf) · [Interactive dashboard](outputs/healthcare_unified_research/dashboard.html) · [Annual results](outputs/healthcare_unified_research/annual_model_performance.csv)

## Research Overview

Do the words used in Healthcare earnings calls help predict post-call abnormal returns after accounting for earnings surprises, stock characteristics, analyst forecasts, and call structure? The study compares financial controls alone with two representations of the same transcripts. Loughran–McDonald (LM) measures word use against finance-specific dictionaries. FinBERT scores sentences in context and supplies sentiment and sentence-count features. All five models share the same controls, realized returns, and eligible events within each test setting.

## Data and Event-Time Definitions

The study uses Capital IQ earnings-call components, CRSP security returns and index returns, Compustat company fundamentals, and I/B/E/S earnings and analyst forecasts. The FinBERT archive contains **3,551 Healthcare calls across 2010–2026** and **1,388,987 sentence scores**. The labeled evaluation ends earlier because the CRSP market-data cutoff is **June 30, 2026**. A call enters a test horizon only when its target is available.

The pipeline maps each call timestamp to New York time and uses the CRSP trading calendar. A call before 16:00 ET on a trading day is assigned to that session. Calls at or after 16:00 ET and calls on nontrading days are assigned to the next trading session. The study timestamps the signal just after the Day +1 close bound and before the Day +2 open.

CAR is the sum of daily security returns minus daily benchmark returns over sessions **[+2,+5]** or **[+2,+20]**, inclusive. The market benchmark is the S&P 500 price-return series (SPRTRN). The sector benchmark uses the study’s Healthcare basket, dated membership, and market-capitalization weights. The archived 2024 targets retain their original CRSP vintage. The validated extension supplies later targets. [Label validation](outputs/healthcare_window_horizon_comparison/target_validation.csv) · [Cohort definitions](outputs/healthcare_window_horizon_comparison/methodology.md)

| Annual test | CAR[+2,+5] calls | CAR[+2,+20] calls | Matched across horizons |
|---|---:|---:|---:|
| 2024 | 205 | 204 | 204 |
| 2025 | 211 | 211 | 211 |
| 2026 (partial) | 106 | 104 | 104 |

Counts are per benchmark. In partial 2026, the last eligible calls are June 4 for the short horizon and May 27 for the long horizon. [Test cohorts](outputs/healthcare_window_horizon_comparison/training_cohorts.csv) · [Horizon overlap](outputs/healthcare_window_horizon_comparison/horizon_cohorts.csv)

## Feature Engineering

- **Financial and structural controls.** The 16 controls cover standardized and relative earnings surprise, analyst dispersion and count, pre-event volatility, early CAR[−1,+1], momentum, book-to-market, size, word and turn counts, average answer length, and the number of previously observed calls.
- **LM8.** Prepared and answer tone, prepared hedging, answer uncertainty, the tone gap, and three changes in tone or hedging. The corrected feature set retains the answer-uncertainty level because the archived answer-hedging level has identical values and missingness.
- **FinBERT13.** The pinned `ProsusAI/finbert` checkpoint scores prepared remarks and management answers from Q&A. Each section contributes mean and standard deviation of signed sentiment, positive and negative sentence shares, and a sentence count. The remaining three features are the prepared-minus-answer sentiment gap and the two section-level sentiment changes. Analyst questions are excluded from answer sentiment.
- **Call history.** Each language change compares the current call with the observed mean over up to four strictly earlier calls for the same PERMNO. Archived LM changes use call dates, which have no same-PERMNO ties in the Healthcare archive. FinBERT changes enforce strict UTC ordering. The prior-call control counts past calls independently of whether they later received a CAR label.

The FinBERT feature definitions follow the earlier IT extractor at `9cf2d0e`. This Healthcare implementation adds strict timestamp ordering and a resumable local cache. [LM and control definitions](outputs/healthcare_window_horizon_comparison/methodology.md) · [FinBERT extraction and cache methodology](outputs/healthcare_unified_research/finbert_methodology.md)

## Model Framework

| Model | Predictors | Language feature selection |
|---|---|---|
| A | Financial and structural controls | None |
| B | Controls plus selected LM features | Inner validation chooses between forward selection, a jointly fitted univariate-screened subset, and controls alone |
| B* | Controls plus all LM8 features | Fixed LM8 block |
| C | Controls plus selected FinBERT features | Greedy forward selection on inner validation, with controls alone if no feature clears the minimum gain |
| C* | Controls plus all FinBERT13 features | Fixed FinBERT13 block |

## Experimental Design

Test 2024, Test 2025, and Test 2026 (Partial) are evaluated separately. For each year, the study crosses **Expanding** training from 2010 onward and **Rolling 5Y** training with both CAR horizons and both benchmarks. This yields **24 settings**. A direct comparison between horizons uses their event intersection.

The outer training cohort ends before the test year. Each fit uses three chronological inner validation folds. A training label must mature strictly before the applicable fold or refit cutoff. Winsorization, imputation, scaling, and target midranks are fitted within each training fold. The estimator grid consists of Ridge at α = 1, 10, and 50 and Elastic Net at α = 0.05 with an L1 ratio of 0.7. The forward searches require a mean inner-fold Rank IC gain greater than 0.001 per step.

The historical Pierre pipeline compared several strategies or estimators on holdout Rank IC when choosing what to report. The corrected pipeline freezes the model recipes using inner validation before scoring the annual test predictors. It retains Pierre’s preprocessing and event-time helpers from `06273f1`. The final unified run carries forward the saved A/B/B* forecasts and fits C/C* on the same cohorts. [Frozen A/B/B* protocol](outputs/healthcare_window_horizon_comparison/frozen_research_spec.json) · [Frozen unified protocol](outputs/healthcare_unified_research/frozen_unified_spec.json) · [Historical methodology comparison](outputs/healthcare_original_pierre_replication/methodology_comparison.md)

## Empirical Results

The table shows Spearman Rank IC for **Healthcare-adjusted CAR[+2,+20] with Expanding training**. Each row uses a matched annual cohort. [Full results for both benchmarks, horizons, and windows](outputs/healthcare_unified_research/annual_model_performance.csv)

| Annual test | Calls | A | B | B* | C | C* |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 204 | 0.0199 | 0.0535 | 0.0404 | 0.0696 | 0.1014 |
| 2025 | 211 | 0.0924 | 0.0966 | 0.0983 | 0.1560 | 0.2265 |
| 2026 (partial) | 104 | 0.2726 | 0.2708 | 0.2676 | 0.3055 | 0.2617 |

- Model C exceeded Model B in **20 of 24 settings** and improved on Model A in **21 of 24**.
- In the 2025 setting shown above, C* achieved **0.2265** versus **0.0924** for A. The paired increment was **+0.1341**, with a pointwise issuer-cluster 95% confidence interval of **[+0.0350, +0.2311]**.
- Most paired confidence intervals include zero. The settings share events, and the intervals are unadjusted for multiple comparisons. The evidence is exploratory. C−B compares complete modeling procedures, including their selected predictors and estimators.

[Incremental Rank IC](outputs/healthcare_unified_research/annual_increments.csv) · [Paired statistical inference](outputs/healthcare_unified_research/paired_inference.csv)

## Repository Structure and Reproducibility

| Source | Role |
|---|---|
| [scripts/healthcare_research.py](scripts/healthcare_research.py) | Current entry point for import and provenance checks and separate local replays |
| [run_comparison.py](outputs/healthcare_window_horizon_comparison/run_comparison.py) and [research_metrics.py](outputs/healthcare_window_horizon_comparison/research_metrics.py) | Frozen A/B/B* training and evaluation implementation |
| [extract_finbert.py](outputs/healthcare_unified_research/extract_finbert.py) and [finbert_runtime.py](outputs/healthcare_unified_research/finbert_runtime.py) | Resumable local FinBERT feature extraction and inference |
| [fit_finbert.py](outputs/healthcare_unified_research/fit_finbert.py), [evaluate_unified.py](outputs/healthcare_unified_research/evaluate_unified.py), and [verify_unified.py](outputs/healthcare_unified_research/verify_unified.py) | C/C* fitting, five-model evaluation, and release checks |
| [render_comparison.py](outputs/healthcare_window_horizon_comparison/render_comparison.py) and [render_unified.py](outputs/healthcare_unified_research/render_unified.py) | Renderers for the comparison and unified reports |
| [experiments/legacy_pipeline/](experiments/legacy_pipeline/) | Pinned upstream code for historical provenance and the preprocessing and timing helpers called by the current pipeline |
| [src/linguistic_alpha/](src/linguistic_alpha/) and [docs/pipeline.md](docs/pipeline.md) | Shared pipeline code and documentation for the broader project |

The published reports and aggregate tables can be reviewed directly from GitHub. For the dashboard’s interactive controls, download and open the HTML locally. Keep its companion CSVs in the same directory to use its table links.

For an authorized local replay, use Python 3.12 from the repository root. A full Git checkout must contain the preservation commit. Install the analysis dependencies and check the recorded source hashes and imports before using licensed data.

```text
python -m pip install -e ".[models]" scipy matplotlib
python scripts/healthcare_research.py check
```

Set up the licensed source and derived inputs at the paths in the provenance manifest. C/C* replay also requires the preserved local A/B/B* forecasts and the validated FinBERT cache under `data/derived/finbert/`. It reads those saved features without running inference. The model weights, runtime installation, event-level predictions, and caches stay local. [Local input guide](outputs/healthcare_unified_research/publication.md) · [Data provenance](outputs/healthcare_window_horizon_comparison/data_provenance.json) · [FinBERT setup](outputs/healthcare_unified_research/finbert_methodology.md)

Use fresh output directories for each replay. The C/C* command uses the preserved A/B/B* inputs in the comparison directory. The separate A/B/B* replay below leaves those inputs in place.

```text
python scripts/healthcare_research.py replay-ab --out data/cache/healthcare_replay/ab
python scripts/healthcare_research.py replay-finbert --out data/cache/healthcare_replay/unified
```

To render local replay results, pass the new directories to the report scripts. The unified PDF renderer needs a local Chromium browser. The original completion launcher and historical fit entry points are retained in Git for provenance.

```text
python outputs/healthcare_window_horizon_comparison/render_comparison.py --out data/cache/healthcare_replay/ab
python outputs/healthcare_unified_research/render_unified.py --results data/cache/healthcare_replay/unified
```

## Research Outputs

| Artifact | Contents |
|---|---|
| [English PDF](outputs/healthcare_unified_research/Healthcare_Linguistic_Alpha_Unified_Research_Report.pdf) and [HTML dashboard](outputs/healthcare_unified_research/dashboard.html) | Unified research report and interactive comparisons |
| [Annual performance](outputs/healthcare_unified_research/annual_model_performance.csv), [increments](outputs/healthcare_unified_research/annual_increments.csv), and [paired inference](outputs/healthcare_unified_research/paired_inference.csv) | Per-year model metrics, baseline contrasts, and confidence intervals |
| [LM selections](outputs/healthcare_window_horizon_comparison/selected_features.csv), [FinBERT selections](outputs/healthcare_unified_research/finbert_selected_features.csv), and [FinBERT forward paths](outputs/healthcare_unified_research/finbert_forward_paths.csv) | Features, estimators, and inner-validation decisions |
| [Methodology](outputs/healthcare_unified_research/finbert_methodology.md), [fold audit](outputs/healthcare_window_horizon_comparison/cv_folds.csv), [aggregate validation](outputs/healthcare_unified_research/aggregate_validation.json), and [integrity manifest](outputs/healthcare_unified_research/integrity.json) | Design, label maturity, and reproducibility evidence |

## Limitations and Future Research

- **Information timing.** Actual transcript delivery and financial-data availability at signal time remain unverified. The first close-to-close return starts at the Day +1 close, so some movement may occur before the signal is formed.
- **Statistical uncertainty.** The study is retrospective. Overlapping return windows, repeated issuers, and many comparisons limit the strength of the evidence. The intervals condition on the fitted models, so uncertainty from feature and estimator selection remains. Partial 2026 has too few independent calendar blocks for time-block inference.
- **Tradability.** The study reports event-level predictive performance. Transaction costs and portfolio construction are outside its scope, and tradable alpha remains unconfirmed.
- **Next comparison.** Separate FinBERT sentiment scores from sentence counts and other transcript structure to measure each block’s contribution on matched events.
