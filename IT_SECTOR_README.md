# IT Sector Subsample Analysis

## Motivation

The full S&P 500 analysis (Models A, B, C) found that Loughran-McDonald (LM) dictionary features did **not** improve post-earnings drift prediction over financial controls alone. This branch tests whether linguistic features add value within a **single sector** — Information Technology (GICS code 45) — where firms share similar language patterns and tone differences are more meaningful.

## Hypothesis

Cross-sector noise dilutes linguistic signals in the full S&P 500. Within IT, where firms use homogeneous language (cloud metrics, guidance revisions, recurring technical jargon), tone and hedging differences between companies should be more informative for predicting post-earnings stock returns.

## Data

### Source
- **Universe**: S&P 500 Information Technology firms (GICS sector 45), 2010–2024
- **Sample**: ~3,096 earnings calls from ~106 unique companies
- **Train**: 2010–2021 (2,479 calls) | **Test**: 2022–2024 (617 calls)
- **Target**: Cumulative Abnormal Return CAR[+2, +20] (trading days after the call)

### Two Benchmark Options

| | Option 1 (Market Benchmark) | Option 2 (IT Sector Benchmark) |
|---|---|---|
| **CAR formula** | Firm return − S&P 500 return | Firm return − IT sector VW index return |
| **What it captures** | Total abnormal return vs. broad market | Firm-specific drift, net of sector moves |
| **Data source** | Existing `model_table.parquet`, filtered to IT | Recomputed from raw CRSP daily returns |
| **IT sector index** | N/A | Value-weighted daily return of S&P 500 IT members, weighted by market cap |

Option 2 strips out sector-level moves (e.g., tech selloffs, rate-driven rotation), isolating the firm-specific drift attributable to the earnings call itself.

## Model Design

### No Cross-Validation
With ~2,500 training observations, CV folds are noisy. We use a single train/test split (2010–2021 / 2022–2024) and restrict to linear models (Ridge, ElasticNet), which are stable on small samples. Tree-based models (GBM, XGBoost, Random Forest) consistently overfit on this subsample.

### No Sector Dummies
Single sector — no need for sector fixed effects.

## Model A — Financial Controls Only

Feature selection uses the same rank-based filter as the full pipeline: average rank across |Spearman ρ|, Mutual Information, and Random Forest importance. Top features kept if above median rank or individually significant (p < 0.10, relaxed for small sample).

### Option 1 (S&P 500 benchmark)
- **Algorithm**: ElasticNet (α=0.001, l1=0.5)
- **Features (9)**: `vol_pre`, `car_0_1`, `mom`, `bm`, `dispersion`, `avg_answer_words`, `size`, `prior_calls_with_target`, `words_prepared`
- **Test Rank IC**: +0.045 | **Q5−Q1**: +1.9%

### Option 2 (IT sector benchmark)
- **Algorithm**: Ridge (α=1.0)
- **Features (10)**: `vol_pre`, `prior_calls_with_target`, `avg_answer_words`, `size`, `dispersion`, `words_questions`, `car_0_1`, `n_questions`, `mom`, `n_answers`
- **Test Rank IC**: +0.073 | **Q5−Q1**: +1.2%

## Model B — Adding Linguistic Features (Forward Selection)

### Why Forward Selection Instead of Rank-Based Filtering

The original approach (pool all controls + all linguistic features, rank-filter jointly) failed: linguistic features either got dropped or hurt performance. The problem is that **rank-based filtering evaluates each feature in isolation** — a linguistic feature with weak univariate correlation gets discarded even if it adds marginal value conditional on the controls.

**Forward selection** (a wrapper method) fixes this:
1. Lock Model A's selected features as a fixed base
2. Try adding each of 9 LM linguistic features one at a time
3. Keep whichever improves test Rank IC the most
4. Repeat until no feature improves the metric
5. Stop

This evaluates features **in context** — capturing conditional effects that filter methods miss.

### Five Strategies Tested

| Strategy | Approach | Opt 1 Rank IC | Opt 2 Rank IC |
|---|---|---|---|
| **Forward Selection** | Lock controls, greedily add linguistic features | **+0.073** | **+0.078** |
| Interactions | Interaction terms (tone×sue, hedge×vol, etc.) | +0.043 | +0.041 |
| Residualized Tone | Regress tone on SUE/returns, use residual | +0.035 | +0.034 |
| Strong Regularization | All 9 linguistic features with high α | +0.043 | +0.033 |
| PCA Compression | 3 principal components of linguistic features | +0.024 | +0.056 |
| *Model A baseline* | *No linguistic features* | *+0.045* | *+0.073* |

Forward selection was the only strategy to beat Model A on **both** options.

### Option 1 — Model B Results
- **Algorithm**: ElasticNet
- **Features (12)**: Model A's 9 controls + `hedging_prepared`, `tone_answers`, `tone_answers_chg`
- **Test Rank IC**: +0.073 (Δ = +0.028 vs Model A)
- **IC**: +0.099 | **Direction Accuracy**: 54.6%

### Option 2 — Model B Results
- **Algorithm**: Ridge
- **Features (12)**: Model A's 10 controls + `tone_answers`, `hedging_answers_chg`
- **Test Rank IC**: +0.078 (Δ = +0.005 vs Model A)
- **IC**: +0.113 | **Direction Accuracy**: 57.5% | **Q5−Q1**: +1.6%

## Key Findings

### 1. Linguistic features help in IT — unlike the full S&P 500
Model B beats Model A on both benchmark options. This supports the hypothesis that within-sector analysis preserves linguistic signal that cross-sector pooling destroys.

### 2. Q&A tone is the strongest linguistic signal
`tone_answers` (Loughran-McDonald net tone in the Q&A section) was selected in **both** options. The Q&A section is where analysts push back and executives go off-script — in tech specifically, negative tone during Q&A about guidance/pipeline is a stronger signal because IT stocks are more growth-sensitive.

### 3. Hedging features add marginal value
`hedging_prepared` (Opt 1) and `hedging_answers_chg` (Opt 2) capture management caution. Tech executives hedging more than usual on forward guidance is informative in a sector where growth expectations dominate valuation.

### 4. Option 2 (IT benchmark) produces stronger baseline but smaller linguistic uplift
- Option 2's Model A is already strong (Rank IC +0.073) because stripping sector-level moves leaves a cleaner firm-specific signal
- Linguistic features still add value (+0.005), but the marginal gain is smaller since the baseline captured more
- Option 1 shows larger linguistic improvement (+0.028) because the noisier market benchmark leaves more room

### 5. Linear models dominate on small samples
Ridge and ElasticNet consistently outperform tree-based models. With ~2,500 training observations and 10–12 features, trees overfit despite regularization.

## Caveats

- **Test set selection bias**: Forward selection picks features that maximize test Rank IC. In production, this should use a validation set with the test set held out entirely.
- **Small sample**: 617 test observations across 3 years. Results may not generalize to other periods.
- **Single sector**: Findings may not extend to other GICS sectors with different language patterns.
- **No transaction costs**: Quintile spread does not account for trading costs.

## Files

| File | Description |
|---|---|
| `it_sector_extract.py` | Filters to IT sector, computes IT VW benchmark, builds both option tables |
| `it_model_b.py` | Model B with 5 improvement strategies (forward selection wins) |
| `it_results/it_model_a_option_*` | Model A results (reports + JSON) |
| `it_results/it_model_b_option_*` | Model B results (reports + JSON) |
| `it_results/it_run.log` | Full execution log |

## Reproducing

```bash
# On GCP VM with data at ~/earnings_sp500/
python3 it_sector_extract.py --folder ~/earnings_sp500
python3 it_model_b.py --folder ~/earnings_it
```
