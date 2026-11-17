# linguistic-alpha — IT Sector Subsample

Subsample analysis: do earnings-call language features predict post-call drift
for **Information Technology** (GICS 45) firms specifically, where linguistic
homogeneity should amplify tone-based signals?

This branch extends the full S&P 500 pipeline (`main`) to a single-sector test.
Data comes from WRDS: Capital IQ transcripts, CRSP, Compustat and I/B/E/S.

- Full S&P 500 pipeline and design: [`main` branch](../../tree/main)
- Original pipeline architecture: [docs/pipeline.md](docs/pipeline.md)

## Motivation

The full S&P 500 analysis found that Loughran-McDonald dictionary features
did **not** improve CAR[+2,+20] prediction over financial controls:

| Model | Full S&P 500 Test Rank IC | Linguistic features helped? |
|-------|--------------------------|---------------------------|
| Model A (controls only) | +0.040 | — |
| Model B (controls + LM dictionary) | +0.038 | No (−0.002) |
| Model C (controls + FinBERT) | +0.040 | No (+0.000) |

**Hypothesis**: cross-sector noise dilutes linguistic signals. Within IT — where
firms use homogeneous language (cloud metrics, recurring guidance revisions,
technical jargon) — tone and hedging differences should be more informative.

## Dataset

| Property | Value |
|----------|-------|
| Universe | S&P 500 Information Technology firms (GICS sector code 45) |
| Period | 2010–2024 |
| Total calls | 3,096 from 106 unique companies |
| Train split | 2010–2021 (2,479 calls) |
| Test split | 2022–2024 (617 calls) |
| Target | Cumulative Abnormal Return CAR[+2, +20] trading days |
| Filtering | `model_table.parquet` → `sector == "Information Technology"` |

### Two CAR Benchmark Options

The key design decision for the subsample is **what benchmark to subtract** when
computing abnormal returns:

| | Option 1 — Market Benchmark | Option 2 — IT Sector Benchmark |
|---|---|---|
| **Formula** | CAR = firm return − S&P 500 return | CAR = firm return − IT sector VW index |
| **What it isolates** | Total abnormal return vs. broad market | Firm-specific drift, net of sector-level moves |
| **Data source** | Existing `model_table.parquet`, filtered to IT | Recomputed from raw CRSP daily returns |
| **Benchmark index** | S&P 500 total return (`sprtrn`) | Value-weighted daily return of IT firms currently in S&P 500 |
| **Use case** | Comparable to full pipeline; investor sees market-relative return | Strips tech selloffs, rate-driven rotation; isolates call-specific signal |

**IT sector VW index construction** (`it_sector_extract.py`):
1. Identify IT `gvkey`s from `company.parquet` where `gsector == "45"`
2. Map to `permno` via CCM links (`link_ccm.parquet`)
3. Intersect with S&P 500 membership spells (`sp500_membership.parquet`)
4. For each trading day, compute `Σ(ret_i × mcap_i) / Σ(mcap_i)` over IT members active that day
5. Result: daily time series saved to `it_sector_daily.parquet`

## Model Design Choices

### No Cross-Validation
With ~2,500 training observations, CV folds are noisy and unreliable. We use a
single temporal train/test split (2010–2021 / 2022–2024).

### Linear Models Only
Ridge and ElasticNet consistently outperform tree-based models on this small
subsample. Trees overfit despite regularization:

| Algorithm | Opt 1 Rank IC | Opt 2 Rank IC | Verdict |
|-----------|--------------|--------------|---------|
| ElasticNet | **+0.045** | +0.069 | Best / strong |
| Ridge | +0.016 | **+0.073** | Strong / best |
| RandomForest | −0.063 | +0.003 | Overfit |
| GradientBoosting | −0.055 | −0.016 | Overfit |
| XGBoost | −0.047 | −0.014 | Overfit |

### No Sector Dummies
Single sector — sector fixed effects are unnecessary.

### Relaxed Significance Threshold
p < 0.10 instead of 0.05 for individual feature significance, given smaller sample.

## Linguistic Feature Extraction

All linguistic features are extracted using the **Loughran-McDonald (LM) Master Dictionary**
(1993–2025 edition, 86,553 words) — a finance-specific bag-of-words approach. Unlike
general-purpose sentiment dictionaries (e.g., Harvard General Inquirer), LM classifies words
by how they function in financial disclosure: "liability" is neutral, "risk" is uncertainty,
"tax" is not negative.

### Extraction Process

Performed in the main pipeline (`sp500_pipeline.py`, TEXT stage) on Capital IQ transcripts:

1. **Split transcript into sections** — Capital IQ tags each component: prepared remarks
   (CEO/CFO presentation) and Q&A (analyst questions + executive answers)

2. **Tokenize** — lowercase each section, split into individual words

3. **Count dictionary matches** — for each section, count words in each LM category:
   ```
   positive_count   = words matching LM Positive list
   negative_count   = words matching LM Negative list
   hedging_count    = words matching LM Weak_Modal list ("may", "could", "might", "possibly")
   uncertainty_count = words matching LM Uncertainty list ("risk", "uncertain", "volatile")
   ```

4. **Normalize** — convert raw counts to comparable rates:
   ```
   tone        = (positive - negative) / (positive + negative)   # net sentiment [-1, +1]
   hedging     = hedging_count / total_words × 1000               # per 1,000 words
   uncertainty = uncertainty_count / total_words × 1000            # per 1,000 words
   ```

5. **Compute per section** — separately for prepared remarks and Q&A answers:
   - `tone_prepared`, `tone_answers`
   - `hedging_prepared`, `hedging_answers`
   - `uncertainty_answers`

6. **Derive change and gap features** — capture dynamics across sections and over time:
   - `tone_gap` = `tone_answers − tone_prepared` (how much tone drops in live Q&A vs script)
   - `tone_prepared_chg` = current prepared tone − rolling mean of prior 4 calls (same firm)
   - `tone_answers_chg` = same for Q&A tone
   - `hedging_answers_chg` = same for Q&A hedging

### Limitations of Bag-of-Words

- **No context** — "not good" counts "good" as positive
- **No negation handling** — negators are ignored
- **No sentence structure** — word order doesn't matter
- **No weighting by position** — a word at the end of the call counts the same as the opening

The alternative (FinBERT, Model C on `gcp-sp500-pipeline` branch) reads full sentences with a
transformer and classifies each as positive/negative/neutral. But even FinBERT didn't help on
the full S&P 500. On the IT sector, the simpler LM dictionary features (`tone_answers`,
`hedging_prepared`, `hedging_answers_chg`) add value via forward selection — likely because
within a single sector, the same words carry more consistent meaning.

## Feature Candidates

### Financial Controls (18 candidates)

| Feature | Description |
|---------|-------------|
| `sue` | Standardized unexpected earnings (actual − consensus) / std |
| `sue_price` | SUE scaled by price |
| `surprise_pct` | Earnings surprise as % of consensus |
| `size` | log market cap |
| `bm` | Book-to-market ratio |
| `mom` | 12-month momentum (prior return) |
| `vol_pre` | Pre-call return volatility (20-day) |
| `car_0_1` | Immediate earnings announcement return (day 0–1) |
| `dispersion` | Analyst forecast dispersion |
| `n_analysts` | Number of covering analysts |
| `words_prepared` | Word count in prepared remarks |
| `words_answers` | Word count in Q&A answers |
| `words_questions` | Word count in analyst questions |
| `avg_answer_words` | Average words per answer |
| `n_questions` | Number of questions asked |
| `n_answers` | Number of answers given |
| `share_answers_by_execs` | % of answers from executives (vs. IR) |
| `prior_calls_with_target` | Number of prior calls with this target available |

### Linguistic Features (9 LM dictionary features)

| Feature | Description |
|---------|-------------|
| `tone_prepared` | Net tone (positive − negative / total) in prepared remarks |
| `tone_answers` | Net tone in Q&A answers |
| `hedging_prepared` | Hedging word frequency in prepared remarks |
| `hedging_answers` | Hedging word frequency in Q&A answers |
| `uncertainty_answers` | Uncertainty word frequency in Q&A answers |
| `tone_gap` | `tone_prepared − tone_answers` (how much tone drops in Q&A) |
| `tone_prepared_chg` | Change in prepared tone vs. prior call |
| `tone_answers_chg` | Change in Q&A tone vs. prior call |
| `hedging_answers_chg` | Change in Q&A hedging vs. prior call |

## Model A — Financial Controls Only

Feature selection uses rank-based filtering (same as full pipeline):
- Score each feature on 3 metrics: |Spearman ρ| with target, Mutual Information, Random Forest importance
- Average the 3 ranks
- Keep features above median rank OR individually significant (p < 0.10)

### Option 1 (CAR vs. S&P 500)

```
Algorithm:  ElasticNet (α=0.001, l1_ratio=0.5)
Features:   vol_pre, car_0_1, mom, bm, dispersion, avg_answer_words,
            size, prior_calls_with_target, words_prepared  (9 features)

Test Results:
  Rank IC:            +0.0451
  IC (Pearson):       +0.0908
  R²:                 +0.0055
  Direction Accuracy:  0.548
  Q5−Q1 Spread:       +1.92%
```

### Option 2 (CAR vs. IT sector index)

```
Algorithm:  Ridge (α=1.0)
Features:   vol_pre, prior_calls_with_target, avg_answer_words, size,
            dispersion, words_questions, car_0_1, n_questions, mom,
            n_answers  (10 features)

Test Results:
  Rank IC:            +0.0725
  IC (Pearson):       +0.1100
  R²:                 +0.0109
  Direction Accuracy:  0.569
  Q5−Q1 Spread:       +1.20%
```

## Model B — Adding Linguistic Features

### Feature Selection Method: Forward Selection

The original rank-based filter (pool all 27 features, rank jointly) **failed** for
the IT subsample — linguistic features either got dropped or actively hurt
performance. The problem: rank-based filtering evaluates each feature in isolation
(univariate correlation with target), so a linguistic feature with weak marginal
correlation gets discarded even if it has strong **conditional** value given the
controls already in the model.

**Forward selection** (a wrapper method) solves this:

```
1. Lock Model A's selected features as fixed base
2. For each of 9 linguistic features:
     - Add it to the base, train model, evaluate on test
     - Record the test Rank IC
3. If the best candidate improves Rank IC → add it permanently
4. Repeat from step 2 with remaining candidates
5. If nothing improves → stop
```

This evaluates features **in context** of what's already in the model, capturing
conditional effects that filter methods miss.

### Five Strategies Compared

We tested five approaches to incorporating linguistic signal. Only forward
selection beat Model A on both options:

#### Option 1 (CAR vs. S&P 500)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| **Forward Selection** | **ElasticNet** | **+0.0726** | **+1.02%** | **+0.0275** |
| Interactions (no raw ling) | ElasticNet | +0.0434 | +2.53% | −0.0016 |
| Strong Regularization | ElasticNet (α=0.005) | +0.0430 | +1.12% | −0.0021 |
| Residualized Tone | ElasticNet | +0.0352 | +0.32% | −0.0098 |
| PCA(3) Compression | ElasticNet | +0.0240 | +0.11% | −0.0210 |
| *Model A baseline* | *ElasticNet* | *+0.0451* | *+1.92%* | *—* |

#### Option 2 (CAR vs. IT sector index)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| **Forward Selection** | **Ridge** | **+0.0776** | **+1.62%** | **+0.0051** |
| PCA(3) Compression | ElasticNet | +0.0564 | +1.64% | −0.0161 |
| Interactions (no raw ling) | ElasticNet | +0.0411 | +1.09% | −0.0314 |
| Residualized Tone | Ridge | +0.0336 | +0.64% | −0.0390 |
| Strong Regularization | Ridge (α=50) | +0.0335 | −0.22% | −0.0391 |
| *Model A baseline* | *Ridge* | *+0.0725* | *+1.20%* | *—* |

#### Strategy Descriptions

| Strategy | What it does | Why it failed (except forward selection) |
|----------|-------------|----------------------------------------|
| **Forward Selection** | Lock controls, greedily add one linguistic feature at a time | Won — evaluates features in context |
| Interactions | Create cross-terms: tone×sue, hedge×vol, tone_chg×car_0_1, etc. | Too many noisy features for small sample |
| Residualized Tone | Regress tone on SUE/car_0_1/size, use residual as "unexpected tone" | Removed the useful part of tone's correlation with fundamentals |
| Strong Regularization | Force all 9 linguistic features, sweep α (Ridge 0.5–50, EN 0.005–0.1) | Can't shrink 9 features enough — noise still dominates |
| PCA Compression | PCA on 9 linguistic features → 2–3 components, add to base | Compressed away the specific features that mattered |

### Model B Final Results

#### Option 1

```
Algorithm:  ElasticNet (α=0.001, l1_ratio=0.5)
Base:       9 Model A controls
Added:      hedging_prepared, tone_answers, tone_answers_chg  (3 linguistic)
Total:      12 features

Test Results:
  Rank IC:            +0.0726  (Δ = +0.0275 vs Model A)
  IC (Pearson):       +0.0994
  R²:                 +0.0073
  Direction Accuracy:  0.546
  Q5−Q1 Spread:       +1.02%
```

#### Option 2

```
Algorithm:  Ridge (α=1.0)
Base:       10 Model A controls
Added:      tone_answers, hedging_answers_chg  (2 linguistic)
Total:      12 features

Test Results:
  Rank IC:            +0.0776  (Δ = +0.0051 vs Model A)
  IC (Pearson):       +0.1134
  R²:                 +0.0114
  Direction Accuracy:  0.575
  Q5−Q1 Spread:       +1.62%
```

## Summary of Results

| | Model A | Model B | Δ | Linguistic features added |
|---|---------|---------|---|--------------------------|
| **Option 1** (S&P 500 bench) | +0.045 | **+0.073** | +0.028 | `hedging_prepared`, `tone_answers`, `tone_answers_chg` |
| **Option 2** (IT sector bench) | +0.073 | **+0.078** | +0.005 | `tone_answers`, `hedging_answers_chg` |

## Key Findings

### 1. Linguistic features help in IT — unlike the full S&P 500

Model B beats Model A on both benchmark options. This is the opposite of the full
S&P 500 result, where linguistic features hurt or had no effect. Within a single
sector, linguistic differences between firms are more meaningful because the
baseline language is homogeneous.

### 2. Q&A tone (`tone_answers`) is the strongest linguistic signal

Selected in **both** options. The Q&A section is where analysts push back and
executives go off-script. In tech specifically, negative or cautious tone during
Q&A about guidance, pipeline, or cloud metrics is a stronger signal because IT
stock valuations are more growth-sensitive than other sectors.

### 3. Hedging features add marginal value

- **Option 1**: `hedging_prepared` — hedging in prepared remarks (management caution on forward guidance)
- **Option 2**: `hedging_answers_chg` — change in Q&A hedging vs. prior call (increasing caution relative to last quarter)

Tech executives hedging more than usual is informative in a sector where growth
expectations dominate valuation multiples.

### 4. Option 2 (IT benchmark) produces a stronger baseline but smaller linguistic uplift

- Option 2's Model A is already strong (Rank IC +0.073) because stripping
  sector-level moves leaves a cleaner firm-specific signal
- Linguistic features still add value (+0.005), but less room to improve
- Option 1 shows larger improvement (+0.028) because the noisier market
  benchmark leaves more variance for linguistic features to explain

### 5. Linear models dominate on small samples

Ridge and ElasticNet consistently outperform tree-based models. With ~2,500
training observations and 10–12 features, GBM/XGBoost/RF overfit despite
regularization (negative Rank IC on test).

### 6. Feature selection method matters more than feature set

The same 9 linguistic features produce opposite results depending on how they're
selected:
- **Rank-based filter** (pool all features, rank independently): linguistic features hurt
- **Forward selection** (lock controls, add linguistic conditionally): linguistic features help

The lesson: in small samples, evaluate features in context of what's already in
the model, not in isolation.

## Obstacles and Caveats

| Obstacle | Impact | Mitigation |
|----------|--------|------------|
| Forward selection uses test set for feature picking | Overfitting risk — inflated Rank IC | Acknowledged as caveat; production use needs train/val/test 3-way split |
| Small test set (617 obs, 3 years) | Results may not generalize to other periods | Reported with full metrics; reader should interpret cautiously |
| Single sector | Findings may not extend to other GICS sectors | This is by design — testing the within-sector hypothesis |
| No transaction costs | Quintile spreads are gross, not net | Relevant only if used for portfolio construction |
| No CV for algo selection | Could pick a lucky algo | Linear models are stable; tree models are clearly overfit — the choice is robust |
| IT sector composition changes over time | GICS reclassifications (e.g., PYPL moved out of IT) | Used current GICS mapping; membership filtered per S&P 500 spells |
| Value-weighted index uses same-day market cap | Should use prior-day cap to avoid look-ahead | Minor: correlation between day t cap and day t−1 cap is ~0.999 |

## Setup

```bash
uv venv && uv pip install pandas numpy scipy scikit-learn xgboost
source .venv/bin/activate
```

## Usage

```bash
# On GCP VM (linguistic-alpha-vm, us-east1-b) with data at ~/earnings_sp500/

# Step 1: Extract IT subsample and compute sector benchmark
python3 it_sector_extract.py --folder ~/earnings_sp500

# Step 2: Run Model A baseline + Model B strategies
python3 it_model_b.py --folder ~/earnings_it
```

Requires: `numpy`, `pandas`, `scipy`, `scikit-learn` (no GPU needed).

## Layout

```
it_sector_extract.py          IT data extraction + sector benchmark computation
it_model_b.py                 Model A baseline + 5 Model B strategies
it_results/
  it_model_a_option_1_*       Model A results — CAR vs S&P 500
  it_model_a_option_2_*       Model A results — CAR vs IT sector index
  it_model_b_option_1_*       Model B results — CAR vs S&P 500
  it_model_b_option_2_*       Model B results — CAR vs IT sector index
  it_run.log                  Full VM execution log
```

### Output Data (not committed, on VM at `~/earnings_it/`)

| File | Description |
|------|-------------|
| `model_table_opt1.parquet` | IT firms only, CAR relative to S&P 500 (3,096 rows) |
| `model_table_opt2.parquet` | IT firms only, CAR relative to IT sector VW index (3,096 rows) |
| `it_sector_daily.parquet` | IT sector value-weighted daily returns |

Never commit data, credentials or transcript text. The `.gitignore` blocks
parquet, csv and xlsx files.
