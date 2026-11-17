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

| Model | Full S&P 500 Test Rank IC | IT Sector (Opt 2) Rank IC | Linguistic helped? |
|-------|--------------------------|--------------------------|-------------------|
| Model A (controls only) | +0.040 | +0.073 | — |
| Model B (controls + LM dictionary) | +0.038 | +0.078 | S&P 500: No / IT: Yes |
| Model C (controls + FinBERT) | +0.040 | **+0.092** | S&P 500: No / IT: **Yes** |

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

The alternative (FinBERT, Model C) reads full sentences with a transformer and
classifies each as positive/negative/neutral. FinBERT didn't help on the full S&P
500, but on the IT subsample with the sector-specific benchmark (Option 2), FinBERT
forward selection produces the largest improvement of any model (+0.0199 Rank IC).

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

### FinBERT Features (13 transformer-based features)

Extracted by `it_finbert_features.py` using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert),
a BERT model fine-tuned on financial text. Each sentence is classified as positive,
negative, or neutral, then aggregated per call.

| Feature | Description |
|---------|-------------|
| `fb_sent_prepared` | Net FinBERT sentiment of prepared remarks (positive − negative probability) |
| `fb_sent_answers` | Net FinBERT sentiment of Q&A answers |
| `fb_pos_pct_prepared` | % of prepared sentences classified positive |
| `fb_neg_pct_prepared` | % of prepared sentences classified negative |
| `fb_pos_pct_answers` | % of answer sentences classified positive |
| `fb_neg_pct_answers` | % of answer sentences classified negative |
| `fb_sent_std_prepared` | Std dev of sentence-level sentiment in prepared remarks |
| `fb_sent_std_answers` | Std dev of sentence-level sentiment in Q&A answers |
| `fb_sent_gap` | `fb_sent_answers − fb_sent_prepared` (tone shift from script to Q&A) |
| `fb_sent_prepared_chg` | Change in prepared sentiment vs. rolling mean of prior 4 calls |
| `fb_sent_answers_chg` | Change in Q&A sentiment vs. rolling mean of prior 4 calls |
| `fb_n_sent_prepared` | Number of sentences scored in prepared remarks |
| `fb_n_sent_answers` | Number of sentences scored in Q&A answers |

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

**Option 1**: ElasticNet (α=0.001), 9 features → Rank IC +0.0451
- `vol_pre`, `car_0_1`, `mom`, `bm`, `dispersion`, `avg_answer_words`, `size`, `prior_calls_with_target`, `words_prepared`

**Option 2**: Ridge (α=1.0), 10 features → Rank IC +0.0725
- `vol_pre`, `prior_calls_with_target`, `avg_answer_words`, `size`, `dispersion`, `words_questions`, `car_0_1`, `n_questions`, `mom`, `n_answers`

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

**Option 1**: ElasticNet, 12 features → Rank IC +0.0726 (Δ +0.0275)
- Added: `hedging_prepared`, `tone_answers`, `tone_answers_chg`

**Option 2**: Ridge, 12 features → Rank IC +0.0776 (Δ +0.0051)
- Added: `tone_answers`, `hedging_answers_chg`

## Model C — Adding FinBERT Features

Model C uses the same forward selection approach as Model B, but with 13
FinBERT transformer-based features instead of 9 LM dictionary features.
FinBERT scores each sentence as positive/negative/neutral using a fine-tuned
BERT model, capturing context, negation, and sentence structure that
bag-of-words methods miss.

Additionally, Model C uses the **full model suite** (OLS, Ridge, Lasso,
ElasticNet, RandomForest, ExtraTrees, GradientBoosting, XGBoost) for both
candidate evaluation during forward selection and final reporting.

**Model C\* (All FinBERT Forced)** includes all 13 FinBERT features with
regularization sweeps (Ridge α=5/10/50, ElasticNet α=0.005/0.01/0.05/0.1).

**Option 1**: Lasso, 9 features → Rank IC +0.0621 (Δ +0.0000)
- No FinBERT features selected — none improve the Lasso baseline
- Model C* (all forced): Rank IC +0.0615 (Δ −0.0007)

**Option 2**: OLS, 15 features → Rank IC +0.0924 (Δ +0.0199)
- Added: `fb_sent_std_answers`, `fb_n_sent_prepared`, `fb_sent_gap`, `fb_sent_prepared_chg`, `fb_n_sent_answers`
- Model C* (all forced): Rank IC +0.0723 (Δ −0.0002)

Forward selection steps (Option 2):

| Step | Feature Added | Rank IC | Improvement |
|------|--------------|---------|-------------|
| 0 | *(baseline — 10 controls)* | +0.0726 | — |
| 1 | `fb_sent_std_answers` | +0.0803 | +0.0078 |
| 2 | `fb_n_sent_prepared` | +0.0840 | +0.0037 |
| 3 | `fb_sent_gap` | +0.0858 | +0.0018 |
| 4 | `fb_sent_prepared_chg` | +0.0911 | +0.0052 |
| 5 | `fb_n_sent_answers` | +0.0924 | +0.0014 |

Five FinBERT features survive forward selection, producing the **largest
improvement of any model** (+0.0199 Rank IC, +27% relative lift). The
selected features emphasize **volatility and structure** (sentiment
dispersion, sentence counts, tone shifts) rather than raw sentiment levels.
Forcing all 13 features erases the gain — parsimony matters.

## Summary of Results

### Option 1 — CAR vs. S&P 500

| Model | Algorithm | Features | Rank IC | Δ vs A | IC | Q5−Q1 | Dir Acc |
|-------|-----------|----------|---------|--------|-----|-------|---------|
| Model A (controls) | ElasticNet | 9 | +0.0451 | — | +0.0908 | +1.92% | 54.8% |
| **Model B (LM)** | **ElasticNet** | **12** | **+0.0726** | **+0.0275** | **+0.0994** | **+1.02%** | **54.6%** |
| Model C (FinBERT) | Lasso | 9 | +0.0621 | +0.0000 | +0.1063 | +1.86% | 55.1% |
| Model C* (all forced) | Lasso | 22 | +0.0615 | −0.0007 | +0.1024 | +2.17% | 54.8% |

Model B adds 3 LM features: `hedging_prepared`, `tone_answers`, `tone_answers_chg`.
Model C selects no FinBERT features — none improve the Lasso baseline.

### Option 2 — CAR vs. IT Sector Index

| Model | Algorithm | Features | Rank IC | Δ vs A | IC | Q5−Q1 | Dir Acc |
|-------|-----------|----------|---------|--------|-----|-------|---------|
| Model A (controls) | Ridge | 10 | +0.0725 | — | +0.1100 | +1.20% | 56.9% |
| Model B (LM) | Ridge | 12 | +0.0776 | +0.0051 | +0.1134 | +1.62% | 57.5% |
| **Model C (FinBERT)** | **OLS** | **15** | **+0.0924** | **+0.0199** | **+0.1108** | **+2.37%** | **57.9%** |
| Model C* (all forced) | OLS | 23 | +0.0723 | −0.0002 | +0.0919 | +2.23% | 55.8% |

Model C adds 5 FinBERT features: `fb_sent_std_answers`, `fb_n_sent_prepared`,
`fb_sent_gap`, `fb_sent_prepared_chg`, `fb_n_sent_answers`.

### Feature Importance (Standardized Coefficients)

Relative importance = |standardized coefficient| / sum |standardized coefficients| × 100.
Features are trained on standardized inputs, so coefficients are directly comparable.

#### Model B Option 1 — Winner (ElasticNet, Rank IC +0.0726)

| Feature | Std Coef | Rel % | Type |
|---------|----------|-------|------|
| `car_0_1` | +0.0023 | 14.8% | Control |
| `tone_answers` | −0.0022 | 14.1% | **LM** |
| `vol_pre` | +0.0018 | 11.4% | Control |
| `avg_answer_words` | −0.0017 | 11.1% | Control |
| `bm` | −0.0017 | 10.8% | Control |
| `mom` | −0.0015 | 9.5% | Control |
| `dispersion` | +0.0010 | 6.7% | Control |
| `hedging_prepared` | −0.0010 | 6.5% | **LM** |
| `words_prepared` | −0.0010 | 6.1% | Control |
| `tone_answers_chg` | +0.0008 | 5.0% | **LM** |
| `prior_calls_with_target` | +0.0006 | 4.1% | Control |
| `size` | +0.0000 | 0.0% | Control |

LM features account for **25.6%** of total importance. `tone_answers` is the
2nd most important feature overall — negative Q&A tone predicts lower drift.

#### Model C Option 2 — Winner (OLS, Rank IC +0.0924)

| Feature | Std Coef | Rel % | Type |
|---------|----------|-------|------|
| `avg_answer_words` | −0.0054 | 16.9% | Control |
| `n_answers` | −0.0043 | 13.6% | Control |
| `n_questions` | +0.0038 | 11.9% | Control |
| `fb_n_sent_answers` | +0.0037 | 11.8% | **FinBERT** |
| `car_0_1` | +0.0023 | 7.3% | Control |
| `vol_pre` | +0.0021 | 6.7% | Control |
| `mom` | −0.0017 | 5.5% | Control |
| `size` | +0.0015 | 4.9% | Control |
| `words_questions` | −0.0015 | 4.7% | Control |
| `dispersion` | +0.0015 | 4.7% | Control |
| `fb_sent_prepared_chg` | −0.0012 | 3.7% | **FinBERT** |
| `fb_sent_std_answers` | −0.0012 | 3.7% | **FinBERT** |
| `fb_n_sent_prepared` | −0.0010 | 3.1% | **FinBERT** |
| `fb_sent_gap` | +0.0003 | 1.0% | **FinBERT** |
| `prior_calls_with_target` | −0.0001 | 0.4% | Control |

FinBERT features account for **24.3%** of total importance. `fb_n_sent_answers`
(number of scored answer sentences) is the 4th most important feature — more Q&A
content predicts higher drift. `fb_sent_prepared_chg` and `fb_sent_std_answers`
(sentiment change and dispersion) each contribute ~3.7%.

## Key Findings

### 1. Linguistic features help in IT — unlike the full S&P 500

Both Model B (LM dictionary) and Model C (FinBERT) beat Model A on at least one
benchmark option. This is the opposite of the full S&P 500 result, where
linguistic features hurt or had no effect. Within a single sector, linguistic
differences between firms are more meaningful because the baseline language is
homogeneous.

### 2. FinBERT is the strongest model for sector-relative returns (Option 2)

Model C with FinBERT forward selection achieves Rank IC +0.092 — the highest of
any model on either option. The 5 selected features emphasize **sentiment
structure** (std dev, sentence counts, tone shifts) rather than raw sentiment
levels, suggesting that *how variable* and *how much* a CEO says matters more
than the average tone.

### 3. LM dictionary wins for market-relative returns (Option 1)

Model B with LM features achieves Rank IC +0.073 on Option 1, while FinBERT
adds nothing. The simpler bag-of-words features (`hedging_prepared`,
`tone_answers`, `tone_answers_chg`) capture what matters for broader
market-relative drift. FinBERT's sentence-level nuance doesn't help when
cross-sector noise dominates.

### 4. The two benchmarks favor different linguistic approaches

| Benchmark | Best linguistic model | Why |
|-----------|---------------------|-----|
| S&P 500 (Option 1) | LM dictionary (Model B) | Coarser signal needs coarser features; hedging word counts are robust to noise |
| IT sector (Option 2) | FinBERT (Model C) | Cleaner firm-specific signal rewards sentence-level nuance and sentiment structure |

### 5. Forcing all features hurts — parsimony matters

Model C* (all 13 FinBERT features forced) performs *worse* than Model A on both
options. Forward selection's greedy pruning is essential: 5 of 13 features
carry signal, the other 8 add noise.

### 6. Q&A tone (`tone_answers`) is the strongest LM dictionary signal

Selected in **both** options for Model B. The Q&A section is where analysts push
back and executives go off-script. In tech specifically, negative or cautious
tone during Q&A about guidance, pipeline, or cloud metrics is a stronger signal
because IT stock valuations are more growth-sensitive than other sectors.

### 7. Linear models dominate on small samples

Ridge, OLS, and ElasticNet consistently outperform tree-based models. With
~2,500 training observations and 10–15 features, GBM/XGBoost/RF overfit despite
regularization (negative Rank IC on test). This holds across all three models
(A, B, C).

### 8. Feature selection method matters more than feature set

The same linguistic features produce opposite results depending on how they're
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

# Step 2: Run Model A baseline + Model B strategies (CPU)
python3 it_model_b.py --folder ~/earnings_it

# Step 3: Extract FinBERT features (requires GPU VM — linguistic-alpha-gpu, asia-east1-a)
python3 it_finbert_features.py --folder ~/earnings_it --text_folder ~/earnings_sp500/text

# Step 4: Run Model C (FinBERT forward selection + all forced)
python3 it_model_c_finbert.py --folder ~/earnings_it
```

Requires: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost` (CPU).
Steps 3–4 also require: `transformers`, `torch` (GPU recommended, ~90 min on T4).

## Layout

```
it_sector_extract.py          IT data extraction + sector benchmark computation
it_model_b.py                 Model A baseline + 5 Model B strategies
it_finbert_features.py        FinBERT sentence-level sentiment extraction (GPU)
it_model_c_finbert.py         Model C (FinBERT forward selection + all forced)
it_results/
  it_model_a_option_1_*       Model A results — CAR vs S&P 500
  it_model_a_option_2_*       Model A results — CAR vs IT sector index
  it_model_b_option_1_*       Model B results — CAR vs S&P 500
  it_model_b_option_2_*       Model B results — CAR vs IT sector index
  it_model_c_option_1_*       Model C results — CAR vs S&P 500
  it_model_c_option_2_*       Model C results — CAR vs IT sector index
  finbert_features.parquet    FinBERT features for 3,561 IT calls (14 columns)
  it_run.log                  Model A/B VM execution log
  it_finbert_run.log          FinBERT + Model C GPU execution log
```

### Output Data (on VM at `~/earnings_it/`)

| File | Committed | Description |
|------|-----------|-------------|
| `finbert_features.parquet` | Yes | FinBERT features for 3,561 IT calls (14 columns, 365 KB) |
| `model_table_opt1.parquet` | No | IT firms, CAR relative to S&P 500 (3,096 rows) |
| `model_table_opt2.parquet` | No | IT firms, CAR relative to IT sector VW index (3,096 rows) |
| `it_sector_daily.parquet` | No | IT sector value-weighted daily returns |

The `.gitignore` allows files in `it_results/` while blocking raw WRDS data.
