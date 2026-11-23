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
did **not** improve CAR[+2,+20] prediction over financial controls.

**Hypothesis**: cross-sector noise dilutes linguistic signals. Within IT — where
firms use homogeneous language (cloud metrics, recurring guidance revisions,
technical jargon) — tone and hedging differences should be more informative.

## Dataset

| Property | Value |
|----------|-------|
| Universe | S&P 500 Information Technology firms (GICS sector code 45) |
| Period | 2010–2026 |
| Total calls | 3,561 from ~106 unique companies |
| Train split | 2010–2023 (2,892 calls) |
| Test split | 2024–2026 (204 calls) |
| Target | Cumulative Abnormal Return CAR[+2, +20] trading days |
| Day 0 | Assigned via 4pm ET cutoff: calls before 4pm → day0 = call date; after 4pm → day0 = next trading day |

### Two CAR Benchmark Options

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

### Time-Series Expanding-Window Cross-Validation

All feature selection and hyperparameter decisions use time-series CV to avoid
look-ahead bias. Three expanding folds:

| Fold | Train | Validation |
|------|-------|------------|
| 1 | 2010–2020 | 2021 |
| 2 | 2010–2021 | 2022 |
| 3 | 2010–2022 | 2023 |

Final evaluation is on the held-out test set (2024–2026, 204 observations),
which is never used during feature selection or model tuning.

### Ranked Target Training

Models train on `rankdata(y_train)` to directly optimize Rank IC (Spearman
correlation) rather than MSE. This is the primary evaluation metric.

### Linear Models Only

Ridge and ElasticNet consistently outperform tree-based models on this
subsample. Trees overfit despite regularization.

### No Sector Dummies

Single sector — sector fixed effects are unnecessary.

## Feature Candidates

### Financial Controls (16 features — all used in Model A)

| Feature | Description |
|---------|-------------|
| `sue` | Standardized unexpected earnings (actual − consensus) / std |
| `surprise_pct` | Earnings surprise as % of consensus |
| `size` | log market cap |
| `bm` | Book-to-market ratio |
| `mom` | 12-month momentum (prior return) |
| `vol_pre` | Pre-call return volatility (20-day) |
| `car_-1_1` | Immediate earnings announcement return (day −1 to +1) |
| `dispersion` | Analyst forecast dispersion |
| `n_analysts` | Number of covering analysts |
| `words_prepared` | Word count in prepared remarks |
| `words_answers` | Word count in Q&A answers |
| `words_questions` | Word count in analyst questions |
| `avg_answer_words` | Average words per answer |
| `n_questions` | Number of questions asked |
| `n_answers` | Number of answers given |
| `prior_calls_with_target` | Number of prior calls with this target available |

### LM Dictionary Features (9 features)

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

## Linguistic Feature Extraction

All LM dictionary features are extracted using the **Loughran-McDonald (LM) Master Dictionary**
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
classifies each as positive/negative/neutral, capturing context and negation that
bag-of-words methods miss.

## Model A — Financial Controls Only

All 16 control features are included without feature selection. Regularized
linear models (Ridge, ElasticNet) handle feature weighting automatically,
avoiding the overfitting risk of greedy feature selection on small CV folds.

**Option 1** (S&P 500 bench): EN_strong → Rank IC **+0.0379**

**Option 2** (IT sector bench): RidgeCV → Rank IC **+0.0682**

## Model B — Adding LM Dictionary Features

Model B adds LM dictionary features on top of Model A's 16 controls. Six
strategies are compared; the best becomes Model B. Additionally, Model B*
forces all 9 LM features in with Ridge regularization.

### Six Strategies

| # | Strategy | What it does |
|---|----------|-------------|
| 1 | Forward Selection | Lock controls, greedily add LM features via time-series CV |
| 2 | Interactions | Cross-terms: tone×sue, hedge×vol, tone_chg×car_-1_1, etc. |
| 3 | Residualized Tone | Regress tone on SUE/car_-1_1/size, use residuals |
| 4 | All Ling + RidgeCV | Force all 9 LM features, sweep regularization |
| 5 | PCA Compression | PCA on 9 LM features → 2–4 components |
| 6 | Single-Feature Ensemble | Build separate models each with 1 LM feature, average predictions |

### Option 1 Results (CAR vs. S&P 500)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| *Model A baseline* | *EN_strong* | *+0.0379* | *+0.99%* | *—* |
| **Single Best (tone_prepared)** | **RidgeCV** | **+0.0374** | **+2.36%** | **−0.0005** |
| PCA(3) | RidgeCV | +0.0131 | +1.85% | −0.0248 |
| Residualized Tone | RidgeCV | −0.0100 | +0.47% | −0.0479 |
| Forward Selection (6 LM) | RidgeCV | −0.0184 | +1.61% | −0.0563 |
| Interactions | EN_strong | −0.0211 | +1.62% | −0.0590 |
| All Ling + RidgeCV | RidgeCV | −0.0274 | +1.34% | −0.0653 |
| **Model B\*** (all 9 LM) | RidgeCV | **−0.0401** | +0.88% | **−0.0780** |

**Model B** = Single Best (tone_prepared): 16 controls + `tone_prepared` → Rank IC +0.0374.
Essentially flat with Model A — adding just one LM feature neither helps nor hurts.

### Option 2 Results (CAR vs. IT sector index)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| *Model A baseline* | *RidgeCV* | *+0.0682* | *+1.40%* | *—* |
| **Single-Feature Ensemble** | **Ensemble** | **+0.0692** | **+1.20%** | **+0.0010** |
| PCA(2) | RidgeCV | +0.0649 | +1.01% | −0.0033 |
| Residualized Tone | EN_strong | +0.0592 | +1.52% | −0.0090 |
| Forward Selection (4 LM) | RidgeCV | +0.0518 | +1.54% | −0.0164 |
| All Ling + RidgeCV | Ridge_a100 | +0.0401 | +0.07% | −0.0281 |
| Interactions | EN_strong | +0.0140 | +1.43% | −0.0542 |
| **Model B\*** (all 9 LM) | RidgeCV | **+0.0401** | +0.07% | **−0.0281** |

**Model B** = Single-Feature Ensemble: averages predictions from 5 models each
with one LM feature (hedging_answers, uncertainty_answers, tone_prepared_chg,
tone_prepared, tone_answers_chg) → Rank IC +0.0692. Marginal improvement (+0.0010).

### Why LM Features Don't Help

With all 16 controls already in the model, LM dictionary features add noise
rather than signal. The only strategies that don't hurt are ones that use LM
features very sparingly — either a single feature (tone_prepared) or an ensemble
where each model sees just one. Forcing all 9 features in (Model B*) actively
degrades performance on both options.

## Model C — Adding FinBERT Features

Model C adds FinBERT transformer-based features on top of Model A's 16
controls. Four strategies are compared: forward selection, all forced,
ensemble of single-feature models, and FinBERT + LM combined.

### Option 1 Results (CAR vs. S&P 500)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| *Model A baseline* | *EN_strong* | *+0.0379* | *+0.99%* | *—* |
| **FinBERT Ensemble (11)** | **Ensemble** | **+0.0483** | **+3.20%** | **+0.0104** |
| All FinBERT forced | EN_a0.1 | +0.0424 | −0.51% | +0.0045 |
| FinBERT forward sel (1 feat) | RidgeCV | +0.0406 | +2.21% | +0.0027 |
| FinBERT + LM combined | GradientBoosting | +0.0360 | −0.05% | −0.0019 |

**Best Model C**: FinBERT Ensemble (11 features) → Rank IC **+0.0483** (Δ +0.0104).

### Option 2 Results (CAR vs. IT sector index)

| Strategy | Algorithm | Rank IC | Q5−Q1 | Δ vs Model A |
|----------|-----------|---------|-------|-------------|
| *Model A baseline* | *RidgeCV* | *+0.0682* | *+1.40%* | *—* |
| **FinBERT forward sel (4 feats)** | **Ridge** | **+0.0943** | **+2.47%** | **+0.0261** |
| All FinBERT forced | Ridge | +0.0844 | +2.72% | +0.0162 |
| FinBERT Ensemble (7) | Ensemble | +0.0775 | +1.81% | +0.0093 |
| FinBERT + LM combined | RidgeCV | +0.0452 | +2.47% | −0.0230 |

**Best Model C**: FinBERT forward selection → Rank IC **+0.0943** (Δ +0.0261).
Selected features: `fb_sent_answers_chg`, `fb_sent_gap`, `fb_pos_pct_answers`,
`fb_sent_answers`.

Forward selection steps (Option 2):

| Step | Feature Added | TS-CV Rank IC | Improvement |
|------|--------------|---------------|-------------|
| 0 | *(baseline — 16 controls)* | +0.0062 | — |
| 1 | `fb_sent_answers_chg` | +0.0112 | +0.0050 |
| 2 | `fb_sent_gap` | +0.0113 | +0.0001 |
| 3 | `fb_pos_pct_answers` | +0.0138 | +0.0025 |
| 4 | `fb_sent_answers` | +0.0187 | +0.0049 |

## Summary of Results

### Option 1 — CAR vs. S&P 500

| Model | Algorithm | Rank IC | Δ vs A | Q5−Q1 | Dir Acc |
|-------|-----------|---------|--------|-------|---------|
| Model A (16 controls) | EN_strong | +0.0379 | — | +0.99% | 52.0% |
| Model B (tone_prepared) | RidgeCV | +0.0374 | −0.0005 | +2.36% | 52.0% |
| Model B* (all 9 LM) | RidgeCV | −0.0401 | −0.0780 | +0.88% | 52.0% |
| **Model C (FinBERT Ens 11)** | **Ensemble** | **+0.0483** | **+0.0104** | **+3.20%** | **52.0%** |

### Option 2 — CAR vs. IT Sector Index

| Model | Algorithm | Rank IC | Δ vs A | Q5−Q1 | Dir Acc |
|-------|-----------|---------|--------|-------|---------|
| Model A (16 controls) | RidgeCV | +0.0682 | — | +1.40% | 38.2% |
| Model B (Ensemble) | Ensemble | +0.0692 | +0.0010 | +1.20% | 38.2% |
| Model B* (all 9 LM) | RidgeCV | +0.0401 | −0.0281 | +0.07% | 38.2% |
| **Model C (FinBERT fwd sel)** | **Ridge** | **+0.0943** | **+0.0261** | **+2.47%** | **38.2%** |

## Key Findings

### 1. FinBERT adds real signal — LM dictionary features do not

The core result: sentence-level transformer sentiment (FinBERT) consistently
improves prediction over 16 financial controls, while bag-of-words dictionary
features (LM) add no value. Model C achieves the highest Rank IC on both
options (+0.0483 and +0.0943), while Model B is essentially flat with Model A.

### 2. Forcing all features hurts — parsimony matters

Model B* (all 9 LM forced) is the worst model on both options. Even with
Ridge regularization, the noise from 9 correlated dictionary features
overwhelms any signal. The only LM strategies that don't hurt are those
using features very sparingly (1 feature or single-feature ensemble).

### 3. FinBERT captures what bag-of-words cannot

FinBERT reads full sentences with context and negation. The selected FinBERT
features for Option 2 (`fb_sent_answers_chg`, `fb_sent_gap`,
`fb_pos_pct_answers`, `fb_sent_answers`) capture **sentiment dynamics and
structure** — how sentiment changes over time and shifts between prepared
remarks and Q&A — rather than raw word counts.

### 4. Option 2 (sector benchmark) produces stronger results

Stripping out sector-level moves isolates firm-specific signal. Model A
starts stronger (+0.0682 vs +0.0379) and Model C lifts it further to +0.0943.
The sector benchmark removes noise from tech selloffs and rate-driven rotation.

### 5. The contrast between Model B and C is itself a finding

LM dictionary features and FinBERT features measure overlapping constructs
(sentiment, tone) but through different methods. That FinBERT helps while LM
doesn't argues that **how** sentiment is measured matters as much as **what** is
measured. Context-aware sentence-level classification captures nuance that word
counting misses.

### 6. Linear models dominate on small samples

Ridge and ElasticNet consistently outperform tree-based models. With ~2,900
training observations and 16–20 features, GBM/XGBoost/RF overfit despite
regularization. This holds across all models (A, B, C).

## Obstacles and Caveats

| Obstacle | Impact | Mitigation |
|----------|--------|------------|
| Small test set (204 obs, 2.5 years) | Results may not generalize | Reported with full metrics; reader should interpret cautiously |
| Time-series CV has only 3 folds | Feature selection may overfit to CV folds | Used all controls (no Model A selection); FinBERT forward selection is parsimonious |
| Single sector | Findings may not extend to other GICS sectors | By design — testing the within-sector hypothesis |
| No transaction costs | Quintile spreads are gross, not net | Relevant only if used for portfolio construction |
| IT sector composition changes over time | GICS reclassifications (e.g., PYPL moved out of IT) | Used current GICS mapping; membership filtered per S&P 500 spells |

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

# Step 2: Run Model A + Model B (6 strategies + B*)
python3 it_model_b.py --folder ~/earnings_it

# Step 3: Extract FinBERT features (requires GPU VM)
python3 it_finbert_features.py --folder ~/earnings_it --text_folder ~/earnings_sp500/text

# Step 4: Run Model C (FinBERT strategies)
python3 it_model_c_finbert.py --folder ~/earnings_it
```

Requires: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost` (CPU).
Steps 3–4 also require: `transformers`, `torch` (GPU recommended, ~90 min on T4).

## Layout

### Scripts

| File | Description |
|------|-------------|
| [`it_sector_extract.py`](it_sector_extract.py) | IT data extraction + sector benchmark computation |
| [`it_model_b.py`](it_model_b.py) | Model A (all controls) + Model B (6 strategies) + Model B* |
| [`it_finbert_features.py`](it_finbert_features.py) | FinBERT sentence-level sentiment extraction (GPU) |
| [`it_model_c_finbert.py`](it_model_c_finbert.py) | Model C (FinBERT forward selection, ensemble, all forced) |

### Results (committed in `it_results/`)

| File | Description |
|------|-------------|
| `it_model_b_improved_option_1_s&p_500_bench_report.txt` | Model A + B results — Option 1 |
| `it_model_b_improved_option_2_it_sector_bench_report.txt` | Model A + B results — Option 2 |
| `it_model_c_option_1_s&p_500_bench_report.txt` | Model C results — Option 1 |
| `it_model_c_option_2_it_sector_bench_report.txt` | Model C results — Option 2 |
| `model_a_features_option_1_s&p_500_bench.json` | Model A feature list + baseline — Option 1 |
| `model_a_features_option_2_it_sector_bench.json` | Model A feature list + baseline — Option 2 |

### Datasets (committed in `it_results/`)

| File | Rows | Description |
|------|------|-------------|
| `model_table_opt1.parquet` | 3,561 | IT firms, CAR relative to S&P 500 |
| `model_table_opt2.parquet` | 3,561 | IT firms, CAR relative to IT sector VW index |
| `finbert_features.parquet` | 3,561 | FinBERT features (13 columns) |

### Not committed (on VM at `~/earnings_it/`)

| File | Description |
|------|-------------|
| `it_sector_daily.parquet` | IT sector value-weighted daily returns |

The `.gitignore` allows files in `it_results/` while blocking raw WRDS data.
