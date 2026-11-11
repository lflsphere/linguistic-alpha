# Brief for the coding agent: Linguistic Alpha from Earnings Calls

## Project in one paragraph
We test whether the language management uses on earnings calls predicts each stock's abnormal return after the call, beyond the earnings surprise and standard factors. One row = one S&P 500 earnings call. The primary target is **CAR[+2, +20]** (`car_2_20`): the sum of daily (stock return − S&P 500 return) from trading day +2 to +20 after Day 0. **Model A** uses controls only (SUE, size, book-to-market, momentum, announcement return). **Model B** adds language features. Linguistic alpha = Model B beating Model A out of sample, ultimately after trading costs.

## Files

| File | Purpose |
|---|---|
| `sp500_pipeline.py` | Data pipeline: WRDS download → one modeling table. **Treat as working code; change it only to fix a real problem.** |
| `requirements.txt` | Python packages: wrds, pandas, numpy, pyarrow |
| `LOCAL_SETUP.md` | How to run it on a computer or a GCP virtual machine |
| `sp500_pipeline.sh` | Wrapper to run it as a WRDS Cloud batch job (`qsub`) |
| Loughran-McDonald Master Dictionary CSV (to be added) | Word lists for tone/hedging; put it in the project folder |

## Step 1: run the pipeline

```
pip install -r requirements.txt
python -c "import wrds; wrds.Connection(wrds_username='bhuvighosh3')"   # once; the user types password + approves Duo
python sp500_pipeline.py
```

- Needs about 16 GB RAM and 10 GB disk; the first run takes hours (mostly transcripts).
- Resumable: if it stops, run the same command again; saved pieces are skipped.
- `--stages panel targets text final` rebuilds the tables from downloaded data without WRDS.
- **The WRDS login is interactive (password + Duo on the user's phone). Ask the user to do it; never ask for, store or print the password.**

Stages: `pull` → `panel` → `targets` → `text` → `final`. Output folder: `~/earnings_sp500/`.

## Step 2: check the output before doing anything else

Read `~/earnings_sp500/data_report.txt` and confirm:
- `duplicate calls: 0`
- `check: every holdout company is in train: True`
- roughly 25,000+ calls, ~600–900 companies, calls in every year 2010–2024
- low share of "unmapped section" pieces (a few % at most)
- `car_2_20` mean near 0; most calls 2010–2024 have `sue`, `size`, `bm`, `mom`

If WRDS returns "column does not exist" or a table is missing, fix that query in `stage_pull` only, and say what changed.

## What the pipeline produces

`model_table.parquet` (also `.csv`) — one row per call:

| Group | Columns |
|---|---|
| IDs / timing (not model inputs) | `keydevid`, `transcriptid`, `companyid`, `companyname`, `headline`, `call_date`, `call_time_utc`, `gvkey`, `permno`, `ticker`, `day0` |
| Sector (for industry controls) | `sector` (GICS), `gind`, `sic` |
| **Targets** | `car_2_5`, **`car_2_20` (primary)**, `car_2_60`, `vol_post_2_20` |
| **Controls** | `car_0_1`, `sue`, `sue_price`, `surprise_pct`, `size`, `bm`, `mom`, `n_analysts`, `dispersion`, `vol_pre` |
| Earnings inputs | `fiscal_period_end`, `actual_eps`, `consensus_eps`, `eps_surprise`, `split_adjusted_consensus`, `mcap_m` |
| **Language features** | `n_questions`, `n_answers`, `words_*`, `avg_answer_words`, `share_answers_by_execs`, `share_other_sections`; with the dictionary: `tone_prepared`, `tone_answers`, `tone_gap`, `hedging_prepared`, `hedging_answers`, `uncertainty_answers`, `*_chg` (vs. the company's previous 4 calls) |
| Split | `period`, **`sample`**, `prior_calls_with_target` |

`model_table_with_text.parquet` = the same plus `prepared_text` and `qa_text` (questions/answers labeled `QUESTION:` / `ANSWER:`).

`sample` values:
- `train`: calls 2010–2022 with a target
- `holdout`: calls 2023–2024 with a target, **only companies with ≥ 4 training calls**
- `holdout_no_history`: 2023–2024 calls of companies without enough training history (not used for testing)
- `no_target`: calls after the return data ends (2024-12-31), no CAR yet

## Rules (do not break these)

1. **The holdout (2023–2024) is used once, at the very end.** Don't look at holdout results while choosing features, models or settings. For model choices, use walk-forward tests inside 2010–2022 (fit on earlier years, test on the next year, repeat).
2. **No lookahead.** Every input must be known before Day 0. Don't standardize or rank features using full-sample statistics; use only data up to each date (e.g., cross-sectional ranks within each quarter).
3. **No survivorship bias.** Keep companies that left the index or were delisted in training. Don't drop rows because a company has no future data.
4. **Model A vs. Model B on the same rows.** Compare them on identical calls (drop rows missing any input used by either model).
5. **Multiple testing.** Many features × targets get tested: report all of them; require t-statistic > 3 or apply a false-discovery-rate correction.
6. **Data license.** WRDS data stays on private storage. Never commit data files, `.pgpass` or passwords to git; add `earnings_*/` and `*.parquet` to `.gitignore`.
7. **Don't display transcript text in full tables or logs** (it's huge); preview with `.str[:150]`.

## Next tasks, in order

1. Run the pipeline and verify `data_report.txt` (Step 2).
2. Add the Loughran-McDonald dictionary; rerun `--stages text final`.
3. Event study: sort train calls into quintiles by each language feature; plot average CAR from day −5 to +60; double-sort by SUE × feature.
4. Regressions on train: `car_2_20` on controls (Model A) vs. controls + language features (Model B); Fama-MacBeth by quarter and pooled with standard errors clustered by firm and quarter; sector fixed effects.
5. Walk-forward out-of-sample prediction inside 2010–2022 (information coefficient, Model A vs. B).
6. Only after 1–5 are fixed: run once on the holdout.
7. Later: evasiveness (F3) with embeddings/an LLM on question–answer pairs; spoken vs. written press release (F5) using 8-K filings from `wrdssec`; long-short backtest with transaction costs (TAQ spreads, Markit borrow fees), capacity and decay.

## Known limits

- CRSP returns end 2024-12-31; 2025–2026 calls have no target.
- `sp500_pipeline.py` was tested on realistic fake data (CAR, momentum, volatility, splits, delistings, share classes, membership dates all checked against hand calculations), but it has **not yet run against real WRDS data**. WRDS column names may differ by subscription; fix queries if needed.
- Section labels come from S&P's component type names (`component_types.parquet`); check the unmapped share in the report.
