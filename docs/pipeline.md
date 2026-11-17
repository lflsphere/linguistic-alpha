# Pipeline architecture

Question: do earnings-call language features (Model B: dictionary counts; Model C: + FinBERT
sentence sentiment) predict CAR[+2,+20] beyond the earnings
surprise and standard controls (Model A), out of sample and net of costs?

```
data/raw  (WRDS exports: CRSP, Compustat, I/B/E/S, CIQ transcripts, links)
   │  la pull              data/wrds_pull.py         (optional; files can come from the web UI)
   ▼
data/interim
   │  la transcripts       data/transcripts.py      components, qa_pairs, call_sections   ✅
   │  la events            data/linking.py, data/universe.py, events/timing.py (Day 0 ✅)
   ▼
data/processed
   │  la targets           targets/car.py           CAR windows, post-call vol
   │  la controls          controls/earnings.py, controls/firm.py
   │  la features          features/*               F1–F5 by family
   │  la model             models/specs.py, splits.py, regression.py
   ▼
   la evaluate             evaluation/metrics.py, backtest.py
```

✅ = implemented and tested. All other stages have fixed signatures and a docstring spec,
and raise `NotImplementedError` until their data arrives.

## Stage contracts

| Stage | Reads | Writes | Key |
|-------|-------|--------|-----|
| transcripts | `ciq_transcript_detail`, `ciq_transcript_component`, `ciq_transcript_person` | `interim/transcript_calls`, `components`, `qa_pairs`, `call_sections` | transcriptid |
| events | `interim/transcript_calls`, `ibes_actu_epsus` (release time), `ciq_wrds_gvkey`, `crsp_ccmxpf_lnkhist`, `wrdsapps_ibcrsphist`, `crsp_msp500list_v2`, `comp_co_hgic`, `crsp_dsf_v2` (trading calendar) | `processed/events` (`schemas.EVENTS`) | keydevid |
| targets | `processed/events`, `crsp_dsf_v2`, `crsp_dsp500_v2` / `crsp_vwmkt_daily`, `ff_factors_daily` | `processed/targets` (`schemas.TARGETS`) | keydevid |
| controls | `processed/events`, `crsp_dsf_v2`, `comp_fundq`, `ibes_actu_epsus`, `ibes_statsumu_epsus` | `processed/controls` (`schemas.CONTROLS`) | keydevid |
| features | `interim/*`, `processed/events`, (8-K text for F5) | `processed/features_<family>` | transcriptid |
| model | `processed/{targets,controls,features_*}` | `processed/predictions`, coefficient tables | keydevid × model |
| evaluate | predictions, returns, cost data | reports | |

Raw file names mirror WRDS tables (`library_table`). Big tables are sharded `name_<yyyy>`.
`io.read_table` reads both parquet and CSV, single files or shards.

## Event timing (Day 0 and windows)

The earnings release and the call count as one event. There is no call-timing control.

- **Day 0** is the first session whose close comes after the release (I/B/E/S `anndats` +
  `anntims`, New York time). A release before the open or during the session gets that day.
  A release after 16:00, or on a non-trading day, gets the next session. When the time is
  unknown, assume the release came after the close.
- **Call check:** the call must fall in Day 0 or Day +1 (`call_lag_sessions`). Otherwise the
  event is flagged and left out of the main sample. In the S&P 500 sample, 98% of calls
  come within a day of their release (median 2 hours).
- **Reaction legs (two):** every model runs twice, differing only in the reaction control
  added to its controls (`controls.reaction_legs`): the main leg uses CAR[-1, +1]
  (`car_m1_1`; Day -1 is the last close before the news), the comparison leg CAR[0, +1]
  (`car_0_1`). Neither is a target.
- **Benchmark: S&P 500 total return** (`crsp_dsp500_v2.vwretd`). AR = stock return minus
  index return. The index is point in time by construction (its members on each day).
  - **Why total, not price return (`sprtrn`):** `dlyret` includes dividends, so subtracting a
    price-only index leaves the index dividend yield in every AR. The bias is about
    yield × days / 252 and grows with the horizon: on our 179 events the price-return CARs
    are higher by 0.01 pp over [0, +1], 0.18 pp over [+2, +20] and 0.46 pp over [+2, +60].
  - **What it changes:** levels, OOS R² against a zero forecast, the "long" drift and backtest
    P&L. Cross-sectional ranks barely move (the shift is the same for every firm on a date),
    so Rank IC is nearly unchanged.
  - **Trade consistency:** the backtest's S&P 500 hedge (futures or an ETF) also earns
    dividends.
  - The CRSP value-weighted market and FF4 are robustness checks only.
- **Target:** CAR[+2, +20] (also [+2, +5] and [+2, +60]) starts after the reaction and
  after the call text is out.
- **Robustness:** rerun with `events.anchor: call`, which takes Day 0 from the Capital IQ
  call time instead.
- **`anntims` time zone (checked):** New York local time with daylight saving. 80% of 41,142
  releases match Capital IQ Key Dev type 28 within 15 min, 92% within 1 h. Key Dev
  `announceddateutc` is 5 h off when `announcedatetimezone` is blank; use `announcedate` +
  `announcetime` as UTC for those rows.
- **Transcript version and timing:** each call (`keydevid`) uses the **latest version
  created by the close of Day 0** (`transcripts.select_latest_before`, cutoff from
  `day0_close_utc`). Versions are ordered by `transcriptcreationdate_utc` +
  `transcriptcreationtime_utc`, never by `transcriptid`. Calls with no version by then are
  left out of the main sample (`transcripts.no_version_before_cutoff: drop`).
- **Robustness: the earliest version** of each call (`transcripts.version_rule: earliest`).
- **Text-quality control: `version_final`**, one variable in every model: 1 when the chosen
  version has reached the Final stage (Edited or later), 0 when it is still Preliminary
  (Spellchecked). It is a single control because the two Capital IQ fields are tied: every
  Preliminary version is Spellchecked (45,073) and no Final version is (120,466), so the
  editing stages form one ladder (Spellchecked, Edited, Proofed, Audited) and Final starts
  at Edited.
- **Why that rule (51,052 calls, 163,844 versions, 2008–2026):** versions arrive in a fixed
  order, each one cleaner than the last:

  | Version | Calls with it | Median hours after call | Created by Day 0 close |
  |---|---|---|---|
  | Preliminary / Spellchecked | 84% | 1.2 | 94% |
  | Final / Edited | 89% | 4.5 | 76% |
  | Final / Proofed | 65% | 8.6 | 36% |
  | Final / Audited | 40% | 67 | 6% |

  - **Coverage:** the latest version by Day 0 close exists for 86% of calls overall, and for
    94–99% from 2011. In 2008–10 versions were mostly created later (17–52% coverage), which
    is why the text sample effectively starts in 2010–11.
  - **Version picked:** Final / Edited 55%, Final / Proofed 25%, Preliminary / Spellchecked
    12% (the only text by Day 0 close for those calls), other 8%.
  - **The mix shifts over time:** 2013–17 mostly Proofed; from 2020 about 88% Edited, as Final
    versions now arrive faster (median 3.5 h after the call vs. 5 h in 2012). Text quality
    therefore varies by year; `version_final` and year effects absorb it.
  - **Day +1 close alternative:** also look-ahead safe, since the trade enters at the Day +1
    close. It covers 94% of calls and almost no Preliminary text (0.2%). Kept as a
    robustness cutoff.

### Missing daily returns

CRSP `dlyret` is null on 3,932 rows. 3,851 fall outside membership; 81 fall inside it, at most
one per firm (75 `NS`, the first day of a new security such as a spin-off).

- **Compounding:** never drop a null day and carry on. Compound the stock's valid returns and
  compound the benchmark over every day of the window. The return after a null day does not
  reliably cover the gap (only half match the price change since the last valid price).
- **Reaction window** CAR[-1, +1]: any null day sets the control to missing.
- **Target windows:** missing when valid days are below `targets.min_window_coverage` (0.8);
  the coverage share is stored per event.
- **Lookback features** (momentum, `vol_pre`, beta): same 80% valid-day minimum, else missing.

## Models

| Model | Regressors | Question it answers |
|-------|------------|---------------------|
| A | Controls: SUE, size, B/M, momentum, pre-event volatility, coverage, dispersion, plus the leg's reaction window | Baseline |
| B | A + dictionary measures (method 1: NLTK tokens + Loughran-McDonald lists) | Does dictionary language add to A? |
| C | B + FinBERT sentence tone per section (method 2) | Does FinBERT add to the dictionary? |
| C_alone | A + FinBERT only | Dictionary or FinBERT: which is better on its own? |

Each model exists in both reaction legs (`model_specs_by_leg`): CAR[-1, +1] (main) and
CAR[0, +1] (comparison). Comparisons (`models.specs.COMPARISONS`, mirrored in `scripts/common.py`)
are made within a leg; comparing the same model across legs shows the effect of the window. B, C and C_alone
each vs. A; C vs. B; C_alone vs. B. Every increment is ΔIC and ΔOOS R² with quarter-block
bootstrap p-values. Clark-West is used only for nested pairs, so not for C_alone vs. B.
Measures from methods 3–4 (embeddings, LLM rating) stay out of A/B/C and are tested one at
a time. The current runs are `scripts/04_oos.py` (dev) and `scripts/05_backtest_holdout.py`
(dev backtest, then a single holdout run that scores every frozen model). C and C_alone need
the FinBERT columns from `01_build_features.py --finbert`; without them only A and B run.
The scripts write one folder per leg (`outputs/oos/<tag>_m1_1`, `<tag>_0_1`).
`scripts/build_reaction_windows.py` rebuilds every event window from the 2008 daily pull
against the S&P 500 total return, on the panel's Day 0: `AR_M11` (CAR[-1, +1], not in the
panel), `AR01` and the `CAR_2_h` targets, written to `data/reaction_windows.csv`.
`common.load_events` replaces the panel's columns with these and keeps the panel's as
`<name>_PX`. The same construction against the price return reproduces the panel to 1e-6,
which shows the panel used `sprtrn`. `LOGVOL_2_20` stays the panel's. `--freeze` selects on the main leg, freezes both, and records the hashes of
`features.csv` and `reaction_windows.csv`.

## Look-ahead rules

- Features use only the call's own text, plus the same firm's **earlier** calls (F4).
- Controls use only data dated before Day 0: `rdq < day0`, `statpers < anndats`, prices at
  the end of the month before the call.
- Targets start at Day +2. Walk-forward training drops events whose target window overlaps
  the test period (embargo = 20 sessions).
- The holdout (`split.holdout_start`) is excluded from every split until the final run.

## Feature families (research framework v2)

| Family | Measures | Scope ID |
|--------|----------|----------|
| Communication state | tone_prepared, tone_answers, tone_gap, uncertainty, hedging | F1, F2 |
| Interaction quality | qa_relevance, evasiveness, non_answer_rate | F3 |
| Consistency / deviation | d_tone, d_uncertainty, d_evasiveness, novelty, div_qa_prepared, div_press_release | F4, F5 |

Consistency measures are transformations of the first two families (`depends_on`), not
primitive factors. Later extensions are kept out of the core path: FinBERT and embeddings
(`nlp` extra), and the professor's suggestions of a RavenPack benchmark and audio/readability.

## Raw data

`scripts/pull_sp500_pit.py` pulls the point-in-time S&P 500 universe from 2008 into
`data/raw/sp500_pit_2008`: CRSP quarterly (`crspq`, to 2026-09-30), Compustat, I/B/E/S
(EPS and non-EPS, guidance, surprise history) and Capital IQ (transcripts, Key Dev release
events). CIZ `dlyret` already includes delisting returns, so no separate delisting file is
spliced in.
