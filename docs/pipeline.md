# Pipeline architecture

Question: do earnings-call language features (Model B: dictionary counts; Model C: + FinBERT
sentence sentiment) predict CAR[+2,+20] beyond the earnings
surprise and standard controls (Model A), out of sample and net of costs?

```
data/raw  (WRDS exports: CRSP, Compustat, I/B/E/S, CIQ transcripts, links)
   │  scripts/pull_sp500_pit.py                     point-in-time S&P 500 from 2008   ✅
   ▼
data/interim
   │  la transcripts  (1) parse   data/transcripts.py   every version: components, speakers  ✅
   │                  (2) clean   features/lexicons.py  operator lines, safe harbor, anonymise
   │  la events       data/linking.py, data/universe.py, events/timing.py, events/build.py  ✅
   │  la transcripts  (3) select  data/transcripts.py   one version per keydevid, capped at
   │                              the entry close; then qa_pairs, call_sections  ✅
   ▼
data/processed
   │  la targets           targets/car.py           CAR windows, post-call vol
   │  la controls          controls/earnings.py, controls/firm.py   + release metrics
   │  la features          features/*               F1–F4, interactions, standardised
   │  la select            (EDA + 5 filters, inside each training window)
   │  la model             models/specs.py, splits.py, regression.py
   ▼
   la evaluate             evaluation/metrics.py (gross)   →   backtest.py (net of costs)
   la holdout              once, on frozen models
```

✅ = implemented and tested. All other stages have fixed signatures and a docstring spec,
and raise `NotImplementedError` until their data arrives.

**Stage order.** Version selection needs the release's Day 0 and the entry close
(`transcripts.entry_close_utc`), which only exist after `la events`. So transcripts run in two
passes: parse and clean every version first, then select one version per call after events, and
build `qa_pairs` / `call_sections` from the selected versions only:
`la transcripts --pass parse`, then `la events`, then `la transcripts --pass select`. The select
pass covers every call, not only the events, so `pair_id`s do not depend on the sample filters;
the entry cap applies wherever `la events` found a release Day 0 (it binds only for calls 2+
sessions after the release). `la events` picks the same version with the same function
(`transcripts.select_versions`), so `processed/events.transcriptid` matches `transcript_calls`.

**Two tracks today.** `scripts/01–05` run on the legacy 179-event panel (`data/panel.csv`:
AAPL, MSFT, NVDA, 2010–2024). The full pull has not been through stages 5–12. Connecting it means
writing `data/processed/{events,targets,controls,features_*}` with the panel's column names (or
mapping them in `scripts/common.py`), then pointing `common.load_events` there instead of
`panel.csv`; `HOLDOUT_START` and the "panel ends Nov 2024" note go with it.

## Stage contracts

| Stage | Reads | Writes | Key |
|-------|-------|--------|-----|
| transcripts (parse, clean) | `ciq_transcript_detail`, `ciq_transcript_component`, `ciq_transcript_person` | `interim/transcript_versions`, `components_<yyyy>` (all versions, one shard per raw component shard) | transcriptid |
| events | `interim/transcript_versions` (deduplicated to keydevid; `transcript_calls` only exists after the select pass), `ibes_actu_epsus` (release time), `ciq_wrds_gvkey`, `crsp_ccmxpf_lnkhist`, `wrdsapps_ibcrsphist`, `crsp_msp500list_v2`, `comp_co_hgic`, `crsp_dsf_v2` (trading calendar, Day -1 market cap, return history) | `processed/events` (`schemas.EVENTS`), `processed/events_excluded` (with `exclusion_reason`), `outputs/validation/events_log.md` | keydevid |
| transcripts (select) | `interim/transcript_versions`, `components_<yyyy>`, `processed/events` and `events_excluded` (Day 0 for the entry cap) | `interim/transcript_calls`, `qa_pairs`, `call_sections` (every call) | keydevid → transcriptid |
| targets | `processed/events`, `crsp_dsf_v2`, `crsp_dsp500_v2` (main and market model), `crsp_vwmkt_daily` (index check) | `processed/targets` (`schemas.TARGETS`) | keydevid |
| controls | `processed/events`, `crsp_dsf_v2`, `comp_fundq`, `ibes_actu_epsus`, `ibes_statsumu_epsus`, `ibes_act_xepsus`, `ibes_statsum_xepsus`, `ibes_det_guidance`, `ibes_surpsum` | `processed/controls` (`schemas.CONTROLS`) | keydevid |
| features | `interim/*`, `processed/events`, `processed/controls` (interactions) | `processed/features_<family>` | transcriptid |
| select | features, targets, controls, splits | `processed/selection` (one row per feature × window, reason kept or dropped) | feature × window |
| model | `processed/{targets,controls,features_*,selection}` | `processed/predictions`, coefficient tables | keydevid × model × leg |
| evaluate | predictions, targets | IC, OOS R², bootstrap and Clark-West tables (gross) | model pair × leg |
| backtest | predictions, CRSP returns, spread and borrow data | portfolio returns gross and net of costs | |

Raw file names mirror WRDS tables (`library_table`). Big tables are sharded `name_<yyyy>`.
`io.read_table` reads both parquet and CSV, single files or shards.

## Sample period

**Sample: 2011 Q1 – 2026 Aug** (`sample.start` 2011-01-01, `sample.end` 2026-08-31): build
2011 Q1 – 2023 Q4 (52 quarters, 13 years), holdout 2024 Q1 – 2026 Aug (10 quarters + 2 months).
CRSP ends 2026-09-30, so September 2026 events have no complete CAR[+2, +20]; they are dropped
entirely rather than kept without a target. CAR[+2, +60] is still incomplete for events from
about June 2026: the [+2, +60] holdout results use the events that have it, and say so. Data are pulled from 2008, so 2008–10 feed lookbacks only (momentum,
pre-event volatility, the SUE scale, F4's prior calls).

Why 2011: the sample starts when the text does. Under the version rule (latest version created
by the Day 0 close), the share of S&P 500 calls with a usable transcript is:

| Period | Calls per quarter | Covered by Day 0 close |
|---|---|---|
| 2009 | about 660 | 26–36% |
| 2010 Q1–Q2 | 681 | 41–44% |
| 2010 Q3–Q4 | about 695 | 55–67% |
| 2011 Q1 | 709 | 93% |
| 2011 Q2–Q4 | about 710 | 81–91% |
| 2012 onward | about 710 | 87–98% |

- **Coverage:** 2011 Q1 is the first quarter above 90%, and coverage never falls back below 80%.
- **Selection:** in 2010 only the calls Capital IQ transcribed fastest are covered, likely the
  larger, more heavily covered firms. Text features from 2010 would describe a different set of
  firms from later years, so 2010 calls would bias the sample rather than add to it.
- **Lookbacks:** three years of 2008–10 data give every history-based feature its full window
  in 2011 (`history_min_calls` = 4 needs one year of calls).
- **Thresholds:** the build period has 13 years, so feature selection's sign-stability rule is
  "same sign in at least 9 of 13 years" (two-thirds, as 8 of 12 was).

Coverage figures approximate Day 0 by the call time (calls follow releases by a median 2 h),
on the 2026-11-17 pull (`data/raw/sp500_pit_2008`).

## Universe

- **Membership:** S&P 500 on Day 0 (`crsp_msp500list_v2`), never today's list. Index changes
  fall mostly in late December, outside earnings season, so Day 0 vs. call date rarely matters.
- **Links** (`data/linking.py`): CIQ companyid → gvkey (`ciq_wrds_gvkey`, date-valid on the
  call's New York date) → permno (CCM `LU` / `LC`, date-valid; every `linkprim` is kept as a
  candidate, including the 18 secondary-class `J` / `N` links) → I/B/E/S ticker
  (`wrdsapps_ibcrsphist`, score ≤ 2, lowest score). `ibcrsphist` ends its open links on its
  refresh date (`edate` 2025-12-31), so an `edate` on the table's last date counts as open
  (`events.iclink_open_end: auto`); otherwise every 2026 event would lose its ticker.
- **Release match** (`events.release_match_days: [-3, 1]`): the I/B/E/S quarterly EPS release
  (`ibes_actu_epsus`) whose `anndats` falls from 3 calendar days before to 1 day after the call's
  New York date; the latest one on or before the call date wins. When several periods are
  announced on one day (catch-up filings), the latest `pends`. Every candidate permno is carried
  through the links and the release match; membership on Day 0 and the share-class rule then pick
  one permno among the members (membership needs Day 0, which needs the release, which needs the
  ticker, which needs the permno). Joint calls (one keydevid under two CIQ companies, e.g. Lam
  Research / KLA-Tencor 2015, HP / HPE 2015: 6 calls) carry both companies as candidates and go
  through the same rule.
- **Result (2026-11-27 run, `outputs/validation/events_log.md`): 29,799 events**, 2011-01 to
  2026-08 by Day 0. Kept events: release on the call's New York date 84%, the day before 16%;
  call lag 0 for 29,661, 1 for 138. Agent C's call-date table had 29,968 calls (−0.6%: 103 with
  no I/B/E/S release within the window, 35 with the call outside Day 0 / Day +1, 20 with no
  I/B/E/S ticker, 20 with under 60 days of returns, 8 duplicates resolved differently and 6 that
  lose their only version to the entry cap, plus 23 placeholder-time calls that gain a
  transcript; 3 joint calls keep a different firm). The 30,102 releases under "Release metrics" (−1.0%) count releases, with or
  without a usable call transcript; 611 member calls have no version by the cutoff.
- **One call per release** (`events.duplicate_calls: keep_first`): when several calls map to
  one release, keep the first by call time and log the others. Two calls are one release when
  they have the same gvkey and fiscal label (parsed from the CIQ headline; ported from agent C's
  `scripts/full_source.py`), and calls still matched to the same I/B/E/S release after that are
  dropped as `duplicate_release`. Two changes by agent C, **pending team approval**, each behind
  a switch so it can be turned off:
  - `events.placeholder_time_last: true`: CIQ's 00:00:00 UTC call time is a placeholder (mostly
    pre-recorded remarks): it sorts last within its date for keep_first, and counts as an unknown
    time for the call's session (the call date, `missing_call_time: next_session`), in
    `la events` and in the transcript cutoff. `false` reads it as a real 00:00 UTC.
  - `events.duplicate_window_days: 3`: same gvkey and fiscal label count as one release only
    within 3 calendar days (Day 0 / Day +1 across a weekend); beyond that the label was reused
    after a fiscal-year change. `null`: at any distance.
- **Biotech is kept** (team decision 2026-11-17; `sample.exclude_gics_industries: []`). The 10/23
  concern (news comes from medical conferences) is checked in EDA by sector, not by exclusion.
- **One permno per event** (`sample.share_class_rule`): firms with two share classes (GOOGL /
  GOOG, FOXA / FOX, NWSA / NWS, BRK.B) map one call to two permnos. Keep the CCM primary link
  (`linkprim = P`); when that is ambiguous, keep the class with the largest market cap on Day -1.
  Market cap is preferred to volume because it is what the size control and the index weight use.
- Every exclusion keeps an `exclusion_reason` instead of dropping rows silently
  (`processed/events_excluded`). Reasons, in the order applied: `outside_sample_period` (Day 0,
  or the call date without a release), `no_gvkey`, `no_permno`, `not_sp500_member`,
  `no_ibes_ticker`, `no_ibes_release`, `no_transcript_version` (before keep_first, so the call
  kept for a release has usable text), `duplicate_call`, `duplicate_release`, `call_lag`,
  `gics_excluded`, `short_return_history` (fewer than
  `sample.min_prior_return_days` valid returns before Day 0). A missing EPS consensus is dropped
  in `la controls`. Attrition: `outputs/validation/events_log.md`.

## Text cleaning (before any feature)

Run on every version, before selection, so the robustness versions are cleaned the same way.

1. Drop operator turns and call-logistics lines (speaker role `Operator`; "next question").
2. Strip the safe-harbor / forward-looking-statement block from prepared remarks
   (`lexicons.strip_safe_harbor`; the legacy regex is in `scripts/01_build_features.py`).
3. Remove bracketed transcription notes (`[indiscernible]`, `[technical difficulty]`).
4. Anonymise firm, ticker and person names before any pre-trained model (FinBERT, embeddings).
5. Record words removed per step per call, so heavy trimming is visible.

## Event timing (Day 0 and windows)

The earnings release and the call count as one event. There is no call-timing control.

- **Day 0** is the first session whose close comes after the release (I/B/E/S `anndats` +
  `anntims`, New York time). A release before the open or during the session gets that day.
  A release after 16:00, or on a non-trading day, gets the next session. When the time is
  unknown, assume the release came after the close.
- **Call check:** the call must fall in Day 0 or Day +1 (`events.max_call_lag_sessions`). Otherwise the
  event is flagged and left out of the main sample (`call_lag`: 35 events on the 2026-11-27 run).
  Most are a call *before* Day 0 (lag −1): the I/B/E/S `anntims` is after the close (16:01–23:15)
  although the call was that morning, so the recorded time is later than the real release. In the S&P 500 sample, 98% of calls
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
  - **Robustness: market model.** Same index, but beta is estimated instead of fixed at 1:
    AR = R(i) − (α̂ + β̂ · R(S&P 500 total return)), with α̂, β̂ from OLS on sessions −250 to −30
    before Day 0 (`targets.market_model`; at least 80% valid days, else missing). The window ends
    before any announcement run-up, so the estimate uses only past data. This isolates the effect
    of beta, one step at a time. Events in about the first year of 2008 have no full window (the
    pull has no earlier data), so their market-model AR is missing.
    Precision: 2 parameters on 177–221 days. With typical daily volatilities (1.5% idiosyncratic,
    1.1% market), SE(β̂) ≈ 1.5 / (1.1 · √220) ≈ 0.09, which moves CAR[+2, +20] by about 0.4 pp,
    small against its idiosyncratic noise (about 6.5 pp). FF4 (5 parameters) would be noisier.
  - The CRSP value-weighted market is only an index check (correlation 0.994 with the S&P 500).
  - **FF4 is left for a future risk-management leg.** It is a risk model (size, value, momentum
    loadings), not a market benchmark, so it is not comparable with the main AR. Model A's size,
    B/M and momentum controls partly cover factor exposure meanwhile. Note it would need about a
    year of returns before 2008 for its estimation window.
- **Target:** CAR[+2, +20] (also [+2, +5] and [+2, +60]) starts after the reaction and
  after the call text is out.
- **Robustness:** rerun with `events.anchor: call`, which takes Day 0 from the Capital IQ
  call time instead.
- **`anntims` time zone (checked):** New York local time with daylight saving. 80% of 41,142
  releases match Capital IQ Key Dev type 28 within 15 min, 92% within 1 h. Key Dev
  `announceddateutc` is 5 h off when `announcedatetimezone` is blank; use `announcedate` +
  `announcetime` as UTC for those rows.
- **Transcript version and timing:** each call (`keydevid`) uses the **latest version
  created by the close of the call's own session** (`transcripts.select_latest_before`,
  cutoff from `day0_close_utc`, which takes Day 0 from the call time). For a call on the
  release's Day 0 that is the Day 0 close. For a call on Day +1 (e.g. release before the
  open, call after the close or the next morning) it is the Day +1 close, the trade entry,
  so these calls keep their transcript. The cutoff is capped at the entry
  (`entry_close_utc`), so no text created after the position opens is used even if a call
  time is wrong. "Day 0 close" below means this cutoff. Versions are ordered by `transcriptcreationdate_utc` +
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
    is why the sample starts on 2011-01-01 (see Sample period).
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

- **CAR = compounded stock return minus compounded index return** over the window
  (`targets/car.py`): the two legs are compounded separately, then differenced.
- **Compounding:** never drop a null day and carry on. Compound the stock's valid returns and
  compound the benchmark over every day of the window. The return after a null day does not
  reliably cover the gap (only half match the price change since the last valid price).
- **Reaction window** CAR[-1, +1]: any null day sets the control to missing.
- **Target windows:** missing when valid days are below `targets.min_window_coverage` (0.8);
  the coverage share is stored per event.
- **Lookback features** (momentum, `vol_pre`, beta): same 80% valid-day minimum, else missing.

## Release metrics and interactions

The call matters only where it adds to what the release already told the market.

- **Release metrics are Model A controls** (`controls.release_metrics`), all from I/B/E/S and
  dated before the Day 0 close: revenue surprise (`rev_sue`, SAL from the adjusted `act_xepsus` /
  `statsum_xepsus`),
  guidance midpoint vs. consensus, guidance width, guidance raised / kept / cut
  (`det_guidance`), net revisions before the release ((numup − numdown) / numest) and the
  forecast range ((highest − lowest) / price). `surpsum` is only a cross-check on our SUE.
- **Dictionary interactions are in Model B, so also in C but not in C − B**
  (`models.measures.interactions`): the tone residual on release metrics, beat-but-negative /
  miss-but-positive flags, surprise × section tone, guidance change × tone gap, guidance width ×
  uncertainty.
- **C_alone gets the same interactions on FinBERT tone** (`models.measures.finbert_interactions`,
  7 terms; guidance width × uncertainty has no FinBERT version). C keeps only the dictionary
  interactions, so C − B is FinBERT alone, and C_alone vs. B compares each tone method with its
  own interactions.
- **Residual terms are model-stage features.** `tone_resid` and `finbert_tone_resid` regress tone
  on the release metrics, so they are fitted inside each training window and applied to its test
  window (`models/regression.py`), never computed once in the features stage.

### Coverage of release metrics (drop vs. keep)

S&P 500 releases with an I/B/E/S quarterly EPS actual, member on the release date,
2011-01 to 2026-08: **30,102 events** (2026-11-17 pull; ticker links with score ≤ 2). Share of
events lost if each metric had to be present:

| Metric | Source | Available | Lost if dropped |
|---|---|---|---|
| EPS consensus (SUE), net revisions, forecast range | `statsumu_epsus` (fpi 6, last statpers before the release) | 99.7% | 0.3% |
| Cash flow per share actual | `actu_xepsus` (CPS) | 87.7% | 12.3% |
| Any guidance within ±1 day of the release | `det_guidance` | 76.4% | 23.6% |
| EPS guidance with consensus at date (`guidance_vs_consensus`) | `det_guidance` (EPS, `mean_at_date`) | 50.0% | 50.0% |
| EPS guidance as a range (`guidance_width`) | `det_guidance` (EPS, `val_2`) | 45.9% | 54.1% |
| Revenue surprise (`rev_sue`: actual and consensus) | `act_xepsus`, `statsum_xepsus` (SAL) | 99.5% | 0.5% |
| EBITDA surprise (actual and consensus) | `act_xepsus`, `statsum_xepsus` (EBT) | 89.9% | 10.1% |
| Every EPS, revenue and guidance metric above | | about 45% | **about 55%** |

EPS guidance coverage is steady at 50–54% a year, except 2020–22 (37–45%, guidance withdrawn
during COVID).

- **Rule (decided, `controls.release_missing: flag_and_zero`): every missing guidance metric is
  set to 0**, with `guidance_given = 0` when the firm gave no guidance within 1 day of the release.
  Partial guidance is zero-filled the same way (point guidance has `guidance_width` = 0; no
  consensus at the date gives `guidance_vs_consensus` = 0). No event is dropped for missing
  guidance: "no guidance" is itself information, and dropping would halve the sample and keep
  only firms that guide (larger, more stable ones). Events are dropped only when the EPS
  consensus is missing (0.3%).
- **Revenue and EBITDA come from the adjusted tables.** SAL and EBT are in `act_xepsus`,
  `statsum_xepsus` and `surpsum`, but not in the unadjusted `actu_xepsus` / `statsumu_xepsus`,
  which hold per-share measures (CPS, DPS) only. Amounts are not affected by splits, so the
  adjusted tables are correct for them; no re-pull is needed.

## Standardisation (per feature)

All schemes are computed **within the call's calendar quarter** (the cross-section the model
ranks), so they use no future data and need no training-window fit, except where noted.

| Feature group | Examples | Scheme | Why |
|---|---|---|---|
| Bounded tone ratios | tone_prepared, tone_answers, finbert_tone_* | Winsorise 1/99%, then z-score | Already in [-1, 1], roughly symmetric |
| Differences | tone_gap, d_tone, d_uncertainty, div_qa_prepared | Winsorise 1/99%, z-score, no demeaning by firm | Already relative; demeaning would double-count F4 |
| Word densities | uncertainty, hedging (per 1,000 words) | log(1 + x), then z-score | Right-skewed counts |
| Shares | non_answer_rate, evasiveness | Rank to [0, 1] | Many zeros; ranks handle ties |
| Similarities | qa_relevance, novelty | z-score | Near-normal cosine ranges |
| Interactions | sue_x_tone_*, guidance_x_* | Rank each input to [-0.5, 0.5] first, then multiply | Products of ranks are comparable across years |
| Tone residuals | tone_resid, finbert_tone_resid | z-score of the residual; built in the model stage, fitted inside each training window | Its regression must not see the test quarter |
| Flags | beat_neg_tone, guidance_given, guidance_change, version_final | None (0/1 or −1/0/+1; 0 when no guidance) | Categorical |
| Surprises | sue, rev_sue | Winsorise 1/99%, then rank | Heavy tails, a few huge surprises |
| Firm controls | size (log), bm, mom, vol_pre, dispersion, n_analysts | Winsorise 1/99%, z-score (log for dispersion, n_analysts) | Standard in the PEAD literature |
| Reaction windows | car_m1_1, car_0_1 | Winsorise 1/99% only | Kept in return units so coefficients read as drift per unit reaction |
| Targets | car_2_20 etc. | Winsorise 1/99% for fitting; never for the backtest | Raw returns are what is traded |

## Models

| Model | Regressors | Question it answers |
|-------|------------|---------------------|
| A | Controls: SUE (SUE_TS), size, B/M, momentum, pre-event volatility, coverage, dispersion, `version_final`, release metrics, plus the leg's reaction window | Baseline |
| B | A + dictionary measures (method 1: NLTK tokens + Loughran-McDonald lists) + release × call interactions | Does dictionary language add to A? |
| C | B + FinBERT sentence tone per section (method 2) | Does FinBERT add to the dictionary? |
| C_alone | A + FinBERT + FinBERT-tone interactions | Dictionary or FinBERT: which is better on its own? |

**Fixed effects by demeaning** (`models.fe_method: demean`): inside each training window, the
target and every regressor are demeaned by industry and by quarter (alternating until stable,
since the panel is unbalanced), and the model is fitted with no dummies. This equals unpenalised
dummies, so the penalty shrinks only predictors: penalised dummies would leave industry and
quarter levels in the residual for language features to pick up, lasso would drop dummies
arbitrarily, and B − A would partly reflect how much the dummies were shrunk. In the test window,
the training industry means are subtracted from the regressors; the test quarter's effect cannot
be estimated and drops out of the prediction (harmless for IC, which ranks within the quarter, but
it shifts the level in OOS R²). OLS uses the same transformation so all estimators see identical
inputs.

**Only language is penalised** (`models.unpenalized`): the controls, release metrics and
reaction window are never shrunk, in any model. Model A therefore reduces to OLS under every
estimator, and in B, C and C_alone the penalty falls only on the dictionary, interaction and
FinBERT measures. Model A's regressors are thus estimated the same way in every model, so B − A
measures language rather than a different treatment of the controls, and the penalty cannot trade
a control off against a correlated language feature (e.g. tone against SUE). With about 15
regressors on 3,500–4,000 training events per window, Model A needs no shrinkage.

**Estimators** (`models.estimators`): primary OLS, lasso, ridge and elastic net. Exploratory: tree-based models (XGBoost), Fama-MacBeth slopes by
quarter, and SHAP interaction values; these are reported but do not decide the A/B/C comparison.

**SUE:** SUE_TS is primary (`controls.sue_scaler: ts`): surprise over the SD of the firm's prior
surprises (previous 8 quarters, at least 4, shifted so the current surprise is not in its own
scale). Analyst-SD scaling (NaN when the SD is 0) and price scaling are robustness runs.

### Splits and validation

- **Rolling windows for now** (`split.walk_forward.scheme: rolling`): train on 8 quarters, test on
  the next 2, roll by 2. **The train / validation / test design is still under discussion**
  (rolling vs. expanding, window lengths); `models/splits.py` keeps it in config.
- **Validation set:** time-ordered cross-validation inside each training window
  (`split.inner_cv`, forward chaining, 3 folds): each fold validates on one of the window's last
  quarters and trains on the quarters before it. Penalties are chosen there, never on the test block.
- **Embargo per target** (`split.walk_forward.embargo: target_window`): a training event is dropped
  when its target window ends on or after the first test Day 0, so CAR[+2, +20] gets 20 sessions
  and CAR[+2, +60] gets 60. The same rule applies inside validation folds and before the holdout.
- **Leakage tests** (`tests/test_splits.py`): for CAR[+2, +5], [+2, +20] and [+2, +60], no training
  target window reaches the test block; the longer target always has the smaller training set;
  dev splits never touch the holdout; the holdout is one embargoed split; validation folds are
  time-ordered and embargoed.

Each model exists in both reaction legs (`model_specs_by_leg`): CAR[-1, +1] (main) and
CAR[0, +1] (comparison). Comparisons (`models.specs.COMPARISONS`, mirrored in `scripts/common.py`)
are made within a leg; comparing the same model across legs shows the effect of the window. B, C and C_alone
each vs. A; C vs. B; C_alone vs. B. Every increment is ΔIC and ΔOOS R² with quarter-block
bootstrap p-values. Clark-West is used only for nested pairs, so not for C_alone vs. B.
Measures from methods 3–4 (embeddings, LLM rating) stay out of A/B/C and are tested one at
a time, each added to A and to B, in both reaction legs. The current runs are `scripts/04_oos.py` (dev) and `scripts/05_backtest_holdout.py`
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

### Regressor counts and sample size

| Model | Controls + release + reaction | Dictionary + interactions | FinBERT | Regressors | Parameters with FE* | Events per parameter* |
|-------|-------------------------------|---------------------------|---------|------------|---------------------|-----------------------|
| A | 8 + 7 + 1 = 16 | – | – | 16 | ≈ 34 | ≈ 110 |
| B | 16 | 8 + 8 = 16 | – | 32 | ≈ 50 | ≈ 75 |
| C | 16 | 16 | 3 | 35 | ≈ 53 | ≈ 70 |
| C_alone | 16 | – | 3 + 7 interactions | 26 | ≈ 44 | ≈ 85 |

Controls (`controls.include`): sue, size, bm, mom, vol_pre, n_analysts, dispersion, version_final;
release metrics (`controls.release_metrics`); plus the leg's reaction window. Dictionary,
interaction and FinBERT lists: `models.measures`. Counts are before feature selection.
\* Per walk-forward window: intercept + 10 industry dummies (if industry = 11 GICS sectors) + 7 quarter
dummies (8 training quarters), on about 3,500–4,000 training events (≈ 500 firms × 8 quarters).
The interactions are now in B and C; with about 70 events per parameter, ridge or elastic net
is the safer primary fit for B and C.

The binding constraint is signal, not observations: post-earnings CAR R² is well under 1%, so a
single window's coefficients are imprecise. Keep the measure lists fixed and small, use shrinkage
(ridge) or ranked features if interactions are added, and judge models on out-of-sample IC pooled
across windows, not on one window's t-stats.

## Look-ahead rules

- Everything is known by the **trade entry, the Day +1 close**; the target starts at Day +2.
- Features use only the call's own text (version created by the transcript cutoff, at the
  latest the entry), plus the same firm's **earlier** calls (F4).
- Controls: consensus dated before the release (`statpers < anndats`), fundamentals
  reported before Day 0 (`rdq < day0`), prices at the end of the month before the call. The
  reported EPS in SUE and the reaction window (CAR[-1, +1] or CAR[0, +1]) are known by the
  entry. Day -1 holds no news (it ends before the release); it guards the window against
  wrong or missing release times.
- Targets start at Day +2. Walk-forward training drops events whose target window overlaps
  the test period (embargo = the target's window: 20 sessions for CAR[+2, +20], 60 for
  CAR[+2, +60]); `tests/test_splits.py` checks it.
- The holdout (`split.holdout_start`) is excluded from every split until the final run.

## Evaluation and backtest

- **Evaluation is gross:** ΔIC and ΔOOS R² with quarter-block bootstrap p-values, Clark-West for
  nested pairs, all within a leg.
- **Backtest: further work.** Portfolio rules and cost sources are still to be decided.
- **Trading costs enter only in the backtest, the last stage** (`backtest`): enter at the Day +1
  close, exit at Day +20, long-short by predicted CAR within the week. Spreads, borrow fees and
  the S&P 500 hedge are applied there, so IC results stay comparable with the literature. The cost
  data source is still open (`backtest.costs`).

## Feature families (research framework v2)

| Family | Measures | Scope ID |
|--------|----------|----------|
| Communication state | tone_prepared, tone_answers, tone_gap, uncertainty, hedging | F1, F2 |
| Interaction quality | qa_relevance, evasiveness, non_answer_rate | F3 |
| Consistency / deviation | d_tone, d_uncertainty, d_evasiveness, novelty, div_qa_prepared | F4, F5 |

Consistency measures are transformations of the first two families (`depends_on`), not
primitive factors. FinBERT and embeddings need the `nlp` extra.

**F3 labels:** see "Q&A labels for F3" below (build period only, never the holdout).

## Future work (out of the core path)

- **RavenPack: dropped.** It holds no call text (only short news tags), so it cannot supply
  language features, and its earnings sentiment largely restates the beat or miss. A benchmark
  comparison with the professor's counterfactual test is left for future work.
- **F5 `div_press_release`:** prepared remarks vs. the same-day 8-K Ex.99.1; needs 8-K text
  (`features.future`).
- **Volatility target** (`future_targets`): `vol_post_2_20` and `vol_ratio_2_20` are built but
  modelled only if the CAR / PEAD results are inconclusive.
- **FF4 risk-management leg**, audio features (no GCP credits) and readability.

## Q&A labels for F3

F3 (`evasiveness`, `non_answer_rate`) is calibrated and validated on hand labels for a sample of
question–answer pairs. The classifier then scores every pair; the full history (about 800,000
pairs) is not labelled.

- **Labels (binary, per pair):**
  - `evasive = 1`: the answer does not give the specific information requested (it changes the
    topic, falls back on generic talking points, answers an easier question, or skips the main
    ask of a multi-part question).
  - `non_answer = 1`: management explicitly declines or cannot answer ("we don't disclose",
    "too early", "we'll follow up offline").
  - A refusal is usually both labels. A frank answer with bad news is `evasive = 0`.
  - Each pair is judged from its question and answer alone, blind to returns, features and
    scores. `notes` gives a reason for every 1 and marks borderline cases.
- **Sample:** about 450 pairs, build period only (2011 Q1 – 2023 Q4, never the holdout).
  Pairs are stratified by year × GICS sector (about 3 per cell), with at most 1 call per firm
  per year and 2 pairs per call, seed 42. Calls are S&P 500 members on the call date, biotech
  included (team decision 2026-11-17), using the latest transcript version. The sample is
  `data/qa_label_sample_full.csv` and the labels are `data/qa_labels.csv`. This replaces the
  legacy 150-pair, 3-firm sample.
- **Reliability:** an independent, blind second labeller relabels 90 overlapping pairs
  (seed 7, at least 6 per year). Cohen's kappa per label goes to
  `outputs/validation/label_interrater.csv`, and disagreements to `label_disagreements.csv`.
  Target kappa ≥ 0.6; below 0.4, the rules are reworded before F3 is used. Both labellers are
  LLM agents, so a person checks the disagreements plus 20 random pairs.
- **Use:** a logistic classifier on pair features (`REL`, `REL_JAC`, lengths, hedging,
  `NONANS_RULE`), fitted out of fold with folds grouped by call (`EVADE_CLF`, `NONANS_CLF`).
  F3 is kept if its out-of-fold AUC is about 0.7 or more.
- **Open:** `scripts/01_build_features.py` and `02_validate.py` still read the legacy panel, so
  agent C adds `--source full`, which reads the full-history pairs (`data/interim/qa_pairs`, built by `la transcripts`), so the labelled `pair_id`s match.
- Handoffs: `docs/labeling_handoff.md` (labeller A), `docs/labeling_handoff_calibration.md` (agent C: calibrate, rerun 01/02), `docs/labeling_handoff_teammate.md`
  (labeller B).

## Raw data

`scripts/pull_sp500_pit.py` pulls the point-in-time S&P 500 universe from 2008 into
`data/raw/sp500_pit_2008`: CRSP quarterly (`crspq`, to 2026-09-30), Compustat, I/B/E/S
(EPS and non-EPS, guidance, surprise history) and Capital IQ (transcripts, Key Dev release
events). CIZ `dlyret` already includes delisting returns, so no separate delisting file is
spliced in.
