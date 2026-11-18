# Legacy ingestion audit and rebuild plan

Subject: `Linguistic_Alpha.ipynb` (`earnings_pipeline.py` cell) and its output `Data/panel.xlsx`
(179 calls, AAPL / MSFT / NVDA, 2010-01 → 2024-11).
Reproduce the measured findings with `python scripts/audit_legacy_panel.py`.

**Verdict:** use the legacy panel only as a sanity reference. **Do not** use its EPS-based
controls or its text columns for research. Rebuild from raw WRDS tables through this pipeline.

## Findings

Severity: **H** = changes results, **M** = bias or loss of sample, **L** = hygiene.

| # | Area | Issue | Evidence | Sev | Fix (module) |
|---|------|-------|----------|-----|--------------|
| 1 | SUE | Split-adjusted I/B/E/S rounds consensus to 2 decimals while actuals keep 4. Surprises are then nonsense (AAPL Q1-2010: actual 0.1311 vs consensus 0.07 → +87%) | 94/179 rows; 13 rows with consensus = 0 (NVDA) | H | Unadjusted `actu_epsus` / `statsumu_epsus`, put on one split basis with I/B/E/S `adj` / `adjsum`; never CRSP share factors on EPS (`controls.earnings`) |
| 2 | SUE | Scaled by forecast stdev, not by price as in the scope doc. Undefined when stdev = 0 | 54/179 `sue` NaN | H | `sue` = surprise / price; keep `sue_std` for robustness |
| 3 | Text | `panel.xlsx` cuts every cell at Excel's 32,767-character limit | 47 Q&A and 12 prepared texts truncated | H | Never store text in xlsx; use parquet (`io`) |
| 4 | Precision | Every float64 is downcast to float32 at pull time (EPS, prices, returns) | notebook `pull()` | M | No downcast (`data.wrds_pull`) |
| 5 | Survivorship | `crsp_dsedelist` is pulled but never merged into returns | notebook step 3 | M | **Resolved by CIZ:** quarterly CRSP `dlyret` already includes the delisting return, so never add `delret` again. The 2008 pull has no unpriced delistings (`DM`/`DP`) |
| 6 | Benchmark | `sprtrn` is a price return; CRSP `ret` includes dividends, so AR drifts up by roughly the dividend yield | definition | M | **Done:** S&P 500 total return (`vwretd`); `scripts/build_reaction_windows.py` rebuilds AR01 and CAR_2_h (price-return CARs were 0.18 pp too high over [+2,+20]) |
| 7 | Timing | Day 0 used the **UTC** calendar date with the NY clock time. A call after 20:00 ET lands a day late | 0 rows hit in this sample; it will hit on the full universe | M | NY date (`events.timing`, tested) |
| 8 | Timing | A missing call time defaults to "before close", i.e. same-day reaction | 0 rows here | M | Configurable; conservative `next_session` default |
| 9 | Versions | "Latest version" = max `transcriptid`, but re-collected transcripts get new, larger ids (NVDA 2010 → 3634xxx) | 11 out-of-order ids | M | **Decided (revised):** latest version created by the Day 0 close, ordered by creation time; earliest version as robustness |
| 10 | Linking | CCM `linkprim` filter is applied only to the gvkey list, not to the join. ICLINK `score` is not used in the join. `ciq.wrds_gvkey` dates are ignored. `drop_duplicates("keydevid")` then keeps an arbitrary row | notebook `build_event_panel` | M | Date-valid, ranked links with attrition log (`data.linking`) |
| 11 | Sections | CIQ tags some IR/operator hand-offs as *Answer* before the first question; they end up in answer text | 3 calls start Q&A with ANSWER | L | Dropped by `pair_qa` |
| 12 | Roles | No speaker roles, so analyst follow-ups and executive answers cannot be told apart | n/a | M | `wrds_transcript_person.speakertypename` |
| 13 | SUE match | Actual EPS is matched within ±5 calendar days of `call_date`, not on announcement vs. Day 0 | notebook | L | `controls.earnings` |
| 14 | Controls | Momentum fills missing daily returns with 0. B/M uses `ceqq` only and keeps negative book equity | notebook | L | `controls.firm` |
| 15 | Targets | `vol_post_2_20` is a level, but the scope defines a ratio to pre-event vol | definition | L | Both: `vol_post_2_20`, `vol_ratio_2_20` |
| 16 | F4 | Rolling baseline uses `min_periods=1`, so a "4-call mean" can be a single call | notebook | L | `features.history_min_calls` |
| 17 | Coverage | MSFT is missing Q3-FY2010 (Apr-2010); the gap is not flagged | 1 gap > 120 days | L | `universe.apply_exclusions` reports gaps |
| 18 | Hygiene | WRDS username hard-coded in the notebook; outputs written to Colab paths | notebook | L | env var `WRDS_USERNAME`, `configs/default.yaml` |

Still open, needing raw data to verify:
- whether CAR / size / mom values match a clean recomputation;
- CIQ component type ids beyond 1–4;
- the share of calls with no time stamp (releases: no I/B/E/S `anntims` is missing in the 2008 pull; 15% of Capital IQ type-28 rows have no time).

## Rebuild plan

Ordered. Each step ends with a check against the legacy panel on AAPL/MSFT/NVDA. Differences
are expected where a finding above applies, and should match within tolerance elsewhere.

1. **Pull (core + transcripts) for the 3-ticker test set.** `la pull` writes to `data/raw/`.
   Verify table names and columns first: CIZ migration, CIQ type tables.
2. **Transcripts.** `la transcripts` is implemented and checked against the legacy text on
   non-truncated calls. Inspect the `other` section ids and the speaker-role coverage.
3. **Events.** Build `data.linking` and `data.universe`, then Day 0 (`events.timing`, done).
   Report link attrition per step. Check: Day 0 matches the legacy value on all 179 calls.
4. **Targets.** `targets.car` with delisting and benchmark choices. Check: CAR[2,20] matches
   the legacy value within 1e-6 when using `sprtrn` (same definition), then switch the benchmark.
5. **Controls.** `controls.firm`, then `controls.earnings` on the unadjusted basis. Check: the
   AAPL Q1-2010 surprise is small and positive, and no consensus is 0.
6. **Scale to the full S&P 500.** Done: `scripts/pull_sp500_pit.py` pulled 2008–2026 (CRSP to
   2026-09-30) into `data/raw/sp500_pit_2008`, and its validation is in that folder's README.
   Still to re-run on the built panel: link attrition per step and truncation = 0.
7. Only after 1–6: features (structure in place), then models and evaluation.

## Decisions (settled)

- **Sample and holdout:** 2011 Q1 – 2026 Q3, 63 quarters (quarterly CRSP runs to 2026-09-30):
  build 2011–2023 (52 quarters), holdout 2024 Q1 – 2026 Q3 (11 quarters, `split.holdout_start` = 2024-01-01).
- **AR benchmark:** S&P 500 total return (`crsp_dsp500_v2.vwretd`, beta = 1); market model on the
  same index for robustness (finding 6); CRSP VW market as an index check only. FF4 is left for a
  future risk-management leg.
- **Transcript versions:** latest version created by the Day 0 close, earliest as robustness,
  `version_final` as a control (finding 9, `docs/pipeline.md`).
- **Sample start:** 2011-01-01, the first quarter in which at least 90% of calls have a
  transcript by the Day 0 close; 2008–10 data feed lookbacks only (`docs/pipeline.md`, Sample period).
- **Biotech kept** (team decision 2026-11-17, reversing meeting 10/23). `comp_co_hgic` is in the 2008 pull for sector EDA.
