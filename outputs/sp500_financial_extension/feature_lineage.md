# S&P 500 financial-feature reconstruction

The production CSV retains its input header, all 29,698 archive events, row order, event identifiers, linguistic columns, CAR values, CAR provenance and all 2010–2024 values. Only the approved 2025–2026 financial fields are replaced. The SHA-256 of the input CAR extension table is `8d8e65ee1ebd6b28a27cea4f288bdf45517c70888c5a52b75404270ff829d767`.

## Traced historical definitions

The archived CSV uses Git `2a3a90a:sp500_pipeline.py`, particularly `stage_targets()` at lines 286–445 and source queries at 175–197. The financial helpers in `src/linguistic_alpha/controls/firm.py:36–49,61–90` and `controls/earnings.py:115–131` are later alternatives. In particular, the former uses a CEQQ fallback book-equity method, and the latter puts price-scaled SUE in the field named `sue`. Neither naming choice is adopted here.

| Output | Source and convention |
| --- | --- |
| `mcap_m`, `size` | Lines 325, 380–386. Use `abs(PRC) × SHROUT / 1000` in millions of dollars on the last market day before the archive call month. On that exact PERMNO only, allow up to 10 market sessions since the last observed capitalization. Use `ln(mcap_m × 1e6)` for size. CIZ `dlyprc` and `shrout` preserve the unit conversion. |
| `mom` | Lines 337–338, 387–394. Compound observed total stock returns from the start of call month minus 12 months to strictly before the start of call month minus one month. Require at least 150 observed returns and a fully bounded calendar window, and exclude months without a verified endpoint. |
| `vol_pre`, `vol_post_2_20` | Lines 340–352, 371–378. Sample standard deviation with ddof=1 of daily stock return minus S&P 500 `SPRTRN`, times √252. Inclusive session windows [−60,−5] and [+2,+20] with at least 40 and 15 paired observations. The full calendar window must exist. **Post-event volatility is a realized outcome.** |
| `bm` | Lines 396–404. Positive Compustat quarterly `CEQQ` divided by `mcap_m`. Latest RDQ strictly before Day 0, within 200 calendar days. Production also requires RDQ before the New York call date and DATADATE ≤ RDQ, and declines conflicting records for the latest RDQ. There is no DATADATE-based availability fallback. |

Day 0 for these reconstructions uses the original UTC archive date and 16:00 Eastern clock from lines 340–352. Timing uses original event-specific calendar positions and retains the previously published Day 0 string as-is. A post-cutoff event never falls back to the first stock row or to a calendar-edge price. The stock side uses legacy daily `RET` through 2024 and CIZ daily `dlyret` thereafter. Legacy same-session `DLRET` is compounded with regular return, or used alone when that is the observed return. CIZ already includes the delisting return. The `SPRTRN` price-index benchmark also switches to CIZ from 2025. Missing prices and returns remain missing except the specified same-security market-cap carry-forward.

The local `ff_factors_daily.parquet` date index matches every CIZ index session in the common range. Its dates extend Day 0 anchoring through 2026-09-30 for features whose underlying inputs already exist, such as some July pre-windows and book-to-market. The file supplies session dates only. There is no Fama–French return substitution, and the stock and benchmark windows still end at the actual CIZ return cutoff of 2026-06-30.

### Earnings surprise

The original pulls unadjusted `actu_epsus` and `statsumu_epsus` (lines 189–197). The dated PERMNO-to-I/B/E/S link selects the lowest score (265–270). Lines 406–443 match the nearest quarterly actual within ±5 calendar days of the archived call date, then the latest FPI=6 forecast snapshot strictly before ANNDATS, at most 120 days before it, for the same ticker and fiscal period. This run additionally verifies the actual predates the call or has an observed non-midnight same-day announcement time at or before the call in Eastern time. A later release is not used. Conflicting same-key observations, mixed currencies and a fiscal period-end after release are rejected.

| Field | Definition |
| --- | --- |
| fiscal_period_end, actual_eps | Matching I/B/E/S QTR PENDS and unadjusted actual VALUE |
| consensus_eps | Unadjusted median on the actual announcement share basis, MED_EST × k |
| eps_surprise | Actual EPS minus share-basis consensus |
| sue | Surprise / (STDEV × k), only when the adjusted standard deviation is positive |
| sue_price | Surprise / absolute unadjusted price on the trading session preceding the index at or before ANNDATS |
| surprise_pct | Surprise / abs(adjusted consensus), provided consensus is nonzero |
| n_analysts | NUMEST from the same eligible forecast snapshot |
| dispersion | (STDEV × k) / abs(MED_EST × k), identical to the unadjusted ratio for positive k |
| split_adjusted_consensus | Whether the verified split factor k differs from one |

`k` is the announcement-session CRSP price-event factor divided by the consensus-session factor, following the original `CFACPR` ratio. During production the two CIZ `dlycumfacpr` endpoints use the same CIZ reference share basis. Where daily `dlyfacprc` is complete, its inverse product across the interval must agree with the endpoint ratio. A same-date non-unit event factor is left unresolved if the split timing relative to release cannot be established. Validation replays the original float32 source precision for EPS values used in arithmetic. Production keeps raw numeric precision. The adjusted I/B/E/S tables are inventoried and never used for the surprise calculation.

The local I/B/E/S–CRSP link extract has a common terminal date of 2025-12-31. For a call beyond that boundary, the pipeline requires the prior link to end exactly at the extract boundary and an independently observed CRSP security-history spell with the **same eight-character CUSIP** spanning the boundary and the call. The archived date-valid PERMNO–GVKEY and exact CIQ transcript-clock links must also pass. A nonempty archived I/B/E/S ticker must agree. The sidecar records the resulting bridge for 853 matched EPS blocks. This does not verify subsequent changes after the CUSIP data end. No security link is extended beyond its independent coverage.

## Sources and localized defects

Actual Parquet contents were checked against the directory and the tracked README and manifest at Git `06273f1:data/raw/sp500_pit/`. The corresponding local team copies of those metadata files are absent, so only selected tracked manifest entries were checked against the local Parquet row counts. The adjusted snapshots below are inventory only.

| Source | First date | Last date | Rows |
| --- | --- | --- | --- |
| ff_factors_daily.parquet (calendar dates only) | 2004-01-02 | 2026-09-30 | 5,722 |
| crsp_dsf_2020.parquet | 2020-01-02 | 2020-12-31 | 178,100 |
| crsp_dsf_2021.parquet | 2021-01-04 | 2021-12-31 | 174,500 |
| crsp_dsf_2023.parquet | 2023-01-03 | 2023-12-29 | 168,098 |
| crsp_dsf_2024.parquet | 2024-01-02 | 2024-12-31 | 168,949 |
| crsp_dsf_v2_2024.parquet | 2024-01-02 | 2024-12-31 | 167,016 |
| crsp_dsf_v2_2025.parquet | 2025-01-02 | 2025-12-31 | 163,590 |
| crsp_dsf_v2_2026.parquet | 2026-01-02 | 2026-06-30 | 78,938 |
| crsp_dsp500_v2.parquet SPRTRN | 2004-01-02 | 2026-06-30 | 5,658 |
| raw/comp_fundq.parquet datadate | 2008-01-31 | 2026-09-30 | 59,553 |
| raw/comp_fundq.parquet RDQ | 2008-02-06 | 2026-11-09 | 58,419 |
| raw unadjusted actual ANNDATS | 2008-01-03 | 2026-08-20 | 56,395 |
| raw/2020 unadjusted consensus STATPERS | 2020-01-16 | 2020-12-17 | 17,014 |
| raw/2021 unadjusted consensus STATPERS | 2021-01-14 | 2021-12-16 | 16,677 |
| raw/2023 unadjusted consensus STATPERS | 2023-01-19 | 2023-12-14 | 16,114 |
| raw/2024 unadjusted consensus STATPERS | 2024-01-18 | 2024-12-19 | 16,088 |
| sp500_pit/comp_fundq.parquet datadate | 2002-01-31 | 2026-09-30 | 75,503 |
| sp500_pit/comp_fundq.parquet RDQ | 2002-02-06 | 2026-11-06 | 73,718 |
| sp500_pit unadjusted actual ANNDATS | 2004-01-05 | 2026-08-20 | 66,558 |
| sp500_pit/2024 unadjusted consensus STATPERS | 2024-01-18 | 2024-12-19 | 15,908 |
| sp500_pit/2025 unadjusted consensus STATPERS | 2025-01-16 | 2025-12-18 | 15,697 |
| sp500_pit/2026 unadjusted consensus STATPERS | 2026-01-15 | 2026-08-20 | 10,240 |
| crsp_msp500list_v2 (nonempty spell dates) | 1925-12-31 | 2026-06-30 | 1,010 |
| crsp_ccmxpf_lnkhist (nonempty spell dates) | 1949-07-01 | 2026-04-30 | 1,188 |
| ciq_wrds_gvkey (nonempty spell dates) | 1968-04-01 | 2023-09-11 | 1,024 |
| crsp_stksecurityinfohist (nonempty spell dates) | 1925-12-31 | 2026-06-30 | 9,424 |
| comp_co_hgic (nonempty spell dates) | 1999-06-30 | 2026-10-05 | 1,702 |
| wrdsapps_ibcrsphist (nonempty spell dates) | 1976-01-15 | 2025-12-31 | 1,557 |
| adjusted actual ANNDATS (inventory only) | 2004-01-05 | 2026-08-20 | 66,558 |
| adjusted 2025 STATPERS (inventory only) | 2025-01-16 | 2025-12-18 | 15,697 |
| adjusted 2026 STATPERS (inventory only) | 2026-01-15 | 2026-08-20 | 10,240 |

The linkage audit matched 6,508 of 6,508 attempted historical-validation and production event mappings. It uses exact CIQ transcript clocks and dated CIQ-to-GVKEY / CCM links. CIZ membership is checked against real spells, with 40 of 2,631 archive events through the CIZ limit outside their saved membership spell. The user-supplied archived universe defines the output, including those rows.

The cutoff-session capitalisation and size for the archive's first sorted security match all 3,348 archived future cap rows and 3,348 size rows. Lines 348–353 mapped invalid late events to stock row zero, and lines 382–394 used that row for cap and momentum even though their guard only checked whether the original PERMNO was valid. The end-of-calendar clip froze cap at the last 2024 date. The first-stock momentum readback matches 836 of 836 nonmissing archived entries. Post-2024 volatility was empty because Day 0 was outside the old calendar. The old EPS-price denominator also clipped at the old calendar edge, so existing post-2024 price-scaled SUE is replaced from dated prices instead of copied.

## Validation and output

The historical comparisons use the 2021 events through October and all 2024 archive calls, with original dates and source definitions. The earlier slice keeps validation windows in the available legacy stock files. `feature_validation.csv` reports overlap, availability, mean, median, maximum absolute error and tolerance pass rates. The production checks separately enforce book equity and earnings cutoffs. Approved features: actual_eps, bm, consensus_eps, dispersion, eps_surprise, fiscal_period_end, mcap_m, mom, n_analysts, size, split_adjusted_consensus, sue, sue_price, surprise_pct, vol_post_2_20, vol_pre. Unapproved fields: none.

The local `feature_provenance.csv` has one row per 2025–2026 archive event, with dates and per-field status. `reconstructed` means sourced, including genuine unchanged values. Other status strings give the unavailability reason or identify a validation-gated field kept unchanged. The model table adds no provenance columns. The tracked `feature_coverage_aggregate.csv` includes original and final coverage, distinct counts, modal frequencies, changed and cleared archived values, new values and unresolved reasons for the full universe and Healthcare subset. Exact modal values are present only in the local `feature_coverage.csv`. Historical coverage rows are based on preserved cells.

Replay from the root with Python, pandas, NumPy and PyArrow. The existing `outputs/sp500_car_extension/extend_car.py` supplies the event mapping and Day 0 helpers. The historical input and existing outputs are kept read-only. Run with a fresh destination.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python outputs/sp500_financial_extension/reconstruct_financials.py --out outputs/sp500_financial_extension/reproduction
```

No models, feature selection, performance metrics or strategies were evaluated. `vol_post_2_20`, `car_0_1` and `car_-1_1` must remain outside call-time feature lists. Static vendor snapshots do not establish original release vintages. Transcript delivery is a separate gate for future models.
