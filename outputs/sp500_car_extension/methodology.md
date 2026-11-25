# S&P 500 CAR label extension

## Source definitions

The input is `model_table_with_linguistic.csv`, with 29,698 uniquely identified calls from the archived S&P 500 universe. The CSV is preserved as supplied, including the original Day 0 column, cohort fields, and all nonmissing CAR values. Input SHA-256 is `833d76f72ed6ce22674ea551ff60e135360d62d76806a0dfbc7f7681eb7c3ffb`. Production windows use the existing CRSP trading-session index, inclusive endpoints and the arithmetic sum of daily stock return minus the date-matched benchmark. When a new label is available, every day in its window has an observed stock and benchmark return. No daily missing observation is replaced with zero, and no truncated window is published.

### Legacy columns

Git `2a3a90a:sp500_pipeline.py:175–181` defines the original daily sources. `stage_targets()` at lines 286–369 builds abnormal returns and the four windows. Lines 340–352 use the archived UTC calendar date with the New York clock for the 16:00 cutoff. A trading-day call at or after 16:00 Eastern rolls forward, as does a call on a nontrading day. This historical calendar-date convention is retained for the legacy fields and reproduces the saved Day 0 on the entire original calendar. `firm_controls()` in `src/linguistic_alpha/controls/firm.py:73–81` is a separate producer used by other reports. It is not the source of the CSV under audit. The legacy target formulas are the sums of `RET - SPRTRN` over [0,1], [2,5], [2,20], or [2,60]. `SPRTRN` is the S&P 500 price-return series. Stock `RET` is the vendor return, including distributions. The 2009–2024 legacy daily-stock files are separate from the three selected CIZ shards.

On legacy days with an observed DLRET, the stock return is `(1 + RET) × (1 + DLRET) − 1`. If only DLRET is observed, the observed stock return for that session is DLRET. Delisting dates are rolled to the first session on or after the recorded date, following the original stage. Beginning with 2025, the pipeline uses CIZ `dlyret` directly. It already incorporates delisting returns. The old pipeline could accept 80% coverage or truncate at a delisting. The extension uses full windows throughout, including around delistings. The 2024 verification sample of existing values has complete windows, so that stricter eligibility did not change the comparisons.

### Early CAR

The full-universe Early CAR uses **Original Sector Option 1**. In Git `06273f1766daeb29de4b9313dd20639439a09671:sector_extract.py:125–143,145–153,167–196`, the stock grid is the daily `RET` grid, the benchmark is `vwretd` if the index file supplies it and otherwise `sprtrn`, and the event window is [-1,1]. The Healthcare replication input adapter at `outputs/healthcare_original_pierre_replication/run_original.py:31` exposed only `sprtrn`. The native Early computation therefore sums raw-source stock `RET` minus S&P 500 `SPRTRN` on those three observations. Unlike upstream `stage_targets()`, the Sector extractor records delisting dates for possible truncation and does not splice DLRET into the stock grid. Day 0 follows the New York date and 16:00 boundary in `src/linguistic_alpha/events/timing.py:15–57`.

For Early CAR, the extension uses the legacy return grid without a separate delisting splice through 2024. After 2024 it uses CIZ observations. Early CAR remains missing if its window crosses a CIZ day flagged as a delisting, since the original pre-splice stock-return definition cannot be established from the bundled return on that day. The original Healthcare Option 2 uses a separate sector-return series. It is tested against the saved Healthcare benchmark as a robustness check and is not used in the full-universe CSV. Both Early CAR and `car_0_1` require Day +1 returns and are unavailable at call time.

## Inputs and identity verification

| Input | Rows | First date | Last date |
| --- | --- | --- | --- |
| data/team_pit/sp500_pit/crsp_dsp500_v2.parquet | 5,658 | 2004-01-02 | 2026-06-30 |
| data/team_pit/sp500_pit/crsp_dsf_v2_2024.parquet | 167,016 | 2024-01-02 | 2024-12-31 |
| data/team_pit/sp500_pit/crsp_dsf_v2_2025.parquet | 163,590 | 2025-01-02 | 2025-12-31 |
| data/team_pit/sp500_pit/crsp_dsf_v2_2026.parquet | 78,938 | 2026-01-02 | 2026-06-30 |

The legacy benchmark is `data/raw/crsp_dsi.parquet`, with `SPRTRN` from 2008-01-02 through 2024-12-31. Legacy stock inputs are `data/raw/crsp_dsf_2009.parquet` through `data/raw/crsp_dsf_2024.parquet`. The legacy delist input is `data/raw/crsp_dsedelist.parquet`. The calendar and equity sources end on **2026-06-30**. Windows spanning year-end 2024 use legacy data through 2024 and CIZ thereafter. The same legacy/CIZ switch applies to stock and benchmark inputs.

The event map verifies the CIQ transcript ID and exact UTC timestamp in `data/raw/ciq_transcript_detail.parquet`, then the date-valid CIQ-company-to-GVKEY and GVKEY-to-archived-PERMNO spells from the local `ciq_wrds_gvkey.parquet` and `crsp_ccmxpf_lnkhist.parquet` copies in `data/raw/` and `data/team_pit/sp500_pit/`. CCM uses LC/LU and P/C links. The archive's PERMNOs are checked and never replaced. This verifies 29,694 of 29,698 mappings. In-period CIZ S&P 500 membership is audited as a diagnostic. 40 of 2,631 archive events from 2025 through the CIZ cutoff fall outside a membership spell in `crsp_msp500list_v2.parquet`. All remain in the output, and no membership-based selection is applied to the archived universe.

## Validation and provenance

- `car_validation.csv` compares saved values on all 2024 events across sectors, using legacy daily inputs and CIZ separately. Early CAR is matched on transcript ID, PERMNO and UTC call timestamp to both saved Original Healthcare Sector options. The legacy agreement gate is absolute error ≤ 1e−7 in decimal CAR, with exact agreement on label availability. The CIZ bridge gate is ≤ 1e−5 except for the isolated vendor revision documented in the validation row. The known revision is reconciled at the stock-day level, and legacy observations are retained on that date. Agreement rates are proportions.
- Each horizon has `<label>_source` and `<label>_missing_reason` fields. `original` preserves the exact original CSV cell string. `reconstructed_legacy_crsp` means a newly calculated label uses legacy daily inputs only. `extended_crsp_ciz` means the label uses CIZ for at least one day and may also include legacy days across the year-end boundary. `missing` means the label is empty and the companion reason is populated.
- Missing reasons identify an unverified dated security link, Day 0 outside the calendar, a return window outside the calendar, a missing benchmark, a missing stock observation, or CIZ Early CAR delisting semantics that could not be verified. The first blocking condition is recorded.
- Before publication the script reparses the output, checks every archived cell outside the eligible CAR fills, verifies identical row order and counts, and checks every label and its flags. Row joins used for validation are one-to-one. Production returns use unique security/session keys. The input file is hashed again before publication.

## Reproduction

From the repository root, run the script with Python, NumPy, pandas, and PyArrow available. Use a fresh destination. The script stops on a failed validation gate before creating the production CSV.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python outputs/sp500_car_extension/extend_car.py --out outputs/sp500_car_extension/reproduction
```

The script calculates realized labels and label availability only. No 2025–2026 holdout labels were used for feature selection, model tuning, fitting, or performance evaluation. The unchanged archive Day 0 column can remain blank after 2024. Label-specific signal and target maturity should be obtained from the applicable rule and CRSP calendar when planning a future holdout evaluation. Transcript delivery, feature availability, historical financial revisions and benchmark-vintage continuity remain separate verification tasks.
