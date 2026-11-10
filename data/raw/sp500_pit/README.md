# S&P 500 point-in-time raw data (WRDS), pulled 2026-11-09 (CRSP quarterly update)

Produced by `linguistic-alpha/scripts/pull_sp500_pit.py`; row counts per file are in `manifest.csv`.
WRDS data is licensed: share it only within the team.

## Universe

987 PERMNOs that were S&P 500 members at any time from 2005-01-01 (`crspq.msp500list_v2`).
CRSP comes from the **quarterly** update (`crspq`, data through 2026-06-30), not the annual
`crsp` schema (through 2025-12-31); values are identical on the overlap. A `mbrenddt` of
2026-06-30 means "still a member at the end of the data", not an exit.
Rows are **not** trimmed to membership: keep history for lookbacks, then apply membership on
the event date:

    member = (mbrstartdt <= event_date) & (event_date <= mbrenddt)    # crsp_msp500list_v2

Check: 500–505 members on each tested date (2005-01-03 to 2026-06-30), and every member has a
CRSP daily row on that date.

## Files (parquet, zstd)

| File | Source | Coverage | Notes |
| --- | --- | --- | --- |
| `crsp_msp500list_v2` | crspq.msp500list_v2 | memberships ending 2005+ | The PIT universe |
| `crsp_ccmxpf_lnkhist` | crspq.ccmxpf_lnkhist | LU/LC links | permno -> gvkey; filter on linkdt/linkenddt |
| `wrdsapps_ibcrsphist` | wrdsapps.ibcrsphist | all | permno -> I/B/E/S ticker; `score` 1 = best |
| `ciq_wrds_gvkey` | ciq.wrds_gvkey | all | gvkey -> CIQ companyid; filter on dates |
| `crsp_stksecurityinfohist` | crspq.stksecurityinfohist | all | CUSIP/ticker/name history (replaces stocknames) |
| `crsp_dsf_v2_YYYY` | crspq.dsf_v2 (CIZ) | 2004–2026-06-30 | Daily ret/prc/vol/cap, `shrout`, adjustment factors, share type, exchange |
| `crsp_stkdelists` | crspq.stkdelists (CIZ) | all | `dlyret` already includes the delisting return: do **not** splice `delret` in again. Use `delretmisstype` to spot unpriced delistings |
| `crsp_dsp500_v2` | crspq.dsp500_v2 (CIZ) | 2004–2026-06-30 | S&P 500 index: `sprtrn` (price return), `vwretd` (total return) |
| `crsp_vwmkt_daily` | crspq.inddlyseriesdata, indno 1000200 | 2004–2026-06-30 | CRSP value-weighted market: `vwretd` (total), `vwretx` (price only) |
| `ff_factors_daily` | ff.factors_daily | 2004–2026-09 | mktrf, smb, hml, umd, rf |
| `comp_fundq` | comp.fundq | datadate 2002–2026-09 | Standard INDL/STD/C/D filters |
| `comp_co_hgic` | comp.co_hgic | all | GICS history (biotech exclusion) |
| `ibes_act_epsus` / `ibes_actu_epsus` | adjusted / unadjusted actuals | anndats 2004–2026-08 | Use unadjusted for SUE (audit finding 1) |
| `ibes_statsum_epsus_YYYY` / `ibes_statsumu_epsus_YYYY` | adjusted / unadjusted consensus | 2004–2026-08 | fpi 6 and 7 |
| `ciq_transcript_detail` | ciq.wrds_transcript_detail | calls 2005–2026-10 | Every version of every earnings call (event type 48) |
| `ciq_transcript_person_YYYY` | ciq.wrds_transcript_person | 2005–2026 | Speaker name and role per component, no text |
| `ciq_transcript_component_all_YYYY` | ciq.ciqtranscriptcomponent | 2005–2026 | Text of **all** versions; join on transcriptid |

## Known gaps

- CRSP ends 2026-06-30: of 2,236 calls in 2026, 1,266 fall on or before that date; the rest
  (and calls late in June) have no forward-return targets until the next quarterly update.
- Transcript coverage is thin before 2008 (30 calls in 2005, 636 in 2006, about 2,700 a year from 2008).
- 54,197 calls from 923 firms, 3.14 versions per call on average; pick one version per call downstream.
- 169,967 transcripts; about 600 (0.4%) have no speaker rows, so check text coverage per version.
- 3 members have no CCM link and 5 no I/B/E/S link (mostly the 2026 additions); 7 duplicate gvkey-datadate rows in `comp_fundq`.
