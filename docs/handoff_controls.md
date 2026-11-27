# Handoff: `la controls` (agent F)

Starts after agent D (`docs/handoff_events.md`) has committed `data/processed/events`. Read
`docs/handoff_next_steps.md` (rules for every agent) and `docs/pipeline.md`: Release metrics
and interactions, Coverage of release metrics, Standardisation, Models (SUE). Config:
`controls.*` in `configs/default.yaml`.

## Goal
`la controls` writes `data/processed/controls` (one row per `keydevid`, `schemas.CONTROLS` plus
the release metrics in `controls.release_metrics`, checked with `schemas.validate`). This is
everything Model A needs.

## Inputs
- `data/processed/events` (Day 0, permno, gvkey, IBES ticker), `data/interim/transcript_calls`
  (`version_final`)
- `crsp_dsf_v2_<yyyy>`, `crsp_dsp500_v2` (reaction windows on the same benchmark as the targets)
- `comp_fundq` (book equity for B/M, point-in-time by report date)
- `ibes_actu_epsus`, `ibes_act_xepsus`, `ibes_det_guidance`, `ibes_surpsum`, `ibes_adj`
- `ibes_statsumu_epsus_<yyyy>`, `ibes_statsum_xepsus_<yyyy>` (sharded by year; `io.read_table`
  reads the shards)

## Work
1. `controls/earnings.py`: `earnings_surprise`. SUE_TS is primary (`controls.sue_scaler: ts`);
   also `sue` (scaled by price) and `sue_std` (scaled by forecast stdev), as in `schemas.CONTROLS`.
   Consensus is the last one before Day 0 only.
2. `controls/firm.py`: `firm_controls`: size, bm, mom, vol_pre, n_analysts, dispersion, all
   measured before Day 0.
3. Reaction windows `car_m1_1` and `car_0_1` (`controls.reaction_legs`): same AR definition as
   `targets/car.py`. If agent E has not landed, use a small shared helper and tell agent E which.
4. Release metrics (`controls.release_metrics`) with `release_missing: flag_and_zero`: each
   missing metric gets a 0 plus a missing flag.
5. `version_final` from the selected transcript.
6. Wire `controls` into `cli.py` (`STAGES`), replacing `_not_yet("controls")`.
7. Do not standardise here: winsorising and z-scoring happen inside each training window
   (Standardisation section), so the controls stay raw.

## Checks to report
- Missing rate of every column, by year, compared with the coverage table in `docs/pipeline.md`
  (30,102 events, 2026-11-17 pull).
- Look-ahead check: for each control, the latest input date used is before Day 0. Report the
  number of violations (it must be 0).
- On the legacy 179 events: compare SUE and the reaction windows with the panel's columns.
- Tests: synthetic I/B/E/S and Compustat rows for the surprise, a forecast dated after Day 0
  (must be ignored), and a missing release metric.

## Do not
- Do not touch `targets/`, `events/`, `data/linking.py` or the transcripts code. Report
  problems with events to the user.

## Report back
Commit hash, the missing-rate table, the look-ahead check, the legacy comparison and the test result.
