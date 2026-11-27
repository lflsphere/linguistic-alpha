# Handoff: `la targets` (agent E)

Starts after agent D (`docs/handoff_events.md`) has committed `data/processed/events`. Read
`docs/handoff_next_steps.md` (rules for every agent) and `docs/pipeline.md`: Event timing,
Missing daily returns and Sample period. Config: `targets.*` in `configs/default.yaml`.

## Goal
`la targets` writes `data/processed/targets` (one row per `keydevid`, `schemas.TARGETS`, checked
with `schemas.validate`).

## Inputs
- `data/processed/events` (Day 0, permno)
- `crsp_dsf_v2_<yyyy>` (daily `dlyret`, delisting returns included), `crsp_stkdelists`
- `crsp_dsp500_v2` (`vwretd`: S&P 500 total return, the main benchmark)
- `crsp_vwmkt_daily` (index check only)

## Work
1. `targets/car.py`: implement `abnormal_returns`, `event_car` and `post_call_volatility` to the
   docstring spec.
   - AR = R(stock) − R(S&P 500 total return).
   - CAR windows from `targets.car_windows`: `car_2_20` (primary), `car_2_5`, `car_2_60`.
   - Missing when valid days fall below `targets.min_window_coverage` (0.8).
   - Market model as robustness (`targets.market_model`: estimation window [−250, −30], at least
     80% valid days).
   - `vol_post_2_20` and `vol_ratio_2_20` (post vol / pre vol).
2. Wire `targets` into `cli.py` (`STAGES`), replacing `_not_yet("targets")`.
3. Reuse what fits from `scripts/build_reaction_windows.py` (legacy panel), but take Day 0 from
   `data/processed/events`.

## Checks to report
- Share of events with each target missing, by year. September 2026 events must be dropped
  (`sample.end`); `car_2_60` is expected to be missing from about June 2026.
- On the legacy 179 events: compare `car_2_20` with `data/reaction_windows.csv` and explain any
  difference over 1 bp.
- Mean CAR with the `vwretd` benchmark vs. `sprtrn`: the gap should be close to the dividend yield
  (`docs/pipeline.md` quotes +0.18 pp over [+2, +20]).
- Tests: synthetic returns with known CARs, missing days at and below the coverage threshold, and
  a delisting inside the window.

## Do not
- Do not compute the reaction windows `car_m1_1` / `car_0_1`: they are controls (agent F).
- Do not touch `controls/` or the events and transcripts code. If events look wrong, report it to
  the user rather than fixing agent D's stage.

## Report back
Commit hash, the missing-rate table, the legacy comparison, the benchmark check and the test result.
