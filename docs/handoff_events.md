# Handoff: `la events` and the transcripts select pass (agent D)

Read `docs/handoff_next_steps.md` (rules for every agent) and `docs/pipeline.md`: Stage
contracts, Universe, Event timing and Sample period. That spec is the source of truth. This
brief only adds what has changed since.

## Goal
`la events` writes `data/processed/events` (one row per `keydevid`, `schemas.EVENTS`, checked with
`schemas.validate`) for the S&P 500 sample, 2011-01-01 to 2026-08-31. Each exclusion is kept with an
`exclusion_reason` in a side table (`data/processed/events_excluded`), not dropped silently.
Then `la transcripts` is split into two passes, as `docs/pipeline.md` ("Stage order") asks.

## Inputs
- `data/interim/transcript_calls`, `components` (from `la transcripts`, already run)
- `data/raw/sp500_pit_2008/`: `ciq_wrds_gvkey`, `crsp_ccmxpf_lnkhist`, `wrdsapps_ibcrsphist`,
  `crsp_msp500list_v2`, `comp_co_hgic`, `ibes_actu_epsus` (release `anndats` / `anntims`),
  `crsp_dsf_v2_<yyyy>` (trading calendar and Day -1 market cap)

## Work
1. **Linking** (`data/linking.py`): implement `link_ciq_to_gvkey`, `link_gvkey_to_permno`
   (CCM `LU`/`LC`, date-valid), `link_permno_to_ibes` (`wrdsapps_ibcrsphist`, score ≤ 2, date-valid)
   and `build_event_links`.
2. **Universe** (`data/universe.py`): `sp500_member_on` on Day 0 and `apply_exclusions`
   (`sample.*` in the config). One permno per event: `share_class_rule: ccm_primary_then_mktcap`.
3. **Day 0** (`events/timing.py` is implemented and tested): release time from I/B/E/S
   (New York time), `market_close_et`, `missing_call_time: next_session`, `call_lag_sessions`,
   flag `max_call_lag_sessions` > 1.
4. **One call per release** (`events.duplicate_calls: keep_first`). Port the rule from
   `scripts/full_source.py` into `src/`, including agent C's two changes, which still need team
   approval:
   - CIQ's 00:00:00 UTC placeholder time sorts last within its date.
   - Two calls are duplicates only for the same gvkey and fiscal label within 3 calendar days.

   Write both in `docs/pipeline.md` (Universe) marked "pending team approval", behind config
   switches (`events.placeholder_time_last`, `events.duplicate_window_days`), so they can be turned off.
5. **Transcripts select pass:** `la transcripts` today selects the version before events, without
   the entry cap. Split it: (1) parse and clean every version → `interim/transcript_versions`,
   `components`; (2) after events, select one version per `keydevid` with
   `transcripts.entry_close_utc` → `interim/transcript_calls`, `qa_pairs`, `call_sections`.
   Keep the old outputs under `data/interim/legacy_select/` until the new ones are checked.
   `pair_id`s for labelled pairs must not change: check that all 414 `pair_id`s in
   `data/qa_labels.csv` still exist, and report any that move.
6. Wire `events` into `cli.py` (`STAGES`), replacing `_not_yet("events")`.

## Checks to report
- An attrition table like `outputs/validation/full_source_log.md`. Compare with agent C's call
  table (31,974 calls after keep_first; 29,968 written for 2011-01 to 2026-08) and with the
  30,102 events in `docs/pipeline.md` (release metric coverage). Explain each gap over 1%.
- How many events have a missing release time, a call lag of 2+ sessions, or no IBES ticker.
- Events per year and per GICS sector.
- Tests: synthetic cases for each link rule, share classes, the placeholder time and the 3-day window.

## Do not
- Do not run `01`, `02` or `03`–`05`. Do not rebuild embeddings.
- Do not change `scripts/full_source.py` beyond importing the rule from `src/`, if that is cleaner.

## Report back
Commit hash, the attrition table, the checks above, the counts of excluded events by reason, and
anything in `docs/pipeline.md` that turned out wrong.
