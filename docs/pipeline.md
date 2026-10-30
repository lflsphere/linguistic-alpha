# Pipeline architecture

Question: do earnings-call language features (Model B) predict CAR[+2,+20] beyond the earnings
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
| events | `interim/transcript_calls`, `ciq_wrds_gvkey`, `crsp_ccmxpf_lnkhist`, `wrdsapps_ibcrsphist`, `crsp_msp500list`, `comp_co_hgic`, `crsp_dsi` | `processed/events` (`schemas.EVENTS`) | keydevid |
| targets | `processed/events`, `crsp_dsf`, `crsp_dsi`, `crsp_dsedelist`, `ff_factors_daily` | `processed/targets` (`schemas.TARGETS`) | keydevid |
| controls | `processed/events`, `crsp_dsf`, `comp_fundq`, `ibes_actu_epsus`, `ibes_statsumu_epsus` | `processed/controls` (`schemas.CONTROLS`) | keydevid |
| features | `interim/*`, `processed/events`, (8-K text for F5) | `processed/features_<family>` | transcriptid |
| model | `processed/{targets,controls,features_*}` | `processed/predictions`, coefficient tables | keydevid × model |
| evaluate | predictions, returns, cost data | reports | |

Raw file names mirror WRDS tables (`library_table`). Big tables are sharded `name_<yyyy>`.
`io.read_table` reads both parquet and CSV, single files or shards.

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

## Data still to retrieve

CRSP (`dsf`, `dsi`, `dsedelist`, `msp500list`, `ccmxpf_lnkhist`), Compustat (`fundq`,
`co_hgic`), I/B/E/S (`actu_epsus`, `statsumu_epsus`, `surpsum`), ICLINK, FF factors.
All queries are listed in `data/wrds_pull.py::QUERIES`.
