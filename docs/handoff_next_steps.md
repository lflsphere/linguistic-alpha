# Next steps after F3 calibration (index)

Status on 2026-11-27: labeller A, labeller B and agent C are done. `01` / `02 --source full` ran
(commit `0bb1a46` on `spec/event-timing-models`). F3 does **not** pass: OOF AUC 0.629
(EVADE_CLF) and 0.674 (NONANS_CLF), against a bar of 0.7. Kappa A vs. B: 0.824 (evasive),
0.883 (non_answer).

## Agent tasks

| Brief | Agent | Starts when | Writes (main files) |
|---|---|---|---|
| `docs/handoff_events.md` | D: `la events` + transcripts select | now | `data/linking.py`, `data/universe.py`, `data/transcripts.py`, `cli.py` (events, transcripts), `data/processed/events` |
| `docs/handoff_targets.md` | E: `la targets` | after D lands | `targets/car.py`, `cli.py` (targets), `data/processed/targets` |
| `docs/handoff_controls.md` | F: `la controls` | after D lands | `controls/*.py`, `cli.py` (controls), `data/processed/controls` |
| `docs/handoff_f3_relevance.md` | G: a stronger F3 relevance signal | now | `scripts/f3/`, `outputs/validation/f3_relevance/` |

D and G can run at the same time: they touch different files. E and F can run at the same time
after D. In `cli.py`, each agent edits only its own stage's entry in `STAGES` and its own stage
function. After E and F, `03`–`05` can be connected to `data/processed/` (a later brief).

## Rules for every agent
- Main checkout `linguistic-alpha/`, branch `spec/event-timing-models`. Run
  `git branch --show-current` first and stop if it differs. Do not switch branches, stash or reset.
- Python: `.venv/bin/python`. Config: `configs/default.yaml`. Raw data: `data/raw/sp500_pit_2008/`.
- Do not edit or overwrite `data/qa_labels.csv`, `data/qa_labels_b.csv`,
  `data/qa_label_sample_full.csv`, `data/features.csv` or `data/features_meta.json`.
- Do not run `03`–`05`, `la holdout` or `--freeze`. Do not look at holdout results (2024-01-01 on).
- Tests go in `tests/`. `pytest` must pass before you report.
- Commit only your own files, on this branch, with a message saying what changed and why. Do not push.
- Never ask for, store or print the WRDS password.
- Run anything over about 2 minutes in the background, and do not poll it.

## To-dos for a person (not agents)
1. **Human check:** fill the blank human columns in `outputs/validation/label_human_check.csv`
   (B's 7 disagreements plus 20 random pairs where A and B agree). Both labellers are LLMs.
2. **B's resolutions:** sign off, or not, on B's 7 proposed resolutions
   (`outputs/validation/label_disagreements.csv`). 5 of them would change A's labels. Until then
   `data/qa_labels.csv` stays as A wrote it.
3. **B's rule wording:** decide on the two changes B proposed (evasive only when the question
   asked for something specific; non_answer only when the decline covers what was asked).
4. **keep_first changes from agent C:** approve or reject (a) CIQ's 00:00 UTC placeholder time
   sorting last within its date, and (b) duplicates counting only within 3 calendar days for the
   same gvkey and fiscal label. Agent D implements them as agent C did and flags them in
   `docs/pipeline.md`, so they can be reverted.
