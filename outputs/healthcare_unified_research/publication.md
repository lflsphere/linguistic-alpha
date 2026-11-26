# Healthcare research checkpoint

The unified report and dashboard cover the 24 annual test settings and all five models. The sibling `healthcare_window_horizon_comparison` directory preserves the A/B/B* comparison. Its `phase1_style.css` is the stylesheet from the original Phase 1 dashboard and lets the comparison renderer work from the archived aggregates. The pinned Pierre code under `experiments/pierre_original/` supplies the historical preprocessing and event-time implementation.

## Review from a fresh checkout

Open `dashboard.html` locally to explore the saved aggregates. Its CSV links resolve within this directory. The English PDF links to the same aggregate tables on the GitHub research branch. The appendix link targets were updated for publication. The rendered pages and research numbers are identical to the completed local report.

Use `annual_model_performance.csv`, `annual_increments.csv`, `paired_inference.csv` and `finbert_selected_features.csv` for numerical review. The frozen specification, input digests, and validation records document the methods and checks. The scripts can assemble both reports from the saved aggregate tables using Python 3.12, pandas and Chromium. The earlier comparison renderer also uses NumPy and Matplotlib.

## Inputs for a full rerun

Fit and replay checks require an authorized local copy of the source data and event outputs. The local file digests and relative paths are recorded in `data_provenance.json` and `integrity.json`. The pinned Pierre provenance manifest is generated locally by the original replication staging script.

- Earlier A/B/B* event predictions and timing reside in the sibling comparison directory as `matched_predictions_local.csv` and `event_timing_local.csv`.
- The original Healthcare option tables, benchmark files and workstation provenance reside in the Pierre replication directory. Its `reproduction.md` documents how to reconstruct them from authorized inputs. The CAR and financial extension directories document the upstream local tables.
- The FinBERT call features, sentence scores, SQLite cache, public model weights, and runtime files reside under `data/derived/finbert/`. The extraction and inference validation require the pinned model and dependencies described in `finbert_methodology.md`.
- The five-model event predictions and fitted transformations are local Parquet outputs. The saved model, factor, coverage and interval tables on this branch are aggregates.

The report assembly uses the aggregate tables in this checkpoint. Replaying the underlying fits requires the licensed inputs and local predictions described above.
