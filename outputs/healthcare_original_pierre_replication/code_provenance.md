# Pierre source provenance and input adapter

Pierre P’s local `origin/main` revision `06273f1766daeb29de4b9313dd20639439a09671` is the verified source. Its commit subject is `refactor in sector`. The working branch stayed `feature/healthcare-linguistic-alpha` at HEAD `66d1bcfda80194d9f3b768d5b2a8d5f309a11965`. Before staging or running the experiment, a local-only backup captured 23 source and test files outside the repository.

The original backup was created under the workstation’s temporary directory. Its local manifest records each file’s SHA-256. Backup contents were limited to source and tests. Reproduction requires a fresh source-only backup for the current branch and HEAD.

| Exact Git blob copied to `experiments/legacy_pipeline/` | SHA-256 |
|---|---|
| `sector_config.py` | `df0caffdc9cb039e58b6e7fca28a69b64db9c5fee0d83287b2074d27464b5e6d` |
| `sector_extract.py` | `33313ebf346e6b3ac52224f8de8a64a6929f8b5aef4917082464376289dc5956` |
| `sector_model_b.py` | `e2f9986a8f809e5b368b97eac0421adfa6009e49a07c3bafa9a4d416afa124f2` |
| `src/linguistic_alpha/events/timing.py` | `e32879c0a26d656b4f2c09badb7c00425acfcda8186ed8a9c1bcf1470bd18d98` |

The source diff is empty. The three source files and the helper package were extracted with `git show` and compared byte-for-byte with the commit. The original extraction calls the staged timing helper. The original model’s formulas, ranked target, features, candidate estimators and reporting code execute untouched. A return-only Python profiler captures the original `evaluate()` predictions and split metadata. A second execution of the original CLI without the profiler produces the identical two native metric JSONs.

## Local interface mapping

| Original logical input | Adapter data source | What changed |
|---|---|---|
| `model_table.parquet` | `model_table_with_linguistic.csv` | Read the CSV in memory with round-trip float parsing and integral identifiers; retain the original sector label, controls, structure and nine LM inputs. |
| `event_panel.parquet` | Projection of the same CSV | Read `keydevid`, GVKEY, PERMNO, UTC date and time. The original extractor then applies its own sector and Day 0 logic. A distinct full VM event panel was unavailable. |
| `raw/company.parquet` | `comp_company.parquet`, CSV GICS industry codes and `comp_co_hgic.parquet` | Populate one static classification per GVKEY. Use the CSV snapshot for companies present in the export, and the latest locally dated GICS record for other Compustat firms. The latter may be an ended classification spell. Keeps the original static-classification rule. |
| `raw/sp500_index_daily.parquet` | `crsp_dsi.parquet`, `date,sprtrn` | Pierre’s upstream `sp500_pipeline.py:177` requests these two columns. Only SPRTRN is exposed, so the extractor uses its original SPRTRN branch. |
| `raw/crsp_daily_*.parquet` | `crsp_dsf_*.parquet` | Alias shards and expose `permno,date,ret,prc,shrout`. Convert nullable numeric storage to native floats so NumPy receives NaN. Keep observed values, membership and missingness. |
| `raw/link_ccm.parquet` | `crsp_ccmxpf_lnkhist.parquet` | Project GVKEY, PERMNO and link dates. Normalize GVKEY padding for joins. |
| `raw/sp500_membership.parquet` | `crsp_msp500list.parquet` | Project original date and security fields. |
| `raw/crsp_delist.parquet` | `crsp_dsedelist.parquet` | Project delisting date and return in the extractor’s expected schema. |

These mappings substitute for the original VM input snapshot. This is a local-input replication of unchanged code. Of the 128 static Healthcare GVKEYs in the reconstructed company input, 96 come from the CSV and 32 from the latest-record fallback. The reconstructed snapshot agrees with the Healthcare label of all exported calls. A total of 545 company rows lacked classification evidence. The original unrestricted full company-sector snapshot and VM index/return files were unavailable for byte-level data replication. Option 2 therefore carries an additional benchmark membership approximation.

The adapter supplies virtual read aliases rather than copying raw shards. It scopes Windows `USERPROFILE` to the new output folder during extraction, so the untouched `~/earnings_health` path resolves there. The second, unprofiled model execution uses copies of the two newly derived small Healthcare tables in a separate verification folder. Existing research outputs are read-only inputs to the comparison.

The stage script generates a local `provenance.json` with the pinned hashes and the backup path.

[Schema and coverage audit](input_schema_audit.json) · [External adapter and capture](run_original.py) · [Independent native report verification](unprofiled_verification.json)
