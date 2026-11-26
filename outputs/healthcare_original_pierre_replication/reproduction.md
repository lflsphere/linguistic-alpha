# Local reproduction in PowerShell

Run from the repository root. The commands recover the exact Pierre Git revision and write to a fresh destination inside the results directory. The completed experiment used Python 3.12.10, pandas 2.2.3 and scikit-learn 1.9.1.

## Local prerequisites

- The local Git object for revision `06273f1766daeb29de4b9313dd20639439a09671`. The staging script reconstructs `experiments/legacy_pipeline/` from that revision.
- Authorized local inputs `model_table_with_linguistic.csv` and the CRSP and Compustat Parquets listed in the adapter mapping, plus `data/raw/ciq_transcript_detail.parquet` for clock verification.
- Local research helper modules at the repository root: `cross_sector.py`, `sector_model_b.py`, `sector_config.py`, `sector_pit_audit.py`, and `sector_extract.py`. The comparison uses these helpers and the repository’s `src/linguistic_alpha/` package. The original extraction and fit use the isolated Pierre sources.
- The prior `outputs/healthcare_sector_model_corrected/` run. The comparison reads its extracted option tables and sector benchmark, the frozen predictions from the transcript runs, and the metrics and frozen specifications from the transcript and immediate runs.

The commit contains scripts, documentation and aggregate results. An authorized local copy of the inputs and prior research is required for a full rerun. Event ledgers, predictions, Parquets, native logs, and workstation-specific manifests are generated locally. [Input mapping](code_provenance.md)

## Run the replication

```powershell
$entry = 'outputs/healthcare_original_pierre_replication'
$run = "$entry/reproduction"
$backup = Join-Path ([IO.Path]::GetTempPath()) ('linguistic-alpha-pierre-source-' + [Guid]::NewGuid().ToString('N'))
$py = (Resolve-Path '.venv/Scripts/python.exe').Path
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONPATH = '.;src'

@'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path.cwd().resolve()
backup = Path(sys.argv[1]).resolve()
if backup.is_relative_to(repo):
    raise ValueError("Use a backup directory outside the repository")
backup.mkdir(parents=True, exist_ok=False)
git = ["git", "-c", f"safe.directory={repo.as_posix()}"]

def git_text(*args):
    return subprocess.check_output([*git, *args], encoding="utf-8").strip()

changed = git_text("ls-files", "-z", "--modified", "--others", "--exclude-standard")
sources = {Path(name) for name in changed.split("\0") if name and Path(name).suffix == ".py"}
sources.update(Path("tests").rglob("test*.py"))
manifest = {
    "branch": git_text("branch", "--show-current"),
    "head": git_text("rev-parse", "HEAD"),
    "captured_utc": datetime.now(timezone.utc).isoformat(),
    "files": {},
}
for source in sorted(sources):
    if source.is_file():
        contents = source.read_bytes()
        copy = backup / source
        copy.parent.mkdir(parents=True, exist_ok=True)
        with copy.open("xb") as handle:
            handle.write(contents)
        manifest["files"][source.as_posix()] = hashlib.sha256(contents).hexdigest()
(backup / "backup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
'@ | & $py - $backup
if ($LASTEXITCODE -ne 0) { throw 'Source-only backup failed' }

& $py "$entry/stage_original.py" --out $run --backup $backup
if ($LASTEXITCODE -ne 0) { throw 'Source or backup verification failed' }

& $py "$entry/run_original.py" extract --out $run
if ($LASTEXITCODE -ne 0) { throw 'Original extraction failed' }

& $py "$entry/run_original.py" model --out $run
if ($LASTEXITCODE -ne 0) { throw 'Original Model A/B fitting failed' }

& $py "$entry/compare_predictions.py" --out $run --bootstrap 2000
if ($LASTEXITCODE -ne 0) { throw 'Frozen forecast comparison failed' }

& $py "$entry/verify_replication.py" --out $run
if ($LASTEXITCODE -ne 0) { throw 'Independent verification failed' }
```

The stage command compares the external backup with the current working source and verifies the isolated copies against the pinned revision. Create the backup after any branch or HEAD change. The final verification runs the original model CLI without a profiler on copies of the newly built tables and compares the native JSON reports.

[Original results](original_results.csv) · [Frozen comparison](matched_event_comparison.csv)
