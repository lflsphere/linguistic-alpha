"""Independently validate the staged source, output files and native model replay."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd

STUDIES = [
    "healthcare_sector_model_corrected", "cross_sector_linguistic_alpha",
    "it_bhuvi_replication_bridge",
]


def verify(output, original, raw):
    provenance = json.loads((output / "provenance.json").read_bytes())
    revision = provenance["revision"]
    git = ["git", "-c", f"safe.directory={Path.cwd().resolve().as_posix()}"]
    assert subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() == provenance["worktree_head"]
    assert subprocess.check_output([*git, "branch", "--show-current"], text=True).strip() == provenance["branch"]
    for name, digest in provenance["sha256"].items():
        b = (original / name).read_bytes()
        assert b == subprocess.check_output([*git, "show", f"{revision}:{name}"])
        assert hashlib.sha256(b).hexdigest() == digest
    backup_dir = Path(provenance["backup"])
    backup = json.loads((backup_dir / "backup_manifest.json").read_bytes())
    for name, digest in backup["files"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        assert (backup_dir / name).read_bytes() == Path(name).read_bytes()
    start = datetime.fromisoformat(backup["captured_utc"]).timestamp()
    prior = [p for run in STUDIES for p in (Path("outputs") / run).rglob("*") if p.is_file()]
    changed = [str(p) for p in prior if p.stat().st_mtime > start]
    if changed:
        raise AssertionError(f"Prior experiment files changed after source backup: {changed}")

    outputs = output / "earnings_health"
    check = output / "unprofiled_model_verification"
    if not check.exists():
        check.mkdir(exist_ok=False)
        for file in outputs.glob("model_table_opt*.parquet"):
            shutil.copyfile(file, check / file.name)
        command = [sys.executable, str(original / "sector_model_b.py"), "--sector", "health",
                   "--folder", str(check)]
        with (check / "run.log").open("xb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = sorted(check.glob("health_model_b_improved_*_results.json"))
    assert len(reports) == 2
    for file in reports:
        assert json.loads(file.read_bytes()) == json.loads((outputs / file.name).read_bytes())
    raw_check = output / "unprofiled_verification.json"
    native_verified = {"same_native_JSON_with_and_without_capture": True,
                       "reports": [r.name for r in reports]}
    if raw_check.exists():
        assert json.loads(raw_check.read_bytes()) == native_verified
    else:
        raw_check.write_text(json.dumps(native_verified, indent=2) + "\n", encoding="utf-8")

    native = pd.read_csv(output / "original_results.csv")
    matches = pd.read_csv(output / "matched_event_comparison.csv")
    targets = pd.read_csv(output / "target_reconciliation_summary.csv")
    assert len(native) == 6 and set(native.n_test) == {204} and set(native.n_train) == {2939}
    assert set(pd.to_datetime(native.test_first).dt.year) == {2024}
    assert set(pd.to_datetime(native.test_last).dt.year) == {2024}
    assert set(matches.n_test) == {175} and len(matches) == 32
    assert set(targets.n_benchmark_only_equal_within_1e_7) == {175}
    assert set(targets.day0_differences) == {0}
    native_scores = pd.read_csv(output / "native_strategy_scores.csv")
    assert len(native_scores) == 12 and native_scores.groupby("option").selected_as_b.sum().eq(1).all()
    health = pd.read_parquet(outputs / "model_table_opt1.parquet",
                             columns=["transcriptid", "call_date", "call_time_utc", "sector"])
    assert health.sector.eq("Health Care").all() and not health.transcriptid.duplicated().any()
    detail = pd.read_parquet(raw / "ciq_transcript_detail.parquet", columns=[
        "transcriptid", "mostimportantdateutc", "mostimportanttimeutc",
    ])
    detail = detail.loc[detail.transcriptid.isin(health.transcriptid)].drop_duplicates()
    detail["call_ts"] = pd.to_datetime(
        detail.mostimportantdateutc + " " + detail.mostimportanttimeutc, utc=True, errors="raise",
    )
    health["call_ts"] = pd.to_datetime(
        pd.to_datetime(health.call_date).dt.strftime("%Y-%m-%d") + " " + health.call_time_utc,
        utc=True, errors="raise",
    )
    exact = health.merge(detail[["transcriptid", "call_ts"]], on=["transcriptid", "call_ts"],
                         validate="one_to_one")
    assert len(exact) == len(health)
    for file in output.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8-sig"), str(file))
    for name in ("summary.md", "methodology_comparison.md", "reproduction.md", "code_provenance.md"):
        doc = output / name
        if doc.is_file():
            links = re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8"))
            assert all(link.startswith("https://") or (output / link).is_file() for link in links)
    artifact = {
        "original_hashes_and_native_reports_verified": True,
        "source_backup_files_unchanged": len(backup["files"]),
        "historical_output_files_unchanged_since_backup": len(prior),
        "original_results_rows": len(native), "matched_results_rows": len(matches),
        "shared_events_each_benchmark": 175,
        "native_healthcare_transcripts_with_exact_ciq_clock": len(exact),
    }
    report = output / "verification.json"
    if report.exists():
        assert json.loads(report.read_bytes()) == artifact
    else:
        report.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path("experiments/pierre_original"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    verify(args.out, args.original, args.raw)

