"""Check or replay the published Healthcare research in a fresh local directory."""

import argparse
from hashlib import file_digest
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "outputs/healthcare_window_horizon_comparison"
UNIFIED = ROOT / "outputs/healthcare_unified_research"
PROTOCOL = COMPARISON / "frozen_research_spec.json"
WORKSTATION_MANIFEST = "outputs/healthcare_original_pierre_replication/provenance.json"
GIT = ["git", "--no-optional-locks", "-c", f"safe.directory={ROOT.as_posix()}"]
MODULES = {
    "run_comparison": COMPARISON,
    "research_metrics": COMPARISON,
    "render_comparison": COMPARISON,
    "extract_finbert": UNIFIED,
    "fit_finbert": UNIFIED,
    "evaluate_unified": UNIFIED,
    "verify_unified": UNIFIED,
    "render_unified": UNIFIED,
}


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_frozen_baseline():
    """Check the preservation commit by ancestry so descendants can replay the run."""
    protocol = read_json(PROTOCOL)
    static_dir = "outputs/healthcare_extended_oos"
    evaluator = f"{static_dir}/evaluate.py"
    relocated = (evaluator, f"{static_dir}/methodology.md")
    archived_model = Path(read_json(ROOT / static_dir / "frozen_experiment_spec.json")["source_model"])
    legacy_dir = Path("experiments/legacy_pipeline")
    subprocess.run(
        [*GIT, "merge-base", "--is-ancestor", protocol["preservation_commit"], "HEAD"],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [*GIT, "diff", "--quiet", protocol["protected_baseline"], "--",
         static_dir, "outputs/sp500_car_extension", "outputs/sp500_financial_extension",
         *(f":(exclude){name}" for name in relocated)],
        cwd=ROOT, check=True,
    )
    # The protected files may differ only by the historical source directory name.
    for name in relocated:
        archived = subprocess.check_output(
            [*GIT, "show", f"{protocol['protected_baseline']}:{name}"], cwd=ROOT,
        )
        expected = archived.replace(archived_model.parent.as_posix().encode(), legacy_dir.as_posix().encode())
        if (ROOT / name).read_bytes().replace(b"\r\n", b"\n") != expected:
            raise ValueError(f"Frozen baseline code or documentation has changed: {name}")
    manifest = read_json(COMPARISON / "data_provenance.json")
    for filename, expected in manifest["analysis_source_sha256"].items():
        if sha256(COMPARISON / filename) != expected:
            raise ValueError(f"Frozen comparison code has changed: {filename}")
    sources = manifest["source_sha256"]
    sources[(legacy_dir / archived_model.name).as_posix()] = sources.pop(archived_model.as_posix())
    sources[evaluator] = sha256(ROOT / evaluator)
    return sources


def local_module(name):
    module = import_module(name)
    if Path(module.__file__).resolve() != MODULES[name] / f"{name}.py":
        raise ValueError(f"Unexpected source for {name}: {module.__file__}")
    return module


def fresh_directory(path):
    path = path.resolve()
    if path.is_relative_to(ROOT):
        # Use a regular output name to check that the destination is ignored.
        probe = (path / "integrity.json").relative_to(ROOT).as_posix()
        subprocess.run([*GIT, "check-ignore", "--quiet", "--", probe], cwd=ROOT, check=True)
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_frozen_inputs(expected, observed_digest):
    # The input adapter validates the revision and source digests in this local manifest.
    for name, digest in expected.items():
        if name != WORKSTATION_MANIFEST and observed_digest(name) != digest:
            raise ValueError(f"Replay input differs from the frozen run: {name}")


def replay_ab(path, frozen_inputs):
    baseline = local_module("run_comparison")
    metrics = local_module("research_metrics")
    out = fresh_directory(path)
    (out / PROTOCOL.name).write_bytes(PROTOCOL.read_bytes())
    native, frames, provenance = baseline.inputs(out)
    check_frozen_inputs(frozen_inputs, provenance.get)
    predictions, cutoffs = baseline.experiment(out, native, frames)
    metrics.evaluate(out, frames, predictions, cutoffs)
    baseline.need(
        all(sha256(ROOT / name) == digest for name, digest in provenance.items()),
        "Input changed during replay",
    )


def replay_finbert(path, frozen_inputs):
    fitter = local_module("fit_finbert")
    feature_store = local_module("extract_finbert")
    verifier = local_module("verify_unified")
    spec = read_json(UNIFIED / "frozen_unified_spec.json")
    if fitter.snapshot() != spec["existing_outputs_sha256"]:
        raise ValueError("Local A/B/B* artifacts differ from the frozen unified run")
    if sha256(feature_store.STORE / "feature_manifest.json") != spec["finbert_feature_manifest_sha256"]:
        raise ValueError("Local FinBERT feature manifest differs from the frozen unified run")
    check_frozen_inputs(frozen_inputs, lambda name: sha256(ROOT / name))
    out = fresh_directory(path)
    fitter.run(out)
    verifier.verify_results(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check frozen sources and imports without reading licensed data")
    for name, help_text in (
        ("replay-ab", "Fit the A/B/B* comparison using the frozen research protocol"),
        ("replay-finbert", "Fit C/C* using the saved A/B/B* forecasts and FinBERT cache"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--out", type=Path, required=True, help="New ignored or external directory")
    args = parser.parse_args()

    sys.dont_write_bytecode = True
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path[:0] = [str(COMPARISON), str(UNIFIED)]
    frozen_inputs = check_frozen_baseline()
    if args.command == "check":
        for name in MODULES:
            local_module(name)
        print("Preservation history, frozen source hashes and Healthcare imports verified.")
    elif args.command == "replay-ab":
        replay_ab(args.out, frozen_inputs)
    else:
        replay_finbert(args.out, frozen_inputs)


if __name__ == "__main__":
    main()
