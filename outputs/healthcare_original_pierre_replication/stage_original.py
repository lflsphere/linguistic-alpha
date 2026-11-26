"""Export Pierre's exact Git blobs and validate the existing source-only backup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = "06273f1766daeb29de4b9313dd20639439a09671"
ORIGINAL = (
    "sector_config.py", "sector_extract.py", "sector_model_b.py",
    "src/linguistic_alpha/__init__.py",
    "src/linguistic_alpha/events/__init__.py",
    "src/linguistic_alpha/events/timing.py",
)


def stage(original, output, backup):
    git = ["git", "-c", f"safe.directory={Path.cwd().resolve().as_posix()}"]

    def read(*args):
        return subprocess.check_output([*git, *args])

    saved = json.loads((backup / "backup_manifest.json").read_text(encoding="utf-8"))
    branch = read("branch", "--show-current").decode().strip()
    head = read("rev-parse", "HEAD").decode().strip()
    if (branch, head) != (saved["branch"], saved["head"]):
        raise ValueError("The saved source backup belongs to a different branch or HEAD")
    for filename, checksum in saved["files"].items():
        if hashlib.sha256((backup / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Source backup copy failed its checksum: {filename}")
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Working research source has changed since backup: {filename}")
    digests = {}
    for filename in ORIGINAL:
        contents = read("show", f"{REVISION}:{filename}")
        target = original / filename
        if target.exists():
            if contents != target.read_bytes():
                raise FileExistsError(f"Existing isolated source differs from Pierre: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(contents)
        digests[filename] = hashlib.sha256(contents).hexdigest()
    details = {
        "revision": REVISION, "branch": branch, "worktree_head": head,
        "backup": str(backup.resolve()), "backup_source_files": len(saved["files"]),
        "sha256": digests, "source_diff": "empty (byte-for-byte original blobs)",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "provenance.json").open("x", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
        f.write("\n")
    print(json.dumps(details, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("experiments/legacy_pipeline"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    stage(args.original, args.out, args.backup)

