"""Command line entry point: ``la <stage> [--config path]``.

Stages run in order; each reads the previous stage's parquet files, so any stage can be
re-run on its own. Unimplemented stages fail with NotImplementedError by design.
"""

from __future__ import annotations

import argparse

import pandas as pd

from linguistic_alpha.config import Config, load_config
from linguistic_alpha.data import transcripts
from linguistic_alpha.io import read_table, write_table


def stage_pull(cfg: Config, args: argparse.Namespace) -> None:
    from linguistic_alpha.data.wrds_pull import run_pull

    run_pull(cfg, groups=tuple(args.groups))


def stage_transcripts(cfg: Config, args: argparse.Namespace) -> None:
    """raw CIQ tables (or legacy panel.xlsx) -> interim components, qa_pairs, call_sections."""
    if args.legacy:
        panel = pd.read_excel(cfg.paths.legacy_panel, dtype={"gvkey": str})
        components = transcripts.components_from_legacy_panel(panel)
    else:
        detail = transcripts.select_latest_version(read_table(cfg.paths.raw,
                                                              "ciq_transcript_detail"))
        raw = read_table(cfg.paths.raw, "ciq_transcript_component")
        raw = raw[raw["transcriptid"].isin(detail["transcriptid"])]
        try:
            persons = read_table(cfg.paths.raw, "ciq_transcript_person")
        except FileNotFoundError:
            persons = None
        components = transcripts.build_components(raw, persons)
        write_table(detail, cfg.paths.interim, "transcript_calls")

    pairs = transcripts.pair_qa(components)
    sections = transcripts.call_sections(components)
    for name, df in [("components", components), ("qa_pairs", pairs),
                     ("call_sections", sections)]:
        path = write_table(df, cfg.paths.interim, name)
        print(f"wrote {path.relative_to(cfg.paths.interim.parent.parent)}: {len(df):,} rows")


def _not_yet(stage: str):
    def run(cfg: Config, args: argparse.Namespace) -> None:
        raise NotImplementedError(f"stage '{stage}' is specified but not implemented yet; "
                                  "see docs/pipeline.md")
    return run


STAGES = {
    "pull": stage_pull,
    "transcripts": stage_transcripts,
    "events": _not_yet("events"),        # linking + universe + Day 0
    "targets": _not_yet("targets"),
    "controls": _not_yet("controls"),
    "features": _not_yet("features"),
    "model": _not_yet("model"),
    "evaluate": _not_yet("evaluate"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="la", description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=None)
    parser.add_argument("--legacy", action="store_true",
                        help="transcripts: read the legacy panel.xlsx instead of raw CIQ files")
    parser.add_argument("--groups", nargs="+", default=["core", "transcripts"],
                        help="pull: query groups (core, transcripts, optional)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.paths.ensure()
    STAGES[args.stage](cfg, args)


if __name__ == "__main__":
    main()
