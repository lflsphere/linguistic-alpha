"""Command line entry point: ``la <stage> [--config path]``.

Stages run in order; each reads the previous stage's parquet files, so any stage can be
re-run on its own. Unimplemented stages fail with NotImplementedError by design.
"""

from __future__ import annotations

import argparse

import pandas as pd

from linguistic_alpha.config import REPO_ROOT, Config, load_config
from linguistic_alpha.data import transcripts
from linguistic_alpha.io import read_table, read_table_isin, write_table


def stage_pull(cfg: Config, args: argparse.Namespace) -> None:
    from linguistic_alpha.data.wrds_pull import run_pull

    run_pull(cfg, groups=tuple(args.groups))


def _trading_days(cfg: Config) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(read_table(cfg.paths.raw, "crsp_dsf_v2",
                                       columns=["dlycaldt"])["dlycaldt"].unique())


def _wrote(cfg: Config, path, n: int) -> None:
    print(f"wrote {path.relative_to(cfg.paths.interim.parent.parent)}: {n:,} rows")


def _raw_shards(folder, name: str) -> list:
    return sorted(folder.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))


def stage_transcripts(cfg: Config, args: argparse.Namespace) -> None:
    """Two passes (docs/pipeline.md, Stage order), with ``la events`` in between.

    ``--pass parse``: every version -> ``interim/transcript_versions`` (all-version metadata) and
    ``interim/components_<yyyy>`` (one shard per raw component shard; all versions do not fit in
    one frame). ``--pass select``: one version per keydevid (``transcripts.select_versions``,
    capped at the entry close where ``la events`` found a release Day 0), for every call, then
    ``interim/transcript_calls``, ``qa_pairs``, ``call_sections``.
    ``--legacy``: components from the legacy panel.xlsx, then pairs and sections.
    """
    interim = cfg.paths.interim
    if args.legacy:
        panel = pd.read_excel(cfg.paths.legacy_panel, dtype={"gvkey": str})
        components = transcripts.components_from_legacy_panel(panel)
        _write_text_tables(cfg, components, with_components=True)
        return

    if args.pass_ == "parse":
        detail = read_table(cfg.paths.raw, "ciq_transcript_detail")
        _wrote(cfg, write_table(transcripts.annotate_versions(detail), interim,
                                "transcript_versions"), len(detail))
        for shard in _raw_shards(cfg.paths.raw, "ciq_transcript_component"):
            year = shard.stem[-4:]
            raw = pd.read_parquet(shard)
            tids = raw["transcriptid"].dropna().unique()
            try:
                persons = read_table_isin(cfg.paths.raw, "ciq_transcript_person", "transcriptid",
                                          tids, columns=["transcriptid", "transcriptpersonid",
                                                         "speakertypename"])
            except FileNotFoundError:
                persons = None
            comp = transcripts.build_components(raw, persons)
            _wrote(cfg, write_table(comp, interim, f"components_{year}"), len(comp))
        # io.read_table prefers a single file over shards: retire the old selected-only table
        single = interim / "components.parquet"
        if single.exists():
            legacy = interim / "legacy_select" / "components.parquet"
            legacy.parent.mkdir(exist_ok=True)
            if legacy.exists():
                single.unlink()
            else:
                single.replace(legacy)
            print(f"retired {single.name} (copy in {legacy.parent.name}/)")
        return

    # select pass
    versions = read_table(interim, "transcript_versions")
    day0 = None
    parts = []
    for name in ("events", "events_excluded"):
        try:
            parts.append(read_table(cfg.paths.processed, name, columns=["keydevid", "day0"]))
        except FileNotFoundError:
            pass
    if parts:
        d = pd.concat(parts).dropna().drop_duplicates("keydevid")
        day0 = d.set_index("keydevid")["day0"]
    else:
        print("no processed/events yet: version cutoff not capped at the entry (run la events)")
    detail = transcripts.select_versions(versions, _trading_days(cfg), cfg.raw, release_day0=day0)
    keep = detail["transcriptid"].dropna().unique()
    components = read_table_isin(interim, "components", "transcriptid", keep)
    _wrote(cfg, write_table(detail, interim, "transcript_calls"), len(detail))
    _write_text_tables(cfg, components, with_components=False)


def _write_text_tables(cfg: Config, components: pd.DataFrame, with_components: bool) -> None:
    out = [("qa_pairs", transcripts.pair_qa(components)),
           ("call_sections", transcripts.call_sections(components))]
    if with_components:
        out.insert(0, ("components", components))
    for name, df in out:
        _wrote(cfg, write_table(df, cfg.paths.interim, name), len(df))


def stage_events(cfg: Config, args: argparse.Namespace) -> None:
    """interim/transcript_versions + raw links, I/B/E/S, CRSP -> processed/events, events_excluded,
    and the attrition log outputs/validation/events_log.md."""
    from linguistic_alpha.events.build import build_events, write_log

    raw = {name: read_table(cfg.paths.raw, name) for name in
           ("ciq_wrds_gvkey", "crsp_ccmxpf_lnkhist", "wrdsapps_ibcrsphist", "crsp_msp500list_v2",
            "comp_co_hgic", "ibes_actu_epsus")}
    raw["crsp_dsf_v2"] = read_table(cfg.paths.raw, "crsp_dsf_v2",
                                    columns=["permno", "dlycaldt", "dlyret", "dlycap"])
    versions = read_table(cfg.paths.interim, "transcript_versions")
    events, excluded, log = build_events(versions, raw, cfg.raw)
    for name, df in (("events", events), ("events_excluded", excluded)):
        path = write_table(df, cfg.paths.processed, name)
        print(f"wrote {path.relative_to(cfg.paths.processed.parent.parent)}: {len(df):,} rows")
    out = write_log(REPO_ROOT / "outputs" / "validation" / "events_log.md", events, excluded, log,
                    cfg.raw)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def stage_targets(cfg: Config, args: argparse.Namespace) -> None:
    """processed/events + CRSP daily (crsp_dsf_v2, crsp_dsp500_v2, crsp_stkdelists) ->
    processed/targets (schemas.TARGETS plus coverage, market-model and delisting columns)."""
    from linguistic_alpha import schemas
    from linguistic_alpha.targets.car import build_targets

    tcfg = cfg.raw["targets"]
    events = read_table(cfg.paths.processed, "events", columns=["keydevid", "permno", "day0"])
    s = cfg.raw["sample"]
    events = events[(events["day0"] >= s["start"]) & (events["day0"] <= s["end"])]
    events = events.reset_index(drop=True)
    dsf = read_table_isin(cfg.paths.raw, "crsp_dsf_v2", "permno", events["permno"].unique(),
                          columns=["permno", "dlycaldt", "dlyret"])
    if tcfg.get("benchmark") == "vwmkt":
        bench = read_table(cfg.paths.raw, "crsp_vwmkt_daily")
    else:
        bench = read_table(cfg.paths.raw, "crsp_dsp500_v2")
    try:
        delists = read_table(cfg.paths.raw, "crsp_stkdelists", columns=["permno", "delistingdt"])
    except FileNotFoundError:
        delists = None
    out = build_targets(events, dsf, bench, delists, tcfg)
    schemas.validate(out, schemas.TARGETS, "targets", key="keydevid")
    path = write_table(out, cfg.paths.processed, "targets")
    print(f"wrote {path.relative_to(cfg.paths.processed.parent.parent)}: {len(out):,} rows")
    for c in tcfg["car_windows"]:
        print(f"  {c}: missing {out[c].isna().mean():.2%}")


def _not_yet(stage: str):
    def run(cfg: Config, args: argparse.Namespace) -> None:
        raise NotImplementedError(f"stage '{stage}' is specified but not implemented yet; "
                                  "see docs/pipeline.md")
    return run


STAGES = {
    "pull": stage_pull,
    "transcripts": stage_transcripts,
    "events": stage_events,              # linking + universe + Day 0
    "targets": stage_targets,             # CARs + post-call volatility
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
    parser.add_argument("--pass", dest="pass_", choices=["parse", "select"], default="parse",
                        help="transcripts: parse (every version, before la events) or select "
                             "(one version per call, after la events)")
    parser.add_argument("--groups", nargs="+", default=["core", "transcripts"],
                        help="pull: query groups (core, transcripts, optional)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.paths.ensure()
    STAGES[args.stage](cfg, args)


if __name__ == "__main__":
    main()
