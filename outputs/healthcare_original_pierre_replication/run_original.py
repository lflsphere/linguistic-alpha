"""Run the unmodified Pierre Healthcare extractor and model with local input aliases."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

IDENTITY = ["keydevid", "gvkey", "permno", "call_date", "call_time_utc"]
TARGET = "car_2_20"
CONTROLS = (
    "sue", "surprise_pct", "vol_pre", "mom", "bm", "size", "dispersion",
    "avg_answer_words", "words_prepared", "words_questions", "words_answers",
    "n_questions", "n_answers", "n_analysts", "prior_calls_with_target",
)
LM = (
    "tone_prepared", "tone_answers", "hedging_prepared", "hedging_answers",
    "uncertainty_answers", "tone_gap", "tone_prepared_chg", "tone_answers_chg",
    "hedging_answers_chg",
)
RAW = {
    "sp500_index_daily.parquet": ("crsp_dsi.parquet", ["date", "sprtrn"]),
    "link_ccm.parquet": ("crsp_ccmxpf_lnkhist.parquet", ["gvkey", "permno", "linkdt", "linkenddt"]),
    "sp500_membership.parquet": ("crsp_msp500list.parquet", ["permno", "start", "ending"]),
    "crsp_delist.parquet": ("crsp_dsedelist.parquet", ["permno", "dlstdt", "dlret"]),
}
STOCK_COLUMNS = ["permno", "date", "ret", "prc", "shrout"]


def assert_unique(frame, keys, name):
    if not set(keys).issubset(frame.columns):
        raise ValueError(f"{name} is missing {sorted(set(keys) - set(frame.columns))}")
    if frame[keys].isna().any(axis=None) or frame.duplicated(keys).any():
        raise ValueError(f"{name} has duplicate or null keys: {keys}")


def gvkeys(series):
    return series.astype("string").str.zfill(6)


def company_snapshot(table, raw):
    """Use original CSV sector codes, then most recent local classifications for other firms."""
    supplied = table[["gvkey", "gind"]].drop_duplicates().copy()
    supplied["gsector"] = supplied.gind.str[:2]
    supplied = supplied[["gvkey", "gsector"]].drop_duplicates()
    assert_unique(supplied, ["gvkey"], "model export snapshot")
    companies = pd.read_parquet(raw / "comp_company.parquet", columns=["gvkey"])
    companies["gvkey"] = gvkeys(companies.gvkey)
    assert_unique(companies, ["gvkey"], "Compustat company IDs")
    fields = ["gvkey", "gsector", "indfrom", "indthru"]
    history = pd.read_parquet(raw / "comp_co_hgic.parquet", columns=fields)
    history["gvkey"] = gvkeys(history.gvkey)
    history["indfrom"] = pd.to_datetime(history.indfrom, errors="raise")
    history = history.dropna(subset=["gsector", "indfrom"]).drop_duplicates()
    most_recent = history.loc[history.indfrom.eq(history.groupby("gvkey").indfrom.transform("max"))]
    most_recent = most_recent[["gvkey", "gsector"]].drop_duplicates()
    assert_unique(most_recent, ["gvkey"], "most recent GICS snapshot fallback")
    result = companies.merge(supplied, how="left", on="gvkey", validate="one_to_one")
    result["snapshot_from_csv"] = result.gsector.notna()
    result["gsector"] = result.gsector.fillna(result.gvkey.map(most_recent.set_index("gvkey").gsector))
    missing_csv = supplied.loc[~supplied.gvkey.isin(result.gvkey)].assign(snapshot_from_csv=True)
    result = pd.concat([result, missing_csv], ignore_index=True)
    result["gsector"] = result.gsector.astype("string")
    if not table.sector.eq("Health Care").eq(table.gvkey.map(
        result.set_index("gvkey").gsector,
    ).eq("35")).all():
        raise AssertionError("The snapshot changes Healthcare classification of an exported call")
    return result


class InputAliases:
    """Virtual source paths are handled solely at the file-read boundary."""
    def __init__(self, csv, raw, virtual):
        self.virtual = virtual.resolve()
        self.raw = raw.resolve()
        self.read = pd.read_parquet
        self.glob = glob.glob
        dtype = {"gvkey": "string", "gind": "string", "keydevid": "int64",
                 "transcriptid": "int64", "permno": "int64"}
        self.table = pd.read_csv(csv, dtype=dtype, float_precision="round_trip", low_memory=False)
        names = {*CONTROLS, *LM, *IDENTITY, "sector", "gind", "transcriptid"}
        if names - set(self.table):
            raise ValueError(f"Linguistic input lacks {sorted(names - set(self.table))}")
        self.table["gvkey"] = gvkeys(self.table.gvkey)
        assert_unique(self.table, ["keydevid"], "CSV model events")
        assert_unique(self.table, ["transcriptid"], "CSV transcripts")
        self.company = company_snapshot(self.table, self.raw)
        self.annual = {}
        for source in sorted(self.raw.glob("crsp_dsf_[0-9][0-9][0-9][0-9].parquet")):
            self.annual[source.name.replace("crsp_dsf_", "crsp_daily_", 1)] = source
        if not self.annual:
            raise ValueError("CRSP DSF shards are missing")
        for source, _ in RAW.values():
            if not (self.raw / source).is_file():
                raise FileNotFoundError(self.raw / source)
        self.stock_checks = []
        self.calendar = self.parquet(self.virtual / "raw" / "sp500_index_daily.parquet")
        assert_unique(self.calendar, ["date"], "S&P 500 market sessions")
        if self.calendar.sprtrn.isna().any() or not self.calendar.date.is_monotonic_increasing:
            raise ValueError("Unexpected S&P price return/calendar schema")

    def parquet(self, path, *args, **kwargs):
        if not isinstance(path, (str, os.PathLike)):
            return self.read(path, *args, **kwargs)
        p = Path(path).resolve()
        if not p.is_relative_to(self.virtual):
            return self.read(path, *args, **kwargs)
        if p.name == "model_table.parquet":
            return self.table.copy()
        if p.name == "event_panel.parquet":
            return self.table[IDENTITY].copy()
        if p.name == "company.parquet":
            return self.company[["gvkey", "gsector"]].copy()
        if p.name in self.annual:
            frame = self.read(self.annual[p.name], columns=STOCK_COLUMNS)
            for col in ("ret", "prc", "shrout"):
                frame[col] = frame[col].astype("float64")
            if len(frame):
                assert_unique(frame, ["permno", "date"], p.name)
                frame["permno"] = frame.permno.astype("int64")
            else:
                frame = frame.astype({"permno": "int64", "date": "string"})
            self.stock_checks.append({
                "logical_name": p.name, "actual_name": self.annual[p.name].name, "rows": len(frame),
                "first_date": str(frame.date.min()), "last_date": str(frame.date.max()),
            })
            return frame
        if p.name in RAW:
            source, cols = RAW[p.name]
            frame = self.read(self.raw / source, columns=cols)
            if "gvkey" in frame:
                frame["gvkey"] = gvkeys(frame.gvkey)
            for c in ("sprtrn", "dlret"):
                if c in frame:
                    frame[c] = frame[c].astype("float64")
            return frame
        raise FileNotFoundError(f"Unexpected original input: {p}")

    def find(self, pattern, *args, **kwargs):
        p = Path(pattern)
        if p.parent.resolve() == self.virtual / "raw" and p.name == "crsp_daily_*.parquet":
            return [str(p.parent / name) for name in self.annual]
        return self.glob(pattern, *args, **kwargs)

    def describe(self):
        health = self.table.loc[self.table.sector.eq("Health Care")]
        company = self.company
        return {
            "input_rows": len(self.table), "healthcare_csv_rows": len(health),
            "healthcare_csv_issuers": health.gvkey.nunique(),
            "healthcare_text_columns": list(LM),
            "duplicate_csv_event_ids": int(self.table.keydevid.duplicated().sum()),
            "duplicate_csv_transcript_ids": int(self.table.transcriptid.duplicated().sum()),
            "current_company_rows": len(company),
            "current_company_rows_missing_gics": int(company.gsector.isna().sum()),
            "health_gvkeys_from_csv": int((company.gsector.eq("35") & company.snapshot_from_csv).sum()),
            "health_gvkeys_from_latest_gics_fallback": int(
                (company.gsector.eq("35") & ~company.snapshot_from_csv).sum()
            ),
            "calendar_rows": len(self.calendar), "market_column": "CRSP DSI SPRTRN",
            "first_market_date": str(self.calendar.date.min()), "last_market_date": str(self.calendar.date.max()),
            "annual_shards": self.stock_checks,
        }


def load(path):
    name = "pierre_" + path.stem
    info = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(info)
    sys.modules[name] = module
    info.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise AssertionError("Historical module resolved to an unexpected source")
    return module


def assert_provenance(original, output):
    manifest = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((original / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Historical source checksum mismatch: {name}")


def run_extract(original, output, csv, raw):
    sut = load(original / "sector_extract.py")
    expected = (original / "src/linguistic_alpha/events/timing.py").resolve()
    timing = sys.modules["linguistic_alpha.events.timing"]
    if Path(timing.__file__).resolve() != expected:
        raise AssertionError("Use the staged revision of the original timing helper")
    aliases = InputAliases(csv, raw, output / "virtual_earnings_sp500")
    sys.argv = [str(original / "sector_extract.py"), "--sector", "health", "--folder",
                str(aliases.virtual)]
    # On Windows expanduser() reads USERPROFILE. Only the isolated runner sees this override.
    with (patch.dict(os.environ, {"USERPROFILE": str(output.resolve())}),
          patch.object(pd, "read_parquet", aliases.parquet), patch.object(glob, "glob", aliases.find),
          (output / "extraction.log").open("x", encoding="utf-8") as log,
          redirect_stdout(log), redirect_stderr(log)):
        sut.main()
    with (output / "input_schema_audit.json").open("x", encoding="utf-8") as report:
        json.dump(aliases.describe(), report, indent=2)
        report.write("\n")
    for i in (1, 2):
        path = output / "earnings_health" / f"model_table_opt{i}.parquet"
        frame = pd.read_parquet(path)
        assert_unique(frame, ["keydevid"], f"native option {i}")
        if len(frame) != aliases.table.sector.eq("Health Care").sum() or not frame.sector.eq("Health Care").all():
            raise AssertionError("The original extraction changed Healthcare row identity")
        base = aliases.table.loc[aliases.table.sector.eq("Health Care")].set_index("keydevid")
        pd.testing.assert_frame_equal(frame.set_index("keydevid")[list(CONTROLS + LM)],
                                      base.loc[frame.keydevid, list(CONTROLS + LM)], check_exact=True)
        if frame[TARGET].notna().sum() == 0:
            raise ValueError(f"Option {i} extraction produced no labels. Read extraction.log.")
        print(f"Option {i}: {len(frame)} Healthcare events, {frame[TARGET].notna().sum()} observed CARs", flush=True)


def run_models(original, output):
    sut = load(original / "sector_model_b.py")
    folder = output / "earnings_health"
    opt = {f"model_table_opt{i}.parquet": i for i in (1, 2)}
    for table in opt:
        if not (folder / table).is_file():
            raise FileNotFoundError(folder / table)
    sys.argv = [str(original / "sector_model_b.py"), "--sector", "health", "--folder", str(folder)]
    evaluations, selected = {}, {}

    def capture(frame, event, result):
        if event != "return" or result is None:
            return
        if frame.f_code is sut.evaluate.__code__:
            evaluations[id(result)] = (result, np.asarray(frame.f_locals["y_pred"]).copy())
        if frame.f_code is sut.run_option.__code__:
            state = frame.f_locals
            b, b_star, a, base_features, routes = result
            selected[opt[state["opt_file"]]] = (
                state["train"].copy(), state["test"].copy(), state["a_best_name"],
                a, base_features, b, b_star, routes,
            )

    with (output.joinpath("model.log").open("x", encoding="utf-8") as log,
          redirect_stdout(log), redirect_stderr(log)):
        previous = sys.getprofile()
        try:
            sys.setprofile(capture)
            sut.main()
        finally:
            sys.setprofile(previous)

    if set(selected) != {1, 2}:
        raise AssertionError("Both options must finish and expose native forecasts")
    forecast_frames, rows, splits, strategy_rows = [], [], [], []
    for option, (train, test, algo, a, base_features, b, star, routes) in selected.items():
        a_spec = {"metrics": a, "algo": algo, "name": "baseline", "added_ling": [],
                  "features": base_features}
        for name, model in {"A": a_spec, "B": b, "B*": star}.items():
            metrics = model["metrics"]
            predictions = evaluations[id(metrics)][1]
            np.testing.assert_allclose(sut.evaluate(test[TARGET].values, predictions)["rank_ic"],
                                       metrics["rank_ic"], atol=1e-14, rtol=0)
            rows.append({
                "option": option, "benchmark": "sp500_price" if option == 1 else "healthcare_sector",
                "model": name, "algorithm": model["algo"], "strategy": model["name"],
                "features": ",".join(model["features"]), "lm_inputs_native": ",".join(model["added_ling"]),
                "n_train": len(train), "n_test": len(test), "test_securities": test.permno.nunique(),
                "train_first": train.call_date.min(), "train_last": train.call_date.max(),
                "test_first": test.call_date.min(), "test_last": test.call_date.max(),
                **{f"native_{k}": v for k, v in metrics.items()},
                "incremental_rank_ic_vs_a": metrics["rank_ic"] - a["rank_ic"],
                "target_scale": "training CAR midranks (not continuous CAR)",
                "test_selection": True,
            })
            forecast_frames.append(
                test[["keydevid", "transcriptid", "permno", "gvkey", "call_date", "call_time_utc", "day0", TARGET]]
                .assign(option=option, model=name, prediction=predictions),
            )
        for route in routes:
            strategy_rows.append({
                "option": option, "strategy": route["name"], "algorithm": route["algo"],
                "lm_inputs_native": ",".join(route["added_ling"]), "test_selected": True,
                "rank_ic": route["metrics"]["rank_ic"], "selected_as_b": route is b,
                "q5_q1": route["metrics"]["quintile_spread"],
            })
        for sample, frame in (("train", train), ("test", test)):
            splits.append(frame[["keydevid", "transcriptid", "permno", "call_date", "call_time_utc", "day0"]]
                          .assign(option=option, sample=sample))
    pd.DataFrame(rows).to_csv(output / "native_model_metrics.csv", index=False)
    pd.DataFrame(strategy_rows).to_csv(output / "native_strategy_scores.csv", index=False)
    pd.concat(forecast_frames, ignore_index=True).to_parquet(
        output / "original_test_predictions.parquet", index=False,
    )
    pd.concat(splits, ignore_index=True).to_csv(output / "original_model_splits.csv", index=False)
    print("Captured native forecasts and verified their reported Rank ICs for both options.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "model"))
    parser.add_argument("--original", type=Path, default=Path("experiments/legacy_pipeline"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--csv", type=Path, default=Path("model_table_with_linguistic.csv"))
    args = parser.parse_args()
    original = args.original.resolve()
    assert_provenance(original, args.out)
    sys.path[:0] = [str(original), str(original / "src")]
    if args.phase == "extract":
        run_extract(original, args.out, args.csv, args.raw)
    else:
        run_models(original, args.out)

