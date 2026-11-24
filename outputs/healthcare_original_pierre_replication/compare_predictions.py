"""Compare frozen Pierre and corrected Healthcare forecasts on reconciled labels."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sector import issuer_rank_intervals
from linguistic_alpha.events.timing import call_timestamp_utc
from sector_model_b import corr, paired_intervals, rank_ic

TARGET = "car_2_20"
KEY = ["transcriptid", "permno", "call_ts_utc"]
ORIG_MODELS, CORRECTED_MODELS = ("A", "B", "B*"), ("A0", "A1", "S1", "B", "B*")
CUTOFF = pd.Timestamp("2024-01-01")


def keyed(frame):
    out = frame.copy()
    if "call_ts_utc" not in out:
        out["call_ts_utc"] = call_timestamp_utc(out.call_date, out.call_time_utc)
    for name in ("transcriptid", "permno"):
        out[name] = out[name].astype("int64")
    if out[KEY].isna().any(axis=None):
        raise ValueError("An event lacks the transcript, security or UTC clock")
    return out


def unique(frame, keys, label):
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} has duplicate {keys}")


def score(frame, models, base_name, draws):
    """Evaluate frozen held-out forecasts with paired time and issuer intervals."""
    y = frame["target"]
    metrics = []
    a = rank_ic(y, frame[base_name])
    for model in models:
        rank = rank_ic(y, frame[model])
        cuts = pd.qcut(frame[model], 5, labels=False, duplicates="drop")
        groups = y.groupby(cuts).mean()
        metrics.append({
            "model": model, "baseline_model": base_name, "rank_ic": rank,
            "pearson_ic": corr(y, frame[model]), "incremental_rank_ic": rank - a,
            "q5_q1_if_five_bins": groups.iloc[-1] - groups.iloc[0] if len(groups) == 5 else np.nan,
            "n_test": len(frame), "test_issuers": frame.gvkey.nunique(),
            "test_first_utc": frame.call_ts_utc.min(), "test_last_utc": frame.call_ts_utc.max(),
        })

    base = frame[["transcriptid", "call_ts_utc", "gvkey", "target"]].rename(columns={
        "transcriptid": "keydevid", "call_ts_utc": "signal_utc", "target": "actual",
    })
    stacked = pd.concat([base.assign(model="A1" if model == base_name else model,
                                    prediction=frame[model].to_numpy(), training_mean=0.0,
                                    sample="test")
                         for model in models], ignore_index=True)
    intervals = paired_intervals(stacked, draws)
    intervals = intervals.loc[intervals.metric.eq("rank_ic") & intervals.comparison.eq("incremental_vs_A1"),
                              ["model", "low_95", "high_95", "valid_replicates"]].rename(columns={
        "low_95": "time_ci_low", "high_95": "time_ci_high", "valid_replicates": "time_defined_draws",
    })
    intervals = intervals.merge(issuer_rank_intervals(stacked, draws), on="model", validate="one_to_one")
    intervals["model"] = intervals.model.replace({"A1": base_name})
    return pd.DataFrame(metrics).merge(intervals, on="model", validate="one_to_one")


def wide_original(long, option):
    part = keyed(long.loc[long.option.eq(option)])
    unique(part, [*KEY, "model"], "native forecasts")
    idx = [*KEY, "gvkey", TARGET, "day0"]
    wide = part.pivot(index=idx, columns="model", values="prediction").reset_index()
    wide = wide.rename(columns={TARGET: "original_car", "day0": "original_day0"})
    if wide[list(ORIG_MODELS)].isna().any(axis=None):
        raise ValueError("Native models use different holdout events")
    return wide


def wide_corrected(previous, option, panel):
    tag = "market" if option == 1 else "sector"
    folder = previous / f"{tag}_transcript"
    info = panel[["keydevid", *KEY, "gvkey", "day0", TARGET]]
    part = pd.read_parquet(folder / "predictions.parquet")
    part = part.loc[part["sample"].eq("test") & part.model.isin(CORRECTED_MODELS)]
    part = part.drop(columns="permno").merge(info, on="keydevid", validate="many_to_one")
    np.testing.assert_allclose(part.actual, part[TARGET], atol=1e-12, rtol=0)
    unique(part, [*KEY, "model"], "corrected forecasts")
    idx = [*KEY, "gvkey", TARGET, "day0"]
    wide = part.pivot(index=idx, columns="model", values="prediction").reset_index()
    wide = wide.rename(columns={TARGET: "corrected_car", "day0": "corrected_day0"})
    if wide[list(CORRECTED_MODELS)].isna().any(axis=None):
        raise ValueError("Corrected models use different holdout events")
    return wide


def frozen_compare(original, corrected, option, draws):
    fields = dict(zip(ORIG_MODELS, ["orig_A", "orig_B", "orig_B*"]))
    common = original.rename(columns=fields).merge(corrected, on=KEY, suffixes=("_orig", "_corr"),
                                                    validate="one_to_one")
    if not common.gvkey_orig.astype(int).eq(common.gvkey_corr.astype(int)).all():
        raise ValueError("Matched events have conflicting issuer IDs")
    common["gvkey"] = common.gvkey_corr
    outputs = []
    for method, cols, base in (
        ("original", tuple(fields.values()), "orig_A"),
        ("corrected", CORRECTED_MODELS, "A1"),
    ):
        for basis in ("original_car", "corrected_car"):
            measured = score(common.assign(target=common[basis]), cols, base, draws)
            measured[["model", "baseline_model"]] = measured[["model", "baseline_model"]].replace(
                {v: k for k, v in fields.items()}
            )
            outputs.append(measured.assign(option=option, forecast_origin=method, label_basis=basis))
    return pd.concat(outputs, ignore_index=True), common


def target_diagnostics(common, option, output, previous, raw):
    dsi = pd.read_parquet(raw / "crsp_dsi.parquet", columns=["date", "sprtrn", "vwretd"])
    dsi = dsi.set_index(pd.to_datetime(dsi.date)).sort_index()
    if option == 1:
        delta = dsi.sprtrn - dsi.vwretd
    else:
        sect = pd.read_parquet(output / "earnings_health" / "health_sector_daily.parquet")
        p = pd.read_csv(previous / "extraction" / "sector_benchmark_daily.csv", parse_dates=["date"])
        delta = sect.set_index("date").sector_ret.reindex(dsi.index) - p.set_index("date").vwretd.reindex(dsi.index)
    rows = common[[*KEY, "original_day0", "corrected_day0", "original_car", "corrected_car"]].copy()
    anchors = dsi.index.get_indexer(rows.corrected_day0)
    offsets = []
    for index, same in zip(anchors, rows.original_day0.eq(rows.corrected_day0)):
        window = delta.iloc[index + 2:index + 21]
        offsets.append(window.sum() if same and index >= 0 and len(window) == window.count() == 19
                       else np.nan)
    rows["benchmark_only_delta"] = offsets
    rows["label_delta_corrected_minus_original"] = rows.corrected_car - rows.original_car
    rows["delta_remaining_after_benchmark"] = rows.label_delta_corrected_minus_original - rows.benchmark_only_delta
    valid = rows.delta_remaining_after_benchmark.dropna()
    diagnostics = {
        "option": option, "n_shared": len(rows), "day0_differences": int(rows.original_day0.ne(rows.corrected_day0).sum()),
        "car_spearman": rank_ic(rows.original_car, rows.corrected_car),
        "car_pearson": corr(rows.original_car, rows.corrected_car),
        "mean_car_delta": rows.label_delta_corrected_minus_original.mean(),
        "rmse_car_delta": np.sqrt(np.square(rows.label_delta_corrected_minus_original).mean()),
        "n_benchmark_only_check": len(valid), "n_benchmark_only_equal_within_1e_7": int(valid.abs().le(1e-7).sum()),
        "max_abs_remaining": valid.abs().max(),
    }
    return rows.assign(option=option), diagnostics


def attrition(original, corrected, original_test, corrected_test, common, option, all_export, calendar):
    masks = [
        ("01_all_original_csv_events", all_export),
        ("02_original_healthcare_label", len(original)),
        ("03_original_day0_in_calendar", int(original.day0.notna().sum())),
        ("04_original_complete_20_session_calendar", int((
            original.day0.notna() & (calendar.get_indexer(original.day0) + 20 < len(calendar))
        ).sum())),
        ("05_original_nonmissing_car", int(original[TARGET].notna().sum())),
        ("06_original_development", int((original[TARGET].notna() & original.call_date.lt(CUTOFF)).sum())),
        ("07_original_holdout", len(original_test)),
        ("08_corrected_historical_healthcare_rows", len(corrected)),
    ]
    eligible = corrected.verified_event.copy()
    for label, test in (
        ("09_corrected_identifiers_and_membership", eligible.copy()),
        ("10_corrected_verified_financial_archive", corrected.archive_verified),
        ("11_corrected_usable_call_time", corrected.immediate_available),
        ("12_corrected_nonmissing_car", corrected[TARGET].notna()),
        ("13_corrected_transcript_before_target", corrected.transcript_available_before_target),
    ):
        eligible &= test
        masks.append((label, int(eligible.sum())))
    masks.extend([
        ("14_corrected_transcript_holdout", len(corrected_test)),
        ("15_shared_test_forecast_keys", len(common)),
        ("16_native_test_keys_without_corrected_test_forecast", len(original_test) - len(common)),
        ("17_corrected_test_keys_without_native_test_forecast", len(corrected_test) - len(common)),
    ])
    return pd.DataFrame(masks, columns=["step", "events"]).assign(option=option)


def correlations(original, common, option):
    rows = []
    for a, b in combinations(ORIG_MODELS, 2):
        rows.append({"option": option, "comparison": "original_native", "left": a, "right": b,
                     "n": len(original), "spearman": rank_ic(original[a], original[b]),
                     "pearson": corr(original[a], original[b])})
    for a, b in (("orig_A", "A0"), ("orig_A", "A1"), ("orig_B", "B"), ("orig_B*", "B*")):
        rows.append({"option": option, "comparison": "original_vs_corrected_matched", "left": a, "right": b,
                     "n": len(common), "spearman": rank_ic(common[a], common[b]),
                     "pearson": corr(common[a], common[b])})
    return pd.DataFrame(rows)


def native_corrected_matrices(previous, option):
    rows = []
    for cohort in ("transcript", "immediate"):
        path = previous / f"{'market' if option == 1 else 'sector'}_{cohort}"
        frozen = json.loads((path / "frozen_specifications.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(path / "metrics.csv")
        test = metrics.loc[metrics["sample"].eq("test") & metrics.period.eq("pooled") &
                           metrics.model.isin(CORRECTED_MODELS)]
        test = test.assign(
            option=option, forecast_origin=f"corrected_{cohort}",
            label_basis="corrected_car", n_test=test.n,
            cv_rank_ic=test.model.map(frozen["selected_model_cv_rank_ic"]),
            algorithm=test.model.map({key: val["algorithm"] for key, val in frozen["specifications"].items()}),
        )
        rows.append(test[["option", "forecast_origin", "label_basis", "model", "algorithm", "n_train",
                          "n_test", "cv_rank_ic", "rank_ic", "incremental_rank_ic", "pearson_ic",
                          "oos_r2", "rmse", "mae", "q5_q1"]])
    return pd.concat(rows, ignore_index=True)


def attribution(original, matched, option):
    rows = []
    for model in ("B", "B*"):
        native = original.loc[original.model.eq(model)].iloc[0]
        def find(origin, label):
            return matched.loc[matched.forecast_origin.eq(origin) & matched.label_basis.eq(label) &
                               matched.model.eq(model)].iloc[0]
        old_common = find("original", "original_car")
        new_target = find("original", "corrected_car")
        new_forecast = find("corrected", "corrected_car")
        for metric in ("rank_ic", "incremental_rank_ic"):
            chain = [native[metric], old_common[metric], new_target[metric], new_forecast[metric]]
            for step, left, right in zip(
                ("sample_only", "target_only", "remaining_frozen_pipeline_difference"), chain, chain[1:],
            ):
                rows.append({"option": option, "model": model, "metric": metric, "step": step,
                             "before": left, "after": right, "change": right - left})
    return pd.DataFrame(rows)


def report(output, previous, raw, draws):
    native_metrics = pd.read_csv(output / "native_model_metrics.csv")
    prepared = pd.read_parquet(output / "original_test_predictions.parquet")
    schema = json.loads((output / "input_schema_audit.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.read_parquet(raw / "crsp_dsi.parquet", columns=["date"]).date)
    original_rows, matched_rows, events, attr, correl, targets, checks, native, decomposed = ([] for _ in range(9))
    for option in (1, 2):
        filename = f"model_table_opt{option}.parquet"
        old = keyed(pd.read_parquet(output / "earnings_health" / filename))
        corrected = keyed(pd.read_parquet(previous / "extraction" / filename))
        for label, frame in (("original events", old), ("corrected events", corrected)):
            unique(frame, KEY, label)
            unique(frame, ["transcriptid"], label)
        wide = wide_original(prepared, option)
        cv = wide_corrected(previous, option, corrected)
        matching, common = frozen_compare(wide, cv, option, draws)
        scored = score(wide.assign(target=wide.original_car), ORIG_MODELS, "A", draws)
        joined = native_metrics.loc[native_metrics.option.eq(option)].merge(
            scored.drop(columns=["n_test"]), on="model", validate="one_to_one",
        )
        np.testing.assert_allclose(joined.native_rank_ic, joined.rank_ic, atol=1e-13, rtol=0)
        original_rows.append(joined)
        matched_rows.append(matching)
        decomposed.append(attribution(scored, matching, option))
        audit, summary = target_diagnostics(common, option, output, previous, raw)
        events.append(old[[*KEY, "keydevid", TARGET, "day0"]].merge(
            corrected[[*KEY, "keydevid", TARGET, "day0", "verified_event",
                       "archive_verified", "transcript_available_before_target"]],
            on=KEY, how="outer", suffixes=("_original", "_corrected"), indicator="link_status",
            validate="one_to_one",
        ).merge(wide[KEY].assign(original_test=True), on=KEY, how="left", validate="one_to_one")
          .merge(cv[KEY].assign(corrected_test=True), on=KEY, how="left", validate="one_to_one")
          .assign(option=option))
        attr.append(attrition(old, corrected, wide, cv, common, option, schema["input_rows"], calendar))
        correl.append(correlations(wide, common, option))
        targets.append(audit)
        checks.append(summary)
        native.append(native_corrected_matrices(previous, option))
    for name, frames in (
        ("original_results", original_rows), ("matched_event_comparison", matched_rows),
        ("event_matching", events), ("sample_attrition", attr), ("prediction_correlations", correl),
        ("target_reconciliation_events", targets), ("native_corrected_comparison", native),
        ("attribution", decomposed),
    ):
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "target_reconciliation_summary.csv", index=False)
    print("Wrote native and frozen matched diagnostics for both Healthcare benchmarks.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, default=Path("outputs/healthcare_sector_model_corrected"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    report(args.out, args.corrected, args.raw, args.bootstrap)

