"""Release checks for the held-out scores, matched cohorts and static report inputs."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_unified import FULL, MATCHED, NATIVE, FOCAL_PAIRS, INCREMENTS
from fit_finbert import BASELINE, CONTEXT, FEATURES, HERE, MODELS, load_csv, snapshot
from extract_finbert import json_write, stable_digest


def verify_results(directory: Path):
    integrity = json.loads((directory / "integrity.json").read_text(encoding="utf-8"))
    predictions_file = directory / "five_model_predictions_local.parquet"
    assert integrity["preserved_a_b_outputs"] == snapshot(), "A/B/B* source digest changed"
    assert integrity["predictions_sha256"] == stable_digest(predictions_file), "Prediction digest changed"
    case = pd.read_parquet(predictions_file).sort_values([*CONTEXT, "transcriptid"]).reset_index(drop=True)
    actual = load_csv(BASELINE, "matched_predictions_local.csv").sort_values(
        [*CONTEXT, "transcriptid"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(case[[*CONTEXT, "transcriptid", "actual_car", "A", "B", "B*"]],
                                  actual[[*CONTEXT, "transcriptid", "actual_car", "A", "B", "B*"]],
                                  check_exact=True)

    perf = load_csv(directory, "annual_model_performance.csv")
    incs = load_csv(directory, "annual_increments.csv")
    assert len(perf) == 24 * 5 * 3 and len(incs) == 24 * 5 * 3
    model_checks = 0
    for context, rows in case.groupby(CONTEXT, sort=False):
        assert rows[["actual_car", *MODELS]].notna().all().all()
        assert rows.transcriptid.is_unique
        year, benchmark, horizon, window = context
        different = case.loc[case.year.eq(year) & case.benchmark.eq(benchmark) &
                             case.window.eq(window) & case.horizon.ne(horizon)]
        masks = [(FULL, np.ones(len(rows), dtype=bool)),
                 (MATCHED, rows.transcriptid.isin(different.transcriptid)),
                 (NATIVE, rows.finbert_complete)]
        for name, mask in masks:
            select = rows.loc[mask]
            key = (*context, name)
            result = perf.set_index([*CONTEXT, "sample"]).loc[[key]].set_index("model")
            estimates = []
            for m in MODELS:
                if len(select) < 3 or select[m].nunique() < 2 or select.actual_car.nunique() < 2:
                    actual_ic, actual_pearson = np.nan, np.nan
                else:
                    actual_ic = stats.spearmanr(select.actual_car, select[m]).statistic
                    actual_pearson = stats.pearsonr(select.actual_car, select[m]).statistic
                metric = result.loc[m]
                assert metric.events == len(select) and metric.issuers == select.gvkey.nunique()
                np.testing.assert_allclose([metric.rank_ic, metric.pearson], [actual_ic, actual_pearson],
                                           rtol=0, atol=1e-12, equal_nan=True)
                estimates.append(actual_ic)
                model_checks += 1
            observed = incs.set_index([*CONTEXT, "sample"]).loc[[key]].set_index("contrast")
            calc = dict(zip(MODELS, estimates))
            np.testing.assert_allclose(observed.loc[list(INCREMENTS), "delta_rank_ic"],
                                       [calc[a] - calc[b] for a, b in FOCAL_PAIRS],
                                       rtol=0, atol=1e-12, equal_nan=True)

    specs = load_csv(directory, "finbert_selected_features.csv")
    paths = load_csv(directory, "finbert_forward_paths.csv")
    importance = load_csv(directory, "finbert_feature_importance.csv")
    assert len(specs) == 48
    for key, group in specs.groupby(CONTEXT, sort=False):
        selected_path = paths.set_index(CONTEXT).loc[[key]]
        c, star = group.set_index("model").loc["C"], group.set_index("model").loc["C*"]
        accepted = selected_path.loc[selected_path.accepted].sort_values("step")
        built = "|".join(accepted.candidate)
        assert (c.selection_order if isinstance(c.selection_order, str) else "") == built
        assert int(c.feature_count) == len(accepted)
        assert (accepted.gain > .001).all()
        assert star.features.split("|") == list(FEATURES)
        fitted = importance.set_index(CONTEXT).loc[[key]]
        assert set(fitted.loc[(fitted.model == "C*") & (fitted.block == "FinBERT"), "feature"]) == set(FEATURES)
    results = dict(verified_model_rows=model_checks, verified_settings=24,
                   verified_selection_rows=len(specs), preserved_archive=True)
    json_write(directory / "aggregate_validation.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    verify_results(HERE)
