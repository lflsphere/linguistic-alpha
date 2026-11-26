"""Fit Healthcare C/C* from the validated local FinBERT store, preserving A/B/B*."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

from extract_finbert import (DEFAULT_MODEL_TABLE, FEATURES, HERE, STORE, events_from, json_write,
                             parquet_write, stable_digest, verify)

BASELINE = HERE.parent / "healthcare_window_horizon_comparison"
sys.path.append(str(BASELINE))
import run_comparison as baseline

CONTEXT = ["year", "benchmark", "horizon", "window"]
CONTROLS = baseline.CONTROLS
FIELDS = CONTROLS + FEATURES
MODELS = ("A", "B", "B*", "C", "C*")
BASE_FILES = (
    "matched_predictions_local.csv", "event_timing_local.csv", "training_cohorts.csv", "cv_folds.csv",
    "selected_features.csv", "frozen_research_spec.json", "data_provenance.json", "annual_model_performance.csv",
    "run_comparison.py", "research_metrics.py", "factor_ic.csv", "factor_stability.csv", "factor_inference.csv",
    "factor_quintiles.csv", "feature_importance.csv", "feature_selection_paths.csv", "estimator_cv.csv",
    "feature_missingness.csv", "correlations.csv",
)


def load_csv(directory, name):
    return pd.read_csv(directory / name, float_precision="round_trip", keep_default_na=True)


def snapshot():
    return {file: stable_digest(BASELINE / file) for file in BASE_FILES}


def freeze(out, checksums):
    protocol = json.loads((BASELINE / "frozen_research_spec.json").read_text(encoding="utf-8"))
    settings = {
        "existing_protocol": protocol,
        "existing_outputs_sha256": checksums,
        "finbert_feature_manifest_sha256": stable_digest(STORE / "feature_manifest.json"),
        "finbert_features": list(FEATURES),
        "c_selection": "Greedy forward only. At each step, maximize the mean of the three inner fold "
        "Spearman ICs across candidate features and the original estimator grid. Require >0.001 gain each "
        "step. Ties within 1e-12 follow declared feature and estimator order. Empty selection copies stored A.",
        "c_star_selection": "All thirteen features, estimator selected on the same inner chronological folds.",
        "cohort_policy": "Exact existing test cohorts. Missing FinBERT predictors use train-fitted imputation "
        "and are included. Record native, partial and complete feature coverage separately.",
    }
    json_write(out / "frozen_unified_spec.json", settings)


def matrices(native, train, predict):
    train_fields = train.loc[:, list(FIELDS)].copy()
    pred_fields = predict.loc[:, list(FIELDS)].copy()
    empty = train_fields.columns[train_fields.isna().all()]
    train_fields.loc[:, empty] = 0.
    pred_fields.loc[:, empty] = 0.
    x, z, scaler, fitted = native.preprocess(train_fields, pred_fields, list(FIELDS))
    baseline.need(np.isfinite(x).all() and np.isfinite(z).all(), "Nonfinite model input")
    return x, z, scaler, fitted, empty


def chronological_folds(native, train, h, saved, refit):
    folds = []
    maturity = f"mature_{h}"
    target = f"car_2_{h}"
    for row in saved.sort_values("validation_year").itertuples(index=False):
        validation = train.loc[train.year.eq(row.validation_year)]
        cutoff = validation.day1_close_bound.min()
        history = train.loc[train.year.lt(row.validation_year)]
        eligible = history.loc[history[maturity].lt(cutoff)]
        baseline.need(len(eligible) == row.train_events and len(validation) == row.validation_events
                      and eligible.gvkey.nunique() == row.train_issuers
                      and validation.gvkey.nunique() == row.validation_issuers
                      and len(history) - len(eligible) == row.purged
                      and cutoff == pd.Timestamp(row.inner_cutoff)
                      and eligible[maturity].max() == pd.Timestamp(row.max_fit_maturity)
                      and validation[maturity].max() == pd.Timestamp(row.max_validation_maturity)
                      and validation[maturity].lt(refit).all(), "Inner-fold audit drift")
        x, z, *_ = matrices(native, eligible, validation)
        folds.append(baseline.Fold(x, z, stats.rankdata(eligible[target]),
                                   stats.rankdata(validation[target])))
    baseline.need(len(folds) == 3, "Expected three chronological folds")
    return folds


class ForwardSearch:
    def __init__(self, folds, context):
        self.folds = folds
        self.context = context
        self.scores = {}
        self.grid = []
        self.paths = []

    def evaluate(self, subset):
        ordered = tuple(name for name in FEATURES if name in subset)
        if ordered in self.scores:
            return self.scores[ordered]
        ids = [FIELDS.index(name) for name in CONTROLS + ordered]
        grid = []
        for algorithm in baseline.ALGORITHMS:
            scores = []
            for fold in self.folds:
                fit = baseline.estimator(algorithm).fit(fold.x[:, ids], fold.ranks)
                ranks = stats.rankdata(fit.predict(fold.predict_x[:, ids]))
                ic = 0. if np.ptp(ranks) == 0 else np.corrcoef(ranks, fold.val_ranks)[0, 1]
                scores.append(float(ic))
            result = dict(**self.context, features="|".join(ordered), estimator=algorithm,
                          mean_rank_ic=float(np.mean(scores)), fold_1=scores[0],
                          fold_2=scores[1], fold_3=scores[2])
            grid.append(result)
        self.grid.extend(grid)
        self.scores[ordered] = baseline.choose_best(grid, lambda row: row["mean_rank_ic"])
        return self.scores[ordered]

    def select(self, archived):
        controls = self.evaluate(())
        baseline.need(controls["estimator"] == archived.estimator
                      and abs(controls["mean_rank_ic"] - archived.inner_rank_ic) < 1e-12,
                      "FinBERT baseline selection deviated from stored Model A")
        ordered = []
        current = controls
        for step in range(1, len(FEATURES) + 1):
            candidates = [(name, self.evaluate([*ordered, name])) for name in FEATURES
                          if name not in ordered]
            feature, chosen = baseline.choose_best(candidates, lambda row: row[1]["mean_rank_ic"])
            gain = chosen["mean_rank_ic"] - current["mean_rank_ic"]
            accept = gain > baseline.MIN_GAIN + 1e-12
            for field, result in candidates:
                self.paths.append(dict(**self.context, model="C", step=step, candidate=field,
                                       prior_selection="|".join(ordered), estimator=result["estimator"],
                                       inner_rank_ic=result["mean_rank_ic"],
                                       gain=result["mean_rank_ic"] - current["mean_rank_ic"],
                                       accepted=accept and field == feature))
            if not accept:
                break
            ordered.append(feature)
            current = chosen
        return {
            "C": (tuple(ordered), current, "Forward" if ordered else "Controls"),
            "C*": (FEATURES, self.evaluate(FEATURES), "All FinBERT13"),
        }


def select_case(frame, context, columns=CONTEXT):
    for key in columns:
        frame = frame.loc[frame[key].eq(context[key])]
    return frame


def model_log(model, context, recipe, refit, n, test_count):
    selected, result, route = recipe
    algorithm = result["estimator"]
    family, alpha = baseline.ALGORITHMS[algorithm]
    return dict(**context, model=model, route=route, features="|".join(
                field for field in FEATURES if field in selected), selection_order="|".join(selected),
                feature_count=len(selected), estimator=algorithm, estimator_family=family, alpha=alpha,
                l1_ratio=.7 if family == "elastic" else None,
                max_iter=5000 if family == "elastic" else None, inner_rank_ic=result["mean_rank_ic"],
                training_count=n, refit_cutoff=refit, test_count=test_count, validation_folds=3)


def record_coefficients(context, model, sample_size, fit, selected, preprocessing):
    scaler, fitted, empty, log_transforms = preprocessing
    total = np.abs(fit.coef_).sum()
    rows = []
    parameters = []
    for field, coefficient in zip(selected, fit.coef_):
        index = FIELDS.index(field)
        low, high, median = fitted[field]
        rows.append(dict(**context, model=model, feature=field,
                         block="FinBERT" if field in FEATURES else "Control",
                         standardized_coefficient=coefficient, scaled_by_train_n=coefficient / sample_size,
                         absolute_share=abs(coefficient) / total if total else 0.))
        parameters.append(dict(**context, model=model, feature=field, coefficient=coefficient,
                               intercept=float(fit.intercept_), log1p=field in log_transforms,
                               all_missing_in_training=field in empty, winsor_lo=low, winsor_hi=high,
                               imputation_median=median, scaler_center=scaler.mean_[index],
                               scaler_scale=scaler.scale_[index]))
    return rows, parameters


def run(out=HERE):
    checksums = snapshot()
    features = verify(STORE, require_complete=False)
    manifest = json.loads((STORE / "feature_manifest.json").read_text(encoding="utf-8"))
    key = f"{stable_digest(DEFAULT_MODEL_TABLE)}:Health Care"
    contract = manifest.get("validated_cohorts", {}).get(key)
    if not contract or not contract["complete"]:
        raise ValueError("FinBERT coverage has not been validated for the complete Healthcare universe")
    coverage = pd.DataFrame(contract["annual"])
    universe = events_from(DEFAULT_MODEL_TABLE, ["Health Care"])
    if coverage.eligible_universe.sum() != len(universe) or not coverage.validated_cached.equals(
            coverage.text_accessible):
        raise ValueError("Healthcare extraction ledger is incomplete")
    from validate_feature_store import check as reconcile_feature_store
    reconcile_feature_store(STORE, out)
    freeze(out, checksums)
    json_write(out / "finbert_source_snapshot.json", {
        "coverage": contract["annual"], "definitions": manifest["definitions"], "revision": manifest["revision"],
        "sentence_rows": int(coverage.prepared_sentences.sum() + coverage.answer_sentences.sum()),
        "event_universe_sha256": contract["event_universe_sha256"],
        "runtime_validation": json.loads((HERE / "runtime_validation.json").read_text(encoding="utf-8")),
    })
    native, frames, _ = baseline.inputs(out)
    scores = load_csv(BASELINE, "matched_predictions_local.csv")
    stored_training = load_csv(BASELINE, "training_cohorts.csv")
    stored_folds = load_csv(BASELINE, "cv_folds.csv")
    stored_selected = load_csv(BASELINE, "selected_features.csv")
    features = features.loc[features.transcriptid.isin(universe.transcriptid)]
    counts = Counter()
    selections, grids, paths, importance, parameters, outputs, coverage_rows = [], [], [], [], [], [], []

    for option, source in frames.items():
        source = source.merge(features[["transcriptid", *FEATURES]], on="transcriptid",
                              validate="one_to_one", how="left")
        source["finbert_available"] = source.transcriptid.isin(features.transcriptid)
        frames[option] = source
        for saved in stored_training.loc[stored_training.benchmark.eq(option)].itertuples(index=False):
            year, horizon, refit = int(saved.year), int(saved.horizon), pd.Timestamp(saved.refit_cutoff)
            context = dict(year=year, benchmark=option, horizon=horizon, window=saved.window)
            target = f"car_2_{horizon}"
            train, purged = baseline.outer_train(source, year, horizon, saved.train_start, refit)
            test = source.loc[source.year.eq(year) & source[f"eligible_{horizon}"]]
            historic = select_case(scores, context).sort_values("transcriptid").reset_index(drop=True)
            test = test.sort_values("transcriptid").reset_index(drop=True)
            baseline.need(len(train) == saved.train_events and train.gvkey.nunique() == saved.train_issuers
                          and purged == saved.purged_at_refit
                          and train[f"mature_{horizon}"].lt(refit).all()
                          and train[f"mature_{horizon}"].max() == pd.Timestamp(saved.max_training_maturity),
                          "Outer training audit drift")
            baseline.need(np.array_equal(test.transcriptid.to_numpy(), historic.transcriptid.to_numpy()),
                          f"Test transcript membership mismatch: {context}")
            baseline.need(test.gvkey.eq(historic.gvkey.astype("string").str.zfill(6)).all(),
                          f"Issuer identity mismatch: {context}")
            baseline.need(np.array_equal(test[target].to_numpy(float), historic.actual_car.to_numpy(float)),
                          f"Realized CAR mismatch: {context}")
            baseline.need(test.signal.eq(pd.to_datetime(historic.signal, utc=True)).all(),
                          f"Prediction timestamp mismatch: {context}")
            baseline.need(set(test.transcriptid).isdisjoint(train.transcriptid), "Overlapping outer cohorts")
            archived = select_case(stored_selected, context).loc[lambda d: d.model.eq("A")]
            baseline.need(len(archived) == 1, "Missing baseline model specification")
            archived = next(archived.itertuples(index=False))
            search = ForwardSearch(chronological_folds(native, train, horizon, select_case(
                stored_folds, context), refit), context)
            recipes = search.select(archived)
            grids.extend(search.grid)
            paths.extend(search.paths)
            x, z, scaler, fitted, empty = matrices(native, train, test.loc[:, list(FIELDS)])
            ranks = stats.rankdata(train[target])
            a_fit = baseline.estimator(archived.estimator).fit(x[:, :len(CONTROLS)], ranks)
            np.testing.assert_allclose(a_fit.predict(z[:, :len(CONTROLS)]), historic.A,
                                       rtol=0, atol=1e-10, err_msg=str(context))
            common = historic.assign(gvkey=test.gvkey, call_date=test.call_date,
                                     finbert_available=test.finbert_available,
                                     finbert_complete=test.loc[:, list(FEATURES)].notna().all(axis=1))
            for model, recipe in recipes.items():
                chosen, result, _ = recipe
                selected = CONTROLS + tuple(field for field in FEATURES if field in chosen)
                ids = [FIELDS.index(field) for field in selected]
                if model == "C" and not chosen:
                    fit, common[model] = a_fit, historic.A
                else:
                    fit = baseline.estimator(result["estimator"]).fit(x[:, ids], ranks)
                    common[model] = fit.predict(z[:, ids])
                selections.append(model_log(model, context, recipe, refit, len(train), len(test)))
                coefficients, fitted_state = record_coefficients(context, model, len(train), fit, selected,
                    (scaler, fitted, empty, native.LOG_TRANSFORM))
                importance.extend(coefficients)
                parameters.extend(fitted_state)
                counts[model] += 1
            coverage_rows.append(dict(**context, events=len(test), issuers=test.gvkey.nunique(),
                                      transcript_covered=int(common.finbert_available.sum()),
                                      features_complete=int(common.finbert_complete.sum()),
                                      training_events=len(train),
                                      training_transcript_covered=int(train.finbert_available.sum()),
                                      training_features_complete=int(train[list(FEATURES)].notna().all(axis=1).sum())))
            outputs.append(common)
            print(f"{year} {option} +{horizon} {saved.window}: {len(test)} matched events", flush=True)
    baseline.need(counts == Counter(C=24, **{"C*": 24}), "Incomplete C/C* grid")
    baseline.need(snapshot() == checksums, "Archived A/B/B* research changed")
    predictions = pd.concat(outputs, ignore_index=True)
    baseline.need(predictions[list(MODELS)].notna().all().all(), "Incomplete matched five-model scores")
    for name, rows in [("finbert_selected_features", selections), ("finbert_estimator_cv", grids),
                       ("finbert_forward_paths", paths), ("finbert_feature_importance", importance),
                       ("finbert_native_coverage", coverage_rows)]:
        baseline.table(out, name, rows)
    parquet_write(out / "fitted_transformations_local.parquet", pd.DataFrame(parameters))
    parquet_write(out / "five_model_predictions_local.parquet", predictions)
    from evaluate_unified import evaluate
    evaluate(out, predictions, frames, stored_training)
    if snapshot() != checksums:
        raise ValueError("Archived research changed during evaluation")
    json_write(out / "integrity.json", {"preserved_a_b_outputs": checksums, "new_model_fits": dict(counts),
                                      "predictions_sha256": stable_digest(out / "five_model_predictions_local.parquet")})


if __name__ == "__main__":
    run()
