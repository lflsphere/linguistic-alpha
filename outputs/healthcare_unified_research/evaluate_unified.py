"""Create annual, matched-event aggregate tables from the saved five-model scores."""

import numpy as np
import pandas as pd
from scipy import stats

from fit_finbert import BASELINE, CONTROLS, FEATURES, MODELS, load_csv
from extract_finbert import json_write
from research_metrics import correlation, summarize, uncertainty

from run_comparison import HORIZONS, LM, OPTIONS, SEED, WINDOWS, YEARS, need, table

FOCAL_PAIRS = (("B", "A"), ("B*", "A"), ("C", "A"), ("C*", "A"), ("C", "B"))
INCREMENTS = tuple(f"{upper}−{lower}" for upper, lower in FOCAL_PAIRS)
FULL = "Five-model common (train-imputed)"
MATCHED = "Five-model horizon intersection"
NATIVE = "All 13 observed (sensitivity)"


def test_metrics(case, context, performance, quintiles, increments):
    results = {}
    n, issuers = len(case), case.gvkey.nunique()
    actual = case.actual_car.to_numpy(float)
    for model in MODELS:
        score = case[model].to_numpy(float)
        rank, pearson = correlation(score, actual)
        sorts, spread, shape = summarize(score, actual, issuers)
        results[model] = rank
        performance.append(dict(**context, model=model, events=n, issuers=issuers, rank_ic=rank,
                                pearson=pearson, q5_minus_q1=spread, monotonicity=shape))
        quintiles.extend(dict(**context, model=model, **row) for row in sorts)
    increments.extend(dict(**context, contrast=label, events=n, issuers=issuers,
                           delta_rank_ic=results[upper] - results[lower])
                      for label, (upper, lower) in zip(INCREMENTS, FOCAL_PAIRS))
    return results


def append_intervals(distribution, data, models, labels, year, seed, context, cols=None, pairs=None):
    eye = np.eye(len(models))
    contrasts = np.stack([eye[models.index(upper)] - eye[models.index(lower)]
                          for _, upper, lower in labels])
    columns = [*models, "actual_car"] if cols is None else cols
    model_pairs = [(index, len(models)) for index in range(len(models))] if pairs is None else pairs
    for index, row in uncertainty(data, columns, model_pairs, contrasts, year, seed):
        distribution.append(dict(**context, contrast=labels[index][0], **row))


def saved_performance(out, forecasts):
    performance, quintiles, increments, windows, horizons, inference = [], [], [], [], [], []
    selected = load_csv(BASELINE, "annual_model_performance.csv")
    rows = {}
    for year in YEARS:
        for benchmark in OPTIONS:
            sample = forecasts.loc[forecasts.year.eq(year) & forecasts.benchmark.eq(benchmark)]
            cohorts = {h: set(sample.loc[sample.horizon.eq(h), "transcriptid"]) for h in HORIZONS}
            intersection = set.intersection(*cohorts.values())
            data = {}
            year_context = dict(year=year, benchmark=benchmark)
            for horizon in HORIZONS:
                for schedule in WINDOWS:
                    case = sample.loc[sample.horizon.eq(horizon) & sample.window.eq(schedule)]
                    case = case.sort_values("transcriptid").reset_index(drop=True)
                    matched = case.loc[case.transcriptid.isin(intersection)]
                    complete = case.loc[case.finbert_complete]
                    need(len(case) == len(cohorts[horizon]), "Repeated test event within configuration")
                    data[horizon, schedule] = case, matched.reset_index(drop=True)
                    for definition, group in [(FULL, case), (MATCHED, matched), (NATIVE, complete)]:
                        context = dict(**year_context, horizon=horizon, window=schedule, sample=definition)
                        result = test_metrics(group, context, performance, quintiles, increments)
                        rows[horizon, schedule, definition] = result
                        if definition in (FULL, MATCHED):
                            legacy_sample = "Full horizon" if definition == FULL else "Matched horizons"
                            eligible = selected.loc[selected.year.eq(year) & selected.benchmark.eq(benchmark)
                                & selected.horizon.eq(horizon) & selected.window.eq(schedule)
                                & selected["sample"].eq(legacy_sample)]
                            for old in eligible.itertuples(index=False):
                                need(old.events == len(group) and abs(result[old.model] - old.rank_ic) < 1e-12,
                                     "Original A/B/B* metrics changed")
                        if definition == FULL:
                            append_intervals(inference, group, list(MODELS),
                                [(label, m, base) for label, (m, base) in zip(INCREMENTS, FOCAL_PAIRS)],
                                year, SEED + year * 100 + horizon * 10 + WINDOWS.index(schedule),
                                dict(**context, family="Language"))
                exp, _ = data[horizon, WINDOWS[0]]
                roll, _ = data[horizon, WINDOWS[1]]
                need(exp.transcriptid.equals(roll.transcriptid) and exp.actual_car.equals(roll.actual_car),
                     "Window comparison cohorts differ")
                titles = [(f"{prefix} {m}") for prefix in ("Exp", "Roll") for m in MODELS]
                merged = exp.rename(columns=dict(zip(MODELS, titles[:5]))).join(
                    roll[list(MODELS)].rename(columns=dict(zip(MODELS, titles[5:]))))
                append_intervals(inference, merged, titles,
                    [(m, f"Roll {m}", f"Exp {m}") for m in MODELS], year, SEED + year * 100 + horizon,
                    dict(**year_context, horizon=horizon, window="Both", family="Training window", sample=FULL))
                for model in MODELS:
                    a = rows[horizon, WINDOWS[0], FULL][model]
                    b = rows[horizon, WINDOWS[1], FULL][model]
                    windows.append(dict(**year_context, horizon=horizon, model=model, events=len(exp),
                                        issuers=exp.gvkey.nunique(), expanding_rank_ic=a, rolling_rank_ic=b,
                                        rolling_minus_expanding=b - a))
            for schedule in WINDOWS:
                _, short = data[5, schedule]
                _, long = data[20, schedule]
                need(short.transcriptid.equals(long.transcriptid), "Horizon comparison cohorts differ")
                names = [f"{prefix} {m}" for prefix in ("Short", "Long") for m in MODELS]
                joined = short.rename(columns=dict(zip((*MODELS, "actual_car"), (*names[:5], "short_car")))).join(
                    long[[*MODELS, "actual_car"]].rename(columns=dict(zip(
                        (*MODELS, "actual_car"), (*names[5:], "long_car")))))
                append_intervals(inference, joined, names, [(m, f"Long {m}", f"Short {m}") for m in MODELS],
                    year, SEED + year * 100 + WINDOWS.index(schedule), dict(**year_context, horizon=0,
                    window=schedule, family="Prediction horizon", sample=MATCHED),
                    cols=[*names, "short_car", "long_car"],
                    pairs=[(i, 10 if i < 5 else 11) for i in range(10)])
                for model in MODELS:
                    a = rows[5, schedule, MATCHED][model]
                    b = rows[20, schedule, MATCHED][model]
                    horizons.append(dict(**year_context, window=schedule, model=model, events=len(short),
                                         issuers=short.gvkey.nunique(), short_rank_ic=a, long_rank_ic=b,
                                         long_minus_short=b - a))
    result_files = (
        ("annual_model_performance", performance), ("annual_increments", increments),
        ("window_comparison", windows), ("horizon_comparison", horizons),
        ("model_quintiles", quintiles), ("paired_inference", inference),
    )
    for file, entries in result_files:
        table(out, file, entries)
    selected.to_csv(out / "original_ab_performance.csv", index=False, float_format="%.17g")


def factor_metrics(out, frames, refit):
    rows, stable, quints, associations, missing, inference = [], [], [], [], [], []
    for benchmark, frame in frames.items():
        for h in HORIZONS:
            cohort = frame.loc[frame[f"eligible_{h}"]].copy()
            target = f"car_2_{h}"
            development = cohort.loc[cohort.year.lt(2024) & cohort[f"mature_{h}"].lt(refit[2024])]
            for yr in (0, *YEARS):
                group = development if yr == 0 else cohort.loc[cohort.year.eq(yr)]
                context = dict(year=yr, scope="Development" if yr == 0 else "Outer test",
                               horizon=h, benchmark=benchmark)
                matrix = group[[*FEATURES, *LM, *CONTROLS]].corr(method="spearman", min_periods=20)
                for name in (*FEATURES, *LM):
                    subset = group.loc[group[name].notna()]
                    v, actual = subset[name].to_numpy(float), subset[target].to_numpy(float)
                    n, issuers = len(subset), subset.gvkey.nunique()
                    rank, pearson = correlation(v, actual)
                    quintiles, spread, shape = summarize(v, actual, issuers)
                    block = "LM" if name in LM else "FinBERT"
                    factor_context = dict(**context, factor=name, representation=block)
                    rows.append(dict(**factor_context, events=n, issuers=issuers, eligible_events=len(group),
                                     rank_ic=rank, pearson=pearson, q5_minus_q1=spread, monotonicity=shape))
                    missing.append(dict(**factor_context, observations=n, missing=len(group) - n,
                                        eligible_events=len(group), missing_share=1 - n / len(group)))
                    quints.extend(dict(**factor_context, **row) for row in quintiles)
                    for period, quarter in subset.groupby(subset.call_date.dt.to_period("Q")):
                        count, firms = len(quarter), quarter.gvkey.nunique()
                        eligible = count >= 25 and firms >= 15
                        value = correlation(quarter[name].to_numpy(), quarter[target].to_numpy())[0] if eligible else np.nan
                        stable.append(dict(**factor_context, quarter=str(period), events=count, issuers=firms,
                                           rank_ic=value, eligible=eligible, partial=str(period) == "2026Q2"))
                    if yr:
                        for _, ci in uncertainty(subset, [name, target], [(0, 1)], np.ones((1, 1)), yr,
                            SEED + 10000 + 100 * yr + 10 * h + (*LM, *FEATURES).index(name)
                            + OPTIONS.index(benchmark)):
                            inference.append(dict(**factor_context, **ci))
                for left, right, name in [(LM, LM, "LM × LM"), (FEATURES, FEATURES, "FinBERT × FinBERT"),
                                           (LM, FEATURES, "LM × FinBERT"), (LM, CONTROLS, "LM × controls"),
                                           (FEATURES, CONTROLS, "FinBERT × controls")]:
                    for x in left:
                        for y in right:
                            associations.append(dict(**context, block=name, row=x, column=y,
                                                     rank_corr=matrix.at[x, y],
                                                     events=int((group[x].notna() & group[y].notna()).sum())))
    for name, entries in [("factor_ic", rows), ("factor_stability", stable), ("factor_quintiles", quints),
                          ("correlations", associations), ("feature_missingness", missing),
                          ("factor_inference", inference)]:
        table(out, name, entries)


def evaluate(out, forecasts, frames, training):
    saved_performance(out, forecasts)
    cutoffs = {year: pd.Timestamp(group.refit_cutoff.iloc[0]) for year, group in training.groupby("year")}
    factor_metrics(out, frames, cutoffs)
    verify_factor_tables(out, frames, cutoffs)


def verify_factor_tables(out, frames, refit):
    matrix = load_csv(out, "correlations.csv")
    factors = load_csv(out, "factor_ic.csv")
    count_matrix, count_factors = 0, 0
    for option, frame in frames.items():
        for horizon in HORIZONS:
            eligible = frame.loc[frame[f"eligible_{horizon}"]]
            target = f"car_2_{horizon}"
            for year in (0, *YEARS):
                group = (eligible.loc[eligible.year.lt(2024) & eligible[f"mature_{horizon}"].lt(refit[2024])]
                         if year == 0 else eligible.loc[eligible.year.eq(year)])
                ours = matrix.loc[matrix.benchmark.eq(option) & matrix.horizon.eq(horizon) & matrix.year.eq(year)]
                for row in ours.itertuples(index=False):
                    x, y = group[row.row], group[row.column]
                    use = x.notna() & y.notna()
                    need(use.sum() == row.events, "Wrong pairwise count in correlation table")
                    value = np.nan
                    if use.sum() >= 20 and x[use].nunique() > 1 and y[use].nunique() > 1:
                        value = stats.spearmanr(x[use], y[use]).statistic
                    np.testing.assert_allclose(row.rank_corr, value, atol=1e-12, rtol=0, equal_nan=True)
                    count_matrix += 1
                reported = factors.loc[factors.benchmark.eq(option) & factors.horizon.eq(horizon)
                                       & factors.year.eq(year)]
                for row in reported.itertuples(index=False):
                    seen = group.loc[group[row.factor].notna()]
                    need(len(seen) == row.events and seen.gvkey.nunique() == row.issuers,
                         "Wrong factor cohort or reported count")
                    if len(seen) >= 3 and seen[row.factor].nunique() > 1 and seen[target].nunique() > 1:
                        value = stats.spearmanr(seen[row.factor], seen[target]).statistic
                    else:
                        value = np.nan
                    np.testing.assert_allclose(row.rank_ic, value, atol=1e-12, rtol=0, equal_nan=True)
                    count_factors += 1
    need(count_matrix == len(matrix) and count_factors == len(factors), "Incomplete factor check")
    json_write(out / "factor_validation.json", dict(factor_rows=count_factors, correlation_cells=count_matrix))
