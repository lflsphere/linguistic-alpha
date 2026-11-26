"""Replay the committed Healthcare evaluation with one maturity-gated refit per year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / 'healthcare_extended_oos'
ROOT = HERE.parents[1]
BASE = 'ffee3b4a2caaa5c450b2ffcda353be70c229e9ac'

import importlib.util

loader = importlib.util.spec_from_file_location('static_healthcare', STATIC / 'evaluate.py')
static = importlib.util.module_from_spec(loader)
loader.loader.exec_module(static)
TARGET = static.TARGET
PERIODS = static.PERIODS
OPTIONS = static.OPTIONS
MODELS = static.MODELS
DATES = ['call_date', 'day0']
IDENTITY = ['transcriptid', 'keydevid', 'gvkey', 'permno', 'call_date', 'call_time_utc', 'day0']
TIMEZONE = 'America/New_York'
TABLES = ['model_performance', 'incremental_comparison', 'quintile_returns', 'quarterly_ic', 'stability_summary']
COMMITTED = ['evaluate.py', 'render_report.py', 'methodology.md', 'frozen_experiment_spec.json',
             'bootstrap_inference.csv', 'sector_benchmark_validation.csv', 'target_coverage.csv',
             *[f'{table}.csv' for table in TABLES]]
CONTRASTS = [('LM', 'B', 1, 0), ('LM', 'B*', 2, 0),
             ('schedule', 'A', 0, 3), ('schedule', 'B', 1, 4), ('schedule', 'B*', 2, 5)]


def require(condition, message):
    static.require(condition, message)


def write(data, out, name):
    data.to_csv(out / f'{name}.csv', index=False, float_format='%.14g')


def csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def checked_inputs():
    baseline_files = [STATIC / name for name in COMMITTED]
    git = ['git', '--no-optional-locks', '-c', f'safe.directory={ROOT.as_posix()}']
    spec = json.loads((STATIC / 'frozen_experiment_spec.json').read_text(encoding='utf-8'))
    archived_dir = Path(spec['source_model']).parent.as_posix().encode()
    legacy_dir = static.ORIGINAL.relative_to(ROOT).as_posix().encode()
    for path in baseline_files:
        ref = subprocess.check_output([*git, 'show', f'{BASE}:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT)
        if path.name in ('evaluate.py', 'methodology.md'):
            ref = ref.replace(archived_dir, legacy_dir)
        require(ref == path.read_bytes().replace(b'\r\n', b'\n'), f'Static commit mismatch for {path.name}')
    native = static.module(static.ORIGINAL / 'sector_model_b.py', 'frozen_native_models')
    provenance = static.check_inputs()
    require(provenance['revision'] == spec['source_revision'], 'Original revision changed')
    require(static.path_hash(static.EXTENDED) == spec['input_sha256'], 'Extended feature archive changed')
    archived_evaluator = (STATIC / 'evaluate.py').read_bytes().replace(legacy_dir, archived_dir)
    require(static.sha256(archived_evaluator).hexdigest() == spec['evaluation_script_sha256'], 'Static evaluator hash mismatch')
    require(static.path_hash(static.ARCHIVE / 'original_test_predictions.parquet') == spec['native_predictions_sha256'],
            'Archived model predictions changed')
    require(csv(STATIC / 'sector_benchmark_validation.csv').passed.all(), 'Saved sector benchmark failed validation')
    for option in OPTIONS:
        for entry in spec['specifications'][option].values():
            require(static.estimator(native, entry['algorithm']).get_params() == entry['resolved_parameters'],
                    'Frozen estimator differs from the installed original algorithm')
    return native, spec


def calendar_inputs():
    legacy = pd.read_parquet(static.RAW / 'crsp_dsi.parquet', columns=['date', 'sprtrn'])
    ciz = pd.read_parquet(static.CIZ / 'crsp_dsp500_v2.parquet', columns=['caldt', 'sprtrn'])
    old = pd.DatetimeIndex(pd.to_datetime(legacy.date)).astype('datetime64[ns]')
    cal = pd.DatetimeIndex(pd.to_datetime(ciz.caldt)).astype('datetime64[ns]')
    for index in (old, cal):
        require(index.is_unique and index.is_monotonic_increasing and not index.hasnans, 'Invalid CRSP calendar')
    require(old.isin(cal).all() and cal[-1] == pd.Timestamp('2026-06-30'), 'Calendar or vintage cutoff changed')
    return old, cal, legacy.set_index(old).sprtrn


def closes(days):
    # The refit boundaries are January regular sessions. Use the nominal model close
    # for the window ledger, conservatively timestamping any earlier half-day closes.
    return (days + pd.Timedelta(hours=16)).dt.tz_localize(TIMEZONE)


def session_offset(frame, calendar, offset):
    positions = calendar.get_indexer(pd.to_datetime(frame.day0).astype('datetime64[ns]'))
    i = positions + offset
    ok = (positions >= 0) & (i >= 0) & (i < len(calendar))
    dates = np.full(len(frame), np.datetime64('NaT'), dtype='datetime64[ns]')
    dates[ok] = calendar.to_numpy()[i[ok]]
    return pd.Series(dates, index=frame.index)


def normalize(frame):
    frame['gvkey'] = frame.gvkey.astype('string').str.zfill(6)
    for c in DATES:
        frame[c] = pd.to_datetime(frame[c]).astype('datetime64[ns]')
    return frame


def assert_events(candidates, saved, context):
    cols = IDENTITY + [TARGET]
    sort = lambda x: x[cols].sort_values('transcriptid').reset_index(drop=True)
    a, b = sort(candidates), sort(saved)
    require(a.transcriptid.is_unique and b.transcriptid.is_unique, f'Duplicate IDs in {context}')
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True, obj=context)


def event_inputs(cal, old):
    baseline = normalize(csv(STATIC / 'matched_predictions_local.csv', dtype={'gvkey': 'string'}))
    require(not baseline.duplicated(['option', 'period', 'transcriptid', 'model']).any(), 'Duplicated Static scores')
    sector = csv(STATIC / 'healthcare_sector_extended_car.csv', parse_dates=DATES)
    require(sector.transcriptid.is_unique, 'Duplicated sector labels')
    sector = sector.set_index('transcriptid')
    health, _ = static.observed_frame()
    eligible = static.eligible_new(health, sector.reset_index(), static.membership(static.CIZ, True))
    results = {}
    new = eligible.loc[eligible.verified & eligible.member].copy()
    for i, option in enumerate(OPTIONS, 1):
        archive = normalize(pd.read_parquet(static.NATIVE / f'model_table_opt{i}.parquet'))
        archive = archive.loc[archive.call_date.lt('2025-01-01')].copy()
        archive['origin'] = 'archived_native'
        archive['label_end'] = session_offset(archive, old, 20)
        extension = new.copy()
        if option == 'sector':
            extension[TARGET] = extension.sector_target
            extension[static.EARLY] = extension.transcriptid.map(sector.sector_early_car)
            extension['day0'] = extension.transcriptid.map(sector.day0)
        else:
            extension['day0'] = static.assign_day0(extension.clock, cal, extension.call_date).day0
        extension = normalize(extension.loc[extension[TARGET].notna()].copy())
        extension['origin'] = 'validated_extension'
        extension['label_end'] = session_offset(extension, cal, 20)
        require(extension.label_end.notna().all(), 'Maturity date missing for a validated extension label')
        cohort = [archive.loc[archive.call_date.dt.year.eq(2024) & archive[TARGET].notna()],
                  *[extension.loc[extension.call_date.dt.year.eq(y)] for y in (2025, 2026)]]
        for period, period_rows in zip(PERIODS, cohort):
            for model in MODELS:
                saved = baseline.loc[baseline.option.eq(option) & baseline.period.eq(period) & baseline.model.eq(model)]
                assert_events(period_rows, saved, f'{option} {period} {model}')
        results[option] = archive, extension, cohort
    a = baseline.loc[baseline.option.eq('market') & baseline.model.eq('A')]
    b = baseline.loc[baseline.option.eq('sector') & baseline.model.eq('A')]
    pd.testing.assert_frame_equal(a[IDENTITY].reset_index(drop=True), b[IDENTITY].reset_index(drop=True), check_dtype=False)
    return baseline, results


def archived_return_counts(archive, calendar, market):
    """Check archived return availability separately from elapsed window maturity."""
    sp = pd.concat([pd.read_parquet(static.RAW / f'crsp_dsf_{y}.parquet', columns=['permno', 'date', 'ret'])
                    for y in range(2010, 2025)], ignore_index=True)
    sp = sp.loc[sp.permno.isin(archive.permno) & sp.ret.notna(), ['permno', 'date']].copy()
    sp['date'] = pd.to_datetime(sp.date).dt.normalize().astype('datetime64[ns]')
    sp['permno'] = sp.permno.astype('int64')
    panel = pd.MultiIndex.from_frame(sp)
    require(panel.is_unique, 'Archived stock date keys are not unique')
    used = archive.loc[archive.label_end.notna()]
    ids = calendar.get_indexer(used.day0)[:, None] + np.arange(2, 21)
    query = pd.MultiIndex.from_arrays([np.repeat(used.permno.astype('int64').to_numpy(), 19),
                                       calendar[ids.ravel()]])
    raw = (panel.get_indexer(query).reshape(-1, 19) >= 0)
    dates = calendar[ids.ravel()]
    sector = pd.read_parquet(static.NATIVE / 'health_sector_daily.parquet', columns=['date', 'sector_ret'])
    sector = sector.set_index(pd.to_datetime(sector.date)).sector_ret
    counts = []
    for returns in [market, sector]:
        v = returns.reindex(dates).notna().to_numpy().reshape(-1, 19)
        counts.append(pd.Series((raw & v).sum(axis=1), index=used.transcriptid))
    return dict(zip(OPTIONS, counts))


def cutoff_training(history, test, cal, year):
    signal_day = session_offset(test, cal, 1)
    require(signal_day.notna().all(), f'Could not recover prediction time for {year}')
    cutoff = closes(signal_day).min()
    require(cutoff.year == year and cutoff.month == 1, 'Unexpected refit boundary. Check session hours.')
    pool = history.loc[history.call_date.dt.year.lt(year)].copy()
    pool['label_close_et'] = closes(pool.label_end)
    pool['maturity_status'] = 'eligible'
    pool.loc[pool.label_close_et.ge(cutoff), 'maturity_status'] = 'not_strictly_before_cutoff'
    pool.loc[pool.label_close_et.isna(), 'maturity_status'] = 'window_end_unavailable'
    pool.loc[pool[TARGET].isna(), 'maturity_status'] = 'missing_saved_target'
    train = pool.loc[pool.maturity_status.eq('eligible')]
    require(train.transcriptid.is_unique and train.label_close_et.lt(cutoff).all() and
            train[TARGET].notna().all() and train.call_date.dt.year.lt(year).all() and
            set(train.transcriptid).isdisjoint(test.transcriptid), 'Training maturity or overlap violation')
    return cutoff, train, pool


def fit_years(native, spec, baseline, sources, cal, raw_counts, out):
    audit, ledger, pred = [], [], []
    for option, (archive, extension, tests) in sources.items():
        history = pd.concat([archive, extension], ignore_index=True)
        history['paired_return_sessions'] = history.transcriptid.map(raw_counts[option])
        history.loc[history.origin.eq('validated_extension'), 'paired_return_sessions'] = 19
        for year, period, test in zip((2024, 2025, 2026), PERIODS, tests):
            cutoff, train, pool = cutoff_training(history, test, cal, year)
            n_legacy_short = int(train.paired_return_sessions.lt(19).sum())
            require(train.paired_return_sessions.ge(16).all(), 'Unverified coverage for an original training label')
            annual = train.call_date.dt.year
            audit.append(dict(option=option, period=period, prediction_year=year, refit_cutoff_et=cutoff.isoformat(),
                              refit_cutoff_utc=cutoff.tz_convert('UTC').isoformat(), historical_candidates=len(pool),
                              training_events=len(train), training_issuers=train.gvkey.nunique(),
                              train_through_2023=int(annual.le(2023).sum()), train_2024=int(annual.eq(2024).sum()),
                              train_2025=int(annual.eq(2025).sum()), missing_target=int(pool[TARGET].isna().sum()),
                              unresolved_window=int(pool.maturity_status.eq('window_end_unavailable').sum()),
                              at_or_after_cutoff=int(pool.maturity_status.eq('not_strictly_before_cutoff').sum()),
                              eligible_under_19_paired_sessions=n_legacy_short,
                              latest_training_call=train.call_date.max().date(),
                              latest_label_close_et=train.label_close_et.max().isoformat(),
                              same_year_training_events=int(annual.eq(year).sum()),
                              matched_events=len(test), prediction_issuers=test.gvkey.nunique()))
            ledger.append(pool[IDENTITY + ['origin', 'label_end', 'label_close_et', 'paired_return_sessions',
                                         'maturity_status']].assign(option=option, prediction_year=year, refit_cutoff_et=cutoff))
            target = stats.rankdata(train[TARGET], method='average')
            for model in MODELS:
                recipe = spec['specifications'][option][model]
                columns = recipe['columns']
                require(all(train[c].count() > 0 for c in columns), 'Training window cannot support frozen imputation')
                x, x_test, _, _ = native.preprocess(train, test, columns)
                fit = static.estimator(native, recipe['algorithm']).fit(x, target)
                saved = baseline.loc[baseline.option.eq(option) & baseline.period.eq(period) & baseline.model.eq(model)]
                scores = test[IDENTITY + [TARGET]].assign(prediction=fit.predict(x_test)).merge(
                    saved[['transcriptid', 'prediction']].rename(columns={'prediction': 'static_prediction'}),
                    on='transcriptid', validate='one_to_one')
                require(np.isfinite(scores[['prediction', 'static_prediction']]).all(axis=None), 'Nonfinite scores')
                if year == 2024:
                    require(np.array_equal(scores.prediction, scores.static_prediction), '2024 diagnostic replay is not exact')
                pred.append(scores.assign(option=option, model=model, period=period, train_n=len(train),
                                          refit_cutoff_et=cutoff, algorithm=recipe['algorithm']))
            print(f'{option} {year}: {len(train)} training events, cutoff {cutoff}, {len(test)} predictions/model.', flush=True)
    write(pd.DataFrame(audit), out, 'refit_training_audit')
    pd.concat(ledger, ignore_index=True).to_csv(out / 'training_maturity_local.csv', index=False)
    predictions = pd.concat(pred, ignore_index=True)
    predictions.to_csv(out / 'matched_predictions_local.csv', index=False, float_format='%.17g')
    return predictions


def metrics(pred, baseline, out):
    with TemporaryDirectory() as p:
        work = Path(p)
        static.statistics(baseline, work)
        for table in TABLES:
            a = csv(work / f'{table}.csv')
            b = csv(STATIC / f'{table}.csv')
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12, obj=f'Static {table} replay')
        static.statistics(pred, work)
        results = {}
        for table in TABLES:
            a = csv(STATIC / f'{table}.csv').assign(training='Static')
            b = csv(work / f'{table}.csv').assign(training='Expanding')
            results[table] = pd.concat([a, b], ignore_index=True)
            write(results[table], out, table)
    a = results['model_performance']
    keys = ['option', 'period', 'model', 'events', 'issuers']
    gains = a.loc[a.training.eq('Static')].drop(columns='training').merge(
        a.loc[a.training.eq('Expanding')].drop(columns='training'), on=keys, validate='one_to_one',
        suffixes=('_static', '_expanding'))
    gains['delta_rank_ic'] = gains.rank_ic_expanding - gains.rank_ic_static
    gains['delta_q5_q1'] = gains.q5_q1_expanding - gains.q5_q1_static
    write(gains, out, 'static_to_expanding')
    cohort = gains.loc[gains.model.eq('A'), ['option', 'period', 'events', 'issuers']].copy()
    cohort['exact_static_ids'] = True
    cohort['exact_static_targets'] = True
    cohort['exact_matched_models'] = True
    write(cohort, out, 'evaluation_match_audit')
    return results['incremental_comparison'], gains


def infer_point(values):
    rank = stats.rankdata(values, axis=0, method='average')
    ic = np.corrcoef(rank, rowvar=False)[-1, :-1]
    return np.array([ic[l]-ic[r] for _, _, l, r in CONTRASTS])


def paired_inference(pred, incremental, gains, out):
    rows = []
    for option, period, expanding in static.pdata(pred):
        base = pred.loc[pred.option.eq(option)]
        base = base.loc[base.period.eq(period)] if period != PERIODS[-1] else base.loc[base.period.ne(PERIODS[0])]
        fixed = base.pivot(index='transcriptid', columns='model', values='static_prediction').reindex(expanding.transcriptid)
        matrix = np.column_stack([expanding[list(MODELS)], fixed[list(MODELS)], expanding[TARGET]])
        require(np.isfinite(matrix).all(), 'Unmatched event in paired Bootstraps')
        observed = infer_point(matrix)
        delta = incremental.loc[incremental.training.eq('Expanding') & incremental.option.eq(option) & incremental.period.eq(period)]
        updates = gains.loc[gains.option.eq(option) & gains.period.eq(period)]
        np.testing.assert_allclose(observed, np.r_[delta.delta_rank_ic, updates.delta_rank_ic], rtol=0, atol=1e-12)
        start = '2025-01' if period == PERIODS[-1] else f'{period[:4]}-01'
        end = '2026-06' if period in PERIODS[2:] else f'{period[:4]}-12'
        calendar = pd.period_range(start, end, freq='M')
        issuer_groups = list(expanding.groupby('gvkey').indices.values())
        months = expanding.groupby(expanding.call_date.dt.to_period('M')).indices
        month_groups = [months.get(month, np.array([], dtype=int)) for month in calendar]
        nominal_blocks = len(calendar)//static.BLOCK
        sufficient = len(issuer_groups) >= 30 and nominal_blocks >= 5 and period != PERIODS[0]
        for method in ('issuer_cluster', 'time_block'):
            rng = np.random.default_rng(static.SEED + 11*OPTIONS.index(option) + PERIODS.index(period))
            draws = []
            if sufficient:
                for _ in range(static.N_BOOT):
                    if method == 'issuer_cluster':
                        blocks = rng.integers(0, len(issuer_groups), size=len(issuer_groups))
                        idx = np.concatenate([issuer_groups[i] for i in blocks])
                    else:
                        n_blocks = int(np.ceil(len(calendar)/static.BLOCK))
                        beginnings = rng.integers(0, len(calendar)-static.BLOCK+1, size=n_blocks)
                        positions = (beginnings[:, None]+np.arange(static.BLOCK)).ravel()[:len(calendar)]
                        idx = np.concatenate([month_groups[i] for i in positions])
                    if len(idx) >= 30:
                        v = infer_point(matrix[idx])
                        if np.isfinite(v).all():
                            draws.append(v)
            usable = len(draws) >= .95*static.N_BOOT
            for j, (family, model, _, _) in enumerate(CONTRASTS):
                sim = np.asarray(draws)[:, j] if usable else np.array([])
                lo, hi = np.quantile(sim, [.025, .975]) if usable else (np.nan, np.nan)
                p = ((1+np.count_nonzero(abs(sim-observed[j]) >= abs(observed[j])))/(len(sim)+1)
                     if usable and period == PERIODS[-1] else np.nan)
                rows.append(dict(option=option, period=period, family=family, model=model, method=method,
                                 events=len(expanding), issuer_clusters=len(issuer_groups), calendar_months=len(calendar),
                                 nominal_time_blocks=nominal_blocks, n_requested=static.N_BOOT if sufficient else 0,
                                 n_valid=len(draws), point=observed[j], lower=lo, upper=hi,
                                 crosses_zero=(lo <= 0 <= hi) if usable else None, p_centered=p,
                                 status='exploratory pointwise' if usable else 'descriptive only'))
    result = pd.DataFrame(rows)
    result['holm_p_pooled'] = np.nan
    result['passes_exploratory_screen'] = False
    pooled = result.loc[result.period.eq(PERIODS[-1])]
    for family, expected in [('LM', 4), ('schedule', 6)]:
        family_p = pooled.loc[pooled.family.eq(family)].groupby(['option', 'model']).p_centered.max().sort_values()
        if len(family_p) != expected or not family_p.notna().all():
            continue
        adjusted = (family_p*np.arange(expected, 0, -1)).cummax().clip(upper=1)
        for (o, model), value in adjusted.items():
            sel = result.period.eq(PERIODS[-1]) & result.family.eq(family) & result.model.eq(model) & result.option.eq(o)
            v = result.loc[sel]
            result.loc[sel, 'holm_p_pooled'] = value
            result.loc[sel, 'passes_exploratory_screen'] = value <= .05 and (v.lower.gt(0).all() or v.upper.lt(0).all())
    write(result, out, 'bootstrap_inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE)
    out = parser.parse_args().out.resolve()
    require(out != STATIC and out != static.ARCHIVE, 'Choose a fresh output location')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'expanding_experiment_spec.json'
    require(not manifest.exists(), 'Run into a fresh output location')
    native, frozen = checked_inputs()
    protected = [STATIC / name for name in COMMITTED] + [static.INPUT, static.CAR, static.EXTENDED,
                 STATIC/'matched_predictions_local.csv', STATIC/'healthcare_sector_extended_car.csv',
                 STATIC/'sector_benchmark_daily.csv', static.ARCHIVE/'original_test_predictions.parquet',
                 *[static.NATIVE/f'model_table_opt{x}.parquet' for x in (1,2)],
                 static.NATIVE/'health_sector_daily.parquet']
    hashes = {p.relative_to(ROOT).as_posix(): static.path_hash(p) for p in protected}
    experiment = {'static_baseline_commit': BASE, 'status': 'post-hoc exploratory',
                  'static_frozen_specification': frozen, 'protected_inputs_sha256': hashes,
                  'changes': 'Annual update schedule and its necessary availability gate only. Original fits and benchmark definitions are unchanged.',
                  'evaluation_sha256': static.path_hash(__file__),
                  'prediction_years': [2024, 2025, 2026],
                  'training_ranges': ['2010–2023', '2010–2024', '2010–2025'],
                  'refit_time': 'First Day +1 close among the existing matched test events in each year. Use original CRSP session mapping and a 16:00 ET close in the cutoff ledger. Both cutoff and maturity are recorded. Include only prior-calendar-year calls with a saved target whose final Day +20 close is strictly earlier.',
                  'archive_labels': 'Use saved native labels and original missing-return convention through 2024. Verify the full time span and paired-return coverage. Append only the validated Static 2025 matched events in 2026. Do not fill the missing native 2024 target.',
                  'pooled': 'Recompute IC and quintiles from concatenated raw held-out midrank scores with no annual normalization. Rank-score scales can drift with changing training-set sizes.',
                  'inference': 'Same resampling scheme, metrics, seed, block size, sample and replication thresholds as Static. Pooled Holm is separate across the four LM effects and six schedule effects. All inference is exploratory and conditional on the fitted scores.'}
    with manifest.open('x', encoding='utf-8') as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False)
    old, cal, market = calendar_inputs()
    baseline, sources = event_inputs(cal, old)
    history = sources['market'][0]
    coverage = archived_return_counts(history, old, market)
    pred = fit_years(native, frozen, baseline, sources, cal, coverage, out)
    increments, gains = metrics(pred, baseline, out)
    paired_inference(pred, increments, gains, out)
    require(all(static.path_hash(ROOT/p) == h for p,h in hashes.items()), 'A protected input changed')
    print('Exact Static cohorts, targets and archived predictions verified.', flush=True)


if __name__ == '__main__':
    main()
