"""Freeze and present the three saved Healthcare test cohorts without fitting models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'outputs/healthcare_original_pierre_replication'
ANNUAL = ROOT / 'outputs/healthcare_expanding_oos'
PHASE1 = ROOT / 'outputs/healthcare_factor_evaluation/dashboard.html'
BASELINE = 'ffee3b4a2caaa5c450b2ffcda353be70c229e9ac'
MODELS = ('A', 'B', 'B*')
BENCH = {'market': 'S&P 500-adjusted', 'sector': 'Healthcare sector-adjusted'}
YEARS = {2024: 'Test 2024', 2025: 'Test 2025', 2026: 'Test 2026 (Partial)'}
TITLE = 'Linguistic Alpha Evaluation'
SUBTITLE = 'Original A/B/B* Models | Annual Expanding Window | Test 2024–2026'
BRAND = 'HEALTHCARE LINGUISTIC ALPHA RESEARCH'
PDF = 'Healthcare_Linguistic_Alpha_Final_Report.pdf'
PALETTE = {'A': '#335f91', 'B': '#08766c', 'B*': '#795b93'}
INK, NAVY, MUTE, LINE, PAPER, SOFT, TEAL = '#172f45', '#112d43', '#496074', '#d6e3e8', '#f0f5f6', '#e6f3ef', '#08766c'
SECTIONS = ('Overview', 'Model Rank IC by Test Year', 'Incremental Rank IC from LM Features',
            'Predicted-Score Quintile Returns', 'Quarterly Rank IC', 'Statistical Inference',
            'Data Coverage and Model Methodology')
FILES = ('annual_model_performance', 'annual_incremental_rank_ic', 'annual_quintile_returns',
         'annual_stability', 'annual_inference', 'annual_coverage')
IDS = ('transcriptid', 'keydevid', 'permno', 'call_date', 'call_time_utc', 'day0')
TARGET = 'car_2_20'


def verify(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def select(frame, **kw):
    mask = np.logical_and.reduce([frame[name].eq(value) for name, value in kw.items()])
    return frame.loc[mask]


def normalize(frame):
    for field in ('call_date', 'day0'):
        frame[field] = pd.to_datetime(frame[field]).astype('datetime64[ns]')
    if 'gvkey' in frame:
        frame['gvkey'] = frame.gvkey.astype('string').str.zfill(6)
    return frame


def exact_match(a, b, fields, context):
    def order(frame):
        verify(frame.transcriptid.is_unique, f'Duplicate events in {context}')
        return frame[list(fields)].sort_values('transcriptid').reset_index(drop=True)
    pd.testing.assert_frame_equal(order(a), order(b), check_exact=True, check_dtype=False, obj=context)


def year_rows(frame, year, **kw):
    period = {2025: '2025', 2026: '2026 partial'}[year]
    return select(frame, period=period, **kw)


def assert_values(actual, expected, name):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=name)


def freeze(out):
    from scipy import stats

    original_names = ('original_test_predictions.parquet', 'native_model_metrics.csv', 'original_model_splits.csv',
                      'native_strategy_scores.csv', 'provenance.json', 'earnings_health/model_table_opt1.parquet',
                      'earnings_health/model_table_opt2.parquet')
    annual_names = ('matched_predictions_local.csv', 'model_performance.csv', 'incremental_comparison.csv',
                    'quintile_returns.csv', 'quarterly_ic.csv', 'bootstrap_inference.csv', 'refit_training_audit.csv',
                    'expanding_experiment_spec.json')
    files = [ORIGINAL/p for p in original_names] + [ANNUAL/p for p in annual_names] + [PHASE1]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    original = normalize(pd.read_parquet(ORIGINAL/original_names[0]))
    archive_metrics = read(ORIGINAL/original_names[1])
    archived_splits = normalize(read(ORIGINAL/original_names[2]))
    archive_routes = read(ORIGINAL/original_names[3])
    annual = normalize(read(ANNUAL/annual_names[0], dtype={'gvkey': 'string'}))
    reports = {name.removesuffix('.csv'): read(ANNUAL/name) for name in annual_names[1:-1]}
    spec = json.loads((ANNUAL/annual_names[-1]).read_text(encoding='utf-8'))
    recipes = spec['static_frozen_specification']['specifications']
    verify(spec['static_baseline_commit'] == BASELINE, 'Source baseline provenance mismatch')
    saved_original_hash = spec['static_frozen_specification']['native_predictions_sha256']
    verify(digest(ORIGINAL/original_names[0]) == saved_original_hash, 'Original predictions hash mismatch')
    verify(original.call_date.dt.year.eq(2024).all(), 'Archived prediction table contains a different test year')
    original['option'] = original.option.map({1:'market',2:'sector'})
    verify(set(original.option) == set(BENCH), 'Unexpected archive benchmark')
    annual = annual.loc[annual.call_date.dt.year.isin((2025, 2026))].copy()
    verify(annual.loc[annual.call_date.dt.year.eq(2025), 'period'].eq('2025').all() and
           annual.loc[annual.call_date.dt.year.eq(2026), 'period'].eq('2026 partial').all(), 'Annual prediction year mismatch')
    cohorts, performance, increments, quintiles, stability, inference, coverage = {}, [], [], [], [], [], []
    for year in YEARS:
        for number, option in enumerate(BENCH, 1):
            test = select(original, option=option) if year == 2024 else year_rows(annual, year, option=option)
            base = select(test, model='A')
            n, issuers = len(base), base.gvkey.nunique()
            verify(bool(n and issuers), f'Missing cohort {year} {option}')
            if year == 2024:
                splits = select(archived_splits, option=number, sample='test')
                exact_match(base, splits, IDS, 'Original archive split')
                archived_table = normalize(pd.read_parquet(ORIGINAL/original_names[4+number], columns=[*IDS,'gvkey',TARGET]))
                archived_table = archived_table.loc[archived_table.call_date.dt.year.eq(2024) & archived_table[TARGET].notna()]
                exact_match(base, archived_table, [*IDS,'gvkey',TARGET], 'Original pre-extension CARs')
                saved = select(archive_metrics, option=number).set_index('model')
                train_n, cutoff = int(saved.n_train.iloc[0]), '2023-12-31 (training end date)'
                verify(saved.n_test.eq(n).all() and saved.test_securities.eq(base.permno.nunique()).all(), 'Native counts mismatch')
                winner = select(archive_routes, option=number, selected_as_b=True).iloc[0]
                verify(winner.strategy==saved.loc['B','strategy'] and winner.lm_inputs_native==saved.loc['B','lm_inputs_native'], 'Archive winner mismatch')
            else:
                audit = select(reports['refit_training_audit'], option=option, prediction_year=year).iloc[0]
                train_n, cutoff = int(audit.training_events), audit.refit_cutoff_et
                clock = pd.to_datetime(test.refit_cutoff_et, utc=True)
                verify((n,issuers)==(audit.matched_events,audit.prediction_issuers) and test.train_n.eq(train_n).all() and
                       clock.eq(pd.Timestamp(cutoff).tz_convert('UTC')).all(), 'Refit audit mismatch')
            common = dict(year=year, test=YEARS[year], option=option, benchmark=BENCH[option])
            cohorts[year,option] = base
            coverage.append(dict(**common, events=n, issuers=issuers, first_call=str(base.call_date.min().date()),
                                 last_call=str(base.call_date.max().date()), training_events=train_n,
                                 training_history=f'2010–{year-1}', training_cutoff_et=cutoff,
                                 source='Original pre-extension archive' if year==2024 else 'Saved annual expanding predictions',
                                 crsp_cutoff='2024-12-31' if year==2024 else '2026-06-30'))
            scores = {}
            for model in MODELS:
                group = select(test, model=model)
                exact_match(base, group, [*IDS,'gvkey',TARGET], f'Matched targets {year} {option} {model}')
                pred, y = group.prediction.to_numpy(), group[TARGET].to_numpy()
                verify(np.isfinite(pred).all() and np.isfinite(y).all(), 'Nonfinite forecast or target')
                ic, pearson = float(stats.spearmanr(pred, y).statistic), float(stats.pearsonr(pred, y).statistic)
                recipe = recipes[option][model]
                if year == 2024:
                    ref = saved.loc[model]
                    verify(ic==ref.native_rank_ic and pearson==ref.native_ic, 'Archived IC differs from original saved result')
                    verify(recipe['columns']==ref.features.split(',') and recipe['algorithm']==ref.algorithm and
                           recipe['strategy']==ref.strategy, 'Saved model is inconsistent with the original archive')
                else:
                    ref = year_rows(reports['model_performance'],year,option=option,model=model,training='Expanding').iloc[0]
                    verify(group.algorithm.eq(recipe['algorithm']).all() and (n,issuers)==(ref.events,ref.issuers), 'Annual model metadata mismatch')
                    assert_values([ic,pearson],[ref.rank_ic,ref.pearson],'Annual IC and Pearson')
                verify(n>=50 and issuers>=20, 'Frozen sample minimum for quintiles not met')
                codes = pd.qcut(pred, 5, labels=False, duplicates='drop')
                verify(len(set(codes))==5 and not pd.isna(codes).any(), 'Frozen quintile grouping failed')
                groups = [y[codes==g] for g in range(5)]
                means = np.array([v.mean() for v in groups])
                differences = np.diff(means)
                profile = 'increasing' if (differences>0).all() else 'decreasing' if (differences<0).all() else 'mixed'
                spread = float(means[-1]-means[0])
                assert_values(spread, ref.native_quintile_spread if year==2024 else ref.q5_q1, 'Q5 minus Q1')
                for k, v in enumerate(groups,1):
                    values = [len(v),float(means[k-1]),float(np.median(v)),float(np.std(v,ddof=1))]
                    if year!=2024:
                        q = year_rows(reports['quintile_returns'],year,option=option,model=model,training='Expanding',group=k).iloc[0]
                        assert_values(values,[q.n,q.mean_car,q.median_car,q.sd_car],'Annual quintile data')
                    quintiles.append(dict(**common,model=model,quintile=k,events=values[0],mean_car=values[1],
                                          median_car=values[2],sd_car=values[3]))
                performance.append(dict(**common,model=model,events=n,issuers=issuers,rank_ic=ic,pearson=pearson,
                                        q5_minus_q1=spread,quintile_monotonicity=profile,algorithm=recipe['algorithm']))
                scores[model] = ic
                if year!=2024:
                    quarters = group.call_date.dt.to_period('Q').astype(str)
                    saved_q = select(reports['quarterly_ic'], option=option, model=model, training='Expanding')
                    for quarter in sorted(quarters.unique()):
                        sample = group.loc[quarters.eq(quarter)]
                        q = select(saved_q,quarter=quarter).iloc[0]
                        verify((len(sample),sample.gvkey.nunique())==(q.events,q.issuers), 'Quarterly sample mismatch')
                        rho = stats.spearmanr(sample.prediction,sample[TARGET]).statistic
                        assert_values(rho,q.rank_ic,'Quarterly Rank IC')
                        stability.append(dict(**common,model=model,quarter=quarter,events=int(q.events),issuers=int(q.issuers),
                                              rank_ic=float(rho),partial=bool(q.partial)))
            for model in MODELS[1:]:
                effect = scores[model]-scores['A']
                contrast = f'{model}−A'
                increments.append(dict(**common,model=model,contrast=contrast,events=n,issuers=issuers,delta_rank_ic=effect))
                if year==2024:
                    assert_values(effect, saved.loc[model,'incremental_rank_ic_vs_a'], 'Original increment')
                    inference.append(dict(**common,model=model,contrast=contrast,events=n,issuers=issuers,point=effect,
                                          method='Unavailable',lower=None,upper=None,draws=0,nominal_time_blocks=None,
                                          status='Original test outcomes informed model selection'))
                    continue
                ref = year_rows(reports['incremental_comparison'],year,option=option,model=model,training='Expanding').iloc[0]
                assert_values(effect,ref.delta_rank_ic,'Annual increment')
                bands = year_rows(reports['bootstrap_inference'],year,option=option,model=model,family='LM')
                verify(len(bands)==2 and set(bands.method)=={'issuer_cluster','time_block'} and
                       bands.events.eq(n).all() and bands.issuer_clusters.eq(issuers).all(), 'Annual inference mismatch')
                for b in bands.itertuples():
                    assert_values(effect,b.point,'Annual bootstrap point')
                    if year==2025:
                        verify(b.n_valid==2499 and np.isfinite(b.lower+b.upper) and b.nominal_time_blocks==6,
                               'Unavailable annual 2025 interval')
                    else:
                        verify(b.n_valid==0 and pd.isna(b.lower) and pd.isna(b.upper) and b.nominal_time_blocks==3,
                               'Mislabelled interval for partial 2026')
                    inference.append(dict(**common,model=model,contrast=contrast,events=n,issuers=issuers,point=effect,
                                          method=b.method,lower=b.lower,upper=b.upper,draws=b.n_valid,
                                          nominal_time_blocks=b.nominal_time_blocks,
                                          status='Year-specific paired 95% interval' if year==2025 else 'Descriptive, insufficient time blocks'))
    for year in YEARS:
        exact_match(cohorts[year,'market'],cohorts[year,'sector'],[*IDS,'gvkey'],f'Common cohort {year}')
    verify(all(digest(ROOT/p)==h for p,h in hashes.items()), 'Source files changed during verification')
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in zip(FILES,(performance,increments,quintiles,stability,inference,coverage)):
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,float_format='%.17g')
    manifest = {'report':TITLE, 'protected_baseline_commit':BASELINE, 'no_model_fit':True,
                'sources_by_year':{YEARS[y]:[p.relative_to(ROOT).as_posix() for p in
                                  ([ORIGINAL/x for x in original_names] if y==2024 else [ANNUAL/x for x in annual_names])]
                                   for y in YEARS}, 'visual_reference':PHASE1.relative_to(ROOT).as_posix(),
                'source_sha256':hashes, 'report_builder_sha256':digest(Path(__file__)),
                'aggregate_sha256':{name:digest(out/f'{name}.csv') for name in FILES},
                'model_definitions':recipes, 'controls':spec['static_frozen_specification']['controls'],
                'lm9':spec['static_frozen_specification']['LM9'],
                'validation':['Identical event IDs and realized targets across A/B/B* within each benchmark and year.',
                              'Exact match to original test split, pre-extension realized labels and unrounded IC in 2024.',
                              'Annual expanding counts, metrics, quintiles and quarterly IC verified from saved predictions.',
                              '2025 confidence intervals retained from saved within-year paired bootstrap.']}
    (out/'source_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified 2024 original archive and saved 2025–26 evaluations. Saved six annual aggregate tables.',flush=True)


def num(value, places=4):
    return '—' if pd.isna(value) else f'{value:.{places}f}'.replace('-', '−')


def exact(value):
    return str(float(value)).replace('-', '−')


def delta(value):
    return ('+' if value > 0 else '') + num(value)


def results(data, year=None, option=None):
    use = data
    if year is not None:
        use = use.loc[use.year.eq(year)]
    if option is not None:
        use = use.loc[use.option.eq(option)]
    return use


def make_pdf(out, data, manifest):
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','axes.unicode_minus':False})
    perf,inc,qs,stability,bands,cov = [data[x] for x in FILES]
    papers = [SECTIONS[0],SECTIONS[1],SECTIONS[2],SECTIONS[3],SECTIONS[3],SECTIONS[4],SECTIONS[5],SECTIONS[6]]

    def put(ax,x,y,value,size=9,color=INK,weight='normal',serif=False,ha='left'):
        ax.text(x,y,str(value),va='top',ha=ha,fontsize=size,color=color,fontweight=weight,
                fontfamily=['Georgia','DejaVu Serif'] if serif else 'DejaVu Sans')

    def rect(ax,x,y,w,h,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='none'))

    def body(ax,y,value,x=43,width=102,size=9.2,color=INK):
        for line in textwrap.wrap(value,width=width,break_long_words=False):
            put(ax,x,y,line,size,color)
            y+=size+4
        return y

    def title(ax,y,value):
        put(ax,43,y,value,13,NAVY,'bold')

    def note(ax,y,value):
        body(ax,y,value,width=115,size=7.7,color=MUTE)

    def layout(page, description=None):
        f=plt.figure(figsize=(8.5,11),facecolor='white')
        a=f.add_axes([0,0,1,1]); a.set(xlim=(0,612),ylim=(792,0)); a.axis('off')
        tall=221 if page==1 else 83
        rect(a,0,0,612,tall,NAVY); rect(a,0,0,612,4,'#3cac99')
        put(a,43,23,BRAND,9,'#86d5c6','bold')
        put(a,43,43,TITLE,32 if page==1 else 23,'white',serif=True)
        if page==1:
            put(a,43,88,SUBTITLE,10,'#d5e5e9')
        else:
            put(a,43,108,papers[page-1],24,NAVY,serif=True)
            if description:
                body(a,146,description,width=109,size=8.9,color=MUTE)
        a.plot([43,569],[747,747],color=LINE,lw=.8)
        put(a,43,761,'HEALTHCARE RESEARCH  /  CAR[+2,+20]  /  TEST 2024–2026',7.5,MUTE)
        put(a,569,760,f'{page:02}',8.5,NAVY,'bold',ha='right')
        return f,a

    def table(a,y,header,rows,xs,h=20,size=8.4):
        rect(a,43,y,526,h,SOFT)
        for x,v in zip(xs,header):
            put(a,x,y+(h-size)/2-1,v,size,NAVY,'bold')
        for j,row in enumerate(rows):
            pos=y+(j+1)*h
            if j%2==1:
                rect(a,43,pos,526,h,'#f2f6f7')
            for x,v in zip(xs,row):
                put(a,x,pos+(h-size)/2-1,v,size,PALETTE.get(v,INK),'bold' if v in MODELS else 'normal')

    def axis(f,x,y,w,h,label):
        a=f.add_axes([x/612,(792-y-h)/792,w/612,h/792]); a.set_axisbelow(True)
        a.grid(axis='y',color=LINE,lw=.65); a.axhline(0,color='#798c9b',lw=.85)
        a.set_ylabel(label,fontsize=8,color=MUTE)
        a.tick_params(axis='both',labelsize=8,length=0,colors=MUTE,pad=5)
        for spine in a.spines.values():
            spine.set_visible(False)
        return a

    def legend(a,y,palette):
        for j,(name,color) in enumerate(palette.items()):
            x=43+j*141
            rect(a,x,y+4,12,4,color); put(a,x+17,y,name,8.8,MUTE)

    def save(pdf,f):
        pdf.savefig(f); plt.close(f)

    tick_years=['2024','2025','2026 (Partial)']
    inc_colors={'B−A':PALETTE['B'],'B*−A':PALETTE['B*']}
    metadata={'Title':f'{BRAND} | {TITLE}','Subject':SUBTITLE,'CreationDate':None,'ModDate':None}
    with PdfPages(out/PDF,metadata=metadata) as pdf:
        f,a=layout(1)
        for j,r in enumerate(results(cov,option='market').itertuples()):
            x=43+j*178
            rect(a,x,130,171,67,'#19394f')
            put(a,x+13,137,f'{r.events:,}',29,'white',serif=True)
            put(a,x+13,174,r.test,8.8,'#d5e5e9')
        put(a,43,246,SECTIONS[0],26,NAVY,serif=True)
        title(a,290,'Test Results')
        rows=[(c.test,c.events,c.issuers,c.training_history,f'{c.training_events:,}') for c in results(cov,option='market').itertuples()]
        table(a,316,['Period','Events','Issuers','Eligible training history','Training events'],rows,[49,227,286,351,482],24)
        note(a,421,'Model A, B and B* share the same events within each test and benchmark. The two benchmark options also have identical event IDs.')
        title(a,458,'Models and targets')
        body(a,483,'A uses 16 financial and structural controls. B adds the original five selected Loughran–McDonald (LM) features. B* adds all nine. Both targets cover CRSP sessions +2 to +20. The benchmarks are the S&P 500 price return and a cap-weighted Healthcare return.')
        title(a,553,'Saved predictions by test year')
        body(a,578,'Test 2024 uses the original archived scores and CARs. The original experiment used 2024 test outcomes in its strategy and estimator choices. Test 2025 and Test 2026 (Partial) use the saved annual expanding predictions.')
        body(a,646,'The final 2026 event is the May 27 call. Returns through June 30, 2026 provide its complete target window.')
        save(pdf,f)


        f,a=layout(2,'Spearman correlation between the saved model score and realized CAR[+2,+20].')
        legend(a,179,PALETTE)
        lim=max(.30,np.ceil(abs(perf.rank_ic).max()*10)/10)
        for j,option in enumerate(BENCH):
            x=43+j*265; put(a,x,205,BENCH[option],10,NAVY,'bold')
            d=axis(f,x+36,228,208,122,'Rank IC')
            for k,m in enumerate(MODELS):
                v=select(perf,option=option,model=m)
                d.bar(np.arange(3)+(k-1)*.21,v.rank_ic,width=.18,color=PALETTE[m]); d.set_ylim(-lim,lim)
            d.set_xticks(range(3),tick_years)
        note(a,374,'Figure 1. Matched events per benchmark are 204, 211 and 104 by test year. Rank IC is dimensionless. Table entries retain the original Test 2024 precision.')
        for j,option in enumerate(BENCH):
            y=406+157*j; put(a,43,y,BENCH[option],9.5,NAVY,'bold')
            rows=[(r.test,r.model,r.events,r.issuers,exact(r.rank_ic) if r.year==2024 else num(r.rank_ic),num(r.pearson))
                  for r in results(perf,option=option).itertuples()]
            table(a,y+16,['Period','Model','Events','Issuers','Rank IC','Pearson (suppl.)'],rows,[49,181,232,285,349,483],13,7.6)
        save(pdf,f)

        f,a=layout(3,'Subtract Model A’s Rank IC from the corresponding LM model’s Rank IC on identical events.')
        legend(a,184,inc_colors)
        for j,option in enumerate(BENCH):
            x=43+j*265; put(a,x,216,BENCH[option],10,NAVY,'bold')
            d=axis(f,x+39,242,200,193,'Δ Rank IC')
            for k,(contrast,color) in enumerate(inc_colors.items()):
                d.bar(np.arange(3)+(k-.5)*.3,select(inc,option=option,contrast=contrast).delta_rank_ic,width=.24,color=color)
            d.set_xticks(range(3),tick_years); d.set_ylim(-.05,.05)
        note(a,461,'Figure 2. N = 204, 211 and 104 events per benchmark by test year. The zero line represents the corresponding Model A rank correlation.')
        rows=[]
        for option in BENCH:
            for year in YEARS:
                v=results(inc,year,option).set_index('model')
                rows.append((BENCH[option],YEARS[year],v.events.iloc[0],delta(v.loc['B','delta_rank_ic']),delta(v.loc['B*','delta_rank_ic'])))
        table(a,504,['Benchmark','Period','Events','B−A','B*−A'],rows,[49,227,364,433,504],22,8.5)
        note(a,680,'The original strategy selection affects Test 2024 estimates. See Statistical Inference for the year-specific intervals and limitations.')
        save(pdf,f)

        for j,option in enumerate(BENCH):
            f,a=layout(4+j,BENCH[option]+' CAR[+2,+20]. Q1 is the lowest predicted-score group.')
            legend(a,179,PALETTE)
            span=max(3.,np.ceil(abs(qs.mean_car).max()*100))
            for k,year in enumerate(YEARS):
                y=211+142*k
                row=results(cov,year,option).iloc[0]
                put(a,43,y,f'{row.test}  /  {row.events} events, {row.issuers} issuers',9.5,NAVY,'bold')
                d=axis(f,108,y+22,442,69,'Mean CAR (%)')
                counts=[]
                for m in MODELS:
                    q=select(qs,year=year,option=option,model=m)
                    counts.append(q.events.tolist())
                    d.plot(q.quintile,q.mean_car*100,'-o',color=PALETTE[m],lw=1.55,ms=4.1)
                d.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5']); d.set_ylim(-span,span)
                n=' / '.join(map(str,counts[0])) if counts.count(counts[0])==3 else 'See quintile CSV for model-specific counts'
                v=results(perf,year,option).set_index('model').q5_minus_q1
                message=f'Q1–Q5 events {n}  ·  Q5−Q1 (pp)  A {num(v.A*100,2)}  /  B {num(v.B*100,2)}  /  B* {num(v["B*"]*100,2)}'
                note(a,y+112,message)
            note(a,659,'Figure 3. Mean event-level CAR by model score, in percent. Spreads use percentage points. Common y-axis scales apply across both benchmark pages.')
            body(a,700,'Holding windows can overlap. These groups describe individual event outcomes.',size=8.5)
            save(pdf,f)

        f,a=layout(6,'Available within-year quarterly Spearman Rank IC series from the saved annual evaluations.')
        legend(a,178,PALETTE)
        span=max(.35,np.ceil(abs(stability.rank_ic).max()*10)/10)
        for j,option in enumerate(BENCH):
            for k,year in enumerate((2025,2026)):
                x=43+k*265; y=208+j*220
                short='S&P 500' if option=='market' else 'Healthcare sector'
                put(a,x,y,f'{short} / {YEARS[year]}',9.1,NAVY,'bold')
                z=results(stability,year,option)
                d=axis(f,x+34,y+26,201,122,'Rank IC')
                for m in MODELS:
                    s=select(z,model=m)
                    d.plot(range(len(s)),s.rank_ic,'-o',color=PALETTE[m],lw=1.55,ms=4.2)
                v=select(z,model='A')
                d.set_xticks(range(len(v)),[f'{r.quarter[-2:]}{"*" if r.partial else ""}\nn={r.events}' for r in v.itertuples()]); d.set_ylim(-span,span)
        note(a,635,'Figure 4. Event counts appear below each quarter. * Q2 2026 covers only calls with complete targets through June 30. Test 2024 has no saved quarterly series for this report.')
        save(pdf,f)

        f,a=layout(7,'Test 2025 provides the available within-year pointwise paired 95% intervals.')
        for j,option in enumerate(BENCH):
            x=43+j*263; put(a,x,190,BENCH[option],9.8,NAVY,'bold')
            d=axis(f,x+35,218,203,138,''); d.axvline(0,color='#798c9b',lw=.85)
            d.set_xlabel('Δ Rank IC',color=MUTE,fontsize=8)
            d.set(ylim=(-.5,1.5),xlim=(-.10,.10),yticks=[0,1],yticklabels=['B*−A','B−A'])
            d.set_xticks([-.10,-.05,0,.05,.10],['−0.10','−0.05','0','0.05','0.10'])
            for k,m in enumerate(('B*','B')):
                for h,method in enumerate(('issuer_cluster','time_block')):
                    v=select(bands,year=2025,option=option,model=m,method=method).iloc[0]; y=k+(.11 if h==0 else -.11)
                    d.hlines(y,v.lower,v.upper,color=PALETTE[m],lw=1.7,ls='-' if h==0 else '--')
                    d.plot(v.point,y,marker='o' if h==0 else 's',color=PALETTE[m],mfc=PALETTE[m] if h==0 else 'white',ms=4.3)
        note(a,387,'Figure 5. Test 2025, 211 events, 61 issuer clusters and six nominal two-month blocks. 2,499 paired draws per method. Circles are issuer clusters. Squares are calendar blocks.')
        rows=[]
        for (option,model),v in results(bands,2025).groupby(['option','model'],sort=False):
            v=v.set_index('method'); line=lambda m:f'[{num(v.loc[m,"lower"])}, {num(v.loc[m,"upper"])}]'
            rows.append((BENCH[option],f'{model}−A',delta(v.point.iloc[0]),line('issuer_cluster'),line('time_block')))
        table(a,433,['Benchmark','Contrast','Point','Issuer 95% CI','Time-block 95% CI'],rows,[49,218,286,357,464],22,8.1)
        body(a,568,'Each Test 2025 interval includes zero. Intervals are exploratory, pointwise, and conditional on the saved predictions.',size=9.0)
        body(a,616,'Test 2024 reports point estimates because the original test outcomes informed strategy and estimator choices. Test 2026 (Partial) has three nominal time blocks, below the frozen minimum of five.',size=9.0)
        save(pdf,f)

        f,a=layout(8,'Sources, training cutoffs and the point-in-time limits of the reported predictions.')
        rows=[(r.test,r.first_call,r.last_call,r.training_cutoff_et[:10],f'{r.training_events:,}') for r in results(cov,option='market').itertuples()]
        table(a,177,['Period','First call','Last call','Training cutoff ET','Train n'],rows,[49,199,283,372,510],22,8.6)
        note(a,276,'The Test 2024 entry is the original training end date. Later cutoffs are at 16:00 ET on the listed date. The 2026 CRSP data cutoff is June 30.')
        titles=['Benchmark construction','Model and feature definitions','Label maturity and timing','Provenance and replication']
        notes=['S&P 500 adjustment uses SPRTRN. The Healthcare adjustment uses the original static GICS Healthcare basket and daily same-day cap weights. Both sum daily stock-minus-benchmark returns across sessions +2 through +20.',
               'A uses 16 controls. B adds the archive’s ordered LM5 per benchmark and B* adds LM9. Market A/B use Ridge α=1 and B* uses α=10. Sector A uses Ridge α=1, B uses Elastic Net α=0.05 with L1 ratio 0.7, and B* uses Ridge α=50. The two archived Q&A hedge and uncertainty columns coincide.',
               'The annual refits use prior-year calls with a known Day +20 close strictly before the annual cutoff. Early CAR is observable after Day +1 close. The original analysis does not establish real-time availability of every input.',
               '2024 predictions, CARs and cohorts trace to the original replication. 2025 and 2026 trace to the saved annual expansion. The provenance manifest contains SHA-256 checksums. The reporting script verifies sources and rebuilds aggregate tables without training.']
        for i,(heading,message) in enumerate(zip(titles,notes)):
            y=313+i*103; title(a,y,heading); body(a,y+24,message,width=101,size=8.9)
        save(pdf,f)


STYLE = '''
:root{--paper:#f0f5f6;--panel:#fff;--ink:#172f45;--muted:#496074;--line:#d6e3e8;--teal:#08766c;--soft:#e6f3ef;--navy:#112d43}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 "Segoe UI",system-ui,sans-serif}
a{color:#08695f;text-underline-offset:3px}a:focus-visible,summary:focus-visible,select:focus-visible,[tabindex="0"]:focus-visible{outline:3px solid #08766c;outline-offset:3px}.skip{position:absolute;left:10px;top:-100px;background:white;padding:9px;z-index:4}.skip:focus{top:10px}
.wrap{width:min(1180px,calc(100% - 72px));margin-inline:auto}.hero{background:var(--navy);color:white;border-top:5px solid #3cac99}.hero .wrap{padding:46px 0 42px}.brand{color:#86d5c6;font-size:12.5px;font-weight:750;letter-spacing:.15em;margin:0 0 17px}.hero h1{font:500 clamp(40px,6vw,75px)/1.02 Georgia,serif;letter-spacing:-.042em;margin:0}.intro{color:#d5e5e9;font-size:18px;margin:18px 0 27px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#456073;border:1px solid #456073;border-radius:12px;overflow:hidden}.stats>div{padding:16px 20px;background:#19394f}.stats strong{display:block;font:normal 34px/1 Georgia,serif;letter-spacing:-.025em}.stats span{display:block;margin-top:8px;color:#d5e5e9;font-size:13px}
nav{position:sticky;top:0;z-index:2;background:white;border-bottom:1px solid var(--line)}nav .wrap{display:flex;gap:4px;overflow:auto;padding:9px 0}nav a{padding:6px 12px;white-space:nowrap;border-radius:6px;text-decoration:none;font-size:14px;font-weight:650;color:var(--ink)}nav a:hover,nav a[aria-current]{background:var(--soft);color:var(--teal)}
main{padding:30px 0 42px}.section{scroll-margin-top:67px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:26px 31px;margin-bottom:22px}.section h2{font:500 clamp(27px,4vw,37px)/1.15 Georgia,serif;letter-spacing:-.027em;margin:0 0 11px;color:var(--navy)}h3{font-size:17px;margin:0 0 8px;color:var(--navy)}p{margin:.5em 0 .8em}.sub,figcaption{font-size:.87rem;color:var(--muted)}.columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}.columns>*{min-width:0}.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 17px}.tile{background:#f6f9fa;border:1px solid var(--line);border-radius:9px;padding:13px 15px}.tile h3{font:600 18px Georgia,serif}.tile strong{color:var(--teal);font-size:1.28rem}.tile p{font-size:.85rem;color:var(--muted);margin:.35em 0 .15em}.disclosure{margin-top:15px;background:#f5f9fa;border-left:3px solid #7d9dac;padding:10px 14px;font-size:.89rem}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:8px 16px;margin:15px 0 12px}.controls label{font-weight:650;font-size:.87rem;color:var(--navy)}select{display:block;font:inherit;color:var(--navy);background:white;border:1px solid #99aeb9;border-radius:6px;max-width:100%;padding:.45em .66em;margin-top:.25em}.legend{display:flex;gap:9px 19px;flex-wrap:wrap;font-size:.84rem;color:var(--muted);margin:8px 0 12px}.sw{display:inline-block;width:12px;height:5px;margin-right:5px;vertical-align:middle;border-radius:2px}
figure{margin:0;min-width:0;padding:13px 15px;border:1px solid var(--line);border-radius:10px;background:#fff}figure svg{display:block;width:100%;height:auto;font:12px "Segoe UI",sans-serif}figure figcaption{margin:.6em 0 .2em}.plot-wrap{max-width:870px;margin-inline:auto}.tip{min-height:2.1em;margin:.5em 0 .1em;background:#edf4f5;border-radius:5px;padding:.3em .57em;font-size:.83rem;color:var(--navy)}
.table-wrap{overflow:auto;margin:10px 0 15px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}thead{background:#edf4f5}td,th{text-align:left;padding:8px 12px;white-space:nowrap}tbody tr+tr{border-top:1px solid var(--line)}th[scope="col"]{color:var(--navy)}.caption{margin:.5em 0 0;color:var(--muted);font-size:.86rem}
details{border-top:1px solid var(--line);padding:11px 0;margin:7px 0}summary{cursor:pointer;color:var(--teal);font-size:.89rem;font-weight:700}details td{white-space:normal}code{overflow-wrap:anywhere}.downloads{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.86rem}footer{border-top:1px solid var(--line);color:var(--muted);font-size:13px;padding:17px 0 28px}
@media(max-width:810px){.wrap{width:calc(100% - 26px)}.hero .wrap{padding-block:30px}.stats,.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.section{padding:17px 15px}.columns{grid-template-columns:minmax(0,1fr)}.intro{font-size:1rem}figure{padding:10px}.tiles .tile:last-child{grid-column:1/-1}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
'''

CLIENT = r'''
const D=JSON.parse(document.getElementById('data').textContent);
const years=[2024,2025,2026], models=['A','B','B*'], bench={market:'S&P 500-adjusted',sector:'Healthcare sector-adjusted'};
const yearly={2024:'Test 2024',2025:'Test 2025',2026:'Test 2026 (Partial)'}, short=['2024','2025','2026 (Partial)'];
const color={A:'#335f91',B:'#08766c','B*':'#795b93','B−A':'#08766c','B*−A':'#795b93'};
const perf=D.annual_model_performance, inc=D.annual_incremental_rank_ic, qs=D.annual_quintile_returns, stb=D.annual_stability, bands=D.annual_inference;
const $=id=>document.getElementById(id), ref=(source,o,y)=>source.filter(r=>r.option===o&&(y==null||r.year===y));
const fmt=(x,d=4)=>x==null?'—':x.toFixed(d).replace('-','−');
const plus=x=>(x>0?'+':'')+fmt(x), car=x=>fmt(x*100,2), C=r=>`${r.events} events, ${r.issuers} issuers`;
function table(headers,rows){return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function mark(x,y,fill,info,bar){
 const glyph=bar?`<rect x="${x-bar.w/2}" y="${Math.min(y,bar.zero)}" width="${bar.w}" height="${Math.max(1,Math.abs(y-bar.zero))}" rx="1" fill="${fill}"/>`:`<circle cx="${x}" cy="${y}" r="4.7" fill="${fill}" stroke="white" stroke-width="1.2"/>`;
 return `<g tabindex="0" data-info="${info}" aria-label="${info}"><title>${info}</title>${glyph}<circle cx="${x}" cy="${y}" r="10" fill="transparent"/></g>`;
}
function bind(id){for(const el of $(id).querySelectorAll('[data-info]'))for(const e of ['mouseenter','focus','click'])el.addEventListener(e,()=>{$(id+'-tip').textContent=el.dataset.info;});}
function chart(id, data, labels, limit, unit, kind='line'){
 const wid=505,left=63,right=483, y=v=>130-v*95/limit, slot=(right-left)/(labels.length),
       x=i=>kind==='bar'?left+slot*(i+.5):left+12+i*(right-left-24)/(labels.length-1);
 const precision=limit>=1?1:limit<=.10?3:2;
 const axis=[-limit,-limit/2,0,limit/2,limit].map(v=>`<path d="M${left} ${y(v)}H${right}" stroke="${v===0?'#7d8d9a':'#d6e3e8'}"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${fmt(v,precision)}</text>`).join('');
 const p=data.map((s,j)=>{
  let line=kind==='line'?`<polyline fill="none" stroke="${color[s.key]}" stroke-width="2.15" points="${s.rows.map((r,i)=>`${x(i)},${y(r.value)}`).join(' ')}"/>`:'';
  return line+s.rows.map((r,i)=>mark(x(i)+(kind==='bar'?(j-(data.length-1)/2)*21:0),y(r.value),color[s.key],r.comment,kind==='bar'?{w:18,zero:y(0)}:null)).join('');
 }).join('');
 $(id).innerHTML=`<svg viewBox="0 0 ${wid} 263" aria-label="${unit}"><g fill="#496074">${axis}<text x="${left}" y="18">${unit}</text>${labels.map((l,i)=>`<text x="${x(i)}" y="247" text-anchor="middle">${l}</text>`).join('')}</g>${p}</svg>`;
 $(id+'-tip').textContent='Focus or hover over a mark for its value and sample size.';bind(id);
}
function ranking(){
 const y=Number($('rank-year').value);
 for(const o of Object.keys(bench)){
  const a=ref(perf,o);
  chart('rank-'+o,models.map(m=>({key:m,rows:a.filter(r=>r.model===m).map(r=>({value:r.rank_ic,comment:`${r.test} · ${m} · Rank IC ${r.rank_ic_exact} · ${C(r)}`}))})),short,.30,'Spearman Rank IC','bar');
  $('table-'+o).innerHTML=table(['Model','Rank IC','Pearson (suppl.)','Calls','Issuers'],ref(perf,o,y).map(r=>[r.model,y===2024?r.rank_ic_exact:fmt(r.rank_ic),fmt(r.pearson),r.events,r.issuers]));
 }
}
function incremental(){
 const o=$('inc-bench').value,a=ref(inc,o);
 $('inc-title').textContent=bench[o];
 chart('inc-plot',['B−A','B*−A'].map(m=>({key:m,rows:a.filter(r=>r.contrast===m).map(r=>({value:r.delta_rank_ic,comment:`${r.test} · ${m} · Δ Rank IC ${plus(r.delta_rank_ic)} · ${C(r)}`}))})),short,.05,'Δ Rank IC','bar');
 $('inc-table').innerHTML=table(['Test','Calls','Issuers','B−A','B*−A'],years.map(y=>{
  const r=ref(inc,o,y);return [yearly[y],r[0].events,r[0].issuers,...r.map(x=>plus(x.delta_rank_ic))];}));
}
function quintiles(){
 const o=$('q-bench').value,y=Number($('q-year').value),v=ref(qs,o,y),p=ref(perf,o,y);
 const limit=Math.max(3,Math.ceil(Math.max(...qs.map(r=>Math.abs(r.mean_car*100)))));
 $('q-title').textContent=`${bench[o]} · ${yearly[y]}`;
 chart('q-plot',models.map(m=>({key:m,rows:v.filter(r=>r.model===m).map(r=>({value:r.mean_car*100,comment:`${m} · Q${r.quintile} · mean CAR ${car(r.mean_car)}% · n=${r.events}`}))})),['Q1','Q2','Q3','Q4','Q5'],limit,'Mean realized CAR (%)');
 $('q-count').textContent=C(p[0]);
 $('q-table').innerHTML=table(['Model','Q1','Q2','Q3','Q4','Q5','Q5−Q1 (pp)','Shape'],models.map((m,i)=>[m,...v.filter(r=>r.model===m).map(r=>`${car(r.mean_car)}% (n=${r.events})`),car(p[i].q5_minus_q1),p[i].quintile_monotonicity]));
}
function stability(){
 const o=$('st-bench').value;
 for(const y of [2025,2026]){
  const a=ref(stb,o,y), key=a.filter(r=>r.model==='A'), id='st-'+y;
  $('st-title-'+y).textContent=`${bench[o]} · ${yearly[y]}`;
  chart(id,models.map(m=>({key:m,rows:a.filter(r=>r.model===m).map(r=>({value:r.rank_ic,comment:`${r.quarter} · ${m} · Rank IC ${fmt(r.rank_ic)} · ${C(r)}`}))})),key.map(r=>r.quarter.slice(-2)+(r.partial?'*':'')+' · n='+r.events),.40,'Quarterly Rank IC');
 }
}
function inference(){
 const o=$('ci-bench').value,y=Number($('ci-year').value), a=ref(bands,o,y), values=a[0].lower!=null;
 const canvas=$('ci-figure');canvas.hidden=!values;
 $('ci-title').textContent=`${bench[o]} · ${yearly[y]}`;
 const rows=['B','B*'].map(m=>{const p=a.filter(x=>x.model===m);return [`${m}−A`,plus(p[0].point),...(values?p.map(x=>`[${fmt(x.lower)}, ${fmt(x.upper)}]`):['—','—'])];});
 $('ci-table').innerHTML=table(['Contrast','Δ Rank IC','Issuer 95% CI','Time-block 95% CI'],rows);
 $('ci-sample').textContent=C(a[0])+'. '+(values?'2,499 paired draws per method. Six nominal two-month blocks.':y===2024?'Point estimates. The original 2024 test outcomes informed model selection.':'Point estimates. Three nominal time blocks fall below the frozen five-block threshold.');
 if(!values)return;
 const x=v=>75+(v+.10)*405/.2;
 const ticks=[-.10,-.05,0,.05,.10].map(v=>`<path d="M${x(v)} 27V166" stroke="${v===0?'#7d8d9a':'#d6e3e8'}"/><text x="${x(v)}" y="187" text-anchor="middle">${fmt(v,2)}</text>`).join('');
 const marks=a.map(r=>{const yc=(r.model==='B'?61:124)+(r.method==='issuer_cluster'?-7:7),
   info=`${r.contrast} · ${r.method==='issuer_cluster'?'Issuer cluster':'Time block'} · [${fmt(r.lower)}, ${fmt(r.upper)}] · n=${r.events}`;
   return `<path d="M${x(r.lower)} ${yc}H${x(r.upper)}" stroke="${color[r.model]}" stroke-width="2.1" ${r.method==='time_block'?'stroke-dasharray="5 4"':''}/>`+mark(x(r.point),yc,color[r.model],info,null);}).join('');
 $('ci-plot').innerHTML=`<svg viewBox="0 0 500 207" aria-label="Paired confidence intervals"><g fill="#496074">${ticks}<text x="75" y="16">Δ Rank IC</text><text x="17" y="65">B−A</text><text x="17" y="128">B*−A</text></g>${marks}</svg>`;
 $('ci-plot-tip').textContent='Focus or hover over a point for the paired interval.';bind('ci-plot');
}
const mapping={'rank-year':ranking,'inc-bench':incremental,'q-bench':quintiles,'q-year':quintiles,'st-bench':stability,'ci-bench':inference,'ci-year':inference};
for(const [id,fn] of Object.entries(mapping)) $(id).addEventListener('change',fn);
ranking();incremental();quintiles();stability();inference();
'''


def html_table(headers, rows):
    from html import escape
    head=''.join(f'<th scope="col">{escape(str(v))}</th>' for v in headers)
    body=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def make_html(out, data, manifest):
    from html import escape

    cov = results(data['annual_coverage'],option='market')
    def choices(id, label, options, current=None):
        body=''.join(f'<option value="{k}"{" selected" if k==current else ""}>{escape(v)}</option>' for k,v in options.items())
        return f'<label for="{id}">{label}<select id="{id}">{body}</select></label>'

    def plot(id, name, caption, selectable_title=False):
        heading=f'<h3 id="{id[:-5]}-title">{escape(name)}</h3>' if selectable_title else f'<h3>{escape(name)}</h3>'
        return (f'<figure>{heading}<div id="{id}"></div><p id="{id}-tip" class="tip" aria-live="polite"></p>'
                f'<figcaption>{escape(caption)}</figcaption></figure>')

    def legend(items):
        return '<div class="legend">'+''.join(f'<span><i class="sw" style="background:{v}"></i>{k}</span>' for k,v in items.items())+'</div>'

    stats=''.join(f'<div><strong>{r.events}</strong><span>{r.test} calls</span></div>' for r in cov.itertuples())
    stats+='<div><strong>9</strong><span>Original LM features</span></div>'
    tiles=''.join(f'<article class="tile"><h3>{r.test}</h3><strong>{r.events} calls</strong><p>{r.issuers} issuers · {r.training_events:,} training events</p><p>{escape(r.source)}</p></article>' for r in cov.itertuples())
    yoptions=YEARS.copy()
    pal=legend(PALETTE)
    rank_charts=''.join(plot('rank-'+o,name,'Figure 1. Spearman Rank IC. Calls per year are 204, 211 and 104.') for o,name in BENCH.items())
    rank_tables=''.join(f'<div><h3>{escape(name)}</h3><div id="table-{o}"></div></div>' for o,name in BENCH.items())
    stability=''.join(f'<figure><h3 id="st-title-{y}">{YEARS[y]}</h3><div id="st-{y}"></div><p id="st-{y}-tip" class="tip" aria-live="polite"></p><figcaption>Figure 4. Quarterly Rank IC. Event counts appear along the x-axis.</figcaption></figure>' for y in (2025,2026))
    coverage=html_table(['Test','First call','Last call','Events','Issuers','Training history','Training events','Training cutoff ET'],[
        (r.test,r.first_call,r.last_call,r.events,r.issuers,r.training_history,r.training_events,r.training_cutoff_et[:10]) for r in cov.itertuples()])
    specs=[]
    for o in BENCH:
        for m,recipe in manifest['model_definitions'][o].items():
            params=recipe['parameters']; estimator=f"{recipe['estimator']} α={params['alpha']:g}"
            if 'l1_ratio' in params:
                estimator=f"Elastic Net α={params['alpha']:g}, L1 ratio {params['l1_ratio']:g}"
            specs.append((BENCH[o],m,estimator,len(recipe['columns'])))
    spec_table=html_table(['Benchmark','Model','Estimator','Predictors'],specs)
    feature_fields=[('Original controls',manifest['controls']),('S&P 500 B LM5',manifest['model_definitions']['market']['B']['columns'][16:]),
                    ('Healthcare B LM5',manifest['model_definitions']['sector']['B']['columns'][16:]),('LM9',manifest['lm9'])]
    features=''.join(f'<p><strong>{escape(label)}</strong> <code>{escape(", ".join(values))}</code></p>' for label,values in feature_fields)
    nav=['Overview','Rank IC','Incremental Alpha','Quintiles','Stability','Inference','Coverage & Methodology']
    ids=['overview','rank','incremental','quintiles','stability','inference','methods']
    navigation=''.join(f'<a href="#{i}">{escape(n)}</a>' for i,n in zip(ids,nav))
    labels=['Model performance','Incremental Rank IC','Quintile returns','Quarterly Rank IC','Paired inference','Cohort coverage']
    links=''.join(f'<a href="{n}.csv">{label} CSV</a>' for n,label in zip(FILES,labels))
    links+=f'<a href="{PDF}">PDF report</a><a href="source_provenance.json">Source hashes</a><a href="methodology.md">Methodology</a><a href="summary_zh.md">中文摘要</a>'
    payload={}
    for name,original in data.items():
        frame=original.copy()
        if name=='annual_model_performance':
            frame['rank_ic_exact']=[exact(x) for x in frame.rank_ic]
        payload[name]=frame.astype(object).where(frame.notna(),None).to_dict(orient='records')
    payload=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{BRAND} | {TITLE}</title><style>{STYLE}</style></head>
<body><a class="skip" href="#content">Skip to test results</a><header class="hero"><div class="wrap"><p class="brand">{BRAND}</p><h1>{TITLE}</h1><p class="intro">{SUBTITLE}</p><div class="stats">{stats}</div></div></header>
<nav aria-label="Research sections"><div class="wrap">{navigation}</div></nav><main class="wrap" id="content">
<section class="section" id="overview"><h2>{SECTIONS[0]}</h2><h3>Test Results</h3><div class="tiles">{tiles}</div>
<div class="columns"><div><h3>Model definitions</h3><p>A uses 16 financial and structural controls. B adds five selected Loughran–McDonald (LM) features. B* adds all nine.</p></div><div><h3>Benchmarks and data</h3><p>CAR[+2,+20] is adjusted by either the S&amp;P 500 price return or a cap-weighted Healthcare return. The partial 2026 test uses complete windows through June 30.</p></div></div>
<p class="disclosure">Test 2024 reports original archived results. Those test outcomes informed strategy and estimator selection in the original experiment. Test 2025 and Test 2026 (Partial) use saved expanding-window predictions.</p></section>
<section class="section" id="rank"><h2>{SECTIONS[1]}</h2><p>Model scores are ranked against realized CAR within each benchmark and test year.</p>{pal}<div class="columns">{rank_charts}</div>
<div class="controls">{choices('rank-year','Test results table',yoptions,2024)}</div><div class="columns">{rank_tables}</div><p class="sub">Pearson correlation is supplementary. Test 2024 Rank IC is shown at the precision of the original archived result.</p></section>
<section class="section" id="incremental"><h2>{SECTIONS[2]}</h2><p>Each contrast subtracts Model A’s Rank IC on the same calls. Zero is the reference.</p><div class="controls">{choices('inc-bench','Benchmark',BENCH)}</div>
{legend({'B−A':PALETTE['B'],'B*−A':PALETTE['B*']})}<div class="plot-wrap">{plot('inc-plot',BENCH['market'],'Figure 2. Δ Rank IC within each test year.',True)}</div><div id="inc-table"></div></section>
<section class="section" id="quintiles"><h2>{SECTIONS[3]}</h2><p>Groups are formed from model scores. Q1 contains the lowest scores. The vertical scale is shared across selections.</p>
<div class="controls">{choices('q-year','Test year',yoptions,2025)}{choices('q-bench','Benchmark',BENCH)}</div>{pal}<p id="q-count" class="sub"></p><div class="plot-wrap">{plot('q-plot','Predicted-score groups','Figure 3. Event-level mean realized CAR by predicted-score quintile.',True)}</div><div id="q-table"></div>
<p class="sub">Each cell shows the mean and call count. The spread is Q5 minus Q1 in percentage points. Return windows may overlap.</p></section>
<section class="section" id="stability"><h2>{SECTIONS[4]}</h2><p>The saved annual series covers 2025 and partial 2026.</p><div class="controls">{choices('st-bench','Benchmark',BENCH)}</div>{pal}<div class="columns">{stability}</div><p class="sub">* Q2 2026 includes only events with complete CAR windows through June 30. Test 2024 has no saved quarterly series for this report.</p></section>
<section class="section" id="inference"><h2>{SECTIONS[5]}</h2><p>Available paired intervals are pointwise, exploratory and conditional on the saved predictions.</p><div class="controls">{choices('ci-year','Test year',yoptions,2025)}{choices('ci-bench','Benchmark',BENCH)}</div>
<h3 id="ci-title"></h3><p id="ci-sample" class="sub" aria-live="polite"></p><div id="ci-figure" class="plot-wrap">{plot('ci-plot','Paired confidence intervals','Figure 5. Test 2025 issuer-cluster intervals are solid. Calendar two-month block intervals are dashed.')}</div><div id="ci-table"></div><p class="sub">All Test 2025 intervals include zero. The 2024 and partial 2026 entries are descriptive point estimates.</p></section>
<section class="section" id="methods"><h2>{SECTIONS[6]}</h2>{coverage}<p class="sub">The 2024 date is the original training end date. Later refit cutoffs are at 16:00 ET. Coverage counts are identical across benchmarks.</p>
<details><summary>Benchmarks and label maturity</summary><p>SPRTRN is the S&amp;P 500 price return. The Healthcare series uses dated eligible members of the original static GICS Healthcare basket and same-day capitalization weights. Both targets sum daily stock-minus-benchmark returns from CRSP session +2 through +20. Annual training retains prior-year labels only if Day +20 closes before the refit cutoff. The prediction convention starts after Day +1 close, when Early CAR is known. The original workflow does not establish real-time availability of every predictor.</p></details>
<details><summary>Frozen model and feature specifications</summary>{spec_table}{features}<p>The original Q&amp;A hedging and uncertainty variables coincide in the archived data. The original candidate and fitted feature conventions are preserved. Raw model scores are rank-scale predictions.</p></details>
<details><summary>Verified sources and report downloads</summary><p>Test 2024 uses the original prediction Parquet, native metrics and original test split. The remaining years use the saved annual expanding prediction ledger, refit audit and within-year aggregates. The report builder verifies SHA-256 inputs and checks each year independently.</p><p class="downloads">{links}</p></details></section></main>
<footer><div class="wrap">HEALTHCARE RESEARCH / CAR[+2,+20] / TEST 2024–2026</div></footer><script id="data" type="application/json">{payload}</script><script>{CLIENT}</script></body></html>'''
    (out/'dashboard.html').write_text(html,encoding='utf-8')


def make_notes(out,data):
    docs = r'''# Healthcare Linguistic Alpha Research

## Saved predictions and test years

The report evaluates the original A/B/B* model definitions against S&P 500-adjusted and Healthcare sector-adjusted CAR[+2,+20]. Each test year, model and benchmark is scored on its own saved predictions and realized labels. For each test year, the two benchmark cohorts have identical event IDs. For each benchmark and year, the three models have identical event IDs and realized outcomes.

- **Test 2024** uses `outputs/healthcare_original_pierre_replication/original_test_predictions.parquet`. The original prediction rows, event IDs, Day 0 and realized targets agree exactly with the original test split and option-specific archive tables. The six Spearman and Pearson correlations agree at full saved precision with `native_model_metrics.csv`. The original experiment used 2024 test outcomes to compare estimators and strategies. The S&P 500 route also used these outcomes for individual feature screening. This year represents the original test and model-selection process. Its cohort and return labels predate the later CAR and financial-feature extensions.
- **Test 2025** and **Test 2026 (Partial)** use the `prediction` and `car_2_20` columns in `outputs/healthcare_expanding_oos/matched_predictions_local.csv`, filtered to their call-year rows. The saved expanding refits used 3,143 and 3,354 training calls respectively, at 16:00 ET on January 17, 2025 and January 22, 2026. New training observations needed a known Day +20 close strictly before the associated cutoff. The 2026 test ends at the May 27 earnings call, with a June 30 CRSP return cutoff. The archive for Test 2024 used 2,939 training calls through 2023-12-31.

## Models and metrics

Model A uses 16 financial and structural controls. Model B adds the selected five LM features in their original benchmark-specific order. Model B* uses all nine LM features. The manifest records the exact ordered inputs and estimator parameters. The original winner and feature choices are retained for later saved predictions. Training uses one-based average CAR ranks and original train-only preprocessing for transformations, winsorization, replacement of missing predictors and scaling. The original Q&A hedging and uncertainty variables are identical in the archived data. The model definitions include Early CAR `car_-1_1`, observable after Day +1 close. The event analysis does not establish that the full predictor set or an execution price was available at that time.

Both benchmark targets sum daily stock-minus-benchmark returns over CRSP sessions +2 to +20. The S&P 500 convention uses `SPRTRN`. The Healthcare benchmark uses the originally validated static GICS Healthcare basket, dated index membership and contemporaneous market-cap weights. The report preserves the realized targets as saved.

Rank IC is Spearman correlation between model scores and realized CAR. Pearson correlation is supplementary. The yearly increments are B−A and B*−A, each computed within a benchmark on matched events. CSV values retain floating-point precision. The report shows the six original Test 2024 Rank IC values at full archive precision. Other display values are rounded for readability.

Within each year and benchmark, quantiles divide predictions into five groups with `pandas.qcut`, preserving score ties. Q1 is the lowest group. The table reports each group's count, mean, median and sample SD of realized CAR. Q5−Q1 is the difference in mean CAR, shown in percentage points. Overlapping windows make these event-level distributions distinct from strategy returns. Quarterly Spearman IC is verified against the expanding source for 2025 and partial 2026, with no Test 2024 quarterly series in this report.

Test 2025 paired 95% intervals are retained from the saved within-year annual expanding LM contrasts and checked against independently recalculated point estimates. Each method has 2,499 valid draws. The issuer bootstrap resamples GVKEYs and the time bootstrap uses overlapping two-calendar-month blocks. The 2025 sample has 61 issuers and six nominal blocks. These are exploratory pointwise intervals conditional on the saved predictions. Test 2024 is descriptive given historical test-based selection. Partial 2026 has three nominal blocks, below the minimum of five.

## Reproduce

Run from the repository root. Use a Python environment with pandas, NumPy, SciPy and PyArrow for source verification. Use a Python environment with pandas, NumPy and Matplotlib to render the reports. The report uses Georgia headings and the Phase 1 visual palette and layout. No modeling libraries are needed by this builder.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe outputs/healthcare_final_research/build_final_report.py freeze
python outputs/healthcare_final_research/build_final_report.py render
```

Pass the same `--out` directory to both commands to rebuild in a fresh destination. The renderer verifies the hashes of all six frozen tables before rendering. `source_provenance.json` lists each source relative to the repository, its SHA-256, the frozen feature recipes and the aggregate hashes. The dashboard contains only derived aggregate tables. The licensed prediction and event ledgers remain in their original local directories.
'''
    (out/'methodology.md').write_text(docs,encoding='utf-8')
    p=data['annual_model_performance']; d=data['annual_incremental_rank_ic']; c=data['annual_coverage']
    zh={2024:'Test 2024',2025:'Test 2025',2026:'Test 2026 (Partial)'}
    labels={'market':'标普 500 调整','sector':'医疗保健行业调整'}
    lines=['# Healthcare 语言因子评估','',
           'Test 2024 直接使用原始复现目录中的 `original_test_predictions.parquet`，其预测与原生模型指标、样本清单、原始 CAR 逐项核对。Test 2025 与 Test 2026 (Partial) 使用已保存的年度扩展窗口预测。','',
           '| 年度 | 场次 | 发行人 | 训练场次 |','|---|---:|---:|---:|']
    for r in results(c,option='market').itertuples():
        lines.append(f'| {zh[r.year]} | {r.events} | {r.issuers} | {r.training_events:,} |')
    lines+=['','## 年度 Rank IC 与 LM 增量','',
            '| 基准 | 年度 | Model A | Model B | Model B* | B−A | B*−A |','|---|---|---:|---:|---:|---:|---:|']
    for o in BENCH:
        for year in YEARS:
            score=results(p,year,o); effects=results(d,year,o)
            vals=[exact(v) if year==2024 else num(v) for v in score.rank_ic]
            lines.append('| '+' | '.join([labels[o],zh[year],*vals,*map(delta,effects.delta_rank_ic)])+' |')
    lines+=['','Test 2025 的全部配对 95% 区间均包含零。两种方法各有 2,499 次有效重抽样。原始 Test 2024 的结果曾参与历史策略与估计器筛选。Test 2026 (Partial) 只有三个名义双月块。后两种限制均按对应年度披露。2026 的 CRSP 截止日期为 2026-06-30，最晚纳入 2026-05-27 的电话会议。','',
            '[交互式仪表板](dashboard.html) · [年度表现](annual_model_performance.csv) · [年度增量](annual_incremental_rank_ic.csv) · [配对区间](annual_inference.csv) · [方法及复现](methodology.md)']
    (out/'summary_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','render'))
    parser.add_argument('--out',type=Path,default=HERE)
    args=parser.parse_args()
    out=args.out.resolve()
    verify(out==HERE or HERE in out.parents or (out!=ROOT and ROOT not in out.parents),
           'Use this experiment directory or a fresh external destination')
    if args.action=='freeze':
        freeze(out)
    else:
        manifest=json.loads((out/'source_provenance.json').read_text(encoding='utf-8'))
        verify(all(digest(out/f'{n}.csv')==manifest['aggregate_sha256'][n] for n in FILES), 'Frozen aggregate changed')
        data={name:read(out/f'{name}.csv') for name in FILES}
        make_pdf(out,data,manifest)
        make_html(out,data,manifest)
        make_notes(out,data)


if __name__=='__main__':
    main()

