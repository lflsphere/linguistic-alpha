"""Run the frozen Healthcare window and horizon experiments with temporal selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet, Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATIC = HERE.parent / 'healthcare_extended_oos'
EARLIER = 'ffee3b4a2caaa5c450b2ffcda353be70c229e9ac'
PRESERVATION = '6c1363f'
WINDOWS = ('Expanding', 'Rolling 5Y')
OPTIONS = ('S&P 500', 'Healthcare')
YEARS = (2024, 2025, 2026)
HORIZONS = (5, 20)
MODELS = ('A', 'B', 'B*')
CONTROLS = ('sue', 'surprise_pct', 'vol_pre', 'car_-1_1', 'mom', 'bm', 'size', 'dispersion',
            'avg_answer_words', 'words_prepared', 'words_questions', 'words_answers', 'n_questions',
            'n_answers', 'n_analysts', 'prior_calls_observed')
LM = ('tone_prepared', 'tone_answers', 'hedging_prepared', 'uncertainty_answers', 'tone_gap',
      'tone_prepared_chg', 'tone_answers_chg', 'hedging_answers_chg')
ALGORITHMS = {'Ridge 1': ('ridge', 1.), 'Ridge 10': ('ridge', 10.), 'Ridge 50': ('ridge', 50.),
              'Elastic Net .05': ('elastic', .05)}
MIN_GAIN = .001
SEED = 18431
DRAWS = 999
BLOCK = 2
TZ = 'America/New_York'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def table(out, name, rows):
    pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False, float_format='%.17g')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def declare(out):
    git=['git','--no-optional-locks','-c',f'safe.directory={ROOT.as_posix()}']
    revision=subprocess.check_output([*git,'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    need(revision.startswith(PRESERVATION), 'Unexpected preservation baseline')
    subprocess.run([*git,'diff','--exit-code',EARLIER,'--','outputs/healthcare_extended_oos',
                    'outputs/sp500_car_extension','outputs/sp500_financial_extension'],cwd=ROOT,check=True)
    spec = dict(status='Post-hoc retrospective reconstruction. Outer outcomes never enter new model selection.',
        protected_baseline=EARLIER, preservation_commit=revision, test_years=YEARS, benchmark_options=OPTIONS,
        horizons_sessions={str(h):[2,h] for h in HORIZONS},
        schedules={'Expanding':'2010 through previous calendar year', 'Rolling 5Y':'Five calendar years before test year'},
        controls=CONTROLS, lm=LM,
        history_count='Prior calls of the same PERMNO observed before current call, irrespective of eventual CAR availability. Replaces target-availability-derived count.',
        native_source='Native option tables through 2024. Saved archive labels retain legacy coverage and delisting rules. Preserve native 2024 return vintage.',
        extension_source='2025 onward: verified snapshot/link/membership rules. Validated SPRTRN targets. Rebuild sector short horizon from original benchmark and CIZ stock returns. Reconcile legacy short and extended long before scoring.',
        clock='CRSP trading dates, original 16:00 ET Day 0 rule. Wait until 16:00 ET of Day +1 plus 1 microsecond. The 16:00 bound is conservative for early closes. Predict prior to Day +2 session open (09:30 ET). Close-to-close CAR begins at the preceding close and cannot be traded in full after this signal.',
        eligibility='Require valid early CAR and complete calendar endpoints. Refits use the earliest eligible signal across horizons and benchmarks for the test year. Strict Day +h close before refit for training. Train year strictly earlier than test year.',
        preprocessing='Fit original log, 1st/99th-percentile winsorization, training median, scaling and unnormalized average target midranks separately within each inner fold and final refit. Fold all-missing inputs contribute a zero standardized column.',
        algorithms={k:{'family':family,'alpha':a,**({'l1_ratio':.7,'max_iter':5000} if family=='elastic' else {})}
                    for k,(family,a) in ALGORITHMS.items()},
        inner_folds='Last three prior calendar years for validation. Expand from each schedule start year. Purge training labels ending at or after the earliest validation signal. Validation labels strictly before the outer refit. Require 100 training events / 25 issuers and 50 validation events / 20 issuers in every fold.',
        selection='Mean inner-fold Spearman Rank IC. A and B* choose estimators independently. B compares greedy forward and jointly fitted positive univariate LM subsets against A. Require >0.001 mean IC gain. Greedy steps also require >0.001. Ties within 1e-12 follow declared feature and estimator order, then prefer the shorter subset and forward route. Empty selection uses the exact A model.',
        evaluation='Raw Spearman and supplementary Pearson by test year. Match schedules by event within horizon. For direct horizon comparisons score both predictions on the intersection, against each respective target. No cross-year pooling.',
        quintiles='Prediction and factor sorts use qcut, preserve ties, require five distinct groups, 50 events and 20 issuers. Report means, counts, SD and Q5 minus Q1. No strategy returns.',
        factors='Development discovery uses mature eligible pre-2024 events, independent of test factor results. Evaluate LM8 individually. Quarterly gate 25 pairs/15 issuers. Prespecified conditional groups SUE >= 0 vs < 0 and prevol above vs at/below pre-2024 median, using observed predictors only. Correlation heatmaps use development and each test year separately.',
        inference='999 pointwise paired percentile draws by firm and overlapping noncircular two-month calendar blocks. Require 50 events / 30 issuers plus five nominal blocks for time inference. Partial 2026 has issuer-only intervals where feasible. Intervals conditional on fitted scores; no confirmatory multiple testing claims.',
        importance='Signed standardized-input coefficient divided by training sample size, plus absolute coefficient share. Joint fits, descriptive rather than causal interpretation.',
        seed=SEED)
    path=out/'frozen_research_spec.json'
    saved=json.dumps(spec,ensure_ascii=False,indent=2)+'\n'
    if path.exists():
        need(path.read_text(encoding='utf-8')==saved,'Frozen specification differs')
    else:
        path.write_text(saved,encoding='utf-8')


def offset(dates, cal, amount):
    i=cal.get_indexer(pd.to_datetime(dates).astype('datetime64[ns]'))
    j=i+amount
    ok=(i>=0)&(j>=0)&(j<len(cal))
    values=np.full(len(i),np.datetime64('NaT','ns'),dtype='datetime64[ns]')
    values[ok]=cal.to_numpy()[j[ok]]
    return pd.Series(values,index=dates.index)


def stamp(days, h, m=0):
    return (days+pd.to_timedelta(f'{h:02}:{m:02}:00')).dt.tz_localize(TZ).dt.tz_convert('UTC')


def timeline(f, calendar, vintage):
    f=f.copy()
    f['day1']=offset(f.day0,calendar,1)
    f['day1_close_bound']=stamp(f.day1,16)
    f['signal']=f.day1_close_bound+pd.to_timedelta('1us')
    f['day2_open']=stamp(offset(f.day0,calendar,2),9,30)
    f['timely']=f.clock.lt(f.signal)&f.signal.lt(f.day2_open)&f['car_-1_1'].notna()
    f['year']=f.call_date.dt.year
    for h in HORIZONS:
        f[f'mature_{h}']=stamp(offset(f.day0,calendar,h),16)
        f[f'eligible_{h}']=f.timely&f[f'mature_{h}'].notna()&f[f'car_2_{h}'].notna()
    f['vintage']=vintage
    return f


def prior_calls(health):
    history=health[['transcriptid','permno','clock']].sort_values(['permno','clock','transcriptid']).copy()
    need(history.clock.notna().all(), 'Missing call clock in history')
    # Simultaneous records cannot contribute to each other's count.
    history['count']=history.groupby('permno').cumcount()
    return history.assign(
        count=history.groupby(['permno','clock'])['count'].transform('min')).set_index('transcriptid')['count']


def independent_short(events, calendar, stock, benchmark):
    """Independent four-session join and sum for the new Healthcare target."""
    i=calendar.get_indexer(events.day0)
    inside=(i>=0)&(i+5<len(calendar))
    chosen=events.loc[inside]
    days=calendar[(i[inside,None]+np.arange(2,6)).ravel()]
    pairs=pd.MultiIndex.from_arrays([np.repeat(chosen.permno.to_numpy(),4),days])
    ret=stock.set_index(['permno','date']).ret.reindex(pairs).to_numpy(float)
    increments=(ret-benchmark.reindex(days).to_numpy(float)).reshape(-1,4)
    result=pd.Series(np.nan,index=events.index)
    result.loc[inside]=increments.sum(axis=1)
    return result


def inputs(out):
    s=load_module('static_source',STATIC/'evaluate.py')
    native=load_module('pinned_native',s.ORIGINAL/'sector_model_b.py')
    s.check_inputs()
    health,_=s.observed_frame()
    raise_if_fail=lambda f: need(f.hedging_answers.isna().equals(f.uncertainty_answers.isna()) and
                                f.hedging_answers.equals(f.uncertainty_answers), 'LM duplicate mismatch')
    raise_if_fail(health)
    old_index=s.RAW/'crsp_dsi.parquet'; ciz_index=s.CIZ/'crsp_dsp500_v2.parquet'
    l=pd.read_parquet(old_index,columns=['date','sprtrn']); l['date']=pd.to_datetime(l.date).astype('datetime64[ns]')
    c=pd.read_parquet(ciz_index,columns=['caldt','sprtrn']); c['caldt']=pd.to_datetime(c.caldt).astype('datetime64[ns]')
    old=pd.DatetimeIndex(l.date); cal=pd.DatetimeIndex(c.caldt)
    need(old.is_unique and cal.is_unique and old.is_monotonic_increasing and cal.is_monotonic_increasing and
         old.isin(cal).all() and cal[-1]==pd.Timestamp('2026-06-30'),'Unexpected CRSP calendar')
    known=pd.read_csv(STATIC/'healthcare_sector_extended_car.csv',parse_dates=['day0'],float_precision='round_trip')
    original_sector=pd.read_parquet(s.NATIVE/'health_sector_daily.parquet',columns=['date','sector_ret'])
    sector=pd.read_csv(STATIC/'sector_benchmark_daily.csv',parse_dates=['date'],float_precision='round_trip').set_index('date').sector_return
    original_sector['date']=pd.to_datetime(original_sector.date).astype('datetime64[ns]')
    old_stock=s.stocks(s.RAW/'crsp_dsf_2024.parquet',False,set(health.permno))
    new_stock=pd.concat([s.stocks(s.CIZ/f'crsp_dsf_v2_{y}.parquet',True,set(health.permno)) for y in (2025,2026)],ignore_index=True)
    future=s.eligible_new(health,known,s.membership(s.CIZ,True))
    future=future.loc[future.verified&future.member].copy()
    known=known.set_index('transcriptid')
    merged_days=s.assign_day0(future.clock,cal,future.call_date).day0
    comparable=merged_days.notna()
    need(merged_days[comparable].equals(future.transcriptid.map(known.day0)[comparable]),'Extended Day 0 mismatch')
    past_count=prior_calls(health)
    sources=[s.EXTENDED,s.CAR,s.INPUT,old_index,ciz_index,s.RAW/'crsp_dsf_2024.parquet',
             *[s.CIZ/f'crsp_dsf_v2_{y}.parquet' for y in (2025,2026)],
             STATIC/'healthcare_sector_extended_car.csv',STATIC/'sector_benchmark_daily.csv',s.NATIVE/'health_sector_daily.parquet',
             STATIC/'sector_benchmark_validation.csv',s.ARCHIVE/'provenance.json',s.ORIGINAL/'sector_model_b.py',
             s.CIZ/'crsp_msp500list_v2.parquet',s.RAW/'ciq_transcript_detail.parquet',
             *[d/p for d in (s.RAW,s.CIZ) for p in ('ciq_wrds_gvkey.parquet','crsp_ccmxpf_lnkhist.parquet')]]
    reconciled=[]
    by_option={}
    def agree(reference, calc, name, tol, full=True):
        known=reference.notna(); observed=calc.notna(); both=known&observed
        err=abs(reference[both]-calc[both]); n=len(err)
        need(n>0 and err.max()<=tol and (known.equals(observed) if full else known[both].sum()==known.sum()),name)
        reconciled.append(dict(check=name,events=n,max_abs_error=err.max(),tolerance=tol))
    for i,option in enumerate(OPTIONS,1):
        path=s.NATIVE/f'model_table_opt{i}.parquet'; sources.append(path)
        a=pd.read_parquet(path)
        a['call_date']=pd.to_datetime(a.call_date).astype('datetime64[ns]')
        a['day0']=pd.to_datetime(a.day0).astype('datetime64[ns]')
        a=a.loc[a.call_date.dt.year.between(2010,2024)].copy()
        a['clock']=s.call_timestamp_utc(a.call_date,a.call_time_utc)
        need(s.assign_day0(a.clock,old,a.call_date).day0.equals(a.day0),'Native Day 0 mismatch')
        raise_if_fail(a)
        hist=a.loc[a.call_date.dt.year.eq(2024)]
        benchmark=l.set_index('date').sprtrn if i==1 else original_sector.set_index('date').sector_ret
        short=s.calculate(hist,old,old_stock,benchmark,2,5).value
        if i==1:
            agree(hist.car_2_5,short,'Legacy 2024 S&P 500 short',1e-10)
        else:
            agree(hist.car_2_5,short,'Legacy 2024 Healthcare short',1e-10)
        f=future.copy(); f['day0']=merged_days
        new_bench=c.set_index('caldt').sprtrn if i==1 else sector
        for h in HORIZONS:
            name=f'car_2_{h}'
            rebuilt=s.calculate(f,cal,new_stock,new_bench,2,h)
            if i==2:
                f[name]=rebuilt.value
                if h==20:
                    agree(f.transcriptid.map(known.sector_car_2_20),f[name],'CIZ Healthcare long against frozen extension',1e-10)
                    f[name]=f.transcriptid.map(known.sector_car_2_20)
            else:
                agree(f[name],rebuilt.value,f'CIZ S&P 500 horizon {h}',1e-10)
        if i==2:
            agree(f.car_2_5,independent_short(f,cal,new_stock,sector),'CIZ Healthcare short independent four-session join',1e-12)
            f['car_-1_1']=f.transcriptid.map(known.sector_early_car)
        by_option[option]=pd.concat([timeline(a,old,'Legacy original'), timeline(f,cal,'CIZ extension')],ignore_index=True)
        g=by_option[option]
        g['gvkey']=g.gvkey.astype('string').str.zfill(6)
        g['prior_calls_observed']=g.transcriptid.map(past_count)
        need(g.transcriptid.is_unique and g.prior_calls_observed.notna().all(),'Unmatched predictor history')
    sources.extend([STATIC/'evaluate.py',ROOT/'outputs/sp500_car_extension/extend_car.py'])
    provenance={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    payload={'source_sha256':provenance,'analysis_source_sha256':{p:digest(HERE/p) for p in ('run_comparison.py','research_metrics.py')}}
    (out/'data_provenance.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    table(out,'target_validation',reconciled)
    table(out,'lm_redundancy',[dict(source='Validated Healthcare feature table',calls=len(health),
                                    joint_missing=int(health.hedging_answers.isna().sum()),
                                    equal_missing=True,equal_observed=True,dropped='hedging_answers',kept='uncertainty_answers')])
    return native,by_option,provenance


@dataclass(frozen=True)
class Fold:
    x: np.ndarray
    predict_x: np.ndarray
    ranks: np.ndarray
    val_ranks: np.ndarray


def arrays(native, train, predict):
    columns=[*CONTROLS,*LM]
    x=train[columns].copy()
    z=predict[columns].copy()
    empty=x.columns[x.isna().all()]
    if not empty.empty:
        x.loc[:,empty]=0.
        z.loc[:,empty]=0.
    x,z,_,_=native.preprocess(x,z,columns)
    need(np.isfinite(x).all() and np.isfinite(z).all(),'Nonfinite model inputs after train-only preprocessing')
    return x,z


def estimator(name):
    family,a=ALGORITHMS[name]
    return Ridge(alpha=a) if family=='ridge' else ElasticNet(alpha=a,l1_ratio=.7,max_iter=5000)


def choose_best(rows, score):
    best=max(score(x) for x in rows)
    return next(x for x in rows if score(x)>=best-1e-12)


class InnerSearch:
    def __init__(self, folds, context):
        self.folds=folds
        self.context=context
        self.cache={}
        self.estimator_rows=[]
        self.path_rows=[]

    def evaluate(self, selected):
        subset=tuple(name for name in LM if name in selected)
        if subset not in self.cache:
            columns=list(range(len(CONTROLS)))+[len(CONTROLS)+LM.index(name) for name in subset]
            scores=[]
            for algo in ALGORITHMS:
                fold_scores=[]
                for f in self.folds:
                    pred=estimator(algo).fit(f.x[:,columns],f.ranks).predict(f.predict_x[:,columns])
                    ranks=stats.rankdata(pred)
                    ic=0. if np.ptp(ranks)==0 else float(np.corrcoef(ranks,f.val_ranks)[0,1])
                    fold_scores.append(ic)
                mean=float(np.mean(fold_scores))
                scores.append((algo,mean))
                self.estimator_rows.append(dict(**self.context,features='|'.join(subset) or '(controls only)',
                                           estimator=algo,mean_rank_ic=mean,fold_1=fold_scores[0],
                                           fold_2=fold_scores[1],fold_3=fold_scores[2]))
            name,value=choose_best(scores,lambda x:x[1])
            self.cache[subset]=(name,value)
        return self.cache[subset]

    def record(self, route, step, feature, subset, score, gain, accepted):
        self.path_rows.append(dict(**self.context,route=route,step=step,feature=feature,
                                  features='|'.join(subset),inner_rank_ic=score,gain=gain,accepted=accepted))

    def select(self):
        algo,baseline=self.evaluate(())
        chosen=[]; current=baseline
        single=tuple(m for m in LM if self.evaluate((m,))[1]>baseline+MIN_GAIN+1e-12)
        for m in LM:
            v=self.evaluate((m,))[1]
            self.record('Univariate screen',1,m,(m,),v,v-baseline,m in single)
        for step in range(len(LM)):
            candidates=[m for m in LM if m not in chosen]
            winner=choose_best(candidates,lambda m:self.evaluate([*chosen,m])[1])
            new=self.evaluate([*chosen,winner])[1]
            accept=new>current+MIN_GAIN+1e-12
            for m in candidates:
                v=self.evaluate([*chosen,m])[1]
                self.record('Forward',step+1,m,[*chosen,m],v,v-current,m==winner and accept)
            if not accept:
                break
            chosen.append(winner); current=new
        routes=[(0,'Controls',(),algo,baseline)]
        for priority,route,subset in [(1,'Forward',tuple(chosen)),(2,'Screened combination',single)]:
            if subset:
                a,v=self.evaluate(subset)
                self.record(route,0,'',subset,v,v-baseline,v>baseline+MIN_GAIN+1e-12)
                if v>baseline+MIN_GAIN+1e-12:
                    routes.append((priority,route,subset,a,v))
        routes.sort(key=lambda row:(len(row[2]),row[0]))
        _,route,subset,b,b_score=choose_best(routes,lambda row:row[-1])
        star,star_ic=self.evaluate(LM)
        return {'A':('Controls',(),algo,baseline),'B':(route,subset,b,b_score),'B*':('All LM8',LM,star,star_ic)}


def fold_history(train, year, h, native, context):
    folds,audit=[],[]
    for y in range(year-3,year):
        val=train.loc[train.year.eq(y)].copy()
        cutoff=val.day1_close_bound.min()
        history=train.loc[train.year.lt(y)]
        use=history.loc[history[f'mature_{h}'].lt(cutoff)]
        need(len(use)>=100 and use.gvkey.nunique()>=25 and len(val)>=50 and val.gvkey.nunique()>=20,
             f'Insufficient inner fold: {context}, {y}')
        need(set(use.transcriptid).isdisjoint(val.transcriptid),'Inner fold overlap')
        x,z=arrays(native,use,val)
        folds.append(Fold(x,z,stats.rankdata(use[f'car_2_{h}']),stats.rankdata(val[f'car_2_{h}'])))
        audit.append(dict(**context,validation_year=y,inner_cutoff=cutoff,train_events=len(use),
                          train_issuers=use.gvkey.nunique(),validation_events=len(val),validation_issuers=val.gvkey.nunique(),
                          purged=len(history)-len(use),max_fit_maturity=use[f'mature_{h}'].max(),
                          max_validation_maturity=val[f'mature_{h}'].max()))
    return folds,audit


def outer_train(g, year, horizon, start, cutoff):
    candidates=g.loc[g.year.between(start,year-1)&g[f'eligible_{horizon}']]
    train=candidates.loc[candidates[f'mature_{horizon}'].lt(cutoff)]
    return train,len(candidates)-len(train)


def experiment(out, native, frames):
    timing,training,folds=[],[],[]
    selected,grid,paths,weights,predictions=[],[],[],[],[]
    cutoffs={}
    for y in YEARS:
        stamps=[f.loc[f.year.eq(y)&(f.eligible_5|f.eligible_20),'day1_close_bound'].min() for f in frames.values()]
        cutoffs[y]=min(stamps)
    for option,g in frames.items():
        check=g.loc[g.year.isin(YEARS)].copy()
        timing.append(check[['transcriptid','gvkey','call_date','clock','day0','day1','day1_close_bound','signal','day2_open',
                             'mature_5','mature_20','timely','eligible_5','eligible_20','vintage']].assign(benchmark=option,
                             refit_cutoff=check.year.map(cutoffs)))
        for y in YEARS:
            cutoff=cutoffs[y]
            for h in HORIZONS:
                test=g.loc[g.year.eq(y)&g[f'eligible_{h}']].sort_values(['call_date','transcriptid'])
                need(test.signal.gt(cutoff).all() and test.signal.lt(test.day2_open).all(),'Prediction clock violation')
                for name in WINDOWS:
                    start=2010 if name=='Expanding' else y-5
                    train,purged=outer_train(g,y,h,start,cutoff)
                    need(train[f'mature_{h}'].lt(cutoff).all() and train.year.lt(y).all() and
                         set(train.transcriptid).isdisjoint(test.transcriptid),'Outer fit isolation violation')
                    context=dict(year=y,horizon=h,benchmark=option,window=name)
                    training.append(dict(**context,train_start=start,train_events=len(train),train_issuers=train.gvkey.nunique(),
                                         purged_at_refit=purged,refit_cutoff=cutoff,max_training_maturity=train[f'mature_{h}'].max(),
                                         test_events=len(test),test_issuers=test.gvkey.nunique(),first_call=test.call_date.min(),
                                         last_call=test.call_date.max()))
                    fold_data,audit=fold_history(train,y,h,native,context)
                    folds.extend(audit)
                    search=InnerSearch(fold_data,context)
                    recipes=search.select()
                    grid.extend(search.estimator_rows); paths.extend(search.path_rows)
                    x,z=arrays(native,train,test[list(CONTROLS)+list(LM)])
                    target=stats.rankdata(train[f'car_2_{h}'])
                    forecasts={}
                    for model,(route,subset,algo,cv) in recipes.items():
                        fields=[*CONTROLS,*[m for m in LM if m in subset]]
                        ids=[(*CONTROLS,*LM).index(m) for m in fields]
                        fit=estimator(algo).fit(x[:,ids],target)
                        forecasts[model]=fit.predict(z[:,ids])
                        if model=='B' and not subset:
                            need(np.array_equal(forecasts['A'],forecasts['B']),'Controls fallback does not match A')
                        selected.append(dict(**context,model=model,route=route,features='|'.join(subset),
                                             feature_count=len(subset),estimator=algo,inner_rank_ic=cv))
                        total=abs(fit.coef_).sum()
                        weights.extend(dict(**context,model=model,feature=f,block='LM' if f in LM else 'control',
                                            standardized_coefficient=float(v),scaled_by_train_n=float(v/len(train)),
                                            absolute_share=float(abs(v)/total) if total else 0.)
                                       for f,v in zip(fields,fit.coef_))
                    # The held-out labels are brought into the predictions artifact after all decisions and scoring.
                    p=test[['transcriptid','gvkey','call_date','signal',f'car_2_{h}']].rename(columns={f'car_2_{h}':'actual_car'})
                    predictions.append(p.assign(**context,**forecasts))
                print(f'{option} {y} CAR[+2,+{h}]: {len(test)} test events, both windows fitted.',flush=True)
    pd.concat(timing).to_csv(out/'event_timing_local.csv',index=False)
    for name,rows in [('training_cohorts',training),('cv_folds',folds),('selected_features',selected),
                      ('estimator_cv',grid),('feature_selection_paths',paths),('feature_importance',weights)]:
        table(out,name,rows)
    p=pd.concat(predictions,ignore_index=True)
    p.to_csv(out/'matched_predictions_local.csv',index=False,float_format='%.17g')
    return p,cutoffs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve()
    need(out==HERE or HERE in out.parents or ROOT not in out.parents and out!=ROOT,'Protected destination')
    out.mkdir(parents=True,exist_ok=True)
    declare(out)
    native,frames,provenance=inputs(out)
    if args.audit_only:
        for opt,g in frames.items():
            print(opt, g.groupby('year')[['eligible_5','eligible_20']].sum().tail(3).to_string(),flush=True)
        return
    p,cutoffs=experiment(out,native,frames)
    from research_metrics import evaluate
    evaluate(out,frames,p,cutoffs)
    need(all(digest(ROOT/name)==h for name,h in provenance.items()),'Input changed during experiment')


if __name__=='__main__':
    main()
