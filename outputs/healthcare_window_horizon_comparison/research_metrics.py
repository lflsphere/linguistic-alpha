"""Aggregate held-out scores and factor measurements by annual test cohort."""

import numpy as np
import pandas as pd
from scipy import stats

from run_comparison import BLOCK, CONTROLS, DRAWS, HORIZONS, LM, MODELS, OPTIONS, SEED, WINDOWS, YEARS, need, table


def correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan,np.nan
    return float(stats.spearmanr(x,y).statistic),float(np.corrcoef(x,y)[0,1])


def summarize(values, actual, issuers):
    if len(values)<50 or issuers<20:
        return [],np.nan,'insufficient sample'
    groups=pd.qcut(values,5,labels=False,duplicates='drop')
    if pd.isna(groups).any() or len(set(groups))!=5:
        return [],np.nan,'insufficient distinct score groups'
    parts=[]
    for i in range(5):
        r=actual[groups==i]
        parts.append(dict(quintile=i+1,n=len(r),mean_car=float(r.mean()),median_car=float(np.median(r)),
                          sd_car=float(r.std(ddof=1)) if len(r)>1 else np.nan))
    means=np.array([r['mean_car'] for r in parts]); step=np.diff(means)
    shape='increasing' if (step>0).all() else 'decreasing' if (step<0).all() else 'mixed'
    return parts,float(means[-1]-means[0]),shape


def paired_statistics(data,pairs):
    r=stats.rankdata(data,axis=0)
    centered=r-r.mean(axis=0)
    norms=np.sqrt((centered*centered).sum(axis=0))
    rho=np.array([np.dot(centered[:,a],centered[:,b])/(norms[a]*norms[b]) if norms[a] and norms[b] else np.nan
                  for a,b in pairs])
    return rho


def uncertainty(d, columns, pairs, contrasts, year, seed):
    """Resample shared event indices before calculating all contrasts in a family."""
    matrix=d[list(columns)].to_numpy(float)
    groups=list(d.groupby('gvkey',sort=True).indices.values())
    first=pd.Period(f'{year}-01',freq='M'); last=pd.Period(f'{year}-{6 if year==2026 else 12:02}',freq='M')
    grid=pd.period_range(first,last,freq='M')
    months=d.groupby(d.call_date.dt.to_period('M')).indices
    monthly=[months.get(m,np.array([],dtype=int)) for m in grid]
    blocks=len(grid)//BLOCK
    rng=np.random.default_rng(seed)
    results=[]
    for method in ('Issuer cluster','Two-month block'):
        observed=contrasts@paired_statistics(matrix,pairs) if len(d)>=3 else np.full(len(contrasts),np.nan)
        draws=[]
        sufficient=len(d)>=50 and len(groups)>=30 and (method=='Issuer cluster' or blocks>=5)
        if sufficient:
            for _ in range(DRAWS):
                if method=='Issuer cluster':
                    index=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
                else:
                    starts=rng.integers(0,len(grid)-BLOCK+1,size=int(np.ceil(len(grid)/BLOCK)))
                    slots=(starts[:,None]+np.arange(BLOCK)).ravel()[:len(grid)]
                    index=np.concatenate([monthly[j] for j in slots])
                if len(index)>=30:
                    draws.append(contrasts@paired_statistics(matrix[index],pairs))
        for j,point in enumerate(observed):
            one=np.asarray(draws)[:,j] if draws else np.array([])
            valid=one[np.isfinite(one)]; count=len(valid)
            enough=count>=.95*DRAWS and np.isfinite(point)
            low,high=np.percentile(valid,[2.5,97.5]) if enough else (np.nan,np.nan)
            results.append((j,dict(method=method,events=len(d),issuers=len(groups),nominal_blocks=blocks,
                                   draws=count,point=point,lower=low,upper=high,available=bool(enough))))
    return results


def model_tables(out,predictions):
    rows,quints,window,horizon,increments,coverage,inference=[],[],[],[],[],[],[]
    cohorts={}
    for y in YEARS:
        for b in OPTIONS:
            all_cases=predictions.loc[predictions.year.eq(y)&predictions.benchmark.eq(b)]
            common=set(all_cases.loc[all_cases.horizon.eq(5),'transcriptid']) & set(all_cases.loc[all_cases.horizon.eq(20),'transcriptid'])
            short_long={}
            for h in HORIZONS:
                sl=all_cases.loc[all_cases.horizon.eq(h)]
                e=sl.loc[sl.window.eq(WINDOWS[0])].sort_values('transcriptid').reset_index(drop=True)
                r=sl.loc[sl.window.eq(WINDOWS[1])].sort_values('transcriptid').reset_index(drop=True)
                need(e.transcriptid.equals(r.transcriptid) and e.actual_car.equals(r.actual_car),'Window cohort mismatch')
                common_case=e.transcriptid.isin(common)
                coverage.append(dict(year=y,benchmark=b,horizon=h,full_events=len(e),full_issuers=e.gvkey.nunique(),
                                     matched_horizons_events=int(common_case.sum()),matched_horizons_issuers=e.loc[common_case,'gvkey'].nunique()))
                for d,w in [(e,WINDOWS[0]),(r,WINDOWS[1])]:
                    contexts=[('Full horizon',d),('Matched horizons',d.loc[d.transcriptid.isin(common)])]
                    short_long[h,w]=contexts[1][1]
                    for sample,case in contexts:
                        context=dict(year=y,benchmark=b,horizon=h,window=w,sample=sample)
                        ic={}
                        for m in MODELS:
                            y_0=case.actual_car.to_numpy()
                            x=case[m].to_numpy(); n=case.gvkey.nunique()
                            rank,pearson=correlation(x,y_0)
                            sorts,spread,shape=summarize(x,y_0,n)
                            ic[m]=rank
                            rows.append(dict(**context,model=m,events=len(case),issuers=n,rank_ic=rank,pearson=pearson,
                                             q5_minus_q1=spread,monotonicity=shape))
                            quints.extend(dict(**context,model=m,**sort) for sort in sorts)
                        for m in MODELS[1:]:
                            increments.append(dict(**context,model=m,events=len(case),issuers=case.gvkey.nunique(),
                                                   delta_rank_ic=ic[m]-ic['A']))
                        cohorts[y,b,h,w,sample]=ic
                e2=e.rename(columns={m:f'Exp {m}' for m in MODELS}); r2=r.rename(columns={m:f'Roll {m}' for m in MODELS})
                matrix=e2.join(r2[[f'Roll {m}' for m in MODELS]])
                cols=[f'{prefix} {m}' for prefix in ('Exp','Roll') for m in MODELS]+['actual_car']
                eye=np.eye(6)
                coefficients=np.stack([*(eye[3:]-eye[:3]),eye[1]-eye[0],eye[2]-eye[0],eye[4]-eye[3],eye[5]-eye[3]])
                labels=[('Window','Rolling 5Y − Expanding',m) for m in MODELS]+[(
                    'LM',w,m) for w in WINDOWS for m in MODELS[1:]]
                for j,ci in uncertainty(matrix,cols,[(i,6) for i in range(6)],coefficients,y,SEED+100*y+10*h+OPTIONS.index(b)):
                    family,comparison,model=labels[j]
                    inference.append(dict(year=y,benchmark=b,horizon=h,family=family,comparison=comparison,model=model,
                                          sample='Full horizon',**ci))
                for m in MODELS:
                    e_ic=cohorts[y,b,h,WINDOWS[0],'Full horizon'][m]; r_ic=cohorts[y,b,h,WINDOWS[1],'Full horizon'][m]
                    window.append(dict(year=y,benchmark=b,horizon=h,model=m,events=len(e),issuers=e.gvkey.nunique(),
                                       expanding_rank_ic=e_ic,rolling_rank_ic=r_ic,rolling_minus_expanding=r_ic-e_ic))
            for w in WINDOWS:
                s=short_long[5,w]; l=short_long[20,w]; a=s.transcriptid.tolist()
                need(a==l.transcriptid.tolist(),'Horizon ordering or eligibility mismatch')
                joined=s.rename(columns={m:f'Short {m}' for m in (*MODELS,'actual_car')}).reset_index(drop=True).join(
                    l[[*MODELS,'actual_car']].rename(columns={m:f'Long {m}' for m in (*MODELS,'actual_car')}).reset_index(drop=True))
                columns=[f'{prefix} {m}' for prefix in ('Short','Long') for m in (*MODELS,'actual_car')]
                models=[(0,3),(1,3),(2,3),(4,7),(5,7),(6,7)]
                contrast=np.eye(6)[3:]-np.eye(6)[:3]
                for j,ci in uncertainty(joined,columns,models,contrast,y,SEED+10*y+7*WINDOWS.index(w)+OPTIONS.index(b)):
                    inference.append(dict(year=y,benchmark=b,horizon=0,family='Horizon',comparison=w,
                                          model=MODELS[j],sample='Matched horizons',**ci))
                for m in MODELS:
                    g5=cohorts[y,b,5,w,'Matched horizons']; g20=cohorts[y,b,20,w,'Matched horizons']
                    horizon.append(dict(year=y,benchmark=b,window=w,model=m,events=len(s),issuers=s.gvkey.nunique(),
                                        short_rank_ic=g5[m],long_rank_ic=g20[m],long_minus_short=g20[m]-g5[m],
                                        short_increment=g5[m]-g5['A'],long_increment=g20[m]-g20['A']))
    for name,entries in [('annual_model_performance',rows),('model_quintiles',quints),('window_comparison',window),
                         ('horizon_comparison',horizon),('lm_increments',increments),('horizon_cohorts',coverage),
                         ('paired_inference',inference)]:
        table(out,name,entries)


def factor_tables(out,frames,cutoffs):
    summary,stability,quints,assoc,conditional,missing,ci=[],[],[],[],[],[],[]
    for b,g in frames.items():
        for h in HORIZONS:
            t=f'car_2_{h}'
            cohort=g.loc[g[f'eligible_{h}']].copy()
            train=cohort.loc[cohort.year.lt(2024)&cohort[f'mature_{h}'].lt(cutoffs[2024])]
            split=train.vol_pre.median()
            for yr in (0,*YEARS):
                d=train if yr==0 else cohort.loc[cohort.year.eq(yr)]
                context=dict(year=yr,scope='Development' if yr==0 else 'Outer test',benchmark=b,horizon=h)
                for name in (*CONTROLS,*LM):
                    missing.append(dict(**context,feature=name,events=len(d),missing=int(d[name].isna().sum()),
                                        missing_share=float(d[name].isna().mean())))
                correlation_fields=[*LM,*CONTROLS]
                matrix=d[correlation_fields].corr(method='spearman',min_periods=20)
                for f in LM:
                    for other in correlation_fields:
                        assoc.append(dict(**context,factor=f,feature=other,block='LM × LM' if other in LM else 'LM × controls',
                                          n=int(d[[f,other]].notna().all(axis=1).sum()),rank_corr=matrix.at[f,other]))
                for k,f in enumerate(LM):
                    complete=d.loc[d[f].notna()].copy()
                    x=complete[f].to_numpy(); y=complete[t].to_numpy()
                    cnt=complete.gvkey.nunique()
                    rank,pearson=correlation(x,y)
                    sorts,spread,shape=summarize(x,y,cnt)
                    summary.append(dict(**context,factor=f,events=len(complete),eligible_events=len(d),issuers=cnt,
                                        rank_ic=rank,pearson=pearson,q5_minus_q1=spread,monotonicity=shape))
                    quints.extend(dict(**context,factor=f,**sort) for sort in sorts)
                    if yr:
                        for _,bounds in uncertainty(complete,[f,t],[(0,1)],np.array([[1.]]),yr,
                                                   SEED+10000+100*yr+10*h+k+OPTIONS.index(b)):
                            ci.append(dict(**context,factor=f,**bounds))
                    for period,grp in complete.groupby(complete.call_date.dt.to_period('Q')):
                        enough=len(grp)>=25 and grp.gvkey.nunique()>=15
                        v=correlation(grp[f].to_numpy(),grp[t].to_numpy())[0] if enough else np.nan
                        stability.append(dict(**context,factor=f,quarter=str(period),events=len(grp),issuers=grp.gvkey.nunique(),
                                              rank_ic=v,eligible=bool(enough),partial=str(period)=='2026Q2'))
                    groups=[('SUE ≥ 0',complete.sue.ge(0)),('SUE < 0',complete.sue.lt(0)),
                            ('Pre-vol above development median',complete.vol_pre.gt(split)),
                            ('Pre-vol at/below development median',complete.vol_pre.le(split))]
                    for label,mask in groups:
                        group=complete.loc[mask]; n=group.gvkey.nunique()
                        v=correlation(group[f].to_numpy(),group[t].to_numpy())[0] if len(group)>=30 and n>=15 else np.nan
                        conditional.append(dict(**context,factor=f,condition=label,events=len(group),issuers=n,rank_ic=v,
                                                prevol_cutoff=split if 'Pre-vol' in label else np.nan))
    for name,entries in [('factor_ic',summary),('factor_quintiles',quints),('factor_stability',stability),
                          ('correlations',assoc),('conditional_factor_ic',conditional),('feature_missingness',missing),
                          ('factor_inference',ci)]:
        table(out,name,entries)


def evaluate(out,frames,predictions,cutoffs):
    model_tables(out,predictions)
    factor_tables(out,frames,cutoffs)
