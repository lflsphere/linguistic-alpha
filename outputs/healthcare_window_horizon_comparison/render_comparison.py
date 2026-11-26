"""Build the comparison dashboard, PDF and Chinese brief from saved aggregates."""

import argparse
from html import escape
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd


HERE=Path(__file__).resolve().parent
BRAND='HEALTHCARE LINGUISTIC ALPHA RESEARCH'
TITLE='Linguistic Alpha Evaluation'
SUBTITLE='Training Window and Prediction Horizon Comparisons | Healthcare | Test 2024–2026'
PDF='Healthcare_Training_Window_and_Prediction_Horizon_Report.pdf'
YEARS=(2024,2025,2026)
OPTIONS=('S&P 500','Healthcare')
MODELS=('A','B','B*')
WINDOWS=('Expanding','Rolling 5Y')
COLOR={'A':'#335f91','B':'#08766c','B*':'#795b93'}
NAVY,INK,MUTE,LINE,SOFT,TEAL='#112d43','#172f45','#496074','#d6e3e8','#e6f3ef','#08766c'
SECTIONS=('Research Overview','Training Window Comparison','Prediction Horizon Comparison','Annual Model Rank IC',
          'Incremental Linguistic Alpha','Single-Factor Rank IC','Factor Stability','Quintile Analysis',
          'Correlation and Redundancy','Feature Selection and Importance','Statistical Inference','Research Insights and Methodology')
NAMES=('training_cohorts','target_validation','horizon_cohorts','annual_model_performance','window_comparison',
       'horizon_comparison','lm_increments','model_quintiles','factor_ic','factor_stability','factor_quintiles',
       'conditional_factor_ic','correlations','paired_inference','factor_inference','selected_features',
       'feature_selection_paths','estimator_cv','feature_importance','feature_missingness','lm_redundancy','cv_folds')
ALIASES={'tone_prepared':'Prepared tone','tone_answers':'Answer tone','hedging_prepared':'Prepared hedge',
         'uncertainty_answers':'Answer uncertainty','tone_gap':'Tone gap','tone_prepared_chg':'Δ Prepared tone',
         'tone_answers_chg':'Δ Answer tone','hedging_answers_chg':'Δ Answer hedge'}


def filt(frame, **kwargs):
    return frame.loc[np.logical_and.reduce([frame[k].eq(v) for k,v in kwargs.items()])]


def num(v,d=4):
    return '—' if pd.isna(v) else f'{v:.{d}f}'.replace('-','−')


def delta(v):
    return ('+' if v>0 else '')+num(v)


def pp(v):
    return num(v*100,2)


def yearname(y):
    return 'Development 2010–23' if y==0 else f'{y}'+(' (Partial)' if y==2026 else '')


def count_table(D):
    W,H,I=D['window_comparison'],D['horizon_comparison'],D['lm_increments']
    rows=[]
    for year in YEARS:
        w=filt(W,year=year).rolling_minus_expanding
        h=filt(H,year=year).long_minus_short
        i=filt(I,year=year,sample='Full horizon')
        pos=sum(w>0); hp=sum(h>0); b=sum(filt(i,model='B').delta_rank_ic>0); star=sum(filt(i,model='B*').delta_rank_ic>0)
        rows.append((year,pos,hp,b,star))
    return rows


def conclusions(D):
    counts=count_table(D)
    chosen=filt(D['selected_features'],model='B')
    note=chosen.features.dropna().str.split('|').explode().value_counts()
    intervals=filt(D['paired_inference'],family='LM',method='Issuer cluster')
    excluded=int((intervals.lower.gt(0)|intervals.upper.lt(0)).sum())
    factor=filt(D['factor_ic'],factor='tone_answers_chg')
    factor=factor.loc[factor.year.ne(0)]
    return [f'Rolling improved Rank IC in {counts[0][1]}, {counts[1][1]} and {counts[2][1]} of 12 cases in 2024, 2025 and partial 2026.',
            f'The long horizon led on matched calls in {counts[0][2]}, {counts[1][2]} and {counts[2][2]} of 12 cases. The partial 2026 comparison has issuer-cluster intervals only.',
            f'{excluded} of {len(intervals)} model-LM issuer intervals exclude zero. B selected no LM inputs in {int(chosen.feature_count.eq(0).sum())} of {len(chosen)} fits.',
            f'Prepared-tone change appeared in {note.get("tone_prepared_chg",0)} B fits and answer-hedging change in {note.get("hedging_answers_chg",0)}. Raw answer-tone change had negative Rank IC in {int(factor.rank_ic.lt(0).sum())} of {len(factor)} test benchmark-horizon cells.']


def make_pdf(out,D,spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.unicode_minus':False})
    C=D['horizon_cohorts']; W=D['window_comparison']; H=D['horizon_comparison']; P=D['annual_model_performance']
    I=D['lm_increments']; F=D['factor_ic']; B=D['paired_inference']; S=D['selected_features']
    shortlong=(5,20)

    def put(a,x,y,s,size=9,c=INK,b='normal',serif=False):
        a.text(x,y,str(s),ha='left',va='top',fontsize=size,color=c,fontweight=b,
               fontfamily=['Georgia','DejaVu Serif'] if serif else 'DejaVu Sans')

    def rect(a,x,y,w,h,c):
        a.add_patch(Rectangle((x,y),w,h,facecolor=c,edgecolor='none'))

    def body(a,y,s,size=9.2,width=102):
        for line in textwrap.wrap(s,width=width,break_long_words=False):
            put(a,43,y,line,size,MUTE)
            y+=size+4
        return y

    def label(a,y,s):
        put(a,43,y,s,12,NAVY,'bold')

    def layout(n,sub):
        f=plt.figure(figsize=(8.5,11)); f.patch.set_facecolor('white')
        a=f.add_axes([0,0,1,1]); a.set(xlim=(0,612),ylim=(792,0));a.axis('off')
        rect(a,0,0,612,121 if n==1 else 83,NAVY); rect(a,0,0,612,4,'#3cac99')
        put(a,43,22,BRAND,8.6,'#86d5c6','bold'); put(a,43,42,TITLE,29 if n==1 else 22,'white',serif=True)
        if n==1: put(a,43,87,SUBTITLE,8.8,'#d5e5e9')
        y=143 if n==1 else 105
        put(a,43,y,SECTIONS[n-1],25,NAVY,serif=True)
        body(a,y+38,sub,size=8.9,width=107)
        a.plot([43,569],[746,746],color=LINE,lw=.85)
        put(a,43,759,'HEALTHCARE  /  RETROSPECTIVE EVALUATION  /  TEST 2024–2026',7.4,MUTE)
        put(a,554,759,f'{n:02}',8,NAVY,'bold')
        return f,a

    def tbl(a,y,header,rows,xs,h=19,fs=7.9):
        rect(a,43,y,526,h,SOFT)
        for x,v in zip(xs,header): put(a,x,y+4,v,fs,NAVY,'bold')
        for k,row in enumerate(rows):
            yp=y+(k+1)*h
            if k%2: rect(a,43,yp,526,h,'#f2f6f7')
            for x,v in zip(xs,row): put(a,x,yp+4,v,fs,COLOR.get(v,INK),'bold' if v in MODELS else 'normal')
        return y+h*(len(rows)+1)

    def axis(f,x,y,w,h,ylabel='',grid=True):
        a=f.add_axes([x/612,(792-y-h)/792,w/612,h/792]); a.set_axisbelow(True)
        if grid:
            a.grid(axis='y',color=LINE,lw=.65);a.axhline(0,color='#8a9aa4',lw=.8)
        a.set_ylabel(ylabel,fontsize=7.6,color=MUTE);a.tick_params(axis='both',labelsize=7.6,length=0,colors=MUTE,pad=5)
        for sp in a.spines.values():sp.set_visible(False)
        return a

    def legend(a,y,labels=COLOR):
        for k,(m,c) in enumerate(labels.items()):
            x=43+135*k;rect(a,x,y+3,10,4,c);put(a,x+15,y,m,8,MUTE)

    def save(p,f): p.savefig(f);plt.close(f)

    with PdfPages(out/PDF,metadata={'Title':f'{BRAND} | {TITLE}','Subject':SUBTITLE,'CreationDate':None,'ModDate':None}) as pdf:
        f,a=layout(1,'The two comparisons use the same controls, temporal selection rule and matched-event scoring convention.')
        label(a,221,'Test cohorts')
        rows=[]
        for y in YEARS:
            s=filt(C,year=y,benchmark='Healthcare').set_index('horizon')
            rows.append((yearname(y),f'{s.loc[5,"full_events"]} / {s.loc[5,"full_issuers"]}',f'{s.loc[20,"full_events"]} / {s.loc[20,"full_issuers"]}',
                         f'{s.loc[20,"matched_horizons_events"]} / {s.loc[20,"matched_horizons_issuers"]}'))
        tbl(a,246,['Test','Short calls / issuers','Long calls / issuers','Matched calls / issuers'],rows,[49,206,329,448],23)
        body(a,352,'Coverage is identical at the two benchmarks. Partial 2026 uses completed outcomes through the June 30 CRSP cutoff.')
        label(a,395,'Training and estimation')
        body(a,421,'Expanding starts in 2010. Rolling 5Y uses the five preceding calendar years. A contains 16 controls. B adds a subset of eight candidate LM factors chosen in inner temporal folds. B* contains all eight.')
        body(a,483,'Both benchmarks preserve the original model table conventions. The short CAR covers sessions +2 to +5. The long CAR covers sessions +2 to +20. Each measures the sum of daily stock returns less its daily benchmark.')
        label(a,546,'Verification and interpretation')
        body(a,573,'All model decisions use prior data. Inner training labels must be complete strictly before the first validation prediction. A separate maturity gate applies to the annual refit.')
        body(a,628,'The outer periods were already examined in earlier research. The new estimates are retrospective and exploratory. Figures show each test year separately.')
        save(pdf,f)

        f,a=layout(2,'Expanding history and the preceding five training years on the same benchmark, horizon and test events.')
        rows=[]
        for y in YEARS:
            for b in OPTIONS:
                for h in shortlong:
                    d=filt(W,year=y,benchmark=b,horizon=h).set_index('model')
                    rows.append((yearname(y),b,f'+2…+{h}',int(d.events.iloc[0]),*[delta(d.loc[m,'rolling_minus_expanding']) for m in MODELS]))
        tbl(a,184,['Test','Benchmark','CAR','Calls','A Δ IC','B Δ IC','B* Δ IC'],rows,[49,151,259,316,370,437,503],19)
        body(a,443,'Table 1. Each entry is Rolling 5Y minus Expanding Rank IC. The models in a row share event IDs and realized returns.')
        label(a,489,'Fit coverage by year')
        r=[]
        for y in YEARS:
            a0=filt(D['training_cohorts'],year=y,window=WINDOWS[0]); a1=filt(D['training_cohorts'],year=y,window=WINDOWS[1])
            span=lambda s:f'{s.min():,}–{s.max():,}' if s.min()!=s.max() else f'{s.min():,}'
            r.append((yearname(y),span(a0.train_events),span(a0.train_issuers),span(a1.train_events),span(a1.train_issuers)))
        tbl(a,513,['Test','Expanding n','Exp. issuers','Rolling n','Rolling issuers'],r,[49,182,287,381,479],21)
        body(a,619,conclusions(D)[0]); body(a,668,'A change in the training-window Rank IC contrasts whole-model predictions. On the next pages, LM models are compared with their financial baseline within a window.')
        save(pdf,f)

        f,a=layout(3,'Both models use the intersection of eligible test calls for this comparison. Each is scored against its own realized CAR.')
        rows=[]
        for y in YEARS:
            for b in OPTIONS:
                for w in WINDOWS:
                    d=filt(H,year=y,benchmark=b,window=w).set_index('model')
                    rows.append((yearname(y),b,w,int(d.events.iloc[0]),*[delta(d.loc[m,'long_minus_short']) for m in MODELS]))
        tbl(a,187,['Test','Benchmark','Window','Calls','A Δ IC','B Δ IC','B* Δ IC'],rows,[49,150,249,322,374,439,505],20)
        body(a,467,'Table 2. Each entry is long-horizon minus short-horizon Rank IC on identical events. The 2024 and partial 2026 short targets have additional eligible calls in their full-horizon cohorts.')
        label(a,530,'Target verification')
        body(a,556,'The legacy Healthcare short target reconciles to 205 original event labels. The new short target is computed from CIZ stock returns and the validated Healthcare benchmark, and verified in an independent four-session join.')
        body(a,609,'The same modern stock-and-benchmark alignment reproduces the saved 315 Healthcare long labels. Market target checks reconcile the validated short and long S&P 500 labels.')
        body(a,670,'Rank IC compares ordering within a horizon. A longer CAR window has different return exposure, so raw quintile spread magnitudes do not measure a change in predictive accuracy.')
        save(pdf,f)

        f,a=layout(4,'Spearman correlations between each saved prediction and its realized CAR on the full eligible horizon cohort.')
        legend(a,172)
        lim=max(.30,np.ceil(abs(P.rank_ic).max()*10)/10)
        for i,b in enumerate(OPTIONS):
            for j,h in enumerate(shortlong):
                x=43+263*j;y=205+222*i
                put(a,x,y,f'{b}  /  CAR[+2,+{h}]',9.6,NAVY,'bold')
                ax=axis(f,x+37,y+26,199,133,'Rank IC')
                for model in MODELS:
                    for w in WINDOWS:
                        d=filt(P,benchmark=b,horizon=h,window=w,model=model,sample='Full horizon')
                        ax.plot(range(3),d.rank_ic,'-o' if w=='Expanding' else '--s',color=COLOR[model],lw=1.4,ms=4)
                ax.set_ylim(-lim,lim);ax.set_xticks(range(3),['2024','2025','2026 partial'])
                n=' / '.join(str(v) for v in filt(C,benchmark=b,horizon=h).full_events)
                put(a,x,y+190,f'Calls by year  {n}',7.9,MUTE)
        body(a,648,'Figure 1. Solid circles indicate Expanding. Dashed squares indicate Rolling 5Y. Correlations are calculated independently in each test year. Pearson IC is supplementary in the annual CSV.',size=8.4)
        save(pdf,f)

        f,a=layout(5,'The LM increment compares each language model with A on the same test calls, benchmark, horizon and training window.')
        for i,b in enumerate(OPTIONS):
            label(a,179+240*i,b)
            rows=[]
            for y in YEARS:
                for h in shortlong:
                    d=filt(I,year=y,benchmark=b,horizon=h,sample='Full horizon')
                    v=[delta(filt(d,window=w,model=m).delta_rank_ic.iloc[0]) for w in WINDOWS for m in ('B','B*')]
                    rows.append((yearname(y),f'+2…+{h}',int(d.events.iloc[0]),*v))
            tbl(a,202+240*i,['Test','CAR','Calls','Exp. B−A','Exp. B*−A','Roll B−A','Roll B*−A'],
                rows,[49,161,215,273,345,421,495],20)
        body(a,642,'Table 3. Rank IC differences are dimensionless. B equals A when the inner search retains controls only. The comparisons hold the control block fixed. Each model can select a different estimator within training.')
        body(a,701,conclusions(D)[2],size=8.4)
        save(pdf,f)

        f,a=layout(6,'Raw LM measurements are evaluated separately from fitted-model predictions. These factor results do not enter model selection.')
        label(a,182,'Test 2025 Rank IC by benchmark and horizon')
        rows=[]
        for field in spec['lm']:
            z=filt(F,year=2025,factor=field)
            values=[num(filt(z,benchmark=b,horizon=h).rank_ic.iloc[0]) for b in OPTIONS for h in shortlong]
            rows.append((ALIASES[field],*values))
        tbl(a,204,['Factor','S&P +5','S&P +20','Healthcare +5','Healthcare +20'],rows,[49,239,316,386,474],18)
        body(a,379,'Table 4. Raw Spearman IC on observed factor and CAR pairs. For exact coverage, see factor_ic.csv. Imputed model features do not enter the factor measures.')
        label(a,420,'Test 2025 Healthcare long horizon, issuer intervals')
        fx=filt(D['factor_inference'],year=2025,benchmark='Healthcare',horizon=20,method='Issuer cluster')
        ax=axis(f,155,454,396,176,grid=False);ax.axvline(0,color='#8a9aa4',lw=.8)
        for i,r in enumerate(fx.itertuples()):
            y=len(fx)-1-i
            ax.hlines(y,r.lower,r.upper,colors=TEAL,lw=1.45);ax.plot(r.point,y,'o',ms=4,color=TEAL)
        ax.set_yticks(range(len(fx)),[f'{ALIASES[r.factor]} (n={r.events})' for r in fx.itertuples()][::-1]); ax.set_xlabel('Factor Rank IC',fontsize=7.6,color=MUTE)
        h=max(.20,np.ceil(abs(fx[['lower','upper']].to_numpy()).max()*10)/10);ax.set_xlim(-h,h)
        body(a,655,'Figure 2. Pointwise paired resampling by issuer. Observation counts are factor-specific. Annual tables include issuer and calendar-time intervals wherever eligible.',size=8.3)
        save(pdf,f)

        f,a=layout(7,'Quarterly IC and prespecified observable financial conditions. Each estimate uses its own eligible return window.')
        label(a,178,'Answer tone, Healthcare CAR[+2,+20]')
        t=filt(D['factor_stability'],benchmark='Healthcare',horizon=20,factor='tone_answers')
        t=t.loc[t.year.ne(0)]
        ax=axis(f,95,211,449,148,'Quarterly Rank IC')
        for i,y in enumerate(YEARS):
            m=t.year.eq(y).to_numpy();x=np.flatnonzero(m);ax.plot(x,t.loc[m,'rank_ic'],'-o',color=['#335f91',TEAL,'#795b93'][i],lw=1.5,ms=4)
        ax.set_xticks(range(len(t)),[f'{r.quarter[-2:]}{"*" if r.partial else ""}\n{r.events}' for r in t.itertuples()])
        ax.set_ylim(-.55,.55)
        body(a,391,'Figure 3. From left to right, four 2024 quarters, four 2025 quarters and two 2026 quarters. Counts appear under the quarter. * The last quarter reflects the partial CRSP year.',size=8.3)
        label(a,435,'Financial conditions, Test 2025 Healthcare long horizon')
        q=filt(D['conditional_factor_ic'],year=2025,benchmark='Healthcare',horizon=20)
        conditions=list(q.condition.unique()); rows=[]
        for k in spec['lm']:
            z=filt(q,factor=k).set_index('condition')
            rows.append((ALIASES[k],*[f'{num(z.loc[c,"rank_ic"])} ({z.loc[c,"events"]})' for c in conditions]))
        tbl(a,461,['Factor','SUE ≥ 0','SUE < 0','High pre-vol','Low pre-vol'],rows,[49,218,309,396,485],17,7.7)
        body(a,632,'Table 5. Spearman IC with paired n in parentheses. Pre-vol is classified using the pre-2024 development median. Conditional cells require 30 events and 15 issuers.',size=8.4)
        yearly=filt(F,benchmark='Healthcare',horizon=20,factor='tone_answers')
        yearly=yearly.loc[yearly.year.ne(0)]
        body(a,684,'Annual answer-tone IC  ·  '+'  /  '.join(f'{r.year} {num(r.rank_ic)} (n={r.events})' for r in yearly.itertuples()),size=8.4)
        body(a,709,'The development quarterly series is available in factor_stability.csv.',size=8.4)
        save(pdf,f)

        f,a=layout(8,'Prediction sorts and raw factor sorts are shown separately. Realized CAR is expressed as percent.')
        legend(a,170)
        Q=D['model_quintiles']; q=filt(Q,year=2025,benchmark='Healthcare',sample='Matched horizons')
        span=max(2,np.ceil(abs(q.mean_car).max()*100))
        for i,h in enumerate(shortlong):
            for j,w in enumerate(WINDOWS):
                x=43+j*263;y=198+i*146
                put(a,x,y,f'Model scores / {w} / +{h}',9.3,NAVY,'bold')
                ax=axis(f,x+42,y+22,190,90,'Mean CAR (%)')
                for m in MODELS:
                    z=filt(q,horizon=h,window=w,model=m)
                    ax.plot(z.quintile,z.mean_car*100,'-o',lw=1.45,ms=4,color=COLOR[m])
                ax.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5']);ax.set_ylim(-span,span)
        body(a,484,'Figure 4. Test 2025, Healthcare benchmark, 211 horizon-matched events per schedule. Identical y-axis across prediction sorts. Full group counts and Q5−Q1 appear in the tables and dashboard.',size=8.2)
        label(a,530,'Raw factor sorts, same benchmark and year')
        rows=[]
        for k in spec['lm']:
            v=[filt(F,year=2025,horizon=h,benchmark='Healthcare',factor=k).iloc[0] for h in shortlong]
            rows.append((ALIASES[k],pp(v[0].q5_minus_q1),pp(v[1].q5_minus_q1),min(v[0].events,v[1].events)))
        tbl(a,554,['Factor','Short Q5−Q1 (pp)','Long Q5−Q1 (pp)','Paired calls'],rows,[49,239,365,497],16,7.6)
        body(a,710,'Table 6. Sorts use overlapping event returns. Spread units are percentage points.',size=8.3)
        save(pdf,f)

        f,a=layout(9,'Development correlations use observed pairs with at least 20 observations. Quarterly and annual results remain separate.')
        v=filt(D['correlations'],year=0,benchmark='Healthcare',horizon=20)
        cols=list(spec['lm']);controls=list(spec['controls'])
        for i,(fields,label_name) in enumerate([(cols,'LM × LM'),(controls,'LM × controls')]):
            y=179+i*234;label(a,y,label_name)
            uv=v.pivot(index='factor',columns='feature',values='rank_corr').reindex(index=cols,columns=fields)
            ax=axis(f,159,y+28,375,121,grid=False)
            cmap=LinearSegmentedColormap.from_list('factor_corr',['#795b93','#f5f8f8','#08766c'])
            img=ax.imshow(uv.to_numpy(float),aspect='auto',vmin=-1,vmax=1,cmap=cmap)
            xt=[ALIASES.get(m,m).replace('Prepared','Prep.').replace('Answer','Ans.') for m in fields]
            ax.set_xticks(range(len(fields)),xt,rotation=49,ha='right');ax.tick_params(axis='x',labelsize=6.2)
            ax.set_yticks(range(len(cols)),[ALIASES[m] for m in cols]);ax.tick_params(axis='y',labelsize=7.0)
            ca=f.add_axes([544/612,(792-y-149)/792,7/612,121/792]);cb=f.colorbar(img,cax=ca,ticks=[-1,0,1]);cb.ax.tick_params(labelsize=6.5,length=0)
        red=D['lm_redundancy'].iloc[0]
        body(a,652,f'Figure 5. Development 2010–2023, Healthcare long horizon. Hedging and uncertainty Q&A inputs agree across {red.calls:,} source calls, with {red.joint_missing} joint missing rows. LM8 retains uncertainty_answers.',size=8.1)
        body(a,700,'Feature missingness is measured before any imputation. Its annual breakdown is in feature_missingness.csv.',size=8.1)
        save(pdf,f)

        f,a=layout(10,'All selections come from the recorded inner temporal folds. The feature coefficient table describes the fitted scale.')
        selected=filt(S,model='B')
        freq=selected.features.dropna().str.split('|').explode().value_counts().reindex(spec['lm'],fill_value=0)
        ax=axis(f,159,181,382,169,grid=False)
        ax.barh(range(len(freq)),freq[::-1],color=TEAL,height=.60)
        ax.set_yticks(range(len(freq)),[ALIASES[k] for k in freq.index[::-1]])
        ax.set_xlim(0,max(freq.max()+1,8));ax.set_xlabel('B fits selecting feature (of 24)',fontsize=8,color=MUTE)
        body(a,379,'Figure 6. Counts summarize the feature choices. No outer outcome contributes to selection.',size=8.3)
        label(a,418,'Test 2025 Model B selections')
        abbrev={'tone_prepared':'TP','tone_answers':'TA','hedging_prepared':'HP','uncertainty_answers':'UA',
                'tone_gap':'TG','tone_prepared_chg':'ΔTP','tone_answers_chg':'ΔTA','hedging_answers_chg':'ΔHA'}
        rows=[]
        for r in filt(selected,year=2025).itertuples():
            fields=' / '.join(abbrev[k] for k in r.features.split('|')) if isinstance(r.features,str) else 'Controls'
            rows.append((r.benchmark,f'+{r.horizon}',r.window,fields,r.estimator,num(r.inner_rank_ic)))
        tbl(a,443,['Benchmark','CAR','Window','LM subset','Estimator','CV IC'],rows,[49,151,205,302,423,520],18,7.4)
        body(a,623,'TP prepared tone. TA answer tone. HP prepared hedge. UA answer uncertainty. TG tone gap. ΔHA change in answer hedge. Δ denotes change in the archived measure.',size=8.1)
        body(a,682,'The dashboard exposes every fitted subset, the selection paths, estimator CV scores, and signed standardized-input coefficients divided by training n. Shares use absolute coefficients.',size=8.1)
        save(pdf,f)

        f,a=layout(11,'Examples of pointwise paired intervals. The downloadable annual tables contain every window, horizon and LM contrast.')
        subset=filt(B,year=2025,benchmark='Healthcare')
        a1=filt(subset,family='Window',horizon=20,model='A')
        a2=filt(subset,family='Horizon',comparison='Expanding',model='A')
        a3=filt(subset,family='LM',horizon=20,comparison='Expanding',model='B')
        examples=[('Rolling−Expanding A (+20)',a1),('Long−Short A (Expanding)',a2),('B−A (Expanding +20)',a3)]
        ax=axis(f,220,204,326,193,grid=False);ax.axvline(0,color='#8296a0',lw=.8)
        for j,(heading,d) in enumerate(examples):
            for k,r in enumerate(d.itertuples()):
                y=2-j+(0.10 if k==0 else -.10)
                ax.hlines(y,r.lower,r.upper,lw=1.6,color=TEAL,ls='-' if k==0 else '--')
                ax.plot(r.point,y,'o' if k==0 else 's',color=TEAL,ms=4)
        ax.set_yticks([2,1,0],[x for x,_ in examples]);ax.set_ylim(-.45,2.45);ax.set_xlabel('Δ Rank IC',fontsize=8,color=MUTE)
        body(a,432,'Figure 7. Test 2025 Healthcare, 211 matched calls and 61 issuers. Circles resample issuer clusters. Squares resample two-month blocks. Each eligible method requests 999 paired draws.',size=8.3)
        rows=[]
        for name,d in examples:
            r=d.set_index('method'); bounds=lambda m:f'[{num(r.loc[m,"lower"])}, {num(r.loc[m,"upper"])}]'
            rows.append((name,delta(r.point.iloc[0]),bounds('Issuer cluster'),bounds('Two-month block')))
        tbl(a,486,['Contrast','Point','Issuer 95% CI','Time-block 95% CI'],rows,[49,285,359,465],22,8)
        body(a,606,'Partial 2026 has too few calendar blocks for time-block intervals. Issuer intervals can still be calculated and capture a different dependence structure. Intervals condition on fitted scores and are exploratory.',size=8.9)
        body(a,670,'The pointwise intervals do not adjust for the number of comparisons. Statistical interpretation accounts for repeated issuers, overlapping CAR windows and limited time blocks.',size=8.9)
        save(pdf,f)

        f,a=layout(12,'Annual findings, assumptions and reproducibility.')
        label(a,179,'Research insights')
        for i,line in enumerate(conclusions(D)):
            body(a,205+53*i,line,size=9.3)
        label(a,427,'Timing and economic interpretation')
        body(a,454,'The prediction clock waits past the conservative Day +1 close and precedes the Day +2 session open. The first daily return includes close-to-close overnight movement. These estimates cannot establish live trading performance.')
        body(a,510,'Short windows may capture immediate repricing. Longer windows may reflect slower incorporation or other price movement. This design evaluates ranking and cannot identify a causal mechanism.')
        body(a,562,'Call timing and original Day 0 are verified against the saved CRSP sessions. Transcript delivery and point-in-time financial availability remain assumptions. All fits and validation folds exclude immature training targets.')
        label(a,629,'Reproduction')
        body(a,653,'The frozen research specification, source hashes, target validation, aggregate CSVs, event-timing ledger, prediction ledger and tests accompany this report. Event-level ledgers stay local.',size=8.9)
        save(pdf,f)


CSS = '''
figure{border:1px solid var(--line);border-radius:10px;margin:13px 0;padding:12px 15px;min-width:0}figure svg{width:100%;height:auto;display:block;font:12px "Segoe UI",sans-serif}figure h3{margin:2px 0 8px!important}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid>*{min-width:0}.context{padding:10px 0 1px}.filter-set{display:flex;align-items:end;flex-wrap:wrap;gap:9px 16px;padding:9px 0}.filter-set label{font-size:.86rem;font-weight:650;color:var(--navy)}select{display:block;padding:.49em .64em;margin-top:.25em;max-width:100%;font:inherit;background:white;border:1px solid #94a9b6;color:var(--navy);border-radius:6px}select:focus-visible,[tabindex="0"]:focus-visible{outline:3px solid var(--teal);outline-offset:2px}
.badge-info{background:#eff7f7;border-radius:8px;padding:10px 14px;font-size:.88rem}figcaption{min-height:1em}.tip{display:block;min-height:2em;padding:.3em .5em;background:#edf4f5;border-radius:5px;font-size:.83rem;color:var(--navy)}
.legend{display:flex;flex-wrap:wrap;gap:9px 19px;font-size:.85rem;color:var(--muted);margin-top:.45em}.legend span i{display:inline-block;width:11px;height:5px;margin-right:6px;vertical-align:middle;border-radius:2px}.thin{max-width:800px;margin:auto}.table-wrap td{font-size:.86rem}.heat th{min-width:58px;text-align:center;white-space:nowrap}.heat tbody th{text-align:left}.heat td{text-align:center;min-width:49px;padding:7px 9px}.compact-list{padding-left:20px}.compact-list li{margin-bottom:.5rem}.table-wrap input{width:auto}.small{font-size:.83rem;color:var(--muted)}
@media(max-width:800px){.grid{grid-template-columns:minmax(0,1fr)}.filter-set{gap:7px 12px}.table-wrap th,.table-wrap td{white-space:nowrap}.table-wrap td.wrapped{white-space:normal;min-width:220px}}
'''


SCRIPT=r'''
const D=JSON.parse(document.getElementById('research-data').textContent),$=id=>document.getElementById(id);
const Y=[2024,2025,2026], M=['A','B','B*'], C={A:'#335f91',B:'#08766c','B*':'#795b93',second:'#788ba0',primary:'#08766c'}, W=['Expanding','Rolling 5Y'];
const names={0:'Development 2010–23',2024:'Test 2024',2025:'Test 2025',2026:'Test 2026 (Partial)'};
const A=JSON.parse(document.getElementById('factor-names').textContent), lm=Object.keys(A);
const e=s=>String(s??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const fmt=(x,n=4)=>x==null?'—':x.toFixed(n).replace('-','−'),delta=x=>x!=null&&x>0?'+'+fmt(x):fmt(x),pct=x=>x==null?'—':fmt(x*100,2);
const nField=r=>`${r.events} calls · ${r.issuers} issuers`, timeLabel=h=>`CAR[+2,+${h}]`;
const get=(t,k)=>D[t].filter(r=>Object.entries(k).every(([p,v])=>r[p]===v));
function table(id,keys,rows,style=''){
 const body=rows.map(r=>`<tr>${r.map((v,i)=>`<td${i===3?' class="wrapped"':''}>${e(v)}</td>`).join('')}</tr>`).join('');
 $(id).innerHTML=`<div class="table-wrap"><table class="${style}"><thead><tr>${keys.map(k=>`<th scope="col">${e(k)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function mark(x,y,c,txt,bar){
 const symbol=bar?`<rect x="${x-8}" y="${Math.min(y,bar.zero)}" width="16" height="${Math.max(Math.abs(y-bar.zero),.7)}" rx="1" fill="${c}"/>`:`<circle cx="${x}" cy="${y}" r="4.2" fill="${c}" stroke="white" stroke-width="1.1"/>`;
 return `<g tabindex="0" data-tip="${e(txt)}" aria-label="${e(txt)}"><title>${e(txt)}</title>${symbol}<circle cx="${x}" cy="${y}" r="9" fill="transparent"/></g>`;
}
function live(id){
 $(id+'-tip').textContent='Focus or hover over a mark for its value.';
 for(const g of $(id).querySelectorAll('[data-tip]')) for(const k of ['focus','mouseenter','click']) g.addEventListener(k,()=>{$(id+'-tip').textContent=g.dataset.tip;});
}
function chart(id,labels,series,lim,units,kind='line'){
 const left=60,right=480,slot=(right-left)/labels.length,y=v=>127-v*92/lim;
 const x=i=>kind==='bar'?left+slot*(i+.5):left+11+i*(right-left-22)/Math.max(labels.length-1,1);
 const axis=[-lim,-lim/2,0,lim/2,lim].map(v=>`<path d="M${left} ${y(v)}H${right}" stroke="${v===0?'#7d8d9a':'#d6e3e8'}"/><text x="51" y="${y(v)+4}" text-anchor="end">${fmt(v,lim>=1?1:lim<=.1?3:2)}</text>`).join('');
 const values=series.map((s,j)=>s.vals.map((a,i)=>{
  if(!a||a.v==null)return '';
  const xp=x(i)+(kind==='bar'?(j-(series.length-1)/2)*20:0), yp=y(a.v);
  const prev=s.vals[i-1], line=kind==='line'&&prev&&prev.v!=null?`<path d="M${x(i-1)} ${y(prev.v)}L${xp} ${yp}" stroke="${C[s.key]}" stroke-width="2.1"/>`:'';
  return line+mark(xp,yp,C[s.key],a.tip,kind==='bar'?{zero:y(0)}:null);
 }).join('')).join('');
 $(id).innerHTML=`<svg viewBox="0 0 505 256" aria-label="${e(units)}"><g fill="#496074">${axis}<text x="60" y="18">${e(units)}</text>${labels.map((l,i)=>`<text x="${x(i)}" y="241" text-anchor="middle">${e(l)}</text>`).join('')}</g>${values}</svg>`;
 live(id);
}
function intervalPlot(id,data){
 const groups=[...new Set(data.map(r=>r.label))], bounds=data.flatMap(r=>[r.lower,r.upper,r.point]).filter(x=>x!=null);
 const m=Math.max(.1,Math.ceil(Math.max(...bounds.map(Math.abs))*10)/10),x=v=>99+(v+m)*370/(2*m), tall=58+groups.length*42;
 const ticks=[-m,-m/2,0,m/2,m].map(v=>`<path d="M${x(v)} 26V${tall-24}" stroke="${v===0?'#7d8d9a':'#d6e3e8'}"/><text text-anchor="middle" x="${x(v)}" y="${tall-8}">${fmt(v,2)}</text>`).join('');
 const points=data.map(r=>{
  if(r.point==null)return '';
  const y=43+groups.indexOf(r.label)*42+(r.method==='Issuer cluster'?-6:6);
  const band=r.available?`<path d="M${x(r.lower)} ${y}H${x(r.upper)}" stroke="${C.primary}" stroke-width="2" ${r.method==='Two-month block'?'stroke-dasharray="5 4"':''}/>`:'';
  return band+mark(x(r.point),y,C.primary,`${r.label} · ${r.method} · ${delta(r.point)} · ${r.available?bracket(r):'interval unavailable'} · ${nField(r)}`);
 }).join('');
 $(id).innerHTML=`<svg viewBox="0 0 500 ${tall}" aria-label="Paired Rank IC intervals"><g fill="#496074"><text x="99" y="15">Δ Rank IC</text>${ticks}${groups.map((g,i)=>`<text x="9" y="${47+42*i}">${e(g)}</text>`).join('')}</g>${points}</svg>`;
 live(id);
}
const bracket=r=>r&&r.available?`[${fmt(r.lower)}, ${fmt(r.upper)}]`:'—';
function update(){
 const year=Number($('year').value), benchmark=$('benchmark').value,horizon=Number($('horizon').value),window=$('window').value, factor=$('factor').value;
 const key={year,benchmark,horizon}, fit={...key,window}, scope=$('factor-scope').value==='dev'?0:year, raw={...key,year:scope};
 for(const el of document.querySelectorAll('.active-context')) el.textContent=`${names[year]} · ${benchmark} · ${timeLabel(horizon)} · ${window}`;
 const w=get('window_comparison',key); $('window-context').textContent=`${names[year]} · ${benchmark} · ${timeLabel(horizon)} · ${nField(w[0])}`;
 chart('window-plot',M,[{key:'second',vals:w.map(r=>({v:r.expanding_rank_ic,tip:`Expanding ${r.model} · ${fmt(r.expanding_rank_ic)} · ${nField(r)}`}))},
 {key:'primary',vals:w.map(r=>({v:r.rolling_rank_ic,tip:`Rolling 5Y ${r.model} · ${fmt(r.rolling_rank_ic)} · ${nField(r)}`}))}],.45,'Rank IC','bar');
 table('window-table',['Model','Expanding','Rolling 5Y','Rolling − Expanding'],w.map(r=>[r.model,fmt(r.expanding_rank_ic),fmt(r.rolling_rank_ic),delta(r.rolling_minus_expanding)]));
 const tr=get('training_cohorts',key);
 table('train-table',['Window','Training calls','Training issuers','Refit cutoff UTC','Purged at refit'],tr.map(r=>[r.window,r.train_events,r.train_issuers,r.refit_cutoff,r.purged_at_refit]));
 const h=get('horizon_comparison',{year,benchmark,window});$('horizon-context').textContent=`${names[year]} · ${benchmark} · ${window} · ${nField(h[0])} matched across horizons`;
 chart('horizon-plot',M,[{key:'second',vals:h.map(r=>({v:r.short_rank_ic,tip:`Short ${r.model} · ${fmt(r.short_rank_ic)} · ${nField(r)}`}))},
 {key:'primary',vals:h.map(r=>({v:r.long_rank_ic,tip:`Long ${r.model} · ${fmt(r.long_rank_ic)} · ${nField(r)}`}))}],.45,'Rank IC','bar');
 table('horizon-table',['Model','Short IC','Long IC','Long − Short','Short LM vs A','Long LM vs A'],h.map(r=>[r.model,fmt(r.short_rank_ic),fmt(r.long_rank_ic),delta(r.long_minus_short),r.model==='A'?'—':delta(r.short_increment),r.model==='A'?'—':delta(r.long_increment)]));
 table('cohort-table',['Horizon ending','Full events','Full issuers','Matched events','Matched issuers'],get('horizon_cohorts',{year,benchmark}).map(r=>['+'+r.horizon,r.full_events,r.full_issuers,r.matched_horizons_events,r.matched_horizons_issuers]));
 const p=get('annual_model_performance',{benchmark,horizon,window,sample:'Full horizon'});
 chart('annual-plot',Y.map(y=>y===2026?'2026 partial':String(y)),M.map(m=>({key:m,vals:p.filter(r=>r.model===m).map(r=>({v:r.rank_ic,tip:`${names[r.year]} ${m} · ${fmt(r.rank_ic)} · ${nField(r)}`}))})),.45,'Annual Spearman Rank IC','bar');
 table('annual-table',['Test','Model','Rank IC','Pearson (suppl.)','Calls','Issuers'],p.map(r=>[names[r.year],r.model,fmt(r.rank_ic),fmt(r.pearson),r.events,r.issuers]));
 const lm_year=get('lm_increments',{...key,sample:'Full horizon'});
 const lm_all=get('lm_increments',{benchmark,horizon,window,sample:'Full horizon'});
 chart('lm-plot',Y.map(y=>y===2026?'2026 partial':String(y)),M.slice(1).map(m=>({key:m,vals:lm_all.filter(r=>r.model===m).map(r=>({v:r.delta_rank_ic,tip:`${names[r.year]} ${m} − A · ${delta(r.delta_rank_ic)} · ${nField(r)}`}))})),.09,'LM incremental Rank IC','bar');
 table('lm-table',['Window','Contrast','Δ Rank IC','Calls','Issuers'],lm_year.map(r=>[r.window,`${r.model}−A`,delta(r.delta_rank_ic),r.events,r.issuers]));
 const factors=get('factor_ic',raw);
 $('factor-context').textContent=`${names[scope]} · ${benchmark} · ${timeLabel(horizon)} · Observed factor and CAR pairs`;
 const fin=get('factor_inference',raw);
 chart('factor-plot',lm.map((_,i)=>`LM ${i+1}`),[{key:'primary',vals:factors.map(r=>({v:r.rank_ic,tip:`${A[r.factor]} · ${fmt(r.rank_ic)} · ${nField(r)}`}))}],.35,'Single-factor Rank IC','bar');
 table('factor-table',['Factor','Rank IC','Calls / eligible','Issuers','Issuer CI','Time-block CI'],factors.map((r,i)=>[`${i+1}. ${A[r.factor]}`,fmt(r.rank_ic),`${r.events} / ${r.eligible_events}`,r.issuers,bracket(fin.find(x=>x.factor===r.factor&&x.method==='Issuer cluster')),bracket(fin.find(x=>x.factor===r.factor&&x.method==='Two-month block'))]));
 const qst=get('factor_stability',{...key,factor});
 $('stability-context').textContent=`${names[year]} · ${benchmark} · ${timeLabel(horizon)} · ${A[factor]}`;
 chart('stability-plot',qst.map(r=>r.quarter.slice(-2)+(r.partial?'*':'')+' · n='+r.events),[{key:'primary',vals:qst.map(r=>({v:r.rank_ic,tip:`${r.quarter} · ${fmt(r.rank_ic)} · ${nField(r)}`}))}],.60,'Quarterly Rank IC');
 const annualFactor=get('factor_ic',{benchmark,horizon,factor}).filter(r=>r.year!==0);
 chart('annual-factor-plot',Y.map(y=>y===2026?'2026 partial':String(y)),[{key:'primary',vals:annualFactor.map(r=>({v:r.rank_ic,tip:`${names[r.year]} ${A[factor]} · ${fmt(r.rank_ic)} · ${nField(r)}`}))}],.35,'Annual factor Rank IC');
 table('condition-table',['Observable condition','Rank IC','Calls','Issuers'],get('conditional_factor_ic',{...raw,factor}).map(r=>[r.condition,fmt(r.rank_ic),r.events,r.issuers]));
 const modelq=get('model_quintiles',{...fit,sample:'Full horizon'}), factorq=get('factor_quintiles',{...raw,factor});
 const qLim=Math.ceil(Math.max(...D.model_quintiles.map(d=>Math.abs(d.mean_car*100)),...D.factor_quintiles.map(d=>Math.abs(d.mean_car*100))));
 const index=['Q1','Q2','Q3','Q4','Q5'];
 const makeQ=(r,label)=>({v:r.mean_car*100,tip:`${label} Q${r.quintile} · ${pct(r.mean_car)}% · n=${r.n}`});
 chart('model-q',index,M.map(m=>({key:m,vals:modelq.filter(r=>r.model===m).map(r=>makeQ(r,m))})),qLim,'Mean realized CAR (%)');
 chart('factor-q',index,[{key:'primary',vals:factorq.map(r=>makeQ(r,A[factor]))}],qLim,'Mean realized CAR (%)');
 $('quant-factor').textContent=`Raw factor sort · ${A[factor]} · ${names[scope]}`;
 const perf=p.filter(r=>r.year===year).map(r=>[r.model,modelq.filter(q=>q.model===r.model).map(q=>q.n).join(' / ')||'—',pct(r.q5_minus_q1),r.monotonicity]);
 const fq=factors.find(f=>f.factor===factor);
 perf.push([A[factor],factorq.map(r=>r.n).join(' / ')||'—',pct(fq.q5_minus_q1),fq.monotonicity]);
 table('q-table',['Scores','Q1 / Q2 / Q3 / Q4 / Q5 counts','Q5−Q1 (pp)','Shape'],perf);
 const block=$('heat-kind').value, corr=get('correlations',{...raw,block}), labels=[...new Set(corr.map(r=>r.feature))];
 const shade=v=>v==null?'#eef2f5':`hsl(${v<0?267:164} 41% ${96-Math.abs(v)*27}%)`;
 $('heat').innerHTML=`<div class="table-wrap"><table class="heat"><thead><tr><th></th>${labels.map(k=>`<th>${e(A[k]||k)}</th>`).join('')}</tr></thead><tbody>${lm.map(f=>`<tr><th>${e(A[f])}</th>${labels.map(k=>{const r=corr.find(x=>x.factor===f&&x.feature===k);return `<td style="background:${shade(r.rank_corr)}" title="${r.n} observed pairs">${fmt(r.rank_corr,2)}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;
 table('missing-table',['Predictor','Observed','Events','Missing share'],get('feature_missingness',raw).map(r=>[r.feature,r.events-r.missing,r.events,pct(r.missing_share)+'%']));
 const chosen=get('selected_features',fit);
 table('choice-table',['Model','Strategy','Estimator','Selected LM inputs','Inner CV IC'],chosen.map(r=>[r.model,r.route,r.estimator,r.features?.split('|').map(f=>A[f]).join(', ')||'—',fmt(r.inner_rank_ic)]));
 const steps=get('feature_selection_paths',fit).filter(r=>r.accepted||r.step===0);
 table('path-table',['Route','Step','Candidate','Subset','Inner CV IC','Gain'],steps.map(r=>[r.route,r.step,A[r.feature]||'—',r.features??'—',fmt(r.inner_rank_ic),delta(r.gain)]));
 const cv=get('estimator_cv',fit).filter(r=>chosen.some(q=>(q.features?.split('|').sort().join('|')||'(controls only)')===r.features.split('|').sort().join('|')));
 table('estimator-table',['Subset','Estimator','Mean IC','Fold 1','Fold 2','Fold 3'],cv.map(r=>[r.features,r.estimator,fmt(r.mean_rank_ic),fmt(r.fold_1),fmt(r.fold_2),fmt(r.fold_3)]));
 table('importance-table',['Model','Factor','Coefficient / train n','Absolute coefficient share'],get('feature_importance',{...fit,block:'LM'}).map(r=>[r.model,A[r.feature],fmt(r.scaled_by_train_n),pct(r.absolute_share)+'%']));
 const family=$('inference-kind').value, def=family==='Horizon'?{year,benchmark,family,comparison:window}:family==='LM'?{...key,family,comparison:window}:{...key,family};
 const inf=get('paired_inference',def).map(r=>({...r,label:family==='LM'?r.model+'−A':r.model}));
 intervalPlot('inference-plot',inf);
 $('inference-context').textContent=`${names[year]} · ${benchmark} · ${family==='Horizon'?'matched horizons':timeLabel(horizon)} · ${nField(inf[0])}`;
 table('inference-table',['Contrast','Method','Point','95% CI','Valid draws','Nominal blocks'],inf.map(r=>[r.label,r.method,delta(r.point),bracket(r),r.draws,r.nominal_blocks]));
}
for(const s of document.querySelectorAll('select'))s.addEventListener('change',update);
update();
'''


def make_html(out,D,spec):
    phase=(HERE/'phase1_style.css').read_text(encoding='utf-8')
    ids=['overview','windows','horizons','ranking','incremental','factors','stability','quintiles',
         'correlation','selection','inference','methodology']
    nav=['Overview','Training Window','Prediction Horizon','Rank IC','Incremental LM','Single Factors','Stability',
         'Quintiles','Correlations','Selection','Inference','Methodology']
    links=''.join(f'<a href="#{i}">{n}</a>' for i,n in zip(ids,nav))
    n=filt(D['horizon_cohorts'],benchmark='Healthcare',horizon=20)
    tiles=''.join(f'<div><strong>{r.matched_horizons_events}</strong><span>Test {yearname(r.year)} matched calls</span></div>'
                  for r in n.itertuples())+'<div><strong>8</strong><span>Nonredundant LM inputs</span></div>'

    def select(id,label,items,chosen=None):
        options=''.join(f'<option value="{escape(str(k))}"{" selected" if k==chosen else ""}>{escape(value)}</option>' for k,value in items)
        return f'<label for="{id}">{label}<select id="{id}">{options}</select></label>'

    def figure(id,title,caption):
        return (f'<figure><h3>{title}</h3><div id="{id}"></div><figcaption>{caption}</figcaption>'
                f'<span class="tip" id="{id}-tip" aria-live="polite"></span></figure>')

    def legend(entries):
        return '<div class="legend">'+''.join(f'<span><i style="background:{c}"></i>{m}</span>' for m,c in entries.items())+'</div>'

    color_legend=legend(COLOR)
    y=select('year','Test year',[(k,'Test '+yearname(k)) for k in YEARS],2025)
    b=select('benchmark','Benchmark',[(k,k) for k in OPTIONS],'Healthcare')
    h=select('horizon','Target',[(k,f'CAR[+2,+{k}]') for k in (5,20)],20)
    w=select('window','Training window',[(k,k) for k in WINDOWS],'Expanding')
    fs=select('factor-scope','Factor cohort',[('test','Selected test year'),('dev','Development 2010–23')])
    f=select('factor','LM factor',list(ALIASES.items()),'tone_answers')
    corr=select('heat-kind','Pair block',[(k,k) for k in ('LM × LM','LM × controls')])
    inf=select('inference-kind','Paired contrast',[('Window','Rolling − Expanding'),('Horizon','Long − Short'),('LM','LM minus A')])
    notes=''.join(f'<li>{escape(line)}</li>' for line in conclusions(D))
    files=''.join(f'<li><a href="{name}.csv">{name.replace("_"," ").title()} CSV</a></li>' for name in NAMES)
    files+=f'<li><a href="{PDF}">Research PDF</a></li><li><a href="methodology.md">Methodology</a></li><li><a href="frozen_research_spec.json">Frozen specification</a></li><li><a href="data_provenance.json">Input hashes</a></li><li><a href="summary_zh.md">中文摘要</a></li>'
    payload={k:frame.astype(object).where(frame.notna(),None).to_dict(orient='records') for k,frame in D.items()}
    blob=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    aliases=json.dumps(ALIASES,ensure_ascii=False)
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{BRAND} | {TITLE}</title><style>{phase}{CSS}</style></head>
<body><a class="skip" href="#content">Skip to research</a><header class="hero"><div class="wrap"><p class="brand">{BRAND}</p><h1>{TITLE}</h1><p class="intro">{SUBTITLE}</p><div class="stats">{tiles}</div></div></header>
<nav aria-label="Research sections"><div class="wrap">{links}</div></nav>
<main class="wrap" id="content"><section class="section" id="overview"><h2>{SECTIONS[0]}</h2><p>Compare expanding and five-year rolling training, and post-call CAR horizons ending at Day +5 or +20. Model A uses the common control block, B selects LM inputs in historical temporal validation, and B* uses all eight retained LM inputs.</p>
<div class="filter-set" aria-label="Research filters">{y}{b}{h}{w}</div><p class="badge-info active-context" aria-live="polite"></p><p class="note">The outer years were examined in earlier research. The new model fits and feature decisions use only information from earlier calls and mature labels.</p></section>
<section class="section" id="windows"><h2>{SECTIONS[1]}</h2><p>Rolling versus Expanding on the full eligible cohort for the selected benchmark and horizon.</p><p class="note" id="window-context"></p>{legend({'Expanding':'#788ba0','Rolling 5Y':'#08766c'})}
<div class="grid">{figure('window-plot','Rank IC by training window','Each model pair shares event IDs and realized CAR.')}<div id="window-table"></div></div><details><summary>Training cohort, cutoff and label maturity</summary><div id="train-table"></div></details></section>
<section class="section" id="horizons"><h2>{SECTIONS[2]}</h2><p>CAR[+2,+5] versus CAR[+2,+20] on their common calls. Each forecast is scored against its respective horizon.</p><p class="note" id="horizon-context"></p>{legend({'CAR[+2,+5]':'#788ba0','CAR[+2,+20]':'#08766c'})}
<div class="grid">{figure('horizon-plot','Rank IC by prediction horizon','Both horizon scores are recomputed on the intersection.')}<div id="horizon-table"></div></div><details><summary>Full and matched horizon coverage</summary><div id="cohort-table"></div></details></section>
<section class="section" id="ranking"><h2>{SECTIONS[3]}</h2><p class="note active-context"></p>{color_legend}<div class="grid">{figure('annual-plot','Annual model correlations','Figure 1. Full-horizon calls. Each year is evaluated independently.')}<div id="annual-table"></div></div></section>
<section class="section" id="incremental"><h2>{SECTIONS[4]}</h2><p class="note active-context"></p>{legend({'B−A':COLOR['B'],'B*−A':COLOR['B*']})}<div class="grid">{figure('lm-plot','Rank IC differences from Model A','Rank IC is dimensionless. Zero is the Model A reference.')}<div id="lm-table"></div></div></section>
<section class="section" id="factors"><h2>{SECTIONS[5]}</h2><div class="filter-set">{fs}{f}</div><p class="note" id="factor-context"></p>{figure('factor-plot','Raw LM8 Spearman Rank IC','Figure 2. Rank each observed factor against CAR. Pairwise coverage and 95% intervals follow.')}
<div id="factor-table"></div><p class="note">A blank interval denotes an unavailable test interval. Development factor tables and outer-year measurements have separate roles.</p></section>
<section class="section" id="stability"><h2>{SECTIONS[6]}</h2><p class="note" id="stability-context"></p><div class="grid">{figure('stability-plot','Quarterly factor Rank IC','Figure 3. Counts appear on the x-axis. An asterisk marks the partially observed quarter.')}{figure('annual-factor-plot','Annual factor Rank IC','Each point ranks the selected factor within a single test year.')}</div><h3>Observable financial conditions</h3><div id="condition-table"></div><p class="note">The SUE split uses zero. The pre-volatility split uses the development median. Conditions use the factor cohort selector.</p></section>
<section class="section" id="quintiles"><h2>{SECTIONS[7]}</h2><p class="note active-context"></p>{color_legend}<div class="grid">{figure('model-q','Model prediction sorts','Figure 4. Mean realized CAR by model score. Q1 is the lowest score group.')}{figure('factor-q','Raw LM factor quintiles','The raw factor panel uses the selected factor and factor cohort.')}</div>
<p id="quant-factor" class="note"></p><div id="q-table"></div><p class="note">Event windows may overlap. Spreads are event-level differences in percentage points.</p></section>
<section class="section" id="correlation"><h2>{SECTIONS[8]}</h2><p>Figure 5. Observed-pair Spearman correlations. Purple denotes negative association. Teal denotes positive association. The selected factor cohort applies.</p><div class="filter-set">{corr}</div><div id="heat"></div>
<p class="badge-info">The full Healthcare input confirms equal Q&amp;A hedging and uncertainty values and missingness. The experiment retains uncertainty_answers in LM8.</p><details><summary>Missingness before within-fold imputation</summary><div id="missing-table"></div></details></section>
<section class="section" id="selection"><h2>{SECTIONS[9]}</h2><p class="note active-context"></p><div id="choice-table"></div><p class="note">Figure 6 in the PDF summarizes selection frequencies. These tables expose the individual fit and accepted decisions. The path CSV contains every tried candidate.</p>
<details open><summary>Train-only selection paths</summary><div id="path-table"></div></details><details><summary>Estimator comparisons on inner folds</summary><div id="estimator-table"></div></details><details><summary>Standardized-input LM coefficients</summary><div id="importance-table"></div><p class="note">Signed coefficients are divided by training sample size because target ranks scale with n. Absolute shares refer to the fitted coefficient vector.</p></details></section>
<section class="section" id="inference"><h2>{SECTIONS[10]}</h2><p>Pointwise 95% intervals use paired issuer-cluster and two-month-block draws. Intervals condition on the fitted forecasts and remain exploratory.</p><div class="filter-set">{inf}</div><p id="inference-context" class="note"></p><div class="grid">{figure('inference-plot','Paired Δ Rank IC','Figure 7. Solid issuer-cluster intervals. Dashed time-block intervals. A point can appear without an interval.')}<div id="inference-table"></div></div><p class="note">Partial 2026 has insufficient time blocks. The issuer intervals address repeated issuers and cannot substitute for calendar-time dependence.</p></section>
<section class="section" id="methodology"><h2>{SECTIONS[11]}</h2><ul class="compact-list">{notes}</ul>
<details><summary>Timing, label maturity and interpretation</summary><p>Predictions use the original Day 0 mapping and occur after the conservative Day +1 close and before the Day +2 session open. CAR uses close-to-close daily returns. The first return includes overnight repricing after the preceding close. These event results cannot establish executable returns. Short horizons may capture immediate repricing. Longer horizons may capture slower incorporation and other price movement.</p></details>
<details><summary>Research tables and reproducibility</summary><ul class="downloads">{files}</ul></details></section></main><footer><div class="wrap">HEALTHCARE RESEARCH / ANNUAL WINDOW AND HORIZON COMPARISONS / CRSP THROUGH 30 JUN 2026</div></footer>
<script id="factor-names" type="application/json">{aliases}</script><script id="research-data" type="application/json">{blob}</script><script>{SCRIPT}</script></body></html>'''
    (out/'dashboard.html').write_text(html,encoding='utf-8')


def make_notes(out,D):
    docs=r'''# Healthcare Linguistic Alpha Research

## Frozen design

The two experiments compare annually expanding versus rolling five-year training, and separately compare CAR[+2,+5] versus CAR[+2,+20]. Models are fitted separately by test year, window, target and benchmark. The frozen research specification was saved before scoring. The outer tests are retrospective because past results for these calendar years were previously examined. The new implementation shares controls across the three models and makes all selections within historical training folds.

## Data and label validation

The original native Healthcare option tables supply 2010–24 covariates and targets, with the original 2024 data vintage. Later calls come from the validated financial feature reconstruction. New call eligibility requires verified CIQ/security links and dated S&P 500 membership. For each benchmark, the full eligible short/long counts are 205/204 in 2024, 211/211 in 2025, and 106/104 in partial 2026. Direct horizon comparisons use their intersection. Window comparisons use every eligible event for the selected horizon. The comparison code checks event and target equality before scoring.

Both labels sum stock-minus-benchmark daily returns for sessions +2 to +5 or +2 to +20. The S&P 500 price benchmark uses SPRTRN. The Healthcare return uses the original static GICS 35 basket, dated membership and same-day market capitalization weights. Native labels preserve original availability and delisting conventions. New sector short CAR uses original validated daily sector returns and the same CIZ stock return type used in the long benchmark. The short target replicates the 205 archived 2024 sector labels from legacy data and is verified with a separate four-session join on each new eligible event. The rebuilt modern Healthcare long labels match the saved series. Both modern market labels match the validated extension. See `target_validation.csv` for overlaps, errors and tolerances. `data_provenance.json` hashes source files and the analysis scripts.

## Clock and fit eligibility

Retain the original 16:00 ET Day 0 mapping and CRSP trading dates. Record the conservative Day +1 close at 16:00 ET, the Day +2 scheduled session open, and Day +5 and +20 close timestamps. A 16:00 bound intentionally waits past an early close. A usable prediction is timestamped one microsecond after that bound and precedes the Day +2 session open. This is an ordering convention for the event study. The first close-to-close daily return begins at the previous close and includes overnight movement that may be unavailable to a post-close signal. Accordingly, a live implementation could not capture the entire first return under this clock. The assumed transcript delivery, actual scoring latency and all-vintage financial data availability are not verified. The research does not apply a transcript timing screen.

Use a common annual refit boundary at the earliest eligible Day +1 close bound. Training calls must predate the test year, have observed Early CAR and a valid target, and mature strictly before refit. Each historical validation year has its own cutoff at that year's earliest eligible validation Day +1 close bound. Remove any inner-fit label mature at or after that cutoff. All inner-validation outcomes mature before the annual refit. The timing audit and fold checks exercise strict equality boundary cases. Rolling uses the previous five calendar years. The otherwise comparable expanding history starts in 2010. Training labels are required to have a known completed session window. No model uses either prediction horizon CAR, post-label volatility or the other horizon's outcome as a feature.

## Controls, LM8 and model selection

The sixteen controls are SUE, relative surprise, pre-event volatility, Early CAR, momentum, book-to-market, size, analyst dispersion, average answer length, prepared/question/answer word counts, question and answer counts, analyst count and prior calls observed. The latter counts earlier calls by archived PERMNO irrespective of whether they later have a target. This replaces the archive's target-availability count and keeps it independent of future label availability. Later financial values use the saved reconstruction and its validated source-specific definitions. Early CAR corresponds to the same benchmark as the target.

Verify exact equality, including missingness, between hedging_answers and uncertainty_answers. Keep uncertainty_answers. The other LM inputs are tone_prepared, tone_answers, hedging_prepared, tone_gap, tone_prepared_chg, tone_answers_chg and hedging_answers_chg. All source variables remain intact.

Use the last three eligible calendar years as expanding inner validation folds within each training schedule. Fit original log transforms, quantile winsorization, median replacement and scaling separately in each fold, then form average target midranks in training. Unavailable columns within a fold contribute zero after standardization. Fit four candidate estimators, Ridge at α=1, 10 and 50 and Elastic Net α=0.05, L1 ratio 0.7. Model A and B* choose estimators on the same folds. B compares the forward route and jointly fitted univariate-screened subset with A. Minimum mean IC gain is 0.001. Prespecified tie rules follow feature and estimator order, then subset size and route. B falls back to the identical A estimator and inputs if neither candidate clears the threshold. Train-only folds and scores determine every choice. The fitted estimator then scores the year's held-out predictors once. The comparison records the estimator, subset, inner IC and both accepted and rejected steps.

## Factor analysis and inference

Development-factor calculations use mature pre-2024 calls. Outer-factor analysis works separately by test year, benchmark and horizon. Factors use observed values. Spearman correlations, factor sorts, conditional groups, quarterly stability and missingness are reported separately from model predictions. Pre-volatility conditions use the applicable development median and SUE conditions use zero. Quarterly estimates require 25 calls and 15 issuers. Conditional estimates require 30 calls and 15 issuers. Correlations use pairwise observations, with a minimum of 20. No factor result from an outer year is passed back to the selector.

Spearman Rank IC is primary, with Pearson supplementary. Q1–Q5 sorts use model scores or raw factor inputs. CAR is in decimal returns in CSV and in percent in plots, while Q5−Q1 appears in percentage points. Event windows may overlap. No live portfolio result is estimated. Signed importance is a standardized-input regression coefficient divided by training n, with a supplementary absolute coefficient share. This is a description of a joint model fit.

Pointwise paired percentile intervals use 999 draws by issuer or overlapping noncircular two-month call-date blocks. Resampling preserves both predictions and any outcome pair on each sampled event. Methods address different dependence structures and neither fully addresses both at once. At least 50 events, 30 issuers and 95% valid draws are required. Time-block intervals also require five nominal disjoint blocks. Partial 2026 has three and therefore shows issuer intervals when eligible and time-block estimates without intervals. Many contrasts are exploratory. No multiplicity-adjusted confirmatory claims are made.

The local bridge methodology and its saved IT Sector reports were reviewed as references for the chronological decision design. This experiment uses the verified Healthcare inputs. The previous annual evaluation, final report and Phase 1 reference remain separate.

## Reproduction and file layout

Run from the repository root with the licensed source paths in `data_provenance.json` available locally. Fit and audit with the project's Python 3.12 environment containing pandas, NumPy, SciPy, PyArrow and scikit-learn. Render with a Python environment containing pandas, NumPy and Matplotlib. Provide the same `--out` folder to both scripts to rebuild elsewhere. The renderer reads the archived Phase 1 stylesheet in `phase1_style.css` and the saved aggregate files.

```powershell
$entry = 'outputs/healthcare_window_horizon_comparison'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe "$entry/run_comparison.py"
.\.venv\Scripts\python.exe -m unittest discover -s $entry -p test_comparison.py
python "$entry/render_comparison.py"
```

`training_cohorts.csv`, `cv_folds.csv`, `target_validation.csv` and `horizon_cohorts.csv` contain coverage and maturity evidence. `window_comparison.csv`, `horizon_comparison.csv`, `annual_model_performance.csv`, `lm_increments.csv` and `paired_inference.csv` provide the primary estimates. `model_quintiles.csv` and the factor tables distinguish model prediction sorts from raw factor sorts. `selected_features.csv`, `feature_selection_paths.csv`, `estimator_cv.csv` and `feature_importance.csv` preserve all training decisions. Event-level forecast and timing ledgers are local outputs without dashboard download links. No licensed observations are duplicated into the PDF or dashboard.
'''
    (out/'methodology.md').write_text(docs,encoding='utf-8')
    rows=['# Healthcare 训练窗口与预测期限比较','',
          '| 年度 | Rolling 优于 Expanding | 长期优于短期 | B−A 为正 | B*−A 为正 |',
          '|---|---:|---:|---:|---:|']
    for y,w,h,b,star in count_table(D):
        rows.append(f'| {yearname(y)} | {w}/12 | {h}/12 | {b}/8 | {star}/8 |')
    rows+=['','前两列分别统计两种基准、两种期限或训练窗口、三个模型。后两列统计每种年度下两种基准、两种期限和两个训练窗口中的正 ΔRank IC。每个结果用其年度的匹配样本计算。','',
           '短期和长期共同样本为 2024 年 204 场、2025 年 211 场、2026 部分年度 104 场。六个时间边界均基于 CRSP 交易日。外层拟合和内层验证只使用截点前成熟的训练标签。','']
    picks=filt(D['selected_features'],model='B'); cnt=picks.features.dropna().str.split('|').explode().value_counts()
    a=filt(D['paired_inference'],family='LM',method='Issuer cluster'); excludes=int((a.lower.gt(0)|a.upper.lt(0)).sum())
    rows+=[f'准备稿语调变化和回答措辞变化分别入选 {cnt.get("tone_prepared_chg",0)} 和 {cnt.get("hedging_answers_chg",0)} 次。{int(picks.feature_count.eq(0).sum())} 次 B 拟合只保留控制变量。{len(a)} 个 LM 发行人聚类区间中有 {excludes} 个排除零。经济解释应区分即时重定价、后续信息消化和其他波动。','',
           '这是事后回顾性研究。2026 年时间块过少。发行人聚类区间只能反映其对应的依赖结构。预测时钟在 Day +1 收盘后及 Day +2 开盘前，首个 CAR 日收益包含延后信号无法完整捕捉的隔夜变化。','',
           '[仪表板](dashboard.html) · [窗口对照](window_comparison.csv) · [期限对照](horizon_comparison.csv) · [增量](lm_increments.csv) · [方法](methodology.md)']
    (out/'summary_zh.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,default=HERE)
    out=cli.parse_args().out.resolve()
    data={k:pd.read_csv(out/f'{k}.csv',float_precision='round_trip') for k in NAMES}
    spec=json.loads((out/'frozen_research_spec.json').read_text(encoding='utf-8'))
    make_pdf(out,data,spec)
    make_html(out,data,spec)
    make_notes(out,data)


if __name__=='__main__':
    main()
