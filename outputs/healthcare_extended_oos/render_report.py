"""Render publication PDF, interactive HTML and Chinese summary from saved aggregates."""

from html import escape
from io import StringIO
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TITLE = 'HEALTHCARE LINGUISTIC ALPHA RESEARCH'
SUBTITLE = 'Original Sector Pipeline — Extended Out-of-Sample Evaluation'
PDF = 'Healthcare_Extended_OOS_Research_Report.pdf'
NAVY, TEXT, MUTE, LINE, SOFT = '#12253D', '#293B4C', '#4F6373', '#CBD7DF', '#EFF4F6'
TEAL, WHITE, PALE = '#007E79', '#FFFFFF', '#DCEEEB'
COLOR = dict(A='#244D77', B='#007E79', **{'B*': '#775A87'})
OPT = dict(market='S&P 500-adjusted', sector='Healthcare sector-adjusted')
MODELS = tuple(COLOR)
PERIOD = ['2024 diagnostic', '2025', '2026 partial', '2025–26 pooled']
SHORT = ['2024 diag.', '2025', '2026 partial', '2025–26 pooled']
MET, INC, INF, Q, COVER, FEATURE, VAL, RPL, QUARTER, STAB = [pd.read_csv(HERE / (f+'.csv')) for f in (
    'model_performance', 'incremental_comparison', 'bootstrap_inference', 'quintile_returns',
    'target_coverage', 'feature_coverage', 'sector_benchmark_validation', 'fit_replay_validation',
    'quarterly_ic', 'stability_summary')]
SPEC = json.loads((HERE/'frozen_experiment_spec.json').read_text(encoding='utf-8'))
OPTIONS = tuple(MET.option.unique())
POOLED = PERIOD[-1]
METHOD = dict(issuer_cluster='Issuer', time_block='Time block')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'healthcare-oos'})


def pick(frame, **criteria):
    return frame.loc[np.logical_and.reduce([frame[name].eq(value) for name, value in criteria.items()])]


def f(value, digits=3):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'.replace('-', '−')


def signed(value, digits=3):
    return '—' if pd.isna(value) else f'{value:+.{digits}f}'.replace('-', '−')


def pct(value, digits=2):
    return f(100*value, digits)


def interval(row):
    return '—' if pd.isna(row.lower) else f'[{f(row.lower,4)}, {f(row.upper,4)}]'


def cov(option, period):
    return pick(COVER, option=option, period=period).iloc[0]


def metric_rows(option, periods=PERIOD):
    return [(period, r.model, r.events, r.issuers, f(r.rank_ic,4), f(r.pearson,4),
             pct(r.q5_q1), r.profile.title()) for period in periods
            for r in pick(MET, option=option, period=period).itertuples()]


def delta_rows(option, period):
    rows = []
    for model in MODELS[1:]:
        b = pick(INF, option=option, period=period, model=model).set_index('method')
        r = pick(INC, option=option, period=period, model=model).iloc[0]
        rows.append((f'{model} − A', signed(r.delta_rank_ic,4), interval(b.loc['issuer_cluster']),
                     interval(b.loc['time_block']), f(b.holm_p_pooled.iloc[0],3)))
    return rows


def quint_rows(option, period):
    return [(r.model, f'Q{r.group}', r.n, pct(r.mean_car), pct(r.median_car), pct(r.sd_car))
            for r in pick(Q, option=option, period=period).itertuples()]


def chart(ax, zero=True):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(LINE)
    ax.tick_params(axis='both', length=0, labelsize=8, colors=MUTE, pad=6)
    ax.grid(axis='y', color=LINE, linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=MUTE, linewidth=.85, alpha=.75)


def score_plot(ax, option):
    for j, m in enumerate(MODELS):
        v = pick(MET, option=option, model=m).set_index('period').loc[PERIOD]
        ax.bar(np.arange(4)+(j-1)*.22, v.rank_ic, width=.19, color=COLOR[m], label=m)
    ax.set_xticks(range(4), ['2024 diag.', '2025', '2026 partial', 'Pooled'])
    ax.set_ylabel('Rank IC', color=MUTE, fontsize=8)
    span = max(.20, np.ceil(abs(MET.rank_ic).max()*10)/10)
    ax.set_ylim(-span, span)
    chart(ax)


def quint_plot(ax, option, period):
    for m in MODELS:
        v = pick(Q, option=option, period=period, model=m)
        ax.plot(v.group, v.mean_car*100, '-o', lw=1.4, ms=4.3, color=COLOR[m], label=m)
    span = max(2., np.ceil(abs(Q.mean_car).max()*100))
    ax.set_ylim(-span, span)
    ax.set_yticks(np.linspace(-span, span, 5))
    ax.set_xticks(range(1,6), ['Q1','Q2','Q3','Q4','Q5'])
    ax.set_ylabel('Mean realized CAR (%)', color=MUTE, fontsize=8)
    chart(ax)


def spread_plot(ax, option):
    for j,m in enumerate(MODELS):
        v = pick(MET, option=option, model=m).set_index('period').loc[PERIOD]
        ax.bar(np.arange(4)+(j-1)*.22, v.q5_q1*100, width=.19, color=COLOR[m], label=m)
    span = max(2., np.ceil(abs(MET.q5_q1).max()*100))
    ax.set_ylim(-span, span)
    ax.set_xticks(range(4), ['2024 diag.', '2025', '2026 partial', 'Pooled'])
    ax.set_ylabel('Q5 − Q1 (pp)', color=MUTE, fontsize=8)
    chart(ax)


def quarter_plot(ax, option):
    for m in MODELS:
        v = pick(QUARTER, option=option, model=m)
        ax.plot(np.arange(5), v.rank_ic.iloc[:5], '-o', color=COLOR[m], label=m, lw=1.5, ms=4)
        ax.plot([4,5], v.rank_ic.iloc[4:], '--o', color=COLOR[m], lw=1.15, ms=3.6, markerfacecolor=WHITE)
    n = pick(QUARTER, option=option, model='A')
    ax.set_xticks(range(6), [f'’{r.quarter[2:4]} {r.quarter[4:]}{"*" if r.partial else ""}\nn={r.events}'
                            for r in n.itertuples()])
    lim = max(.35, np.ceil(abs(QUARTER.rank_ic).max()*10)/10)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel('Rank IC', color=MUTE, fontsize=8)
    chart(ax)


def forest(ax, period):
    items = [(o,m) for o in OPTIONS for m in MODELS[1:]]
    labels = [f'{"S&P 500" if o == "market" else "Healthcare"} / {m} − A' for o,m in items]
    for j, (o,m) in enumerate(items):
        rs = pick(INF, option=o, period=period, model=m).set_index('method')
        for k, method in enumerate(METHOD):
            r = rs.loc[method]
            y = (len(items)-j-1)+(k-.5)*.22
            ax.hlines(y, r.lower, r.upper, color=COLOR[m], lw=1.45, ls='-' if k==0 else '--')
            ax.plot(r.point, y, 'o' if k==0 else 's', color=COLOR[m], mfc=COLOR[m] if k==0 else WHITE, ms=4.8)
    limit = max(.10, np.ceil(np.nanmax(abs(INF[['lower','upper']].to_numpy()))*20)/20)
    ax.set_xlim(-limit, limit)
    ax.set_yticks(range(len(items)), labels[::-1])
    ax.set_xlabel('Incremental Rank IC', color=MUTE, fontsize=8)
    chart(ax, zero=False)
    ax.axvline(0, color=MUTE, lw=.85)
    ax.legend(handles=[Line2D([0],[0], color=MUTE, marker='o', lw=1.3, label='Issuer cluster'),
                       Line2D([0],[0], color=MUTE, marker='s', ls='--', markerfacecolor=WHITE,
                              lw=1.3, label='Calendar block')], fontsize=8, frameon=False, loc='lower right')


def coverage_plot(ax):
    d = pick(COVER, option='market').set_index('period').loc[PERIOD[:3]]
    for j, (field,label,color) in enumerate(zip(['archived_calls','valid_target','matched_events'],
                                               ['Source events','Valid target','Matched cohort'],
                                               ['#ACBAC6','#6688A5',TEAL])):
        x = np.arange(3)+(j-1)*.23
        ax.bar(x, d[field], width=.21, label=label, color=color)
        for xpos,n in zip(x,d[field]):
            ax.text(xpos,n+3,str(n),fontsize=7,ha='center',color=MUTE)
    ax.set_ylim(0,max(d.archived_calls)*1.19)
    ax.set_xticks(range(3), SHORT[:3])
    ax.set_ylabel('Events', fontsize=8, color=MUTE)
    chart(ax, zero=False)
    ax.legend(frameon=False, fontsize=7.2, loc='upper center', ncol=3)


def benchmark_plot(ax):
    if len(OPTIONS)<2:
        return
    for j,m in enumerate(MODELS):
        v = pick(MET, period=POOLED, model=m).set_index('option').loc[list(OPTIONS)]
        ax.plot(v.rank_ic,[j,j],'-o',color=COLOR[m],lw=1.4,ms=5)
        for opt, x in v.rank_ic.items():
            ax.annotate(f(x), (x,j), xytext=(0,8), textcoords='offset points', ha='center', fontsize=7.3,color=TEXT)
    ax.set_yticks(range(3), MODELS)
    ax.set_xlim(-.20,.20)
    ax.set_ylim(-.40,2.40)
    ax.set_xlabel('Pooled Rank IC', fontsize=8, color=MUTE)
    chart(ax, zero=False)
    ax.axvline(0,color=MUTE,lw=.8)
    ax.text(.02,.96,'S&P 500',transform=ax.transAxes,color=MUTE,fontsize=7.3,ha='left',va='top')
    ax.text(.98,.96,'Healthcare sector',transform=ax.transAxes,color=MUTE,fontsize=7.3,ha='right',va='top')


def page(num, section, headline, dek):
    fig = plt.figure(figsize=(8.5,11), facecolor=WHITE)
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,612), ylim=(792,0)); ax.axis('off')
    rect(ax, 0,0,612,74,NAVY)
    t(ax,42,25,TITLE,13,WHITE,'bold')
    t(ax,42,48,SUBTITLE,9.6,'#C7DBE2')
    t(ax,42,97,section.upper(),8,TEAL,'bold')
    t(ax,42,115,headline,24,NAVY,'bold')
    t(ax,42,151,dek,9.7,MUTE)
    ax.plot([42,570],[753,753],color=LINE,lw=.8)
    t(ax,42,765,'QUANTITATIVE RESEARCH  /  HEALTH CARE  /  CRSP THROUGH 30 JUN 2026',7.1,MUTE)
    t(ax,570,765,f'{num:02d}',8,NAVY,'bold',ha='right')
    return fig, ax


def t(ax,x,y,s,size=9,color=TEXT,weight='normal',ha='left'):
    ax.text(x,y,str(s),fontsize=size,color=color,fontweight=weight,ha=ha,va='top')


def rect(ax,x,y,w,h,color):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='none'))


def p(ax,x,y,s,width=102,size=9.4,color=TEXT,step=13):
    for line in textwrap.wrap(s,width=width,break_long_words=False,break_on_hyphens=False):
        t(ax,x,y,line,size,color)
        y += step
    return y


def label(ax,y,s):
    t(ax,42,y,s,12,NAVY,'bold')


def note(ax,y,s):
    p(ax,42,y,s,width=116,size=7.65,color=MUTE,step=10)


def subplot(fig,x,y,w,h):
    return fig.add_axes([x/612,(792-y-h)/792,w/612,h/792])


def table(ax,y,headers,rows,cols,spacing=18,size=8.2):
    rect(ax,42,y,528,spacing,NAVY)
    for x,s in zip(cols,headers):
        t(ax,x,y+(spacing-size)/2-1,s,size,WHITE,'bold')
    for j,row in enumerate(rows):
        y += spacing
        rect(ax,42,y,528,spacing,SOFT if j%2==0 else WHITE)
        for x,s in zip(cols,row):
            t(ax,x,y+(spacing-size)/2-1,s,size,COLOR.get(s,TEXT),'bold' if s in MODELS else 'normal')
    return y+spacing


def legend(ax,y):
    for x,(m,title) in zip([42,199,384],[('A','Financial baseline'),('B','Selected LM5'),('B*','Full LM9')]):
        rect(ax,x,y+2,10,4,COLOR[m]); t(ax,x+16,y,f'{m}  {title}',8.4,MUTE)


def add(pdf,fig):
    pdf.savefig(fig)
    plt.close(fig)


def make_pdf():
    d = [cov(OPTIONS[0],v) for v in PERIOD]
    bi = pick(INF, period=POOLED)
    n_validation = int(VAL.passed.sum())
    with PdfPages(HERE/PDF, metadata={'Title': TITLE, 'Subject': SUBTITLE}) as pdf:
        fig, ax = page(1,'01 / Executive research summary','No pooled lift from language',
                       'Do earnings-call language scores improve the original financial baseline?')
        for x,num,caption in [(42,f'{d[3].matched_events}','Pooled OOS calls'), (222,f'{d[3].issuers}','Distinct OOS issuers'),
                              (402,f'{len(OPTIONS)*3}','Frozen specifications')]:
            rect(ax,x,180,168,76,SOFT); t(ax,x+11,190,num,26,NAVY,'bold'); t(ax,x+11,229,caption,8.8,MUTE)
        label(ax,280,'Pooled 2025–26 results on matched events')
        rows = []
        for o in OPTIONS:
            z = pick(MET, option=o, period=POOLED).set_index('model')
            for m in MODELS:
                r = z.loc[m]
                rows.append((OPT[o],m,f(r.rank_ic,4), signed(r.rank_ic-z.loc['A'].rank_ic,4) if m!='A' else '—',pct(r.q5_q1)))
        end = table(ax,303,['Benchmark','Model','Rank IC','Δ Rank IC vs A','Q5 − Q1 (pp)'],rows,[49,224,281,355,477],25,8.8)
        note(ax,end+8,'Table 1. CAR[+2,+20]. Each row uses 315 events and 61 issuers. Rank IC and changes are dimensionless.')
        label(ax,515,'What the new holdout shows')
        p(ax,42,539,'All four pooled LM increments are negative. Every issuer-cluster and calendar-block 95% interval includes zero. The four Holm-adjusted p-values are 1.000.')
        p(ax,42,576,'Financial-only ranks perform better against the Healthcare benchmark. Positive extreme-group spreads coexist with non-monotonic quintile means and weak market-adjusted rank IC.')
        rect(ax,42,628,528,93,PALE)
        t(ax,54,639,'HOW TO READ THE EVIDENCE',8.3,TEAL,'bold')
        p(ax,54,657,'Trained on 2010–23. The 2024 diagnostic informed historical strategy selection. New tests cover full-year 2025 and matured 2026 events through the May 27 call, using June 30 CRSP data. The Early CAR control becomes available after the Day +1 close.',width=96,size=9.3)
        add(pdf,fig)

        fig, ax = page(2,'02 / Research design','Six frozen specifications',
                       'Development fits use archived 2010–23 targets and features. The 2024 scores are reused.')
        rows = []
        for o in OPTIONS:
            for m,spec in SPEC['specifications'][o].items():
                a = spec['parameters']; estimator = f"{spec['estimator']} (α {a['alpha']:g})"
                if 'l1_ratio' in a:
                    estimator = f"Elastic Net (α {a['alpha']:g}, L1 {a['l1_ratio']:g})"
                rows.append((OPT[o],m,estimator,len(spec['columns']),{'A':'Controls','B':'Controls + LM5','B*':'Controls + LM9'}[m]))
        end = table(ax,180,['Benchmark','Model','Estimator','Inputs','Feature block'],rows,[49,224,268,430,479],23,8.6)
        note(ax,end+6,'Table 2. Each model uses 2,939 development events across 95 issuers. All six refits replay 204 archived scores exactly.')
        label(ax,372,'Feature blocks')
        t(ax,42,394,'16 common controls',9.3,TEAL,'bold')
        p(ax,42,410,', '.join(SPEC['controls'])+'.',105,8.4,step=12)
        t(ax,42,455,'Selected LM5',9.3,TEAL,'bold')
        lm = SPEC['specifications']['market']['B']['columns'][len(SPEC['controls']):]
        p(ax,42,471,', '.join(lm)+'. Both benchmarks use these five features in distinct archived orders.',105,8.4,step=12)
        t(ax,42,505,'Full LM9',9.3,TEAL,'bold')
        p(ax,42,521,', '.join(SPEC['LM9'])+'.',105,8.4,step=12)
        label(ax,564,'Target and preprocessing')
        p(ax,42,585,'Fit the unnormalized 1-based average midranks of development CAR. Apply the archived log transforms, 1st/99th percentile winsorization, median imputation and standard scaling using development statistics only.')
        p(ax,42,630,'Against each benchmark, A, B and B* score exactly the same eligible calls. The S&P 500 specification uses SPRTRN price returns. The Healthcare specification uses the original static GICS and cap-weighted S&P 500 Healthcare basket.')
        p(ax,42,675,'The market Early CAR is car_-1_1 in the extended table. The sector option uses the same window with the Healthcare benchmark. Neither model specification includes car_0_1 or AR01.')
        add(pdf,fig)

        fig, ax = page(3,'03 / Cross-period comparison','Results by benchmark and period',
                       'Market-adjusted pooled Rank IC is negative. Sector-adjusted Rank IC remains positive.')
        for k,o in enumerate(OPTIONS):
            y=179+k*235
            label(ax,y,OPT[o]); table(ax,y+19,
                ['Period','Model','Events','Issuers','Rank IC','Pearson','Q5 − Q1 (pp)','Profile'],metric_rows(o),
                [49,158,209,255,304,369,438,529],15.5,7.7)
        note(ax,647,'Table 3. Spearman Rank IC and supplementary Pearson correlation are dimensionless. Q5 − Q1 is the mean realized CAR difference between the highest and lowest predicted quintiles, in percentage points.')
        note(ax,679,'Pooled metrics recompute ranks and quintiles over all 315 calls. The 2024 rows are selected historical diagnostics. 2026 holds only calls with complete forward windows through the June 30 CRSP cutoff.')
        note(ax,711,'Scores are fitted midranks. Continuous-return R², MAE, RMSE and directional accuracy are omitted.')
        add(pdf,fig)

        for k,o in enumerate(OPTIONS):
            fig, ax = page(4+k,f'0{4+k} / Prediction quintiles',OPT[o]+' quintiles',
                           'Realized CAR[+2,+20] grouped by ascending model scores within each sample.')
            legend(ax,177)
            for i,period in enumerate(PERIOD):
                col,row=i%2,i//2; x=42+col*262; y=208+row*148
                c=cov(o,period)
                t(ax,x,y,f'{SHORT[i]}  /  {c.matched_events} calls, {c.issuers} issuers',8.6,NAVY,'bold')
                quint_plot(subplot(fig,x+42,y+22,184,99),o,period)
            note(ax,497,f'Figure {1+k}. Mean realized CAR, percent. Source: frozen-model scores and matched CARs. Q1 is the lowest prediction quintile. All panels use the same y-axis.')
            label(ax,526,'Quintile sizes and extreme-group separation')
            rows=[]
            for i,period in enumerate(PERIOD):
                uq=pick(Q,option=o,period=period).pivot(index='model',columns='group',values='n')
                if uq.nunique().max()==1:
                    counts=' / '.join(str(x) for x in uq.iloc[0])
                else:
                    counts=' | '.join('/'.join(str(x) for x in uq.loc[m]) for m in MODELS)
                v=pick(MET,option=o,period=period).set_index('model')
                rows.append((SHORT[i],counts,*[pct(v.loc[m,'q5_q1']) for m in MODELS],v.profile.iloc[0].title()))
            end=table(ax,547,['Period','Q1 / Q2 / Q3 / Q4 / Q5 n','A (pp)','B (pp)','B* (pp)','All profiles'],
                      rows,[49,161,312,372,430,499],20,7.8)
            note(ax,end+7,'Quintile counts are shared across models when shown once. Spreads are Q5 minus Q1 in percentage points. These event groups overlap in calendar time and do not represent portfolio returns.')
            spread=pick(MET,option=o,period=POOLED).set_index('model').q5_q1
            p(ax,42,688,f'Pooled separation is {pct(spread.A)} pp for A, {pct(spread.B)} pp for B and {pct(spread["B*"])} pp for B*. No profile shows strictly increasing or decreasing mean CAR across all five quintiles.')
            add(pdf,fig)

        fig, ax = page(6,'06 / Temporal stability','Signal varies across periods',
                       'The LM increments depend on the benchmark and the period used to measure them.')
        legend(ax,177)
        for k,o in enumerate(OPTIONS):
            x=42+k*264
            t(ax,x,202,OPT[o],9.2,NAVY,'bold')
            score_plot(subplot(fig,x+37,222,210,109),o)
            t(ax,x,369,OPT[o],9.2,NAVY,'bold')
            quarter_plot(subplot(fig,x+34,390,222,102),o)
        note(ax,349,'Figure 3. Rank IC. N = 204, 211, 104 and 315 per benchmark. Matched-event scores and realized CAR.')
        note(ax,523,'Figure 4. Quarterly Rank IC; eligible quarters require ≥30 events and ≥15 issuers. * Q2 2026 ends at the May 27 eligible call and is excluded from stability summaries.')
        label(ax,558,'Full-quarter stability')
        rows=[(OPT[r.option],r.model,r.adequate_full_quarters,f(r.mean_ic),f(r.sd_ic),pct(r.positive_ic_share,0)+'%',f(r.icir)) for r in STAB.itertuples()]
        end=table(ax,580,['Benchmark','Model','Quarters','Mean IC','IC SD','Positive IC','ICIR'],rows,
                  [49,219,265,332,393,454,531],17,8.2)
        note(ax,end+7,'Sample SD across comparable quarters. Five full quarters are available. The frozen minimum for ICIR is six.')
        add(pdf,fig)

        fig, ax = page(7,'07 / Paired inference','Uncertainty overwhelms the lift',
                       'All pooled estimates fall below zero. Every pointwise 95% interval includes zero.')
        label(ax,176,'Pooled 2025–26 Δ Rank IC')
        forest(subplot(fig,177,208,375,140),POOLED)
        note(ax,380,'Figure 5. Paired percentile intervals, n = 315 events, 61 issuers, 18 calendar months and 9 nominal two-month blocks. All 2,499 draws per resampling method are valid.')
        for j,period in enumerate((POOLED,'2025')):
            y=410+j*119
            label(ax,y,'Pooled intervals and adjusted p-values' if j==0 else '2025 pointwise intervals')
            rows=[(('S&P 500' if o=='market' else 'Healthcare'),*r) for o in OPTIONS for r in delta_rows(o,period)]
            table(ax,y+19,['Benchmark','Model Δ','Δ IC','Issuer 95% CI','Time block 95% CI','Holm p'],
                  rows,[49,135,199,258,382,516],16.5,7.9)
        note(ax,642,'Table 4. Pooled: 315 events, 61 issuers. 2025: 211 events, 61 issuers, 6 nominal blocks. The larger method-specific p-value enters the four-test pooled Holm correction.')
        partial=[f'{"S&P 500" if o=="market" else "Healthcare"} '+', '.join(f'{r[0]} {r[1]}' for r in delta_rows(o,'2026 partial')) for o in OPTIONS]
        p(ax,42,676,'Partial 2026 (104 events, 59 issuers, 3 nominal blocks) is descriptive. '+'  /  '.join(partial)+'.',width=104,size=8.8)
        note(ax,713,'Issuer and time bootstraps probe different dependence structures. Each uses identical event resamples across models. Few independent time blocks limit resolution.')
        add(pdf,fig)

        fig, ax = page(8,'08 / Coverage and benchmarks','What changed in the extension',
                       'The report keeps the historical model definition and validates the newer CRSP return feed.')
        label(ax,177,'Healthcare event coverage')
        coverage_plot(subplot(fig,94,207,443,143))
        note(ax,379,'Figure 6. Counts by coverage stage and period, identical across benchmarks. Six target-valid new events lacked a dated S&P 500 membership match. No target-valid event failed security mapping.')
        rows=[(r.period,r.archived_calls,r.valid_target,r.matched_events,r.issuers,r.last_call) for r in pick(COVER,option='market').itertuples()]
        end=table(ax,415,['Period','Source n','Target n','Matched n','Issuers','Last call'],rows,[49,181,259,337,420,482],17.5,8.3)
        label(ax,end+11,'Benchmark comparison on 315 pooled calls')
        benchmark_plot(subplot(fig,85,end+43,456,119))
        note(ax,end+192,'Figure 7. Pooled Rank IC under both native benchmark and estimator configurations. Both use the same matched event IDs; changing the target and fitting specification can change rankings.')
        note(ax,732,f'All {n_validation} sector validation checks passed. See page 9 for vintage errors and timing conventions.')
        add(pdf,fig)

        fig, ax = page(9,'09 / Technical appendix','Validation and timing',
                       'Data lineage, availability assumptions and reproducibility.')
        label(ax,179,'2024 Healthcare benchmark validation')
        rows=[]
        for r in VAL.itertuples():
            name=r.check.replace('2024 ','').replace(' event ',' / ').replace('car_-1_1','Early CAR').replace('car_2_20','CAR[+2,+20]')
            rows.append((name,r.compared_n,f'{r.max_abs_error*10000:.5f}',f'{r.fraction_within_cutoff:.0%}','Pass' if r.passed else 'Fail'))
        end=table(ax,201,['2024 comparison','Paired n','Max |error| (bp)','Within gate','Status'],rows,[49,277,347,441,521],20,8.3)
        note(ax,end+7,'Table 5. Legacy tolerance: 1e−10 in return units. CIZ bridge: 95% of paired labels within 5 bp and all within 25 bp. Same-day cap weights retain the original benchmark convention.')
        label(ax,361,'Input integrity and benchmark construction')
        p(ax,42,382,'The 29,698-row extended input preserves event identity, the protected historic CARs and LM columns. The approved financial reconstruction changes only designated financial fields from 2025. Missing predictors use archived development statistics.')
        p(ax,42,427,'For Healthcare, classify firms using the original static GICS snapshot and link to securities through CCM. Restrict each trading session to dated S&P 500 members. Weight available stock returns by the same-day absolute price × shares outstanding. Use legacy returns through 2024 and CIZ daily returns from 2025.')
        label(ax,487,'Event time and observability')
        p(ax,42,509,'Day 0 is the CRSP trading date for calls before 16:00 Eastern, with calls at or after 16:00 and non-session dates rolled forward. The target sums stock minus benchmark daily returns from +2 to +20. The Early CAR control spans −1 through +1.')
        p(ax,42,555,'Early CAR is observable after the Day +1 close. These scores cannot be interpreted as available at the call. No clock gate tests transcript, financial-input or execution availability. The original LM dictionary makes hedging_answers and uncertainty_answers identical.')
        p(ax,42,600,'The 2024 diagnostic informed historical Model B strategy and estimator choices. New cohorts are scored once using frozen rules. Partial 2026 supplies three nominal time blocks, too few for inference under the declared gate.')
        label(ax,646,'Reproducibility')
        p(ax,42,669,'The freeze manifest lists ordered feature sets, estimator arguments, historical source hashes, preprocessing and inference. The evaluation and rendering scripts produce the aggregate research tables, report and dashboard from local data and saved inputs.',width=106,size=9.1)
        note(ax,721,'Diagnostic metrics, matched predictions, sector-return checks and all reported quintile statistics are saved with this report.')
        add(pdf,fig)

        fig, ax = page(10,'10 / Robustness and missingness','Extreme groups separate unevenly',
                       'The spread describes two tails. Rank IC summarizes ordering across all events.')
        legend(ax,178)
        for j,o in enumerate(OPTIONS):
            x=42+264*j
            t(ax,x,209,OPT[o],9.3,NAVY,'bold')
            spread_plot(subplot(fig,x+39,232,207,117),o)
        note(ax,376,'Figure 8. Q5 minus Q1, percentage points. N = 204, 211, 104 and 315 per panel. Source: realized CAR conditional on model scores. Full quintile means, medians, SD and n are available in the dashboard and CSV.')
        label(ax,423,'New holdout missingness before imputation')
        feature=pick(FEATURE,option='market').pivot(index='feature',columns='period',values='missing')
        names=feature.loc[(feature[PERIOD[1:3]]>0).any(axis=1)].index
        rows=[(name,int(feature.loc[name,PERIOD[1]]),int(feature.loc[name,PERIOD[2]])) for name in names]
        rows.append(('Other 20 candidate inputs, including LM9',0,0))
        end=table(ax,447,['Predictor','2025 missing / 211','2026 partial missing / 104'],rows,[49,298,427],19,8.5)
        note(ax,end+7,'Table 6. Counts are identical for the two matched cohorts. Missing predictors are retained with development-only imputation. All 16 financial definitions passed the upstream validation gate.')
        label(ax,634,'New holdout ranking changes')
        first,last=pick(INC,period=POOLED).prediction_rank_agreement.agg(['min','max'])
        p(ax,42,658,f'Spearman agreement between A and its LM variants ranges from {f(first)} to {f(last)} in the pooled sample. The LM variants change model ranks without delivering positive pooled Δ Rank IC.')
        p(ax,42,699,'Sector increments are negative in both new periods. Market B is near zero in 2025 and slightly positive in partial 2026. Market B* changes sign.',size=9.1)
        add(pdf,fig)

        for j,o in enumerate(OPTIONS):
            title='S&P 500-adjusted distributions' if o=='market' else 'Sector-adjusted distributions'
            fig, ax = page(11+j,f'{11+j:02d} / Quintile appendix',title,
                           'Mean, median, sample standard deviation and count within each predicted group.')
            for i,period in enumerate(PERIOD):
                y=177+i*135
                r=cov(o,period)
                t(ax,42,y,f'{SHORT[i]}   /   {r.matched_events} events, {r.issuers} issuers',9.5,NAVY,'bold')
                sq = pick(Q,option=o,period=period)
                rows=[]
                for g in range(1,6):
                    rows.append((f'Q{g}',*[f'{a.n}  /  {pct(a.mean_car)}  /  {pct(a.median_car)}  /  {pct(a.sd_car)}'
                                             for a in sq.loc[sq.group.eq(g)].itertuples()]))
                table(ax,y+17,['Group','A  n / mean / median / SD','B  n / mean / median / SD','B*  n / mean / median / SD'],
                      rows,[49,106,257,408],15,7.8)
            note(ax,727,'Return statistics are in percent. SD uses n − 1. Source: within-period predicted-score quintiles in quintile_returns.csv.')
            add(pdf,fig)


def html_table(headers, rows):
    def cell(value):
        style=f' class="model" style="color:{COLOR[value]}"' if value in MODELS else ''
        return f'<td{style}>{escape(str(value))}</td>'
    return ('<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{escape(str(v))}</th>' for v in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(cell(v) for v in row) + '</tr>'
                                                for row in rows) + '</tbody></table></div>')


def inline_chart(callback, *args, wide=False):
    fig=plt.figure(figsize=(7.5 if wide else 5.4, 2.65),facecolor=WHITE)
    fig.subplots_adjust(left=.14,bottom=.20,right=.96,top=.93)
    if callback == forest:
        fig.subplots_adjust(left=.27)
    callback(fig.add_subplot(),*args)
    stream=StringIO()
    fig.savefig(stream,format='svg',metadata={'Date':None})
    plt.close(fig)
    svg=stream.getvalue().split('<svg',1)[1].replace(' id="',' data-svg-id="')
    # Namespace local glyphs and clip paths across embedded SVGs.
    import re
    prefix=f'plot{inline_chart.serial}-'
    inline_chart.serial += 1
    svg=re.sub(r'(data-svg-id="|href="#|url\(#)([\w-]+)',lambda m: ('id="' if m[1]=='data-svg-id="' else m[1])+prefix+m[2],svg)
    svg='\n'.join(line.rstrip() for line in svg.splitlines())
    return '<svg role="img" aria-hidden="true"'+svg


inline_chart.serial=0


def hfig(number,title,detail,fn,*values,wide=False):
    return (f'<figure><h3>Figure {number}. {escape(title)}</h3>{inline_chart(fn,*values,wide=wide)}'
            f'<figcaption>{escape(detail)}</figcaption></figure>')


CSS='''
:root{color-scheme:light;--navy:#12253d;--ink:#293b4c;--muted:#4f6373;--line:#cbd7df;--soft:#eff4f6;--teal:#007e79;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.56 "Segoe UI",sans-serif}
header{background:var(--navy);color:white;padding:2.7rem max(24px,calc((100vw - 1200px)/2)) 2.4rem;border-bottom:4px solid #007e79}
header h1{font-size:clamp(1.25rem,2.6vw,2rem);line-height:1.17;letter-spacing:.075em;margin:0 0 .6rem}header p{margin:0;color:#c7dbe2;font-size:1rem}
main,footer{max-width:1240px;margin:auto;padding:1.55rem 22px}section{margin:0 0 1.15rem;background:white;border:1px solid var(--line);padding:clamp(16px,3.3vw,32px)}
h2,h3{color:var(--navy);line-height:1.2;margin:0 0 .75rem}h2{font-size:clamp(1.52rem,2.6vw,2.1rem)}h3{font-size:1.01rem}p{margin:.5rem 0 1rem}a{color:#00655e;text-underline-offset:3px}
.eyebrow{font-weight:700;font-size:.73rem;color:var(--teal);letter-spacing:.14em}.grid,.tiles,.filters{display:grid;gap:14px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.tiles{grid-template-columns:repeat(3,minmax(0,1fr));margin:1.05rem 0}.tile{background:var(--soft);padding:12px 17px}.tile strong{display:block;color:var(--navy);font-size:1.8rem;font-variant-numeric:tabular-nums}
.filters{grid-template-columns:repeat(2,minmax(145px,300px));margin:1rem 0 1.2rem}label{font-weight:650;color:var(--navy)}select{display:block;width:100%;margin-top:5px;padding:.62rem;background:white;border:1px solid #8299aa;border-radius:2px;font:inherit;color:var(--navy)}:focus-visible{outline:3px solid #007e79;outline-offset:3px}
.table-wrap{overflow:auto;margin:.7rem 0 1.05rem}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.91rem;min-width:400px}th,td{text-align:left;padding:.52rem .65rem;white-space:nowrap}th{background:var(--navy);color:white;font-size:.83rem}tbody tr:nth-child(odd){background:var(--soft)}td.model{font-weight:700}
figure{padding:12px;margin:0 0 .65rem;border:1px solid var(--line);min-width:0}figure svg{width:100%;height:auto;display:block}figcaption,.sub,footer{font-size:.86rem;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:1.6em;font-size:.9rem;margin:.5rem 0 1rem}.swatch{display:inline-block;width:12px;height:5px;margin-right:6px;vertical-align:middle}details{border-top:1px solid var(--line);padding:.66rem 0}summary{cursor:pointer;font-weight:650;color:var(--navy)}[hidden]{display:none!important}.links{display:flex;flex-wrap:wrap;gap:.42rem 1.25rem}.links a{white-space:nowrap}
@media(max-width:740px){.grid{grid-template-columns:1fr}main{padding:15px 12px}.tiles{gap:6px}.tile{padding:8px}.tile strong{font-size:1.45rem}th,td{padding:.48rem}.filters{gap:8px}}@media(max-width:375px){.filters{grid-template-columns:1fr}}
'''


def make_html():
    suite=[]
    head=[(OPT[o],*r[1:]) for o in OPTIONS for r in metric_rows(o,[POOLED])]
    c=cov('market',POOLED)
    options=''.join(f'<option value="{o}">{escape(OPT[o])}</option>' for o in OPTIONS)
    periods=''.join(f'<option value="{i}"{" selected" if i==3 else ""}>{p}</option>' for i,p in enumerate(PERIOD))
    legend_html='<div class="legend">'+''.join(f'<span><i class="swatch" style="background:{COLOR[m]}"></i>{m} {nm}</span>'
                    for m,nm in zip(MODELS,('Financial baseline','Selected LM5','Full LM9')))+'</div>'
    for o in OPTIONS:
        for i,p in enumerate(PERIOD):
            r = cov(o,p); sub = f'{r.matched_events} calls · {r.issuers} issuers · {r.first_call} through {r.last_call}'
            status={'2024 diagnostic':'Previously examined and used for historical model selection.',
                    '2025':'New OOS, full calendar year. Intervals are pointwise.',
                    '2026 partial':'New OOS, complete forward windows through June 30. Descriptive statistics only.',
                    POOLED:'New OOS, pooled ranks across all matched calls. No averaging of yearly Rank IC.'}[p]
            panel=[f'<article data-option="{o}" data-period="{i}" '+('' if (o,i)==('market',3) else 'hidden')+'>',
                   f'<h3>{escape(OPT[o])} · {p}</h3><p class="sub">{sub}. {status}</p>',
                   html_table(['Model','Events','Issuers','Rank IC','Pearson','Q5 − Q1 (pp)','Profile'],[r[1:] for r in metric_rows(o,[p])]),
                   html_table(['Increment','Δ Rank IC','Issuer 95% CI','Time block 95% CI','Holm p'],delta_rows(o,p)),
                   '<div class="grid">',hfig('1' if o=='market' else '2','Prediction quintiles · '+p,
                   f'{sub}. Mean realized CAR (%), sorted by model score. Source: matched scores and CARs.',quint_plot,o,p),
                   html_table(['Model','Group','n','Mean CAR (%)','Median (%)','SD (%)'],quint_rows(o,p)),'</div></article>']
            suite.extend(panel)
    score_figs=''.join(hfig('3',OPT[o]+' Rank IC','N = 204, 211, 104 and 315. Spearman correlation in matched event cohorts.',score_plot,o) for o in OPTIONS)
    time_figs=''.join(hfig('4',OPT[o]+' quarterly stability','Eligible quarterly Spearman IC. * Q2 2026 is truncated and excluded from stability summaries.',quarter_plot,o) for o in OPTIONS)
    spread_figs=''.join(hfig('8',OPT[o]+' extreme-quintile spread','Q5 minus Q1 mean CAR, pp. N = 204, 211, 104 and 315. Source: matched score groups.',spread_plot,o) for o in OPTIONS)
    stab_rows=[(OPT[r.option],r.model,r.adequate_full_quarters,f(r.mean_ic),f(r.sd_ic),pct(r.positive_ic_share,0)+'%',f(r.icir)) for r in STAB.itertuples()]
    val_rows=[(r.check.replace(' event ',' / '),r.compared_n,f'{r.max_abs_error*10000:.5f}',f'{r.fraction_within_cutoff:.0%}', 'Pass' if r.passed else 'Fail') for r in VAL.itertuples()]
    cov_rows=[(r.period,r.archived_calls,r.valid_target,r.matched_events,r.issuers,r.last_call) for r in pick(COVER,option='market').itertuples()]
    files={'model_performance.csv':'Performance CSV','incremental_comparison.csv':'Incremental CSV','bootstrap_inference.csv':'Bootstrap CSV',
           'quintile_returns.csv':'Quintile CSV','quarterly_ic.csv':'Quarterly IC CSV','stability_summary.csv':'Stability CSV',
           'target_coverage.csv':'Target coverage CSV','feature_coverage.csv':'Feature coverage CSV',
           'sector_benchmark_validation.csv':'Benchmark validation CSV','fit_replay_validation.csv':'Replay validation CSV',
           'frozen_experiment_spec.json':'Frozen specification','methodology.md':'Methodology','summary_zh.md':'中文摘要'}
    f_links=''.join(f'<a href="{name}">{label}</a>' for name,label in files.items())
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title><style>{CSS}</style></head><body><header><h1>{TITLE}</h1><p>{SUBTITLE}</p></header><main>
<section><div class="eyebrow">01 / EXECUTIVE RESEARCH SUMMARY</div><h2>No pooled lift from language</h2>
<p>All four pooled LM increments are negative. Both paired 95% intervals for each increment include zero. The Healthcare benchmark produces a stronger financial baseline. Three model variants per benchmark share identical event IDs.</p>
<div class="tiles"><div class="tile"><strong>{c.matched_events}</strong>Pooled OOS calls</div><div class="tile"><strong>{c.issuers}</strong>Pooled OOS issuers</div><div class="tile"><strong>{len(OPTIONS)*3}</strong>Frozen specifications</div></div>
{html_table(['Benchmark','Model','Events','Issuers','Rank IC','Pearson','Q5 − Q1 (pp)','Profile'],head)}
<p class="sub">Development 2010–23. 2024 informed historical strategy selection. New OOS covers 2025 and matured 2026 calls through {c.last_call}. Target CAR[+2,+20].</p></section>
<section><div class="eyebrow">02 / MATCHED-EVENT EXPLORER</div><h2>Compare benchmarks and cohorts</h2>
<div class="filters"><label>Benchmark<select id="bench">{options}</select></label><label>Period<select id="period">{periods}</select></label></div>
{legend_html}<div id="result" aria-live="polite">{''.join(suite)}</div><p class="sub">Groups overlap in event time. Q5 − Q1 compares mean realized CAR, in pp.</p></section>
<section><div class="eyebrow">03 / STABILITY AND INFERENCE</div><h2>Compare years, quarters and uncertainty</h2>{legend_html}
<div class="grid">{score_figs}</div><div class="grid">{time_figs}</div>
{html_table(['Benchmark','Model','Full quarters','Mean IC','IC SD','Positive IC','ICIR'],stab_rows)}
<p class="sub">Positive share and SD use only the five eligible full quarters. Six are required to show ICIR.</p>
<div class="grid">{hfig('5A','Pooled paired 95% intervals','315 calls, 61 issuers and 9 nominal two-month blocks. All 2,499 draws valid per method.',forest,POOLED)}
{hfig('5B','2025 pointwise paired 95% intervals','211 calls, 61 issuers and 6 nominal two-month blocks. All 2,499 draws valid per method.',forest,'2025')}</div></section>
<section><div class="eyebrow">04 / BENCHMARK AND COVERAGE CHECKS</div><h2>Validate the inputs</h2><div class="grid">
{hfig('6','Healthcare event coverage','Archived, target-valid and matched counts. Matched samples are identical across the two benchmarks.',coverage_plot)}
{hfig('7','Pooled rank IC by benchmark','Same 315 calls from 61 issuers. Source: frozen native configurations and matched realized CAR.',benchmark_plot)}</div>
{html_table(['Period','Source events','Valid target','Matched events','Issuers','Last call'],cov_rows)}
{html_table(['Validation','Compared n','Max absolute error (bp)','Within gate','Status'],val_rows)}<div class="grid">{spread_figs}</div></section>
<section><div class="eyebrow">05 / METHODS AND DOWNLOADS</div><h2>Reproduce the evaluation</h2>
<details><summary>Cohorts, training and timing</summary><p>The 29,698-event input preserves archived identity and existing labels. Models are fitted only on 2,939 native development events per option. No additional clock gate is applied. Early CAR[−1,+1] is known after the Day +1 close and the study does not establish transcript or trading availability at that time.</p></details>
<details><summary>Model estimators, transformations and features</summary><p>Market A/B use Ridge α 1 and market B* uses Ridge α 10. Sector A uses Ridge α 1, B uses Elastic Net α 0.05 and L1 ratio 0.7, and B* uses Ridge α 50. All fits use unnormalized development midranks. The selected LM5 and full LM9 follow archived order. <code>hedging_answers</code> and <code>uncertainty_answers</code> coincide under the archived dictionary. See the manifest for exact columns, preprocessing and estimator arguments.</p></details>
<details><summary>Benchmarks, inference and limitations</summary><p>Daily stock minus benchmark returns are summed over offsets +2 to +20. Sector returns use original static GICS, dated S&P 500 membership and same-day market-cap weights. Returns use legacy CRSP through 2024 and CIZ from 2025. Paired issuer and moving-block resampling report pointwise intervals. The larger method p-value enters the pooled four-test Holm adjustment. Partial 2026 has only three nominal time blocks. Scores are rank scale. No continuous-return regression error or directional metric is shown.</p></details>
<p class="links">{f_links}</p></section></main><footer>QUANTITATIVE RESEARCH / HEALTH CARE / CRSP THROUGH 30 JUN 2026</footer>
<script>const b=document.getElementById('bench'),p=document.getElementById('period');function setPanel(){{for(const e of document.querySelectorAll('[data-option]')) e.hidden=e.dataset.option!==b.value||e.dataset.period!==p.value;}} b.addEventListener('change',setPanel);p.addEventListener('change',setPanel);setPanel();</script></body></html>'''
    (HERE/'dashboard.html').write_text(html,encoding='utf-8')


def make_summary_zh():
    names={'market':'标普 500 调整','sector':'医疗保健行业调整'}
    periods=['2024 历史诊断','2025 全年','2026 部分年度','2025–26 合并']
    d=[cov('market',v) for v in PERIOD]
    body=['# 医疗保健语言因子样本外评估', '',
          '六个冻结规格均完成评估，重拟合完全复现六组已保存的 2024 预测分数。每个基准使用 2,939 场 2010–2023 年电话会议拟合。2024 曾被用于选择 Model B 的策略和估计器，属于历史诊断。', '',
          f'两个基准的评估事件相同。2024 为 {d[0].matched_events} 场、{d[0].issuers} 家发行人，2025 为 {d[1].matched_events} 场、{d[1].issuers} 家，2026 部分年度为 {d[2].matched_events} 场、{d[2].issuers} 家。合并样本为 {d[3].matched_events} 场、{d[3].issuers} 家。CRSP 截止 2026-06-30，最晚纳入电话会议日期为 {d[2].last_call}。', '',
          '## Rank IC 与极端五分位差', '',
          '| 基准 | 期间 | A IC | B IC | B* IC | A Q5−Q1 | B Q5−Q1 | B* Q5−Q1 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for o in OPTIONS:
        for label,period in zip(periods,PERIOD):
            v=pick(MET,option=o,period=period).set_index('model').loc[list(MODELS)]
            body.append('| '+' | '.join([names[o],label,*[f(x,4) for x in v.rank_ic],*[pct(x)+' pp' for x in v.q5_q1]])+' |')
    body += ['', '按模型预测分数分组。各年度与合并样本的五组平均实际 CAR 均未呈严格单调变化。合并 Rank IC 用全体事件重算，行业基准下 A 高于 B 和 B*，市场基准下三者均为负。', '',
             '## 增量 Rank IC 与配对 95% 置信区间', '',
             '| 基准 | 期间 | 比较 | Δ Rank IC | 发行人聚类 | 日历时间块 | Holm p |',
             '|---|---|---|---:|---|---|---:|']
    for o in OPTIONS:
        for label,period in zip(periods,PERIOD):
            for row in delta_rows(o,period):
                body.append('| '+' | '.join([names[o],label,*row])+' |')
    body += ['', '合并期四个 LM 增量均为负，各自的两个区间都包含 0。Holm 校正后 p 值均为 1.000，5% 水平下没有显著增量。两种方法各使用 2,499 次有效配对重抽样。2026 仅有 3 个名义双月块，因此只报告描述性点估计。', '',
             '行业调整下，2025 和 2026 的两个增量均为负。市场调整下，B 在 2025 接近零、2026 略正，B* 由略正变负。五个完整季度的正 IC 比例，市场 A/B/B* 为 60%/20%/20%，行业为 80%/80%/80%。五个完整季度低于预设的 ICIR 所需六期。', '',
             '## 数据核验与限制', '',
             f'医疗行业基准通过 {VAL.passed.sum()} 项验证。2024 年历史日基准和两个事件窗口均在 1e−10 原始收益单位内复现。204 个 CIZ 重叠目标的最大误差为 {VAL.iloc[-1].max_abs_error*10000:.5f} bp。', '',
             'Early CAR[−1,+1] 直到 Day +1 收盘后才可观测。原流程未验证电话会议文字稿与财务输入在该时点是否均已发布，不能把结果当作电话会议即时信号或可执行策略。原 LM 字典下 hedging_answers 与 uncertainty_answers 完全相同，按冻结规格保留。预测为秩尺度，收益分组含重叠事件窗口，不能解释为可投资组合收益。', '',
             '## 文件', '',
             '- [交互式仪表板](dashboard.html)，PDF 由 `render_report.py` 本地生成。',
             '- [绩效](model_performance.csv)、[增量](incremental_comparison.csv)、[置信区间](bootstrap_inference.csv)、[五分位结果](quintile_returns.csv)',
             '- [季度 IC](quarterly_ic.csv)、[稳定性](stability_summary.csv)、[目标覆盖](target_coverage.csv)、[特征覆盖](feature_coverage.csv)、[行业基准验证](sector_benchmark_validation.csv)',
             '- [冻结规格](frozen_experiment_spec.json)及[方法与复现](methodology.md)']
    (HERE/'summary_zh.md').write_text('\n'.join(body)+'\n',encoding='utf-8')


if __name__ == '__main__':
    make_pdf()
    make_html()
    make_summary_zh()
