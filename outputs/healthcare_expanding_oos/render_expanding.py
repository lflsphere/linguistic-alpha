"""Build the research report and aggregate-only interactive dashboard from the evaluation CSVs."""

import argparse
from html import escape
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATIC_RENDER = HERE.parent / 'healthcare_extended_oos' / 'render_report.py'
styles = importlib.util.spec_from_file_location('report_style', STATIC_RENDER)
ui = importlib.util.module_from_spec(styles)
styles.loader.exec_module(ui)
TITLE = 'HEALTHCARE LINGUISTIC ALPHA RESEARCH'
SUBTITLE = 'Original Sector Pipeline — Annual Expanding Window Analysis'
PDF = 'Healthcare_Static_vs_Annual_Expanding_Window.pdf'
OPTIONS = {'market': 'S&P 500-adjusted', 'sector': 'Healthcare sector-adjusted'}
PERIODS = ('2024 diagnostic', '2025', '2026 partial', '2025–26 pooled')
SHORT = ('2024 diag.', '2025', '2026 partial', 'Pooled')
MODELS = ('A', 'B', 'B*')
COLOR = {'A': '#244D77', 'B': '#007E79', 'B*': '#775A87'}
MODE = {'Static': '#687A8B', 'Expanding': '#007E79'}
PANEL = ['Static', 'Expanding']
PALE = '#DCEEEB'
NAMES = ('model_performance', 'incremental_comparison', 'static_to_expanding', 'quintile_returns',
         'bootstrap_inference', 'refit_training_audit', 'evaluation_match_audit', 'quarterly_ic', 'stability_summary')


def load(folder):
    return {name: pd.read_csv(folder / f'{name}.csv') for name in NAMES}


def pick(df, **kw):
    return ui.pick(df, **kw)


def fmt(x, places=4):
    return ui.f(x, places)


def signed(x, places=4):
    return ui.signed(x, places)


def pct(x):
    return ui.pct(x)


def band(row):
    return '—' if pd.isna(row.lower) else f'[{fmt(row.lower)}, {fmt(row.upper)}]'


def page(num, section, headline, dek):
    f = plt.figure(figsize=(8.5,11), facecolor='white')
    a = f.add_axes([0,0,1,1]); a.set(xlim=(0,612), ylim=(792,0)); a.axis('off')
    ui.rect(a,0,0,612,76,ui.NAVY); ui.rect(a,0,76,612,2,ui.TEAL)
    ui.t(a,42,23,TITLE,13,'white','bold'); ui.t(a,42,48,SUBTITLE,9.5,'#C7DBE2')
    ui.t(a,42,98,f'{num:02d}  /  {section.upper()}',8.3,ui.TEAL,'bold')
    ui.t(a,42,116,headline,23.4,ui.NAVY,'bold'); ui.t(a,42,153,dek,9.4,ui.MUTE)
    a.plot([42,570],[751,751],color=ui.LINE,lw=.8)
    ui.t(a,42,764,'POST-HOC EXPLORATORY RESEARCH  /  CAR[+2,+20]',7.4,ui.MUTE)
    ui.t(a,570,764,f'{num:02d}',8.2,ui.NAVY,'bold',ha='right')
    return f,a


def text(ax,y,s,**kw):
    return ui.p(ax,42,y,s,**kw)


def training_legend(ax,y):
    ui.rect(ax,42,y+3,11,4,MODE['Static']); ui.t(ax,58,y,'Static',8.4,ui.MUTE)
    ui.rect(ax,142,y+3,11,4,MODE['Expanding']); ui.t(ax,158,y,'Annual Expanding',8.4,ui.MUTE)


def line_chart(ax, table, option, y_field, span, metric):
    for model in MODELS:
        for schedule in PANEL:
            d = pick(table, option=option, model=model, training=schedule).set_index('period').reindex(PERIODS)
            if d[y_field].notna().any():
                ax.plot(range(4),d[y_field], '-o' if schedule=='Expanding' else '--s',
                        color=COLOR[model], lw=1.55 if schedule=='Expanding' else 1.1, ms=4.1,
                        markerfacecolor=COLOR[model] if schedule=='Expanding' else 'white')
    ax.set_xticks(range(4),SHORT)
    ax.set_ylim(-span,span)
    ax.set_ylabel(metric,fontsize=8,color=ui.MUTE)
    ui.chart(ax)


def quint(ax, q, o, schedule, span):
    for m in MODELS:
        d=pick(q,option=o,period=PERIODS[-1],training=schedule,model=m)
        ax.plot(d.group,100*d.mean_car,'-o',color=COLOR[m],lw=1.45,ms=4.3)
    ax.set_xticks(range(1,6),['Q1','Q2','Q3','Q4','Q5'])
    ax.set_ylim(-span,span)
    ax.set_ylabel('Mean CAR (%)',fontsize=8,color=ui.MUTE)
    ui.chart(ax)


def grouped(ax, gains, option, period, span):
    d=pick(gains,option=option,period=period).set_index('model').loc[list(MODELS)]
    for j,series in enumerate(PANEL):
        ax.bar(np.arange(3)+(j-.5)*.26,d[f'rank_ic_{series.lower()}'],width=.23,color=MODE[series])
    ax.set_xticks(range(3),MODELS); ax.set_ylim(-span,span)
    ax.set_ylabel('Rank IC',fontsize=8,color=ui.MUTE)
    ui.chart(ax)


def forest(ax, inf, family):
    d = pick(inf,period=PERIODS[-1],family=family)
    keys = list(d.groupby(['option','model'],sort=False).groups)
    lim=max(.18,np.ceil(np.nanmax(abs(d[['lower','upper']].to_numpy()))*20)/20)
    for j,(o,m) in enumerate(keys):
        v=pick(d,option=o,model=m).set_index('method')
        for k,method in enumerate(['issuer_cluster','time_block']):
            e=v.loc[method]; y=len(keys)-1-j+(.10 if k==0 else -.10)
            ax.hlines(y,e.lower,e.upper,colors=COLOR[m],lw=1.5,ls='-' if k==0 else '--')
            ax.plot(e.point,y,'o' if k==0 else 's',color=COLOR[m],mfc=COLOR[m] if k==0 else 'white',ms=4.2)
    ax.set_xlim(-lim,lim); ax.axvline(0,color=ui.MUTE,lw=.8)
    ax.set_xlabel('Δ Rank IC',fontsize=8,color=ui.MUTE)
    ax.set_yticks(range(len(keys)),[f'{"Market" if o=="market" else "Sector"} {m}' for o,m in keys[::-1]])
    ui.chart(ax,zero=False)


def make_pdf(folder, D, spec):
    G,I,Q,B,R,T,S = [D[x] for x in ('static_to_expanding','incremental_comparison','quintile_returns',
                'bootstrap_inference','refit_training_audit','quarterly_ic','stability_summary')]
    with PdfPages(folder/PDF,metadata={'Title':'Healthcare Linguistic Alpha — Static vs Annual Expanding Window','Subject':SUBTITLE}) as pdf:
        f,a=page(1,'Executive findings','Annual refits did not restore LM alpha',
                 'Small improvements versus the updated financial baseline come with wide uncertainty.')
        for x,big,l in [(42,'315','Matched OOS calls'),(222,'61','Pooled OOS issuers'),(402,'6','Frozen model specifications')]:
            ui.rect(a,x,180,168,72,ui.SOFT); ui.t(a,x+11,191,big,25,ui.NAVY,'bold'); ui.t(a,x+11,226,l,8.8,ui.MUTE)
        ui.label(a,278,'Pooled 2025–26 rank performance')
        rows=[]
        for r in pick(G,period=PERIODS[-1]).itertuples():
            inc=pick(I,period=PERIODS[-1],training='Expanding',option=r.option,model=r.model)
            rows.append((OPTIONS[r.option],r.model,fmt(r.rank_ic_static),fmt(r.rank_ic_expanding),signed(r.delta_rank_ic),
                         signed(inc.delta_rank_ic.iloc[0]) if len(inc) else '—'))
        ui.table(a,302,['Benchmark','Model','Static IC','Exp. IC','Exp. − Static','LM vs A'],rows,[49,219,266,333,404,502],24,8.6)
        ui.note(a,480,'Table 1. Each row uses the same 315 calls from 61 issuers. Rank IC and changes are dimensionless. LM vs A compares Expanding models.')
        ui.label(a,523,'Separate the baseline from the LM increment')
        text(a,548,'The Healthcare-adjusted financial baseline improves in 2025 and partial 2026. The S&P 500-adjusted financial baseline deteriorates in both periods. All six models have lower pooled Rank IC after updating.')
        text(a,600,'The four pooled LM increments range from −0.0007 to +0.0064. Both paired 95% intervals include zero for each increment. Industry-adjusted LM5 trails its annually updated baseline in the pool.')
        ui.rect(a,42,655,528,66,PALE)
        ui.t(a,53,666,'POST-HOC ROBUSTNESS CHECK',8.5,ui.TEAL,'bold')
        ui.p(a,53,686,'The Static OOS results were already known. 2024 informed original model selection. New scores are available only after the Day +1 close under the model convention.',width=101,size=8.5)
        ui.add(pdf,f)


        f,a=page(2,'Research design','Same models, annual refit schedule',
                 'Each year starts with a new rank target, preprocessing fit and estimator fit.')
        ui.label(a,179,'Refits at the first eligible Day +1 close')
        rows=[(r.period,r.refit_cutoff_et[:10],f'{r.training_events:,}',r.train_2024,r.train_2025,r.matched_events)
              for r in pick(R,option='market').itertuples()]
        ui.table(a,203,['Evaluation','Cutoff date ET','Train n','2024 rows','2025 rows','Test n'],rows,[49,171,290,360,433,521],21,8.6)
        ui.note(a,296,'Table 2. Cutoffs are 16:00 Eastern. Counts are per benchmark. Both model schedules reproduce the 204 archived 2024 scores exactly.')
        ui.label(a,334,'Frozen estimators and predictors')
        rows=[]
        for option in OPTIONS:
            for model,recipe in spec['specifications'][option].items():
                params=recipe['parameters']; algo=f"{recipe['estimator']} α {params['alpha']:g}"
                if 'l1_ratio' in params:
                    algo+=f", L1 {params['l1_ratio']:g}"
                rows.append((OPTIONS[option],model,algo,len(recipe['columns']),{'A':'Controls','B':'Controls + LM5','B*':'Controls + LM9'}[model]))
        ui.table(a,358,['Benchmark','Model','Estimator','Inputs','Feature set'],rows,[49,219,262,423,477],20,8.4)
        text(a,520,'Original median imputation, winsorization, scaling and feature ordering are fitted separately on each eligible expanding window. The target is the unnormalized one-based average midrank of the training sample CAR.')
        text(a,576,'Both benchmark versions use the same events. S&P 500 returns use SPRTRN. The industry option reuses saved static GICS Healthcare CAR labels and the original same-day cap-weighted benchmark. No sector return is rebuilt.')
        text(a,633,'The model sees only prior-year calls with outcomes mature strictly before the refit cutoff. The 2026 refit can learn from the 211 matched 2025 calls. The pre-existing 2024 target gap remains excluded.')
        ui.add(pdf,f)

        f,a=page(3,'Rank performance','Updating helps sector IC by year',
                 'Pooled ranking deteriorates for every model under both benchmarks.')
        training_legend(a,179)
        for j,o in enumerate(OPTIONS):
            x=42+264*j
            ui.t(a,x,207,OPTIONS[o],10,ui.NAVY,'bold')
            grouped(ui.subplot(f,x+37,231,207,113),G,o,PERIODS[-1],.30)
        ui.note(a,369,'Figure 1. Pooled Rank IC on 315 matched events and 61 issuers. Each schedule pools its raw predicted scores and recomputes Spearman correlation.')
        for j,o in enumerate(OPTIONS):
            y=407+151*j
            ui.t(a,42,y,OPTIONS[o],10,ui.NAVY,'bold')
            rows=[]
            for k,period in enumerate(PERIODS):
                g=pick(G,option=o,period=period).set_index('model')
                rows.append((SHORT[k],g.events.iloc[0],*[f'{fmt(g.loc[m,"rank_ic_static"])}  →  {fmt(g.loc[m,"rank_ic_expanding"])}' for m in MODELS]))
            ui.table(a,y+20,['Period','Calls','A  Static → Exp.','B  Static → Exp.','B*  Static → Exp.'],rows,
                     [49,157,202,326,450],19,8.2)
        ui.note(a,702,'Table 3. Rank IC and comparison use identical events per benchmark and year. Cross-year shifts in the unnormalized model-score scale can affect the pooled ranks. Annual results retain their within-year scale.')
        ui.add(pdf,f)

        f,a=page(4,'Incremental language','Small LM increments after refitting',
                 'Post-update pooled increments remain small for both benchmark definitions.')
        ui.t(a,42,178,'B  Selected LM5    /    B*  Full LM9     ·     Solid Expanding    /    Dashed Static',8.7,ui.MUTE)
        for j,o in enumerate(OPTIONS):
            x=42+264*j; ui.t(a,x,207,OPTIONS[o],9.6,ui.NAVY,'bold')
            line_chart(ui.subplot(f,x+38,230,207,131),I,o,'delta_rank_ic',.06,'Δ Rank IC vs A')
        ui.note(a,385,'Figure 2. B and B* in consistent model colors. Zero marks the financial baseline. N = 204, 211, 104 and 315 by period per benchmark. 2024 is a historical diagnostic.')
        for j,o in enumerate(OPTIONS):
            y=426+j*125
            ui.t(a,42,y,OPTIONS[o],9.8,ui.NAVY,'bold')
            rows=[]
            for k,period in enumerate(PERIODS):
                v=pick(I,option=o,period=period).pivot(index='model',columns='training',values='delta_rank_ic')
                rows.append((SHORT[k],*[signed(v.loc[m,s]) for m in ('B','B*') for s in PANEL]))
            ui.table(a,y+18,['Period','B−A Static','B−A Expanding','B*−A Static','B*−A Expanding'],rows,[49,168,264,367,465],17,8.2)
        ui.note(a,666,'Table 4. The difference between a language model and A within a schedule measures incremental Rank IC. It can improve while the total model IC falls.')
        text(a,699,'In the 2026 partial sample, sector-adjusted B is slightly above A and B* is slightly below. Both estimates are descriptive.',size=9.0)
        ui.add(pdf,f)

        f,a=page(5,'Economic separation','Quintile separation weakens in the pool',
                 'Scores sort events within a benchmark and schedule. Return windows can overlap.')
        ui.legend(a,178)
        for j,o in enumerate(OPTIONS):
            for k,training in enumerate(PANEL):
                x=42+264*k; y=208+143*j
                ui.t(a,x,y,f'{"S&P 500" if o=="market" else "Healthcare"}  /  {training}',9.6,ui.NAVY,'bold')
                quint(ui.subplot(f,x+42,y+21,198,93),Q,o,training,5.0)
        ui.note(a,493,'Figure 3. Pooled 315 calls and 61 issuers, 63 events in each quintile. Q1 is the lowest predicted-score group. Both schedules use the same vertical scale and the same realized labels.')
        rows=[(OPTIONS[o],s,*[pct(pick(G,option=o,period=PERIODS[-1],model=m)[f'q5_q1_{s.lower()}'].iloc[0])
                               for m in MODELS],'Mixed') for o in OPTIONS for s in PANEL]
        ui.table(a,530,['Benchmark','Schedule','A spread','B spread','B* spread','Profile'],rows,[49,213,307,371,434,505],21,8.3)
        ui.note(a,643,'Table 5. Spreads are Q5 minus Q1 in percentage points. Mixed indicates a non-monotonic profile. Full event counts, mean, median and sample SD are saved for all periods.')
        text(a,687,'Pooled market spreads turn negative under annual refits. Industry-adjusted spreads stay positive. These are realized event-level CAR groups and cannot be interpreted as strategy returns.',size=9.0)
        ui.add(pdf,f)

        f,a=page(6,'Paired inference','LM intervals all cross zero',
                 'Issuer and time resampling quantify uncertainty for both types of comparison.')
        for j,family in enumerate(('LM','schedule')):
            x=42+j*263
            ui.t(a,x,184,'Expanding LM versus A' if j==0 else 'Expanding versus Static',10.2,ui.NAVY,'bold')
            forest(ui.subplot(f,x+54,217,181,143),B,family)
        ui.note(a,399,'Figure 4. Pooled matched n = 315, 61 issuers, 9 nominal two-month blocks. Solid circle = issuer cluster. Dashed square = calendar time block. Both methods use 2,499 paired draws.')
        rows=[]
        for family in ('LM','schedule'):
            for (o,m),d in pick(B,period=PERIODS[-1],family=family).groupby(['option','model'],sort=False):
                d=d.set_index('method'); c=d.loc['issuer_cluster']
                name=f'{m} − A' if family=='LM' else f'{m} update'
                rows.append(('Market' if o=='market' else 'Healthcare', name, signed(c.point),band(c),band(d.loc['time_block']),fmt(c.holm_p_pooled,3)))
        ui.table(a,437,['Benchmark','Comparison','Δ IC','Issuer 95% CI','Time-block 95% CI','Holm p'],rows,
                 [49,133,230,292,402,528],17,7.9)
        ui.note(a,634,'Table 6. Separate pooled Holm adjustments cover four LM contrasts and six schedule contrasts. Pointwise intervals are percentile bootstrap intervals. Partial 2026 has only three nominal time blocks.')
        text(a,672,'Individual market schedule intervals exclude zero. The schedule-family Holm p-value is 0.1128 for each. No pooled contrast passes the exploratory screen.',size=9.1)
        ui.note(a,720,'The inference conditions on these fitted scores. The prior Static analysis and few independent blocks limit interpretation.')
        ui.add(pdf,f)

        f,a=page(7,'Temporal stability','Quarterly IC fluctuates materially',
                 'Five comparable full quarters support sign-share and dispersion summaries.')
        ui.legend(a,177)
        for j,o in enumerate(OPTIONS):
            x=42+j*263
            ui.t(a,x,204,OPTIONS[o],9.6,ui.NAVY,'bold')
            ax=ui.subplot(f,x+36,230,214,138)
            for m in MODELS:
                for s in PANEL:
                    d=pick(T,option=o,training=s,model=m)
                    ax.plot(range(6),d.rank_ic,'-o' if s=='Expanding' else '--',c=COLOR[m],lw=1.4 if s=='Expanding' else .95,ms=3.5)
            ax.set_ylim(-.50,.50); ax.set_ylabel('Rank IC',fontsize=8,color=ui.MUTE)
            ax.set_xticks(range(6),['’25 Q1','Q2','Q3','Q4','’26 Q1','Q2*']); ui.chart(ax)
        ui.note(a,394,'Figure 5. Solid Expanding, dashed Static. Quarterly n = 52, 51, 53, 55, 55 and 49. * Partial Q2 is shown and excluded from the full-quarter stability summary.')
        rows=[]
        for (o,m),z in S.groupby(['option','model'],sort=False):
            v=z.set_index('training')
            rows.append((OPTIONS[o],m,*[fmt(v.loc[s,'sd_ic'],3) for s in PANEL],
                         *[f'{v.loc[s,"positive_ic_share"]:.0%}' for s in PANEL]))
        ui.table(a,450,['Benchmark','Model','Static SD','Exp. SD','Static positive','Exp. positive'],rows,
                 [49,215,270,341,409,493],22,8.1)
        ui.note(a,616,'Table 7. IC dispersion and positive share use the same five full quarters. Six are required for ICIR under the frozen rule. Pearson correlations are supplementary and are available in the aggregate performance CSV.')
        text(a,669,'Model scores are on the rank scale. The analysis contains no continuous-return R², MAE or prediction calibration.',size=9.1)
        ui.add(pdf,f)

        f,a=page(8,'Availability and limitations','Maturity clock and data checks',
                 'Train only on already observed outcomes and preserve the original target conventions.')
        rows=[(r.period,str(r.training_events),r.latest_label_close_et[:10],r.missing_target,r.unresolved_window,r.at_or_after_cutoff)
              for r in pick(R,option='market').itertuples()]
        ui.table(a,181,['Evaluation','Train n','Last label close','Missing target','Unknown end','Too late'],rows,
                 [49,160,225,334,428,519],23,8.0)
        ui.note(a,283,'Table 8. Counts apply to both benchmarks. The latest 2025 training outcomes used for the 2026 fit ended on January 6, 2026, ahead of January 22. No current-year calls enter training.')
        paragraphs=[('Recovered maturity and legacy availability','Training maturity is the close of Day +20 on the saved CRSP calendar. Every used window has a known endpoint. Two older archived labels have 18 paired daily returns out of 19. Both are valid under the original minimum coverage rule and their whole windows elapsed before refit.'),
                    ('Execution clock','Early CAR is available after Day +1 close. The original generic pipeline does not establish transcript delivery or all-feature availability at that point. No additional timing screen was imposed.'),
                    ('Dictionary and inference','Original hedging_answers and uncertainty_answers are identical. The frozen models retain that redundancy. Previously examined 2024 and 2025–26 results limit independent interpretation.'),
                    ('Reproduction','The manifest pins the committed Static files, native models, archived predictions and the validated local inputs. Both schedules use the same event IDs and realized CARs. Local event-level ledgers are excluded from the dashboard.')]
        for j,(heading,body) in enumerate(paragraphs):
            y=322+93*j
            ui.label(a,y,heading); text(a,y+22,body,width=103,step=13)
        ui.add(pdf,f)


CSS = '''
:root{--navy:#12253d;--ink:#263b4e;--mute:#526678;--line:#cad8e0;--teal:#007e79;--pale:#dceeea;--wash:#edf2f5;--paper:#fff;--static:#687a8b}
*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:var(--ink);font:16px/1.52 "Segoe UI",sans-serif}
.wrap{max-width:1250px;margin:auto;padding:0 25px}header{background:var(--navy);color:white;border-bottom:3px solid #37a19a;padding:20px 0}
header strong{display:block;letter-spacing:.1em;font-size:.91rem}header span{color:#c0d7e0;font-size:.9rem}nav{background:#fff;border-bottom:1px solid var(--line)}nav .wrap{display:flex;flex-wrap:wrap;gap:8px 24px;padding-block:10px}a{color:#00655f;text-underline-offset:3px}nav a{text-decoration:none;font-size:.88rem;font-weight:650}
main{padding-bottom:32px}.hero{padding:38px 0 20px;display:grid;grid-template-columns:1.62fr 1fr;gap:25px;align-items:start}.hero h1{margin:4px 0 10px;font-size:clamp(2.1rem,4vw,3.5rem);line-height:1.04;color:var(--navy);letter-spacing:-.04em;max-width:770px}.hero p{max-width:730px;margin:0 0 10px}.kicker{font-weight:750;font-size:.72rem;letter-spacing:.105em;color:#006a64;text-transform:uppercase}.aside{padding:17px 20px;background:var(--pale);border-left:3px solid var(--teal)}.aside p{font-size:.94rem;margin:6px 0}.aside strong{color:var(--navy)}
section{scroll-margin-top:10px;background:var(--paper);border:1px solid var(--line);margin:0 0 17px;padding:clamp(15px,3.2vw,32px)}section h2{font-size:clamp(1.35rem,2.5vw,2rem);line-height:1.14;margin:3px 0 7px;color:var(--navy);letter-spacing:-.02em}h3{margin:.2rem 0 .55rem;color:var(--navy);font-size:1rem}.sub,figcaption{color:var(--mute);font-size:.88rem;margin:.45em 0 .9em}.grid{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:22px;align-items:start}.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}.tile{background:#f6f9fa;padding:10px 12px;border:1px solid var(--line)}.tile span{display:block;color:var(--mute);font-size:.78rem}.tile b{display:block;font-size:1.24rem;color:var(--navy)}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:10px 15px;padding:.8rem 0 .7rem}label{font-size:.85rem;font-weight:650;color:var(--navy)}select{display:block;font:inherit;border:1px solid #8ba0af;background:white;color:var(--navy);padding:.52em .8em;margin-top:.28em;min-width:128px;max-width:100%}:focus-visible{outline:3px solid #007e79;outline-offset:2px}
.table-wrap{overflow:auto;margin:.5rem 0 1rem}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.88rem}th,td{text-align:left;padding:.55rem .6rem;white-space:nowrap}th{background:var(--navy);color:white;font-size:.81rem}tbody tr:nth-child(odd){background:#f0f5f6}.key{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:.85rem;color:var(--mute)}.swatch{display:inline-block;width:12px;height:5px;margin-right:5px;vertical-align:middle}figure{margin:0;min-width:0;position:relative}figure svg{display:block;width:100%;height:auto;font:12px "Segoe UI",sans-serif}figcaption{margin-top:.55rem}.live{min-height:2.1em;padding:.38rem .6rem;background:#f0f5f6;font-size:.86rem;color:var(--navy)}
details{padding:.6rem 0;border-top:1px solid var(--line);margin:0}summary{cursor:pointer;font-weight:650;color:var(--navy)}.download{display:flex;flex-wrap:wrap;gap:5px 18px}.download a{font-size:.86rem}footer{padding:12px 0 33px;color:var(--mute);font-size:.85rem}
.grid>*{min-width:0}details code{overflow-wrap:anywhere}
@media(max-width:790px){.hero,.grid{grid-template-columns:minmax(0,1fr);gap:12px}.hero{padding-top:25px}.wrap{padding-inline:12px}.controls{gap:8px}nav .wrap{gap:5px 14px}}@media(max-width:390px){th,td{padding:.43rem .52rem}select{min-width:110px}}
'''

JS = r'''
const D=JSON.parse(document.getElementById('data').textContent);
const periods=['2024 diagnostic','2025','2026 partial','2025–26 pooled'], short=['2024 diag.','2025','2026 partial','Pooled'];
const choices=['A','B','B*'], model={A:'#244d77',B:'#007e79','B*':'#775a87'}, mode={Static:'#687a8b',Expanding:'#007e79'};
const $=id=>document.getElementById(id), num=(x,d=4)=>x==null?'—':Number(x).toFixed(d).replace('-', '−');
const change=x=>x==null?'—':(x>0?'+':'')+num(x), percent=x=>num(x*100,2);
function table(head,rows){return `<div class="table-wrap"><table><thead><tr>${head.map(x=>`<th scope="col">${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
const lines=(t,s)=>(t===s?`stroke-dasharray="6 5"`:'');
function point(x,y,c,txt,series){
 const icon=series==='Static'?`<rect x="${x-4}" y="${y-4}" width="8" height="8" fill="white" stroke="${c}" stroke-width="2"/>`:`<circle cx="${x}" cy="${y}" r="4.5" fill="${c}"/>`;
 return `<g tabindex="0" data-tip="${txt}" aria-label="${txt}"><title>${txt}</title>${icon}<circle cx="${x}" cy="${y}" r="12" fill="transparent"/></g>`;
}
function linePlot(id,series,labels,lim,ticks,unit){
 const x=i=>65+i*485/(labels.length-1), y=v=>148-v*106/lim;
 const items=ticks.map(v=>`<path d="M65 ${y(v)}H550" stroke="${v===0?'#738697':'#d5dfe5'}"/><text x="54" y="${y(v)+4}" text-anchor="end">${num(v,lim>1?1:2)}</text>`).join('');
 const plotted=series.map(s=>`<polyline fill="none" stroke="${s.color}" stroke-width="2.25" ${lines(s.schedule,'Static')} points="${s.vals.map((r,j)=>`${x(j)},${y(r.value)}`).join(' ')}"/>${s.vals.map((r,j)=>point(x(j),y(r.value),s.color,r.info,s.schedule)).join('')}`).join('');
 $(id).innerHTML=`<svg viewBox="0 0 570 290" aria-label="${unit}"><g fill="#526678">${items}<text x="65" y="25">${unit}</text>${labels.map((l,i)=>`<text x="${x(i)}" y="276" text-anchor="middle">${l}</text>`).join('')}</g>${plotted}</svg>`;
 connect(id);
}
function connect(id){
 for(const hit of $(id).querySelectorAll('[data-tip]')) for(const event of ['mouseenter','focus','click']) hit.addEventListener(event,()=>{$(id+'-tip').textContent=hit.dataset.tip;});
}
function mark(id,benchmark,period){
 const r=D.static_to_expanding.filter(d=>d.option===benchmark&&d.period===period);
 const x=j=>110+160*j,y=v=>136-v*100/.31;
 const ticks=[-.3,-.15,0,.15,.3].map(v=>`<path d="M60 ${y(v)}H530" stroke="${v===0?'#738697':'#d5dfe5'}"/><text x="51" y="${y(v)+4}" text-anchor="end">${num(v,2)}</text>`).join('');
 const marks=r.map((d,j)=>['Static','Expanding'].map((s,k)=>{
  const v=d['rank_ic_'+s.toLowerCase()],xx=x(j)+k*42-21, yy=y(v),h=Math.abs(y(0)-yy);
  const info=`${d.model} ${s} Rank IC ${num(v)} · n=${d.events} · ${d.issuers} issuers`;
  return `<g tabindex="0" data-tip="${info}" aria-label="${info}"><title>${info}</title><rect x="${xx-14}" y="${Math.min(yy,y(0))}" width="28" height="${Math.max(h,.6)}" fill="${mode[s]}"/></g>`;
 }).join('')+`<text x="${x(j)}" y="263" text-anchor="middle">${d.model}</text>`).join('');
 $(id).innerHTML=`<svg viewBox="0 0 550 276" aria-label="Rank IC"><g fill="#526678"><text x="60" y="21">Rank IC</text>${ticks}${marks}</g></svg>`;
 connect(id);
 $('cmp-table').innerHTML=table(['Model','Static IC','Expanding IC','Change','Calls'],r.map(d=>[d.model,num(d.rank_ic_static),num(d.rank_ic_expanding),change(d.delta_rank_ic),d.events]));
}
const noCI='Descriptive only under the frozen time-block rule.';
function infer(benchmark,period,family){
 const a=D.bootstrap_inference.filter(d=>d.option===benchmark&&d.period===period&&d.family===family);
 const keys=choices.filter(m=>a.some(d=>d.model===m));
 const range=D.bootstrap_inference.filter(d=>d.family===family).flatMap(d=>[d.lower,d.upper,d.point]).filter(x=>x!=null);
 const lim=Math.max(.10,Math.ceil(Math.max(...range.map(Math.abs))*20)/20), x=v=>90+(v+lim)*450/(2*lim), h=95+keys.length*47;
 const ticks=[-lim,-lim/2,0,lim/2,lim].map(v=>`<path d="M${x(v)} 32V${h-32}" stroke="${v===0?'#738697':'#d5dfe5'}"/><text x="${x(v)}" y="${h-12}" text-anchor="middle">${num(v,2)}</text>`).join('');
 const marks=a.map(d=>{const y=58+keys.indexOf(d.model)*47+(d.method==='issuer_cluster'?-6:6), s=d.method==='issuer_cluster'?'Issuer cluster':'Time block', txt=`${d.model} · ${s} · ${change(d.point)} · ${d.lower==null?'descriptive':`95% CI [${num(d.lower)}, ${num(d.upper)}]`} · n=${d.events}`;
  return (d.lower==null?'':`<path d="M${x(d.lower)} ${y}H${x(d.upper)}" stroke="${model[d.model]}" stroke-width="2" ${lines(d.method,'time_block')}/>`)+point(x(d.point),y,model[d.model],txt,d.method==='time_block'?'Static':'Expanding');}).join('');
 $('ci-plot').innerHTML=`<svg viewBox="0 0 570 ${h}" aria-label="95% confidence intervals"><g fill="#526678"><text x="90" y="22">Δ Rank IC</text>${ticks}${keys.map((m,j)=>`<text x="20" y="${62+j*47}">${family==='LM'?m+' − A':m}</text>`).join('')}</g>${marks}</svg>`;
 connect('ci-plot');
 const fmt=d=>d.lower==null?'—':`[${num(d.lower)}, ${num(d.upper)}]`;
 $('ci-table').innerHTML=table(['Model','Δ IC','Issuer 95% CI','Time block 95% CI','Holm p'],keys.map(m=>{
  const i=a.find(d=>d.model===m&&d.method==='issuer_cluster'),t=a.find(d=>d.model===m&&d.method==='time_block');return [m,change(i.point),fmt(i),fmt(t),num(i.holm_p_pooled,3)];
 }));
 $('ci-count').textContent=`${a[0].events} calls · ${a[0].issuer_clusters} issuers · ${a[0].nominal_time_blocks} nominal blocks. ${a[0].n_valid?`${a[0].n_valid.toLocaleString()} draws per method.`:noCI}`;
}
function alpha(){
 const option=$('alpha-bench').value, m=$('alpha-model').value, modeFilter=$('alpha-training').value;
 const schedules=modeFilter==='Both'?['Static','Expanding']:[modeFilter];
 const a=D.incremental_comparison.filter(d=>d.option===option&&d.model===m);
 const series=schedules.map(s=>({schedule:s,color:mode[s],vals:periods.map(p=>{
  const {delta_rank_ic:value,events,issuers}=a.find(d=>d.training===s&&d.period===p);return {value,info:`${p} · ${s} · ${m} − A ${change(value)} · n=${events} · ${issuers} issuers`};
 })}));
 linePlot('alpha-plot',series,short,.06,[-.06,-.03,0,.03,.06],'Incremental Rank IC');
 $('alpha-table').innerHTML=table(['Period','Calls',...schedules],periods.map((p,j)=>[p,a.find(d=>d.period===p).events,...series.map(s=>change(s.vals[j].value))]));
}
function quintile(){
 const o=$('q-bench').value,p=$('q-period').value,s=$('q-training').value;
 const a=D.quintile_returns.filter(d=>d.option===o&&d.period===p&&d.training===s);
 const rows=D.model_performance.filter(d=>d.option===o&&d.period===p&&d.training===s);
 const lim=Math.max(6,Math.ceil(Math.max(...D.quintile_returns.map(v=>Math.abs(v.mean_car*100)))));
 linePlot('q-plot',choices.map(m=>({schedule:'Expanding',color:model[m],vals:a.filter(r=>r.model===m).map(r=>({value:r.mean_car*100,info:`${m} Q${r.group} · Mean realized CAR ${percent(r.mean_car)}% · n=${r.n}`}))})),['Q1','Q2','Q3','Q4','Q5'],lim,[-lim,-lim/2,0,lim/2,lim],'Mean realized CAR (%)');
 $('q-table').innerHTML=table(['Model','Calls','Q5 − Q1 (pp)','Monotonicity'],rows.map(d=>[d.model,d.events,percent(d.q5_q1),d.profile]));
 $('q-detail').innerHTML=table(['Model','Group','Calls','Mean (%)','Median (%)','SD (%)'],a.map(r=>[r.model,'Q'+r.group,r.n,percent(r.mean_car),percent(r.median_car),percent(r.sd_car)]));
}
function update(){alpha();mark('cmp-plot',$('cmp-bench').value,$('cmp-period').value);quintile();infer($('ci-bench').value,$('ci-period').value,$('ci-family').value);}
for(const item of document.querySelectorAll('select')) item.addEventListener('change',()=>{for(const e of document.querySelectorAll('.live'))e.textContent='Focus or hover over a chart mark for exact values.';update();});
update();
'''


def select(identifier, label, values, chosen=None):
    tags = ''.join(f'<option value="{escape(k)}"{" selected" if k==chosen else ""}>{escape(v)}</option>'
                   for k,v in values.items())
    return f'<label for="{identifier}">{label}<select id="{identifier}">{tags}</select></label>'


def figure(label, identifier, caption):
    return (f'<figure aria-label="{label}"><div id="{identifier}"></div><figcaption>{caption}</figcaption>'
            f'<div class="live" aria-live="polite" id="{identifier}-tip">Focus or hover over a chart mark for exact values.</div></figure>')


def key(colors):
    return '<div class="key">'+''.join(f'<span><i class="swatch" style="background:{v}"></i>{k}</span>' for k,v in colors.items())+'</div>'


def make_html(folder, D, spec):
    G,I,R,S,C = [D[n] for n in ('static_to_expanding','incremental_comparison','refit_training_audit',
                               'stability_summary','evaluation_match_audit')]
    pooled = pick(C,option='market',period=PERIODS[-1]).iloc[0]
    gains=[]
    for row in pick(G,period=PERIODS[-1]).itertuples():
        m=pick(I,training='Expanding',option=row.option,period=PERIODS[-1],model=row.model)
        gains.append((OPTIONS[row.option],row.model,fmt(row.rank_ic_static),fmt(row.rank_ic_expanding),signed(row.delta_rank_ic),
                      signed(m.delta_rank_ic.iloc[0]) if len(m) else '—'))
    head_table=ui.html_table(['Benchmark','Model','Static Rank IC','Expanding Rank IC','Expanding − Static','Expanding LM vs A'],gains)
    periods={p:p for p in PERIODS}
    plan=ui.html_table(['Year','Cutoff date (16:00 ET)','Train n','Prior 2024','Prior 2025','Last mature target','Missing labels','Too late','Test n'],[
        (r.period,r.refit_cutoff_et[:10],r.training_events,r.train_2024,r.train_2025,r.latest_label_close_et[:10],r.missing_target,
         r.at_or_after_cutoff,r.matched_events) for r in pick(R,option='market').itertuples()])
    stab=ui.html_table(['Benchmark','Schedule','Model','Full quarters','IC SD','Positive IC share','ICIR'],[
        (OPTIONS[r.option],r.training,r.model,r.adequate_full_quarters,fmt(r.sd_ic),f'{r.positive_ic_share:.0%}',fmt(r.icir)) for r in S.itertuples()])
    hp=[]
    for o in OPTIONS:
        for m,s in spec['specifications'][o].items():
            params=s['parameters']; estimator=f"{s['estimator']} (α {params['alpha']:g})"
            if 'l1_ratio' in params:
                estimator=f"Elastic Net (α {params['alpha']:g}, L1 ratio {params['l1_ratio']:g})"
            hp.append((OPTIONS[o],m,estimator,len(s['columns'])))
    specs=ui.html_table(['Benchmark','Model','Frozen estimator','Predictors'],hp)
    fields=''.join(f'<p><strong>{label}</strong> <code>{escape(", ".join(names))}</code></p>' for label,names in [
        ('Controls',spec['controls']),('Market LM5',spec['specifications']['market']['B']['columns'][16:]),
        ('Healthcare LM5',spec['specifications']['sector']['B']['columns'][16:]),('LM9',spec['LM9'])])
    headers={n:n.replace('_',' ').title() for n in NAMES}
    links=''.join(f'<a href="{n}.csv">{label} CSV</a>' for n,label in headers.items())
    links+='<a href="expanding_experiment_spec.json">Frozen specification</a><a href="methodology.md">Methodology</a><a href="summary_zh.md">中文摘要</a>'
    payload=json.dumps({k:json.loads(v.to_json(orient='records',double_precision=15)) for k,v in D.items()},ensure_ascii=False).replace('<','\\u003c')
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><style>{CSS}</style></head>
<body><header><div class="wrap"><strong>{TITLE}</strong><span>{SUBTITLE}</span></div></header><nav><div class="wrap"><a href="#alpha">LM increments</a><a href="#compare">Static vs Expanding</a><a href="#quintiles">Quintiles</a><a href="#intervals">Uncertainty</a><a href="#methods">Methods &amp; data</a></div></nav>
<main class="wrap"><div class="hero"><div class="intro"><div class="kicker">Executive overview / Post-hoc exploratory</div><h1>Annual refits did not restore LM alpha</h1><p>Annual updates improved the Healthcare-adjusted baseline within each new year. The market-adjusted baseline weakened. All six pooled Rank ICs fell and LM increments remained small.</p></div>
<aside class="aside"><strong>Identical evaluation calls</strong><div class="tiles"><div class="tile"><b>{pooled.events}</b><span>Pooled OOS events</span></div><div class="tile"><b>{pooled.issuers}</b><span>Distinct OOS issuers</span></div></div><p><strong>Static</strong> 2010–23 throughout.<br><strong>Annual Expanding</strong> Through 2023, 2024 and 2025 for the 2024, 2025 and 2026 predictions, respectively.</p></aside></div>
<section aria-label="Pooled rank performance"><h2>Pooled 2025–26 headline results</h2>{head_table}<p class="sub">2024 informed original model selection. Both new years were previously examined in the Static baseline. CAR target window is +2 through +20 CRSP sessions.</p></section>
<section id="alpha"><div class="kicker">Primary comparison</div><h2>Incremental LM Rank IC across years</h2><p class="sub">B or B* minus A, within the same training schedule. Use zero as the financial baseline.</p>
<div class="controls">{select('alpha-bench','Benchmark',OPTIONS)}{select('alpha-model','Language model',{'B':'B · selected LM5','B*':'B* · full LM9'})}{select('alpha-training','Training schedule',{'Both':'Compare both',**{v:v for v in PANEL}})}</div>
{key(MODE)}<div class="grid">{figure('Incremental Rank IC line chart','alpha-plot','2024 historical diagnostic, 2025 full-year, 2026 partial-year, and pooled 2025–26.')}<div id="alpha-table"></div></div>
<p class="sub">Both bootstrap intervals include zero for each pooled LM increment. A better relative increment can coincide with worse total model performance.</p></section>
<section id="compare"><h2>Static vs Annual Expanding</h2><p class="sub">Compare each frozen model with itself across the two refit schedules. All bars use the same Rank IC scale.</p>
<div class="controls">{select('cmp-bench','Benchmark',OPTIONS)}{select('cmp-period','Evaluation period',periods,PERIODS[-1])}</div>{key(MODE)}
<div class="grid">{figure('Static and Expanding Rank IC','cmp-plot','Spearman correlation on matched events.')}<div><div id="cmp-table"></div><p class="sub">Pooled Rank IC is recomputed across raw event-level scores. Their scale can shift as the training windows grow.</p></div></div>
<details><summary>Quarterly stability and positive IC share</summary>{stab}<p class="sub">Five full quarters qualify under the frozen rule. The incomplete Q2 2026 appears in the quarterly CSV. Six full quarters are required for ICIR.</p></details></section>
<section id="quintiles"><h2>Prediction quintiles</h2><p class="sub">Sort by model scores. Q1 is the lowest group. Y-axis scales stay fixed across selections.</p>
<div class="controls">{select('q-bench','Benchmark',OPTIONS)}{select('q-period','Evaluation period',periods,PERIODS[-1])}{select('q-training','Training schedule',{v:v for v in PANEL},'Expanding')}</div>{key(COLOR)}
<div class="grid">{figure('Event quintile profiles','q-plot','Mean realized CAR by predicted-score quintile. Overlapping windows limit portfolio interpretation.')}<div id="q-table"></div></div>
<details><summary>All five groups with counts, medians and dispersion</summary><div id="q-detail"></div></details></section>
<section id="intervals"><h2>Paired statistical evidence</h2><p class="sub">Circles are issuer clusters. Squares and dashed lines are calendar-time blocks. All evidence is exploratory.</p>
<div class="controls">{select('ci-bench','Benchmark',OPTIONS)}{select('ci-period','Evaluation period',periods,PERIODS[-1])}{select('ci-family','Contrast',{'LM':'LM minus A','schedule':'Expanding minus Static'})}</div><p id="ci-count" class="sub"></p>
<div class="grid">{figure('Paired 95% bootstrap intervals','ci-plot','Zero is the reference for both families. Missing intervals indicate descriptive estimates.')}<div id="ci-table"></div></div>
<p class="sub">The pooled Holm adjustments treat the four LM contrasts and six schedule contrasts as two families. The three market schedule comparisons have pointwise intervals below zero, with adjusted p = 0.1128 for each. No pooled contrast passes the exploratory screen.</p></section>
<section id="methods"><h2>Data and methodology</h2><p class="sub">Same event IDs, targets and benchmarks as the committed Static baseline.</p>{plan}
<details><summary>Frozen model specifications and feature blocks</summary>{specs}{fields}<p>The archived LM dictionary makes <code>hedging_answers</code> and <code>uncertainty_answers</code> identical. Both are included for faithful replication.</p></details>
<details><summary>Label maturity and execution time</summary><p>A refit starts at the first matched event’s Day +1 close. It uses prior-year calls with a saved target and the Day +20 close strictly before the cutoff. Two legacy training labels retain 18 of 19 paired sessions under the original coverage convention. Their full windows elapsed. Early CAR is known after Day +1 close. This design does not establish real-time transcript or financial input availability.</p></details>
<details><summary>Data downloads and reproduction</summary><p class="download">{links}</p></details></section></main>
<footer class="wrap">HEALTH CARE / CAR[+2,+20] / CRSP THROUGH 30 JUN 2026</footer><script id="data" type="application/json">{payload}</script><script>{JS}</script></body></html>'''
    (folder/'dashboard.html').write_text(html,encoding='utf-8')


def make_zh(folder, D):
    G,I,B,R = [D[n] for n in ('static_to_expanding','incremental_comparison','bootstrap_inference','refit_training_audit')]
    names={'market':'标普 500 调整','sector':'医疗保健行业调整'}
    names_period=dict(zip(PERIODS,('2024 历史诊断','2025 全年','2026 部分年度','2025–26 合并')))
    out=['# Healthcare 语言因子年度扩展窗口评估','',
         '年度重拟合未恢复可靠的语言因子增量。医疗保健行业调整后，A 在两个新年度内有所改善，标普 500 调整后 A 在两个期间均转弱。合并样本中六个模型的 Rank IC 均低于静态版本，四项 LM 增量的两个配对区间均含零。','',
         '六个规格均成功运行，2024 历史诊断逐分数精确复现；两种训练日程及两种基准的事件 ID 均完全一致。','',
         '| 评估期间 | 重拟合截点（纽约 16:00） | 每个基准训练事件 | 评估事件 / 发行人 |','|---|---|---:|---:|']
    for r in pick(R,option='market').itertuples():
        out.append(f'| {names_period[r.period]} | {r.refit_cutoff_et[:10]} | {r.training_events:,} | {r.matched_events} / {r.prediction_issuers} |')
    out+=['','合并样本为 315 场、61 家发行人。2024 一场缺失原生标签的电话会议继续排除。所有使用的标签均在截点前成熟，2025 年最后一批训练标签最迟于 2026-01-06 可观测。','',
          '## Rank IC 和 LM 增量','',
          '| 基准 | 期间 | A 静态 → 扩展 | B 静态 → 扩展 | B* 静态 → 扩展 | 扩展 B−A | 扩展 B*−A |','|---|---|---:|---:|---:|---:|---:|']
    for o in OPTIONS:
        for p in PERIODS:
            g=pick(G,option=o,period=p).set_index('model'); i=pick(I,option=o,period=p,training='Expanding').set_index('model')
            values=[f'{fmt(g.loc[m,"rank_ic_static"])} → {fmt(g.loc[m,"rank_ic_expanding"])}' for m in MODELS]
            out.append('| '+' | '.join([names[o],names_period[p],*values,signed(i.loc['B','delta_rank_ic']),signed(i.loc['B*','delta_rank_ic'])])+' |')
    out+=['','## 合并样本配对 95% 区间','',
          '| 基准 | 比较 | Δ Rank IC | 发行人聚类 | 双月时间块 | Holm p |','|---|---|---:|---|---|---:|']
    for family in ('LM','schedule'):
        for (o,m),pair in pick(B,period=PERIODS[-1],family=family).groupby(['option','model'],sort=False):
            v=pair.set_index('method'); r=v.loc['issuer_cluster']
            out.append('| '+' | '.join([names[o],f'{m}−A' if family=='LM' else f'{m} 扩展−静态',signed(r.point),band(r),band(v.loc['time_block']),fmt(r.holm_p_pooled,4)])+' |')
    out+=['','标普调整下，三个更新效应的单项区间均低于零。六项更新效应经 Holm 校正后均未通过探索性筛查。2026 年仅有三个名义双月块，因此只作描述性比较。区间基于规定的两类重抽样，各有 2,499 次有效配对重抽。','',
          '合并 Rank IC 从全部事件的原始预测分数重算。无归一化的训练秩随训练样本扩展而改变尺度，可能影响跨年排序。按预测分数组成的五分位在合并样本中均未呈严格单调变化，不能解释为可交易组合收益。','',
          '2024 曾用于策略选择，2025–26 的静态结果也已检查。本研究是事后探索。Early CAR 要到 Day +1 收盘后才可获得，原流程未验证文本或全量财务特征当时是否可用。两列相同的旧版语言特征保持原样。','',
          '[交互式仪表板](dashboard.html) · [方法及复现](methodology.md) · [拟合日期及样本](refit_training_audit.csv) · [配对区间](bootstrap_inference.csv)']
    (folder/'summary_zh.md').write_text('\n'.join(out)+'\n',encoding='utf-8')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,default=HERE,help='Directory containing the saved expanding evaluation tables')
    folder=cli.parse_args().out.resolve()
    if folder == STATIC_RENDER.parent:
        raise ValueError('Choose the expanding output directory')
    data=load(folder)
    spec=json.loads((folder/'expanding_experiment_spec.json').read_text(encoding='utf-8'))['static_frozen_specification']
    make_pdf(folder,data,spec)
    make_html(folder,data,spec)
    make_zh(folder,data)


if __name__=='__main__':
    main()

