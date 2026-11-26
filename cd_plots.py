"""Charts for the CD client deck (EDA + train-only CV results). Usage: python cd_plots.py . cd_results/plots"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
wt, out = sys.argv[1], sys.argv[2]
fam = "Inter" if any("Inter" in f.name for f in fm.fontManager.ttflist) else "Helvetica"
INK, SOFT, MUTED, GRID, BG = "#14213D", "#4A5568", "#8A93A3", "#E4E2DC", "#F7F6F2"
BLUE, ORANGE, GRAY = "#2E6FD8", "#C8651B", "#7A8291"
plt.rcParams.update({"font.family": fam, "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": SOFT,
    "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.titlesize": 17,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left"})
def save(fig, n): fig.savefig(f"{out}/{n}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
R = json.load(open(f"{wt}/cd_results/cd_cv_protocol_results.json"))
O1, O2 = list(R)
t1 = pd.read_parquet(f"{wt}/cd_results/model_table_opt1.parquet"); t2 = pd.read_parquet(f"{wt}/cd_results/model_table_opt2.parquet")
for t in (t1, t2): t["call_date"] = pd.to_datetime(t["call_date"])
fb = pd.read_parquet(f"{wt}/cd_results/finbert_features.parquet")
mt = pd.read_csv(f"{wt}/model_table_with_linguistic.csv", usecols=["transcriptid", "call_date", "sector"]); mt["call_date"] = pd.to_datetime(mt.call_date)
col = mt[mt.sector == "Consumer Discretionary"]

# 1 calls per year: kept / added by PIT / dropped
yr = lambda d: d.call_date.dt.year
pit_ids, col_ids = set(t1.transcriptid), set(col.transcriptid)
kept = t1[t1.transcriptid.isin(col_ids)]; added = t1[~t1.transcriptid.isin(col_ids)]; dropped = col[~col.transcriptid.isin(pit_ids)]
years = range(2010, 2027)
k = yr(kept).value_counts().reindex(years, fill_value=0); a = yr(added).value_counts().reindex(years, fill_value=0)
d = yr(dropped).value_counts().reindex(years, fill_value=0)
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.axvspan(2023.5, 2024.5, color="#E9EEF8", zorder=0); ax.axvspan(2024.5, 2026.6, color="#ECEBE6", zorder=0)
ax.bar(years, k, color=BLUE, width=0.72, label="in both samples")
ax.bar(years, a, bottom=k, color=ORANGE, width=0.72, label="added by point-in-time GICS", edgecolor=BG, linewidth=2)
ax.bar(years, -d, color=GRAY, width=0.72, label="dropped (not CD at the time)")
ax.axhline(0, color=MUTED, lw=1)
ax.text(2024, 340, "test\n2024", ha="center", color=INK, fontsize=14, weight="semibold")
ax.text(2025.5, 340, "no target\n(CRSP ends)", ha="center", color=SOFT, fontsize=13)
ax.set_ylabel("earnings calls"); ax.set_xticks(list(years)); ax.set_xticklabels([str(y)[2:] for y in years])
ax.set_ylim(-40, 400); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=14, bbox_to_anchor=(0, 1.08)); save(fig, "eda_calls")

# 2 target distribution (train) both benchmarks
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, t, name in [(axs[0], t1, "Option 1: vs S&P 500"), (axs[1], t2, "Option 2: vs CD sector index")]:
    y = t.loc[t.car_2_20.notna(), "car_2_20"] * 100
    ax.hist(y.clip(-30, 30), bins=60, color=BLUE, edgecolor=BG, linewidth=1)
    ax.axvline(y.median(), color=INK, lw=2)
    ax.text(0.98, 0.92, f"n = {len(y):,}\nmedian {y.median():+.2f}%\nsd {y.std():.1f}%", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=14)
    ax.set_title(name); ax.set_xlabel("CAR[+2,+20], % (clipped at ±30)"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("calls"); save(fig, "eda_target")

# 3 train-only univariate rank correlation, Option 2
feats = {"Controls": ["sue", "surprise_pct", "car_-1_1", "mom", "vol_pre", "bm", "size", "dispersion", "n_analysts", "words_answers"],
         "LM dictionary": ["tone_prepared", "tone_answers", "tone_gap", "tone_answers_chg", "hedging_answers", "uncertainty_answers"],
         "FinBERT": ["fb_sent_prepared", "fb_sent_answers", "fb_neg_pct_prepared", "fb_sent_gap", "fb_sent_answers_chg", "fb_n_sent_answers"]}
tr = t2.merge(fb, on="transcriptid", how="left"); tr = tr[(tr.call_date <= "2023-11-26") & tr.car_2_20.notna()]
rows = [(g, f, stats.spearmanr(tr[f], tr.car_2_20, nan_policy="omit")[0]) for g, fs in feats.items() for f in fs if f in tr]
fig, ax = plt.subplots(figsize=(13, 8.4)); ypos, labels, yv = [], [], 0
gcol = {"Controls": GRAY, "LM dictionary": BLUE, "FinBERT": ORANGE}
for g in feats:
    for gg, f, r in rows:
        if gg != g: continue
        ax.barh(yv, r, color=gcol[g], height=0.7); labels.append(f); ypos.append(yv); yv -= 1
    yv -= 0.6
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=13); ax.axvline(0, color=MUTED, lw=1)
se = 1.96 / np.sqrt(len(tr)); ax.axvspan(-se, se, color="#ECEBE6", zorder=0)
ax.set_xlabel(f"Spearman correlation with CAR[+2,+20] vs CD index, train calls (n = {len(tr):,}); shaded = ±1.96/√n")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=g) for g, c in gcol.items()], frameon=False, loc="upper right", fontsize=14, bbox_to_anchor=(1.02, 1.02))
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); save(fig, "eda_features")

# 4 holdout Rank IC with CI, both options
models = ["A", "B", "B*", "C", "C*"]; mcol = {"A": GRAY, "B": BLUE, "B*": BLUE, "C": ORANGE, "C*": ORANGE}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, o, name in [(axs[0], O1, "Option 1: vs S&P 500"), (axs[1], O2, "Option 2: vs CD index")]:
    for i, m in enumerate(models):
        h = R[o][m]["holdout"]; y = -i
        ax.plot(h["ci95"], [y, y], color=mcol[m], lw=3, solid_capstyle="round")
        ax.plot(h["rank_ic"], y, "o", ms=13, color=mcol[m], mfc=BG if "*" in m else mcol[m], mew=3)
        ax.text(h["ci95"][1] + 0.012, y, f"{h['rank_ic']:+.3f}", va="center", color=INK, fontsize=14)
    ax.axvline(0, color=MUTED, lw=1); ax.set_title(name); ax.set_xlim(-0.33, 0.33)
    ax.set_xlabel("holdout Rank IC (2024), 95% firm-bootstrap CI"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
axs[0].set_yticks([-i for i in range(5)]); axs[0].set_yticklabels(["A  controls", "B  CV-best LM", "B*  all LM", "C  CV-best FinBERT", "C*  all FinBERT"])
save(fig, "res_rankic")

# 5 delta vs A CI
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, o, name in [(axs[0], O1, "Option 1: vs S&P 500"), (axs[1], O2, "Option 2: vs CD index")]:
    for i, m in enumerate(models[1:]):
        h = R[o][m]["holdout"]; y = -i; dlt = h["rank_ic"] - R[o]["A"]["holdout"]["rank_ic"]; lo, hi = h["delta_vs_A_ci95"]
        ax.plot([lo, hi], [y, y], color=mcol[m], lw=3, solid_capstyle="round")
        ax.plot(dlt, y, "o", ms=13, color=mcol[m], mfc=BG if "*" in m else mcol[m], mew=3)
        ax.text(hi + 0.01, y, f"{dlt:+.3f}" + ("  CI > 0" if lo > 0 else ""), va="center", color=INK, fontsize=14)
    ax.axvline(0, color=INK, lw=1.5); ax.set_title(name); ax.set_xlim(-0.17, 0.3)
    ax.set_xlabel("Rank IC minus Model A, 95% CI"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
axs[0].set_yticks([0, -1, -2, -3]); axs[0].set_yticklabels(["B  CV-best LM", "B*  all LM", "C  CV-best FinBERT", "C*  all FinBERT"])
save(fig, "res_delta")

# 6 CV vs holdout for every candidate
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, o, name in [(axs[0], O1, "Option 1"), (axs[1], O2, "Option 2")]:
    for g, c in [("B", BLUE), ("C", ORANGE)]:
        cs = R[o][g]["candidates"]
        ax.scatter([x["cv"] for x in cs], [x["holdout"]["rank_ic"] for x in cs], s=90, color=c, edgecolor=BG, lw=2, label=f"{g} candidates")
        w = cs[0]; ax.scatter(w["cv"], w["holdout"]["rank_ic"], s=330, facecolor="none", edgecolor=INK, lw=2.2)
        if o == O2 and g == "C":
            bh = max(cs, key=lambda x: x["holdout"]["rank_ic"])
            ax.annotate("FinBERT + LM: best on holdout,\n3rd of 4 on CV", (bh["cv"], bh["holdout"]["rank_ic"]), xytext=(12, -6), textcoords="offset points", color=SOFT, fontsize=13, va="top")
    r = np.corrcoef([x["cv"] for g in "BC" for x in R[o][g]["candidates"]], [x["holdout"]["rank_ic"] for g in "BC" for x in R[o][g]["candidates"]])[0, 1]
    ax.set_title(f"{name}: corr(CV, holdout) = {r:+.2f}"); ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("train-only CV Rank IC (used to choose)"); ax.grid(color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("holdout Rank IC (never used)")
axs[1].scatter([], [], s=330, facecolor="none", edgecolor=INK, lw=2.2, label="selected on CV")
axs[1].legend(frameon=False, loc="lower right", fontsize=13); save(fig, "res_cv_vs_holdout")

# 7 protocol comparison: Δ vs A, holdout-picked vs CV-picked
old = {O1: {"B": 0.0442, "B*": 0.0400, "C": 0.0709}, O2: {"B": 0.0908, "B*": 0.0835, "C": 0.0653}}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, o, name in [(axs[0], O1, "Option 1: vs S&P 500"), (axs[1], O2, "Option 2: vs CD index")]:
    x = np.arange(3); new = [R[o][m]["holdout"]["rank_ic"] - R[o]["A"]["holdout"]["rank_ic"] for m in ["B", "B*", "C"]]
    ov = [old[o][m] for m in ["B", "B*", "C"]]
    ax.bar(x - 0.19, ov, 0.36, color=GRAY, label="holdout-picked (IT protocol)")
    ax.bar(x + 0.19, new, 0.36, color=BLUE, label="train-CV-picked (final)")
    for xi, v in zip(x - 0.19, ov): ax.text(xi, v + (0.004 if v >= 0 else -0.012), f"{v:+.3f}", ha="center", fontsize=13, color=INK)
    for xi, v in zip(x + 0.19, new): ax.text(xi, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=13, color=INK)
    ax.set_ylim(-0.055, 0.105)
    if o == O2: ax.axhline(0.0287, color=ORANGE, lw=2, ls="--", label="placebo, Option 2: random FinBERT labels (+0.029)")
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(x); ax.set_xticklabels(["B", "B*", "best C / C"]); ax.set_title(name)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("Rank IC gain over Model A")
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
axs[0].legend(h0 + h1[:1], l0 + l1[:1], frameon=False, loc="lower left", fontsize=13); save(fig, "res_protocols")

# 8 CV fold stability for selected models
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, o, name in [(axs[0], O1, "Option 1"), (axs[1], O2, "Option 2")]:
    for m, c, mk in [("A", GRAY, "o"), ("B", BLUE, "s"), ("C", ORANGE, "D")]:
        cand = next(x for x in R[o][m]["candidates"] if x["name"] == R[o][m]["selected"]) if m != "A" else R[o]["A"]["candidates"][0]
        f = cand["cv_folds"]; ax.plot([2021, 2022, 2023], f, marker=mk, ms=10, lw=2, color=c, label=m)
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks([2021, 2022, 2023]); ax.set_title(f"{name}: validation-year Rank IC"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("Rank IC"); axs[0].legend(frameon=False, loc="upper right", fontsize=13); save(fig, "res_folds")
print("ok", fam)
