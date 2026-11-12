"""
Linguistic Alpha from Earnings Calls: shared configuration and utilities.

Run order (from the project root; data/panel.csv is the only data input):
  python scripts/01_build_features.py      parse transcripts -> targets, Model A controls, linguistic features
  python scripts/02_validate.py            Step 1: transcript EDA, measure validation, labeling sample, leakage checks
  python scripts/03_cross_section.py       Steps 2-3: in-sample significance, Rank IC, sorts, FDR, alpha decay
  python scripts/04_oos.py --freeze        Step 4: purged expanding-window OOS, Model A vs B, freeze the spec
  python scripts/05_backtest_holdout.py    Section 7 on dev OOS predictions; --stage holdout for the one-shot 2024 test

Panel facts that drive design choices (02_validate prints the evidence):
  * 3 firms (AAPL, MSFT, NVDA), 179 calls, 2010-2024, about 12 events per year and 3 per calendar quarter.
    A per-quarter cross-sectional IC has N=3 and firm-clustered SEs have G=3, so neither is usable:
    IC is computed on annual pooled cross-sections and SEs are clustered by calendar quarter of day 0.
  * One GICS sub-industry per firm, so industry FE = firm FE. Default FE: firm + year.
  * CAR windows in the panel start at +2, so CAR[0,+1] never enters a target.
  * Price per share is not in the panel, so the price-scaled SUE of the framework is unavailable (see SUE_VARIANT).
  * Consensus EPS is split-adjusted and rounded to the cent (I/B/E/S adjusted file), which destroys
    surprise precision for heavily split stocks (NVDA before ~2020). Flagged as LOWPREC.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================ paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PANEL_PATH = DATA_DIR / "panel.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
PAIRS_PATH = DATA_DIR / "qa_pairs.csv"
META_PATH = DATA_DIR / "features_meta.json"
LABEL_SAMPLE_PATH = DATA_DIR / "qa_label_sample.csv"   # written by 02_validate (blind, no model scores)
LABELS_PATH = DATA_DIR / "qa_labels.csv"               # the same file once the team has filled the labels
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "outputs"
FROZEN_SPEC_PATH = OUT_DIR / "frozen_spec.json"

# ============================================================================ sample and splits (Sections 2.1, 6)
DEV_START, DEV_END = 2010, 2022
HOLDOUT_START, HOLDOUT_END = 2023, 2024  # CRSP target coverage ends 2024-12-31
FIRST_TEST_YEAR = 2016               # expanding development folds begin after six years
HIST_WINDOW = 4                      # prior calls used by the Delta features
SUE_HIST = (8, 4)                    # (window, min periods) of the time-series SUE scale

# ============================================================================ targets (Section 2.2)
# CAR_a_b is the market-adjusted CAR over trading days [+a, +b] relative to day 0.
TARGETS = ["CAR_2_5", "CAR_2_20", "CAR_2_60", "CAR_6_20", "CAR_21_60", "LOGVOL_2_20"]
CUM_TARGETS = ["CAR_2_5", "CAR_2_20", "CAR_2_60"]
BUCKET_TARGETS = ["CAR_2_5", "CAR_6_20", "CAR_21_60"]          # non-overlapping decay buckets
RETURN_TARGETS = ["CAR_2_5", "CAR_2_20", "CAR_2_60", "CAR_6_20", "CAR_21_60"]
TARGET_END_DAY = {"CAR_2_5": 5, "CAR_2_20": 20, "CAR_2_60": 60,
                  "CAR_6_20": 20, "CAR_21_60": 60, "LOGVOL_2_20": 20}
PRIMARY_TARGET = "CAR_2_20"          # to be confirmed with the client
OOS_TARGETS = ["CAR_2_5", "CAR_2_20", "CAR_2_60", "LOGVOL_2_20"]

# ============================================================================ Model A (Section 3)
# SUE_TS  : surprise / SD of the firm's prior 8 surprises (point-in-time, split-invariant)   [default]
# SUE_PCT : surprise / |consensus|, clipped to [-1, 1]
# SUE_IBES: surprise / SD of analyst forecasts, as delivered in the panel (missing when SD rounds to 0)
SUE_VARIANT = "SUE_IBES"             # scope definition: EPS surprise / pre-Day-0 price
MODEL_A = ["SUE", "DISP", "COV", "SIZE", "BM", "MOM", "VOL"]
DELAYED_EXTRA = ["AR01"]             # CAR[0,+1]: only in the delayed-signal robustness spec

# ============================================================================ Model B (Section 4)
LING_FAMILIES = {
    "communication_state": ["TONE_P", "TONE_Q", "TONE_GAP", "UNC", "HEDGE"],
    "interaction_quality": ["EVADE", "NONANS"],
    "consistency_deviation": ["dTONE", "dUNC", "dEVADE", "dTONE_P", "dTONE_Q", "dTONE_GAP", "NOVEL", "DIV_QP"],
}
LING_ALL = [f for fam in LING_FAMILIES.values() for f in fam]
# Full-rank subset for joint OLS: TONE_GAP = TONE_Q - TONE_P, and EVADE = 1 - REL.
LING_CORE = ["TONE_P", "TONE_GAP", "UNC", "HEDGE", "EVADE", "NONANS", "dTONE", "NOVEL"]
FAMILY_OF = {f: fam for fam, fs in LING_FAMILIES.items() for f in fs}

# ============================================================================ estimation defaults
FE_DEFAULT = ("firm", "year")
CLUSTER_DEFAULT = "event_q"          # calendar quarter of day 0
WINSOR = (0.01, 0.99)
N_QUANTILES = 3                      # ~50 events per bucket in the dev sample
PIT_MIN_HIST = 12                    # prior events needed before a point-in-time bucket is assigned
BOOT_REPS = 1000
SEED = 7
ST_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FINBERT_MODEL = "ProsusAI/finbert"
LSA_DIM = 150
COST_BPS = {"stock": 5.0, "hedge": 1.0}   # one-way cost per side, in bps of notional

ID_COLS = ["transcriptid", "ticker", "firm", "call_date", "day0", "year", "event_q"]


# ============================================================================ I/O helpers
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


def stage_dir(stage: str) -> Path:
    d = OUT_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta() -> dict:
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


def model_a_cols(delayed: bool = False) -> list[str]:
    return MODEL_A + (DELAYED_EXTRA if delayed else [])


def load_events(scope: str = "dev", sample: str = "all") -> pd.DataFrame:
    """Event table. scope in {dev, holdout, dev+holdout, all}; sample in {all, highprec}."""
    if not FEATURES_PATH.exists():
        raise SystemExit("data/features.csv not found: run scripts/01_build_features.py first")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["call_date", "day0"])
    df["SUE"] = df[SUE_VARIANT]
    if scope == "dev":
        df = df[(df["year"] >= DEV_START) & (df["year"] <= DEV_END)]
    elif scope == "holdout":
        df = df[df["year"].between(HOLDOUT_START, HOLDOUT_END)]
    elif scope == "dev+holdout":
        df = df[df["year"].between(DEV_START, HOLDOUT_END)]
    elif scope != "all":
        raise ValueError(scope)
    if sample == "highprec":
        df = df[~df["LOWPREC"].astype(bool)]
    return df.sort_values(["day0", "ticker"]).reset_index(drop=True)


def assert_no_holdout(df: pd.DataFrame) -> None:
    if (df["year"] >= HOLDOUT_START).any():
        raise RuntimeError("Holdout events reached a development-stage computation.")


def label_end(df: pd.DataFrame, target: str) -> pd.Series:
    """Approximate date on which the target becomes observable (business days, holidays ignored)."""
    return df["day0"] + pd.offsets.BDay(TARGET_END_DAY[target])


def purged_train(df: pd.DataFrame, target: str, test_start: pd.Timestamp, first_year: int = DEV_START):
    """Expanding training set whose labels were fully observed before the first test event."""
    return df[(df["year"] >= first_year) & (label_end(df, target) < test_start)]


# ============================================================================ preprocessing
class Preprocessor:
    """Winsorize -> median impute -> z-score; every moment is estimated on the fitting sample only."""

    def __init__(self, cols: list[str], winsor: tuple[float, float] = WINSOR):
        self.cols, self.winsor = list(cols), winsor

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = df[self.cols].astype(float)
        self.lo, self.hi = X.quantile(self.winsor[0]), X.quantile(self.winsor[1])
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        Xf = Xc.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].astype(float).clip(self.lo, self.hi, axis=1).fillna(self.med)
        return ((X - self.mu) / self.sd).fillna(0.0)


def fe_dummies(df: pd.DataFrame, fe, categories: dict | None = None) -> pd.DataFrame:
    parts = []
    for f in fe:
        cats = (categories or {}).get(f) or sorted(df[f].astype(str).unique())
        d = pd.get_dummies(pd.Categorical(df[f].astype(str), categories=cats), prefix=f, dtype=float)
        parts.append(d.iloc[:, 1:])
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(df)))
    out.index = df.index
    return out


# ============================================================================ estimation
def ols_fit(y: pd.Series, X: pd.DataFrame, groups=None):
    """OLS with cluster-robust (groups given) or HC3 standard errors."""
    import statsmodels.api as sm
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.OLS(np.asarray(y, float), Xc)
    if groups is None:
        return model.fit(cov_type="HC3")
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pd.Series(groups))[0]})


def joint_wald(res, names: list[str]) -> tuple[float, float]:
    """Wald F-test that the listed coefficients are jointly zero, using the fitted covariance."""
    params = list(res.params.index)
    R = np.zeros((len(names), len(params)))
    for i, n in enumerate(names):
        R[i, params.index(n)] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)
    return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))


def spearman(a, b, min_n: int = 5) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(stats.rankdata(a[m]), stats.rankdata(b[m]))[0, 1])


def ic_series(df: pd.DataFrame, sig: str, y: str, by: str = "year", min_n: int = 6) -> pd.Series:
    return pd.Series({k: spearman(g[sig], g[y], min_n) for k, g in df.groupby(by)}, dtype=float)


def ic_stats(s: pd.Series) -> dict:
    s = pd.Series(s, dtype=float).dropna()
    n = len(s)
    if n < 2:
        return dict(ic_mean=s.mean() if n else np.nan, ic_sd=np.nan, icir=np.nan, ic_t=np.nan, ic_hit=np.nan, n_periods=n)
    sd = s.std(ddof=1)
    return dict(ic_mean=s.mean(), ic_sd=sd, icir=s.mean() / sd if sd > 0 else np.nan,
                ic_t=s.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan, ic_hit=(s > 0).mean(), n_periods=n)


def block_indices(df: pd.DataFrame, block: str = CLUSTER_DEFAULT, reps: int = BOOT_REPS,
                  seed: int = SEED) -> list[np.ndarray]:
    """Positional row indices for a block bootstrap: whole blocks (default calendar quarters) are
    resampled with replacement, which preserves within-quarter dependence of overlapping CAR windows."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(df[block].to_numpy())[0]
    groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    return [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(reps)]


def bootstrap(fn, draws: list[np.ndarray]) -> np.ndarray:
    """Evaluate fn(idx) for each resampled index array (fn closes over numpy arrays)."""
    return np.array([fn(i) for i in draws], dtype=float)


def boot_p_two_sided(draws: np.ndarray) -> float:
    d = draws[np.isfinite(draws)]
    if len(d) == 0:
        return np.nan
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def nw_se_mean(x, lags: int | None = None) -> float:
    """Newey-West standard error of a sample mean (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (NaNs are ignored and returned as NaN)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[m] = out
    return q


def clark_west(y, f_small, f_big, lags: int = 2) -> tuple[float, float]:
    """Clark-West (2007) MSPE-adjusted test, H1: the larger nested model forecasts better (one-sided)."""
    y, f1, f2 = (np.asarray(v, float) for v in (y, f_small, f_big))
    m = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[m], f1[m], f2[m]
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    se = nw_se_mean(adj, lags)
    if not np.isfinite(se) or se == 0:
        return np.nan, np.nan
    t = adj.mean() / se
    return float(t), float(1 - stats.norm.cdf(t))


def oos_r2(y, f, bench) -> float:
    y, f, b = (np.asarray(v, float) for v in (y, f, bench))
    m = np.isfinite(y) & np.isfinite(f) & np.isfinite(b)
    den = ((y[m] - b[m]) ** 2).sum()
    return float(1 - ((y[m] - f[m]) ** 2).sum() / den) if den > 0 else np.nan


def pit_bucket(dates, values, n_q: int = N_QUANTILES, min_hist: int = PIT_MIN_HIST) -> np.ndarray:
    """Point-in-time quantile: rank each event against signals of strictly earlier events only.
    (Ranking within a calendar quarter would use calls that happen weeks later.)"""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        hist = v[(d < d[i]) & np.isfinite(v)]
        if len(hist) < min_hist:
            continue
        pct = (hist < v[i]).mean() + 0.5 * (hist == v[i]).mean()
        out[i] = min(int(pct * n_q), n_q - 1) + 1
    return out


# ============================================================================ reporting
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = True) -> str:
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)

    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
