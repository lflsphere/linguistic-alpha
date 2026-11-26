"""Reproduce the frozen Healthcare fits and score the new periods without selection."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "outputs"
ARCHIVE = OUTPUTS / "healthcare_original_pierre_replication"
NATIVE = ARCHIVE / "earnings_health"
ORIGINAL = ROOT / "experiments/legacy_pipeline"
RAW = ROOT / "data/raw"
CIZ = ROOT / "data/team_pit/sp500_pit"
EXTENDED = OUTPUTS / "sp500_financial_extension/sp500_model_extended.csv"
CAR = OUTPUTS / "sp500_car_extension/sp500_car_extended.csv"
INPUT = ROOT / "model_table_with_linguistic.csv"
TARGET = "car_2_20"
EARLY = "car_-1_1"
MODELS = ("A", "B", "B*")
OPTIONS = ("market", "sector")
PERIODS = ("2024 diagnostic", "2025", "2026 partial", "2025–26 pooled")
N_BOOT = 2499
BLOCK = 2
SEED = 14483
IDS = ["keydevid", "transcriptid", "permno", "gvkey", "call_date", "call_time_utc"]
PROJECTION = ["keydevid", "transcriptid", "permno", "gvkey", "call_date", "call_time_utc", "day0"]

sys.path[:0] = [str(ORIGINAL), str(ORIGINAL / "src"),
                str(OUTPUTS / "sp500_car_extension"), str(ARCHIVE)]
from linguistic_alpha.events.timing import assign_day0, call_timestamp_utc
from extend_car import read_text_table, verify_mapping
from run_original import company_snapshot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_hash(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def check_inputs():
    expected = json.loads((ARCHIVE / "provenance.json").read_text(encoding="utf-8"))
    require(expected["revision"].startswith("06273f1"), "Unexpected native source version")
    for name, digest in expected["sha256"].items():
        require(path_hash(ORIGINAL / name) == digest, f"Native checksum changed: {name}")
    require(path_hash(INPUT) == "833d76f72ed6ce22674ea551ff60e135360d62d76806a0dfbc7f7681eb7c3ffb",
            "Source archive digest changed")
    financial = pd.read_csv(OUTPUTS / "sp500_financial_extension/feature_validation.csv")
    require(financial.approved_for_reconstruction.all(), "Some financial input definitions did not pass")
    # The financial extension may change only verified future financial fields.
    fields = set(financial.feature)
    for count, (old, new) in enumerate(zip_longest(csv_rows(CAR), csv_rows(EXTENDED)), start=1):
        require(old is not None and new is not None and list(old) == list(new), "Extension header/row mismatch")
        allowed = fields if old["call_date"] >= "2025-01-01" else set()
        require(all(new[k] == x for k, x in old.items() if k not in allowed), f"Protected cell changed on row {count}")
    require(count == 29698, "Wrong model-table size")
    return expected


def freeze(out, native_module, provenance):
    f = out / "frozen_experiment_spec.json"
    require(not f.exists(), "Use a fresh destination")
    native = pd.read_csv(ARCHIVE / "native_model_metrics.csv")
    args = {"Ridge": {"alpha": 1.0}, "Ridge_a10": {"alpha": 10.0}, "Ridge_a50": {"alpha": 50.0},
            "EN_strong": {"alpha": .05, "l1_ratio": .7, "max_iter": 5000}}
    expected = ["Ridge", "Ridge", "Ridge_a10", "Ridge", "EN_strong", "Ridge_a50"]
    require(native.algorithm.tolist() == expected, "Original selected estimators changed")
    spec = {}
    for row in native.itertuples(index=False):
        name = OPTIONS[row.option - 1]
        features = row.features.split(",")
        controls = native_module.CONTROL_CANDIDATES
        require(features[:len(controls)] == controls, "Native controls have changed")
        if row.model == "B*":
            require(features[len(controls):] == native_module.LINGUISTIC_FEATURES, "Native LM9 changed")
        spec.setdefault(name, {})[row.model] = {
            "algorithm": row.algorithm, "strategy": row.strategy, "columns": features,
            "estimator": "ElasticNet" if row.algorithm == "EN_strong" else "Ridge", "parameters": args[row.algorithm],
            "resolved_parameters": estimator(native_module, row.algorithm).get_params(),
        }
    result = {
        "source_revision": provenance["revision"],
        "source_model": "experiments/legacy_pipeline/sector_model_b.py",
        "model_sha256": provenance["sha256"]["sector_model_b.py"],
        "input_sha256": path_hash(EXTENDED),
        "evaluation_script_sha256": path_hash(__file__),
        "native_predictions_sha256": path_hash(ARCHIVE / "original_test_predictions.parquet"),
        "specifications": spec, "controls": native_module.CONTROL_CANDIDATES,
        "LM9": native_module.LINGUISTIC_FEATURES, "log_then_preprocess": native_module.LOG_TRANSFORM,
        "preprocessing": "Original function; log1p(max(x,0)) on listed fields, development 1st/99th percentile winsorization, raw development median for missing, StandardScaler (ddof=0) fit on development only.",
        "target": "sum of daily abnormal returns over trading offsets +2 to +20 inclusive",
        "learning_target": "scipy.stats.rankdata(method=average), 1-based midranks, unnormalized, development CAR only",
        "benchmarks": {
            "market": "S&P 500 SPRTRN price-return benchmark. Use already validated extended labels.",
            "sector": "Original static GICS 35 snapshot and any-CCM link basket; session-valid S&P 500 membership; same-day abs(price)*shares, cap-weighted daily returns; CIZ from 2025. Legacy and CIZ 2024 benchmark bridge gates below.",
        },
        "sector_validation": "Legacy 2024 daily and event labels within 1e-10; CIZ 2024 overlap at least 95% of archived events within 5 bp and all within 25 bp, a documented source-vintage guard. Early CAR is also validated on legacy 2024.",
        "development": "Archived native option-specific rows, 2010–2023, nonmissing target. Fit all frozen models on identical 2,939 rows per option, no retuning or new development exclusion.",
        "diagnostic": "Reuse archived 2024 scores/labels on 204 events per option; verify reproduction by fixed refit. 2024 informed historical estimator and strategy selection.",
        "new_oos": "Existing static Health Care archive events, 2025 calendar year or partial 2026 through 2026-06-30 last source session. Verify exact CIQ/security linkage within observed window and active CIZ S&P membership on the archived call date; require complete target window and valid benchmark label. Apply identical inclusion across models. No transcript clock gate. Original supplied prior_calls_with_target retained; it does not accrue new OOS outcomes.",
        "missingness": "All selected features stay in original order, with original development-statistic imputation; don't discard events for missing features or score/return magnitude. Keep LM redundancy. Sector Early CAR uses sector benchmark, with missing values imputed as originally.",
        "metrics": "Spearman IC, paired B/B*-A, supplementary Pearson, score-sort spreads and change in spread; prediction rank agreement and changed rank share. No raw-scale R2, MAE or RMSE, and no directional hit rate without a frozen sign calibration.",
        "quintiles": "Within option and period, pandas.qcut of predictions only; five distinct groups, no tie-splitting; require 50 events and 20 issuers. Report mean, median, sample SD, n. Monotonic when four successive changes in means share the same direction.",
        "inference": f"{N_BOOT} paired draws per method, seed {SEED} adjusted by period/option. Resample all calls of drawn GVKEY clusters or overlapping noncircular {BLOCK}-calendar-month blocks in call months (include empty months), recompute midranks per draw. Pointwise percentile 95% CIs; at least 30 issuers and five nominal disjoint blocks for dual-method inference and >=95% usable draws. Partial 2026 and accessed 2024 descriptive. Centered two-sided bootstrap p-values with +1 correction; take larger across methods, Holm across exactly four pooled tests; require Holm p<=.05 and both CIs exclude zero in same direction.",
        "stability": "Calendar quarters require 30 events and 15 issuers; incomplete Q2 2026 excluded from stability mean/SD/positive share. Report mean/SD/sign share at >=3 full periods; unannualized ICIR requires >=6.",
    }
    with f.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def estimator(native, algorithm):
    if algorithm == "Ridge_a50":
        return native.Ridge(alpha=50.0)
    return native.get_algos()[algorithm]


def observed_frame():
    dtype = {"gvkey": "string", "gind": "string"}
    table = pd.read_csv(EXTENDED, dtype=dtype, low_memory=False, float_precision="round_trip")
    table["gvkey"] = table.gvkey.str.zfill(6)
    table["call_date"] = pd.to_datetime(table.call_date)
    require(table.keydevid.is_unique and table.transcriptid.is_unique, "Duplicate input identities")
    ref = pd.read_csv(INPUT, dtype=dtype, float_precision="round_trip", low_memory=False)
    ref["gvkey"] = ref.gvkey.str.zfill(6)
    require(table.keydevid.equals(ref.keydevid), "Model archive ordering changed")
    for name in ["sector", "gind", "gvkey", "permno", "transcriptid", "call_time_utc", "day0"]:
        require(table[name].equals(ref[name]), f"Archive {name} changed")
    for name in ["car_0_1", "car_2_5", "car_2_20", "car_2_60"]:
        use = ref[name].notna()
        require(ref.loc[use, name].equals(table.loc[use, name]), f"Archived {name} changed")
    out = table.loc[table.sector.eq("Health Care")].copy().reset_index(drop=True)
    out["clock"] = call_timestamp_utc(out.call_date, out.call_time_utc)
    out["year"] = out.call_date.dt.year
    both = out.hedging_answers.notna() & out.uncertainty_answers.notna()
    require(out.hedging_answers[both].equals(out.uncertainty_answers[both]), "Archived LM redundancy changed")
    return out, ref


def membership(folder, modern):
    if modern:
        file = folder / "crsp_msp500list_v2.parquet"
        d = pd.read_parquet(file, columns=["permno", "mbrstartdt", "mbrenddt"]).rename(
            columns={"mbrstartdt": "start", "mbrenddt": "ending"})
    else:
        d = pd.read_parquet(folder / "crsp_msp500list.parquet", columns=["permno", "start", "ending"])
    for c in ["start", "ending"]:
        d[c] = pd.to_datetime(d[c])
    return d


def stocks(path, modern, securities):
    fields = ({"dlycaldt": "date", "dlyret": "ret", "dlyprc": "prc", "shrout": "shrout", "dlydelflg": "delist"}
              if modern else {"date": "date", "ret": "ret", "prc": "prc", "shrout": "shrout"})
    frame = pd.read_parquet(path, columns=["permno", *fields]).rename(columns=fields)
    frame = frame.loc[frame.permno.isin(securities)].copy()
    frame["date"] = pd.to_datetime(frame.date)
    for field in ("ret", "prc", "shrout"):
        frame[field] = frame[field].astype(float)
    require(not frame.duplicated(["permno", "date"]).any(), "Duplicate stock sessions")
    return frame


def aggregate_sector(stocks, spells, calendar, year, vintage):
    merged = stocks.merge(spells, on="permno", validate="many_to_many")
    d = merged.loc[merged.date.between(merged.start, merged.ending)].drop_duplicates(["permno", "date"]).copy()
    d["cap"] = d.prc.abs() * d.shrout
    known = d.loc[d.cap.gt(0) & d.ret.notna()].copy()
    known["weighted"] = known.cap * known.ret
    total = known.groupby("date").agg(weighted=("weighted", "sum"), cap=("cap", "sum"),
                                      weighted_members=("permno", "size")).reindex(calendar)
    total["sector_return"] = total.weighted / total.cap
    relevant = spells.loc[spells.ending.ge(calendar[0]) & spells.start.le(calendar[-1]), ["permno", "start", "ending"]]
    total["eligible_members"] = [relevant.loc[relevant.start.le(day) & relevant.ending.ge(day), "permno"].nunique()
                                 for day in calendar]
    return total[["sector_return", "weighted_members", "eligible_members"]].rename_axis("date").reset_index().assign(
        year=year, vintage=vintage)


def calculate(e, calendar, daily, benchmark, low, high, early=False):
    day0 = assign_day0(e.clock, calendar, e.call_date).day0
    by = {key: v.set_index("date") for key, v in daily.groupby("permno")}
    vals, status = [], []
    for row, day in zip(e.itertuples(), day0):
        i = calendar.get_indexer([day])[0]
        if i < 0 or i + low < 0 or i + high >= len(calendar):
            vals.append(np.nan)
            status.append("outside_complete_calendar")
            continue
        days = calendar[i+low:i+high+1]
        piece = by.get(row.permno)
        if piece is None:
            vals.append(np.nan)
            status.append("security_returns_unavailable")
            continue
        piece = piece.reindex(days)
        r = piece.ret.to_numpy(float)
        b = benchmark.reindex(days).to_numpy(float)
        if not (np.isfinite(r).all() and np.isfinite(b).all()):
            vals.append(np.nan)
            status.append("incomplete_paired_sessions")
        elif early and "delist" in piece and piece.delist.eq("Y").any():
            vals.append(np.nan)
            status.append("early_delisting_convention_unverified")
        else:
            vals.append(float(np.sum(r-b)))
            status.append("complete")
    return pd.DataFrame({"day0": day0, "value": vals, "status": status}, index=e.index)


def val_row(name, a, b, cutoff, gate):
    a, b = np.asarray(a, float), np.asarray(b, float)
    paired = np.isfinite(a) & np.isfinite(b)
    d = abs(a[paired]-b[paired])
    return dict(check=name, reference_n=int(np.isfinite(a).sum()), compared_n=len(d),
                mean_abs_error=d.mean(), max_abs_error=d.max(), fraction_within_cutoff=np.mean(d <= cutoff),
                tolerance=cutoff, passed=bool(gate(d, len(d), int(np.isfinite(a).sum()))))


def sector_extension(health, reference, out):
    comp = company_snapshot(reference, RAW)
    gv = set(comp.loc[comp.gsector.eq("35"), "gvkey"])
    old_mem, recent_mem = membership(RAW, False), membership(CIZ, True)
    def basket(folder):
        c = pd.read_parquet(folder / "crsp_ccmxpf_lnkhist.parquet", columns=["gvkey", "permno"])
        return set(c.loc[c.gvkey.astype("string").str.zfill(6).isin(gv), "permno"])
    old_basket, modern_basket = basket(RAW), basket(CIZ)
    members = modern_basket | old_basket
    all_relevant = members | set(health.permno)
    old_cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(RAW / "crsp_dsi.parquet", columns=["date"]).date))
    new_cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(CIZ / "crsp_dsp500_v2.parquet", columns=["caldt"]).caldt))
    require(old_cal.isin(new_cal).all() and new_cal[-1] == pd.Timestamp("2026-06-30"), "Calendar mismatch")
    legacy = stocks(RAW / "crsp_dsf_2024.parquet", False, all_relevant)
    sector_old = aggregate_sector(legacy.loc[legacy.permno.isin(old_basket)], old_mem,
                                 old_cal[old_cal.year == 2024], 2024, "legacy_rebuilt")
    modern = {}
    daily = {}
    for year in [2024, 2025, 2026]:
        d = stocks(CIZ / f"crsp_dsf_v2_{year}.parquet", True, all_relevant)
        require(d.date.max() == new_cal[new_cal.year == year][-1], f"Missing CIZ end date {year}")
        modern[year] = d
        daily[year] = aggregate_sector(d.loc[d.permno.isin(members)], recent_mem,
                                       new_cal[new_cal.year == year], year, "ciz")
    archived_daily = pd.read_parquet(NATIVE / "health_sector_daily.parquet")
    archived_daily["date"] = pd.to_datetime(archived_daily.date)
    archived_daily = archived_daily.set_index("date").sector_ret
    old = sector_old.set_index("date").sector_return
    exact = lambda d, n, refn: n == refn and len(d) > 0 and d.max() <= 1e-10
    rows = [val_row("2024 legacy daily sector benchmark", archived_daily.reindex(old.index), old, 1e-10, exact)]
    native = pd.read_parquet(NATIVE / "model_table_opt2.parquet", columns=[*PROJECTION, EARLY, TARGET])
    native["call_date"] = pd.to_datetime(native.call_date)
    check = health.loc[health.year.eq(2024)].merge(native, on=IDS, validate="one_to_one", suffixes=("", "_native"))
    sector_2024 = daily[2024].set_index("date").sector_return
    for col, lo, hi in [(EARLY, -1, 1), (TARGET, 2, 20)]:
        value = calculate(check, old_cal[old_cal.year == 2024], legacy, old, lo, hi)
        require(value.day0.equals(pd.to_datetime(check.day0_native)), "2024 native Day 0 mismatch")
        rows.append(val_row(f"2024 legacy event {col}", check[col+"_native"], value.value, 1e-10, exact))
        bridge = calculate(check, new_cal[new_cal.year == 2024], modern[2024], sector_2024, lo, hi,
                           early=col == EARLY)
        rows.append(val_row(f"2024 CIZ bridge event {col}", check[col+"_native"], bridge.value, .0005,
                            lambda d, n, refn: n == refn and np.mean(d <= .0005) >= .95 and d.max() <= .0025))
    validation = pd.DataFrame(rows)
    validation.to_csv(out / "sector_benchmark_validation.csv", index=False, float_format="%.14g")
    ok = validation.passed.all()
    health_new = health.loc[health.year.ge(2025)]
    if not ok:
        return None, recent_mem
    period_sector = pd.concat([sector_old, daily[2025], daily[2026]], ignore_index=True)
    period_sector.to_csv(out / "sector_benchmark_daily.csv", index=False, float_format="%.14g")
    benchmark = period_sector.set_index("date").sector_return
    calendar = new_cal[new_cal.year >= 2024]
    # Use the archived vintage on every 2024 session needed by a cross-year Early window.
    combined = pd.concat([legacy, modern[2025], modern[2026]], ignore_index=True)
    early = calculate(health_new, calendar, combined, benchmark, -1, 1, early=True)
    target = calculate(health_new, calendar, combined, benchmark, 2, 20)
    ext = health_new[PROJECTION[:-1]].copy()
    ext["day0"] = target.day0
    ext["sector_early_car"] = early.value
    ext["sector_car_2_20"] = target.value
    ext["early_status"] = early.status
    ext["target_status"] = target.status
    ext.to_csv(out / "healthcare_sector_extended_car.csv", index=False, float_format="%.17g")
    return ext, recent_mem


def eligible_new(health, sector, members):
    allrows = health.loc[health.year.ge(2025)].copy().reset_index(drop=True)
    require(allrows.clock.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            .equals(allrows.call_date), "New Healthcare Day 0 month/date convention diverges")
    m, _, _ = verify_mapping(allrows[["keydevid", "transcriptid", "permno", "gvkey", "call_date", "clock"]])
    spells = allrows[["transcriptid", "permno", "call_date"]].merge(members, on="permno", validate="many_to_many")
    active = spells.loc[spells.call_date.between(spells.start, spells.ending), "transcriptid"]
    allrows["verified"] = m
    allrows["member"] = allrows.transcriptid.isin(active)
    allrows["sector_target"] = (
        allrows.transcriptid.map(sector.set_index("transcriptid").sector_car_2_20) if sector is not None else np.nan)
    require(allrows.loc[allrows[TARGET].notna(), "car_2_20_source"].eq("extended_crsp_ciz").all(),
            "Unexpected vintage for new market CAR")
    return allrows


def fits(out, native, frozen, allrows, sector):
    old = pd.read_parquet(ARCHIVE / "original_test_predictions.parquet")
    old["call_date"] = pd.to_datetime(old.call_date)
    old["gvkey"] = old.gvkey.astype("string").str.zfill(6)
    predictions, coverage, feature_coverage, verification = [], [], [], []
    for i, option in enumerate(OPTIONS, 1):
        if option == "sector" and sector is None:
            continue
        archive = pd.read_parquet(NATIVE / f"model_table_opt{i}.parquet")
        archive["call_date"] = pd.to_datetime(archive.call_date)
        archive["gvkey"] = archive.gvkey.astype("string").str.zfill(6)
        train = archive.loc[(archive.call_date < "2024-01-01") & archive[TARGET].notna()]
        diagnostic = archive.loc[archive.call_date.dt.year.eq(2024) & archive[TARGET].notna()]
        require(len(train) == 2939 and len(diagnostic) == 204, "Native split changed")
        new = allrows.copy()
        if option == "sector":
            new[TARGET] = new.sector_target
            new[EARLY] = new.transcriptid.map(sector.set_index("transcriptid").sector_early_car)
            new["day0"] = new.transcriptid.map(sector.set_index("transcriptid").day0)
        else:
            cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(
                CIZ / "crsp_dsp500_v2.parquet", columns=["caldt"]).caldt))
            new["day0"] = assign_day0(new.clock, cal, new.call_date).day0
        valid = new.loc[new.verified & new.member & new[TARGET].notna()].copy()
        require(valid.day0.notna().all() and valid.transcriptid.is_unique, "Final matched window invalid")
        for label, src, used in [
            ("Development", archive.loc[archive.call_date.lt("2024-01-01")], train),
            (PERIODS[0], archive.loc[archive.call_date.dt.year.eq(2024)], diagnostic),
            (PERIODS[1], new.loc[new.year.eq(2025)], valid.loc[valid.year.eq(2025)]),
            (PERIODS[2], new.loc[new.year.eq(2026)], valid.loc[valid.year.eq(2026)]),
            (PERIODS[3], new, valid)
        ]:
            is_new = "verified" in src
            coverage.append(dict(option=option, period=label, archived_calls=len(src),
                                 valid_target=int(src[TARGET].count()),
                                 excluded_target_rows_mapping=int((src[TARGET].notna() & ~src.verified).sum()) if is_new else 0,
                                 excluded_target_rows_membership=int((src[TARGET].notna() & src.verified & ~src.member).sum()) if is_new else 0,
                                 matched_events=len(used), issuers=used.gvkey.nunique(),
                                 first_call=used.call_date.min().date(), last_call=used.call_date.max().date()))
            for feature in frozen["controls"] + frozen["LM9"]:
                n = used[feature].notna().sum()
                feature_coverage.append(dict(option=option, period=label, feature=feature, events=len(used),
                                             observed=n, missing=len(used)-n, observed_share=n/len(used) if len(used) else np.nan))
        for label, spec in frozen["specifications"][option].items():
            predictors = spec["columns"]
            require(all(train[f].count() > 0 for f in predictors), "Unidentified legacy imputation")
            x, xold, _, _ = native.preprocess(train, diagnostic, predictors)
            fit = estimator(native, spec["algorithm"]).fit(x, stats.rankdata(train[TARGET], method="average"))
            saved = old.loc[old.option.eq(i) & old.model.eq(label), ["transcriptid", "prediction", TARGET]]
            pairs = diagnostic[["transcriptid", TARGET]].assign(replay=fit.predict(xold)).merge(
                saved, on="transcriptid", validate="one_to_one", suffixes=("", "_saved"))
            err = abs(pairs.replay-pairs.prediction).max()
            require(len(pairs) == 204 and pairs[TARGET].equals(pairs[TARGET+"_saved"]) and err < 1e-8,
                    f"Frozen refit fails archived prediction replay: {option} / {label} / {err}")
            verification.append(dict(option=option, model=label, n=len(pairs), max_abs_score_difference=err,
                                     fit_n=len(train), parameters=spec["algorithm"]))
            h = diagnostic[PROJECTION+[TARGET]].merge(saved[["transcriptid", "prediction"]],
                                                     on="transcriptid", validate="one_to_one")
            predictions.append(h.assign(option=option, model=label, period=PERIODS[0], source="saved_diagnostic"))
            # Re-run the same original preprocessing with development data only.
            replay_x, xv, _, _ = native.preprocess(train, valid, predictors)
            require(np.array_equal(x, replay_x), "Evaluation features changed development preprocessing")
            n = valid[PROJECTION+[TARGET]].assign(prediction=fit.predict(xv), option=option, model=label,
                                                 source="frozen_development_refit")
            n["period"] = np.where(n.call_date.dt.year.eq(2025), PERIODS[1], PERIODS[2])
            predictions.append(n)
    pd.DataFrame(verification).to_csv(out / "fit_replay_validation.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "target_coverage.csv", index=False)
    pd.DataFrame(feature_coverage).to_csv(out / "feature_coverage.csv", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    require(not pred.duplicated(["option", "period", "model", "transcriptid"]).any(), "Duplicate predictions")
    pred.to_csv(out / "matched_predictions_local.csv", index=False, float_format="%.17g")
    return pred


def pdata(pred):
    for option, d in pred.groupby("option", sort=False):
        for period in PERIODS:
            small = d.loc[d.period.eq(period)] if period != PERIODS[-1] else d.loc[d.period.ne(PERIODS[0])]
            index = ["transcriptid", "call_date", "gvkey", TARGET]
            wide = small.pivot(index=index, columns="model", values="prediction").reset_index()
            require(wide[list(MODELS)].notna().all(axis=None), "Matched-event requirement violated")
            yield option, period, wide.sort_values(["call_date", "transcriptid"]).reset_index(drop=True)


def ic_matrix(data):
    r = stats.rankdata(data, axis=0, method="average")
    return np.corrcoef(r, rowvar=False)[-1, :3]


def sort_groups(pred, actual):
    codes = pd.qcut(pred, 5, labels=False, duplicates="drop")
    if pd.isna(codes).any() or np.unique(codes).size != 5:
        return None
    return [actual[codes == j] for j in range(5)]


def statistics(pred, out):
    metrics, quintiles, increments, quarters, stability = [], [], [], [], []
    for option, period, wide in pdata(pred):
        arr = wide[[*MODELS, TARGET]].to_numpy()
        ic = ic_matrix(arr)
        spread = {}
        issuers = wide.gvkey.nunique()
        for j, model in enumerate(MODELS):
            groups = sort_groups(arr[:, j], arr[:, 3]) if len(wide) >= 50 and issuers >= 20 else None
            means = np.array([x.mean() for x in groups]) if groups else np.full(5, np.nan)
            changes = np.diff(means)
            profile = ("increasing" if (changes > 0).all() else "decreasing" if (changes < 0).all()
                       else "mixed" if groups else "insufficient distinct groups or events")
            spread[model] = means[-1] - means[0]
            for k in range(5):
                x = groups[k] if groups else np.array([])
                quintiles.append(dict(option=option, period=period, model=model, group=k+1, n=len(x),
                                      mean_car=means[k], median_car=np.median(x) if len(x) else np.nan,
                                      sd_car=x.std(ddof=1) if len(x) > 1 else np.nan, profile=profile))
            metrics.append(dict(option=option, period=period, model=model, events=len(wide), issuers=issuers,
                                rank_ic=ic[j], pearson=stats.pearsonr(arr[:, j], arr[:, 3]).statistic,
                                q5_q1=spread[model], profile=profile))
        for j, model in enumerate(MODELS[1:], 1):
            a, b = stats.rankdata(arr[:, 0]), stats.rankdata(arr[:, j])
            increments.append(dict(option=option, period=period, model=model, events=len(wide), issuers=issuers,
                                   delta_rank_ic=ic[j]-ic[0], delta_spread=spread[model]-spread["A"],
                                   prediction_rank_agreement=stats.spearmanr(a, b).statistic,
                                   changed_rank_share=np.mean(a != b)))
        if period != PERIODS[-1]:
            continue
        clock_quarter = wide.call_date.dt.to_period("Q")
        for year in (2025, 2026):
            for q in range(1, (3 if year == 2026 else 5)):
                label = f"{year}Q{q}"
                g = wide.loc[clock_quarter.astype(str).eq(label)]
                adequate = len(g) >= 30 and g.gvkey.nunique() >= 15
                z = ic_matrix(g[[*MODELS, TARGET]].to_numpy()) if adequate else [np.nan]*3
                for j, model in enumerate(MODELS):
                    quarters.append(dict(option=option, quarter=label, model=model, events=len(g),
                                         issuers=g.gvkey.nunique(), rank_ic=z[j], partial=label == "2026Q2"))
        for model in MODELS:
            q = np.array([r["rank_ic"] for r in quarters if r["option"] == option and r["model"] == model and
                          not r["partial"] and np.isfinite(r["rank_ic"])])
            sd = q.std(ddof=1) if len(q) >= 3 else np.nan
            stability.append(dict(option=option, model=model, adequate_full_quarters=len(q),
                                  mean_ic=q.mean() if len(q) >= 3 else np.nan, sd_ic=sd,
                                  positive_ic_share=np.mean(q>0) if len(q) >= 3 else np.nan,
                                  icir=q.mean()/sd if len(q) >= 6 and sd > 0 else np.nan))
    tables = {"model_performance": metrics, "incremental_comparison": increments, "quintile_returns": quintiles,
              "quarterly_ic": quarters, "stability_summary": stability}
    for file, data in tables.items():
        pd.DataFrame(data).to_csv(out / f"{file}.csv", index=False, float_format="%.14g")


def bootstrap(pred, out):
    output = []
    for option, period, wide in pdata(pred):
        matrix = wide[[*MODELS, TARGET]].to_numpy()
        obs = ic_matrix(matrix)
        last = pd.Period("2026-06" if period in PERIODS[2:] else f"{period[:4]}-12", freq="M")
        start = pd.Period("2025-01" if period == PERIODS[-1] else f"{period[:4]}-01", freq="M")
        grid = pd.period_range(start, last, freq="M")
        groups = wide.groupby("gvkey").indices
        months = wide.groupby(wide.call_date.dt.to_period("M")).indices
        month_grp = [months.get(p, np.array([], dtype=int)) for p in grid]
        sufficient = len(groups) >= 30 and len(grid)//BLOCK >= 5 and period != PERIODS[0]
        for method in ("issuer_cluster", "time_block"):
            n_blocks = int(np.ceil(len(grid)/BLOCK))
            rng = np.random.default_rng(SEED+11*OPTIONS.index(option)+PERIODS.index(period))
            idx_groups = list(groups.values())
            draws = []
            if sufficient:
                for _ in range(N_BOOT):
                    if method == "issuer_cluster":
                        indices = np.concatenate([idx_groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
                    else:
                        beginnings = rng.integers(0, len(grid)-BLOCK+1, size=n_blocks)
                        slots = (beginnings[:, None]+np.arange(BLOCK)).ravel()[:len(grid)]
                        indices = np.concatenate([month_grp[i] for i in slots])
                    if len(indices) < 30:
                        continue
                    v = ic_matrix(matrix[indices])
                    if np.isfinite(v).all():
                        draws.append(v[1:]-v[0])
            for j, model in enumerate(MODELS[1:]):
                estimate = obs[j+1]-obs[0]
                usable = len(draws) >= .95*N_BOOT
                simulated = np.asarray(draws)[:, j] if draws else np.array([])
                lo, hi = np.quantile(simulated, [.025,.975]) if usable else [np.nan,np.nan]
                pval = ((1+(abs(simulated-estimate) >= abs(estimate)).sum())/(len(simulated)+1)
                        if usable and period == PERIODS[-1] else np.nan)
                output.append(dict(option=option, period=period, model=model, method=method,
                                   events=len(wide), issuer_clusters=len(groups), calendar_months=len(grid),
                                   nominal_time_blocks=len(grid)//BLOCK, n_requested=N_BOOT if sufficient else 0,
                                   n_valid=len(draws), point=estimate, lower=lo, upper=hi,
                                   crosses_zero=bool(lo <= 0 <= hi) if usable else None,
                                   p_centered=pval, status="pointwise interval" if usable else "descriptive only"))
    inf = pd.DataFrame(output)
    pooled = inf.loc[inf.period.eq(PERIODS[-1])]
    grouped = pooled.groupby(["option", "model"])
    p = grouped.p_centered.max()
    if len(p) == 4 and p.notna().all():
        ordered = p.sort_values()
        adjusted = (ordered*np.arange(4,0,-1)).cummax().clip(upper=1)
    else:
        adjusted = p*0+np.nan
    inf["holm_p_pooled"] = np.nan
    inf["passes_confirmatory_screen"] = False
    for (option, model), correction in adjusted.items():
        mask = inf.option.eq(option) & inf.model.eq(model) & inf.period.eq(PERIODS[-1])
        v = inf.loc[mask]
        same_sign = v.lower.gt(0).all() or v.upper.lt(0).all()
        inf.loc[mask, "holm_p_pooled"] = correction
        inf.loc[mask, "passes_confirmatory_screen"] = same_sign and correction <= .05
    inf.to_csv(out / "bootstrap_inference.csv", index=False, float_format="%.14g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = check_inputs()
    native = module(ORIGINAL / "sector_model_b.py", "original_oos_models")
    frozen = freeze(out, native, provenance)
    print("Frozen all six specifications before evaluation.", flush=True)
    health, base = observed_frame()
    sector, members = sector_extension(health, base, out)
    print("Sector validation:", sector is not None, flush=True)
    samples = eligible_new(health, sector, members)
    pred = fits(out, native, frozen, samples, sector)
    statistics(pred, out)
    bootstrap(pred, out)
    require(path_hash(EXTENDED) == frozen["input_sha256"], "Protected extended source changed")
    print(pd.read_csv(out / "model_performance.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
