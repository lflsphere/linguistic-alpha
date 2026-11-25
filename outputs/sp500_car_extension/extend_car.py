"""Validate and extend the archived S&P 500 CARs without changing any source files."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from linguistic_alpha.events.timing import NY, assign_day0, call_timestamp_utc

HERE = Path(__file__).resolve().parent
INPUT = ROOT / "model_table_with_linguistic.csv"
RAW = ROOT / "data/raw"
TEAM = ROOT / "data/team_pit/sp500_pit"
ARCHIVE = ROOT / "outputs/healthcare_original_pierre_replication/earnings_health"
EARLY = "car_-1_1"
WINDOWS = {"car_0_1": (0, 1), "car_2_5": (2, 5), "car_2_20": (2, 20),
           "car_2_60": (2, 60), EARLY: (-1, 1)}
OLD = list(WINDOWS)[:-1]
IDENTITY = ["transcriptid", "permno", "call_date", "call_time_utc"]
REASONS = ["unverified_event_security_link", "no_day0_in_calendar",
           "window_outside_calendar", "missing_benchmark", "missing_stock_return",
           "early_ciz_delisting_semantics_unverified"]
LEGACY_TOL = 1e-7
CIZ_TOL = 1e-5
ARCHIVED_REVISION_DAY = pd.Timestamp("2024-10-15")
ARCHIVED_REVISION_SECURITY = 81774
FILES = ["sp500_car_extended.csv", "car_validation.csv", "coverage_summary.csv",
         "methodology.md", "summary_zh.md"]


def fail_unless(condition, message):
    if not condition:
        raise ValueError(message)


def dates(frame, *columns):
    for name in columns:
        frame[name] = pd.to_datetime(frame[name], errors="raise").dt.normalize()
    return frame


def numeric(frame, *columns):
    for name in columns:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def read_text_table():
    with INPUT.open(encoding="utf-8", newline="") as f:
        records = csv.reader(f)
        header = next(records)
        rows = list(records)
    fail_unless(len(header) == len(set(header)) and all(len(row) == len(header) for row in rows),
                "The archived CSV has malformed rows or duplicate columns")
    fail_unless(not (set(header) & set(export_columns())), "Archive already contains extension columns")
    names = ["keydevid", "gvkey", "day0", "sector", *IDENTITY, *OLD]
    cols = [header.index(name) for name in names]
    e = pd.DataFrame([[r[c] for c in cols] for r in rows], columns=names)
    numeric(e, "keydevid", "gvkey", "transcriptid", "permno", *OLD)
    dates(e, "call_date", "day0")
    for key in ["keydevid", "transcriptid"]:
        fail_unless(e[key].notna().all() and e[key].is_unique, f"Missing or duplicate archive {key}")
    fail_unless(e[["permno", "gvkey", "call_date"]].notna().all(axis=None), "Missing archived identity")
    fail_unless(all(np.isfinite(e[h].dropna()).all() for h in OLD), "Nonfinite archived label")
    e["clock"] = call_timestamp_utc(e.call_date, e.call_time_utc)
    fail_unless(e.clock.notna().all(), "An archived call timestamp cannot be parsed")
    return header, rows, e


def link_ids(events, spells, keys, start, end):
    numeric(spells, *keys)
    dates(spells, start, end)
    dtype = {key: "int64" for key in keys}
    left = events[["transcriptid", "call_date", *keys]].dropna(subset=keys).astype(dtype)
    right = spells[[*keys, start, end]].dropna(subset=keys).drop_duplicates().astype(dtype)
    joined = left.merge(right, on=keys, validate="many_to_many")
    valid = ((joined[start].isna() | joined.call_date.ge(joined[start]))
             & (joined[end].isna() | joined.call_date.le(joined[end])))
    return joined.loc[valid, "transcriptid"].unique()


def verify_mapping(events):
    keys = ["transcriptid", "clock"]
    detail = pd.read_parquet(RAW / "ciq_transcript_detail.parquet", columns=[
        "transcriptid", "companyid", "mostimportantdateutc", "mostimportanttimeutc"])
    detail["clock"] = call_timestamp_utc(detail.mostimportantdateutc, detail.mostimportanttimeutc)
    detail = detail[keys + ["companyid"]].dropna().drop_duplicates()
    ambiguous = detail.duplicated(keys, keep=False)
    linked = events.merge(detail.loc[~ambiguous], on=keys, how="left", validate="one_to_one")
    fail_unless(linked.transcriptid.equals(events.transcriptid), "CIQ merge changed row order")
    ciq_ok, ccm_ok = set(), set()
    for folder in [RAW, TEAM]:
        ciq_ok.update(link_ids(linked, pd.read_parquet(folder / "ciq_wrds_gvkey.parquet"),
                               ["companyid", "gvkey"], "startdate", "enddate"))
        ccm = pd.read_parquet(folder / "crsp_ccmxpf_lnkhist.parquet")
        ccm = ccm.loc[ccm.linktype.isin(["LC", "LU"]) & ccm.linkprim.isin(["P", "C"])].copy()
        ccm_ok.update(link_ids(linked, ccm, ["gvkey", "permno"], "linkdt", "linkenddt"))
    members = pd.read_parquet(TEAM / "crsp_msp500list_v2.parquet")
    member = events.transcriptid.isin(link_ids(events, members, ["permno"], "mbrstartdt", "mbrenddt"))
    member_applicable = events.call_date.between("2025-01-01", pd.to_datetime(members.mbrenddt).max())
    mapped = (linked.companyid.notna() & linked.transcriptid.isin(ciq_ok & ccm_ok)).to_numpy()
    return mapped, int((member_applicable & ~member).sum()), int(member_applicable.sum())


def benchmark(path, date):
    s = dates(numeric(pd.read_parquet(path, columns=[date, "sprtrn"]), "sprtrn"), date)
    fail_unless(s[date].notna().all() and s[date].is_unique, f"Invalid session dates in {path}")
    return s.set_index(date).sprtrn.sort_index()


@dataclass
class Returns:
    calendar: pd.DatetimeIndex
    securities: pd.Index
    legacy_bench: pd.Series
    ciz_bench: pd.Series
    benchmark: np.ndarray
    stock: np.ndarray
    early_stock: np.ndarray
    ciz: np.ndarray
    ciz_delisted: np.ndarray
    legacy_count: int
    inventory: list[tuple[str, int, str, str]]


def day_string(value):
    return "" if pd.isna(value) else str(pd.Timestamp(value).date())


def load_returns(events):
    legacy = benchmark(RAW / "crsp_dsi.parquet", "date")
    ciz_bench = benchmark(TEAM / "crsp_dsp500_v2.parquet", "caldt")
    overlap = legacy.index.intersection(ciz_bench.index)
    fail_unless(overlap.equals(legacy.index), "CIZ and legacy calendars disagree in the overlap")
    same_year = ciz_bench.loc[ciz_bench.index.year == 2024].index
    fail_unless(same_year.equals(legacy.loc[legacy.index.year == 2024].index), "2024 calendars differ")
    calendar = legacy.index.union(ciz_bench.index[ciz_bench.index > legacy.index[-1]])
    n_legacy = len(legacy)
    p = pd.Index(sorted(events.permno.unique()))
    shape = (len(p), len(calendar))
    ordinary = np.full(shape, np.nan)
    modern = np.full(shape, np.nan)
    delisted = np.zeros(shape, dtype=bool)
    inventory = [(str((TEAM / "crsp_dsp500_v2.parquet").relative_to(ROOT)), len(ciz_bench),
                  day_string(ciz_bench.index[0]), day_string(ciz_bench.index[-1]))]

    def load(path, date, ret, grid, flag=None):
        fields = ["permno", date, ret] + ([flag] if flag else [])
        schema = pq.ParquetFile(path).schema_arrow
        fail_unless(set(fields) <= set(schema.names), f"Missing input fields in {path}")
        f = dates(numeric(pd.read_parquet(path, columns=fields), "permno", ret), date)
        fail_unless(not f[["permno", date]].isna().any(axis=None)
                    and not f.duplicated(["permno", date]).any(), f"Invalid stock-day keys in {path}")
        r, t = p.get_indexer(f.permno), calendar.get_indexer(f[date])
        fail_unless((t >= 0).all(), f"Stock sessions outside the benchmark calendar in {path}")
        use = r >= 0
        x = f[ret].to_numpy(dtype=float, na_value=np.nan)
        fail_unless(not np.isinf(x).any() and not (x < -1).any(), f"Invalid stock return in {path}")
        grid[r[use], t[use]] = x[use]
        if flag:
            fail_unless(f[flag].isin(["Y", "N"]).all(), f"Unexpected CIZ delisting flags in {path}")
            delisted[r[use], t[use]] = f.loc[use, flag].eq("Y")
        return (path.relative_to(ROOT).as_posix(), len(f), day_string(f[date].min()), day_string(f[date].max()))

    for year in range(2009, legacy.index[-1].year + 1):
        load(RAW / f"crsp_dsf_{year}.parquet", "date", "ret", ordinary)
    for year in (2024, 2025, 2026):
        inventory.append(load(TEAM / f"crsp_dsf_v2_{year}.parquet", "dlycaldt", "dlyret",
                              modern, "dlydelflg"))
    fail_unless(inventory[-1][-1] == day_string(calendar[-1]), "CIZ equity and benchmark end dates differ")
    stock = ordinary.copy()
    dl = dates(numeric(pd.read_parquet(RAW / "crsp_dsedelist.parquet", columns=[
        "permno", "dlstdt", "dlret"]), "permno", "dlret"), "dlstdt")
    dl = dl.dropna(subset=["dlret", "dlstdt"]).copy()
    r, t = p.get_indexer(dl.permno), legacy.index.searchsorted(dl.dlstdt)
    use = (r >= 0) & (t < n_legacy)
    r, t = r[use], t[use]
    fail_unless(not pd.MultiIndex.from_arrays([r, t]).has_duplicates, "Duplicate legacy delist stock-day")
    previous = stock[r, t]
    # The legacy delisting-only observation is the recorded DLRET itself.
    valid_dl = dl.loc[use, "dlret"].to_numpy(dtype=float)
    stock[r, t] = np.where(np.isnan(previous), valid_dl, (1 + previous) * (1 + valid_dl) - 1)
    stock[:, n_legacy:] = modern[:, n_legacy:]
    ordinary[:, n_legacy:] = modern[:, n_legacy:]
    bench = pd.concat([legacy, ciz_bench.loc[ciz_bench.index > legacy.index[-1]]]).reindex(calendar)
    return Returns(calendar, p, legacy, ciz_bench, bench.to_numpy(dtype=float), stock, ordinary,
                   modern, delisted, n_legacy, inventory)


def locate_day0(events, calendar, legacy):
    if not legacy:
        value = assign_day0(events.clock, calendar, events.call_date).day0
        return calendar.get_indexer(value)
    utc_date = events.call_date.to_numpy(dtype="datetime64[ns]")
    after_close = events.clock.dt.tz_convert(NY).dt.hour.ge(16).to_numpy()
    i = calendar.searchsorted(utc_date)
    same_day = (i < len(calendar)) & (calendar.take(i.clip(0, len(calendar) - 1)).to_numpy() == utc_date)
    i = i + (same_day & after_close)
    return np.where(i < len(calendar), i, -1)


@dataclass
class Window:
    value: np.ndarray
    end: np.ndarray
    reason: np.ndarray


def measure(security, day0, low, high, stock, benchmark, applicable=None, exclude=None):
    end = day0 + high
    reason = np.full(len(day0), "", dtype=object)
    if applicable is not None:
        reason[~applicable] = REASONS[0]
    reason[(reason == "") & (day0 < 0)] = REASONS[1]
    reason[(reason == "") & ((day0 + low < 0) | (end >= len(benchmark)))] = REASONS[2]
    use = np.flatnonzero(reason == "")
    fail_unless((security[use] >= 0).all(), "A measured event is missing a security row")
    positions = day0[use, None] + np.arange(low, high + 1)
    b = benchmark[positions]
    r = stock[security[use, None], positions]
    reason[use[~np.isfinite(b).all(axis=1)]] = REASONS[3]
    reason[use[(reason[use] == "") & ~np.isfinite(r).all(axis=1)]] = REASONS[4]
    if exclude is not None:
        unsafe = exclude[security[use, None], positions].any(axis=1)
        reason[use[(reason[use] == "") & unsafe]] = REASONS[5]
    value = np.full(len(day0), np.nan)
    ready = reason[use] == ""
    value[use[ready]] = (r[ready] - b[ready]).sum(axis=1)
    fail_unless(np.array_equal(np.isfinite(value), reason == ""), "Window/provenance mismatch")
    return Window(value, end, reason)


def archive_match(events, path):
    native = pd.read_parquet(path, columns=[*IDENTITY, "day0", EARLY])
    dates(native, "call_date", "day0")
    fail_unless(not native.duplicated(IDENTITY).any(), f"Duplicate Early CAR archive keys in {path}")
    joined = events.rename_axis("event_row").reset_index()[["event_row", *IDENTITY]].merge(
        native, on=IDENTITY, how="inner", validate="one_to_one")
    fail_unless(len(joined) == len(native), f"Missing exact event matches for {path}")
    return joined


def summarize_validation(label, reference, scope, match, events, observed, calculated, passed, note=""):
    eligible = np.isfinite(observed) & np.isfinite(calculated)
    fail_unless(eligible.any(), f"No numerical validation observations for {label} / {reference}")
    delta = calculated[eligible] - observed[eligible]
    err = np.abs(delta)
    return {"label": label, "check": reference, "scope": scope, "matched_events": len(match),
            "with_saved_label": int(np.isfinite(observed).sum()), "compared_events": len(delta),
            "compared_securities": events.iloc[match[eligible]].permno.nunique(),
            "first_event_date": day_string(events.iloc[match[eligible]].call_date.min()),
            "last_event_date": day_string(events.iloc[match[eligible]].call_date.max()),
            "mean_signed_difference": delta.mean(), "mean_abs_difference": err.mean(),
            "median_abs_difference": np.median(err), "max_abs_difference": err.max(),
            "agreement_at_1e_8": np.mean(err <= 1e-8), "agreement_at_1e_7": np.mean(err <= LEGACY_TOL),
            "agreement_at_1e_6": np.mean(err <= 1e-6), "agreement_at_1e_5": np.mean(err <= CIZ_TOL),
            "above_1e_5": int(np.sum(err > CIZ_TOL)), "gate_passed": bool(passed), "note": note}


def validate(events, data, sid, old_day0, early_day0):
    rows = []
    before_end = data.calendar[:data.legacy_count]
    archived_days = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    valid = old_day0 >= 0
    archived_days.loc[valid] = before_end.take(old_day0[valid])
    fail_unless(archived_days.equals(events.day0), "Archived Day 0 differs from legacy stage_targets")
    in_2024 = events.call_date.dt.year.eq(2024).to_numpy()
    selected = np.flatnonzero(in_2024)
    ciz_bench = data.ciz_bench.reindex(data.calendar).to_numpy(dtype=float)
    legacy_bench = data.benchmark[:data.legacy_count]
    legacy_s = data.stock[:, :data.legacy_count]
    legacy_early = data.early_stock[:, :data.legacy_count]
    for name in OLD:
        low, high = WINDOWS[name]
        saved = events[name].to_numpy()[selected]
        native = measure(sid[selected], old_day0[selected], low, high, legacy_s, legacy_bench).value
        expected = np.isfinite(saved)
        exact = np.isfinite(native)
        same_missing = np.array_equal(exact, expected)
        agrees = np.all(np.abs(native[expected] - saved[expected]) <= LEGACY_TOL)
        rows.append(summarize_validation(name, "legacy_vs_saved", "all_2024", selected, events,
                                         saved, native, same_missing and agrees,
                                         f"complete-window availability mismatch count={int((exact != expected).sum())}"))
        modern = measure(sid[selected], old_day0[selected], low, high, data.ciz, ciz_bench).value
        finite = expected & np.isfinite(modern)
        extreme = np.flatnonzero(finite & (np.abs(modern - saved) > CIZ_TOL))
        note = "Same Day 0 and full date windows; compares the CIZ vintage directly to the archive."
        resolved = len(extreme) == 0
        if len(extreme) == 1 and name == "car_2_60":
            k = extreme[0]
            i = selected[k]
            pos = data.calendar.get_indexer([ARCHIVED_REVISION_DAY])[0]
            first = old_day0[i] + low
            last = old_day0[i] + high
            if events.permno.iloc[i] == ARCHIVED_REVISION_SECURITY and first <= pos <= last:
                delta = data.ciz[sid[i], pos] - data.stock[sid[i], pos]
                residual = modern[k] - saved[k] - delta
                resolved = (np.isclose(delta, .001535, atol=LEGACY_TOL, rtol=0)
                            and abs(residual) <= CIZ_TOL and not data.ciz_delisted[sid[i], pos])
                note += (" One CIZ outlier was reconciled to a daily return revision. "
                         "Production retains legacy inputs on the affected session.")
        rows.append(summarize_validation(name, "ciz_vs_saved", "all_2024", selected, events,
                                         saved, modern, np.array_equal(finite, expected) and resolved, note))

    for option in (1, 2):
        match = archive_match(events, ARCHIVE / f"model_table_opt{option}.parquet")
        index = match.event_row.to_numpy()
        saved = match[EARLY].to_numpy(dtype=float)
        bench = legacy_bench
        if option == 2:
            sector = dates(pd.read_parquet(ARCHIVE / "health_sector_daily.parquet"), "date")
            bench = sector.set_index("date").sector_ret.reindex(before_end).to_numpy(dtype=float)
        day_index = early_day0[index].copy()
        day_index[day_index >= data.legacy_count] = -1
        fail_unless(np.array_equal(day_index, before_end.get_indexer(match.day0)),
                    f"Original Sector Option {option} Day 0 does not reproduce")
        pred = measure(sid[index], day_index, -1, 1, legacy_early, bench).value
        seen = np.isfinite(saved)
        fits = np.array_equal(np.isfinite(pred), seen) and np.all(np.abs(pred[seen] - saved[seen]) <= LEGACY_TOL)
        rows.append(summarize_validation(EARLY, f"legacy_vs_original_sector_option_{option}",
                                         "matched_healthcare_all_archived_years", index, events, saved, pred, fits,
                                         "Exact transcriptid, PERMNO and UTC call timestamp; Option 2 is validation only."))
        if option == 1:
            restrict = in_2024[index]
            i = index[restrict]
            actual = saved[restrict]
            recent = measure(sid[i], early_day0[i], -1, 1, data.ciz, ciz_bench,
                             exclude=data.ciz_delisted).value
            seen = np.isfinite(actual)
            fits = np.array_equal(np.isfinite(recent), seen) and np.all(np.abs(recent[seen] - actual[seen]) <= CIZ_TOL)
            rows.append(summarize_validation(EARLY, "ciz_vs_original_sector_option_1",
                                             "matched_healthcare_2024", i, events, actual, recent, fits))
    return pd.DataFrame(rows)


def export_columns():
    return [EARLY, *(name + suffix for name in WINDOWS for suffix in ("_source", "_missing_reason"))]


def extend(events, data, sid, mapped, old_day0, early_day0):
    results = {}
    early_exclude = data.ciz_delisted.copy()
    early_exclude[:, :data.legacy_count] = False
    for name, (low, high) in WINDOWS.items():
        is_early = name == EARLY
        original = events[name].notna().to_numpy() if not is_early else np.zeros(len(events), dtype=bool)
        pending = ~original
        day0 = early_day0 if is_early else old_day0
        computed = measure(sid[pending], day0[pending], low, high,
                    data.early_stock if is_early else data.stock, data.benchmark,
                    applicable=mapped[pending], exclude=early_exclude if is_early else None)
        w = Window(np.full(len(events), np.nan), np.zeros(len(events), dtype=int),
                   np.full(len(events), "", dtype=object))
        w.value[pending], w.end[pending], w.reason[pending] = computed.value, computed.end, computed.reason
        ok = np.isfinite(w.value)
        flags = np.full(len(events), "missing", dtype=object)
        flags[ok & (w.end < data.legacy_count)] = "reconstructed_legacy_crsp"
        flags[ok & (w.end >= data.legacy_count)] = "extended_crsp_ciz"
        flags[original] = "original"
        results[name] = (w, flags)
    return results


def coverage(events, results):
    years = events.call_date.dt.year.to_numpy()
    health = events.sector.eq("Health Care").to_numpy()
    output = []
    for scope, selected in [("sp500_archive_all_sectors", np.ones(len(events), bool)), ("healthcare", health)]:
        for year in np.unique(years):
            mask = selected & (years == year)
            for name, (window, flags) in results.items():
                original = mask & (flags == "original")
                ciz = mask & (flags == "extended_crsp_ciz")
                legacy = mask & (flags == "reconstructed_legacy_crsp")
                new = ciz | legacy
                missing = mask & (flags == "missing")
                problems = {"missing_" + r: int(np.sum(missing & (window.reason == r))) for r in REASONS}
                fail_unless(sum(problems.values()) == missing.sum(), f"Unexplained missing {name} values")
                output.append({"scope": scope, "year": year, "horizon": name, "events": int(mask.sum()),
                               "available_before": int(original.sum()), "new_legacy": int(legacy.sum()),
                               "new_ciz": int(ciz.sum()), "newly_filled": int(new.sum()),
                               "available_after": int((original | new).sum()), "missing_after": int(missing.sum()),
                               "latest_new_event_date": day_string(events.loc[new, "call_date"].max()),
                               "latest_available_event_date": day_string(events.loc[original | new, "call_date"].max()),
                               **problems})
    return pd.DataFrame(output)


def write_verify_csv(path, header, source, results):
    old_cols = {header.index(name): name for name in OLD}
    added = export_columns()
    flags = {name: pair[1] for name, pair in results.items()}
    values = {name: pair[0].value for name, pair in results.items()}
    reasons = {name: pair[0].reason for name, pair in results.items()}

    def new_value(name, i):
        return "" if flags[name][i] == "missing" else format(values[name][i], ".17g")

    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + added)
        for i, orig in enumerate(source):
            row = orig.copy()
            for c, name in old_cols.items():
                if not row[c]:
                    row[c] = new_value(name, i)
            row.append(new_value(EARLY, i))
            for name in WINDOWS:
                row.extend([flags[name][i], reasons[name][i]])
            writer.writerow(row)
    with path.open(encoding="utf-8", newline="") as f:
        csv_rows = csv.reader(f)
        fail_unless(next(csv_rows) == header + added, "Output header changed")
        for i, orig in enumerate(source):
            got = next(csv_rows, None)
            fail_unless(got is not None and len(got) == len(header) + len(added), "Output row dropped or malformed")
            for c, cell in enumerate(orig):
                expected = new_value(old_cols[c], i) if c in old_cols and not cell else cell
                fail_unless(got[c] == expected, f"Archive value changed in row {i + 1}, column {header[c]}")
            fail_unless(got[len(header)] == new_value(EARLY, i), "Early CAR write/read mismatch")
            for j, name in enumerate(WINDOWS):
                k = len(header) + 1 + 2 * j
                fail_unless(got[k:k + 2] == [flags[name][i], reasons[name][i]], "Provenance write/read mismatch")
                cell = got[header.index(name)] if name in OLD else got[len(header)]
                fail_unless(bool(cell) == (flags[name][i] != "missing"), "Label and source disagree")
        fail_unless(next(csv_rows, None) is None, "Output added extra rows")


def markdown_table(headings, rows):
    return "\n".join(["| " + " | ".join(headings) + " |",
                      "| " + " | ".join("---" for _ in headings) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def write_documentation(out, events, validation, summary, data, digest, mapped,
                        outside_membership, membership_n):
    inventory = markdown_table(["Input", "Rows", "First date", "Last date"],
                               [(p.replace("\\", "/"), f"{n:,}", low, high)
                                for p, n, low, high in data.inventory])
    method = f"""# S&P 500 CAR label extension

## Source definitions

The input is `model_table_with_linguistic.csv`, with {len(events):,} uniquely identified calls from the archived S&P 500 universe. The CSV is preserved as supplied, including the original Day 0 column, cohort fields, and all nonmissing CAR values. Input SHA-256 is `{digest}`. Production windows use the existing CRSP trading-session index, inclusive endpoints and the arithmetic sum of daily stock return minus the date-matched benchmark. When a new label is available, every day in its window has an observed stock and benchmark return. No daily missing observation is replaced with zero, and no truncated window is published.

### Legacy columns

Git `2a3a90a:sp500_pipeline.py:175–181` defines the original daily sources. `stage_targets()` at lines 286–369 builds abnormal returns and the four windows. Lines 340–352 use the archived UTC calendar date with the New York clock for the 16:00 cutoff. A trading-day call at or after 16:00 Eastern rolls forward, as does a call on a nontrading day. This historical calendar-date convention is retained for the legacy fields and reproduces the saved Day 0 on the entire original calendar. `firm_controls()` in `src/linguistic_alpha/controls/firm.py:73–81` is a separate producer used by other reports. It is not the source of the CSV under audit. The legacy target formulas are the sums of `RET - SPRTRN` over [0,1], [2,5], [2,20], or [2,60]. `SPRTRN` is the S&P 500 price-return series. Stock `RET` is the vendor return, including distributions. The 2009–2024 legacy daily-stock files are separate from the three selected CIZ shards.

On legacy days with an observed DLRET, the stock return is `(1 + RET) × (1 + DLRET) − 1`. If only DLRET is observed, the observed stock return for that session is DLRET. Delisting dates are rolled to the first session on or after the recorded date, following the original stage. Beginning with 2025, the pipeline uses CIZ `dlyret` directly. It already incorporates delisting returns. The old pipeline could accept 80% coverage or truncate at a delisting. The extension uses full windows throughout, including around delistings. The 2024 verification sample of existing values has complete windows, so that stricter eligibility did not change the comparisons.

### Early CAR

The full-universe Early CAR uses **Original Sector Option 1**. In Git `06273f1766daeb29de4b9313dd20639439a09671:sector_extract.py:125–143,145–153,167–196`, the stock grid is the daily `RET` grid, the benchmark is `vwretd` if the index file supplies it and otherwise `sprtrn`, and the event window is [-1,1]. The Healthcare replication input adapter at `outputs/healthcare_original_pierre_replication/run_original.py:31` exposed only `sprtrn`. The native Early computation therefore sums raw-source stock `RET` minus S&P 500 `SPRTRN` on those three observations. Unlike upstream `stage_targets()`, the Sector extractor records delisting dates for possible truncation and does not splice DLRET into the stock grid. Day 0 follows the New York date and 16:00 boundary in `src/linguistic_alpha/events/timing.py:15–57`.

For Early CAR, the extension uses the legacy return grid without a separate delisting splice through 2024. After 2024 it uses CIZ observations. Early CAR remains missing if its window crosses a CIZ day flagged as a delisting, since the original pre-splice stock-return definition cannot be established from the bundled return on that day. The original Healthcare Option 2 uses a separate sector-return series. It is tested against the saved Healthcare benchmark as a robustness check and is not used in the full-universe CSV. Both Early CAR and `car_0_1` require Day +1 returns and are unavailable at call time.

## Inputs and identity verification

{inventory}

The legacy benchmark is `data/raw/crsp_dsi.parquet`, with `SPRTRN` from {day_string(data.legacy_bench.index[0])} through {day_string(data.legacy_bench.index[-1])}. Legacy stock inputs are `data/raw/crsp_dsf_2009.parquet` through `data/raw/crsp_dsf_2024.parquet`. The legacy delist input is `data/raw/crsp_dsedelist.parquet`. The calendar and equity sources end on **{day_string(data.calendar[-1])}**. Windows spanning year-end 2024 use legacy data through 2024 and CIZ thereafter. The same legacy/CIZ switch applies to stock and benchmark inputs.

The event map verifies the CIQ transcript ID and exact UTC timestamp in `data/raw/ciq_transcript_detail.parquet`, then the date-valid CIQ-company-to-GVKEY and GVKEY-to-archived-PERMNO spells from the local `ciq_wrds_gvkey.parquet` and `crsp_ccmxpf_lnkhist.parquet` copies in `data/raw/` and `data/team_pit/sp500_pit/`. CCM uses LC/LU and P/C links. The archive's PERMNOs are checked and never replaced. This verifies {int(mapped.sum()):,} of {len(events):,} mappings. In-period CIZ S&P 500 membership is audited as a diagnostic. {outside_membership:,} of {membership_n:,} archive events from 2025 through the CIZ cutoff fall outside a membership spell in `crsp_msp500list_v2.parquet`. All remain in the output, and no membership-based selection is applied to the archived universe.

## Validation and provenance

- `car_validation.csv` compares saved values on all 2024 events across sectors, using legacy daily inputs and CIZ separately. Early CAR is matched on transcript ID, PERMNO and UTC call timestamp to both saved Original Healthcare Sector options. The legacy agreement gate is absolute error ≤ 1e−7 in decimal CAR, with exact agreement on label availability. The CIZ bridge gate is ≤ 1e−5 except for the isolated vendor revision documented in the validation row. The known revision is reconciled at the stock-day level, and legacy observations are retained on that date. Agreement rates are proportions.
- Each horizon has `<label>_source` and `<label>_missing_reason` fields. `original` preserves the exact original CSV cell string. `reconstructed_legacy_crsp` means a newly calculated label uses legacy daily inputs only. `extended_crsp_ciz` means the label uses CIZ for at least one day and may also include legacy days across the year-end boundary. `missing` means the label is empty and the companion reason is populated.
- Missing reasons identify an unverified dated security link, Day 0 outside the calendar, a return window outside the calendar, a missing benchmark, a missing stock observation, or CIZ Early CAR delisting semantics that could not be verified. The first blocking condition is recorded.
- Before publication the script reparses the output, checks every archived cell outside the eligible CAR fills, verifies identical row order and counts, and checks every label and its flags. Row joins used for validation are one-to-one. Production returns use unique security/session keys. The input file is hashed again before publication.

## Reproduction

From the repository root, run the script with Python, NumPy, pandas, and PyArrow available. Use a fresh destination. The script stops on a failed validation gate before creating the production CSV.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python outputs/sp500_car_extension/extend_car.py --out outputs/sp500_car_extension/reproduction
```

The script calculates realized labels and label availability only. No 2025–2026 holdout labels were used for feature selection, model tuning, fitting, or performance evaluation. The unchanged archive Day 0 column can remain blank after 2024. Label-specific signal and target maturity should be obtained from the applicable rule and CRSP calendar when planning a future holdout evaluation. Transcript delivery, feature availability, historical financial revisions and benchmark-vintage continuity remain separate verification tasks.
"""
    (out / "methodology.md").write_text(method, encoding="utf-8")

    label_order = [*OLD, EARLY]
    recent = summary.loc[summary.year.ge(2024)]
    counts = []
    for (scope, year), block in recent.groupby(["scope", "year"], sort=False):
        b = block.set_index("horizon").loc[label_order]
        counts.append(["全表" if scope == "sp500_archive_all_sectors" else "医疗保健",
                       year, f"{b.events.iloc[0]:,}",
                       *[f"{x.newly_filled:,} / {x.available_after:,}" for x in b.itertuples()]])
    latest = []
    full = summary.loc[summary.scope.eq("sp500_archive_all_sectors")]
    health = summary.loc[summary.scope.eq("healthcare")]
    for h in label_order:
        f, g = full.loc[full.horizon.eq(h)], health.loc[health.horizon.eq(h)]
        latest.append([f"`{h}`", f"{f.newly_filled.sum():,}", f"{g.newly_filled.sum():,}",
                       f.latest_new_event_date.max(), g.latest_new_event_date.max(), f"{f.missing_after.sum():,}"])
    errors = []
    for row in validation.itertuples():
        errors.append([f"`{row.label}`", f"`{row.check}`", f"{row.compared_events:,}",
                       f"{row.mean_abs_difference:.3g}", f"{row.median_abs_difference:.3g}",
                       f"{row.max_abs_difference:.3g}", f"{row.agreement_at_1e_5:.2%}"])
    missing_counts = full[["missing_" + reason for reason in REASONS]].sum()
    missing_text = "；".join(f"`{k[len('missing_'):]}` {v:,}" for k, v in missing_counts.items() if v)
    zh = f"""# 标普 500 CAR 标签扩展审计

**验证通过。** 新 CSV 保留 {len(events):,} 场电话会的原始顺序和所有原始字段。旧列只回填空值。新增 `car_-1_1` 与旧 `car_0_1` 保持独立。所有行业均参与计算，医疗保健单列作覆盖诊断。

## 验证

以下差异为小数 CAR，`1e−4` 等于一个基点。

{markdown_table(["标签", "比较", "匹配且可比较", "平均绝对差", "绝对差中位数", "最大绝对差", "≤1e−5"], errors)}

原四列使用旧版个股日收益与标普 500 价格收益的逐日差值求和。原版 Early CAR 使用纽约日期的 16:00 规则和 Day −1 至 Day +1 三个交易日，按 Original Sector Option 1 计算。原版四列的旧 Day 0 沿用归档 UTC 日期加纽约时钟的规则。医疗保健行业基准仅用于复验历史 Option 2。两套历史 Early CAR 均以精确事件键配对。

CIZ 复验在 2024 年的一个长窗口上发现约 15.4 基点差异，主要来自 2024-10-15 一条非退市个股日收益修订。产出保留该日及 2024 年底前的旧日收益，从 2025 年起接入 CIZ。CIZ 日收益已含退市处理，未重复叠加。所有新值要求完整窗口和逐日有效股票及基准收益。

## 覆盖

每格为「新增 / 扩展后非空」。

{markdown_table(["样本", "年", "事件", *[f"`{h}`" for h in label_order]], counts)}

{markdown_table(["标签", "全表新增合计", "医疗保健新增合计", "全表最晚新增电话会", "医疗保健最晚新增电话会", "全表仍缺"], latest)}

全历史逐年覆盖见 `coverage_summary.csv`。CIZ 缓存实际覆盖至 **{day_string(data.calendar[-1])}**。剩余缺失按标签单元格计数，{missing_text}。

2025–2026 为后续样本外评估保留。本次只生成已实现标签和覆盖审计。2026 年尚未成熟的窗口需要后续交易日历、个股日收益和标普 500 基准。通话时模型应排除两个要求 Day +1 收盘的反应窗口。历史成员、逐字稿到达时间及财务字段的时点资格仍需在未来建模时单独验证。

[方法和复验](methodology.md) · [验证统计](car_validation.csv) · [覆盖统计](coverage_summary.csv)
"""
    (out / "summary_zh.md").write_text(zh, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    fail_unless(not any((out / f).exists() for f in FILES), "Use a fresh output directory; results already exist")
    digest = sha256(INPUT.read_bytes()).hexdigest()
    header, source, events = read_text_table()
    mapped, outside_membership, membership_n = verify_mapping(events)
    data = load_returns(events)
    sid = data.securities.get_indexer(events.permno)
    old_archive_day0 = locate_day0(events, data.legacy_bench.index, legacy=True)
    early_day0 = locate_day0(events, data.calendar, legacy=False)
    validation = validate(events, data, sid, old_archive_day0, early_day0)
    print(validation[["label", "check", "compared_events", "max_abs_difference", "gate_passed"]].to_string(index=False))
    out.mkdir(parents=True, exist_ok=True)
    if not validation.gate_passed.all():
        validation.to_csv(out / "car_validation.csv", index=False, mode="x")
        raise SystemExit("Validation gate failed. No production CSV was generated.")
    old_day0 = locate_day0(events, data.calendar, legacy=True)
    results = extend(events, data, sid, mapped, old_day0, early_day0)
    summary = coverage(events, results)
    with TemporaryDirectory(prefix=".car-extension-", dir=out) as folder:
        tmp = Path(folder)
        write_verify_csv(tmp / FILES[0], header, source, results)
        validation.to_csv(tmp / FILES[1], index=False, float_format="%.12g")
        summary.to_csv(tmp / FILES[2], index=False)
        write_documentation(tmp, events, validation, summary, data, digest, mapped,
                            outside_membership, membership_n)
        fail_unless(sha256(INPUT.read_bytes()).hexdigest() == digest, "Input CSV changed during execution")
        # Publish the production CSV only after the data and documentation pass.
        for name in [*FILES[1:], FILES[0]]:
            (tmp / name).rename(out / name)
    print(f"Verified {len(events):,} rows, {len(header)} archived columns and all label-source flags.")


if __name__ == "__main__":
    main()
