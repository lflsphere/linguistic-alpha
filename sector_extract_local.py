"""
Sector extraction from the LOCAL raw pull (data/raw/sp500_pit_2008, CRSP CIZ format).

Same method as it_sector_extract.py, for any sector:
  Option 1: CAR vs. S&P 500 (sprtrn)
  Option 2: CAR vs. the sector's value-weighted S&P 500 members
Calls are selected by GICS sector on the call date (comp.co_hgic; --sample column uses the
table's current-GICS `sector` column instead); non-CAR
columns come from that table unchanged, CARs and day0 are recomputed (4pm ET cutoff).

CRSP is truncated at --end-date (default 2024-12-31), the last CRSP day behind it_results/,
so the train/test windows are the same as the IT run.

Usage:
    PYTHONPATH=src python sector_extract_local.py --raw <.../data/raw/sp500_pit_2008> \
        --sector "Consumer Discretionary" --out cd_results [--text-out <dir>]
"""
import argparse, glob, os, re, time
import numpy as np
import pandas as pd

from linguistic_alpha.events.timing import call_timestamp_utc, assign_day0

GICS = {"Energy": "10", "Materials": "15", "Industrials": "20", "Consumer Discretionary": "25",
        "Consumer Staples": "30", "Health Care": "35", "Financials": "40",
        "Information Technology": "45", "Communication Services": "50",
        "Utilities": "55", "Real Estate": "60"}
WINDOWS = {"car_-1_1": (-1, 1), "car_2_5": (2, 5), "car_2_20": (2, 20), "car_2_60": (2, 60)}
SECTION = {2: "prepared", 3: "question", 4: "answer"}   # same codes as sp500_pipeline.py


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_crsp(raw, start, end):
    files = [f for f in sorted(glob.glob(f"{raw}/crsp_dsf_v2_*.parquet"))
             if start.year <= int(re.search(r"(\d{4})\.parquet", f).group(1)) <= end.year]
    cols = ["permno", "dlycaldt", "dlyret", "dlyprc", "shrout"]
    c = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    c = c.rename(columns={"dlycaldt": "date", "dlyret": "ret", "dlyprc": "prc"})
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")
    for col in ["ret", "prc", "shrout"]:
        c[col] = pd.to_numeric(c[col]).astype(float)
    c["permno"] = c["permno"].astype(int)
    return c[c["date"].between(start, end)].drop_duplicates(["permno", "date"])


def sector_vw_returns(raw, crsp, gics, pit=True):
    """Daily VW return (same-day mcap weights, as in it_sector_extract) of S&P 500 members in the sector.

    pit=True: a firm counts on date d only if its GICS sector *on d* (comp.co_hgic indfrom..indthru)
    is the sector and its CCM link is active on d. pit=False: current GICS applied to all history
    (the it_sector_extract method; it leaks later reclassifications into the past).
    """
    h = pd.read_parquet(f"{raw}/comp_co_hgic.parquet").sort_values(["gvkey", "indfrom"])
    ccm = pd.read_parquet(f"{raw}/crsp_ccmxpf_lnkhist.parquet")
    ccm = ccm[ccm["linkprim"].isin(["P", "C"])].dropna(subset=["permno"]).copy()
    ccm["permno"] = ccm["permno"].astype(int)
    sp = pd.read_parquet(f"{raw}/crsp_msp500list_v2.parquet")[["permno", "mbrstartdt", "mbrenddt"]]
    for c in ["mbrstartdt", "mbrenddt"]:
        sp[c] = pd.to_datetime(sp[c])
    sp["permno"] = sp["permno"].astype(int)
    if pit:
        h["indfrom"] = pd.to_datetime(h["indfrom"])
        h["indthru"] = pd.to_datetime(h["indthru"]).fillna(pd.Timestamp("2099-12-31"))
        h = h[h["gsector"].astype(str) == gics][["gvkey", "indfrom", "indthru"]]
        ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
        ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
        lk = ccm[ccm["gvkey"].isin(set(h["gvkey"]))][["gvkey", "permno", "linkdt", "linkenddt"]]
        x = crsp[crsp["permno"].isin(set(lk["permno"]) & set(sp["permno"]))].merge(lk, on="permno").merge(h, on="gvkey")
        x = x[x["date"].between(x["linkdt"], x["linkenddt"]) & x["date"].between(x["indfrom"], x["indthru"])]
        x = x.drop_duplicates(["permno", "date"])[crsp.columns]
    else:
        current = h.groupby("gvkey").tail(1)                 # current GICS, like comp.company
        gv = set(current.loc[current["gsector"].astype(str) == gics, "gvkey"])
        pn = set(ccm.loc[ccm["gvkey"].isin(gv), "permno"]) & set(sp["permno"])
        x = crsp[crsp["permno"].isin(pn)].copy()
    log(f"sector index ({'point-in-time' if pit else 'current'} GICS): {x['permno'].nunique()} permnos")
    x = x.copy()
    x["mcap"] = x["prc"].abs() * x["shrout"]
    x = x.dropna(subset=["ret", "mcap"])
    x = x[x["mcap"] > 0].merge(sp, on="permno")
    x = x[(x["date"] >= x["mbrstartdt"]) & (x["date"] <= x["mbrenddt"])].drop_duplicates(["permno", "date"])
    x["w"] = x["ret"] * x["mcap"]
    g = x.groupby("date")[["w", "mcap"]].sum()
    out = (g["w"] / g["mcap"]).rename("sector_ret").reset_index()
    log(f"sector VW: {len(out):,} days, annualized {((1 + out.sector_ret).prod() ** (252 / len(out)) - 1):.4f}, "
        f"avg members/day {x.groupby('date').size().mean():.1f}")
    return out


def compute_cars(calls, crsp, bench, days, delist):
    nd = len(days)
    permnos = np.sort(calls["permno"].unique())
    pidx = {p: i for i, p in enumerate(permnos)}
    sub = crsp[crsp["permno"].isin(pidx)]
    ret = np.full((len(permnos), nd), np.nan)
    ret[sub["permno"].map(pidx).values, np.searchsorted(days, sub["date"].values)] = sub["ret"].values
    ar = ret - bench[None, :]

    def csum(a):
        return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(np.nan_to_num(a), axis=1)], axis=1)
    cs_ar, cs_n = csum(ar), csum((~np.isnan(ar)).astype(float))

    d0 = np.minimum(np.searchsorted(days, calls["day0"].values.astype("datetime64[ns]")), nd - 1)
    valid = calls["day0"].notna().values
    rr = calls["permno"].map(pidx).values
    dl = calls["permno"].map(delist).fillna(-1).astype(int).values
    out = {}
    for name, (a, b) in WINDOWS.items():
        lo, hi = d0 + a, d0 + b + 1
        ok = valid & (lo >= 0) & (hi <= nd)
        s, n = np.full(len(calls), np.nan), np.full(len(calls), np.nan)
        s[ok] = cs_ar[rr[ok], hi[ok]] - cs_ar[rr[ok], lo[ok]]
        n[ok] = cs_n[rr[ok], hi[ok]] - cs_n[rr[ok], lo[ok]]
        hi2 = np.minimum(d0 + b, dl) + 1
        g = valid & (dl >= d0 + a) & (dl < d0 + b) & (lo >= 0) & (hi2 <= nd)
        s[g] = cs_ar[rr[g], hi2[g]] - cs_ar[rr[g], lo[g]]
        out[name] = np.where((n >= 0.8 * (b - a + 1)) | g, s, np.nan)
    return out


def write_text(raw, calls, out_dir):
    """Rebuild prepared_text / qa_text per call exactly as sp500_pipeline.stage_text does."""
    os.makedirs(out_dir, exist_ok=True)
    missing = 0
    for y, grp in calls.groupby(calls["call_date"].dt.year):
        ids = [int(t) for t in grp["transcriptid"]]
        t = pd.read_parquet(f"{raw}/ciq_transcript_component_{y}.parquet",
                            columns=["transcriptid", "componentorder", "transcriptcomponenttypeid", "componenttext"],
                            filters=[("transcriptid", "in", ids)])
        t["section"] = t["transcriptcomponenttypeid"].map(SECTION)
        t = t.dropna(subset=["section"]).sort_values(["transcriptid", "componentorder"])
        t["componenttext"] = t["componenttext"].fillna("")
        rows = []
        for tid, g in t.groupby("transcriptid", sort=False):
            qa = g[g["section"] != "prepared"]
            rows.append({"transcriptid": tid,
                         "prepared_text": "\n\n".join(g.loc[g["section"] == "prepared", "componenttext"]),
                         "qa_text": "\n\n".join(qa["section"].str.upper() + ": " + qa["componenttext"])})
        pd.DataFrame(rows).to_parquet(f"{out_dir}/text_{y}.parquet", index=False)
        missing += len(set(ids) - {r["transcriptid"] for r in rows})
    log(f"text written to {out_dir}; calls without text: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--sector", default="Consumer Discretionary", choices=list(GICS))
    ap.add_argument("--table", default="model_table_with_linguistic.csv")
    ap.add_argument("--out", default="cd_results")
    ap.add_argument("--end-date", default="2024-12-31")
    ap.add_argument("--text-out")
    ap.add_argument("--sample", choices=["pit", "column"], default="pit",
                    help="call selection: GICS sector on the call date (default) or the table's current-GICS `sector` column")
    ap.add_argument("--gics", choices=["pit", "current"], default="pit",
                    help="sector index membership: point-in-time GICS (default) or current GICS (IT method)")
    args = ap.parse_args()
    raw, end = args.raw, pd.Timestamp(args.end_date)
    start = pd.Timestamp("2009-01-01")
    os.makedirs(args.out, exist_ok=True)

    mt = pd.read_csv(args.table, dtype={"gvkey": str, "sic": str, "gind": str}, low_memory=False)
    mt["call_date"] = pd.to_datetime(mt["call_date"])
    mt["gvkey"] = mt["gvkey"].str.zfill(6)
    if args.sample == "pit":
        h = pd.read_parquet(f"{raw}/comp_co_hgic.parquet")
        h["indfrom"] = pd.to_datetime(h["indfrom"])
        h["indthru"] = pd.to_datetime(h["indthru"]).fillna(pd.Timestamp("2099-12-31"))
        x = mt[["transcriptid", "gvkey", "call_date"]].merge(h[["gvkey", "gsector", "indfrom", "indthru"]], on="gvkey")
        x = x[x["call_date"].between(x["indfrom"], x["indthru"])].drop_duplicates("transcriptid")
        pit_ids = set(x.loc[x["gsector"].astype(str) == GICS[args.sector], "transcriptid"])
        col_ids = set(mt.loc[mt["sector"] == args.sector, "transcriptid"])
        log(f"point-in-time GICS sample: {len(pit_ids):,} calls ({len(pit_ids - col_ids)} added, "
            f"{len(col_ids - pit_ids)} dropped vs. the current-GICS `sector` column)")
        calls = mt[mt["transcriptid"].isin(pit_ids)].copy().reset_index(drop=True)
    else:
        calls = mt[mt["sector"] == args.sector].copy().reset_index(drop=True)
    calls["permno"] = calls["permno"].astype(int)
    log(f"{args.sector}: {len(calls):,} calls, {calls['permno'].nunique()} companies")

    crsp = load_crsp(raw, start, end)
    spx = pd.read_parquet(f"{raw}/crsp_dsp500_v2.parquet", columns=["caldt", "sprtrn"])
    spx["caldt"] = pd.to_datetime(spx["caldt"]).astype("datetime64[ns]")
    spx = spx[spx["caldt"].between(start, end)].sort_values("caldt")
    days = spx["caldt"].values
    crsp = crsp[crsp["date"].isin(days)]
    log(f"CRSP daily: {len(crsp):,} rows, {len(days):,} days to {pd.Timestamp(days[-1]).date()}")

    vw = sector_vw_returns(raw, crsp, GICS[args.sector], pit=args.gics == "pit")
    vw.to_parquet(f"{args.out}/sector_daily.parquet", index=False)
    sec_bench = vw.set_index("date")["sector_ret"].reindex(pd.DatetimeIndex(days)).values

    dl = pd.read_parquet(f"{raw}/crsp_stkdelists.parquet").dropna(subset=["delret", "delistingdt"])
    di = np.searchsorted(days, pd.to_datetime(dl["delistingdt"]).values.astype("datetime64[ns]"))
    delist = {int(p): i for p, i in zip(dl["permno"], di) if i < len(days)}

    ts = call_timestamp_utc(calls["call_date"], calls["call_time_utc"])
    calls["day0"] = assign_day0(ts, pd.DatetimeIndex(days), call_date_fallback=calls["call_date"])["day0"].values

    base = calls.drop(columns=[c for c in calls.columns if c.startswith("car_")])
    for opt, bench in [(1, spx["sprtrn"].astype(float).values), (2, sec_bench)]:
        df = base.copy()
        for k, v in compute_cars(calls, crsp, bench, days, delist).items():
            df[k] = v
        df.to_parquet(f"{args.out}/model_table_opt{opt}.parquet", index=False)
        v = df["car_2_20"].dropna()
        tr = df.loc[v.index, "call_date"] <= "2023-12-31"
        log(f"Option {opt}: {len(v):,} with target (train {tr.sum():,} / test {(~tr).sum():,}), "
            f"CAR[+2,+20] mean={v.mean():.4f} std={v.std():.4f} median={v.median():.4f}")
    if args.text_out:
        write_text(raw, calls, args.text_out)
    log("done")


if __name__ == "__main__":
    main()
