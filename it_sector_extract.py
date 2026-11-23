"""
IT Sector Extraction — builds two model tables for the IT-only subsample.

Option 1: CAR relative to S&P 500 (market benchmark) — recomputed with corrected day0.
Option 2: CAR relative to IT sector value-weighted index — recomputed with corrected day0.

Both options use assign_day0() (4pm ET cutoff) and car_-1_1 [-1,+1] EA window.

Requires: ~/earnings_sp500/ with raw/ parquets and model_table.parquet (for non-CAR columns)
Outputs:  ~/earnings_it/
            model_table_opt1.parquet   (market-benchmarked CARs, IT only)
            model_table_opt2.parquet   (IT-benchmarked CARs)
            it_sector_daily.parquet    (IT sector VW daily returns)

Usage:
    python3 it_sector_extract.py [--folder ~/earnings_sp500]
"""
import argparse, gc, os, sys, time, warnings
import numpy as np
import pandas as pd

from linguistic_alpha.events.timing import call_timestamp_utc, assign_day0

warnings.filterwarnings("ignore", category=FutureWarning)

START_YEAR = 2010
IT_GICS_SECTOR = "45"
IT_SECTOR_NAME = "Information Technology"

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def to_dt(df, *cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.expanduser("~/earnings_sp500"))
    args = ap.parse_args()
    BASE = args.folder
    RAW = f"{BASE}/raw"
    OUT = os.path.expanduser("~/earnings_it")
    os.makedirs(OUT, exist_ok=True)

    # --- Load base model_table for non-CAR columns ---
    log("loading model_table.parquet ...")
    mt = pd.read_parquet(f"{BASE}/model_table.parquet")
    to_dt(mt, "call_date")
    log(f"full dataset: {len(mt):,} rows, {mt['permno'].nunique()} companies")
    log(f"sectors: {mt['sector'].value_counts().to_dict()}")

    it = mt[mt["sector"] == IT_SECTOR_NAME].copy().reset_index(drop=True)
    log(f"IT sector: {len(it):,} rows, {it['permno'].nunique()} companies")

    # --- Load raw CRSP for recomputing CARs with corrected day0 ---
    log("loading raw CRSP daily returns ...")
    import glob
    crsp_files = sorted(glob.glob(f"{RAW}/crsp_daily_*.parquet"))
    crsp = pd.concat([pd.read_parquet(f) for f in crsp_files], ignore_index=True)
    crsp = to_dt(crsp, "date")
    log(f"CRSP daily: {len(crsp):,} rows")

    spx = to_dt(pd.read_parquet(f"{RAW}/sp500_index_daily.parquet"), "date").sort_values("date")
    days = spx["date"].values.astype("datetime64[ns]")
    nd = len(days)
    crsp = crsp[crsp["date"].isin(spx["date"])]

    # Identify IT sector permnos from company data
    comp = pd.read_parquet(f"{RAW}/company.parquet")
    it_gvkeys = set(comp[comp["gsector"].astype(str).str[:2] == IT_GICS_SECTOR]["gvkey"].unique())
    log(f"IT gvkeys: {len(it_gvkeys)}")

    # gvkey -> permno mapping from CCM
    ccm = to_dt(pd.read_parquet(f"{RAW}/link_ccm.parquet"), "linkdt", "linkenddt")
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    it_permnos_ccm = set(ccm[ccm["gvkey"].isin(it_gvkeys)]["permno"].unique())

    # S&P 500 membership
    sp = to_dt(pd.read_parquet(f"{RAW}/sp500_membership.parquet"), "start", "ending")
    sp_permnos = set(sp["permno"].unique())

    # IT permnos = in CCM as IT gvkey AND in S&P 500
    it_permnos = it_permnos_ccm & sp_permnos
    log(f"IT permnos (ever in S&P 500): {len(it_permnos)}")

    # Compute value-weighted IT sector daily returns
    crsp_it = crsp[crsp["permno"].isin(it_permnos)].copy()
    crsp_it["mcap"] = crsp_it["prc"].abs() * crsp_it["shrout"]
    crsp_it = crsp_it.dropna(subset=["ret", "mcap"])
    crsp_it = crsp_it[crsp_it["mcap"] > 0]

    # For each day, check S&P 500 membership
    # Merge with membership spells to filter to only current members
    crsp_it_sp = crsp_it.merge(sp, on="permno")
    crsp_it_sp = crsp_it_sp[(crsp_it_sp["date"] >= crsp_it_sp["start"]) &
                             (crsp_it_sp["date"] <= crsp_it_sp["ending"])]
    crsp_it_sp = crsp_it_sp.drop_duplicates(subset=["permno", "date"])

    # Value-weighted return: weight by prior day's market cap (use same-day as proxy)
    daily_vw = (
        crsp_it_sp.groupby("date")
        .apply(lambda g: np.average(g["ret"], weights=g["mcap"]), include_groups=False)
        .reset_index(name="it_ret")
    )
    daily_vw = daily_vw.sort_values("date").reset_index(drop=True)
    log(f"IT sector VW daily returns: {len(daily_vw):,} days")
    log(f"  mean daily return: {daily_vw['it_ret'].mean():.6f}")
    log(f"  annualized return: {((1 + daily_vw['it_ret']).prod() ** (252/len(daily_vw)) - 1):.4f}")

    daily_vw.to_parquet(f"{OUT}/it_sector_daily.parquet", index=False)

    # Merge IT sector return into the trading calendar
    it_ret_map = dict(zip(daily_vw["date"], daily_vw["it_ret"]))
    it_bench = np.array([it_ret_map.get(d, np.nan) for d in days])

    # Build return grid for IT firms only (same approach as sp500_pipeline stage_targets)
    panel = to_dt(pd.read_parquet(f"{BASE}/event_panel.parquet"), "call_date")
    panel_it = panel[panel["gvkey"].isin(it_gvkeys)].copy().reset_index(drop=True)
    log(f"IT event panel: {len(panel_it):,} calls")

    permnos = np.sort(panel_it["permno"].dropna().unique())
    pidx = {p: i for i, p in enumerate(permnos)}
    crsp_sub = crsp[crsp["permno"].isin(pidx)].copy()
    r = crsp_sub["permno"].map(pidx).values
    c = np.searchsorted(days, crsp_sub["date"].values)

    def grid(values):
        a = np.full((len(permnos), nd), np.nan)
        a[r, c] = values
        return a

    ret = grid(crsp_sub["ret"].values)

    # S&P 500 market return for Option 1 benchmark
    spx_ret = spx["vwretd"].values if "vwretd" in spx.columns else spx["sprtrn"].values
    ar_mkt = ret - spx_ret[None, :]

    # Abnormal returns relative to IT sector index
    ar_it = ret - it_bench[None, :]

    # Also handle delisting returns
    dl = to_dt(pd.read_parquet(f"{RAW}/crsp_delist.parquet"), "dlstdt").dropna(subset=["dlret", "dlstdt"])
    dl = dl[dl["permno"].isin(pidx)]
    delisted_day = {}
    if len(dl):
        di = np.searchsorted(days, dl["dlstdt"].values)
        dl = dl[di < nd].copy()
        dl["date"] = days[di[di < nd]]
        delisted_day = dict(zip(dl["permno"], np.searchsorted(days, dl["date"].values)))

    ok_mkt = ~np.isnan(ar_mkt)
    ok_it = ~np.isnan(ar_it)

    def csum(a):
        z = np.nan_to_num(a)
        return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)

    cs_ar_mkt = csum(ar_mkt)
    cs_n_mkt = csum(ok_mkt.astype(float))
    cs_ar = csum(ar_it)
    cs_n = csum(ok_it.astype(float))

    call_ts = call_timestamp_utc(panel_it["call_date"], panel_it["call_time_utc"])
    day0_df = assign_day0(call_ts, pd.DatetimeIndex(days),
                          call_date_fallback=panel_it["call_date"])
    panel_it["day0"] = day0_df["day0"].values
    d0 = np.searchsorted(days, panel_it["day0"].values.astype("datetime64[ns]"))
    d0 = np.minimum(d0, nd - 1)
    row = panel_it["permno"].map(pidx).fillna(-1).astype(int).values
    valid = (row >= 0) & (d0 < nd) & panel_it["day0"].notna().values
    rr = np.where(valid, row, 0)

    def window_sum(cs, a, b):
        lo, hi = d0 + a, d0 + b + 1
        good = valid & (lo >= 0) & (hi <= nd)
        out = np.full(len(panel_it), np.nan)
        out[good] = cs[rr[good], hi[good]] - cs[rr[good], lo[good]]
        return out

    dlday = panel_it["permno"].map(delisted_day).fillna(-1).astype(int).values

    # Compute market-benchmarked CARs (Option 1: S&P 500)
    log("  computing Option 1 CARs (S&P 500 benchmark, corrected day0) ...")
    mkt_cars = {}
    for name, (a, b) in {"car_-1_1": (-1, 1), "car_2_5": (2, 5), "car_2_20": (2, 20), "car_2_60": (2, 60)}.items():
        s = window_sum(cs_ar_mkt, a, b)
        n = window_sum(cs_n_mkt, a, b)
        delisted_in = (dlday >= d0 + a) & (dlday < d0 + b)
        lo, hi = d0 + a, np.minimum(d0 + b, dlday) + 1
        g = valid & delisted_in & (lo >= 0) & (hi <= nd)
        s[g] = cs_ar_mkt[rr[g], hi[g]] - cs_ar_mkt[rr[g], lo[g]]
        mkt_cars[name] = np.where((n >= 0.8 * (b - a + 1)) | g, s, np.nan)
        log(f"    {name}: {np.sum(~np.isnan(mkt_cars[name])):,} valid")

    # Compute IT-benchmarked CARs (Option 2: IT sector VW index)
    log("  computing Option 2 CARs (IT sector benchmark) ...")
    it_cars = {}
    for name, (a, b) in {"car_-1_1": (-1, 1), "car_2_5": (2, 5), "car_2_20": (2, 20), "car_2_60": (2, 60)}.items():
        s = window_sum(cs_ar, a, b)
        n = window_sum(cs_n, a, b)
        delisted_in = (dlday >= d0 + a) & (dlday < d0 + b)
        lo, hi = d0 + a, np.minimum(d0 + b, dlday) + 1
        g = valid & delisted_in & (lo >= 0) & (hi <= nd)
        s[g] = cs_ar[rr[g], hi[g]] - cs_ar[rr[g], lo[g]]
        it_cars[name] = np.where((n >= 0.8 * (b - a + 1)) | g, s, np.nan)
        log(f"    {name}: {np.sum(~np.isnan(it_cars[name])):,} valid")

    # Build CAR + day0 map from panel_it
    car_map = panel_it[["keydevid", "day0"]].copy()
    car_map = car_map.rename(columns={"day0": "day0_new"})
    for name in mkt_cars:
        car_map[f"{name}_mkt"] = mkt_cars[name]
    for name in it_cars:
        car_map[f"{name}_it"] = it_cars[name]

    def replace_cars_and_day0(base_df, car_dict, suffix):
        out = base_df.copy()
        old_car_cols = [c for c in out.columns if c.startswith("car_")]
        out = out.drop(columns=old_car_cols)
        merge_cols = ["keydevid", "day0_new"] + [f"{n}_{suffix}" for n in car_dict]
        out = out.merge(car_map[merge_cols], on="keydevid", how="left")
        out["day0"] = out["day0_new"]
        out = out.drop(columns="day0_new")
        for name in car_dict:
            out[name] = out[f"{name}_{suffix}"]
            out = out.drop(columns=f"{name}_{suffix}")
        return out

    # --- OPTION 1: Replace old CARs + day0 with recomputed values ---
    opt1 = replace_cars_and_day0(it, mkt_cars, "mkt")
    opt1.to_parquet(f"{OUT}/model_table_opt1.parquet", index=False)
    log(f"saved Option 1 (S&P 500 bench, recomputed with corrected day0): {OUT}/model_table_opt1.parquet")

    # --- OPTION 2: Replace old CARs + day0 with IT-benchmarked values ---
    opt2 = replace_cars_and_day0(it, it_cars, "it")
    opt2.to_parquet(f"{OUT}/model_table_opt2.parquet", index=False)
    log(f"saved Option 2 (IT-benchmarked CARs): {OUT}/model_table_opt2.parquet")

    # Summary stats
    for label, df in [("Option 1 (market bench)", opt1), ("Option 2 (IT bench)", opt2)]:
        valid_car = df["car_2_20"].dropna()
        log(f"\n{label}:")
        log(f"  calls: {len(df):,}, companies: {df['permno'].nunique()}")
        log(f"  train (<=2021): {len(df[df['call_date'] <= '2021-12-31']):,}")
        log(f"  test  (>2021):  {len(df[df['call_date'] > '2021-12-31']):,}")
        log(f"  CAR[+2,+20] mean={valid_car.mean():.4f}, std={valid_car.std():.4f}, "
            f"median={valid_car.median():.4f}")

    log("\ndone.")

if __name__ == "__main__":
    main()
