"""Identifier linking: CIQ companyid -> gvkey -> permno -> I/B/E/S ticker, valid on the call date.

Implemented with date-valid CIQ, CCM, and I/B/E/S links.

Rules (fixes vs. the legacy notebook, docs/data_audit.md §Linking):
  1. ciq.wrds_gvkey: keep rows with startdate <= call_date <= enddate (NULL enddate = open).
  2. crsp.ccmxpf_lnkhist: linktype in (LU, LC), linkprim in (P, C), date-valid on call_date.
     If several remain, prefer linkprim P, then LC over LU.
  3. wrdsapps.ibcrsphist: date-valid on call_date, keep the lowest ``score`` (best match), and
     require score <= 2. Never drop to an arbitrary row via ``drop_duplicates``.
  4. Every step logs how many events it loses, so link attrition is reported, not hidden.

Output: ``interim/event_links`` with one row per keydevid and ``link_*`` diagnostic columns.
"""

from __future__ import annotations

import pandas as pd


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _event_date(events: pd.DataFrame) -> pd.Series:
    for col in ("call_date_et", "call_date", "mostimportantdateutc"):
        if col in events:
            return _date(events[col])
    raise ValueError("events require call_date_et, call_date, or mostimportantdateutc")


def _valid(joined: pd.DataFrame, date: pd.Series, start: str, end: str) -> pd.Series:
    lo = _date(joined[start])
    hi = _date(joined[end])
    d = _date(date)
    return (lo.isna() | (lo <= d)) & (hi.isna() | (hi >= d))


def link_ciq_to_gvkey(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame) -> pd.DataFrame:
    c = calls.copy()
    c["_link_date"] = _event_date(c)
    l = ciq_gvkey.copy()
    l["gvkey"] = l["gvkey"].astype("string").str.zfill(6)
    x = c.merge(l, on="companyid", how="left", suffixes=("", "_link"))
    ok = _valid(x, x["_link_date"], "startdate", "enddate") & x["gvkey"].notna()
    x = x[ok].copy()
    if "primaryflag" in x:
        x["_primary"] = x["primaryflag"].astype("string").str.upper().isin(["Y", "1", "T"])
        x = x.sort_values(["keydevid", "_primary", "startdate", "gvkey"],
                          ascending=[True, False, False, True], na_position="last")
    else:
        x = x.sort_values(["keydevid", "startdate", "gvkey"],
                          ascending=[True, False, True], na_position="last")
    x = x.drop_duplicates("keydevid", keep="first")
    return x.drop(columns=["_link_date", "_primary"], errors="ignore").reset_index(drop=True)


def link_gvkey_to_permno(events: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["gvkey"] = e["gvkey"].astype("string").str.zfill(6)
    e["_link_date"] = _event_date(e)
    l = ccm.copy()
    l["gvkey"] = l["gvkey"].astype("string").str.zfill(6)
    x = e.merge(l, on="gvkey", how="left", suffixes=("", "_ccm"))
    ok = _valid(x, x["_link_date"], "linkdt", "linkenddt")
    ok &= x["linktype"].isin(["LU", "LC"]) & x["linkprim"].isin(["P", "C"])
    x = x[ok].assign(_prim=lambda d: d["linkprim"].map({"P": 0, "C": 1}),
                     _type=lambda d: d["linktype"].map({"LC": 0, "LU": 1}))
    x = x.sort_values(["keydevid", "_prim", "_type", "permno"])
    x = x.drop_duplicates("keydevid", keep="first")
    return x.drop(columns=["_link_date", "_prim", "_type"], errors="ignore").reset_index(drop=True)


def link_permno_to_ibes(events: pd.DataFrame, iclink: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["_link_date"] = _event_date(e)
    l = iclink.copy()
    x = e.merge(l, on="permno", how="left", suffixes=("", "_ibes"))
    ok = _valid(x, x["_link_date"], "sdate", "edate")
    score = pd.to_numeric(x["score"], errors="coerce")
    x = x[ok & score.le(2) & x["ticker"].notna()].copy()
    x["score"] = pd.to_numeric(x["score"], errors="coerce")
    x = x.sort_values(["keydevid", "score", "ticker"], na_position="last")
    x = x.drop_duplicates("keydevid", keep="first")
    # Keep events without a valid I/B/E/S link so the sample audit can report them.
    unmatched = events.loc[~events["keydevid"].isin(x["keydevid"])].copy()
    for col in iclink.columns:
        if col not in unmatched:
            unmatched[col] = pd.NA
    out = pd.concat([x, unmatched], ignore_index=True, sort=False)
    return out.drop(columns="_link_date", errors="ignore").sort_values("keydevid").reset_index(drop=True)


def build_event_links(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame, ccm: pd.DataFrame,
                      iclink: pd.DataFrame) -> pd.DataFrame:
    """Chain the three links; return one row per keydevid plus attrition diagnostics."""
    n = len(calls)
    out = link_ciq_to_gvkey(calls, ciq_gvkey)
    print(f"CIQ-GVKEY: {len(out):,}/{n:,} events linked")
    n = len(out)
    out = link_gvkey_to_permno(out, ccm)
    print(f"GVKEY-PERMNO: {len(out):,}/{n:,} events linked")
    n = len(out)
    out = link_permno_to_ibes(out, iclink)
    print(f"PERMNO-IBES: {len(out):,}/{n:,} events linked")
    if out["keydevid"].duplicated().any():
        raise ValueError("event links are not unique by keydevid")
    return out
