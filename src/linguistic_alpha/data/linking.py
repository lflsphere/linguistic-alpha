"""Identifier linking: CIQ companyid -> gvkey -> permno -> I/B/E/S ticker, valid on the call date.

Rules (fixes vs. the legacy notebook, docs/data_audit.md §Linking):
  1. ciq.wrds_gvkey: keep rows with startdate <= call_date <= enddate (NULL enddate = open).
     Several valid rows (none in the 2026-11-17 pull): prefer primaryflag 1, then the latest
     startdate.
  2. crsp.ccmxpf_lnkhist: linktype in (LU, LC), date-valid on call_date. Every linkprim is kept
     as a candidate (P, C, and the secondary share classes J / N: 18 links in the pull), because
     one permno per event is chosen later, among the S&P 500 members on Day 0
     (``sample.share_class_rule``: linkprim P first, then the largest market cap on Day -1).
     Dropping J / N here would lose an event whose only member class is secondary.
  3. wrdsapps.ibcrsphist: date-valid on call_date, ``score`` <= 2, lowest score per permno
     (ties: latest ``sdate``). Never drop to an arbitrary row via ``drop_duplicates``.
     The table's open links end on its refresh date (``edate`` = 2025-12-31 in the pull), so
     rows with ``edate`` on that last date are treated as open; otherwise every 2026 event loses
     its ticker.
  4. Every step logs how many calls it loses, so link attrition is reported, not hidden.

The chain keeps **every candidate permno** (and its ticker) per call: S&P 500 membership on Day 0
needs the release date, which needs the ticker, which needs the permno, so the share-class choice
is made only after the release match (``events/build.py``).
"""

from __future__ import annotations

import pandas as pd

LINKPRIM_RANK = {"P": 0, "C": 1, "J": 2, "N": 3}
LINKTYPE_RANK = {"LC": 0, "LU": 1}


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("string"), errors="coerce").dt.normalize()


def valid_on(df: pd.DataFrame, date_col: str, lo: str, hi: str) -> pd.Series:
    """``lo <= date <= hi``, a missing bound being open."""
    d = df[date_col]
    return (df[lo].isna() | (df[lo] <= d)) & (df[hi].isna() | (d <= df[hi]))


def link_ciq_to_gvkey(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame,
                      date_col: str = "call_date_et") -> pd.DataFrame:
    """One row per matched (keydevid, companyid): ``keydevid``, ``companyid``, ``gvkey``.
    A keydevid can have two companies (joint calls, e.g. a merger announced with earnings)."""
    g = ciq_gvkey.copy()
    g["companyid"] = pd.to_numeric(g["companyid"]).astype("int64")
    g["startdate"], g["enddate"] = _date(g["startdate"]), _date(g["enddate"])
    if "primaryflag" not in g:
        g["primaryflag"] = 1
    m = calls[["keydevid", "companyid", date_col]].merge(
        g[["companyid", "gvkey", "startdate", "enddate", "primaryflag"]], on="companyid")
    m = m[valid_on(m, date_col, "startdate", "enddate")]
    m = m.sort_values(["keydevid", "companyid", "primaryflag", "startdate"],
                      ascending=[True, True, False, False], na_position="last")
    return (m.drop_duplicates(["keydevid", "companyid"])[["keydevid", "companyid", "gvkey"]]
             .reset_index(drop=True))


def link_gvkey_to_permno(events: pd.DataFrame, ccm: pd.DataFrame,
                         date_col: str = "call_date_et") -> pd.DataFrame:
    """All date-valid CCM candidates: one row per (keydevid, permno) with ``gvkey``, ``linkprim``,
    ``linktype`` (best link per pair: linkprim P > C > J > N, then LC over LU)."""
    c = ccm[ccm["linktype"].isin(list(LINKTYPE_RANK))].copy()
    c["linkdt"], c["linkenddt"] = _date(c["linkdt"]), _date(c["linkenddt"])
    m = events[["keydevid", "gvkey", date_col]].merge(
        c[["gvkey", "permno", "linktype", "linkprim", "linkdt", "linkenddt"]], on="gvkey")
    m = m[valid_on(m, date_col, "linkdt", "linkenddt")].dropna(subset=["permno"]).copy()
    m["permno"] = m["permno"].astype("int64")
    m["_p"] = m["linkprim"].map(LINKPRIM_RANK).fillna(9)
    m["_t"] = m["linktype"].map(LINKTYPE_RANK).fillna(9)
    m = m.sort_values(["keydevid", "permno", "_p", "_t"]).drop_duplicates(["keydevid", "permno"])
    return m[["keydevid", "gvkey", "permno", "linkprim", "linktype"]].reset_index(drop=True)


def link_permno_to_ibes(events: pd.DataFrame, iclink: pd.DataFrame,
                        date_col: str = "call_date_et", max_score: float = 2,
                        open_end: str | pd.Timestamp | None = "auto") -> pd.DataFrame:
    """Best I/B/E/S ticker per (keydevid, permno): ``ibes_ticker``, ``ibes_score``.

    ``open_end``: an ``edate`` on or after this date counts as open ("auto": the table's last
    ``edate``, its refresh date; None: take ``edate`` literally).
    """
    ic = iclink[pd.to_numeric(iclink["score"], errors="coerce") <= max_score].copy()
    ic = ic.dropna(subset=["permno", "ticker"])
    ic["permno"] = ic["permno"].astype("int64")
    ic["sdate"], ic["edate"] = _date(ic["sdate"]), _date(ic["edate"])
    if open_end is not None:
        end = ic["edate"].max() if open_end == "auto" else pd.Timestamp(open_end)
        ic.loc[ic["edate"] >= end, "edate"] = pd.NaT
    m = events[["keydevid", "permno", date_col]].merge(
        ic[["permno", "ticker", "score", "sdate", "edate"]], on="permno")
    m = m[valid_on(m, date_col, "sdate", "edate")]
    m = m.sort_values(["keydevid", "permno", "score", "sdate"], ascending=[True, True, True, False])
    m = m.drop_duplicates(["keydevid", "permno"])
    return (m.rename(columns={"ticker": "ibes_ticker", "score": "ibes_score"})
             [["keydevid", "permno", "ibes_ticker", "ibes_score"]].reset_index(drop=True))


def build_event_links(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame, ccm: pd.DataFrame,
                      iclink: pd.DataFrame, date_col: str = "call_date_et",
                      max_score: float = 2, open_end="auto"
                      ) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Chain the three links.

    ``calls`` may hold several companies per keydevid (joint calls): each is a candidate.
    Returns ``(candidates, attrition)``: one row per keydevid × candidate permno (companies with no
    gvkey or no permno keep one row with the missing ids, so nothing disappears), with ``gvkey``,
    ``permno``, ``linkprim``, ``linktype``, ``ibes_ticker``, ``ibes_score``, ``n_permno_candidates``;
    and ``(step, calls before, calls after)`` per link.
    """
    base = calls[["keydevid", "companyid", date_col]].drop_duplicates(["keydevid", "companyid"])
    gv = link_ciq_to_gvkey(base, ciq_gvkey, date_col)
    out = base.merge(gv, on=["keydevid", "companyid"], how="left")
    pm = link_gvkey_to_permno(out.dropna(subset=["gvkey"]), ccm, date_col)
    out = out.merge(pm, on=["keydevid", "gvkey"], how="left")
    out["permno"] = out["permno"].astype("Int64")
    has_p = out.dropna(subset=["permno"]).astype({"permno": "int64"})
    ib = link_permno_to_ibes(has_p, iclink, date_col, max_score, open_end)
    ib["permno"] = ib["permno"].astype("Int64")
    out = out.merge(ib, on=["keydevid", "permno"], how="left")
    out["n_permno_candidates"] = out.groupby("keydevid")["permno"].transform("nunique")

    def n(mask):
        return int(out.loc[mask, "keydevid"].nunique())
    n0 = int(base["keydevid"].nunique())
    n1 = n(out["gvkey"].notna())
    n2 = n(out["permno"].notna())
    n3 = n(out["ibes_ticker"].notna())
    attrition = [("CIQ companyid -> gvkey (date-valid)", n0, n1),
                 ("gvkey -> permno (CCM LU/LC, date-valid)", n1, n2),
                 (f"permno -> I/B/E/S ticker (score <= {max_score:g}, date-valid)", n2, n3)]
    return out.reset_index(drop=True), attrition
