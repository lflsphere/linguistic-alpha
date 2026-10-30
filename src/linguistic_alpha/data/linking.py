"""Identifier linking: CIQ companyid -> gvkey -> permno -> I/B/E/S ticker, valid on the call date.

Status: specified, not implemented (waits on the CRSP/Compustat/IBES pull).

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


def link_ciq_to_gvkey(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def link_gvkey_to_permno(events: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def link_permno_to_ibes(events: pd.DataFrame, iclink: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def build_event_links(calls: pd.DataFrame, ciq_gvkey: pd.DataFrame, ccm: pd.DataFrame,
                      iclink: pd.DataFrame) -> pd.DataFrame:
    """Chain the three links; return one row per keydevid plus attrition diagnostics."""
    raise NotImplementedError
