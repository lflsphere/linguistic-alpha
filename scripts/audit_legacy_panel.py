"""Audit the legacy notebook output (panel.xlsx) for issues visible without raw WRDS data.

Usage:  python scripts/audit_legacy_panel.py [path/to/panel.xlsx]

Each check prints a count and example rows; findings feed docs/data_audit.md. Checks that
need raw CRSP/IBES (recomputing CAR, SUE on the unadjusted basis) live in the rebuild plan.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from linguistic_alpha.config import load_config
from linguistic_alpha.events.timing import NY, call_timestamp_utc


def n_decimals(x: pd.Series) -> pd.Series:
    s = x.dropna().map(lambda v: f"{v:.6f}".rstrip("0").split(".")[-1])
    return s.str.len().reindex(x.index)


def check(title: str, mask: pd.Series, df: pd.DataFrame, cols: list[str]) -> None:
    n = int(mask.sum())
    print(f"\n## {title}: {n} / {len(df)} rows")
    if n:
        print(df.loc[mask, cols].head(5).to_string(index=False))


def main(path: str | None = None) -> None:
    path = path or load_config().paths.legacy_panel
    df = pd.read_excel(path, dtype={"gvkey": str})
    print(f"# Legacy panel audit: {path}\n{len(df)} calls, {df['ticker'].nunique()} tickers, "
          f"{df['call_date'].min():%Y-%m-%d} .. {df['call_date'].max():%Y-%m-%d}")
    id_cols = ["ticker", "call_date"]

    # 1. Split-adjusted I/B/E/S rounding: consensus stored at 2 decimals vs. finer actuals
    rounded = (n_decimals(df["consensus_eps"]) <= 2) & (n_decimals(df["actual_eps"]) > 2)
    check("IBES rounding: consensus <=2 decimals while actual is finer", rounded, df,
          id_cols + ["actual_eps", "consensus_eps", "eps_surprise"])
    check("IBES rounding: consensus == 0 with non-zero actual",
          (df["consensus_eps"] == 0) & (df["actual_eps"] != 0), df,
          id_cols + ["actual_eps", "consensus_eps"])

    # 2. SUE scaled by forecast stdev -> undefined whenever stdev is 0 / missing
    check("SUE missing (stdev-scaled definition)", df["sue"].isna(), df,
          id_cols + ["eps_surprise", "dispersion", "n_analysts"])
    check("Dispersion relative to tiny consensus (|consensus| < 0.05)",
          df["consensus_eps"].abs() < 0.05, df, id_cols + ["consensus_eps", "dispersion"])

    # 3. Timing: legacy Day 0 used the UTC calendar date
    ts = call_timestamp_utc(df["call_date"], df["call_time_utc"])
    et = ts.dt.tz_convert(NY)
    check("Call time missing (legacy treated as before close)", ts.isna(), df,
          id_cols + ["call_time_utc"])
    check("NY call date differs from UTC call date",
          et.dt.tz_localize(None).dt.normalize() != df["call_date"].dt.normalize(), df,
          id_cols + ["call_time_utc", "day0"])
    gap = np.busday_count(df["call_date"].values.astype("datetime64[D]"),
                          df["day0"].values.astype("datetime64[D]"))
    check("Day 0 more than 1 business day after call date (holiday or error)",
          pd.Series(gap > 1, index=df.index), df, id_cols + ["call_time_utc", "day0"])

    # 4. Transcript structure
    excel_cap = 32_767   # max characters per Excel cell: longer text was silently cut
    check("Text truncated at Excel cell limit (prepared or Q&A)",
          (df["prepared_text"].str.len() >= excel_cap) | (df["qa_text"].str.len() >= excel_cap),
          df, id_cols + ["words_prepared", "words_answers"])
    check("Q&A starts with ANSWER (section mislabel / IR hand-off)",
          df["qa_text"].str.lstrip().str.startswith("ANSWER:"), df,
          id_cols + ["n_questions", "n_answers"])
    order = df.sort_values("call_date").groupby("permno")["transcriptid"].diff() < 0
    check("transcriptid not increasing with call date (re-collected versions)",
          order.reindex(df.index).fillna(False), df, id_cols + ["transcriptid"])

    # 5. Coverage: missing quarters per firm
    gaps = df.sort_values("call_date").groupby("permno")["call_date"].diff().dt.days
    check("Gap between consecutive calls > 120 days", gaps.reindex(df.index) > 120, df,
          id_cols + ["headline"])

    # 6. Definition mismatches vs. the scope document (informational)
    print("\n## Definitions differing from the scope doc")
    print("- sue = surprise / stdev (scope: surprise / price)")
    print("- vol_post_2_20 is a level (scope: ratio to pre-event vol); "
          f"median post/pre ratio here = {(df['vol_post_2_20'] / df['vol_pre']).median():.2f}")
    print(f"- missing targets: car_2_20={df['car_2_20'].isna().sum()}, "
          f"car_2_60={df['car_2_60'].isna().sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
