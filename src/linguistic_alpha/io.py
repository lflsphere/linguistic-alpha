"""Read/write helpers for the raw -> interim -> processed data layers.

Raw files are named after the WRDS table they mirror (``crsp_dsf``, ``ibes_statsumu_epsus``,
``ciq_transcript_component``...). Large tables may be split by year: ``<name>_<yyyy>.parquet``.
Both ``.parquet`` and ``.csv`` exports are accepted so web-UI downloads work too.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(folder: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``name`` from ``folder``: a single file, or all ``name_<yyyy>`` yearly shards."""
    for ext in (".parquet", ".csv"):
        single = folder / f"{name}{ext}"
        if single.exists():
            return _read(single, columns)
    shards = sorted(folder.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet")) + sorted(
        folder.glob(f"{name}_[0-9][0-9][0-9][0-9].csv")
    )
    if not shards:
        raise FileNotFoundError(f"No file or yearly shards for '{name}' in {folder}")
    return pd.concat((_read(p, columns) for p in shards), ignore_index=True)


def write_table(df: pd.DataFrame, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


def _read(path: Path, columns: list[str] | None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)
