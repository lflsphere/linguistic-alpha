"""Column contracts for every table the pipeline produces.

Each stage validates its output with :func:`validate` so a broken upstream join fails loudly
instead of silently producing NaN-filled features. Keys are listed first.
"""

from __future__ import annotations

import pandas as pd

# One row per transcript component (speaker turn), canonical form of ciq.ciqtranscriptcomponent.
COMPONENTS = {
    "transcriptid": "int64",
    "componentorder": "int64",
    "section": "string",          # operator | prepared | question | answer | other
    "speaker_id": "Int64",        # ciq transcriptpersonid
    "speaker_role": "string",     # executive | analyst | operator | unknown
    "text": "string",
}

# One row per question with its management answer(s) concatenated.
QA_PAIRS = {
    "transcriptid": "int64",
    "pair_id": "int64",
    "question_speaker_id": "Int64",
    "question": "string",
    "answer": "string",
    "n_answer_turns": "int64",
}

# One row per earnings call (event), after linking and universe filters.
EVENTS = {
    "keydevid": "int64",
    "transcriptid": "int64",
    "companyid": "int64",
    "gvkey": "string",
    "permno": "int64",
    "ibes_ticker": "string",
    "call_ts_utc": "datetime64[ns, UTC]",
    "call_date_et": "datetime64[ns]",
    "day0": "datetime64[ns]",
}

TARGETS = {
    "keydevid": "int64",
    "car_2_20": "float64",
    "car_2_5": "float64",
    "car_2_60": "float64",
    "vol_post_2_20": "float64",
    "vol_ratio_2_20": "float64",  # scope doc definition: post vol / pre vol
}

CONTROLS = {
    "keydevid": "int64",
    "sue": "float64",             # surprise / price (scope doc)
    "sue_std": "float64",         # surprise / forecast stdev (legacy notebook)
    "size": "float64",
    "bm": "float64",
    "mom": "float64",
    "car_0_1": "float64",
    "vol_pre": "float64",
    "n_analysts": "float64",
    "dispersion": "float64",
}


def validate(df: pd.DataFrame, schema: dict[str, str], name: str, key: str | None = None) -> None:
    """Raise if required columns are missing or the key is not unique."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if key is not None and df[key].duplicated().any():
        n = int(df[key].duplicated().sum())
        raise ValueError(f"{name}: {n} duplicate values in key '{key}'")
