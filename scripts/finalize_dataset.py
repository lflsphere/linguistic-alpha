#!/usr/bin/env python3
"""Join computed language features and write aligned Model A/B event tables.

The WRDS inputs and interim files remain local and ignored by git. Run after
``build_wrds_dataset.py`` and ``01_build_features.py`` have completed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from linguistic_alpha.config import load_config
from linguistic_alpha.data.universe import apply_exclusions, sp500_member_on
from linguistic_alpha.io import read_table, write_table


LANGUAGE_COLUMNS = {
    "TONE_P", "TONE_Q", "TONE_M", "TONE_R", "TONE_P_PW", "TONE_Q_PW",
    "TONE_GAP", "TONE_P_FB", "TONE_Q_FB", "UNC", "UNC_P", "UNC_Q",
    "HEDGE", "HEDGE_P", "HEDGE_Q", "REL", "REL_JAC", "EVADE", "NONANS",
    "TONE_R", "DIV_QP", "DIV_PR", "DIV_PR_TONE", "NOVEL", "dTONE",
    "dUNC", "dEVADE", "words_prep_parsed", "words_ans_parsed", "n_pairs",
    "n_q_blocks", "n_a_blocks", "qa_truncated", "prep_truncated", "boiler_share",
    "words_prepared", "words_question", "words_answer", "n_questions", "n_answers",
    "avg_answer_words", "has_transcript",
}
TEXT_COLUMNS = {
    "prepared_text", "qa_text", "question_text", "answer_text", "press_release_text",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--features", type=Path)
    args = ap.parse_args()
    cfg = load_config()
    panel_path = args.panel or cfg.paths.processed / "event_panel_all.parquet"
    feature_path = args.features or cfg.paths.processed / "language_features.csv"
    panel = read_table(panel_path.parent, panel_path.stem)
    features = pd.read_csv(feature_path, low_memory=False)
    membership = read_table(cfg.paths.raw, "crsp_msp500list")
    panel["in_sp500"] = sp500_member_on(panel, membership)
    panel = apply_exclusions(panel, cfg.raw)
    write_table(panel, panel_path.parent, panel_path.stem)

    # Match the configured estimation sample and preserve the exact same rows
    # in the baseline and language-enhanced data sets.
    start = pd.Timestamp(cfg["sample"]["start"])
    events = panel[panel["included"].fillna(False) &
                   (pd.to_datetime(panel["call_date_et"]) >= start)].copy()
    drop = [c for c in features if c != "transcriptid" and c in events]
    features = features.drop(columns=drop)
    full = events.merge(features, on="transcriptid", how="left", validate="one_to_one")
    full = full.drop(columns=[c for c in TEXT_COLUMNS if c in full])
    baseline = full.drop(columns=[c for c in LANGUAGE_COLUMNS if c in full])

    out = cfg.paths.processed
    write_table(full, out, "model_table_with_language")
    write_table(baseline, out, "model_table_baseline")
    # Keep the established project filename as the Model B alias.
    write_table(full, out, "model_table")
    attrition = panel["exclusion_reason"].fillna("included").value_counts().rename_axis(
        "reason").reset_index(name="events")
    attrition.to_csv(out / "sample_attrition.csv", index=False)
    print(f"Model A: {len(baseline):,} rows x {baseline.shape[1]} columns")
    print(f"Model B: {len(full):,} rows x {full.shape[1]} columns")
    for target in ("car_2_5", "car_2_20", "car_2_60"):
        if target in full:
            print(f"{target}: {full[target].notna().sum():,} observed labels")
    present = sorted(LANGUAGE_COLUMNS.intersection(full.columns))
    print(f"Language fields present: {', '.join(present)}")
    print(f"Same event IDs: {baseline['keydevid'].equals(full['keydevid'])}")


if __name__ == "__main__":
    main()
