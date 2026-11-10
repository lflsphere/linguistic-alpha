#!/usr/bin/env python3
"""Score included call transcripts with ProsusAI/finbert and add tone features."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from linguistic_alpha.config import load_config
from linguistic_alpha.io import read_table, write_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("feature_builder", ROOT / "scripts/01_build_features.py")
feature_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_builder)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="score only the first N calls (smoke runs)")
    args = ap.parse_args()
    cfg = load_config()
    needed = ["transcriptid", "call_date_et", "day0", "firm", "ticker", "included",
              "prepared_text", "qa_text"]
    panel = read_table(cfg.paths.processed, "event_panel_all", columns=needed)
    panel["call_date_et"] = pd.to_datetime(panel["call_date_et"], errors="coerce")
    panel = panel[panel["included"].fillna(False) &
                  (panel["call_date_et"] >= pd.Timestamp(cfg["sample"]["start"]))].copy()
    panel = panel.drop_duplicates("transcriptid").sort_values("call_date_et")
    if args.limit:
        panel = panel.head(args.limit)

    prepared, answers = {}, {}
    for r in panel.itertuples(index=False):
        tid = int(r.transcriptid)
        prepared_text, _ = feature_builder.clean_prepared(r.prepared_text or "")
        _, answer_list = feature_builder.parse_qa(r.qa_text or "", False)
        prepared[tid] = _sample_sentences(prepared_text)
        answers[tid] = _sample_sentences(" ".join(answer_list))

    feature_builder.C.CACHE_DIR = cfg.paths.processed / "finbert_cache"
    tone_p = feature_builder.finbert_tone(prepared, "prepared_sample32")
    tone_q = feature_builder.finbert_tone(answers, "answers_sample32")
    scores = pd.DataFrame({"transcriptid": panel["transcriptid"].astype("int64").to_numpy()})
    scores["TONE_P_FB"] = scores["transcriptid"].map(tone_p)
    scores["TONE_Q_FB"] = scores["transcriptid"].map(tone_q)
    scores["TONE_M_FB"] = scores[["TONE_P_FB", "TONE_Q_FB"]].mean(axis=1)
    ordered = panel[["transcriptid", "firm", "call_date_et"]].merge(scores, on="transcriptid")
    ordered = ordered.sort_values(["firm", "call_date_et"])
    prior = ordered.groupby("firm")["TONE_M_FB"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).mean())
    ordered["dTONE_FB"] = ordered["TONE_M_FB"] - prior

    base = read_table(cfg.paths.processed, "model_table_with_language")
    result = base.merge(ordered.drop(columns=["firm", "call_date_et"]),
                        on="transcriptid", how="left", validate="one_to_one")
    write_table(result, cfg.paths.processed, "model_table_with_language_finbert")
    (cfg.paths.processed / "finbert_run.json").write_text(
        '{"model":"ProsusAI/finbert","sampling":"up to 32 evenly spaced sentences per section",'
        '"device":"MPS when available"}\n')
    print(f"FinBERT scores: {len(scores):,} calls")
    for c in ("TONE_P_FB", "TONE_Q_FB", "dTONE_FB"):
        print(f"{c}: {result[c].notna().sum():,} nonmissing rows")
    print(f"Wrote {cfg.paths.processed / 'model_table_with_language_finbert.parquet'}")


def _sample_sentences(text: str, maximum: int = 32) -> str:
    """Use an evenly spaced section sample to keep whole-universe inference feasible."""
    sentences = [s for s in feature_builder.SENT_RE.split(text) if len(s.split()) >= 4]
    if len(sentences) > maximum:
        indices = np.linspace(0, len(sentences) - 1, maximum, dtype=int)
        sentences = [sentences[i] for i in indices]
    return " ".join(sentences)


if __name__ == "__main__":
    main()
