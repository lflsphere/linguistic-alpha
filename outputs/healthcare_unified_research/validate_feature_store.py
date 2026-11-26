"""Reload the real Parquet features and independently check their sentence aggregates."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extract_finbert import FEATURES, HERE, SECTIONS, STORE, json_write, verify


def accumulate(parquet):
    groups = {}
    for batch in parquet.iter_batches(batch_size=32768):
        frame = batch.to_pandas()
        if not frame.top_label.isin(["positive", "negative", "neutral"]).all():
            raise ValueError("Invalid sentence label")
        if not frame.confidence.between(1 / 3 - 1e-6, 1).all():
            raise ValueError("Invalid sentence probability")
        if not frame.token_count_before_truncation.ge(2).all():
            raise ValueError("Invalid tokenizer length")
        sign = frame.top_label.map({"positive": 1, "negative": -1, "neutral": 0})
        frame["net"] = sign * frame.confidence.astype(float)
        frame["positive"] = sign.eq(1).astype(int)
        frame["negative"] = sign.eq(-1).astype(int)
        for key, group in frame.groupby(["transcriptid", "section"], sort=False):
            current = groups.setdefault(key, [[], [], 0, 0])
            current[0].extend(group.sentence_index.tolist())
            current[1].extend(group.net.tolist())
            current[2] += int(group.positive.sum())
            current[3] += int(group.negative.sum())
    return groups


def check(archive=STORE, out=HERE):
    features = verify(archive, require_complete=False)
    parquet = pq.ParquetFile(archive / "sentence_scores.parquet")
    groups = accumulate(parquet)
    if not set(tid for tid, _ in groups).issubset(features.transcriptid):
        raise ValueError("Orphaned sentence rows")
    token_caps = 0
    for batch in parquet.iter_batches(batch_size=32768, columns=["token_count_before_truncation"]):
        token_caps += int((batch.column(0).to_numpy() > 512).sum())
    for row in features.itertuples(index=False):
        for section in SECTIONS:
            order, net, positive, negative = groups.get((row.transcriptid, section), ([], [], 0, 0))
            count = len(net)
            if order != list(range(1, count + 1)) or count != getattr(row, f"fb_n_sent_{section}"):
                raise ValueError("Noncontiguous sentence ordinals or mismatched counts")
            expected = ((np.mean(net), positive / count, negative / count, np.std(net))
                        if count else (np.nan,) * 4)
            actual = [getattr(row, base + "_" + section)
                      for base in ("fb_sent", "fb_pos_pct", "fb_neg_pct", "fb_sent_std")]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(features.fb_sent_gap, features.fb_sent_prepared - features.fb_sent_answers,
                               atol=1e-12, rtol=0, equal_nan=True)
    issuer_history = {key: rows.sort_values(["clock", "transcriptid"]) for key, rows in
                      features.dropna(subset=["permno", "clock"]).groupby("permno")}
    for row in features.itertuples(index=False):
        history = None
        if pd.notna(row.permno) and pd.notna(row.clock):
            issuer = issuer_history[row.permno]
            history = issuer.loc[issuer.clock.lt(row.clock)].tail(4)
        for section in SECTIONS:
            level = f"fb_sent_{section}"
            expected = np.nan if history is None else getattr(row, level) - history[level].mean()
            np.testing.assert_allclose(getattr(row, level + "_chg"), expected, atol=1e-12, rtol=0, equal_nan=True)
    manifest = json.loads((archive / "feature_manifest.json").read_text(encoding="utf-8"))
    summary = {"call_rows": len(features), "feature_columns": len(FEATURES),
               "sentence_rows": parquet.metadata.num_rows, "sentence_groups": len(groups),
               "token_truncations": token_caps, "all_aggregates_reconciled": True,
               "complete_requested": manifest["completed_requested_extraction"],
               "model_inference_invoked": False}
    json_write(out / "parquet_validation.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    check()
