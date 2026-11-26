"""
Score FinBERT for a given list of calls (same sentence split, scoring and aggregation as
it_finbert_features.py). Writes per-call features without the *_chg columns; those need the
whole panel and are computed after merging (see cd_merge_finbert.py).

Usage: python finbert_score_calls.py --text-dir text --ids ids.csv --out fb_scored.parquet [--batch-size 128]
"""
import argparse, glob, time
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
import it_finbert_features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--ids", required=True, help="csv with a transcriptid column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    want = set(pd.read_csv(a.ids)["transcriptid"])
    dev = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    F.log(f"{len(want)} calls to score on {dev}")
    clf = pipeline("text-classification", model="ProsusAI/finbert", device=dev)
    rows = []
    for tf in sorted(glob.glob(f"{a.text_dir}/text_*.parquet")):
        df = pd.read_parquet(tf)
        df = df[df["transcriptid"].isin(want)].reset_index(drop=True)
        if df.empty:
            continue
        t0, ps, asn, pid, aid = time.time(), [], [], [], []
        for i, row in df.iterrows():
            p, ans = F.extract_sections(row)
            s1, s2 = F.split_sentences(p), F.split_sentences(ans)
            ps += s1; pid += [i] * len(s1); asn += s2; aid += [i] * len(s2)
        P, A = F.score_bulk(clf, ps, a.batch_size), F.score_bulk(clf, asn, a.batch_size)
        pid, aid = np.array(pid), np.array(aid)
        for i, row in df.iterrows():
            pm, pp, pn, psd, pc = F.aggregate(P[pid == i] if (pid == i).any() else np.empty((0, 3)))
            am, ap_, an, asd, ac = F.aggregate(A[aid == i] if (aid == i).any() else np.empty((0, 3)))
            rows.append(dict(transcriptid=row["transcriptid"], fb_sent_prepared=pm, fb_sent_answers=am,
                             fb_pos_pct_prepared=pp, fb_neg_pct_prepared=pn, fb_pos_pct_answers=ap_,
                             fb_neg_pct_answers=an, fb_sent_std_prepared=psd, fb_sent_std_answers=asd,
                             fb_n_sent_prepared=pc, fb_n_sent_answers=ac))
        n = len(ps) + len(asn)
        F.log(f"{tf[-12:-8]}: {len(df)} calls, {n:,} sentences, {n / max(time.time() - t0, 1):.0f} sent/s")
    r = pd.DataFrame(rows)
    r["fb_sent_gap"] = r["fb_sent_prepared"] - r["fb_sent_answers"]
    r.to_parquet(a.out, index=False)
    F.log(f"saved {a.out}: {len(r)} calls")


if __name__ == "__main__":
    main()
