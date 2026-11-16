"""
FinBERT feature extraction for earnings-call transcripts.

Collects all sentences across calls, scores them in large batches
through ProsusAI/finbert, then aggregates back to per-call features.

Usage:
    python3 finbert_features.py                    # score all years
    python3 finbert_features.py --years 2020 2021  # score specific years
    python3 finbert_features.py --batch-size 128   # adjust for memory/GPU

Output:  {BASE}/finbert_features.parquet
"""
import os, sys, glob, re, argparse, time, gc
import numpy as np
import pandas as pd
import torch
from datetime import datetime

DEFAULT_FOLDER = os.path.expanduser("~/earnings_sp500")

def log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def split_sentences(text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip()[:512] for s in sents if len(s.strip()) >= 10]

def extract_sections(row):
    prepared = row.get("prepared_text", "") or ""
    qa = row.get("qa_text", "") or ""
    answers = []
    for block in qa.split("\n\n"):
        if block.startswith("ANSWER:"):
            answers.append(block[7:].strip())
    return prepared, "\n".join(answers)

def aggregate(scores):
    if len(scores) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    pos = scores[:, 0]
    neg = scores[:, 1]
    net = pos - neg
    return (float(net.mean()),
            float((pos > neg).mean()),
            float((neg > pos).mean()),
            float(net.std()) if len(net) > 1 else 0.0,
            len(scores))

def score_bulk(classifier, sentences, batch_size):
    """Score all sentences in bulk, return Nx3 array of [pos, neg, neu] scores."""
    if not sentences:
        return np.empty((0, 3))
    all_scores = np.zeros((len(sentences), 3))
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        results = classifier(batch, truncation=True, max_length=512, batch_size=batch_size)
        for j, r in enumerate(results):
            col = label_map[r["label"]]
            all_scores[i + j, col] = r["score"]
            remaining = 1.0 - r["score"]
            if col == 0:
                all_scores[i + j, 2] = remaining
            elif col == 1:
                all_scores[i + j, 2] = remaining
            else:
                pass
        if (i + batch_size) % (batch_size * 50) == 0:
            log(f"  scored {min(i + batch_size, len(sentences)):,}/{len(sentences):,} sentences")
    return all_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=DEFAULT_FOLDER)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    base = args.folder
    text_dir = os.path.join(base, "text")

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"

    from transformers import pipeline as hf_pipeline
    log(f"loading ProsusAI/finbert on {device_name}...")
    classifier = hf_pipeline("text-classification", model="ProsusAI/finbert",
                             device=device)
    log("model loaded")

    text_files = sorted(glob.glob(f"{text_dir}/text_*.parquet"))
    if not text_files:
        log(f"no text files in {text_dir}"); sys.exit(1)

    if args.years:
        text_files = [f for f in text_files
                      if int(re.search(r"text_(\d{4})", f).group(1)) in args.years]

    all_features = []
    for tf in text_files:
        year = re.search(r"text_(\d{4})", tf).group(1)
        out_file = os.path.join(base, f"finbert_features_{year}.parquet")
        if os.path.exists(out_file):
            log(f"finbert {year}: loading cached {out_file}")
            all_features.append(pd.read_parquet(out_file))
            continue

        df = pd.read_parquet(tf)
        log(f"finbert {year}: {len(df):,} calls")
        t0 = time.time()

        # Collect all sentences with their call index and section tag
        prep_sentences, ans_sentences = [], []
        prep_call_ids, ans_call_ids = [], []

        for i, row in df.iterrows():
            prepared_text, answers_text = extract_sections(row)
            ps = split_sentences(prepared_text)
            ans = split_sentences(answers_text)
            prep_sentences.extend(ps)
            prep_call_ids.extend([i] * len(ps))
            ans_sentences.extend(ans)
            ans_call_ids.extend([i] * len(ans))

        log(f"  {year}: {len(prep_sentences):,} prepared sentences, {len(ans_sentences):,} answer sentences")

        # Score all sentences in bulk
        prep_scores = score_bulk(classifier, prep_sentences, args.batch_size)
        ans_scores = score_bulk(classifier, ans_sentences, args.batch_size)

        prep_call_ids = np.array(prep_call_ids)
        ans_call_ids = np.array(ans_call_ids)

        # Aggregate per call
        rows = []
        for i, row in df.iterrows():
            pmask = prep_call_ids == i
            amask = ans_call_ids == i
            ps = prep_scores[pmask] if pmask.any() else np.empty((0, 3))
            asc = ans_scores[amask] if amask.any() else np.empty((0, 3))

            p_mean, p_pos, p_neg, p_std, p_n = aggregate(ps)
            a_mean, a_pos, a_neg, a_std, a_n = aggregate(asc)

            rows.append({
                "transcriptid": row["transcriptid"],
                "fb_sent_prepared": p_mean,
                "fb_sent_answers": a_mean,
                "fb_pos_pct_prepared": p_pos,
                "fb_neg_pct_prepared": p_neg,
                "fb_pos_pct_answers": a_pos,
                "fb_neg_pct_answers": a_neg,
                "fb_sent_std_prepared": p_std,
                "fb_sent_std_answers": a_std,
                "fb_n_sent_prepared": p_n,
                "fb_n_sent_answers": a_n,
            })

        year_df = pd.DataFrame(rows)
        year_df["fb_sent_gap"] = year_df["fb_sent_prepared"] - year_df["fb_sent_answers"]
        year_df.to_parquet(out_file, index=False)
        all_features.append(year_df)
        elapsed = time.time() - t0
        total_sents = len(prep_sentences) + len(ans_sentences)
        log(f"finbert {year}: done in {elapsed:.0f}s ({total_sents/elapsed:.0f} sent/s), saved {out_file}")
        del df, rows, year_df, prep_sentences, ans_sentences, prep_scores, ans_scores
        gc.collect()

    features = pd.concat(all_features, ignore_index=True)

    # Compute deltas vs prior 4-call average
    panel = pd.read_parquet(f"{base}/event_panel.parquet",
                            columns=["transcriptid", "permno", "call_date"])
    features = features.merge(panel, on="transcriptid", how="left")
    features = features.sort_values(["permno", "call_date"])
    for col in ["fb_sent_prepared", "fb_sent_answers"]:
        prev = features.groupby("permno")[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        features[f"{col}_chg"] = features[col] - prev
    features = features.drop(columns=["permno", "call_date"])
    features = features.sort_index()

    out = os.path.join(base, "finbert_features.parquet")
    features.to_parquet(out, index=False)
    log(f"saved {out}: {len(features):,} calls, {features.shape[1]} columns")
    log("columns: " + ", ".join(features.columns.tolist()))

if __name__ == "__main__":
    main()
