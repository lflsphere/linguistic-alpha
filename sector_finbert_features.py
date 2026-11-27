"""
Sector — FinBERT feature extraction for earnings-call transcripts.

Filters to the chosen sector's transcripts only (from ~/earnings_{sector}/ model tables),
scores sentences with ProsusAI/finbert, aggregates to per-call features.

Usage:
    python3 sector_finbert_features.py --sector {it,health,comm}                    # all years
    python3 sector_finbert_features.py --sector it --batch-size 128   # adjust for GPU memory

Output:  ~/earnings_{sector}/finbert_features.parquet
"""
import os, sys, glob, re, argparse, time, gc
import numpy as np
import pandas as pd
import torch
from sector_config import add_sector_arg, get_sector
from datetime import datetime

SP500_FOLDER = os.path.expanduser("~/earnings_sp500")


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
            if col in (0, 1):
                all_scores[i + j, 2] = remaining
        if (i + batch_size) % (batch_size * 50) == 0:
            log(f"  scored {min(i + batch_size, len(sentences)):,}/{len(sentences):,} sentences")
    return all_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500-folder", default=SP500_FOLDER)
    add_sector_arg(ap)
    ap.add_argument("--sector-folder", default=None, help="default: ~/earnings_{sector}")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    sec = get_sector(args)
    sector_folder = args.sector_folder or sec.folder

    text_dir = os.path.join(args.sp500_folder, "text")

    # Get sector transcript IDs from both model tables
    sector_tids = set()
    for f in ["model_table_opt1.parquet", "model_table_opt2.parquet"]:
        fp = os.path.join(sector_folder, f)
        if os.path.exists(fp):
            df = pd.read_parquet(fp, columns=["transcriptid"])
            sector_tids.update(df["transcriptid"].values)
    log(f"{sec.short} transcript IDs: {len(sector_tids):,}")

    if not sector_tids:
        log(f"ERROR: no {sec.short} transcripts found in {sector_folder}"); sys.exit(1)

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

    all_features = []
    total_scored = 0

    for tf in text_files:
        year = re.search(r"text_(\d{4})", tf).group(1)
        df = pd.read_parquet(tf)
        df_sec = df[df["transcriptid"].isin(sector_tids)].copy()
        if len(df_sec) == 0:
            log(f"finbert {year}: 0 {sec.short} calls, skipping")
            continue

        log(f"finbert {year}: {len(df_sec):,} {sec.short} calls (of {len(df):,} total)")
        t0 = time.time()

        prep_sentences, ans_sentences = [], []
        prep_call_ids, ans_call_ids = [], []

        for i, row in df_sec.iterrows():
            prepared_text, answers_text = extract_sections(row)
            ps = split_sentences(prepared_text)
            ans = split_sentences(answers_text)
            prep_sentences.extend(ps)
            prep_call_ids.extend([i] * len(ps))
            ans_sentences.extend(ans)
            ans_call_ids.extend([i] * len(ans))

        log(f"  {year}: {len(prep_sentences):,} prepared sentences, {len(ans_sentences):,} answer sentences")

        prep_scores = score_bulk(classifier, prep_sentences, args.batch_size)
        ans_scores = score_bulk(classifier, ans_sentences, args.batch_size)

        prep_call_ids = np.array(prep_call_ids)
        ans_call_ids = np.array(ans_call_ids)

        rows = []
        for i, row in df_sec.iterrows():
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
        all_features.append(year_df)
        elapsed = time.time() - t0
        n_sents = len(prep_sentences) + len(ans_sentences)
        total_scored += n_sents
        log(f"finbert {year}: done in {elapsed:.0f}s ({n_sents/max(elapsed,1):.0f} sent/s)")
        del df, df_sec, rows, year_df, prep_sentences, ans_sentences, prep_scores, ans_scores
        gc.collect()

    if not all_features:
        log("ERROR: no features extracted"); sys.exit(1)

    features = pd.concat(all_features, ignore_index=True)

    # Compute deltas vs prior 4-call average
    panel = pd.read_parquet(f"{args.sp500_folder}/event_panel.parquet",
                            columns=["transcriptid", "permno", "call_date"])
    features = features.merge(panel, on="transcriptid", how="left")
    features = features.sort_values(["permno", "call_date"])
    for col in ["fb_sent_prepared", "fb_sent_answers"]:
        prev = features.groupby("permno")[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        features[f"{col}_chg"] = features[col] - prev
    features = features.drop(columns=["permno", "call_date"])

    out = os.path.join(sector_folder, "finbert_features.parquet")
    features.to_parquet(out, index=False)
    log(f"saved {out}: {len(features):,} calls, {features.shape[1]} columns")
    log(f"columns: {', '.join(features.columns.tolist())}")
    log(f"total sentences scored: {total_scored:,}")


if __name__ == "__main__":
    main()
