"""
Merge FinBERT scores for calls added by the point-in-time GICS sample into finbert_features.parquet,
check the re-scored overlap calls agree, and recompute the *_chg features over the new panel
(change vs. the mean of the firm's previous 4 sample calls, as in it_finbert_features.py).

Usage: python cd_merge_finbert.py --folder cd_results --new fb_pit_new.parquet --ids ids.csv
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="cd_results")
ap.add_argument("--new", required=True)
ap.add_argument("--ids", required=True, help="ids.csv from the Colab input (check column marks overlap calls)")
a = ap.parse_args()

old = pd.read_parquet(f"{a.folder}/finbert_features.parquet").drop(columns=["fb_sent_prepared_chg", "fb_sent_answers_chg"])
new = pd.read_parquet(a.new)
chk = set(pd.read_csv(a.ids).query("check")["transcriptid"].astype(int))
both = old[old.transcriptid.isin(chk)].set_index("transcriptid").sort_index()
again = new[new.transcriptid.isin(chk)].set_index("transcriptid").sort_index()[both.columns]
diff = (both - again).abs().max()
print(f"overlap check on {len(both)} calls, max abs diff per feature:\n{diff.round(5).to_string()}")
assert diff.max() < 0.02, "re-scored calls disagree with the original FinBERT run"

fb = pd.concat([old, new[~new.transcriptid.isin(old.transcriptid)]], ignore_index=True)
panel = pd.read_parquet(f"{a.folder}/model_table_opt1.parquet", columns=["transcriptid", "permno", "call_date"])
fb = fb[fb.transcriptid.isin(panel.transcriptid)].merge(panel, on="transcriptid")
fb = fb.sort_values(["permno", "call_date"])
for c in ["fb_sent_prepared", "fb_sent_answers"]:
    prev = fb.groupby("permno")[c].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    fb[f"{c}_chg"] = fb[c] - prev
fb = fb.drop(columns=["permno", "call_date"]).reset_index(drop=True)
missing = set(panel.transcriptid) - set(fb.transcriptid)
print(f"merged: {len(fb):,} calls ({len(new) - len(both)} new), sample {len(panel):,}, without scores {len(missing)}")
fb.to_parquet(f"{a.folder}/finbert_features.parquet", index=False)
fb.to_csv(f"{a.folder}/finbert_features.csv", index=False)
