# Q-A labeling guide
File: data/qa_label_sample.csv (150 pairs, stratified by relevance quintile, dev period only).
Fill two binary columns, then save as data/qa_labels.csv:
* evasive = 1 if the answer does not address the specific information the analyst requested
  (topic shift, generic talking points, answering an easier question).
* non_answer = 1 if management explicitly declines or is unable to answer
  (refuses, does not disclose, too early, will follow up offline).
Label without looking at returns or model scores. Double-label ~30 pairs to report inter-rater kappa.
Then rerun 01_build_features.py (calibrates EVADE_CLF / NONANS_CLF) and 02_validate.py (agreement).
