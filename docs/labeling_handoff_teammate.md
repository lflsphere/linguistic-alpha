# Handoff: second labeller (labeller B, the "teammate")

You are an independent second labeller. You check how reliable labeller A's labels are
(`docs/labeling_handoff.md`). Working directory: `linguistic-alpha/`. Python: `.venv/bin/python`.

## Blindness rules (most important)
- **Never open** `data/qa_labels.csv`, or labeller A's notes,
  logs or report, until your own labels are saved. Opening them first invalidates the
  agreement measure.
- Do not look at returns, features, relevance scores or model output.
- Use only the label scheme in `docs/pipeline.md` ("Q&A labels for F3"), repeated below. Do not reread labeller A's handoff for hints on how they decided.

## Your sample
Wait until `data/qa_label_sample_full.csv` exists (labeller A builds it). You can label while
labeller A labels; for procedure step 2, wait until `data/qa_labels.csv` exists (check that it
exists; do not open it before your labels are saved). Then draw your
overlap set with seed `7`: 90 pairs from `data/qa_label_sample_full.csv`, stratified by year
(at least 6 per year) and covering every GICS sector.

Use the sample file only (its label columns are empty). Write
`data/qa_labels_b_sample.csv` (the ids you drew), then label it.

## Labelling rules
- `evasive = 1` when the answer does not give the specific information the analyst asked for:
  it changes the topic, falls back on generic talking points, answers an easier question, or
  covers only part of a multi-part question while skipping the main ask.
- `non_answer = 1` when management explicitly declines or cannot answer: "we don't disclose",
  "too early to say", "we don't guide to that", "we'll follow up offline".
- A refusal is usually both labels. A frank answer gets `evasive = 0`, even with bad news or when it is brief.
- Judge each pair only from its question and answer.
- Text may be cut off: judge only what is there and write `truncated` in `notes`.
- Give a one-line reason in `notes` for every 1 and for every borderline case
  (`borderline: ...`).

## Procedure
1. Label one pair at a time and save `data/qa_labels_b.csv`
   (`pair_id, evasive, non_answer, notes`; integer 0/1 values, no blanks).
2. Only after saving, merge with `data/qa_labels.csv` on `pair_id` and compute, for each label:
   ```python
   from sklearn.metrics import cohen_kappa_score
   ```
   Report Cohen's kappa, raw agreement and each labeller's base rate. Write
   `outputs/validation/label_interrater.csv`.
3. List every disagreement in `outputs/validation/label_disagreements.csv`, with both notes
   and your proposed resolved label.
4. Do not edit labeller A's files. Changes after resolving disagreements are a team decision.

## Report back
- Kappa per label. Target about 0.6 or more; below 0.4, the rules need rewording
  before F3 is used.
- Disagreement patterns, e.g. partial answers or forward-looking deflections, and a suggested
  rule change for each.
- Caveat: two LLM labellers can share the same biases. Have one human check at least the
  disagreements plus 20 random pairs.
