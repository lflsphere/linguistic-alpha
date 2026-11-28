"""Candidate 3: LLM judge on anonymised Q&A pairs, with labeller A's rules (docs/labeling_handoff.md) as prompt.

    .venv/bin/python scripts/f3/llm_judge.py estimate          # token / cost estimate for all pairs, no API call
    .venv/bin/python scripts/f3/llm_judge.py run [--model M]    # the 414 labelled pairs only

`run` needs the `anthropic` SDK and a credential already in the environment; without one it exits and
does nothing (agent G's brief: never ask for or store a key). Answers are cached per pair_id in
data/interim/f3_cache/llm_judge_<model>.jsonl, so a rerun only scores what is missing. It never scores
pairs outside the labelled build sample: scaling to the 654,447 pairs needs the user's approval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3common as G  # noqa: E402

SYSTEM = """You label question-and-answer pairs from company earnings calls. Firm names and years are masked.
Judge each pair only from its question and answer.

evasive = 1 when the answer does not give the specific information the analyst asked for: it changes the
topic, falls back on generic talking points, answers an easier question, or covers only part of a
multi-part question while skipping the main ask.
evasive = 0 when the answer engages with the question, even if the news is bad or the answer is brief.

non_answer = 1 when management explicitly declines or cannot answer: "we don't disclose", "too early to
say", "we don't guide to that", "we'll follow up offline".
A refusal is usually both labels. An answer can be evasive without being a refusal.

Text may be cut off: judge only what is there.
Reply with JSON only: {"p_evasive": <0..1>, "p_non_answer": <0..1>, "reason": "<one short line>"}
where each p is your probability that the label is 1."""

USER = "Question:\n{q}\n\nAnswer:\n{a}"
DEFAULT_MODEL = "claude-haiku-5-5"
# $ per 1M tokens (Anthropic first-party list prices cached 2026-11-03; Batches API = 50% off)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
TOKENS_PER_WORD = 1.35          # rough English rate; verify with messages.count_tokens before any scale-up
SYSTEM_TOKENS = int(len(SYSTEM.split()) * TOKENS_PER_WORD) + 20
OUT_TOKENS = 150                # JSON + one-line reason + some adaptive thinking at low effort (assumption)
N_ALL_PAIRS = 654_447


def estimate(pairs_full=G.PAIRS_FULL):
    p = pd.read_csv(pairs_full, usecols=["q_words", "a_words"])
    tok_in = (p["q_words"] + p["a_words"]) * TOKENS_PER_WORD + SYSTEM_TOKENS + 15
    rows = []
    for m, (pin, pout) in PRICES.items():
        cin, cout = tok_in.sum() / 1e6 * pin, len(p) * OUT_TOKENS / 1e6 * pout
        rows.append(dict(model=m, pairs=len(p), input_mtok=tok_in.sum() / 1e6, output_mtok=len(p) * OUT_TOKENS / 1e6,
                         cost_usd=cin + cout, cost_usd_batch=(cin + cout) / 2,
                         cost_usd_414=(cin + cout) * 414 / len(p)))
    return pd.DataFrame(rows), float(tok_in.mean())


def _parse(text):
    s = text[text.find("{"): text.rfind("}") + 1]
    d = json.loads(s)
    return float(d["p_evasive"]), float(d["p_non_answer"]), str(d.get("reason", ""))


def run(model=DEFAULT_MODEL):
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("no Anthropic credential in the environment: LLM judge skipped")
    try:
        import anthropic
    except ImportError:
        sys.exit("the anthropic SDK is not installed: LLM judge skipped")
    d = pd.read_pickle(G.CACHE / "labelled.pkl") if (G.CACHE / "labelled.pkl").exists() else G.load_labelled()
    out = G.CACHE / f"llm_judge_{model}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["pair_id"] for l in out.read_text().splitlines() if l.strip()}
    client = anthropic.Anthropic()
    with out.open("a") as fh:
        for r in d.itertuples():
            if r.pair_id in done:
                continue
            resp = client.messages.create(
                model=model, max_tokens=1024, output_config={"effort": "low"},
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": USER.format(q=r.q_anon, a=r.a_anon)}])
            text = next((b.text for b in resp.content if b.type == "text"), "")
            try:
                pe, pn, why = _parse(text)
            except (ValueError, KeyError):
                pe = pn = np.nan
                why = "unparsed"
            fh.write(json.dumps(dict(pair_id=r.pair_id, p_evasive=pe, p_non_answer=pn, reason=why,
                                     stop=resp.stop_reason, in_tok=resp.usage.input_tokens,
                                     out_tok=resp.usage.output_tokens)) + "\n")


def load(model=DEFAULT_MODEL):
    f = G.CACHE / f"llm_judge_{model}.jsonl"
    if not f.exists():
        return None
    return pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["estimate", "run"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    if a.mode == "estimate":
        t, mean_in = estimate()
        print(t.round(2).to_string(index=False))
        print(f"mean input tokens per pair ~{mean_in:.0f}")
    else:
        run(a.model)
