"""Clean throughput timing (one model at a time) for the cost estimate: windows/s on the device.

    .venv/bin/python scripts/f3/time_models.py [n_windows]
Writes data/interim/f3_cache/throughput_clean.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_scores as X  # noqa: E402
import f3common as G  # noqa: E402


def main(n=400):
    d = pd.read_pickle(G.CACHE / "labelled.pkl")
    out = {}
    rank_pairs = [(q, w) for q, a in zip(d["q_anon"], d["a_anon"]) for w in X.windows(a, X.RANK_WORDS)][:n]
    nli_prem = [w for a in d["a_anon"] for w in X.windows(a, X.NLI_WORDS)][:n]
    jobs = {**{k: (m, [p[0] for p in rank_pairs], [p[1] for p in rank_pairs]) for k, m in X.RANKERS.items()},
            "nli_one_hypothesis": (X.NLI_MODEL, nli_prem, [X.HYP_A["decline"]] * len(nli_prem))}
    for k, (m, a, b) in jobs.items():
        tok, mod, torch = X._load(m)
        X._score(tok, mod, torch, a[:32], b[:32])            # warm-up
        t0 = time.time()
        X._score(tok, mod, torch, a, b)
        dt = time.time() - t0
        out[k] = dict(model=m, windows=len(a), seconds=dt, windows_per_s=len(a) / dt, device=X._device())
        print(k, f"{len(a) / dt:.1f} windows/s", flush=True)
    (G.CACHE / "throughput_clean.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
