"""How many more labels would tell an AUC of 0.65 from 0.70? (No labelling is done here.)

Two estimates of the AUC standard error as a function of the number of labelled pairs n at the current
base rate p:
  - Hanley & McNeil (1982) formula, at AUC = 0.70;
  - the observed call-bootstrap SE at n = 414, scaled by sqrt(414 / n) (one pair per call, so calls ~ pairs).
Criterion: a one-sided test of AUC = 0.65 against 0.70 at alpha = 0.05 with 80% power needs
SE <= 0.05 / (1.645 + 0.842) ~= 0.0201. A looser criterion is a 95% CI half-width of 0.05.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z_A, Z_B = 1.6449, 0.8416
SE_TEST = 0.05 / (Z_A + Z_B)
SE_CI05 = 0.05 / 1.96


def hanley_se(auc, n, p):
    npos, nneg = max(n * p, 1), max(n * (1 - p), 1)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)
    return math.sqrt(v)


def n_needed(se_target, p, auc=0.70, se_obs=None, n_obs=414):
    n_h = next(n for n in range(50, 100000, 10) if hanley_se(auc, n, p) <= se_target)
    n_b = math.ceil(n_obs * (se_obs / se_target) ** 2) if se_obs else None
    return n_h, n_b


def table(base_rates, se_obs):
    rows = []
    for t, p in base_rates.items():
        for crit, se in (("0.65 vs 0.70, one-sided 5%, 80% power", SE_TEST), ("95% CI half-width 0.05", SE_CI05)):
            n_h, n_b = n_needed(se, p, se_obs=se_obs.get(t))
            rows.append(dict(target=t, base_rate=p, criterion=crit, se_target=se, n_total_hanley=n_h,
                             n_total_bootstrap_scaled=n_b, n_more_hanley=max(n_h - 414, 0),
                             n_more_bootstrap=max((n_b or 0) - 414, 0), positives_at_n=round(n_h * p)))
    return pd.DataFrame(rows)


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import f3common as G
    r = pd.read_csv(G.OUT / "results.csv")
    pick = {"evasive": "E1 rank-mean(C1, T3)", "non_answer": "C1 rich pair features"}
    se = {t: float(((r.candidate == c) & (r.target == t)).pipe(lambda m: (r[m].auc_hi - r[m].auc_lo) / 3.92).iloc[0])
          for t, c in pick.items()}
    lab = pd.read_csv(G.LABELS)
    t = table({k: float(lab[k].mean()) for k in G.TARGETS}, se)
    t.to_csv(G.OUT / "label_count.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
