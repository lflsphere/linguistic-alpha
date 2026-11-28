"""Agent G's F3 relevance helpers (scripts/f3/): no model download, no labels, no API call."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

F3 = Path(__file__).resolve().parents[1] / "scripts" / "f3"
sys.path.insert(0, str(F3))
pytest.importorskip("sklearn")
G = pytest.importorskip("f3common")
import cross_scores as X  # noqa: E402
import label_plan as LP  # noqa: E402
import llm_judge as J  # noqa: E402
import pair_features as P  # noqa: E402


def test_refusal_and_deferral_cues():
    f = P.pair_features("How much did pricing contribute? And what's the margin?",
                        "We don't disclose that and we'll follow up offline.")
    assert f["a_refuse"] >= 1 and f["a_defer"] >= 1
    assert f["q_number"] == 1 and f["num_q_no_num"] == 1 and f["q_qmarks"] == 2
    g = P.pair_features("How much did pricing contribute?", "Pricing added about 3% or $40 million in Q2.")
    assert g["a_refuse"] == 0 and g["a_any_number"] == 1 and g["a_spec_p100"] > f["a_spec_p100"]


def test_feature_names_stable_and_frame():
    df = pd.DataFrame({"q_anon": ["Is it up?", ""], "a_anon": ["Yes, up 5%.", ""]})
    out = P.frame(df)
    assert list(out.columns) == P.RICH and len(out) == 2
    assert out.loc[0, "a_direct"] == 1 and np.isnan(out.loc[1, "q_cover"])


def test_folds_keep_calls_together():
    g = np.repeat(np.arange(20), 3)
    for tr, te in G.folds(g):
        assert not set(g[tr]) & set(g[te])


def test_oof_and_bootstrap_on_signal():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    X1 = np.c_[y + rng.normal(0, 1, n), rng.normal(0, 1, n)]
    g = np.arange(n)
    oof, chosen = G.oof_model(X1, y, g)
    r = G.auc_ci(y, oof, g, n_boot=200)
    assert len(chosen) == G.N_SPLITS and not np.isnan(oof).any()
    assert 0.7 < r["auc"] < 0.9 and r["auc_lo"] < r["auc"] < r["auc_hi"]
    noise = G.auc_ci(y, rng.normal(size=n), g, n_boot=200)
    assert noise["auc_lo"] < 0.5 < noise["auc_hi"]


def test_windows():
    assert X.windows("", 5) == [""]
    w = X.windows(" ".join(map(str, range(12))), 5)
    assert len(w) == 3 and w[-1] == "10 11"


def test_label_plan_monotone():
    assert LP.hanley_se(0.7, 2000, 0.14) < LP.hanley_se(0.7, 400, 0.14)
    t = LP.table({"evasive": 0.15}, {"evasive": 0.038})
    assert (t["n_total_hanley"] > 414).all()


def test_llm_judge_estimate_and_no_key(tmp_path, monkeypatch):
    f = tmp_path / "p.csv"
    pd.DataFrame({"q_words": [50, 70], "a_words": [200, 100]}).to_csv(f, index=False)
    t, mean_in = J.estimate(f)
    assert set(t["model"]) == set(J.PRICES) and mean_in > J.SYSTEM_TOKENS
    assert (t["cost_usd_batch"] == t["cost_usd"] / 2).all()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        J.run()
    assert J._parse('x {"p_evasive": 0.2, "p_non_answer": 0.9, "reason": "r"} y')[:2] == (0.2, 0.9)


def test_llm_judge_load_keeps_last_row(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(J.G, "CACHE", tmp_path)
    rows = [dict(pair_id="a", p_evasive=float("nan"), p_non_answer=float("nan")),
            dict(pair_id="a", p_evasive=0.3, p_non_answer=0.1), dict(pair_id="b", p_evasive=0.9, p_non_answer=0.8)]
    (tmp_path / "llm_judge_m.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = J.load("m").set_index("pair_id")
    assert len(df) == 2 and df.loc["a", "p_evasive"] == 0.3
