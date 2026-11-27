"""`01 --source full` / `02 --source full` on a 3-call synthetic interim set (no model download)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
F = pytest.importorskip("full_source")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REFUSE = "We don't disclose that number and we are not going to break it out."
ANSWER = "Revenue in the cloud segment grew strongly because cloud demand from enterprise customers rose."


def fake_encoder():
    """Deterministic bag-of-words hashing vectors: similar wording -> high cosine."""
    calls = {"n": 0}

    def encode(texts):
        calls["n"] += len(texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 0, n, 1)
    encode.model_name = "fake-hash-64"
    encode.calls = calls
    return encode


def make_interim(root: Path):
    """Call 101 (firm A, Q1 2015, live with Q&A), call 102 (firm A, Q1 2015 pre-recorded at the 00:00 placeholder
    time, the same release -> dropped by keep_first), call 103 (firm B, Q3 2016)."""
    interim, raw = root / "interim", root / "raw"
    interim.mkdir()
    raw.mkdir()
    calls = pd.DataFrame({
        "companyid": [1.0, 1.0, 2.0], "keydevid": [11.0, 12.0, 13.0], "transcriptid": [101.0, 102.0, 103.0],
        "headline": ["Acme Widgets, Inc., Q1 2015 Earnings Call, Apr 20, 2015",
                     "Acme Widgets, Inc., Q1 2015 Pre Recorded Earnings Call, Apr 20, 2015",
                     "Beta Corp., Q3 2016 Earnings Call, Oct 25, 2016"],
        "mostimportantdateutc": ["2015-04-20", "2015-04-20", "2016-10-25"],
        "mostimportanttimeutc": ["21:00:00", "00:00:00", "13:30:00"],
        "keydeveventtypename": ["Earnings Calls"] * 3,
        "companyname": ["Acme Widgets, Inc.", "Acme Widgets, Inc.", "Beta Corp."],
        "version_type": ["Final"] * 3, "version_final": [1, 1, 1]})
    calls.to_parquet(interim / "transcript_calls.parquet")
    rows = []
    for tid, n in ((101, 20), (102, 2), (103, 20)):
        for k in range(n):
            refuse = k % 3 == 0
            rows.append(dict(transcriptid=tid, pair_id=k + 1, question_speaker_id=1,
                             question=f"Can you tell us about cloud segment revenue growth in 2015, question {k}?",
                             answer=REFUSE if refuse else ANSWER + " Acme Widgets thanks you.", n_answer_turns=1))
    rows.append(dict(transcriptid=103, pair_id=21, question_speaker_id=1, question="Last one?", answer="",
                     n_answer_turns=0))                                  # empty answer -> dropped
    pd.DataFrame(rows).to_parquet(interim / "qa_pairs.parquet")
    sec = pd.DataFrame({"transcriptid": [101, 102, 103],
                        "prepared_text": ["Good afternoon. Acme Widgets had a strong quarter with record revenue.",
                                          "Pre-recorded remarks: record revenue.",
                                          "Beta had a difficult quarter; demand was weak and uncertain."],
                        "words_prepared": [10, 4, 10], "question_text": ["q"] * 3, "words_question": [200, 20, 200],
                        "answer_text": [" ".join([ANSWER, REFUSE] * 5), "", " ".join([REFUSE, ANSWER] * 5)],
                        "words_answer": [300, 0, 300], "n_questions": [20, 0, 21], "n_answers": [20, 0, 20],
                        "avg_answer_words": [15.0, 0.0, 15.0]})
    sec.to_parquet(interim / "call_sections.parquet")
    pd.DataFrame({"companyid": [1.0, 2.0], "gvkey": ["000001", "000002"], "companyname": ["Acme", "Beta"],
                  "startdate": [None, None], "enddate": [None, None], "primaryflag": [1, 1]}) \
        .to_parquet(raw / "ciq_wrds_gvkey.parquet")
    pd.DataFrame({"gvkey": ["000001", "000002"], "permno": [10001.0, 10002.0], "permco": [1.0, 2.0],
                  "linktype": ["LU", "LC"], "linkprim": ["P", "P"], "linkdt": ["2000-01-01", "2000-01-01"],
                  "linkenddt": [None, None]}).to_parquet(raw / "crsp_ccmxpf_lnkhist.parquet")
    pd.DataFrame({"permno": pd.array([10001, 10002], dtype="Int64"), "indno": pd.array([1, 1], dtype="Int64"),
                  "mbrstartdt": ["2000-01-01", "2000-01-01"], "mbrenddt": ["2026-09-30", "2026-09-30"],
                  "mbrflg": ["NORM", "NORM"]}).to_parquet(raw / "crsp_msp500list_v2.parquet")
    pd.DataFrame({"gvkey": ["000001", "000002"], "gsector": ["45", "20"], "ggroup": ["4510", "2010"],
                  "gind": ["451010", "201010"], "gsubind": ["45101010", "20101010"],
                  "indfrom": ["1999-06-30", "1999-06-30"], "indthru": [None, None]}) \
        .to_parquet(raw / "comp_co_hgic.parquet")
    # labels: every pair of 101 and 103 (evasive = refusal) + one pair of the dropped duplicate call
    ids = [f"{t}_{k:03d}" for t in (101, 103) for k in range(1, 21)]
    lab = pd.DataFrame({"pair_id": ids, "transcriptid": [int(i.split("_")[0]) for i in ids]})
    lab["call_date"] = np.where(lab["transcriptid"] == 101, "2015-04-20", "2016-10-25")
    lab["evasive"] = [int((int(i.split("_")[1]) - 1) % 3 == 0) for i in ids]
    lab["non_answer"] = lab["evasive"]
    lab = pd.concat([lab, pd.DataFrame([dict(pair_id="102_001", transcriptid=102, call_date="2015-04-20",
                                             evasive=1, non_answer=1)])], ignore_index=True)
    lab.to_csv(root / "qa_labels.csv", index=False)
    return interim, raw, root / "qa_labels.csv"


@pytest.fixture()
def built(tmp_path):
    interim, raw, labels = make_interim(tmp_path)
    paths = dict(features_path=tmp_path / "features_text_full.csv", pairs_path=tmp_path / "qa_pairs_full.csv",
                 meta_path=tmp_path / "meta.json", log_path=tmp_path / "full_source_log.md",
                 cache_dir=tmp_path / "emb_cache")
    enc = fake_encoder()
    meta = F.run(embed="st", interim=interim, raw=raw, labels_path=labels, encode=enc, workers=1, **paths)
    return tmp_path, paths, meta, enc, (interim, raw, labels)


def test_pair_ids_and_keep_first(built):
    tmp, paths, meta, _, _ = built
    ev = pd.read_csv(paths["features_path"])
    pairs = pd.read_csv(paths["pairs_path"], dtype={"pair_id": str})
    assert ev["transcriptid"].tolist() == [101, 103]                 # 102: same release, placeholder time
    assert meta["n_duplicate_calls_dropped"] == 1
    expected = [f"101_{k:03d}" for k in range(1, 21)] + [f"103_{k:03d}" for k in range(1, 21)]
    assert pairs["pair_id"].tolist() == expected                     # 103_021 has an empty answer
    assert ev["firm"].astype(str).str.zfill(6).tolist() == ["000001", "000002"]
    assert ev["day0"].isna().all()
    assert not ev["qa_truncated"].any()
    for c in ("REL", "REL_JAC", "NONANS", "EVADE", "DIV_QP", "EVADE_CLF", "NONANS_CLF"):
        assert ev[c].notna().all(), c
    assert ev["NOVEL"].isna().all()                                  # each firm has a single call
    assert set(pairs["NONANS_RULE"][pairs["pair_id"].str.endswith(("001", "004"))]) == {1}
    log_md = paths["log_path"].read_text()
    assert "duplicate_calls: keep_first" in log_md and "102" in log_md


def test_labels_merge_and_classifiers(built):
    tmp, paths, meta, _, _ = built
    pairs = pd.read_csv(paths["pairs_path"], dtype={"pair_id": str})
    lab = pd.read_csv(tmp / "qa_labels.csv", dtype={"pair_id": str})
    m = pairs.merge(lab, on="pair_id")
    assert len(m) == 40                                              # 102_001 belongs to the dropped call
    assert meta["n_labels"] == {"EVADE_CLF": 40, "NONANS_CLF": 40}
    assert meta["label_classifiers"]["EVADE_CLF"]["labels_not_in_pairs"] == 1
    assert meta["source"] == "full" and meta["firm_key"] == "gvkey"
    assert m["labelled_oof"].all()
    assert meta["oof_auc"]["EVADE_CLF"] > 0.9                        # refusals are trivially separable here
    assert pairs["EVADE_CLF"].between(0, 1).all()


def test_cache_reused(built):
    tmp, paths, meta, enc, (interim, raw, labels) = built
    n_first = enc.calls["n"]
    assert n_first > 0
    enc2 = fake_encoder()
    F.run(embed="st", interim=interim, raw=raw, labels_path=labels, encode=enc2, workers=1, **paths)
    assert enc2.calls["n"] == 0
    with pytest.raises(RuntimeError):
        F.Cache(paths["cache_dir"], "another-model", "pairs")


def test_validate_full(built):
    tmp, paths, meta, _, (_, _, labels) = built
    V = _load("validate02", "02_validate.py")
    out = tmp / "validation_full"
    agree, byg = V.main_full(paths["features_path"], paths["pairs_path"], labels, paths["meta_path"], out)
    a = pd.read_csv(out / "label_agreement.csv")
    assert {"EVADE_CLF", "NONANS_CLF", "REL", "REL_JAC", "NONANS_RULE"} <= set(a["score"])
    assert (a["n"] == 40).all()
    by_year = pd.read_csv(out / "label_agreement_by_year.csv")
    assert set(by_year["year"]) == {2015, 2016}
    assert (out / "label_agreement_by_sector.csv").exists()
    summary = (out / "summary.md").read_text()
    assert "Skipped" in summary and "needs `la targets`" in summary
    checks = pd.read_csv(out / "leakage_checks.csv")
    assert not (checks["status"] == "FAIL").any(), checks


def test_anonymizer_masks_own_name_and_years():
    f = F.anonymizer("Acme Widgets, Inc.")
    assert f("Acme Widgets, Inc. and acme widgets in 2015") == "the company and the company in YEAR"
    assert F.fiscal_label(pd.Series(["Peabody Energy Corp., 2011 Earnings Call, Jan 24, 2012",
                                     "X Inc., Q3 2016 Earnings Call, Oct 25, 2016"])).tolist() == ["Q4 2011", "Q3 2016"]


def test_meta_is_json(built):
    _, paths, _, _, _ = built
    meta = json.loads(Path(paths["meta_path"]).read_text())
    assert meta["embedding_backend"] == "st" and meta["anonymization"] == F.ANON_VERSION
