"""Tests for IT feature semantics, strict time ordering and reusable inference."""

from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extract_finbert import (Call, FEATURES, aggregate, prior_changes, process, publish, sections,
                             sentences, setup)
from fit_finbert import FEATURES as MODEL_FEATURES, ForwardSearch, baseline


class FeatureSemantics(unittest.TestCase):
    def test_legacy_sections_and_truncation(self):
        records = [
            (1, "The operator welcomes the participants."),
            (2, "This is a long prepared remark. No."),
            (3, "Can you explain the bridge to profit?"),
            (4, "Operating results exceeded expectations.\n\nThis continuation was skipped by the IT parser."),
            (4, "We continue to execute the plan."),
        ]
        frame = pd.DataFrame(records, columns=["transcriptcomponenttypeid", "componenttext"])
        (prepared, answers), omitted = sections(frame)
        self.assertEqual(prepared, (("This is a long prepared remark.", False),))
        self.assertEqual(answers, (("Operating results exceeded expectations.", False),
                                  ("We continue to execute the plan.", False)))
        self.assertEqual(omitted, 1)
        self.assertEqual(sentences("x" * 515), (("x" * 512, True),))

    def test_top_label_scores_and_missing_rules(self):
        result = aggregate([(0, .8, 30), (1, .6, 20), (2, .95, 15)], "answers")
        self.assertAlmostEqual(result["fb_sent_answers"], (.8 - .6) / 3)
        self.assertEqual(result["fb_pos_pct_answers"], 1 / 3)
        self.assertEqual(result["fb_neg_pct_answers"], 1 / 3)
        self.assertAlmostEqual(result["fb_sent_std_answers"], np.std([.8, -.6, 0.]))
        self.assertEqual(aggregate([(0, .8, 8)], "prepared")["fb_sent_std_prepared"], 0.)
        empty = aggregate([], "prepared")
        self.assertEqual(empty["fb_n_sent_prepared"], 0)
        self.assertTrue(np.isnan(empty["fb_sent_prepared"]))

    def test_changes_use_prior_four_strictly_earlier_calls(self):
        dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01",
                               "2020-04-01", "2020-04-01", "2020-05-01", "2020-05-01"])
        data = pd.DataFrame(dict(transcriptid=range(7), clock=dates, permno=[1] * 6 + [2],
                                 fb_sent_prepared=[.1, .2, .3, .4, .5, .6, .7],
                                 fb_sent_answers=[np.nan, .2, .3, .4, .5, .6, .7]))
        changes = prior_changes(data)
        self.assertAlmostEqual(changes.fb_sent_prepared_chg.iloc[3], .4 - .2)
        self.assertAlmostEqual(changes.fb_sent_prepared_chg.iloc[4], .5 - .2)
        self.assertAlmostEqual(changes.fb_sent_prepared_chg.iloc[5], .6 - np.mean([.2, .3, .4, .5]))
        self.assertTrue(np.isnan(changes.fb_sent_prepared_chg.iloc[6]))

    def test_call_and_sentence_cache_can_be_reused_without_engine(self):
        class Fake:
            def score(self, texts, batch_size):
                return np.tile([.1, .6, .3], (len(texts), 1)), np.full(len(texts), 12)

        def no_inference():
            raise AssertionError("Validated cached sentence was rescored")

        one = "Exactly the same sentence for both calls."
        with TemporaryDirectory() as tmp, sqlite3.connect(":memory:") as db:
            archive = Path(tmp)
            setup(db)
            for number, make_scorer in [(1, Fake), (2, no_inference)]:
                call = Call(number, "local-source", sha256(str(number).encode()).hexdigest(),
                            (((one, False),), ((one, False),)), 0)
                event = SimpleNamespace(permno=1, year=2024, sector="Health Care",
                                        clock=pd.Timestamp(f"2024-01-0{number}T12:00Z"))
                count = process(db, [(call, event)], make_scorer, 8)
                self.assertEqual(count, 1 if number == 1 else 0)
            publish(db, archive)
            call_table = pd.read_parquet(archive / "call_features.parquet")
            text_table = pq.read_table(archive / "sentence_scores.parquet").to_pandas()
            self.assertEqual(len(call_table), 2)
            self.assertEqual(len(text_table), 4)
            self.assertEqual(text_table.sentence_sha256.nunique(), 1)
            self.assertTrue(set(FEATURES).issubset(call_table.columns))
            self.assertEqual(call_table.fb_sent_answers_chg.iloc[1], 0.)


class FinBERTSelection(unittest.TestCase):
    def test_same_controls_and_allowed_estimators_with_zero_feature_fallback(self):
        rng = np.random.default_rng(12)
        self.assertEqual(len(MODEL_FEATURES), 13)
        x = np.column_stack([rng.normal(size=(100, len(baseline.CONTROLS))), np.zeros((100, 13))])
        y = np.arange(100.)
        search = ForwardSearch([baseline.Fold(x, x, y, y)] * 3, {})
        candidate = search.evaluate(())
        archived = SimpleNamespace(estimator=candidate["estimator"], inner_rank_ic=candidate["mean_rank_ic"])
        results = search.select(archived)
        self.assertEqual(results["C"][0], ())
        self.assertEqual(results["C"][1], candidate)
        self.assertEqual(results["C*"][0], MODEL_FEATURES)
        self.assertTrue(all(not item["accepted"] for item in search.paths))


if __name__ == "__main__":
    unittest.main()
