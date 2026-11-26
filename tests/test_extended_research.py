"""Regression checks for the audited extensions without licensed inputs."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
sys.path[:0] = [str(OUTPUTS / "sp500_car_extension"), str(OUTPUTS / "sp500_financial_extension")]
import extend_car as car
import reconstruct_financials as financial


class ExtendedResearchTests(unittest.TestCase):
    def test_car_sums_paired_returns_and_rejects_incomplete_windows(self):
        bench = np.array([.01, .02, .03, .02, .01])
        stock = np.array([[.03, -.02, .05, np.nan, .02]])
        result = car.measure(np.array([0, 0, 0]), np.array([1, 2, 4]), -1, 1, stock, bench)
        self.assertAlmostEqual(result.value[0], (.03-.01) + (-.02-.02) + (.05-.03))
        self.assertTrue(np.isnan(result.value[1:]).all())
        self.assertEqual(result.reason.tolist(), ["", "missing_stock_return", "window_outside_calendar"])
        self.assertEqual(car.WINDOWS["car_0_1"], (0, 1))
        self.assertEqual(car.WINDOWS["car_-1_1"], (-1, 1))

    def test_only_approved_future_financial_values_are_replaced(self):
        columns = ["car_2_20", *financial.FEATURES]
        archived = [["0.07", *["1"]*len(financial.FEATURES)], ["-0.08", *["2"]*len(financial.FEATURES)]]
        events = pd.DataFrame({"year": [2024, 2025], "sector": ["Health Care", "Health Care"]})
        trial = pd.DataFrame(3.0, index=[1], columns=financial.FEATURES)
        audit = pd.DataFrame(financial.RECON, index=[1], columns=[f+"_status" for f in financial.FEATURES])
        rows, _ = financial.assemble(columns, archived, events, trial, audit, {"sue"})
        self.assertEqual(rows[0], archived[0])
        expected = archived[1].copy()
        expected[columns.index("sue")] = "3"
        self.assertEqual(rows[1], expected)

    def test_frozen_evaluation_and_saved_aggregates_agree(self):
        folder = OUTPUTS / "healthcare_extended_oos"
        spec = json.loads((folder / "frozen_experiment_spec.json").read_text(encoding="utf-8"))
        evaluator = (folder / "evaluate.py").read_bytes()
        legacy_dir = b"experiments/legacy_pipeline"
        self.assertIn(legacy_dir, evaluator)
        archived_dir = Path(spec["source_model"]).parent.as_posix().encode()
        self.assertEqual(hashlib.sha256(evaluator.replace(legacy_dir, archived_dir)).hexdigest(),
                         spec["evaluation_script_sha256"])
        self.assertIn("car_-1_1", spec["controls"])
        self.assertTrue({"car_0_1", "AR01"}.isdisjoint(spec["controls"]))
        self.assertEqual(len(spec["specifications"]), 2)
        performance = pd.read_csv(folder / "model_performance.csv")
        increments = pd.read_csv(folder / "incremental_comparison.csv")
        self.assertEqual(len(performance), 24)
        for (option, period), records in performance.groupby(["option", "period"]):
            self.assertEqual(records.events.nunique(), 1)
            ic = records.set_index("model").rank_ic
            paired = increments[increments.option.eq(option) & increments.period.eq(period)]
            for row in paired.itertuples():
                self.assertAlmostEqual(row.delta_rank_ic, ic[row.model] - ic["A"], places=12)
        results = pd.read_csv(folder / "bootstrap_inference.csv")
        pooled = results[results.n_valid.eq(2499) & results.holm_p_pooled.notna()]
        self.assertEqual(len(pooled), 8)
        self.assertTrue(pooled.crosses_zero.eq(True).all())

    def test_shared_financial_coverage_excludes_exact_modal_values(self):
        folder = OUTPUTS / "sp500_financial_extension"
        with (folder / "feature_coverage_aggregate.csv").open(encoding="utf-8", newline="") as file:
            fieldnames = set(next(csv.reader(file)))
        self.assertTrue({"scope", "year", "feature", "nonmissing", "most_frequent_n"} <= fieldnames)
        self.assertNotIn("most_frequent_value", fieldnames)


if __name__ == "__main__":
    unittest.main()
