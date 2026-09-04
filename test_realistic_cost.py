#!/usr/bin/env python3
"""Automated tests for realistic industrial defect-rate cost re-weighting analysis."""
import json
import math
import os
import unittest

from exp_realistic_cost import (
    BATCH_SIZE,
    COST_RATIOS,
    MVTEC_TEST_COUNTS,
    PRIORS,
    compute_baselines,
    compute_expected_costs,
    compute_rates,
)


class TestRealisticCost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.json_path = "outputs/exp_realistic_cost.json"
        if not os.path.exists(cls.json_path):
            raise FileNotFoundError(f"{cls.json_path} does not exist. Run exp_realistic_cost.py first.")
        with open(cls.json_path, "r") as f:
            cls.data = json.load(f)

    def test_json_structure(self):
        self.assertIn("dataset_ground_truth", self.data)
        self.assertIn("baselines_per_10k_parts", self.data)
        self.assertIn("arm_A_fine_grid", self.data)
        self.assertIn("arms_comparison", self.data)

    def test_ground_truth_totals(self):
        gt = self.data["dataset_ground_truth"]
        self.assertEqual(gt["total_defective"], 1258)
        self.assertEqual(gt["total_good"], 467)
        self.assertEqual(gt["total_test_images"], 1725)
        self.assertAlmostEqual(gt["mvtec_test_prevalence"], 1258 / 1725, places=5)

    def test_baselines_formulas(self):
        baselines = self.data["baselines_per_10k_parts"]
        for r in COST_RATIOS:
            r_str = str(r)
            for p in PRIORS:
                p_str = str(p)
                b = baselines[r_str][p_str]
                expected_scrap = round(BATCH_SIZE * (1.0 - p) * 1.0, 1)
                expected_ship = round(BATCH_SIZE * p * r, 1)
                self.assertAlmostEqual(b["scrap_all"], expected_scrap, places=1)
                self.assertAlmostEqual(b["ship_all"], expected_ship, places=1)

    def test_rates_monotonicity(self):
        """As threshold percentile increases, threshold rises, so FNR increases and FPR decreases."""
        rates = self.data["arm_A_fine_grid"]["rates"]["micro"]
        pcts = self.data["arm_A_fine_grid"]["percentiles"]

        for i in range(len(pcts) - 1):
            p1, p2 = str(pcts[i]), str(pcts[i + 1])
            # FNR should be non-decreasing with percentile
            self.assertGreaterEqual(
                rates["fnr"][p2], rates["fnr"][p1] - 1e-6,
                f"FNR decreased from {p1} to {p2}"
            )
            # FPR should be non-increasing with percentile
            self.assertLessEqual(
                rates["fpr"][p2], rates["fpr"][p1] + 1e-6,
                f"FPR increased from {p1} to {p2}"
            )

    def test_expected_cost_formula(self):
        """Verify C = 10000 * [p * FNR * ratio + (1 - p) * FPR * 1.0]."""
        arm_a = self.data["arm_A_fine_grid"]
        rates = arm_a["rates"]["micro"]
        costs = arm_a["costs_per_10k_micro"]

        for r in COST_RATIOS:
            r_str = str(r)
            for p in PRIORS:
                p_str = str(p)
                for pct in arm_a["percentiles"]:
                    pct_str = str(pct)
                    fnr = rates["fnr"][pct_str]
                    fpr = rates["fpr"][pct_str]
                    expected = round(BATCH_SIZE * (p * fnr * r + (1.0 - p) * fpr * 1.0), 2)
                    actual = costs[r_str][p_str]["curve"][pct_str]
                    self.assertAlmostEqual(actual, expected, places=1)

    def test_p99_vindication_under_realistic_priors(self):
        """Under realistic priors p in [0.001, 0.005, 0.01] at 100:1, p99 must beat p50 substantially."""
        costs = self.data["arm_A_fine_grid"]["costs_per_10k_micro"]["100.0"]
        for p in [0.001, 0.005, 0.01]:
            p_str = str(p)
            c_p99 = costs[p_str]["cost_at_p99"]
            c_p50 = costs[p_str]["cost_at_p50"]
            self.assertLess(c_p99, c_p50, f"At p={p}, p99 ({c_p99}) did not beat p50 ({c_p50})")
            # At p=0.01, p99 should be more than 3x cheaper than p50
            if p == 0.01:
                self.assertGreater(c_p50 / c_p99, 3.0)


if __name__ == "__main__":
    unittest.main()
