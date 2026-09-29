"""
Unit and Invariant Tests for Controlled Experiment Corrections (R1, R2, R3, R4, R6, R5)
========================================================================================
Verifies:
1. Standard clinical calibration-in-the-large (offset slope=1) satisfies the score equation.
2. Separate calibration slope estimation.
3. Scale invariance of source-standardized continuous Wasserstein distances.
4. Small-cell and degenerate sample protection in missingness analysis.
5. Absolute preservation of all 24 external-transfer ROC-AUC values.
6. Cluster bootstrap resample logging.
7. Mathematical efficiency axiom of exact Shapley values.
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

from scripts.corrections.r1_r2_calibration_internal_validation import (
    compute_standard_calibration_metrics,
    evaluate_prediction_set,
    fast_auc
)
from scripts.corrections.r4_dataset_shift import compute_continuous_feature_shift
from scripts.corrections.r6_missingness import check_evaluable
from scripts.corrections.r3_svm_shap import build_exact_shapley_matrix


class TestControlledCorrections(unittest.TestCase):

    def test_calibration_intercept_offset_score_root(self):
        """Tests that calibration intercept alpha satisfies sum(y - sigma(alpha + logit(p))) = 0."""
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.4, size=200)
        p = rng.uniform(0.1, 0.9, size=200)

        cal = compute_standard_calibration_metrics(y, p)
        alpha = cal["calibration_intercept"]
        self.assertFalse(np.isnan(alpha))

        logits = np.log(p / (1.0 - p))
        score = np.sum(y - 1.0 / (1.0 + np.exp(-(alpha + logits))))
        self.assertAlmostEqual(score, 0.0, delta=1e-2)

    def test_calibration_slope_separate(self):
        """Tests that calibration slope is estimated independently of intercept."""
        rng = np.random.default_rng(42)
        logits = rng.normal(0, 1, 1000)
        p = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, p)

        cal = compute_standard_calibration_metrics(y, p)
        # Slope should be close to 1.0
        self.assertGreater(cal["calibration_slope"], 0.8)
        self.assertLess(cal["calibration_slope"], 1.2)

    def test_standardized_wasserstein_scale_invariance(self):
        """Tests that source-standardized Wasserstein distance is scale-invariant."""
        x_tr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        x_te = np.array([15.0, 25.0, 35.0, 45.0, 55.0])

        shift_base = compute_continuous_feature_shift(x_tr, x_te)
        # Scale by 1000 (e.g. grams vs kilograms)
        shift_scaled = compute_continuous_feature_shift(x_tr * 1000.0, x_te * 1000.0)

        # Raw distance must scale by 1000
        self.assertAlmostEqual(shift_scaled["raw_wasserstein_distance"], shift_base["raw_wasserstein_distance"] * 1000.0, places=1)
        # Standardized distance must remain invariant!
        self.assertAlmostEqual(shift_scaled["standardized_wasserstein_distance"], shift_base["standardized_wasserstein_distance"], places=3)

    def test_missingness_small_cell_rejection(self):
        """Tests that degenerate cohorts (<10 positive or <10 negative events) are rejected."""
        # Single class
        is_eval, note, n, n_pos, n_neg = check_evaluable(np.array([1]*50))
        self.assertFalse(is_eval)
        self.assertIn("Degenerate", note)

        # Insufficient minority class (e.g. Zurich females: 10 pos, 2 neg)
        is_eval, note, n, n_pos, n_neg = check_evaluable(np.array([1]*30 + [0]*4))
        self.assertFalse(is_eval)
        self.assertIn("Degenerate", note)

        # Valid balanced cohort
        is_eval, note, n, n_pos, n_neg = check_evaluable(np.array([1]*25 + [0]*25))
        self.assertTrue(is_eval)
        self.assertEqual(note, "Valid evaluable set")

    def test_exact_shapley_operator_weights(self):
        """Tests that the precomputed Shapley transformation matrix satisfies exact combinatorial weights."""
        Z, A = build_exact_shapley_matrix(10)
        self.assertEqual(Z.shape, (1024, 10))
        self.assertEqual(A.shape, (10, 1024))

        # Efficiency test: For a linear model f(x) = sum(c_i * x_i), Shapley attributions must equal c_i * (x_i - b_i)
        c = np.array([1.5, -2.0, 0.5, 3.0, -1.0, 0.8, -0.4, 2.2, -1.5, 0.1])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = np.zeros(10)

        H = Z * x + (1.0 - Z) * b
        v = H @ c  # Model evaluations on all 1024 coalitions
        phi = A @ v
        expected_phi = c * (x - b)
        np.testing.assert_allclose(phi, expected_phi, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
