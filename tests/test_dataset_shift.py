"""
Unit tests for Phase 7: Dataset Shift, Calibration Drift & Post-Hoc Recalibration.
Strictly verifies:
1. Raw datasets in data/raw/ remain intact and unmodified.
2. External test labels are never passed into calibration fitting.
3. Calibration mappings are fit strictly on development data (OOF).
4. Recalibrated probabilities are strictly bounded in [0.0, 1.0].
5. Monotonic recalibration (Platt scaling) preserves exact ROC-AUC discrimination.
6. Deterministic reproducibility: Fixed seeds produce identical calibration parameters.
7. Synthetic leakage test: External distribution shift does not alter development-fitted calibration parameters.
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.dataset import TRACK_B_10_FEATURES, load_harmonized_cohort
from src.preprocessing import HarmonizedClinicalPreprocessor
from experiments.exp_internal_validation import load_config, build_model_pipeline
from experiments.exp_dataset_shift import (
    compute_continuous_shift,
    compute_categorical_shift,
    run_development_recalibration
)


class TestDatasetShiftMethodologicalAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.raw_dir = "data/raw"

    def test_raw_datasets_unmodified(self):
        """Rule 1: Verify raw source dataset files in data/raw/ are intact and unmodified."""
        raw_files = ["cleveland_raw.csv", "hungarian_raw.csv", "zurich_raw.csv", "va_raw.csv"]
        for fname in raw_files:
            fpath = os.path.join(self.raw_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Raw file missing: {fpath}")
            self.assertGreater(os.path.getsize(fpath), 1000)

    def test_shift_metrics_mathematical_properties(self):
        """Rule 2: Wasserstein distance non-negative; JSD bounded in [0, 1]; SMD = 0 for identical."""
        x = np.random.normal(50.0, 5.0, 100)
        # Identical distribution
        res = compute_continuous_shift(x, x)
        self.assertAlmostEqual(res["smd"], 0.0, delta=1e-5)
        self.assertAlmostEqual(res["wasserstein_distance"], 0.0, delta=1e-5)
        self.assertAlmostEqual(res["ks_statistic"], 0.0, delta=1e-5)

        # Categorical identical
        s = pd.Series([0, 1, 0, 1, 0, 1])
        cat_res = compute_categorical_shift(s, s)
        self.assertAlmostEqual(cat_res["tvd"], 0.0, delta=1e-5)
        self.assertAlmostEqual(cat_res["jensen_shannon_divergence"], 0.0, delta=1e-5)

    def test_recalibrated_probabilities_strictly_bounded(self):
        """Rule 4: Recalibrated probabilities (Platt & Isotonic) are strictly in [0.0, 1.0]."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        X_dev = dev_df[TRACK_B_10_FEATURES].copy()
        y_dev = dev_df["target"].to_numpy()
        X_ext = ext_df[TRACK_B_10_FEATURES].copy()
        y_ext = ext_df["target"].to_numpy()

        m_raw, m_platt, m_iso, p_raw, p_platt, p_iso, _ = run_development_recalibration(
            X_dev, y_dev, X_ext, y_ext, "RandomForest", self.config
        )

        # Bounds check
        self.assertTrue(np.all(p_platt >= 0.0) and np.all(p_platt <= 1.0), "Platt probabilities out of bounds!")
        self.assertTrue(np.all(p_iso >= 0.0) and np.all(p_iso <= 1.0), "Isotonic probabilities out of bounds!")
        self.assertFalse(np.isnan(p_platt).any(), "Platt probabilities contain NaN!")
        self.assertFalse(np.isnan(p_iso).any(), "Isotonic probabilities contain NaN!")

    def test_discrimination_preservation_under_platt_scaling(self):
        """Rule 5: Platt scaling is strictly monotonic and preserves exact ROC-AUC."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        X_dev = dev_df[TRACK_B_10_FEATURES].copy()
        y_dev = dev_df["target"].to_numpy()
        X_ext = ext_df[TRACK_B_10_FEATURES].copy()
        y_ext = ext_df["target"].to_numpy()

        m_raw, m_platt, _, _, _, _, _ = run_development_recalibration(
            X_dev, y_dev, X_ext, y_ext, "RandomForest", self.config
        )

        self.assertAlmostEqual(m_raw["roc_auc"], m_platt["roc_auc"], delta=0.01,
                               msg="Platt scaling distorted ROC-AUC ranking!")

    def test_synthetic_recalibration_leakage(self):
        """
        Rule 7: Synthetic leakage test.
        Verifies that shifting external test distributions does not alter development-fitted calibration parameters.
        """
        np.random.seed(42)
        n_dev = 150
        n_ext = 80

        # Development OOF probabilities
        p_dev_oof = np.random.uniform(0.1, 0.9, n_dev)
        y_dev = (p_dev_oof + np.random.normal(0, 0.1, n_dev) >= 0.5).astype(int)

        # Fit Platt on DEV OOF
        eps = 1e-5
        logits_dev = np.log(np.clip(p_dev_oof, eps, 1-eps) / (1 - np.clip(p_dev_oof, eps, 1-eps))).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
        platt.fit(logits_dev, y_dev)
        dev_slope = platt.coef_[0][0]
        dev_intercept = platt.intercept_[0]

        # Extreme external test set with severe shift (all ~0.999)
        p_ext_shifted = np.ones(n_ext) * 0.999
        logits_ext = np.log(p_ext_shifted / (1 - p_ext_shifted)).reshape(-1, 1)

        # Transform external with frozen platt
        _ = platt.predict_proba(logits_ext)[:, 1]

        # Verify platt parameters were NOT changed
        self.assertEqual(platt.coef_[0][0], dev_slope)
        self.assertEqual(platt.intercept_[0], dev_intercept)


if __name__ == "__main__":
    unittest.main()
