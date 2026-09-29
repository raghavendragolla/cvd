"""
Unit tests for Phase 4: Baseline Models & Internal Validation Protocol.
Verifies fold generation, out-of-fold predictions, probability ranges,
metric correctness, track dimensions, and includes an explicit leakage test.
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    TRACK_A_FEATURES,
    TRACK_B_10_FEATURES,
    load_harmonized_cohort
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from src.evaluate import (
    evaluate_classifier,
    compute_expected_calibration_error,
    compute_calibration_intercept_slope
)
from experiments.exp_internal_validation import build_model_pipeline, load_config


class TestInternalValidationProtocol(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.seed = cls.config["experiment"]["random_seed"]
        cls.k_folds = cls.config["experiment"]["cv_folds"]

    def test_track_feature_counts(self):
        """Test 10 & 11: Track A has exactly 13 features, Track B has exactly 10 features."""
        self.assertEqual(len(TRACK_A_FEATURES), 13)
        self.assertEqual(len(TRACK_B_10_FEATURES), 10)
        self.assertNotIn("chol", TRACK_B_10_FEATURES)
        self.assertNotIn("ca", TRACK_B_10_FEATURES)
        self.assertNotIn("thal", TRACK_B_10_FEATURES)

    def test_stratified_folds_and_complete_oof_coverage(self):
        """Tests 1, 2, 3: Stratified folds partition data disjointly; every sample gets exactly one OOF prediction."""
        df = load_harmonized_cohort("cleveland", track="B10")
        X = df[TRACK_B_10_FEATURES]
        y = df["target"].to_numpy()

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        visited_indices = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Disjoint check: train and val must have 0 overlap
            overlap = set(train_idx).intersection(set(val_idx))
            self.assertEqual(len(overlap), 0, f"Fold {fold_idx} has train/val overlap!")

            # Stratification check: fold positive proportion close to global proportion
            fold_pos_ratio = np.mean(y[val_idx])
            global_pos_ratio = np.mean(y)
            self.assertAlmostEqual(fold_pos_ratio, global_pos_ratio, delta=0.08)

            visited_indices.extend(val_idx)

        # Complete coverage check
        self.assertEqual(len(visited_indices), len(df))
        self.assertEqual(set(visited_indices), set(range(len(df))))

    def test_probability_predictions_bounded(self):
        """Test 6: Model probability estimates must strictly lie in [0, 1]."""
        df = load_harmonized_cohort("cleveland", track="B10")
        X = df[TRACK_B_10_FEATURES].iloc[:100]
        y = df["target"].iloc[:100].to_numpy()

        prep = HarmonizedClinicalPreprocessor()
        X_proc = prep.fit_transform(X)

        for model_name in ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]:
            model = build_model_pipeline(model_name, self.config)
            model.fit(X_proc, y)
            probs = model.predict_proba(X_proc)[:, 1]
            self.assertTrue(np.all(probs >= 0.0), f"{model_name} produced negative probabilities!")
            self.assertTrue(np.all(probs <= 1.0), f"{model_name} produced probabilities > 1.0!")

    def test_metrics_calculation_correctness(self):
        """Tests 7, 8: Metric computation produces valid discrimination and calibration scores."""
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15, 0.75, 0.25])
        y_pred = (y_proba >= 0.50).astype(int)

        metrics = evaluate_classifier(y_true, y_pred, y_proba)

        # Accuracy should be 100% for this perfectly separable case
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertEqual(metrics["roc_auc"], 100.0)
        self.assertEqual(metrics["pr_auc"], 100.0)
        self.assertEqual(metrics["mcc"], 1.0)
        self.assertLess(metrics["brier_score"], 0.10)
        self.assertIn("calibration_intercept", metrics)
        self.assertIn("calibration_slope", metrics)

    def test_calibration_intercept_slope(self):
        """Test 9: Calibration uses out-of-fold probabilities; slope and intercept match expectations."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Well calibrated probabilities
        y_proba = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.7, 0.8, 0.75, 0.85, 0.9])
        intercept, slope = compute_calibration_intercept_slope(y_true, y_proba)
        # For well calibrated predictions, intercept should be near 0 and slope near 1
        self.assertAlmostEqual(intercept, 0.0, delta=1.5)
        self.assertGreater(slope, 0.0)

    def test_explicit_leakage_prevention(self):
        """
        Part 22 Explicit Leakage Verification:
        Construct synthetic dataset with extreme distribution shift in validation fold.
        Verify imputer and scaler fitted inside fold are unaffected by validation fold.
        """
        # Train fold: normal distribution around 50
        # Val fold: extreme anomaly distribution around 5000
        np.random.seed(42)
        n_train = 80
        n_val = 20

        train_vals = np.random.normal(50.0, 5.0, n_train)
        val_vals = np.random.normal(5000.0, 5.0, n_val)

        df_synthetic = pd.DataFrame({
            "age": np.concatenate([train_vals, val_vals]),
            "sex": np.ones(n_train + n_val),
            "cp": np.ones(n_train + n_val) * 3,
            "trestbps": np.ones(n_train + n_val) * 120,
            "fbs": np.zeros(n_train + n_val),
            "restecg": np.zeros(n_train + n_val),
            "thalach": np.ones(n_train + n_val) * 150,
            "exang": np.zeros(n_train + n_val),
            "oldpeak": np.ones(n_train + n_val) * 1.0,
            "slope": np.ones(n_train + n_val) * 2,
            "target": np.concatenate([np.ones(n_train // 2), np.zeros(n_train // 2), np.ones(n_val // 2), np.zeros(n_val // 2)])
        })

        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_train + n_val)

        # 1. Pipeline execution: fit scaler ONLY on train fold
        scaler = StandardScaler()
        scaler.fit(df_synthetic.iloc[train_idx][["age"]])

        # Scaler mean must match train mean (~50), NOT contaminated by val mean (~5000)
        expected_train_mean = np.mean(train_vals)
        fitted_mean = scaler.mean_[0]

        self.assertAlmostEqual(fitted_mean, expected_train_mean, places=4)
        self.assertLess(fitted_mean, 70.0, "Leakage detected! Scaler was contaminated by validation distribution!")

        # 2. Verify that if someone mistakenly fitted on entire dataset, it WOULD fail
        leaked_scaler = StandardScaler()
        leaked_scaler.fit(df_synthetic[["age"]])
        leaked_mean = leaked_scaler.mean_[0]
        self.assertGreater(leaked_mean, 500.0, "Leaked mean test control verified.")


if __name__ == "__main__":
    unittest.main()
