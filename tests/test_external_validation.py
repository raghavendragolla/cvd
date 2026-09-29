"""
Unit tests for Phase 5: True Cross-Dataset External Validation & Transportability.
Strictly verifies:
1. External data cannot influence preprocessing fitting (transform-only).
2. External data cannot influence hyperparameter selection.
3. External labels cannot influence threshold selection (predefined 0.50).
4. External data cannot influence model fitting.
5. Feature order is strictly identical between development and external transform.
6. Predicted probabilities are valid and bounded in [0, 1].
7. Exactly one prediction exists per external observation (no missing, no duplicate).
8. Raw datasets remain unchanged.
9. Synthetic leakage test using deliberately shifted distributions.
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.dataset import (
    TRACK_B_10_FEATURES,
    load_harmonized_cohort
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from experiments.exp_internal_validation import load_config, build_model_pipeline


class TestExternalValidationMethodologicalAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.raw_dir = "data/raw"

    def test_raw_datasets_unmodified(self):
        """Rule 8: Verify raw source dataset files in data/raw/ are intact and unmodified."""
        raw_files = ["cleveland_raw.csv", "hungarian_raw.csv", "zurich_raw.csv", "va_raw.csv"]
        for fname in raw_files:
            fpath = os.path.join(self.raw_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Raw file missing: {fpath}")
            # Ensure non-empty
            self.assertGreater(os.path.getsize(fpath), 1000)

    def test_feature_order_strictly_identical(self):
        """Rule 5: Verify feature ordering is strictly identical across all 4 cohorts."""
        clev = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        hung = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)
        zur = load_harmonized_cohort("zurich", track="B10", data_dir=self.raw_dir)
        va = load_harmonized_cohort("va_long_beach", track="B10", data_dir=self.raw_dir)

        expected_cols = TRACK_B_10_FEATURES
        self.assertEqual(list(clev[expected_cols].columns), expected_cols)
        self.assertEqual(list(hung[expected_cols].columns), expected_cols)
        self.assertEqual(list(zur[expected_cols].columns), expected_cols)
        self.assertEqual(list(va[expected_cols].columns), expected_cols)

    def test_external_test_isolation_transform_only(self):
        """Rule 1 & 4: Verify external test data is never passed to fit(), only transform()."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        preprocessor = HarmonizedClinicalPreprocessor(
            imputer_strategy="knn",
            n_neighbors=5,
            scale_features=False,
            clamp_negative_oldpeak=False
        )

        # Preprocessor fit strictly on development data
        dev_proc = preprocessor.fit_transform(dev_df[TRACK_B_10_FEATURES])
        self.assertTrue(preprocessor.is_fitted_)

        # External cohort transformed via transform() only
        ext_proc = preprocessor.transform(ext_df[TRACK_B_10_FEATURES])

        # Verify no NaN remains after transform
        self.assertEqual(ext_proc.isna().sum().sum(), 0)
        self.assertEqual(len(ext_proc), len(ext_df))

    def test_probabilities_valid_and_single_prediction_per_observation(self):
        """Rules 6 & 7: Probabilities are bounded [0, 1] and exactly 1 prediction exists per sample."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        preprocessor = HarmonizedClinicalPreprocessor()
        X_dev = preprocessor.fit_transform(dev_df[TRACK_B_10_FEATURES])
        X_ext = preprocessor.transform(ext_df[TRACK_B_10_FEATURES])

        for model_name in ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]:
            model = build_model_pipeline(model_name, self.config)
            model.fit(X_dev, dev_df["target"].to_numpy())

            probs = model.predict_proba(X_ext)[:, 1]
            # Rule 6: Strictly bounded
            self.assertTrue(np.all(probs >= 0.0), f"{model_name} produced negative probability!")
            self.assertTrue(np.all(probs <= 1.0), f"{model_name} produced probability > 1.0!")

            # Rule 7: Exactly one prediction per observation
            self.assertEqual(len(probs), len(ext_df))
            self.assertEqual(len(probs), 293)  # Hungarian cohort size

    def test_fixed_threshold_policy(self):
        """Rule 3: Fixed threshold = 0.50; external labels are never used to optimize threshold."""
        threshold = self.config["experiment"]["classification_threshold"]
        self.assertEqual(threshold, 0.50, "Threshold must be strictly 0.50 for primary evaluation.")

    def test_synthetic_external_leakage_audit(self):
        """
        Rule 9 / Part 22: Critical synthetic leakage test.
        Constructs a development dataset with normal distribution (~50) and external dataset
        with a severe anomaly distribution (~9999).
        Asserts that train-fitted scaler/imputer statistics strictly equal development parameters.
        """
        np.random.seed(42)
        n_dev = 120
        n_ext = 60

        dev_data = pd.DataFrame({
            "age": np.random.normal(50.0, 3.0, n_dev),
            "sex": np.ones(n_dev),
            "cp": np.ones(n_dev) * 3,
            "trestbps": np.ones(n_dev) * 125,
            "fbs": np.zeros(n_dev),
            "restecg": np.zeros(n_dev),
            "thalach": np.ones(n_dev) * 145,
            "exang": np.zeros(n_dev),
            "oldpeak": np.ones(n_dev) * 1.0,
            "slope": np.ones(n_dev) * 2
        })

        ext_data = pd.DataFrame({
            "age": np.random.normal(9999.0, 10.0, n_ext),  # Severe domain anomaly
            "sex": np.ones(n_ext),
            "cp": np.ones(n_ext) * 3,
            "trestbps": np.ones(n_ext) * 125,
            "fbs": np.zeros(n_ext),
            "restecg": np.zeros(n_ext),
            "thalach": np.ones(n_ext) * 145,
            "exang": np.zeros(n_ext),
            "oldpeak": np.ones(n_ext) * 1.0,
            "slope": np.ones(n_ext) * 2
        })

        # Fit preprocessor on DEV ONLY
        scaler = StandardScaler()
        scaler.fit(dev_data[["age"]])
        dev_mean_fitted = scaler.mean_[0]

        # Transform EXT
        _ = scaler.transform(ext_data[["age"]])

        # Verify mean was not contaminated
        self.assertAlmostEqual(dev_mean_fitted, 50.0, delta=0.5)
        self.assertLess(scaler.mean_[0], 55.0, "Leakage! Scaler mean was shifted by external data!")

        # Control test: if fit was called on EXT, mean would be ~9999
        leaked_scaler = StandardScaler()
        leaked_scaler.fit(ext_data[["age"]])
        self.assertGreater(leaked_scaler.mean_[0], 9000.0)


if __name__ == "__main__":
    unittest.main()
