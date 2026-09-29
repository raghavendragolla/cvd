"""
Unit tests for Phase 3: Research-Grade Preprocessing & Harmonization Pipeline.
Verifies zero-masking, leak-free preprocessing, track partitioning, and duplicate handling.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.dataset import (
    load_harmonized_cohort,
    TRACK_A_FEATURES,
    TRACK_B_10_FEATURES,
    TRACK_B_11_FEATURES
)
from src.preprocessing import (
    mask_clinical_zero_sentinels,
    HarmonizedClinicalPreprocessor,
    prepare_train_test_split
)


class TestClinicalHarmonization(unittest.TestCase):

    def setUp(self):
        # Create representative test dataframe
        self.sample_df = pd.DataFrame({
            "age": [55.0, 60.0, 48.0, 65.0],
            "sex": [1.0, 0.0, 1.0, 1.0],
            "cp": [4.0, 2.0, 3.0, 1.0],
            "trestbps": [140.0, 0.0, 130.0, 150.0],  # One sentinel zero
            "chol": [240.0, 0.0, 190.0, 220.0],      # One sentinel zero
            "fbs": [0.0, 0.0, 1.0, 0.0],
            "restecg": [0.0, 1.0, 0.0, 2.0],
            "thalach": [150.0, 120.0, 160.0, 110.0],
            "exang": [0.0, 1.0, 0.0, 1.0],
            "oldpeak": [-1.2, 0.0, 2.4, 0.0],       # Signed negative oldpeak and legitimate zeros
            "slope": [2.0, 1.0, 2.0, 3.0],
            "ca": [0.0, 1.0, 0.0, 2.0],              # Legitimate zero ca
            "thal": [3.0, 6.0, 3.0, 7.0],
            "target": [1, 0, 1, 0]
        })

    def test_zero_sentinel_masking(self):
        """Verify non-physiological zeros are masked while valid clinical zeros are kept."""
        masked = mask_clinical_zero_sentinels(self.sample_df)

        # Sentinels should be masked
        self.assertTrue(np.isnan(masked.loc[1, "trestbps"]))
        self.assertTrue(np.isnan(masked.loc[1, "chol"]))

        # Valid zeros must be preserved
        self.assertEqual(masked.loc[0, "exang"], 0.0)
        self.assertEqual(masked.loc[1, "oldpeak"], 0.0)
        self.assertEqual(masked.loc[0, "ca"], 0.0)
        self.assertEqual(masked.loc[1, "sex"], 0.0)

    def test_harmonized_preprocessor_leak_free(self):
        """Verify preprocessor fits strictly on train and transforms out-of-sample data."""
        train_df = self.sample_df.iloc[:3].copy()
        test_df = self.sample_df.iloc[3:].copy()

        preprocessor = HarmonizedClinicalPreprocessor(
            imputer_strategy="knn",
            n_neighbors=2,
            clamp_negative_oldpeak=False
        )

        # Fit on train only
        X_train_imp = preprocessor.fit_transform(train_df[TRACK_B_10_FEATURES])
        self.assertTrue(preprocessor.is_fitted_)
        self.assertEqual(X_train_imp.isna().sum().sum(), 0)

        # Transform test out-of-sample
        X_test_imp = preprocessor.transform(test_df[TRACK_B_10_FEATURES])
        self.assertEqual(X_test_imp.isna().sum().sum(), 0)
        self.assertEqual(list(X_test_imp.columns), TRACK_B_10_FEATURES)

    def test_signed_oldpeak_handling(self):
        """Verify signed oldpeak is preserved when clamp=False and clipped when clamp=True."""
        prep_signed = HarmonizedClinicalPreprocessor(clamp_negative_oldpeak=False)
        prep_signed.fit(self.sample_df[TRACK_B_10_FEATURES])
        res_signed = prep_signed.transform(self.sample_df[TRACK_B_10_FEATURES])
        self.assertAlmostEqual(res_signed.loc[0, "oldpeak"], -1.2, places=1)

        prep_clamped = HarmonizedClinicalPreprocessor(clamp_negative_oldpeak=True)
        prep_clamped.fit(self.sample_df[TRACK_B_10_FEATURES])
        res_clamped = prep_clamped.transform(self.sample_df[TRACK_B_10_FEATURES])
        self.assertEqual(res_clamped.loc[0, "oldpeak"], 0.0)

    def test_load_harmonized_cohort_tracks(self):
        """Verify Track A and Track B loadings for Cleveland."""
        clev_a = load_harmonized_cohort("cleveland", track="A")
        self.assertEqual(len(clev_a), 303)
        self.assertIn("ca", clev_a.columns)
        self.assertIn("thal", clev_a.columns)

        clev_b = load_harmonized_cohort("cleveland", track="B10")
        self.assertEqual(len(clev_b), 303)
        self.assertNotIn("ca", clev_b.columns)
        self.assertNotIn("thal", clev_b.columns)
        self.assertNotIn("chol", clev_b.columns)

    def test_deduplication_in_hungarian_and_va(self):
        """Verify verified duplicates are removed."""
        hung = load_harmonized_cohort("hungarian", track="B10", drop_duplicates=True)
        self.assertEqual(len(hung), 293)  # 294 - 1 duplicate = 293

        va = load_harmonized_cohort("va_long_beach", track="B10", drop_duplicates=True)
        self.assertEqual(len(va), 199)    # 200 - 1 duplicate = 199

    def test_prepare_train_test_split_dual_track(self):
        """Verify prepare_train_test_split outputs complete matrices without missingness."""
        clev = load_harmonized_cohort("cleveland", track="B10")
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            clev,
            feature_set=TRACK_B_10_FEATURES,
            test_size=0.2,
            random_state=42,
            impute_missing=True
        )
        self.assertEqual(X_train.isna().sum().sum(), 0)
        self.assertEqual(X_test.isna().sum().sum(), 0)
        self.assertEqual(len(X_train) + len(X_test), len(clev))


if __name__ == "__main__":
    unittest.main()
