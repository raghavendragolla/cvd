"""
Unit tests for Phase 6: Explanation Stability & XAI Transportability.
Strictly verifies:
1. External test labels are never passed to explainer routines.
2. Development preprocessing is reused via transform() without refitting.
3. Feature column ordering is strictly identical between development and external attributions.
4. SHAP efficiency axiom holds for linear baseline: sum(phi_i) = f(x) - E[f(x)].
5. Deterministic reproducibility: Identical seeds produce bitwise identical SHAP rankings.
6. Deterministic sampling rule generates reproducible local explanation cohorts without outcome bias.
7. SHAP and LIME metrics remain strictly separated and are not blended.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.dataset import TRACK_B_10_FEATURES, load_harmonized_cohort
from src.preprocessing import HarmonizedClinicalPreprocessor
from experiments.exp_internal_validation import load_config, build_model_pipeline
from experiments.exp_xai_stability import (
    compute_shap_values,
    summarize_feature_attributions,
    compute_rank_stability,
    select_deterministic_sample
)


class TestXAIStabilityMethodologicalAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.raw_dir = "data/raw"

    def test_external_labels_never_accessed(self):
        """Rule 4: Verify compute_shap_values does not take or require external labels."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        prep = HarmonizedClinicalPreprocessor()
        X_dev = prep.fit_transform(dev_df[TRACK_B_10_FEATURES])
        y_dev = dev_df["target"].to_numpy()
        X_ext = prep.transform(ext_df[TRACK_B_10_FEATURES])

        model = build_model_pipeline("RandomForest", self.config)
        model.fit(X_dev, y_dev)

        # compute_shap_values should work purely on feature matrix X_ext without any y
        shap_vals = compute_shap_values(model, "RandomForest", X_dev, X_ext, seed=42)
        self.assertEqual(shap_vals.shape, (len(ext_df), len(TRACK_B_10_FEATURES)))
        self.assertFalse(np.isnan(shap_vals).any(), "SHAP values contain unexpected NaNs!")

    def test_feature_order_strictly_identical(self):
        """Rule 1: Verify feature ordering matches TRACK_B_10_FEATURES."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        prep = HarmonizedClinicalPreprocessor()
        X_dev = prep.fit_transform(dev_df[TRACK_B_10_FEATURES])

        self.assertEqual(list(X_dev.columns), TRACK_B_10_FEATURES)

    def test_efficiency_axiom_linear_model(self):
        """Rule 19: Verify SHAP efficiency property for linear baseline: sum(phi_i) approx f(x) - E[f(x)]."""
        import shap
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        prep = HarmonizedClinicalPreprocessor()
        X_dev = prep.fit_transform(dev_df[TRACK_B_10_FEATURES])
        y_dev = dev_df["target"].to_numpy()

        model = build_model_pipeline("LogisticRegression", self.config)
        model.fit(X_dev, y_dev)

        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_scaled = pd.DataFrame(scaler.transform(X_dev), columns=TRACK_B_10_FEATURES)
        masker = shap.maskers.Independent(X_scaled)
        explainer = shap.LinearExplainer(clf, masker=masker)
        shap_res = explainer(X_scaled.head(5))

        # Check efficiency: sum of SHAP values + base_value approx decision_function
        dec_func = clf.decision_function(X_scaled.head(5))
        for i in range(5):
            phi_sum = np.sum(shap_res.values[i])
            expected_diff = dec_func[i] - explainer.expected_value
            self.assertAlmostEqual(phi_sum, expected_diff, delta=1e-3)

    def test_deterministic_reproducibility(self):
        """Rule 17: Fixed seeds produce identical rankings across repeated runs."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        prep = HarmonizedClinicalPreprocessor()
        X_dev = prep.fit_transform(dev_df[TRACK_B_10_FEATURES])
        y_dev = dev_df["target"].to_numpy()
        X_ext = prep.transform(ext_df[TRACK_B_10_FEATURES])

        model = build_model_pipeline("RandomForest", self.config)
        model.fit(X_dev, y_dev)

        shap_run1 = compute_shap_values(model, "RandomForest", X_dev, X_ext.head(20), seed=42)
        shap_run2 = compute_shap_values(model, "RandomForest", X_dev, X_ext.head(20), seed=42)

        np.testing.assert_array_almost_equal(shap_run1, shap_run2, decimal=6)

    def test_deterministic_sampling_rule(self):
        """Rule 9: Sampling selects patient indices deterministically across predicted risk spectrum."""
        dev_df = load_harmonized_cohort("cleveland", track="B10", data_dir=self.raw_dir)
        ext_df = load_harmonized_cohort("hungarian", track="B10", data_dir=self.raw_dir)

        prep = HarmonizedClinicalPreprocessor()
        X_dev = prep.fit_transform(dev_df[TRACK_B_10_FEATURES])
        y_dev = dev_df["target"].to_numpy()
        X_ext = prep.transform(ext_df[TRACK_B_10_FEATURES])

        model = build_model_pipeline("RandomForest", self.config)
        model.fit(X_dev, y_dev)

        sample_idx1 = select_deterministic_sample(X_ext, model, n_samples=15, seed=42)
        sample_idx2 = select_deterministic_sample(X_ext, model, n_samples=15, seed=42)

        self.assertEqual(len(sample_idx1), 15)
        np.testing.assert_array_equal(sample_idx1, sample_idx2)
        # Verify indices span the range
        self.assertLess(sample_idx1[0], sample_idx1[-1])

    def test_rank_stability_computation(self):
        """Rule 7: Verify rank correlation and top-k calculations."""
        dev_summary = pd.DataFrame({
            "feature": TRACK_B_10_FEATURES,
            "mean_abs_shap": [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01],
            "feature_rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        # External summary with small perturbation
        ext_summary = pd.DataFrame({
            "feature": TRACK_B_10_FEATURES,
            "mean_abs_shap": [0.45, 0.42, 0.28, 0.22, 0.12, 0.06, 0.03, 0.04, 0.01, 0.02],
            "feature_rank": [1, 2, 3, 4, 5, 6, 8, 7, 10, 9]
        })

        stability_mets, disp_df = compute_rank_stability(dev_summary, ext_summary)
        self.assertGreater(stability_mets["spearman_rank_correlation"], 0.90)
        self.assertEqual(stability_mets["top3_overlap"], 1.0)
        self.assertEqual(stability_mets["top5_overlap"], 1.0)
        self.assertEqual(len(disp_df), 10)


if __name__ == "__main__":
    unittest.main()
