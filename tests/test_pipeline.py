"""
Unit Tests for the CardioPulse Cleveland-cohort Clinical AI Framework:
- Cleveland feature schema handling
- SMOTE physiological boundary validity auditing
- Probability calibration & Expected Calibration Error (ECE)
- Dual-XAI Concordance Index (C_i) calculation & triage threshold tagging
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.dataset import (
    TRACK_A_FEATURES,
    create_synthetic_heart_dataset
)
from src.preprocessing import (
    prepare_train_test_split,
    audit_synthetic_samples,
    compare_imbalance_strategies
)
from src.evaluate import (
    compute_expected_calibration_error,
    evaluate_classifier,
    compute_calibration_data
)
from src.models import get_model_zoo, build_calibrated_model
from src.explainability import ClinicalExplainabilitySuite


class TestCardioPulsePipeline(unittest.TestCase):

    def test_cleveland_feature_structure(self):
        self.assertEqual(len(TRACK_A_FEATURES), 13)
        self.assertTrue("ca" in TRACK_A_FEATURES and "thal" in TRACK_A_FEATURES)

    def test_synthetic_audit_detects_violations(self):
        # Construct synthetic rows with obvious physiological violations
        violated_df = pd.DataFrame([
            {"age": 50, "trestbps": 260.0, "chol": 180, "thalach": 150, "oldpeak": 1.0}, # BP too high (>220)
            {"age": 70, "trestbps": 120.0, "chol": 200, "thalach": 215, "oldpeak": 0.5}, # HR impossible for age (215 > 220-70+15=165)
            {"age": 45, "trestbps": 120.0, "chol": 210, "thalach": 160, "oldpeak": -1.5}, # Negative oldpeak
        ])
        audit = audit_synthetic_samples(violated_df)
        self.assertEqual(audit["total_synthetic_samples"], 3)
        self.assertGreater(audit["clinical_violation_rate_pct"], 60.0)
        self.assertGreaterEqual(audit["violations_breakdown"]["trestbps_out_of_bounds"], 1)
        self.assertGreaterEqual(audit["violations_breakdown"]["impossible_hr_for_age"], 1)
        self.assertGreaterEqual(audit["violations_breakdown"]["oldpeak_negative_or_extreme"], 1)

    def test_expected_calibration_error_computation(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        # Perfectly calibrated probabilities matching true labels
        y_proba_perfect = np.array([0.05, 0.10, 0.90, 0.95, 0.85, 0.15, 0.80, 0.10, 0.90, 0.85])
        ece_low = compute_expected_calibration_error(y_true, y_proba_perfect, n_bins=5)
        
        # Completely uncalibrated probabilities (inverted)
        y_proba_bad = 1.0 - y_proba_perfect
        ece_high = compute_expected_calibration_error(y_true, y_proba_bad, n_bins=5)
        
        self.assertLess(ece_low, 0.20)
        self.assertGreater(ece_high, 0.50)

    def test_dual_xai_concordance_engine(self):
        # Create small dataset and model
        df = create_synthetic_heart_dataset(n_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = prepare_train_test_split(df, feature_set=TRACK_A_FEATURES, test_size=0.2)
        
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        
        xai = ClinicalExplainabilitySuite(clf, X_train)
        patient_df = X_test.iloc[[0]]
        
        shap_res = xai.explain_patient_shap(patient_df)
        lime_res = xai.explain_patient_lime(patient_df, num_features=5)
        
        concordance = xai.compute_explainer_concordance(shap_res, lime_res)
        
        self.assertIn("concordance_index", concordance)
        self.assertTrue(0.0 <= concordance["concordance_index"] <= 1.0)
        self.assertIn(concordance["audit_action"], ["Automated Clinical Recommendation", "Flag for Secondary Clinical Review"])
        self.assertIn("audit_badge", concordance)


if __name__ == "__main__":
    unittest.main()
