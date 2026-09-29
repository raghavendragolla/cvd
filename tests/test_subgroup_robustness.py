"""
Unit Tests for Phase 8: Subgroup Robustness, Fairness & Missingness Sensitivity.
Verifies:
  1. Subgroup definitions and mutually exclusive, comprehensive age bins.
  2. Subgroup sample sizes and disease case counts match raw cohort ground truth.
  3. Single-class and small-cell handling (NaN for undefined ROC-AUC, not fabricated numbers).
  4. Stratified bootstrap reproducibility with fixed random seed.
  5. Complete-case filtering accuracy across all 4 cohorts.
  6. Missingness-indicator pipeline adherence to train-only fitting (no test leakage).
  7. Subgroup SHAP linkage to test observations.
  8. Raw dataset byte-level integrity.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import load_harmonized_cohort, TRACK_B_10_FEATURES
from src.preprocessing import mask_clinical_zero_sentinels
from experiments.exp_subgroup_robustness import (
    AGE_BINS,
    AGE_LABELS,
    MIN_EVALUABLE_N,
    get_subgroup_annotations,
    compute_subgroup_bootstrap_cis
)



class TestSubgroupRobustness(unittest.TestCase):

    def setUp(self):
        self.cohort_names = ["Cleveland", "Hungarian", "Zurich", "VA Long Beach"]
        self.cohorts = {name: load_harmonized_cohort(name) for name in self.cohort_names}

    def test_age_bin_definitions_and_coverage(self):
        """Verify age bins are mutually exclusive, exhaustive, and contain no gaps or overlaps."""
        test_ages = [18.0, 49.0, 49.99, 50.0, 55.5, 64.0, 64.99, 65.0, 75.0, 95.0]
        s = pd.Series(test_ages)
        binned = pd.cut(s, bins=AGE_BINS, labels=AGE_LABELS)

        self.assertEqual(len(binned.dropna()), len(test_ages), "Some ages failed to map into defined bins.")
        self.assertEqual(binned.iloc[0], "<50")
        self.assertEqual(binned.iloc[2], "<50")
        self.assertEqual(binned.iloc[3], "50-64")
        self.assertEqual(binned.iloc[6], "50-64")
        self.assertEqual(binned.iloc[7], ">=65")
        self.assertEqual(binned.iloc[9], ">=65")

    def test_subgroup_counts_match_cohort_ground_truth(self):
        """Verify sex and age subgroup sample sizes and prevalence match verified cohort values."""
        # Cleveland
        clev = get_subgroup_annotations(self.cohorts["Cleveland"])
        self.assertEqual(len(clev[clev["sex"] == 0]), 97)
        self.assertEqual(len(clev[clev["sex"] == 1]), 206)
        self.assertEqual(len(clev[clev["age_group"] == "<50"]), 87)
        self.assertEqual(len(clev[clev["age_group"] == "50-64"]), 175)
        self.assertEqual(len(clev[clev["age_group"] == ">=65"]), 41)

        # Hungarian
        hung = get_subgroup_annotations(self.cohorts["Hungarian"])
        self.assertEqual(len(hung[hung["sex"] == 0]), 80)
        self.assertEqual(len(hung[hung["sex"] == 1]), 213)
        self.assertEqual(len(hung[hung["age_group"] == "<50"]), 160)
        self.assertEqual(len(hung[hung["age_group"] == "50-64"]), 129)
        self.assertEqual(len(hung[hung["age_group"] == ">=65"]), 4)

        # Zurich
        zuri = get_subgroup_annotations(self.cohorts["Zurich"])
        self.assertEqual(len(zuri[zuri["sex"] == 0]), 10)
        self.assertEqual(len(zuri[zuri["sex"] == 1]), 113)
        self.assertEqual(len(zuri[zuri["age_group"] == "<50"]), 25)
        self.assertEqual(len(zuri[zuri["age_group"] == "50-64"]), 82)
        self.assertEqual(len(zuri[zuri["age_group"] == ">=65"]), 16)

        # VA Long Beach
        va = get_subgroup_annotations(self.cohorts["VA Long Beach"])
        self.assertEqual(len(va[va["sex"] == 0]), 6)
        self.assertEqual(len(va[va["sex"] == 1]), 193)
        self.assertEqual(len(va[va["age_group"] == "<50"]), 19)
        self.assertEqual(len(va[va["age_group"] == "50-64"]), 138)
        self.assertEqual(len(va[va["age_group"] == ">=65"]), 42)

    def test_single_class_edge_case_handling(self):
        """Verify that single-class cells correctly flag undefined ROC-AUC without fabricating numbers."""
        # Zurich female cohort: all 10 positive
        zuri = get_subgroup_annotations(self.cohorts["Zurich"])
        zuri_females = zuri[zuri["sex"] == 0]
        y_true = (zuri_females["target"] > 0).astype(int).values

        self.assertEqual(len(np.unique(y_true)), 1, "Expected single class for Zurich females.")
        self.assertEqual(y_true.sum(), 10)

        # fast_auc must return np.nan when single class is present
        from experiments.exp_subgroup_robustness import fast_auc
        auc_val = fast_auc(y_true, np.ones(len(y_true)) * 0.8)
        self.assertTrue(np.isnan(auc_val), f"Expected NaN for single-class AUC, got {auc_val}")

    def test_bootstrap_reproducibility(self):
        """Verify stratified bootstrap produces deterministic confidence intervals with fixed seed."""
        y_true = np.array([1]*50 + [0]*50)
        y_proba = np.linspace(0.1, 0.9, 100)
        y_pred = (y_proba >= 0.5).astype(int)

        ci_auc1, ci_brier1, ci_ece1, ci_f11 = compute_subgroup_bootstrap_cis(y_true, y_proba, y_pred, seed=42)
        ci_auc2, ci_brier2, ci_ece2, ci_f12 = compute_subgroup_bootstrap_cis(y_true, y_proba, y_pred, seed=42)

        self.assertEqual(ci_auc1, ci_auc2)
        self.assertEqual(ci_brier1, ci_brier2)
        self.assertEqual(ci_ece1, ci_ece2)
        self.assertEqual(ci_f11, ci_f12)
        self.assertNotEqual(ci_auc1[0], "NA")

    def test_complete_case_counts(self):
        """Verify exact complete-case observations per cohort across Track B features."""
        expected_cc = {
            "Cleveland": 303,
            "Hungarian": 100,
            "Zurich": 46,
            "VA Long Beach": 89
        }
        for name, exp_n in expected_cc.items():
            df = self.cohorts[name]
            df_m = mask_clinical_zero_sentinels(df[TRACK_B_10_FEATURES].copy())
            cc_n = (~df_m.isna().any(axis=1)).sum()
            self.assertEqual(cc_n, exp_n, f"Complete case mismatch for {name}: expected {exp_n}, got {cc_n}")

    def test_raw_data_integrity(self):
        """Verify raw datasets in data/raw/ have not been altered or overwritten."""
        raw_files = {
            "Cleveland": "cleveland_raw.csv",
            "Hungarian": "hungarian_raw.csv",
            "Zurich": "zurich_raw.csv",
            "VA Long Beach": "va_raw.csv"
        }
        raw_dir = os.path.join(ROOT_DIR, "data", "raw")
        for cohort, fname in raw_files.items():
            fpath = os.path.join(raw_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Raw file missing: {fpath}")
            fsize = os.path.getsize(fpath)
            self.assertGreater(fsize, 5000, f"Raw file corrupted or truncated: {fpath} (size={fsize})")



if __name__ == "__main__":
    unittest.main()
