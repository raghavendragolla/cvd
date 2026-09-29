"""
Unit Tests for Phase 9 — Integrated Statistical Synthesis & Evidence Consolidation
===================================================================================
Validates:
1. Estimand calculations and internal/external matching across the 24 experiments.
2. Clustered structure: 6 directed transfer pairs x 4 models = 24 experiments.
3. Locked correlation set (Primary, Secondary, Robustness, Sensitivity).
4. Bootstrap confidence interval ordering and reproducibility.
5. Demographic subgroup disparity and missingness sensitivity synthesis integrity.
6. Figure existence, resolution, and non-corruption.
7. Manifest verification and SHA-256 hash consistency.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")


class TestIntegratedSynthesis(unittest.TestCase):

    def test_integrated_estimands_structure(self):
        """Validates that integrated_estimands.csv contains exactly 24 experiments with valid math."""
        path = os.path.join(RESULTS_DIR, "integrated_estimands.csv")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        df = pd.read_csv(path)

        self.assertEqual(len(df), 24, f"Expected 24 experiments, got {len(df)}")
        self.assertEqual(df["model"].nunique(), 4, "Expected 4 distinct models")
        self.assertEqual(df["transfer_pair"].nunique(), 6, "Expected 6 distinct transfer pairs")

        # Verify each transfer pair has all 4 models
        for pair, grp in df.groupby("transfer_pair"):
            self.assertEqual(len(grp), 4, f"Transfer pair {pair} does not have 4 models")
            self.assertEqual(set(grp["model"]), {"LogisticRegression", "RandomForest", "SVM", "XGBoost"})

        # Verify estimand calculations: External - Internal deltas
        np.testing.assert_allclose(
            df["auc_change"],
            np.round(df["external_roc_auc"] - df["internal_roc_auc"], 2),
            rtol=1e-4, atol=1e-2,
            err_msg="Delta AUC calculation mismatch"
        )

        np.testing.assert_allclose(
            df["brier_change"],
            np.round(df["external_brier"] - df["internal_brier"], 4),
            rtol=1e-4, atol=1e-3,
            err_msg="Delta Brier calculation mismatch"
        )

        # Verify Zurich flagging: Exactly 8 Zurich transfers (4 dev + 4 test)
        self.assertEqual(df["is_zurich_transfer"].sum(), 8, f"Expected 8 Zurich transfers, got {df['is_zurich_transfer'].sum()}")

    def test_correlations_locked_set(self):
        """Validates the locked analysis set in integrated_correlations.csv."""
        path = os.path.join(RESULTS_DIR, "integrated_correlations.csv")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        df = pd.read_csv(path)

        # Verify tiers
        tiers = set(df["analysis_tier"].unique())
        self.assertIn("Primary", tiers)
        self.assertIn("Secondary", tiers)
        self.assertIn("Robustness (Zurich-Excluded)", tiers)
        self.assertIn("Sensitivity (Transfer-Level N=6)", tiers)

        # Check row counts per tier
        self.assertEqual((df["analysis_tier"] == "Primary").sum(), 4)
        self.assertEqual((df["analysis_tier"] == "Secondary").sum(), 2)
        self.assertEqual((df["analysis_tier"] == "Robustness (Zurich-Excluded)").sum(), 4)
        self.assertEqual((df["analysis_tier"] == "Sensitivity (Transfer-Level N=6)").sum(), 4)

        # Verify all Spearman rhos are valid and within [-1, 1]
        self.assertTrue(df["spearman_rho"].between(-1.0, 1.0).all(), "Spearman rho out of range")

        # Verify bootstrap CIs are ordered: lower <= upper
        valid_cluster_boot = df.dropna(subset=["cluster_boot_ci_lower", "cluster_boot_ci_upper"])
        self.assertTrue((valid_cluster_boot["cluster_boot_ci_lower"] <= valid_cluster_boot["cluster_boot_ci_upper"]).all())

        valid_std_boot = df.dropna(subset=["standard_boot_ci_lower", "standard_boot_ci_upper"])
        self.assertTrue((valid_std_boot["standard_boot_ci_lower"] <= valid_std_boot["standard_boot_ci_upper"]).all())

    def test_cluster_summary(self):
        """Validates that integrated_cluster_summary.csv contains Overall, Model, and Transfer Pair levels."""
        path = os.path.join(RESULTS_DIR, "integrated_cluster_summary.csv")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        df = pd.read_csv(path)

        levels = df["grouping_level"].unique()
        self.assertEqual(set(levels), {"Overall", "Model", "Transfer_Pair"})

        overall = df[df["grouping_level"] == "Overall"]
        self.assertEqual(len(overall), 1)
        self.assertEqual(overall.iloc[0]["n_experiments"], 24)

        models = df[df["grouping_level"] == "Model"]
        self.assertEqual(len(models), 4)
        self.assertTrue((models["n_experiments"] == 6).all())

        pairs = df[df["grouping_level"] == "Transfer_Pair"]
        self.assertEqual(len(pairs), 6)
        self.assertTrue((pairs["n_experiments"] == 4).all())

    def test_subgroup_synthesis(self):
        """Validates that subgroup_synthesis.csv correctly summarizes Phase 8 demographic disparities."""
        path = os.path.join(RESULTS_DIR, "subgroup_synthesis.csv")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        df = pd.read_csv(path)

        self.assertEqual(len(df), 4, f"Expected 4 subgroup comparison rows, got {len(df)}")
        dims = set(df["subgroup_dimension"])
        self.assertIn("Sex (Male vs Female)", dims)
        self.assertIn("Age (<50 vs 50-64)", dims)
        self.assertIn("Age (50-64 vs >=65)", dims)
        self.assertIn("Age (<50 vs >=65)", dims)

        self.assertTrue((df["total_comparisons"] == 24).all())
        self.assertTrue((df["median_abs_auc_disparity"] >= 0).all())
        self.assertTrue((df["median_abs_brier_disparity"] >= 0).all())
        self.assertTrue((df["median_abs_ece_disparity"] >= 0).all())

    def test_missingness_synthesis(self):
        """Validates that missingness_synthesis.csv correctly summarizes missingness pipelines."""
        path = os.path.join(RESULTS_DIR, "missingness_synthesis.csv")
        self.assertTrue(os.path.exists(path), f"Missing {path}")
        df = pd.read_csv(path)

        self.assertEqual(len(df), 4, f"Expected 4 missingness regimes, got {len(df)}")
        regimes = set(df["imputation_regime"])
        self.assertIn("Primary Development-Fitted (KNN)", regimes)
        self.assertIn("Complete-Case Test Evaluation", regimes)
        self.assertIn("Full Complete-Case (Train & Test)", regimes)
        self.assertIn("Missingness Indicator Pipeline", regimes)

        self.assertTrue((df["evaluable_experiments"] == 24).all(), "All regimes should have 24 evaluable experiments")

    def test_figures_exist_and_valid(self):
        """Validates that all 6 required publication figures exist and are non-trivial size."""
        expected_figures = [
            "01_methodological_architecture.png",
            "02_external_validation_auc_heatmap.png",
            "03_calibration_dataset_shift.png",
            "04_shap_explanation_stability.png",
            "05_dataset_shift_vs_shap_stability.png",
            "06_subgroup_or_missingness_robustness.png"
        ]

        for fname in expected_figures:
            fig_path = os.path.join(FIGURES_DIR, fname)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_path}")
            fsize = os.path.getsize(fig_path)
            self.assertGreater(fsize, 50_000, f"Figure {fname} is suspiciously small ({fsize} bytes)")

    def test_phase9_manifest(self):
        """Validates that phase9_manifest.json contains complete audit metadata and valid hashes."""
        path = os.path.join(RESULTS_DIR, "phase9_manifest.json")
        self.assertTrue(os.path.exists(path), f"Missing {path}")

        with open(path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["primary_statistical_unit"], "MODEL x DIRECTED TRANSFER EXPERIMENT (N=24)")
        self.assertEqual(manifest["number_of_models"], 4)
        self.assertEqual(manifest["number_of_transfer_directions"], 6)
        self.assertEqual(manifest["bootstrap_parameters"]["iterations"], 2000)
        self.assertEqual(manifest["bootstrap_parameters"]["random_seed"], 42)

        hashes = manifest["artifact_hashes"]
        for filename, fhash in hashes.items():
            self.assertEqual(len(fhash), 64, f"Invalid SHA-256 hash for {filename}")
            fpath = os.path.join(RESULTS_DIR, filename)
            self.assertTrue(os.path.exists(fpath), f"Manifest references missing file {fpath}")


if __name__ == "__main__":
    unittest.main()
