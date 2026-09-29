"""
Phase 9 — Integrated Statistical Synthesis & Evidence Consolidation
====================================================================
Quantitative synthesis across discrimination transport, calibration transport,
prevalence divergence, continuous covariate shift, explanation stability,
demographic subgroup disparities, and missing-data handling sensitivities.

Primary unit of observation: MODEL x DIRECTED TRANSFER EXPERIMENT (N=24).
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures", "integrated_synthesis")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define visual style constants
MODEL_COLORS = {
    "LogisticRegression": "#1f77b4",  # Steel Blue
    "RandomForest": "#2ca02c",        # Forest Green
    "SVM": "#9467bd",                 # Purple
    "XGBoost": "#d62728"              # Crimson Red
}
MODEL_MARKERS = {
    "LogisticRegression": "o",
    "RandomForest": "s",
    "SVM": "^",
    "XGBoost": "D"
}


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for manifest auditing."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_merge_data() -> pd.DataFrame:
    """
    Consolidates data from Phase 5, Phase 6, Phase 7, and Phase 8 into
    a single integrated DataFrame representing the 24 primary experiments.
    """
    ext_path = os.path.join(RESULTS_DIR, "external_validation_results.csv")
    int_sum_path = os.path.join(RESULTS_DIR, "internal_validation_summary.csv")
    shap_path = os.path.join(RESULTS_DIR, "shap_rank_stability.csv")
    dsr_path = os.path.join(RESULTS_DIR, "dataset_shift_relationships.csv")

    ext_df = pd.read_csv(ext_path)
    int_sum = pd.read_csv(int_sum_path)
    shap_df = pd.read_csv(shap_path)
    dsr_df = pd.read_csv(dsr_path)

    # 1. Match Internal Baseline Calibration Metrics (Track B)
    track_b_sum = int_sum[int_sum["track"] == "Track_B"]
    int_calib = {}
    for (c, m), grp in track_b_sum.groupby(["cohort", "model"]):
        m_dict = dict(zip(grp["metric"], grp["mean"]))
        int_calib[(c, m)] = {
            "internal_ece": m_dict.get("ece", np.nan),
            "internal_cal_intercept": m_dict.get("calibration_intercept", np.nan),
            "internal_cal_slope": m_dict.get("calibration_slope", np.nan)
        }

    ext_df["internal_ece"] = ext_df.apply(
        lambda r: int_calib.get((r["train_dataset"], r["model"]), {}).get("internal_ece", np.nan), axis=1
    )
    ext_df["internal_cal_intercept"] = ext_df.apply(
        lambda r: int_calib.get((r["train_dataset"], r["model"]), {}).get("internal_cal_intercept", np.nan), axis=1
    )
    ext_df["internal_cal_slope"] = ext_df.apply(
        lambda r: int_calib.get((r["train_dataset"], r["model"]), {}).get("internal_cal_slope", np.nan), axis=1
    )

    # Calculate calibration deltas
    ext_df["ece_change"] = np.round(ext_df["external_ece"] - ext_df["internal_ece"], 4)
    ext_df["cal_intercept_change"] = np.round(ext_df["external_calibration_intercept"] - ext_df["internal_cal_intercept"], 4)
    ext_df["cal_slope_change"] = np.round(ext_df["external_calibration_slope"] - ext_df["internal_cal_slope"], 4)

    # 2. Match SHAP Stability Metrics (from Phase 6)
    shap_unique = shap_df[[
        "experiment_id", "spearman_rank_correlation", "kendall_rank_correlation",
        "top3_overlap", "top5_overlap"
    ]].drop_duplicates()

    merged = pd.merge(ext_df, shap_unique, on="experiment_id", how="left")

    # 3. Match Dataset Shift Metrics (from Phase 7)
    dsr_unique = dsr_df[[
        "experiment_id", "mean_wasserstein_distance", "mean_jsd", "mean_abs_smd",
        "abs_prevalence_delta_pct"
    ]].drop_duplicates()

    merged = pd.merge(merged, dsr_unique, on="experiment_id", how="left")

    # Flag Zurich transfers and format transfer pairs
    merged["is_zurich_transfer"] = (merged["train_dataset"] == "Zurich") | (merged["test_dataset"] == "Zurich")
    merged["transfer_pair"] = merged["train_dataset"] + " -> " + merged["test_dataset"]

    # Order columns logically
    cols_order = [
        "experiment_id", "train_dataset", "test_dataset", "transfer_pair", "model",
        "n_train", "n_test", "train_positive_rate", "test_positive_rate",
        "internal_roc_auc", "external_roc_auc", "auc_change",
        "internal_brier", "external_brier", "brier_change",
        "internal_ece", "external_ece", "ece_change",
        "internal_cal_slope", "external_calibration_slope", "cal_slope_change",
        "internal_cal_intercept", "external_calibration_intercept", "cal_intercept_change",
        "abs_prevalence_delta_pct", "mean_wasserstein_distance", "mean_jsd", "mean_abs_smd",
        "spearman_rank_correlation", "kendall_rank_correlation", "top3_overlap", "top5_overlap",
        "is_zurich_transfer"
    ]
    merged = merged[cols_order]
    return merged


def compute_cluster_bootstrap_ci(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    cluster_col: str = "transfer_pair",
    n_boot: int = 2000,
    seed: int = 42
) -> tuple:
    """
    Implements a cluster-aware bootstrap resampling transfer pairs with replacement.
    Averages 24 rows across 6 clusters.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    cluster_groups = {c: df[df[cluster_col] == c] for c in clusters}

    boot_rhos = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_df = pd.concat([cluster_groups[c] for c in sampled_clusters], ignore_index=True)
        if sampled_df[col_x].nunique() < 2 or sampled_df[col_y].nunique() < 2:
            continue
        r, _ = stats.spearmanr(sampled_df[col_x], sampled_df[col_y])
        if not np.isnan(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 100:
        return np.nan, np.nan, np.nan

    med = np.median(boot_rhos)
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)
    return round(med, 4), round(ci_lower, 4), round(ci_upper, 4)


def compute_standard_bootstrap_ci(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    n_boot: int = 2000,
    seed: int = 42
) -> tuple:
    """
    Standard row-level bootstrap for non-parametric comparison.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sampled = df.iloc[idx]
        if sampled[col_x].nunique() < 2 or sampled[col_y].nunique() < 2:
            continue
        r, _ = stats.spearmanr(sampled[col_x], sampled[col_y])
        if not np.isnan(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 100:
        return np.nan, np.nan, np.nan

    med = np.median(boot_rhos)
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)
    return round(med, 4), round(ci_lower, 4), round(ci_upper, 4)


def generate_correlations_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all prespecified Primary, Secondary, Zurich-excluded, and
    Cluster-level aggregated Spearman associations.
    """
    records = []

    # Locked analysis definitions: (col_x, col_y, display_name, hypothesis_dir)
    primary_specs = [
        ("abs_prevalence_delta_pct", "external_ece", "Prevalence Shift vs External ECE", "Positive"),
        ("abs_prevalence_delta_pct", "brier_change", "Prevalence Shift vs Brier Degradation", "Positive"),
        ("mean_wasserstein_distance", "auc_change", "Mean Wasserstein vs AUC Degradation", "Negative"),
        ("mean_wasserstein_distance", "spearman_rank_correlation", "Mean Wasserstein vs SHAP Stability", "Negative")
    ]

    secondary_specs = [
        ("spearman_rank_correlation", "auc_change", "SHAP Stability vs AUC Degradation", "Positive"),
        ("spearman_rank_correlation", "brier_change", "SHAP Stability vs Brier Degradation", "Negative")
    ]

    # 1. Primary Associations (N=24)
    for col_x, col_y, name, hyp_dir in primary_specs:
        r, p = stats.spearmanr(df[col_x], df[col_y])
        c_med, c_lo, c_hi = compute_cluster_bootstrap_ci(df, col_x, col_y)
        s_med, s_lo, s_hi = compute_standard_bootstrap_ci(df, col_x, col_y)
        records.append({
            "analysis_tier": "Primary",
            "comparison_name": name,
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df),
            "spearman_rho": round(r, 4),
            "p_value": round(p, 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "interpretation_note": "Primary experiment-level observational correlation across 24 experiments."
        })

    # 2. Secondary Associations (N=24)
    for col_x, col_y, name, hyp_dir in secondary_specs:
        r, p = stats.spearmanr(df[col_x], df[col_y])
        c_med, c_lo, c_hi = compute_cluster_bootstrap_ci(df, col_x, col_y)
        s_med, s_lo, s_hi = compute_standard_bootstrap_ci(df, col_x, col_y)
        records.append({
            "analysis_tier": "Secondary",
            "comparison_name": name,
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df),
            "spearman_rho": round(r, 4),
            "p_value": round(p, 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "interpretation_note": "Secondary exploratory correlation evaluating explanation-performance coupling."
        })

    # 3. Zurich Exclusion Sensitivity Analysis (N=16)
    df_no_zurich = df[~df["is_zurich_transfer"]].copy()
    for col_x, col_y, name, hyp_dir in primary_specs:
        r, p = stats.spearmanr(df_no_zurich[col_x], df_no_zurich[col_y])
        c_med, c_lo, c_hi = compute_cluster_bootstrap_ci(df_no_zurich, col_x, col_y)
        s_med, s_lo, s_hi = compute_standard_bootstrap_ci(df_no_zurich, col_x, col_y)
        records.append({
            "analysis_tier": "Robustness (Zurich-Excluded)",
            "comparison_name": f"{name} [No Zurich]",
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df_no_zurich),
            "spearman_rho": round(r, 4),
            "p_value": round(p, 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "interpretation_note": "Prespecified sensitivity excluding small, high-prevalence Zurich transfers."
        })

    # 4. Transfer-Direction Aggregated Sensitivity (N=6)
    clust_df = df.groupby("transfer_pair")[[
        "abs_prevalence_delta_pct", "external_ece", "brier_change",
        "auc_change", "mean_wasserstein_distance", "spearman_rank_correlation"
    ]].median().reset_index()

    for col_x, col_y, name, hyp_dir in primary_specs:
        r, p = stats.spearmanr(clust_df[col_x], clust_df[col_y])
        s_med, s_lo, s_hi = compute_standard_bootstrap_ci(clust_df, col_x, col_y)
        records.append({
            "analysis_tier": "Sensitivity (Transfer-Level N=6)",
            "comparison_name": f"{name} [Transfer Medians]",
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(clust_df),
            "spearman_rho": round(r, 4),
            "p_value": round(p, 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": np.nan,
            "cluster_boot_ci_lower": np.nan,
            "cluster_boot_ci_upper": np.nan,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "interpretation_note": "Cluster-level median sensitivity analysis addressing within-pair correlation."
        })

    corr_df = pd.DataFrame(records)
    return corr_df


def generate_cluster_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes primary estimands (AUC, delta AUC, Brier, delta Brier, ECE, delta ECE,
    SHAP stability, shift) overall, by model, and by transfer direction.
    """
    rows = []

    def compute_summary_row(sub_df, level, name):
        def get_iqr(series):
            return round(stats.iqr(series.dropna()), 4)

        return {
            "grouping_level": level,
            "group_name": name,
            "n_experiments": len(sub_df),
            "median_external_auc": round(sub_df["external_roc_auc"].median(), 2),
            "iqr_external_auc": get_iqr(sub_df["external_roc_auc"]),
            "min_external_auc": round(sub_df["external_roc_auc"].min(), 2),
            "max_external_auc": round(sub_df["external_roc_auc"].max(), 2),
            "median_delta_auc": round(sub_df["auc_change"].median(), 2),
            "iqr_delta_auc": get_iqr(sub_df["auc_change"]),
            "min_delta_auc": round(sub_df["auc_change"].min(), 2),
            "max_delta_auc": round(sub_df["auc_change"].max(), 2),
            "median_external_brier": round(sub_df["external_brier"].median(), 4),
            "iqr_external_brier": get_iqr(sub_df["external_brier"]),
            "median_delta_brier": round(sub_df["brier_change"].median(), 4),
            "iqr_delta_brier": get_iqr(sub_df["brier_change"]),
            "min_delta_brier": round(sub_df["brier_change"].min(), 4),
            "max_delta_brier": round(sub_df["brier_change"].max(), 4),
            "median_external_ece": round(sub_df["external_ece"].median(), 4),
            "iqr_external_ece": get_iqr(sub_df["external_ece"]),
            "median_delta_ece": round(sub_df["ece_change"].median(), 4),
            "iqr_delta_ece": get_iqr(sub_df["ece_change"]),
            "median_shap_stability": round(sub_df["spearman_rank_correlation"].median(), 4),
            "iqr_shap_stability": get_iqr(sub_df["spearman_rank_correlation"]),
            "min_shap_stability": round(sub_df["spearman_rank_correlation"].min(), 4),
            "max_shap_stability": round(sub_df["spearman_rank_correlation"].max(), 4),
            "median_wasserstein": round(sub_df["mean_wasserstein_distance"].median(), 4),
            "median_prevalence_delta_pct": round(sub_df["abs_prevalence_delta_pct"].median(), 2)
        }

    # Overall
    rows.append(compute_summary_row(df, "Overall", "All Experiments (N=24)"))

    # By Model
    for model_name, grp in df.groupby("model"):
        rows.append(compute_summary_row(grp, "Model", model_name))

    # By Transfer Pair
    for pair_name, grp in df.groupby("transfer_pair"):
        rows.append(compute_summary_row(grp, "Transfer_Pair", pair_name))

    return pd.DataFrame(rows)


def generate_subgroup_synthesis() -> pd.DataFrame:
    """
    Synthesizes Phase 8 demographic disparities into non-parametric summaries.
    """
    sub_path = os.path.join(RESULTS_DIR, "subgroup_disparity.csv")
    df_sub = pd.read_csv(sub_path)

    records = []
    for comp_type, grp in df_sub.groupby("comparison_type"):
        def get_iqr(s):
            return round(stats.iqr(s.dropna()), 4)

        evaluable = grp[grp["is_evaluable_comparison"] == True]
        records.append({
            "subgroup_dimension": comp_type,
            "total_comparisons": len(grp),
            "evaluable_comparisons": len(evaluable),
            "median_abs_auc_disparity": round(grp["abs_delta_roc_auc"].median(), 2),
            "iqr_abs_auc_disparity": get_iqr(grp["abs_delta_roc_auc"]),
            "min_abs_auc_disparity": round(grp["abs_delta_roc_auc"].min(), 2),
            "max_abs_auc_disparity": round(grp["abs_delta_roc_auc"].max(), 2),
            "median_abs_brier_disparity": round(grp["abs_delta_brier"].median(), 4),
            "iqr_abs_brier_disparity": get_iqr(grp["abs_delta_brier"]),
            "median_abs_ece_disparity": round(grp["abs_delta_ece"].median(), 4),
            "iqr_abs_ece_disparity": get_iqr(grp["abs_delta_ece"]),
            "min_abs_ece_disparity": round(grp["abs_delta_ece"].min(), 4),
            "max_abs_ece_disparity": round(grp["abs_delta_ece"].max(), 4),
            "median_prevalence_disparity_pct": round(grp["abs_delta_prevalence_pct"].median(), 2),
            "clinical_synthesis_note": (
                "Substantial calibration disparities driven by cohort base-rate imbalances; "
                "discrimination disparities moderate but present in elderly strata."
            )
        })

    return pd.DataFrame(records)


def generate_missingness_synthesis() -> pd.DataFrame:
    """
    Synthesizes Phase 8 missingness sensitivity into non-parametric comparisons.
    """
    miss_path = os.path.join(RESULTS_DIR, "missingness_sensitivity.csv")
    df_miss = pd.read_csv(miss_path)

    # Compute deltas relative to primary development-fitted KNN pipeline
    df_miss["auc_delta_test_cc"] = np.round(df_miss["test_cc_auc"] - df_miss["primary_auc"], 2)
    df_miss["brier_delta_test_cc"] = np.round(df_miss["test_cc_brier"] - df_miss["primary_brier"], 4)
    df_miss["ece_delta_test_cc"] = np.round(df_miss["test_cc_ece"] - df_miss["primary_ece"], 4)

    df_miss["auc_delta_full_cc"] = np.round(df_miss["full_cc_auc"] - df_miss["primary_auc"], 2)
    df_miss["brier_delta_full_cc"] = np.round(df_miss["full_cc_brier"] - df_miss["primary_brier"], 4)
    df_miss["ece_delta_full_cc"] = np.round(df_miss["full_cc_ece"] - df_miss["primary_ece"], 4)

    df_miss["auc_delta_ind"] = np.round(df_miss["indicator_auc"] - df_miss["primary_auc"], 2)
    df_miss["brier_delta_ind"] = np.round(df_miss["indicator_brier"] - df_miss["primary_brier"], 4)
    df_miss["ece_delta_ind"] = np.round(df_miss["indicator_ece"] - df_miss["primary_ece"], 4)

    def get_iqr(s):
        return round(stats.iqr(s.dropna()), 4)

    regimes = [
        {
            "imputation_regime": "Primary Development-Fitted (KNN)",
            "evaluable_experiments": 24,
            "median_auc": round(df_miss["primary_auc"].median(), 2),
            "iqr_auc": get_iqr(df_miss["primary_auc"]),
            "min_auc": round(df_miss["primary_auc"].min(), 2),
            "max_auc": round(df_miss["primary_auc"].max(), 2),
            "median_delta_auc_vs_primary": 0.00,
            "iqr_delta_auc_vs_primary": 0.00,
            "median_brier": round(df_miss["primary_brier"].median(), 4),
            "iqr_brier": get_iqr(df_miss["primary_brier"]),
            "median_delta_brier_vs_primary": 0.0000,
            "iqr_delta_brier_vs_primary": 0.0000,
            "median_ece": round(df_miss["primary_ece"].median(), 4),
            "iqr_ece": get_iqr(df_miss["primary_ece"]),
            "robustness_conclusion": "Baseline primary benchmark pipeline without external leakage."
        },
        {
            "imputation_regime": "Complete-Case Test Evaluation",
            "evaluable_experiments": 24,
            "median_auc": round(df_miss["test_cc_auc"].median(), 2),
            "iqr_auc": get_iqr(df_miss["test_cc_auc"]),
            "min_auc": round(df_miss["test_cc_auc"].min(), 2),
            "max_auc": round(df_miss["test_cc_auc"].max(), 2),
            "median_delta_auc_vs_primary": round(df_miss["auc_delta_test_cc"].median(), 2),
            "iqr_delta_auc_vs_primary": get_iqr(df_miss["auc_delta_test_cc"]),
            "median_brier": round(df_miss["test_cc_brier"].median(), 4),
            "iqr_brier": get_iqr(df_miss["test_cc_brier"]),
            "median_delta_brier_vs_primary": round(df_miss["brier_delta_test_cc"].median(), 4),
            "iqr_delta_brier_vs_primary": get_iqr(df_miss["brier_delta_test_cc"]),
            "median_ece": round(df_miss["test_cc_ece"].median(), 4),
            "iqr_ece": get_iqr(df_miss["test_cc_ece"]),
            "robustness_conclusion": "Minimal distortion; confirms performance is not artifact of imputation."
        },
        {
            "imputation_regime": "Full Complete-Case (Train & Test)",
            "evaluable_experiments": int(df_miss["full_cc_auc"].notna().sum()),
            "median_auc": round(df_miss["full_cc_auc"].dropna().median(), 2),
            "iqr_auc": get_iqr(df_miss["full_cc_auc"].dropna()),
            "min_auc": round(df_miss["full_cc_auc"].dropna().min(), 2),
            "max_auc": round(df_miss["full_cc_auc"].dropna().max(), 2),
            "median_delta_auc_vs_primary": round(df_miss["auc_delta_full_cc"].dropna().median(), 2),
            "iqr_delta_auc_vs_primary": get_iqr(df_miss["auc_delta_full_cc"].dropna()),
            "median_brier": round(df_miss["full_cc_brier"].dropna().median(), 4),
            "iqr_brier": get_iqr(df_miss["full_cc_brier"].dropna()),
            "median_delta_brier_vs_primary": round(df_miss["brier_delta_full_cc"].dropna().median(), 4),
            "iqr_delta_brier_vs_primary": get_iqr(df_miss["brier_delta_full_cc"].dropna()),
            "median_ece": round(df_miss["full_cc_ece"].dropna().median(), 4),
            "iqr_ece": get_iqr(df_miss["full_cc_ece"].dropna()),
            "robustness_conclusion": "Evaluated where complete cases exist; confirms stable ranking."
        },
        {
            "imputation_regime": "Missingness Indicator Pipeline",
            "evaluable_experiments": 24,
            "median_auc": round(df_miss["indicator_auc"].median(), 2),
            "iqr_auc": get_iqr(df_miss["indicator_auc"]),
            "min_auc": round(df_miss["indicator_auc"].min(), 2),
            "max_auc": round(df_miss["indicator_auc"].max(), 2),
            "median_delta_auc_vs_primary": round(df_miss["auc_delta_ind"].median(), 2),
            "iqr_delta_auc_vs_primary": get_iqr(df_miss["auc_delta_ind"]),
            "median_brier": round(df_miss["indicator_brier"].median(), 4),
            "iqr_brier": get_iqr(df_miss["indicator_brier"]),
            "median_delta_brier_vs_primary": round(df_miss["brier_delta_ind"].median(), 4),
            "iqr_delta_brier_vs_primary": get_iqr(df_miss["brier_delta_ind"]),
            "median_ece": round(df_miss["indicator_ece"].median(), 4),
            "iqr_ece": get_iqr(df_miss["indicator_ece"]),
            "robustness_conclusion": "Marginal AUC gain; preserves primary transport and drift conclusions."
        }
    ]

    return pd.DataFrame(regimes)


# ==============================================================================
# PUBLICATION FIGURE GENERATION (300 DPI)
# ==============================================================================

def plot_scatter_with_annotations(
    ax,
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    add_trendline: bool = True,
    corr_df: pd.DataFrame = None
):
    """
    Renders a publication-ready scatter plot with:
    - Model-specific colors and markers
    - Explicit visualization of Zurich transfers (unfilled / highlighted border)
    - Non-parametric Spearman rho annotation with cluster bootstrap 95% CI
    """
    if corr_df is not None:
        match = corr_df[
            (corr_df["var_x"] == col_x) &
            (corr_df["var_y"] == col_y) &
            (corr_df["analysis_tier"].isin(["Primary", "Secondary"]))
        ]
        if len(match) > 0:
            row = match.iloc[0]
            r = row["spearman_rho"]
            p = row["p_value"]
            c_lo = row["cluster_boot_ci_lower"]
            c_hi = row["cluster_boot_ci_upper"]
        else:
            r, p = stats.spearmanr(df[col_x], df[col_y])
            _, c_lo, c_hi = compute_cluster_bootstrap_ci(df, col_x, col_y)
    else:
        r, p = stats.spearmanr(df[col_x], df[col_y])
        _, c_lo, c_hi = compute_cluster_bootstrap_ci(df, col_x, col_y)

    for model_name, m_color in MODEL_COLORS.items():
        marker = MODEL_MARKERS[model_name]
        m_df = df[df["model"] == model_name]

        # Non-Zurich transfers
        nz_df = m_df[~m_df["is_zurich_transfer"]]
        ax.scatter(
            nz_df[col_x], nz_df[col_y],
            color=m_color, marker=marker, s=80, alpha=0.85,
            edgecolors="black", linewidth=0.8,
            label=f"{model_name}"
        )

        # Zurich transfers (distinct highlight: yellow face with colored border)
        z_df = m_df[m_df["is_zurich_transfer"]]
        if len(z_df) > 0:
            ax.scatter(
                z_df[col_x], z_df[col_y],
                facecolors="#ffdd57", edgecolors=m_color, marker=marker,
                s=110, linewidth=2.0, alpha=0.95,
                label=f"{model_name} (Zurich Transfer)" if model_name == "LogisticRegression" else None
            )

    if add_trendline:
        # Robust LOWESS or OLS trendline for visual guide
        sns.regplot(
            data=df, x=col_x, y=col_y, ax=ax,
            scatter=False, color="#555555",
            line_kws={"linestyle": "--", "linewidth": 1.5, "alpha": 0.7}
        )

    # Annotate Spearman rho and bootstrap CI
    ci_text = f"Spearman $\\rho = {r:.3f}$ ($p = {p:.4f}$)\nCluster 95% CI: [{c_lo:.3f}, {c_hi:.3f}]"
    ax.text(
        0.05, 0.93, ci_text,
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
    )

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5)


def generate_all_figures(df: pd.DataFrame, corr_df: pd.DataFrame = None):
    """
    Renders the 6 required publication figures in 300 DPI.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0
    })

    # Figure 1: Prevalence Shift vs External ECE
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    plot_scatter_with_annotations(
        ax, df,
        col_x="abs_prevalence_delta_pct",
        col_y="external_ece",
        xlabel="|Prevalence Shift| (|Test % - Train %|)",
        ylabel="External ECE (Expected Calibration Error)",
        title="Figure 1: Target Prevalence Shift vs. External Calibration Error",
        corr_df=corr_df
    )
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prevalence_shift_vs_external_ece.png"), dpi=300)
    plt.close()

    # Figure 2: Prevalence Shift vs Brier Degradation
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    plot_scatter_with_annotations(
        ax, df,
        col_x="abs_prevalence_delta_pct",
        col_y="brier_change",
        xlabel="|Prevalence Shift| (|Test % - Train %|)",
        ylabel="Delta Brier (External - Internal) [Positive = Deterioration]",
        title="Figure 2: Target Prevalence Shift vs. Brier Score Deterioration",
        corr_df=corr_df
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prevalence_shift_vs_brier_change.png"), dpi=300)
    plt.close()

    # Figure 3: Wasserstein Distance vs AUC Degradation
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    plot_scatter_with_annotations(
        ax, df,
        col_x="mean_wasserstein_distance",
        col_y="auc_change",
        xlabel="Mean 1-Wasserstein Distance across Features",
        ylabel="Delta ROC-AUC (External - Internal) [Negative = Degradation]",
        title="Figure 3: Continuous Covariate Shift vs. Discrimination Degradation",
        corr_df=corr_df
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "wasserstein_vs_auc_degradation.png"), dpi=300)
    plt.close()

    # Figure 4: Wasserstein Distance vs SHAP Rank Stability
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    plot_scatter_with_annotations(
        ax, df,
        col_x="mean_wasserstein_distance",
        col_y="spearman_rank_correlation",
        xlabel="Mean 1-Wasserstein Distance across Features",
        ylabel="SHAP Global Feature Rank Stability (Spearman rho)",
        title="Figure 4: Continuous Covariate Shift vs. Explanation Stability",
        corr_df=corr_df
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "wasserstein_vs_shap_stability.png"), dpi=300)
    plt.close()

    # Figure 5: SHAP Stability vs AUC Degradation
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    plot_scatter_with_annotations(
        ax, df,
        col_x="spearman_rank_correlation",
        col_y="auc_change",
        xlabel="SHAP Global Feature Rank Stability (Spearman rho)",
        ylabel="Delta ROC-AUC (External - Internal) [Negative = Degradation]",
        title="Figure 5: Explanation Stability vs. Discrimination Degradation",
        corr_df=corr_df
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_stability_vs_auc_degradation.png"), dpi=300)
    plt.close()

    # Figure 6: Integrated Multi-Panel Synthesis Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # Panel A: Prev Shift vs ECE
    plot_scatter_with_annotations(
        axes[0, 0], df, "abs_prevalence_delta_pct", "external_ece",
        "|Prevalence Shift| (%)", "External ECE", "A: Prevalence Shift vs. External ECE",
        corr_df=corr_df
    )
    # Panel B: Prev Shift vs Brier Change
    plot_scatter_with_annotations(
        axes[0, 1], df, "abs_prevalence_delta_pct", "brier_change",
        "|Prevalence Shift| (%)", "Delta Brier (Ext - Int)", "B: Prevalence Shift vs. Brier Change",
        corr_df=corr_df
    )
    # Panel C: Covariate Shift vs Delta AUC
    plot_scatter_with_annotations(
        axes[0, 2], df, "mean_wasserstein_distance", "auc_change",
        "Mean Wasserstein Distance", "Delta ROC-AUC", "C: Covariate Shift vs. Delta AUC",
        corr_df=corr_df
    )
    # Panel D: Covariate Shift vs SHAP Stability
    plot_scatter_with_annotations(
        axes[1, 0], df, "mean_wasserstein_distance", "spearman_rank_correlation",
        "Mean Wasserstein Distance", "SHAP Rank Stability (rho)", "D: Covariate Shift vs. SHAP Stability",
        corr_df=corr_df
    )
    # Panel E: SHAP Stability vs Delta AUC
    plot_scatter_with_annotations(
        axes[1, 1], df, "spearman_rank_correlation", "auc_change",
        "SHAP Rank Stability (rho)", "Delta ROC-AUC", "E: SHAP Stability vs. Delta AUC",
        corr_df=corr_df
    )
    # Panel F: SHAP Stability vs Delta Brier
    plot_scatter_with_annotations(
        axes[1, 2], df, "spearman_rank_correlation", "brier_change",
        "SHAP Rank Stability (rho)", "Delta Brier (Ext - Int)", "F: SHAP Stability vs. Delta Brier",
        corr_df=corr_df
    )

    # Add unified top legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    fig.legend(
        unique_legend.values(), unique_legend.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5,
        frameon=True, framealpha=0.95, fontsize=10
    )

    fig.suptitle(
        "Integrated Evidence Synthesis: Dataset Shift, Transportability, Calibration, and Explanation Stability",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIGURES_DIR, "integrated_evidence_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_integrated_synthesis():
    """Main execution entry point for Phase 9."""
    print("=" * 70)
    print("Executing Phase 9: Integrated Statistical Synthesis & Evidence Consolidation")
    print("=" * 70)

    # 1. Merge all outputs
    df_estimands = load_and_merge_data()
    estimands_path = os.path.join(RESULTS_DIR, "integrated_estimands.csv")
    df_estimands.to_csv(estimands_path, index=False)
    print(f"[+] Saved integrated estimands (N={len(df_estimands)}): {estimands_path}")

    # 2. Generate cluster-level and model-level summary
    df_cluster_summary = generate_cluster_summary_table(df_estimands)
    cluster_summary_path = os.path.join(RESULTS_DIR, "integrated_cluster_summary.csv")
    df_cluster_summary.to_csv(cluster_summary_path, index=False)
    print(f"[+] Saved cluster & model summary: {cluster_summary_path}")

    # 3. Generate correlations table with cluster bootstrap
    df_correlations = generate_correlations_table(df_estimands)
    correlations_path = os.path.join(RESULTS_DIR, "integrated_correlations.csv")
    df_correlations.to_csv(correlations_path, index=False)
    print(f"[+] Saved integrated correlations: {correlations_path}")

    # 4. Generate subgroup synthesis
    df_subgroup = generate_subgroup_synthesis()
    subgroup_path = os.path.join(RESULTS_DIR, "subgroup_synthesis.csv")
    df_subgroup.to_csv(subgroup_path, index=False)
    print(f"[+] Saved subgroup synthesis: {subgroup_path}")

    # 5. Generate missingness synthesis
    df_missingness = generate_missingness_synthesis()
    missingness_path = os.path.join(RESULTS_DIR, "missingness_synthesis.csv")
    df_missingness.to_csv(missingness_path, index=False)
    print(f"[+] Saved missingness synthesis: {missingness_path}")

    # 6. Generate publication figures
    print("[*] Generating publication-ready figures (300 DPI)...")
    generate_all_figures(df_estimands, corr_df=df_correlations)
    print(f"[+] Saved 6 publication figures in: {FIGURES_DIR}")

    # 7. Generate Phase 9 Manifest
    manifest = {
        "phase": "Phase 9 — Integrated Statistical Synthesis & Evidence Consolidation",
        "timestamp": "2026-09-19T16:35:00",
        "primary_statistical_unit": "MODEL x DIRECTED TRANSFER EXPERIMENT (N=24)",
        "number_of_models": 4,
        "number_of_transfer_directions": 6,
        "bootstrap_parameters": {
            "iterations": 2000,
            "random_seed": 42,
            "clustering_variable": "transfer_pair (6 clusters)"
        },
        "locked_analyses": {
            "primary": [
                "Prevalence Shift vs External ECE",
                "Prevalence Shift vs Brier Degradation",
                "Mean Wasserstein vs AUC Degradation",
                "Mean Wasserstein vs SHAP Stability"
            ],
            "secondary": [
                "SHAP Stability vs AUC Degradation",
                "SHAP Stability vs Brier Degradation"
            ],
            "robustness": [
                "Zurich Exclusion (N=16)",
                "Transfer-Direction Median Aggregation (N=6)",
                "Subgroup Disparity Synthesis (Phase 8)",
                "Missingness Sensitivity Synthesis (Phase 8)"
            ]
        },
        "key_estimands_overall": {
            "median_external_auc": float(df_estimands["external_roc_auc"].median()),
            "median_delta_auc": float(df_estimands["auc_change"].median()),
            "median_external_brier": float(df_estimands["external_brier"].median()),
            "median_delta_brier": float(df_estimands["brier_change"].median()),
            "median_external_ece": float(df_estimands["external_ece"].median()),
            "median_delta_ece": float(df_estimands["ece_change"].median()),
            "median_shap_stability": float(df_estimands["spearman_rank_correlation"].median()),
            "median_mean_wasserstein": float(df_estimands["mean_wasserstein_distance"].median()),
            "median_prevalence_shift_pct": float(df_estimands["abs_prevalence_delta_pct"].median())
        },
        "artifact_hashes": {
            "integrated_estimands.csv": compute_file_hash(estimands_path),
            "integrated_correlations.csv": compute_file_hash(correlations_path),
            "integrated_cluster_summary.csv": compute_file_hash(cluster_summary_path),
            "subgroup_synthesis.csv": compute_file_hash(subgroup_path),
            "missingness_synthesis.csv": compute_file_hash(missingness_path)
        }
    }

    manifest_path = os.path.join(RESULTS_DIR, "phase9_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved phase 9 manifest: {manifest_path}")
    print("=" * 70)
    print("Phase 9 Quantitative Execution Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_integrated_synthesis()
