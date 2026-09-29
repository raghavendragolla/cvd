"""
R5: Integrated Statistical Synthesis & Evidence Consolidation Engine
===================================================================
Implements:
1. Merging of corrected inputs across R1 (Calibration), R2 (Pooled-OOF Internal Validation),
   R3 (Exact Coalition SVM SHAP), and R4 (Standardized Continuous Covariate Shift).
2. Clustered data structure representation (N=24 experiments nested in 6 transfer-pair clusters).
3. Clustered bootstrap inference (B=2000, seed=42) resampling transfer-pair clusters with replacement.
4. Explicit tracking and logging of skipped/degenerate bootstrap resamples.
5. Clear separation between:
   - Exploratory/naive p-values (uncorrected for clustering)
   - Cluster-aware inference (cluster bootstrap 95% CIs and cluster median analysis).
6. Prespecified sensitivity analyses:
   - Zurich-excluded sensitivity analysis (N=16, 4 clusters)
   - Direction-level median sensitivity analysis (N=6 clusters).
7. Outputs saved separately in corrected_results/:
   - integrated_estimands_corrected.csv
   - integrated_correlations_corrected.csv
   - integrated_cluster_summary_corrected.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CORRECTED_DIR = os.path.join(ROOT_DIR, "corrected_results")
os.makedirs(CORRECTED_DIR, exist_ok=True)


def compute_cluster_bootstrap_ci(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    cluster_col: str = "transfer_pair",
    n_boot: int = 2000,
    seed: int = 42
) -> tuple:
    """
    Cluster-aware bootstrap resampling transfer pairs with replacement.
    Logs and counts skipped resamples (e.g. zero variance in resample).
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    cluster_groups = {c: df[df[cluster_col] == c] for c in clusters}

    boot_rhos = []
    n_skipped = 0

    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_df = pd.concat([cluster_groups[c] for c in sampled_clusters], ignore_index=True)

        if sampled_df[col_x].nunique() < 2 or sampled_df[col_y].nunique() < 2:
            n_skipped += 1
            continue

        r, _ = stats.spearmanr(sampled_df[col_x], sampled_df[col_y])
        if np.isnan(r):
            n_skipped += 1
            continue

        boot_rhos.append(r)

    n_valid = len(boot_rhos)
    if n_valid < 50:
        return np.nan, np.nan, np.nan, n_valid, n_skipped

    med = float(np.median(boot_rhos))
    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))
    return round(med, 4), round(ci_lower, 4), round(ci_upper, 4), n_valid, n_skipped


def compute_standard_bootstrap_ci(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    n_boot: int = 2000,
    seed: int = 42
) -> tuple:
    """Row-level bootstrap for non-parametric comparison."""
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_rhos = []
    n_skipped = 0

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sampled = df.iloc[idx]
        if sampled[col_x].nunique() < 2 or sampled[col_y].nunique() < 2:
            n_skipped += 1
            continue
        r, _ = stats.spearmanr(sampled[col_x], sampled[col_y])
        if np.isnan(r):
            n_skipped += 1
            continue
        boot_rhos.append(r)

    n_valid = len(boot_rhos)
    if n_valid < 50:
        return np.nan, np.nan, np.nan, n_valid, n_skipped

    med = float(np.median(boot_rhos))
    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))
    return round(med, 4), round(ci_lower, 4), round(ci_upper, 4), n_valid, n_skipped


def run_r5_integrated_synthesis_correction():
    print("\n=======================================================")
    print("Executing R5: Corrected Integrated Statistical Synthesis")
    print("=======================================================")

    # 1. Load corrected inputs
    ext_orig_path = os.path.join(RESULTS_DIR, "external_validation_results.csv")
    df_ext = pd.read_csv(ext_orig_path)

    int_oof_path = os.path.join(CORRECTED_DIR, "internal_validation_pooled_oof.csv")
    df_int = pd.read_csv(int_oof_path)
    # Filter to Track B
    df_int_b = df_int[df_int["track"] == "Track_B"].copy()

    ext_cal_path = os.path.join(CORRECTED_DIR, "calibration_corrected_external.csv")
    df_ext_cal = pd.read_csv(ext_cal_path)

    shift_path = os.path.join(CORRECTED_DIR, "dataset_shift_relationships_standardized.csv")
    df_shift = pd.read_csv(shift_path)

    # Load SHAP stability: original tree/linear models + corrected SVM
    shap_orig_path = os.path.join(RESULTS_DIR, "shap_rank_stability.csv")
    df_shap_orig = pd.read_csv(shap_orig_path)

    svm_shap_corr_path = os.path.join(CORRECTED_DIR, "svm_shap_rank_stability_corrected.csv")
    if os.path.exists(svm_shap_corr_path):
        df_svm_corr = pd.read_csv(svm_shap_corr_path)
    else:
        df_svm_corr = None

    # Match internal baselines to each external experiment
    int_dict = {}
    for _, r in df_int_b.iterrows():
        int_dict[(r["cohort"], r["model"])] = {
            "internal_roc_auc": r["roc_auc"],
            "internal_brier": r["brier_score"],
            "internal_ece": r["ece"],
            "internal_cal_intercept": r["calibration_intercept"],
            "internal_cal_slope": r["calibration_slope"]
        }

    ext_cal_dict = {}
    for _, r in df_ext_cal.iterrows():
        ext_cal_dict[r["experiment_id"]] = {
            "external_calibration_intercept": r["external_calibration_intercept"],
            "external_calibration_slope": r["external_calibration_slope"],
            "external_ece": r["external_ece"],
            "external_brier": r["external_brier_score"]
        }

    # Build integrated estimands dataframe
    estimand_rows = []

    for _, r in df_ext.iterrows():
        exp_id = r["experiment_id"]
        tr = r["train_dataset"]
        te = r["test_dataset"]
        m = r["model"]

        int_metrics = int_dict.get((tr, m), {})
        cal_metrics = ext_cal_dict.get(exp_id, {})

        ext_auc = r["external_roc_auc"]
        int_auc = int_metrics.get("internal_roc_auc", np.nan)
        auc_delta = round(ext_auc - int_auc, 2) if (not np.isnan(ext_auc) and not np.isnan(int_auc)) else np.nan

        ext_brier = cal_metrics.get("external_brier", r["external_brier"])
        int_brier = int_metrics.get("internal_brier", np.nan)
        brier_delta = round(ext_brier - int_brier, 4) if (not np.isnan(ext_brier) and not np.isnan(int_brier)) else np.nan

        ext_ece = cal_metrics.get("external_ece", r.get("external_ece", np.nan))
        int_ece = int_metrics.get("internal_ece", np.nan)
        ece_delta = round(ext_ece - int_ece, 4) if (not np.isnan(ext_ece) and not np.isnan(int_ece)) else np.nan

        ext_slope = cal_metrics.get("external_calibration_slope", np.nan)
        int_slope = int_metrics.get("internal_cal_slope", np.nan)
        slope_delta = round(ext_slope - int_slope, 4) if (not np.isnan(ext_slope) and not np.isnan(int_slope)) else np.nan

        ext_intercept = cal_metrics.get("external_calibration_intercept", np.nan)
        int_intercept = int_metrics.get("internal_cal_intercept", np.nan)
        intercept_delta = round(ext_intercept - int_intercept, 4) if (not np.isnan(ext_intercept) and not np.isnan(int_intercept)) else np.nan

        # Dataset shift metrics
        shift_match = df_shift[(df_shift["train_dataset"] == tr) & (df_shift["test_dataset"] == te)]
        if len(shift_match) > 0:
            s_row = shift_match.iloc[0]
            mean_std_w1 = s_row["mean_standardized_wasserstein_distance"]
            mean_raw_w1 = s_row["mean_raw_wasserstein_distance"]
            mean_jsd = s_row["mean_jsd"]
            mean_smd = s_row["mean_abs_smd"]
            prev_delta = s_row["abs_prevalence_delta_pct"]
        else:
            mean_std_w1, mean_raw_w1, mean_jsd, mean_smd, prev_delta = np.nan, np.nan, np.nan, np.nan, np.nan

        # SHAP stability metrics
        if m == "SVM" and df_svm_corr is not None:
            svm_m = df_svm_corr[(df_svm_corr["train_dataset"] == tr) & (df_svm_corr["test_dataset"] == te)]
            if len(svm_m) > 0:
                shap_rho = svm_m["corrected_spearman_rho"].iloc[0]
                shap_tau = svm_m["corrected_kendall_tau"].iloc[0]
                top3_ov = svm_m["corrected_top3_overlap"].iloc[0]
                top5_ov = svm_m["corrected_top5_overlap"].iloc[0]
            else:
                shap_rho, shap_tau, top3_ov, top5_ov = np.nan, np.nan, np.nan, np.nan
        else:
            orig_m = df_shap_orig[(df_shap_orig["train_dataset"] == tr) & (df_shap_orig["test_dataset"] == te) & (df_shap_orig["model"] == m)]
            if len(orig_m) > 0:
                shap_rho = orig_m["spearman_rank_correlation"].iloc[0]
                shap_tau = orig_m["kendall_rank_correlation"].iloc[0]
                top3_ov = orig_m["top3_overlap"].iloc[0]
                top5_ov = orig_m["top5_overlap"].iloc[0]
            else:
                shap_rho, shap_tau, top3_ov, top5_ov = np.nan, np.nan, np.nan, np.nan

        estimand_rows.append({
            "experiment_id": exp_id,
            "train_dataset": tr,
            "test_dataset": te,
            "transfer_pair": f"{tr} -> {te}",
            "model": m,
            "is_zurich_transfer": bool((tr == "Zurich") or (te == "Zurich")),
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            # Discrimination
            "internal_roc_auc": int_auc,
            "external_roc_auc": ext_auc,
            "auc_change": auc_delta,
            # Brier
            "internal_brier": int_brier,
            "external_brier": ext_brier,
            "brier_change": brier_delta,
            # Calibration ECE
            "internal_ece": int_ece,
            "external_ece": ext_ece,
            "ece_change": ece_delta,
            # Calibration Slope
            "internal_cal_slope": int_slope,
            "external_calibration_slope": ext_slope,
            "cal_slope_change": slope_delta,
            # Calibration Intercept
            "internal_cal_intercept": int_intercept,
            "external_calibration_intercept": ext_intercept,
            "cal_intercept_change": intercept_delta,
            # Dataset Shift
            "mean_standardized_wasserstein_distance": mean_std_w1,
            "mean_raw_wasserstein_distance": mean_raw_w1,
            "mean_jsd": mean_jsd,
            "mean_abs_smd": mean_smd,
            "abs_prevalence_delta_pct": prev_delta,
            # SHAP Stability
            "spearman_rank_correlation": shap_rho,
            "kendall_rank_correlation": shap_tau,
            "top3_overlap": top3_ov,
            "top5_overlap": top5_ov
        })

    df_estimands = pd.DataFrame(estimand_rows)
    out_est_path = os.path.join(CORRECTED_DIR, "integrated_estimands_corrected.csv")
    df_estimands.to_csv(out_est_path, index=False)
    print(f"[OK] Saved corrected integrated estimands to: {out_est_path}")

    # 2. Cluster Summary (N=6 transfer pairs)
    clust_summary = df_estimands.groupby("transfer_pair").agg(
        train_dataset=("train_dataset", "first"),
        test_dataset=("test_dataset", "first"),
        n_experiments=("experiment_id", "count"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        is_zurich_transfer=("is_zurich_transfer", "first"),
        median_auc_change=("auc_change", "median"),
        median_brier_change=("brier_change", "median"),
        median_external_ece=("external_ece", "median"),
        median_cal_intercept=("external_calibration_intercept", "median"),
        median_cal_slope=("external_calibration_slope", "median"),
        mean_standardized_wasserstein=("mean_standardized_wasserstein_distance", "first"),
        mean_raw_wasserstein=("mean_raw_wasserstein_distance", "first"),
        abs_prevalence_delta_pct=("abs_prevalence_delta_pct", "first"),
        median_shap_spearman=("spearman_rank_correlation", "median")
    ).reset_index()

    out_clust_path = os.path.join(CORRECTED_DIR, "integrated_cluster_summary_corrected.csv")
    clust_summary.to_csv(out_clust_path, index=False)
    print(f"[OK] Saved corrected cluster summary to: {out_clust_path}")

    # 3. Association Specifications
    primary_specs = [
        ("abs_prevalence_delta_pct", "external_ece", "Prevalence Shift vs External ECE", "Positive"),
        ("abs_prevalence_delta_pct", "brier_change", "Prevalence Shift vs Brier Degradation", "Positive"),
        ("mean_standardized_wasserstein_distance", "auc_change", "Standardized Wasserstein vs AUC Degradation", "Negative"),
        ("mean_standardized_wasserstein_distance", "spearman_rank_correlation", "Standardized Wasserstein vs SHAP Stability", "Negative")
    ]

    secondary_specs = [
        ("spearman_rank_correlation", "auc_change", "SHAP Stability vs AUC Degradation", "Positive"),
        ("spearman_rank_correlation", "brier_change", "SHAP Stability vs Brier Degradation", "Negative")
    ]

    correlation_records = []

    # A. Primary Associations (N=24)
    for col_x, col_y, name, hyp_dir in primary_specs:
        r, p = stats.spearmanr(df_estimands[col_x], df_estimands[col_y])
        c_med, c_lo, c_hi, n_valid, n_skip = compute_cluster_bootstrap_ci(df_estimands, col_x, col_y)
        s_med, s_lo, s_hi, _, _ = compute_standard_bootstrap_ci(df_estimands, col_x, col_y)

        correlation_records.append({
            "analysis_tier": "Primary",
            "comparison_name": name,
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df_estimands),
            "n_clusters": df_estimands["transfer_pair"].nunique(),
            "spearman_rho": round(float(r), 4),
            "naive_p_value_uncorrected": round(float(p), 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "bootstrap_valid_resamples": n_valid,
            "bootstrap_skipped_resamples": n_skip,
            "statistical_inference_tier": "Cluster-Aware CI [Confirmatory] | Naive p-value [Exploratory / Cluster-Unaware]",
            "interpretation_note": "Primary experiment-level observational correlation across 24 experiments nested in 6 transfer pairs."
        })

    # B. Secondary Associations (N=24)
    for col_x, col_y, name, hyp_dir in secondary_specs:
        r, p = stats.spearmanr(df_estimands[col_x], df_estimands[col_y])
        c_med, c_lo, c_hi, n_valid, n_skip = compute_cluster_bootstrap_ci(df_estimands, col_x, col_y)
        s_med, s_lo, s_hi, _, _ = compute_standard_bootstrap_ci(df_estimands, col_x, col_y)

        correlation_records.append({
            "analysis_tier": "Secondary",
            "comparison_name": name,
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df_estimands),
            "n_clusters": df_estimands["transfer_pair"].nunique(),
            "spearman_rho": round(float(r), 4),
            "naive_p_value_uncorrected": round(float(p), 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "bootstrap_valid_resamples": n_valid,
            "bootstrap_skipped_resamples": n_skip,
            "statistical_inference_tier": "Cluster-Aware CI [Confirmatory] | Naive p-value [Exploratory / Cluster-Unaware]",
            "interpretation_note": "Secondary exploratory correlation evaluating explanation-performance coupling."
        })

    # C. Zurich-Excluded Sensitivity Analysis (N=16, 4 clusters)
    df_no_zurich = df_estimands[~df_estimands["is_zurich_transfer"]].copy()
    for col_x, col_y, name, hyp_dir in primary_specs:
        r, p = stats.spearmanr(df_no_zurich[col_x], df_no_zurich[col_y])
        c_med, c_lo, c_hi, n_valid, n_skip = compute_cluster_bootstrap_ci(df_no_zurich, col_x, col_y)
        s_med, s_lo, s_hi, _, _ = compute_standard_bootstrap_ci(df_no_zurich, col_x, col_y)

        correlation_records.append({
            "analysis_tier": "Robustness (Zurich-Excluded)",
            "comparison_name": f"{name} [No Zurich]",
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(df_no_zurich),
            "n_clusters": df_no_zurich["transfer_pair"].nunique(),
            "spearman_rho": round(float(r), 4),
            "naive_p_value_uncorrected": round(float(p), 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": c_med,
            "cluster_boot_ci_lower": c_lo,
            "cluster_boot_ci_upper": c_hi,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "bootstrap_valid_resamples": n_valid,
            "bootstrap_skipped_resamples": n_skip,
            "statistical_inference_tier": "Cluster-Aware CI [Confirmatory] | Naive p-value [Exploratory / Cluster-Unaware]",
            "interpretation_note": "Prespecified sensitivity excluding small, high-prevalence Zurich transfers (N=16 across 4 clusters)."
        })

    # D. Direction-Level Median Sensitivity (N=6 clusters)
    for col_x, col_y, name, hyp_dir in primary_specs:
        x_name = "mean_standardized_wasserstein" if col_x == "mean_standardized_wasserstein_distance" else (
            "median_external_ece" if col_y == "external_ece" else (
                "median_brier_change" if col_y == "brier_change" else (
                    "median_auc_change" if col_y == "auc_change" else (
                        "median_shap_spearman" if col_y == "spearman_rank_correlation" else col_y
                    )
                )
            )
        )
        y_name = "median_external_ece" if col_y == "external_ece" else (
            "median_brier_change" if col_y == "brier_change" else (
                "median_auc_change" if col_y == "auc_change" else (
                    "median_shap_spearman" if col_y == "spearman_rank_correlation" else col_y
                )
            )
        )
        r, p = stats.spearmanr(clust_summary[col_x if col_x in clust_summary else x_name], clust_summary[y_name])
        s_med, s_lo, s_hi, n_valid, n_skip = compute_standard_bootstrap_ci(
            clust_summary, col_x if col_x in clust_summary else x_name, y_name
        )

        correlation_records.append({
            "analysis_tier": "Sensitivity (Transfer-Level N=6)",
            "comparison_name": f"{name} [Transfer Medians]",
            "var_x": col_x,
            "var_y": col_y,
            "n_observations": len(clust_summary),
            "n_clusters": len(clust_summary),
            "spearman_rho": round(float(r), 4),
            "naive_p_value_uncorrected": round(float(p), 5),
            "expected_direction": hyp_dir,
            "cluster_boot_median": np.nan,
            "cluster_boot_ci_lower": np.nan,
            "cluster_boot_ci_upper": np.nan,
            "standard_boot_ci_lower": s_lo,
            "standard_boot_ci_upper": s_hi,
            "bootstrap_valid_resamples": n_valid,
            "bootstrap_skipped_resamples": n_skip,
            "statistical_inference_tier": "Non-parametric aggregation across N=6 independent clusters",
            "interpretation_note": "Cluster-aggregated analysis collapsing models to single cluster medians."
        })

    df_corr = pd.DataFrame(correlation_records)
    out_corr_path = os.path.join(CORRECTED_DIR, "integrated_correlations_corrected.csv")
    df_corr.to_csv(out_corr_path, index=False)
    print(f"[OK] Saved corrected integrated correlations to: {out_corr_path}")

    return df_estimands, df_corr, clust_summary


if __name__ == "__main__":
    run_r5_integrated_synthesis_correction()
