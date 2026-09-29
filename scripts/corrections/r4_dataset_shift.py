"""
R4: Dataset Shift Recalculation Engine (Standardized Wasserstein Distance)
========================================================================
Implements:
1. Continuous covariate shift recalculation using strictly source-standardized features:
   z_src = (x_src - mu_src) / sigma_src
   z_tgt = (x_tgt - mu_src) / sigma_src
2. Elimination of unstandardized Wasserstein distance averaging across incompatible units.
3. Reporting of both per-feature raw and source-standardized shift metrics.
4. Recalculation of the aggregate standardized shift measure (mean standardized W1 across continuous features).
5. Association analysis between standardized shift and SHAP rank stability.
6. Preservation of old raw-unit metrics for full audit traceability.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import scipy.spatial.distance as dist

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CORRECTED_DIR = os.path.join(ROOT_DIR, "corrected_results")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(CORRECTED_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import load_harmonized_cohort, TRACK_B_10_FEATURES

CONTINUOUS_FEATURES = ["age", "trestbps", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

TRANSFER_DIRECTIONS = [
    ("Cleveland", "Hungarian"),
    ("Cleveland", "Zurich"),
    ("Cleveland", "VA Long Beach"),
    ("Hungarian", "Cleveland"),
    ("Zurich", "Cleveland"),
    ("VA Long Beach", "Cleveland"),
]


def compute_continuous_feature_shift(x_train: np.ndarray, x_test: np.ndarray) -> dict:
    """Computes both raw and source-standardized continuous distribution shift."""
    x_tr = np.asarray(x_train, dtype=float)
    x_te = np.asarray(x_test, dtype=float)

    mean_tr, std_tr = float(np.mean(x_tr)), float(np.std(x_tr, ddof=1))
    mean_te, std_te = float(np.mean(x_te)), float(np.std(x_te, ddof=1))

    # 1. Raw Wasserstein distance (in physical units)
    raw_w1 = stats.wasserstein_distance(x_tr, x_te)

    # 2. Source-standardized features: (x - mu_src) / sigma_src
    if std_tr > 1e-8:
        z_tr = (x_tr - mean_tr) / std_tr
        z_te = (x_te - mean_tr) / std_tr
        std_w1 = stats.wasserstein_distance(z_tr, z_te)
    else:
        std_w1 = 0.0

    # Standardized Mean Difference (Cohen's d)
    pooled_std = np.sqrt((std_tr**2 + std_te**2) / 2.0)
    smd = (mean_tr - mean_te) / pooled_std if pooled_std > 1e-8 else 0.0

    # Kolmogorov-Smirnov 2-sample test
    ks_res = stats.ks_2samp(x_tr, x_te)

    return {
        "train_mean": round(mean_tr, 3),
        "train_std": round(std_tr, 3),
        "test_mean": round(mean_te, 3),
        "test_std": round(std_te, 3),
        "raw_wasserstein_distance": round(float(raw_w1), 4),
        "standardized_wasserstein_distance": round(float(std_w1), 4),
        "smd": round(float(smd), 4),
        "ks_statistic": round(float(ks_res.statistic), 4),
        "ks_pvalue": round(float(ks_res.pvalue), 6)
    }


def compute_categorical_feature_shift(s_train: pd.Series, s_test: pd.Series) -> dict:
    """Computes TVD, JSD, and Chi-Square for discrete categorical features."""
    all_cats = sorted(list(set(s_train.dropna().unique()).union(set(s_test.dropna().unique()))))

    p_tr = s_train.value_counts(normalize=True).reindex(all_cats, fill_value=0.0).to_numpy()
    p_te = s_test.value_counts(normalize=True).reindex(all_cats, fill_value=0.0).to_numpy()

    tvd = 0.5 * np.sum(np.abs(p_tr - p_te))
    jsd = dist.jensenshannon(p_tr, p_te, base=2)**2
    if np.isnan(jsd):
        jsd = 0.0

    cohort_labels = ["train"] * len(s_train) + ["test"] * len(s_test)
    combined_series = list(s_train) + list(s_test)
    contingency = pd.crosstab(pd.Series(cohort_labels), pd.Series(combined_series))
    try:
        chi2_res = stats.chi2_contingency(contingency)
        chi2_stat = float(chi2_res.statistic)
        chi2_pval = float(chi2_res.pvalue)
    except Exception:
        chi2_stat, chi2_pval = 0.0, 1.0

    return {
        "tvd": round(float(tvd), 4),
        "jensen_shannon_divergence": round(float(jsd), 4),
        "chi2_statistic": round(chi2_stat, 3),
        "chi2_pvalue": round(chi2_pval, 6)
    }


def run_r4_dataset_shift_correction():
    print("\n=======================================================")
    print("Executing R4: Standardized Dataset Shift Recalculation")
    print("=======================================================")

    # Load all harmonized cohorts
    cohorts = {}
    for c_name in ["cleveland", "hungarian", "zurich", "va_long_beach"]:
        disp_name = "VA Long Beach" if c_name == "va_long_beach" else c_name.capitalize()
        cohorts[disp_name] = load_harmonized_cohort(
            c_name, track="B10", data_dir=RAW_DATA_DIR, drop_duplicates=True, mask_zero_sentinels=True
        )

    feature_shift_records = []

    for tr_name, te_name in TRANSFER_DIRECTIONS:
        df_tr = cohorts[tr_name]
        df_te = cohorts[te_name]

        for feat in TRACK_B_10_FEATURES:
            if feat in CONTINUOUS_FEATURES:
                c_res = compute_continuous_feature_shift(df_tr[feat].dropna().values, df_te[feat].dropna().values)
                rec = {
                    "train_dataset": tr_name,
                    "test_dataset": te_name,
                    "feature": feat,
                    "feature_type": "continuous",
                    **c_res,
                    "tvd": np.nan,
                    "jensen_shannon_divergence": np.nan,
                    "chi2_statistic": np.nan,
                    "chi2_pvalue": np.nan
                }
            else:
                d_res = compute_categorical_feature_shift(df_tr[feat], df_te[feat])
                rec = {
                    "train_dataset": tr_name,
                    "test_dataset": te_name,
                    "feature": feat,
                    "feature_type": "categorical",
                    "train_mean": np.nan,
                    "train_std": np.nan,
                    "test_mean": np.nan,
                    "test_std": np.nan,
                    "raw_wasserstein_distance": np.nan,
                    "standardized_wasserstein_distance": np.nan,
                    "smd": np.nan,
                    "ks_statistic": np.nan,
                    "ks_pvalue": np.nan,
                    **d_res
                }
            feature_shift_records.append(rec)

    feat_shift_df = pd.DataFrame(feature_shift_records)
    out_feat_path = os.path.join(CORRECTED_DIR, "feature_shift_summary_standardized.csv")
    feat_shift_df.to_csv(out_feat_path, index=False)
    print(f"[OK] Saved per-feature standardized shifts to: {out_feat_path}")

    # Aggregate direction-level shifts
    cont_df = feat_shift_df[feat_shift_df["feature_type"] == "continuous"]
    cat_df = feat_shift_df[feat_shift_df["feature_type"] == "categorical"]

    agg_cont = cont_df.groupby(["train_dataset", "test_dataset"]).agg(
        mean_standardized_wasserstein_distance=("standardized_wasserstein_distance", "mean"),
        mean_raw_wasserstein_distance=("raw_wasserstein_distance", "mean"),
        mean_abs_smd=("smd", lambda s: float(np.mean(np.abs(s))))
    ).reset_index()

    agg_cat = cat_df.groupby(["train_dataset", "test_dataset"]).agg(
        mean_jsd=("jensen_shannon_divergence", "mean"),
        mean_tvd=("tvd", "mean")
    ).reset_index()

    dir_summary = pd.merge(agg_cont, agg_cat, on=["train_dataset", "test_dataset"])

    # Map prevalence shifts
    prev_dict = {}
    for c_name, df_c in cohorts.items():
        prev_dict[c_name] = float(np.mean(df_c["target"])) * 100.0

    dir_summary["train_prevalence_pct"] = dir_summary["train_dataset"].map(prev_dict)
    dir_summary["test_prevalence_pct"] = dir_summary["test_dataset"].map(prev_dict)
    dir_summary["abs_prevalence_delta_pct"] = np.round(np.abs(dir_summary["train_prevalence_pct"] - dir_summary["test_prevalence_pct"]), 2)

    # Link with 24 experiments
    ext_res_path = os.path.join(RESULTS_DIR, "external_validation_results.csv")
    df_ext = pd.read_csv(ext_res_path)

    # Load SHAP stability (original Phase 6 for baseline linkage)
    shap_path = os.path.join(RESULTS_DIR, "shap_rank_stability.csv")
    df_shap = pd.read_csv(shap_path).drop_duplicates(subset=["experiment_id"])

    merged_rel = pd.merge(df_ext[["experiment_id", "train_dataset", "test_dataset", "model", "external_roc_auc", "auc_change", "external_brier", "brier_change"]], dir_summary, on=["train_dataset", "test_dataset"])
    merged_rel = pd.merge(merged_rel, df_shap[["experiment_id", "spearman_rank_correlation", "kendall_rank_correlation"]], on="experiment_id", how="left")

    out_rel_path = os.path.join(CORRECTED_DIR, "dataset_shift_relationships_standardized.csv")
    merged_rel.to_csv(out_rel_path, index=False)
    print(f"[OK] Saved standardized dataset shift relationships to: {out_rel_path}")

    return feat_shift_df, merged_rel


if __name__ == "__main__":
    run_r4_dataset_shift_correction()
