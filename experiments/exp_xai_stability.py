"""
Phase 6: Explanation Stability & XAI Transportability Engine (SHAP & LIME).
Implements:
  - 24 Primary Pairwise Transfer Experiments (6 directions x 4 models) under Track B (10 features).
  - Absolute explainer isolation: Explainers explain the development-trained model; zero external labels accessed.
  - Dual Explainers: SHAP (primary, Tree/Linear/Kernel) and LIME (secondary local robustness).
  - Global Feature Importance & Rank Stability: Spearman's rho, Kendall's tau, Top-3 and Top-5 Jaccard overlap.
  - Feature Rank Displacement and Attribution Drift: Delta SHAP and relative change.
  - Local Explanation Consistency on deterministic risk-stratified patient cohorts.
  - Empirical synthesis connecting Phase 5 performance/calibration degradation to Phase 6 explanation stability.
  - Generates publication figures (300 DPI), comprehensive CSV registries, and execution manifest.
"""

import os
import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import lime
import lime.lime_tabular

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_B_10_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from experiments.exp_internal_validation import load_config, build_model_pipeline


def compute_shap_values(model, model_name: str, X_dev: pd.DataFrame, X_eval: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """
    Computes SHAP values on X_eval for the given trained model.
    Guarantees no external labels are accessed.
    Returns:
        shap_vals_matrix: np.ndarray of shape (N, 10) representing class 1 (CAD risk) attribution.
    """
    n_features = len(TRACK_B_10_FEATURES)

    if model_name == "LogisticRegression":
        # Logistic Regression pipeline: scaler + clf
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_dev_scaled = pd.DataFrame(scaler.transform(X_dev), columns=TRACK_B_10_FEATURES)
        X_eval_scaled = pd.DataFrame(scaler.transform(X_eval), columns=TRACK_B_10_FEATURES)

        masker = shap.maskers.Independent(X_dev_scaled)
        explainer = shap.LinearExplainer(clf, masker=masker)
        shap_res = explainer(X_eval_scaled)
        vals = shap_res.values
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        return vals

    elif model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_res = explainer(X_eval)
        vals = shap_res.values
        # If binary classification output is 3D (samples, features, classes), extract class 1
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        elif vals.ndim == 1:
            vals = vals.reshape(1, -1)
        return vals

    elif model_name == "SVM":
        # SVM pipeline with RBF kernel: KernelExplainer with deterministic background sample
        bg_sample = shap.sample(X_dev, min(len(X_dev), 50), random_state=seed)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, bg_sample)
        # Suppress stdout progress during kernel calculation
        vals = explainer.shap_values(X_eval, nsamples=100, silent=True)
        if isinstance(vals, list) and len(vals) == 2:
            vals = vals[1]
        elif isinstance(vals, list) and len(vals) == 1:
            vals = vals[0]
        return np.array(vals)

    else:
        raise ValueError(f"Unsupported model family for SHAP: {model_name}")


def summarize_feature_attributions(shap_matrix: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Computes summary statistics (mean, median, std, IQR, rank) for absolute SHAP attributions."""
    abs_matrix = np.abs(shap_matrix)
    rows = []

    mean_vals = np.mean(abs_matrix, axis=0)
    median_vals = np.median(abs_matrix, axis=0)
    std_vals = np.std(abs_matrix, axis=0)
    iqr_vals = stats.iqr(abs_matrix, axis=0)

    # Ranks: highest mean absolute SHAP = rank 1
    rank_indices = np.argsort(-mean_vals)
    ranks = np.empty_like(rank_indices)
    ranks[rank_indices] = np.arange(1, len(feature_names) + 1)

    for idx, f_name in enumerate(feature_names):
        rows.append({
            "feature": f_name,
            "mean_abs_shap": round(float(mean_vals[idx]), 6),
            "median_abs_shap": round(float(median_vals[idx]), 6),
            "std_abs_shap": round(float(std_vals[idx]), 6),
            "iqr_abs_shap": round(float(iqr_vals[idx]), 6),
            "feature_rank": int(ranks[idx])
        })
    return pd.DataFrame(rows)


def compute_rank_stability(dev_summary_df: pd.DataFrame, ext_summary_df: pd.DataFrame) -> tuple:
    """
    Computes rank stability metrics comparing development vs external feature importance:
    - Spearman's rho
    - Kendall's tau
    - Top-3 overlap (Jaccard)
    - Top-5 overlap (Jaccard)
    - Detailed feature displacement dataframe
    """
    merged = pd.merge(
        dev_summary_df,
        ext_summary_df,
        on="feature",
        suffixes=("_dev", "_ext")
    )

    dev_ranks = merged["feature_rank_dev"].to_numpy()
    ext_ranks = merged["feature_rank_ext"].to_numpy()

    # 1. Spearman Rank Correlation
    spearman_res = stats.spearmanr(dev_ranks, ext_ranks)
    spearman_rho = round(float(spearman_res.statistic), 4)

    # 2. Kendall's Tau Correlation
    kendall_res = stats.kendalltau(dev_ranks, ext_ranks)
    kendall_tau = round(float(kendall_res.statistic), 4)

    # 3. Top-k Overlaps (Jaccard Similarity)
    top3_dev = set(merged.sort_values("feature_rank_dev")["feature"].head(3))
    top3_ext = set(merged.sort_values("feature_rank_ext")["feature"].head(3))
    top3_jaccard = round(len(top3_dev.intersection(top3_ext)) / len(top3_dev.union(top3_ext)), 4)

    top5_dev = set(merged.sort_values("feature_rank_dev")["feature"].head(5))
    top5_ext = set(merged.sort_values("feature_rank_ext")["feature"].head(5))
    top5_jaccard = round(len(top5_dev.intersection(top5_ext)) / len(top5_dev.union(top5_ext)), 4)

    # 4. Feature-level Displacement Table
    displacement_rows = []
    for _, row in merged.iterrows():
        f = row["feature"]
        d_rank = int(row["feature_rank_dev"])
        e_rank = int(row["feature_rank_ext"])
        rank_diff = e_rank - d_rank  # Positive = dropped in rank externally; Negative = gained rank

        d_mean = row["mean_abs_shap_dev"]
        e_mean = row["mean_abs_shap_ext"]
        delta_shap = round(e_mean - d_mean, 6)
        rel_chg = round((delta_shap / d_mean) * 100, 2) if d_mean > 0 else 0.0

        displacement_rows.append({
            "feature": f,
            "development_rank": d_rank,
            "external_rank": e_rank,
            "rank_difference": rank_diff,
            "development_mean_abs_shap": d_mean,
            "external_mean_abs_shap": e_mean,
            "delta_shap": delta_shap,
            "relative_change_percent": rel_chg
        })

    stability_metrics = {
        "spearman_rank_correlation": spearman_rho,
        "kendall_rank_correlation": kendall_tau,
        "top3_overlap": top3_jaccard,
        "top5_overlap": top5_jaccard,
        "top3_dev_features": list(top3_dev),
        "top3_ext_features": list(top3_ext),
        "top5_dev_features": list(top5_dev),
        "top5_ext_features": list(top5_ext)
    }
    return stability_metrics, pd.DataFrame(displacement_rows)


def select_deterministic_sample(X_df: pd.DataFrame, model, n_samples: int = 25, seed: int = 42) -> np.ndarray:
    """
    Selects a deterministic, outcome-independent sample of patients evenly spaced across
    the model's predicted probability spectrum (ensuring low, medium, and high-risk strata are covered).
    Never uses external labels.
    """
    probs = model.predict_proba(X_df)[:, 1]
    sorted_indices = np.argsort(probs)
    # Select evenly spaced quantile indices across sorted array
    selected_idx = np.linspace(0, len(X_df) - 1, n_samples, dtype=int)
    return sorted_indices[selected_idx]


def compute_local_explanation_metrics(
    shap_matrix: np.ndarray,
    selected_indices: np.ndarray
) -> dict:
    """
    Calculates local explanation vector consistency across the deterministic cohort:
    - Mean pairwise cosine similarity
    - Mean pairwise Spearman attribution correlation
    """
    sub_matrix = shap_matrix[selected_indices]
    n_sub = len(sub_matrix)

    cos_sims = []
    spear_corrs = []

    for i in range(n_sub):
        v1 = sub_matrix[i]
        norm1 = np.linalg.norm(v1)
        if norm1 == 0:
            continue
        for j in range(i + 1, n_sub):
            v2 = sub_matrix[j]
            norm2 = np.linalg.norm(v2)
            if norm2 == 0:
                continue
            cos_sim = np.dot(v1, v2) / (norm1 * norm2)
            cos_sims.append(cos_sim)

            sp_res = stats.spearmanr(v1, v2)
            if not np.isnan(sp_res.statistic):
                spear_corrs.append(sp_res.statistic)

    return {
        "mean_local_cosine_similarity": round(float(np.mean(cos_sims)), 4) if cos_sims else 0.0,
        "std_local_cosine_similarity": round(float(np.std(cos_sims)), 4) if cos_sims else 0.0,
        "mean_local_spearman_correlation": round(float(np.mean(spear_corrs)), 4) if spear_corrs else 0.0,
        "std_local_spearman_correlation": round(float(np.std(spear_corrs)), 4) if spear_corrs else 0.0
    }


def compute_lime_explanations(
    model,
    X_dev: pd.DataFrame,
    X_ext: pd.DataFrame,
    selected_indices: np.ndarray,
    seed: int = 42
) -> pd.DataFrame:
    """
    Computes complementary local LIME explanations for the deterministic patient subset.
    Quantifies feature rank consistency and attribution directionality.
    """
    cat_indices = [
        TRACK_B_10_FEATURES.index(c)
        for c in ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    ]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_dev),
        feature_names=TRACK_B_10_FEATURES,
        categorical_features=cat_indices,
        class_names=["No CAD", "CAD"],
        mode="classification",
        random_state=seed,
        discretize_continuous=True
    )

    lime_rows = []
    for sample_pos, idx in enumerate(selected_indices):
        patient_vector = X_ext.iloc[idx].values
        # Predict probability for class 1
        exp = lime_explainer.explain_instance(
            patient_vector,
            model.predict_proba,
            num_features=len(TRACK_B_10_FEATURES),
            num_samples=1000
        )

        # Parse LIME weights
        feature_weights = {}
        for feat_desc, weight in exp.as_list():
            # Match feature name from description
            for base_col in TRACK_B_10_FEATURES:
                if base_col in feat_desc:
                    feature_weights[base_col] = weight
                    break

        for f_name in TRACK_B_10_FEATURES:
            w = feature_weights.get(f_name, 0.0)
            lime_rows.append({
                "patient_sample_index": int(idx),
                "deterministic_rank_stratum": int(sample_pos),
                "feature": f_name,
                "lime_weight": round(float(w), 6),
                "abs_lime_weight": round(float(abs(w)), 6)
            })

    return pd.DataFrame(lime_rows)


def run_all_xai_experiments():
    config = load_config("config.yaml")
    seed = config["experiment"]["random_seed"]
    raw_dir = os.path.join(ROOT_DIR, config["paths"]["raw_data_dir"])
    results_dir = os.path.join(ROOT_DIR, config["paths"]["results_dir"])
    figures_xai_dir = os.path.join(ROOT_DIR, "figures", "xai")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_xai_dir, exist_ok=True)

    # 1. Load Phase 5 Pairwise Results for Cross-Phase Degradation Merging
    p5_res_path = os.path.join(results_dir, "external_validation_results.csv")
    if not os.path.exists(p5_res_path):
        raise FileNotFoundError(f"Phase 5 results not found at: {p5_res_path}")
    p5_df = pd.read_csv(p5_res_path)

    pairwise_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland"),
    ]
    models = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]

    all_shap_external_records = []
    all_feature_importance_records = []
    all_rank_stability_records = []
    all_displacement_records = []
    all_local_stability_records = []
    all_lime_records = []

    print("=== Starting Phase 6 Explanation Stability & XAI Transportability Experiments ===")
    exp_counter = 1

    for train_c, test_c in pairwise_directions:
        # Ingest cohorts
        dev_df = load_harmonized_cohort(train_c, track="B10", data_dir=raw_dir)
        ext_df = load_harmonized_cohort(test_c, track="B10", data_dir=raw_dir)

        X_dev_raw = dev_df[TRACK_B_10_FEATURES].copy()
        y_dev = dev_df["target"].to_numpy()
        X_ext_raw = ext_df[TRACK_B_10_FEATURES].copy()
        # External labels strictly sequestered

        # Fit preprocessor strictly on DEV
        preprocessor = HarmonizedClinicalPreprocessor(
            imputer_strategy=config["preprocessing"]["imputer_strategy"],
            n_neighbors=config["preprocessing"]["n_neighbors"],
            scale_features=False,
            clamp_negative_oldpeak=config["preprocessing"]["clamp_negative_oldpeak"]
        )
        X_dev_proc = preprocessor.fit_transform(X_dev_raw)
        X_ext_proc = preprocessor.transform(X_ext_raw)

        for m_name in models:
            exp_id = f"PAIRWISE_E{exp_counter}_{train_c[:4].upper()}_TO_{test_c[:4].upper()}_{m_name}"
            print(f"\n[{exp_id}] Explaining {train_c} -> {test_c} | Model: {m_name}...")

            # 1. Fit Model on DEV
            model = build_model_pipeline(m_name, config)
            model.fit(X_dev_proc, y_dev)

            # 2. Compute SHAP on Development Cohort
            print("  Computing SHAP on Development Cohort...")
            dev_shap = compute_shap_values(model, m_name, X_dev_proc, X_dev_proc, seed=seed)
            dev_summary = summarize_feature_attributions(dev_shap, TRACK_B_10_FEATURES)
            for _, r in dev_summary.iterrows():
                all_feature_importance_records.append({
                    "experiment_id": exp_id,
                    "train_dataset": train_c,
                    "test_dataset": test_c,
                    "model": m_name,
                    "cohort_type": "development",
                    **r.to_dict()
                })

            # 3. Compute SHAP on External Test Cohort (Zero External Labels)
            print("  Computing SHAP on External Test Cohort...")
            ext_shap = compute_shap_values(model, m_name, X_dev_proc, X_ext_proc, seed=seed)
            ext_summary = summarize_feature_attributions(ext_shap, TRACK_B_10_FEATURES)
            for _, r in ext_summary.iterrows():
                all_feature_importance_records.append({
                    "experiment_id": exp_id,
                    "train_dataset": train_c,
                    "test_dataset": test_c,
                    "model": m_name,
                    "cohort_type": "external",
                    **r.to_dict()
                })

            # Log external sample-level SHAP values
            for idx, shap_row in enumerate(ext_shap):
                rec = {
                    "experiment_id": exp_id,
                    "train_dataset": train_c,
                    "test_dataset": test_c,
                    "model": m_name,
                    "sample_index": idx
                }
                for f_idx, f_name in enumerate(TRACK_B_10_FEATURES):
                    rec[f_name] = round(float(shap_row[f_idx]), 6)
                all_shap_external_records.append(rec)

            # 4. Compute Explanation Rank Stability Metrics & Feature Displacement
            stability_mets, disp_df = compute_rank_stability(dev_summary, ext_summary)
            for _, r in disp_df.iterrows():
                all_displacement_records.append({
                    "experiment_id": exp_id,
                    "train_dataset": train_c,
                    "test_dataset": test_c,
                    "model": m_name,
                    **r.to_dict()
                })

            all_rank_stability_records.append({
                "experiment_id": exp_id,
                "train_dataset": train_c,
                "test_dataset": test_c,
                "model": m_name,
                **stability_mets
            })

            # 5. Deterministic Local Explanation Cohort Analysis
            print("  Computing Local Explanation Consistency...")
            det_indices = select_deterministic_sample(X_ext_proc, model, n_samples=25, seed=seed)
            local_mets = compute_local_explanation_metrics(ext_shap, det_indices)
            all_local_stability_records.append({
                "experiment_id": exp_id,
                "train_dataset": train_c,
                "test_dataset": test_c,
                "model": m_name,
                "n_local_samples": len(det_indices),
                **local_mets
            })

            # 6. Complementary LIME Robustness Analysis
            print("  Computing Complementary LIME Local Explanations...")
            lime_df = compute_lime_explanations(model, X_dev_proc, X_ext_proc, det_indices, seed=seed)
            lime_df["experiment_id"] = exp_id
            lime_df["train_dataset"] = train_c
            lime_df["test_dataset"] = test_c
            lime_df["model"] = m_name
            all_lime_records.extend(lime_df.to_dict(orient="records"))

            exp_counter += 1

    # Save Output CSVs
    print("\n--- Saving Phase 6 Explanation Datasets & Registries ---")

    # 1. External SHAP Values Registry (6,096 rows)
    shap_ext_df = pd.DataFrame(all_shap_external_records)
    shap_ext_path = os.path.join(results_dir, "shap_values_external.csv")
    shap_ext_df.to_csv(shap_ext_path, index=False)
    print(f"[OK] Saved external SHAP values ({len(shap_ext_df)} rows) to: {shap_ext_path}")

    # 2. SHAP Feature Importance Summary (24 exps x 2 cohorts x 10 features = 480 rows)
    feat_imp_df = pd.DataFrame(all_feature_importance_records)
    feat_imp_path = os.path.join(results_dir, "shap_feature_importance.csv")
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"[OK] Saved SHAP feature importance summary ({len(feat_imp_df)} rows) to: {feat_imp_path}")

    # 3. SHAP Rank Stability & Displacement Table (240 feature rows + 24 experiment summaries)
    disp_df = pd.DataFrame(all_displacement_records)
    stab_df = pd.DataFrame(all_rank_stability_records)
    rank_stab_df = pd.merge(disp_df, stab_df[["experiment_id", "spearman_rank_correlation", "kendall_rank_correlation", "top3_overlap", "top5_overlap"]], on="experiment_id")
    rank_stab_path = os.path.join(results_dir, "shap_rank_stability.csv")
    rank_stab_df.to_csv(rank_stab_path, index=False)
    print(f"[OK] Saved SHAP rank stability & displacement ({len(rank_stab_df)} rows) to: {rank_stab_path}")

    # 4. Local SHAP Consistency Table (24 rows)
    local_stab_df = pd.DataFrame(all_local_stability_records)
    local_stab_path = os.path.join(results_dir, "shap_local_stability.csv")
    local_stab_df.to_csv(local_stab_path, index=False)
    print(f"[OK] Saved local SHAP consistency ({len(local_stab_df)} rows) to: {local_stab_path}")

    # 5. LIME Stability Table
    lime_all_df = pd.DataFrame(all_lime_records)
    lime_path = os.path.join(results_dir, "lime_stability.csv")
    lime_all_df.to_csv(lime_path, index=False)
    print(f"[OK] Saved LIME stability registry ({len(lime_all_df)} rows) to: {lime_path}")

    # 6. Cross-Phase Synthesis: Explanation Stability vs Performance Degradation
    print("\n--- Merging Phase 5 Performance Degradation with Phase 6 Explanation Stability ---")
    p5_merged = pd.merge(
        p5_df[[
            "experiment_id", "train_dataset", "test_dataset", "model",
            "auc_change", "pr_auc_change", "brier_change",
            "sensitivity_change", "specificity_change", "mcc_change",
            "external_calibration_slope", "external_calibration_intercept", "external_ece"
        ]],
        stab_df[[
            "experiment_id", "spearman_rank_correlation", "kendall_rank_correlation",
            "top3_overlap", "top5_overlap"
        ]],
        on=["experiment_id"]
    )
    # Add mean absolute delta SHAP across 10 features for each experiment
    mean_delta_shap = disp_df.groupby("experiment_id")["delta_shap"].apply(lambda s: round(float(np.mean(np.abs(s))), 6)).reset_index()
    mean_delta_shap.columns = ["experiment_id", "mean_abs_delta_shap"]
    p5_merged = pd.merge(p5_merged, mean_delta_shap, on="experiment_id")

    cross_phase_path = os.path.join(results_dir, "xai_performance_relationships.csv")
    p5_merged.to_csv(cross_phase_path, index=False)
    print(f"[OK] Saved cross-phase performance vs explanation synthesis to: {cross_phase_path}")

    # 7. Execution Manifest
    manifest = {
        "metadata": {
            "title": "Phase 6 Explanation Stability & XAI Transportability Manifest",
            "project": "Beyond Accuracy: Cross-Dataset Generalization, Calibration and Explanation Stability",
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "shap_version": shap.__version__,
            "lime_version": getattr(lime, "__version__", "0.2.0.1"),
            "random_seed": seed,
            "feature_schema": TRACK_B_10_FEATURES,
            "models_evaluated": models,
            "pairwise_transfer_directions": [f"{tr} -> {te}" for tr, te in pairwise_directions],
            "total_experiments": len(stab_df),
            "files_generated": [
                "results/shap_values_external.csv",
                "results/shap_feature_importance.csv",
                "results/shap_rank_stability.csv",
                "results/shap_local_stability.csv",
                "results/lime_stability.csv",
                "results/xai_performance_relationships.csv"
            ],
            "isolation_rule": "Explainers explain development-fitted models; external labels strictly sequestered"
        }
    }
    manifest_path = os.path.join(results_dir, "xai_explanation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Saved XAI manifest to: {manifest_path}")

    # 8. Generate 10 High-Resolution Publication Figures
    print("\n--- Generating Publication-Quality Figures (300 DPI) ---")
    generate_xai_figures(
        feat_imp_df,
        rank_stab_df,
        stab_df,
        p5_merged,
        shap_ext_df,
        figures_xai_dir
    )

    return stab_df, p5_merged


def generate_xai_figures(feat_imp_df, rank_stab_df, stab_df, cross_phase_df, shap_ext_df, out_dir):
    """Generates 10 publication-quality figures documenting explanation stability and transportability."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "SVM": "#ff7f0e", "XGBoost": "#d62728"}
    cross_phase_df = cross_phase_df.copy()
    cross_phase_df["direction"] = cross_phase_df["train_dataset"] + " -> " + cross_phase_df["test_dataset"]

    # 1. Figure: SHAP Global Feature Importance - Cleveland Development vs External
    plt.figure(figsize=(12, 6))
    clev_hung_rf = feat_imp_df[
        (feat_imp_df["train_dataset"] == "Cleveland") &
        (feat_imp_df["test_dataset"] == "Hungarian") &
        (feat_imp_df["model"] == "RandomForest")
    ]
    sns.barplot(
        data=clev_hung_rf,
        x="mean_abs_shap",
        y="feature",
        hue="cohort_type",
        palette=["#2b5c8f", "#d95f02"],
        order=clev_hung_rf[clev_hung_rf["cohort_type"] == "development"].sort_values("mean_abs_shap", ascending=False)["feature"]
    )
    plt.title("SHAP Global Feature Importance: Development (Cleveland) vs External (Hungarian) [Random Forest]", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model Log-Odds / Risk)", fontsize=11)
    plt.ylabel("Clinical Biomarker", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_shap_importance_comparison.png"), dpi=300)
    plt.close()

    # 2. Figure: Representative Beeswarm Plot (Cleveland -> Hungarian Random Forest)
    plt.figure(figsize=(10, 6))
    sample_exp_id = "PAIRWISE_E2_CLEV_TO_HUNG_RandomForest"
    sub_shap = shap_ext_df[shap_ext_df["experiment_id"] == sample_exp_id][TRACK_B_10_FEATURES].to_numpy()
    # Mock summary bar / beeswarm projection
    mean_abs = np.mean(np.abs(sub_shap), axis=0)
    sorted_order = np.argsort(mean_abs)
    plt.barh(np.array(TRACK_B_10_FEATURES)[sorted_order], mean_abs[sorted_order], color="#2ca02c", alpha=0.85)
    plt.title("External SHAP Attributions: Cleveland -> Hungarian (Random Forest)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP| on External Hungarian Patients", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_shap_beeswarm_representative.png"), dpi=300)
    plt.close()

    # 3. Figure: Feature Rank Displacement Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=rank_stab_df,
        x="development_rank",
        y="external_rank",
        hue="model",
        style="feature",
        palette=palette,
        s=120,
        alpha=0.85
    )
    plt.plot([1, 10], [1, 10], "k--", alpha=0.6, label="Ideal Stability Line (Rank Unchanged)")
    plt.title("Feature Rank Displacement: Development Rank vs External Test Rank", fontsize=13, fontweight="bold")
    plt.xlabel("Development Feature Rank (1 = Most Important)", fontsize=11)
    plt.ylabel("External Test Feature Rank (1 = Most Important)", fontsize=11)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_shap_rank_displacement.png"), dpi=300)
    plt.close()

    # 4. Figure: SHAP Rank Correlation Heatmap Across Transfer Directions
    plt.figure(figsize=(11, 6))
    stab_df["direction"] = stab_df["train_dataset"] + " -> " + stab_df["test_dataset"]
    heatmap_pivot = stab_df.pivot(index="direction", columns="model", values="spearman_rank_correlation")
    sns.heatmap(heatmap_pivot, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={"label": "Spearman Rank Correlation (rho)"}, vmin=0.3, vmax=1.0)
    plt.title("Explanation Rank Stability (Spearman rho) by Hospital Transfer Direction", fontsize=13, fontweight="bold")
    plt.ylabel("Hospital Transfer Direction", fontsize=11)
    plt.xlabel("Model Family", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_shap_rank_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 5. Figure: Top-3 and Top-5 Feature Overlap by Transfer Direction
    plt.figure(figsize=(13, 6))
    melted_overlap = pd.melt(
        stab_df,
        id_vars=["direction", "model"],
        value_vars=["top3_overlap", "top5_overlap"],
        var_name="overlap_metric",
        value_name="jaccard_similarity"
    )
    melted_overlap["overlap_label"] = melted_overlap["overlap_metric"].map({"top3_overlap": "Top-3 Features Overlap", "top5_overlap": "Top-5 Features Overlap"})
    sns.barplot(data=melted_overlap, x="direction", y="jaccard_similarity", hue="overlap_label", palette="Set2")
    plt.title("Top-3 and Top-5 Feature Overlap (Jaccard Index) Across Transfer Directions", fontsize=13, fontweight="bold")
    plt.xlabel("Hospital Transfer Direction", fontsize=11)
    plt.ylabel("Jaccard Overlap Index", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_top_k_overlap_comparison.png"), dpi=300)
    plt.close()

    # 6. Figure: Explanation Stability vs ROC-AUC Degradation
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=cross_phase_df,
        x="auc_change",
        y="spearman_rank_correlation",
        hue="model",
        palette=palette,
        s=140,
        alpha=0.9
    )
    # Regression trendline
    sns.regplot(
        data=cross_phase_df,
        x="auc_change",
        y="spearman_rank_correlation",
        scatter=False,
        color="gray",
        line_kws={"linestyle": "--", "alpha": 0.7}
    )
    plt.axvline(0, color="k", linestyle=":", alpha=0.5)
    plt.title("Explanation Stability (Spearman rho) vs Discrimination Degradation (Delta AUC)", fontsize=13, fontweight="bold")
    plt.xlabel("Delta ROC-AUC (pp) [External AUC - Internal Ref AUC]", fontsize=11)
    plt.ylabel("SHAP Feature Rank Correlation (rho)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "6_stability_vs_auc_degradation.png"), dpi=300)
    plt.close()

    # 7. Figure: Explanation Stability vs Brier Degradation
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=cross_phase_df,
        x="brier_change",
        y="spearman_rank_correlation",
        hue="model",
        palette=palette,
        s=140,
        alpha=0.9
    )
    sns.regplot(
        data=cross_phase_df,
        x="brier_change",
        y="spearman_rank_correlation",
        scatter=False,
        color="gray",
        line_kws={"linestyle": "--", "alpha": 0.7}
    )
    plt.axvline(0, color="k", linestyle=":", alpha=0.5)
    plt.title("Explanation Stability (Spearman rho) vs Calibration Deterioration (Delta Brier)", fontsize=13, fontweight="bold")
    plt.xlabel("Delta Brier Score [Positive = Degraded Calibration Error]", fontsize=11)
    plt.ylabel("SHAP Feature Rank Correlation (rho)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "7_stability_vs_brier_degradation.png"), dpi=300)
    plt.close()

    # 8. Figure: Directional Explanation Stability Comparison (Asymmetry)
    plt.figure(figsize=(10, 6))
    # Paired bidirectional comparisons
    pairs = [
        ("Cleveland -> Hungarian", "Hungarian -> Cleveland"),
        ("Cleveland -> VA Long Beach", "VA Long Beach -> Cleveland"),
        ("Cleveland -> Zurich", "Zurich -> Cleveland")
    ]
    pair_rows = []
    for fwd, rev in pairs:
        rho_fwd = stab_df[stab_df["direction"] == fwd]["spearman_rank_correlation"].mean()
        rho_rev = stab_df[stab_df["direction"] == rev]["spearman_rank_correlation"].mean()
        pair_rows.append({"Pair": f"{fwd}\nvs\n{rev}", "Direction": "Forward", "Mean_rho": rho_fwd})
        pair_rows.append({"Pair": f"{fwd}\nvs\n{rev}", "Direction": "Reverse", "Mean_rho": rho_rev})
    pair_df = pd.DataFrame(pair_rows)
    sns.barplot(data=pair_df, x="Pair", y="Mean_rho", hue="Direction", palette=["#2ca02c", "#1f77b4"])
    plt.title("Directional Asymmetry in Explanation Stability (Mean Spearman rho)", fontsize=13, fontweight="bold")
    plt.ylabel("Mean Spearman Rank Correlation across 4 Models", fontsize=11)
    plt.xlabel("Bidirectional Pair", fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "8_directional_asymmetry_stability.png"), dpi=300)
    plt.close()

    # 9. Figure: Mean Absolute Delta SHAP across Models
    plt.figure(figsize=(11, 6))
    sns.barplot(data=cross_phase_df, x="direction", y="mean_abs_delta_shap", hue="model", palette=palette)
    plt.title("Mean Absolute Attribution Drift (|Delta SHAP|) Across Transfer Directions", fontsize=13, fontweight="bold")
    plt.xlabel("Hospital Transfer Direction", fontsize=11)
    plt.ylabel("Mean |Delta SHAP| Across 10 Features", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "9_mean_attribution_drift.png"), dpi=300)
    plt.close()

    # 10. Figure: Explanation Stability vs ECE Calibration Error
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=cross_phase_df,
        x="external_ece",
        y="spearman_rank_correlation",
        hue="model",
        palette=palette,
        s=140,
        alpha=0.9
    )
    plt.title("Explanation Stability (Spearman rho) vs External Calibration Error (ECE)", fontsize=13, fontweight="bold")
    plt.xlabel("External Expected Calibration Error (ECE)", fontsize=11)
    plt.ylabel("SHAP Feature Rank Correlation (rho)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "10_stability_vs_ece.png"), dpi=300)
    plt.close()

    print(f"[OK] Generated all 10 publication-quality figures in: {out_dir}")


if __name__ == "__main__":
    run_all_xai_experiments()
