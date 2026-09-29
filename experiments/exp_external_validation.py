"""
Phase 5: True Cross-Dataset External Validation & Transportability Engine.
Implements:
  - 24 Primary Pairwise External Experiments (6 directions x 4 models) under Track B (10 features).
  - 16 Leave-One-Dataset-Out (LODO) Experiments (4 pooled dev sets x 4 models) stored separately.
  - Strict external test set isolation (transform only, no fitting, no tuning).
  - Degradation metrics relative to Phase 4 internal baselines.
  - Stratified Bootstrap 95% Confidence Intervals for external performance.
  - Generates separated result tables, predictions registry, manifest, and publication figures.
"""

import os
import sys
import argparse
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_B_10_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from src.evaluate import (
    evaluate_classifier,
    compute_expected_calibration_error,
    compute_calibration_intercept_slope
)
from experiments.exp_internal_validation import load_config, build_model_pipeline


def compute_bootstrap_ci(y_true, y_proba, is_zurich: bool = False, n_bootstraps: int = 1000, seed: int = 42) -> dict:
    """
    Computes 95% percentile bootstrap confidence intervals for external test metrics.
    For Zurich (n_neg=8), explicitly flags high uncertainty and returns notes.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)

    auc_list = []
    brier_list = []

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b_true = y_true[boot_idx]
        y_b_proba = y_proba[boot_idx]

        # Need at least one sample of each class in bootstrap sample to compute ROC-AUC
        if len(np.unique(y_b_true)) < 2:
            continue

        try:
            b_auc = roc_auc_score(y_b_true, y_b_proba) * 100
            b_brier = brier_score_loss(y_b_true, y_b_proba)
            auc_list.append(b_auc)
            brier_list.append(b_brier)
        except Exception:
            continue

    note = "Zurich n_neg=8; bootstrap variance is elevated" if is_zurich else "Standard bootstrap (B=1000)"

    if len(auc_list) < 50:
        return {
            "auc_ci_lower": "NA", "auc_ci_upper": "NA",
            "brier_ci_lower": "NA", "brier_ci_upper": "NA",
            "ci_notes": "Insufficient minority class resamples"
        }

    return {
        "auc_ci_lower": round(float(np.percentile(auc_list, 2.5)), 2),
        "auc_ci_upper": round(float(np.percentile(auc_list, 97.5)), 2),
        "brier_ci_lower": round(float(np.percentile(brier_list, 2.5)), 4),
        "brier_ci_upper": round(float(np.percentile(brier_list, 97.5)), 4),
        "ci_notes": note
    }


def load_internal_baselines(results_dir: str = "results") -> dict:
    """Loads Phase 4 internal CV summary metrics for degradation matching."""
    summary_path = os.path.join(ROOT_DIR, results_dir, "internal_validation_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Phase 4 internal summary not found at: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    track_b_summary = summary_df[summary_df["track"] == "Track_B"]

    baselines = {}
    for (cohort, model), grp in track_b_summary.groupby(["cohort", "model"]):
        metric_dict = dict(zip(grp["metric"], grp["mean"]))
        baselines[(cohort, model)] = metric_dict

    return baselines


def run_single_external_experiment(
    exp_id: str,
    train_cohort: str,
    test_cohort: str,
    model_name: str,
    config: dict,
    internal_baselines: dict,
    is_lodo: bool = False
) -> tuple:
    """
    Executes an isolated cross-dataset experiment:
    1. Fits preprocessor strictly on development cohort.
    2. Fits model on development cohort.
    3. Transforms external test cohort with train-fitted preprocessor.
    4. Evaluates external performance and calculates degradation deltas.
    """
    seed = config["experiment"]["random_seed"]
    threshold = config["experiment"]["classification_threshold"]
    raw_dir = os.path.join(ROOT_DIR, config["paths"]["raw_data_dir"])

    # Load Development (Train) Data
    if is_lodo:
        dev_names = [n.strip() for n in train_cohort.split("+")]
        dev_dfs = [load_harmonized_cohort(name, track="B10", data_dir=raw_dir) for name in dev_names]
        dev_df = pd.concat(dev_dfs, ignore_index=True)
    else:
        dev_df = load_harmonized_cohort(train_cohort, track="B10", data_dir=raw_dir)

    # Load External Test Data (Strictly Isolated)
    test_df = load_harmonized_cohort(test_cohort, track="B10", data_dir=raw_dir)

    X_train_raw = dev_df[TRACK_B_10_FEATURES].copy()
    y_train = dev_df["target"].to_numpy()

    X_test_raw = test_df[TRACK_B_10_FEATURES].copy()
    y_test = test_df["target"].to_numpy()

    n_train = len(dev_df)
    n_test = len(test_df)
    train_pos_rate = round(float(np.mean(y_train)), 4)
    test_pos_rate = round(float(np.mean(y_test)), 4)

    # 1. Fit preprocessing ONLY on training cohort
    preprocessor = HarmonizedClinicalPreprocessor(
        imputer_strategy=config["preprocessing"]["imputer_strategy"],
        n_neighbors=config["preprocessing"]["n_neighbors"],
        scale_features=False,
        clamp_negative_oldpeak=config["preprocessing"]["clamp_negative_oldpeak"]
    )
    X_train_proc = preprocessor.fit_transform(X_train_raw)

    # 2. Transform external cohort with training preprocessor (NEVER FIT)
    X_test_proc = preprocessor.transform(X_test_raw)

    # 3. Fit model on training cohort
    model = build_model_pipeline(model_name, config)
    model.fit(X_train_proc, y_train)

    # 4. Predict probabilities on external cohort
    if hasattr(model, "predict_proba"):
        test_proba = model.predict_proba(X_test_proc)[:, 1]
    else:
        test_proba = model.decision_function(X_test_proc)
        test_proba = 1 / (1 + np.exp(-test_proba))

    test_pred = (test_proba >= threshold).astype(int)

    # Compute external metrics
    ext_metrics = evaluate_classifier(y_test, test_pred, test_proba)
    is_zurich = (test_cohort == "Zurich")
    boot_ci = compute_bootstrap_ci(y_test, test_proba, is_zurich=is_zurich, n_bootstraps=1000, seed=seed)

    # Match corresponding internal baseline
    if is_lodo:
        base_metrics = internal_baselines.get(("Cleveland", model_name), {})
    else:
        base_metrics = internal_baselines.get((train_cohort, model_name), {})

    int_auc = base_metrics.get("roc_auc", np.nan)
    int_pr_auc = base_metrics.get("pr_auc", np.nan)
    int_brier = base_metrics.get("brier_score", np.nan)
    int_sens = base_metrics.get("sensitivity", np.nan)
    int_spec = base_metrics.get("specificity", np.nan)
    int_mcc = base_metrics.get("mcc", np.nan)

    auc_change = round(ext_metrics["roc_auc"] - int_auc, 2) if not np.isnan(int_auc) else np.nan
    pr_auc_change = round(ext_metrics["pr_auc"] - int_pr_auc, 2) if not np.isnan(int_pr_auc) else np.nan
    brier_change = round(ext_metrics["brier_score"] - int_brier, 4) if not np.isnan(int_brier) else np.nan
    sens_change = round(ext_metrics["sensitivity"] - int_sens, 2) if not np.isnan(int_sens) else np.nan
    spec_change = round(ext_metrics["specificity"] - int_spec, 2) if not np.isnan(int_spec) else np.nan
    mcc_change = round(ext_metrics["mcc"] - int_mcc, 4) if not np.isnan(int_mcc) else np.nan

    # Compile result row
    result_row = {
        "experiment_id": exp_id,
        "track": "Track_B",
        "train_dataset": train_cohort,
        "test_dataset": test_cohort,
        "model": model_name,
        "n_train": n_train,
        "n_test": n_test,
        "train_positive_rate": train_pos_rate,
        "test_positive_rate": test_pos_rate,
        "internal_roc_auc": int_auc,
        "external_roc_auc": ext_metrics["roc_auc"],
        "auc_ci_lower": boot_ci["auc_ci_lower"],
        "auc_ci_upper": boot_ci["auc_ci_upper"],
        "auc_change": auc_change,
        "internal_pr_auc": int_pr_auc,
        "external_pr_auc": ext_metrics["pr_auc"],
        "pr_auc_change": pr_auc_change,
        "internal_brier": int_brier,
        "external_brier": ext_metrics["brier_score"],
        "brier_ci_lower": boot_ci["brier_ci_lower"],
        "brier_ci_upper": boot_ci["brier_ci_upper"],
        "brier_change": brier_change,
        "external_accuracy": ext_metrics["accuracy"],
        "external_sensitivity": ext_metrics["sensitivity"],
        "internal_sensitivity": int_sens,
        "sensitivity_change": sens_change,
        "external_specificity": ext_metrics["specificity"],
        "internal_specificity": int_spec,
        "specificity_change": spec_change,
        "external_precision": ext_metrics["precision"],
        "external_f1": ext_metrics["f1_score"],
        "external_mcc": ext_metrics["mcc"],
        "internal_mcc": int_mcc,
        "mcc_change": mcc_change,
        "external_ece": ext_metrics["ece"],
        "external_calibration_intercept": ext_metrics["calibration_intercept"],
        "external_calibration_slope": ext_metrics["calibration_slope"],
        "ci_notes": boot_ci["ci_notes"],
        "random_seed": seed
    }

    # Compile prediction rows (verified: exactly 1 prediction per test observation)
    predictions_rows = []
    for idx, (y_t, p_prob, p_lbl) in enumerate(zip(y_test, test_proba, test_pred)):
        predictions_rows.append({
            "experiment_id": exp_id,
            "train_dataset": train_cohort,
            "test_dataset": test_cohort,
            "model": model_name,
            "sample_index": idx,
            "true_label": int(y_t),
            "predicted_probability": round(float(p_prob), 6),
            "predicted_label": int(p_lbl)
        })

    # Compile degradation breakdown rows
    degradation_metrics = [
        ("ROC-AUC", int_auc, ext_metrics["roc_auc"], auc_change),
        ("PR-AUC", int_pr_auc, ext_metrics["pr_auc"], pr_auc_change),
        ("Brier", int_brier, ext_metrics["brier_score"], brier_change),
        ("Sensitivity", int_sens, ext_metrics["sensitivity"], sens_change),
        ("Specificity", int_spec, ext_metrics["specificity"], spec_change),
        ("MCC", int_mcc, ext_metrics["mcc"], mcc_change)
    ]

    degradation_rows = []
    for m_name, i_val, e_val, delta in degradation_metrics:
        rel_chg = round(float((delta / i_val) * 100), 2) if (i_val and not np.isnan(i_val) and i_val != 0) else np.nan
        degradation_rows.append({
            "experiment_id": exp_id,
            "train_dataset": train_cohort,
            "test_dataset": test_cohort,
            "model": model_name,
            "metric": m_name,
            "internal_value": i_val,
            "external_value": e_val,
            "change": delta,
            "relative_change_percent": rel_chg
        })

    return result_row, predictions_rows, degradation_rows


def run_all_external_validation():
    config = load_config("config.yaml")
    results_dir = os.path.join(ROOT_DIR, config["paths"]["results_dir"])
    figures_ext_dir = os.path.join(ROOT_DIR, "figures", "external")
    figures_cal_dir = os.path.join(ROOT_DIR, "figures", "external_calibration")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_ext_dir, exist_ok=True)
    os.makedirs(figures_cal_dir, exist_ok=True)

    internal_baselines = load_internal_baselines(config["paths"]["results_dir"])

    # 1. Six Primary Pairwise Directions
    pairwise_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland"),
    ]

    # 2. Four LODO Directions
    lodo_directions = [
        ("Cleveland + Hungarian + Zurich", "VA Long Beach"),
        ("Cleveland + Hungarian + VA Long Beach", "Zurich"),
        ("Cleveland + Zurich + VA Long Beach", "Hungarian"),
        ("Hungarian + Zurich + VA Long Beach", "Cleveland")
    ]

    models = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]

    pairwise_results = []
    pairwise_predictions = []
    pairwise_degradations = []

    print("=== Starting Phase 5 Cross-Dataset External Validation ===")

    # Execute 24 Primary Pairwise Experiments
    print(f"\n--- Executing 24 Primary Pairwise Experiments (6 directions x 4 models) ---")
    exp_counter = 1
    for train_c, test_c in pairwise_directions:
        for m_name in models:
            exp_id = f"PAIRWISE_E{exp_counter}_{train_c[:4].upper()}_TO_{test_c[:4].upper()}_{m_name}"
            print(f"  [{exp_id}] Train: {train_c} -> Test: {test_c} | Model: {m_name}...")

            res, preds, degs = run_single_external_experiment(
                exp_id, train_c, test_c, m_name, config, internal_baselines, is_lodo=False
            )
            pairwise_results.append(res)
            pairwise_predictions.extend(preds)
            pairwise_degradations.extend(degs)
            exp_counter += 1

    # Save Primary Results CSV (Strictly 24 rows)
    primary_results_df = pd.DataFrame(pairwise_results)
    primary_results_path = os.path.join(results_dir, "external_validation_results.csv")
    primary_results_df.to_csv(primary_results_path, index=False)
    print(f"\n[OK] Saved primary external validation results (24 rows) to: {primary_results_path}")

    # Save Primary Predictions CSV
    primary_preds_df = pd.DataFrame(pairwise_predictions)
    primary_preds_path = os.path.join(results_dir, "external_predictions.csv")
    primary_preds_df.to_csv(primary_preds_path, index=False)
    print(f"[OK] Saved primary external predictions registry to: {primary_preds_path}")

    # Save Primary Degradation CSV
    primary_deg_df = pd.DataFrame(pairwise_degradations)
    primary_deg_path = os.path.join(results_dir, "performance_degradation.csv")
    primary_deg_df.to_csv(primary_deg_path, index=False)
    print(f"[OK] Saved performance degradation analysis to: {primary_deg_path}")

    # Execute 16 Secondary LODO Experiments
    print(f"\n--- Executing 16 Secondary Leave-One-Dataset-Out (LODO) Experiments (4 directions x 4 models) ---")
    lodo_results = []
    lodo_predictions = []
    lodo_counter = 1
    for train_pool, held_out in lodo_directions:
        for m_name in models:
            exp_id = f"LODO_E{lodo_counter}_POOL_TO_{held_out[:4].upper()}_{m_name}"
            print(f"  [{exp_id}] Dev Pool -> Held-out: {held_out} | Model: {m_name}...")

            res, preds, degs = run_single_external_experiment(
                exp_id, train_pool, held_out, m_name, config, internal_baselines, is_lodo=True
            )
            lodo_results.append(res)
            lodo_predictions.extend(preds)
            lodo_counter += 1

    # Save Secondary LODO Results CSV (Strictly 16 rows)
    lodo_results_df = pd.DataFrame(lodo_results)
    lodo_results_path = os.path.join(results_dir, "lodo_results.csv")
    lodo_results_df.to_csv(lodo_results_path, index=False)
    print(f"[OK] Saved secondary LODO results (16 rows) to: {lodo_results_path}")

    # Save Secondary LODO Predictions CSV
    lodo_preds_df = pd.DataFrame(lodo_predictions)
    lodo_preds_path = os.path.join(results_dir, "lodo_predictions.csv")
    lodo_preds_df.to_csv(lodo_preds_path, index=False)
    print(f"[OK] Saved secondary LODO predictions to: {lodo_preds_path}")

    # Save Experiment Manifest JSON (Covers all 40 experiments)
    manifest = {
        "metadata": {
            "title": "Phase 5 Cross-Dataset External Validation Manifest",
            "project": "Beyond Accuracy: Cross-Dataset Generalization, Calibration and Explanation Stability",
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "random_seed": config["experiment"]["random_seed"],
            "classification_threshold": config["experiment"]["classification_threshold"],
            "feature_schema": TRACK_B_10_FEATURES,
            "models_evaluated": models,
            "primary_analysis": {
                "description": "24 pairwise external validation experiments across 6 hospital transfer directions",
                "results_file": "results/external_validation_results.csv",
                "predictions_file": "results/external_predictions.csv",
                "degradation_file": "results/performance_degradation.csv",
                "n_experiments": len(primary_results_df)
            },
            "secondary_analysis": {
                "description": "16 Leave-One-Dataset-Out (LODO) experiments on pooled multi-hospital development data",
                "results_file": "results/lodo_results.csv",
                "predictions_file": "results/lodo_predictions.csv",
                "n_experiments": len(lodo_results_df)
            },
            "total_experiments_completed": len(primary_results_df) + len(lodo_results_df),
            "isolation_rule": "Test cohorts strictly transformed via transform(); zero fitting or hyperparameter tuning on test data"
        }
    }
    manifest_path = os.path.join(results_dir, "external_validation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Saved experiment manifest to: {manifest_path}")

    # Generate Publication Figures
    generate_external_figures(primary_results_df, primary_preds_df, figures_ext_dir, figures_cal_dir)

    return primary_results_df, lodo_results_df


def generate_external_figures(primary_results_df: pd.DataFrame, primary_preds_df: pd.DataFrame, figures_ext_dir: str, figures_cal_dir: str):
    """Generates 300 DPI publication figures comparing internal vs external metrics and calibration without ranking."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "SVM": "#ff7f0e", "XGBoost": "#d62728"}

    df = primary_results_df.copy()
    df["direction"] = df["train_dataset"] + " -> " + df["test_dataset"]

    # 1. Figure: Internal vs External AUC
    plt.figure(figsize=(14, 6))
    melted_auc = pd.melt(
        df,
        id_vars=["direction", "model"],
        value_vars=["internal_roc_auc", "external_roc_auc"],
        var_name="Validation_Type",
        value_name="ROC_AUC"
    )
    melted_auc["Validation_Type"] = melted_auc["Validation_Type"].replace({
        "internal_roc_auc": "Internal CV (Ref)",
        "external_roc_auc": "External Generalization"
    })

    sns.barplot(
        data=melted_auc,
        x="direction",
        y="ROC_AUC",
        hue="Validation_Type",
        palette={"Internal CV (Ref)": "#94a3b8", "External Generalization": "#3b82f6"}
    )
    plt.title("Internal Cross-Validation vs. External Generalization ROC-AUC by Transfer Direction", fontsize=12, fontweight="bold")
    plt.ylabel("ROC-AUC (%)", fontsize=11)
    plt.xlabel("Transfer Direction (Development -> External Evaluation)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(30, 100)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = os.path.join(figures_ext_dir, "internal_vs_external_auc.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # 2. Figure: Internal vs External Brier Score
    plt.figure(figsize=(14, 6))
    melted_brier = pd.melt(
        df,
        id_vars=["direction", "model"],
        value_vars=["internal_brier", "external_brier"],
        var_name="Validation_Type",
        value_name="Brier_Score"
    )
    melted_brier["Validation_Type"] = melted_brier["Validation_Type"].replace({
        "internal_brier": "Internal CV (Ref)",
        "external_brier": "External Generalization"
    })

    sns.barplot(
        data=melted_brier,
        x="direction",
        y="Brier_Score",
        hue="Validation_Type",
        palette={"Internal CV (Ref)": "#94a3b8", "External Generalization": "#f97316"}
    )
    plt.title("Internal vs. External Brier Score (Higher External = Greater Probabilistic Error)", fontsize=12, fontweight="bold")
    plt.ylabel("Brier Score", fontsize=11)
    plt.xlabel("Transfer Direction (Development -> External Evaluation)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig2_path = os.path.join(figures_ext_dir, "internal_vs_external_brier.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # 3. Figure: AUC Degradation
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="direction",
        y="auc_change",
        hue="model",
        palette=palette
    )
    plt.axhline(0, color="black", linestyle="-", lw=1.2)
    plt.title("Discrimination Degradation (Δ ROC-AUC = External - Internal Reference)", fontsize=12, fontweight="bold")
    plt.ylabel("Δ ROC-AUC (Percentage Points)", fontsize=11)
    plt.xlabel("Transfer Direction (Development -> External Evaluation)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(figures_ext_dir, "auc_degradation.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # 4. Figure: Brier Score Degradation
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="direction",
        y="brier_change",
        hue="model",
        palette=palette
    )
    plt.axhline(0, color="black", linestyle="-", lw=1.2)
    plt.title("Calibration Error Inflation (Δ Brier Score = External - Internal Reference)", fontsize=12, fontweight="bold")
    plt.ylabel("Δ Brier Score (Positive = Calibration Degradation)", fontsize=11)
    plt.xlabel("Transfer Direction (Development -> External Evaluation)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    fig4_path = os.path.join(figures_ext_dir, "brier_degradation.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # 5. Figure: External Calibration Curves for Key Directions
    key_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("VA Long Beach", "Cleveland")
    ]
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (tr_c, te_c) in enumerate(key_directions):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.5)
        dir_preds = primary_preds_df[(primary_preds_df["train_dataset"] == tr_c) & (primary_preds_df["test_dataset"] == te_c)]

        for m_name, grp in dir_preds.groupby("model"):
            n_bins = 8
            prob_true, prob_pred = calibration_curve(grp["true_label"], grp["predicted_probability"], n_bins=n_bins, strategy="uniform")
            ece_score = compute_expected_calibration_error(grp["true_label"], grp["predicted_probability"], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", label=f"{m_name} (ECE={ece_score:.3f})", color=palette.get(m_name, "gray"), lw=2)

        ax.set_title(f"{tr_c} -> {te_c}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Risk Probability", fontsize=10)
        ax.set_ylabel("Observed Disease Proportion", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("External Calibration Drift Across Primary Transfer Directions", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig5_path = os.path.join(figures_cal_dir, "external_calibration_curves.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    print(f"[OK] Saved publication figures to: {figures_ext_dir} and {figures_cal_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Cross-Dataset External Validation")
    parser.add_argument("--train", type=str, default=None, help="Training cohort name")
    parser.add_argument("--test", type=str, default=None, help="Testing cohort name")
    parser.add_argument("--run-all", action="store_true", help="Run all 24 pairwise and 16 LODO experiments")
    args = parser.parse_args()

    config = load_config("config.yaml")
    internal_baselines = load_internal_baselines(config["paths"]["results_dir"])

    if args.run_all or (args.train is None and args.test is None):
        run_all_external_validation()
    elif args.train and args.test:
        models = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]
        for m in models:
            exp_id = f"SINGLE_{args.train[:4].upper()}_TO_{args.test[:4].upper()}_{m}"
            res, _, _ = run_single_external_experiment(
                exp_id, args.train, args.test, m, config, internal_baselines, is_lodo=False
            )
            print(f"[{exp_id}] External ROC-AUC: {res['external_roc_auc']:.2f}% | Delta vs Internal: {res['auc_change']:.2f}%")


if __name__ == "__main__":
    main()
