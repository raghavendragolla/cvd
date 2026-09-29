"""
Phase 4: Internal Validation Baseline Experiment Runner.
Executes 5-fold Stratified Cross-Validation on:
  - Cleveland Track A (13 features)
  - Cleveland Track B (10 features)
  - Hungarian Track B (10 features)
  - Zurich Track B (10 features)
  - VA Long Beach Track B (10 features)
across 4 clinical ML models: Logistic Regression, Random Forest, SVM, and XGBoost.
Guarantees strict leak-free fitting, out-of-fold predictions, calibration metrics,
summary statistics with 95% CIs, and publication-quality figures.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_A_FEATURES,
    TRACK_B_10_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from src.evaluate import (
    evaluate_classifier,
    compute_expected_calibration_error,
    compute_calibration_intercept_slope
)


def load_config(config_path: str = "config.yaml") -> dict:
    # Try importing PyYAML if installed in environment
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback to mirror config.json for zero-dependency standard library execution
        if not os.path.isabs(config_path):
            config_path = os.path.join(ROOT_DIR, config_path)
        json_path = config_path.replace(".yaml", ".json").replace(".yml", ".json")
        if not os.path.exists(json_path):
            json_path = os.path.join(ROOT_DIR, "config.json")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)


def build_model_pipeline(model_name: str, config: dict):
    """Instantiates model pipeline with scaling if required by architecture."""
    seed = config["experiment"]["random_seed"]

    if model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed))
        ])
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=seed
        )
    elif model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=seed))
        ])
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            eval_metric="logloss", random_state=seed
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_internal_validation():
    config = load_config(os.path.join(ROOT_DIR, "config.yaml"))
    seed = config["experiment"]["random_seed"]
    k_folds = config["experiment"]["cv_folds"]
    threshold = config["experiment"]["classification_threshold"]

    results_dir = os.path.join(ROOT_DIR, config["paths"]["results_dir"])
    figures_dir = os.path.join(ROOT_DIR, config["paths"]["figures_dir"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Evaluation Configurations: (Cohort Name, Track Code, Features)
    eval_tasks = [
        ("Cleveland", "Track_A", TRACK_A_FEATURES),
        ("Cleveland", "Track_B", TRACK_B_10_FEATURES),
        ("Hungarian", "Track_B", TRACK_B_10_FEATURES),
        ("Zurich", "Track_B", TRACK_B_10_FEATURES),
        ("VA Long Beach", "Track_B", TRACK_B_10_FEATURES),
    ]

    model_names = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]

    oof_records = []
    fold_results = []

    print(f"=== Starting Phase 4 Internal Validation ({k_folds}-Fold Stratified CV) ===")

    for cohort_name, track_name, feature_list in eval_tasks:
        print(f"\n--- Cohort: {cohort_name} | Track: {track_name} ({len(feature_list)} features) ---")
        track_key = "A" if track_name == "Track_A" else "B10"

        # Load raw harmonized cohort
        df = load_harmonized_cohort(
            cohort_name,
            track=track_key,
            data_dir=os.path.join(ROOT_DIR, config["paths"]["raw_data_dir"]),
            drop_duplicates=True,
            mask_zero_sentinels=True
        )

        X = df[feature_list].copy()
        y = df["target"].to_numpy()
        n_samples = len(df)
        pos_rate = round(float(np.mean(y)), 4)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for model_name in model_names:
            print(f"  > Training Model: {model_name}...")
            oof_proba = np.zeros(n_samples, dtype=float)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_raw = X.iloc[train_idx].copy()
                y_train = y[train_idx]
                X_val_raw = X.iloc[val_idx].copy()
                y_val = y[val_idx]

                # Fit preprocessor strictly on train fold
                preprocessor = HarmonizedClinicalPreprocessor(
                    imputer_strategy="knn",
                    n_neighbors=5,
                    scale_features=False,
                    clamp_negative_oldpeak=False
                )
                X_train_proc = preprocessor.fit_transform(X_train_raw)
                X_val_proc = preprocessor.transform(X_val_raw)

                # Fit model pipeline strictly on train fold
                model = build_model_pipeline(model_name, config)
                model.fit(X_train_proc, y_train)

                # Predict probabilities on out-of-fold validation set
                if hasattr(model, "predict_proba"):
                    y_val_proba = model.predict_proba(X_val_proc)[:, 1]
                else:
                    y_val_proba = model.decision_function(X_val_proc)
                    y_val_proba = 1 / (1 + np.exp(-y_val_proba))

                oof_proba[val_idx] = y_val_proba
                y_val_pred = (y_val_proba >= threshold).astype(int)

                # Compute fold metrics
                fold_metrics = evaluate_classifier(y_val, y_val_pred, y_val_proba)
                fold_metrics.update({
                    "cohort": cohort_name,
                    "track": track_name,
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "n_val": len(val_idx),
                    "random_seed": seed
                })
                fold_results.append(fold_metrics)

                # Record OOF predictions per sample
                for local_i, global_i in enumerate(val_idx):
                    oof_records.append({
                        "cohort": cohort_name,
                        "track": track_name,
                        "model": model_name,
                        "sample_idx": int(global_i),
                        "true_label": int(y[global_i]),
                        "predicted_probability": round(float(y_val_proba[local_i]), 6),
                        "fold": fold_idx + 1
                    })

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_records)
    oof_csv_path = os.path.join(ROOT_DIR, config["paths"]["oof_predictions_file"])
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"\n[OK] Saved out-of-fold predictions to: {oof_csv_path}")

    # Save fold-level results
    results_df = pd.DataFrame(fold_results)
    results_csv_path = os.path.join(ROOT_DIR, config["paths"]["results_file"])
    results_df.to_csv(results_csv_path, index=False)
    print(f"[OK] Saved fold-level results to: {results_csv_path}")

    # Compute Summary Statistics (Mean, SD, 95% CI)
    metric_cols = [
        "roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity",
        "precision", "f1_score", "mcc", "brier_score", "ece",
        "calibration_intercept", "calibration_slope"
    ]

    summary_rows = []
    grouped = results_df.groupby(["cohort", "track", "model"])

    for (cohort_val, track_val, model_val), grp in grouped:
        for metric in metric_cols:
            vals = grp[metric].values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            # 95% Confidence Interval: t-multiplier for df=4 is 2.776
            ci_margin = 2.776 * (std_val / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

            summary_rows.append({
                "cohort": cohort_val,
                "track": track_val,
                "model": model_val,
                "metric": metric,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "ci_lower": round(mean_val - ci_margin, 4),
                "ci_upper": round(mean_val + ci_margin, 4)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(ROOT_DIR, config["paths"]["summary_file"])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[OK] Saved summary results (with 95% CIs) to: {summary_csv_path}")

    # Generate Publication Figures
    generate_validation_figures(oof_df, figures_dir)
    return results_df, summary_df, oof_df


def generate_validation_figures(oof_df: pd.DataFrame, figures_dir: str):
    """Generates ROC curves, PR curves, calibration curves, and confusion matrices."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "SVM": "#ff7f0e", "XGBoost": "#d62728"}

    # Figure 1: ROC Curve for Cleveland Track A
    plt.figure(figsize=(7, 6))
    clev_a = oof_df[(oof_df["cohort"] == "Cleveland") & (oof_df["track"] == "Track_A")]
    for model_name, grp in clev_a.groupby("model"):
        fpr, tpr, _ = roc_curve(grp["true_label"], grp["predicted_probability"])
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(grp["true_label"], grp["predicted_probability"])
        plt.plot(fpr, tpr, label=f"{model_name} (OOF AUC = {auc_val:.3f})", color=palette.get(model_name, "gray"), lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
    plt.ylabel("Sensitivity (True Positive Rate)", fontsize=11)
    plt.title("Internal Validation ROC: Cleveland Track A (13 Features)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "internal_roc_cleveland_track_a.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Multi-Cohort ROC Curves for Track B
    cohorts_track_b = ["Cleveland", "Hungarian", "Zurich", "VA Long Beach"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, cohort in enumerate(cohorts_track_b):
        ax = axes[idx]
        c_df = oof_df[(oof_df["cohort"] == cohort) & (oof_df["track"] == "Track_B")]
        for model_name, grp in c_df.groupby("model"):
            fpr, tpr, _ = roc_curve(grp["true_label"], grp["predicted_probability"])
            from sklearn.metrics import roc_auc_score
            auc_val = roc_auc_score(grp["true_label"], grp["predicted_probability"])
            ax.plot(fpr, tpr, label=f"{model_name} ({auc_val:.3f})", color=palette.get(model_name, "gray"), lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax.set_title(f"{cohort} Cohort (Track B: 10 Features)", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Internal 5-Fold CV ROC Curves: Track B Across International Cohorts", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "internal_roc_track_b.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Precision-Recall Curves Across Cohorts (Track B)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, cohort in enumerate(cohorts_track_b):
        ax = axes[idx]
        c_df = oof_df[(oof_df["cohort"] == cohort) & (oof_df["track"] == "Track_B")]
        from sklearn.metrics import average_precision_score
        base_rate = float(np.mean(c_df[c_df["model"] == "LogisticRegression"]["true_label"]))
        for model_name, grp in c_df.groupby("model"):
            prec, rec, _ = precision_recall_curve(grp["true_label"], grp["predicted_probability"])
            ap_val = average_precision_score(grp["true_label"], grp["predicted_probability"])
            ax.plot(rec, prec, label=f"{model_name} (PR-AUC = {ap_val:.3f})", color=palette.get(model_name, "gray"), lw=2)
        ax.axhline(y=base_rate, color="k", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence ({base_rate:.2f})")
        ax.set_title(f"{cohort} Precision-Recall (Track B)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall / Sensitivity", fontsize=10)
        ax.set_ylabel("Precision / PPV", fontsize=10)
        ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Internal 5-Fold CV Precision-Recall Curves: Track B", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "internal_pr_curves.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Calibration Curves Across Cohorts (Track B)
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, cohort in enumerate(cohorts_track_b):
        ax = axes[idx]
        c_df = oof_df[(oof_df["cohort"] == cohort) & (oof_df["track"] == "Track_B")]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", lw=1.5)
        for model_name, grp in c_df.groupby("model"):
            n_bins = 5 if cohort == "Zurich" else 8
            prob_true, prob_pred = calibration_curve(grp["true_label"], grp["predicted_probability"], n_bins=n_bins, strategy="uniform")
            ece_score = compute_expected_calibration_error(grp["true_label"], grp["predicted_probability"], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", label=f"{model_name} (ECE={ece_score:.3f})", color=palette.get(model_name, "gray"), lw=2)
        ax.set_title(f"{cohort} Calibration Curves (Track B)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Proportion", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Out-of-Fold Probability Calibration Across 4 International Cohorts", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "internal_calibration_curves.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Figure 5: Confusion Matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for idx, model_name in enumerate(["LogisticRegression", "RandomForest", "SVM", "XGBoost"]):
        ax = axes[idx]
        m_df = oof_df[(oof_df["cohort"] == "Cleveland") & (oof_df["track"] == "Track_A") & (oof_df["model"] == model_name)]
        preds = (m_df["predicted_probability"] >= 0.50).astype(int)
        cm = confusion_matrix(m_df["true_label"], preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 13})
        ax.set_title(f"{model_name}\n(Cleveland Track A, thr=0.50)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Condition", fontsize=10)
        ax.set_ylabel("True Condition", fontsize=10)
        ax.set_xticklabels(["Healthy", "CAD"])
        ax.set_yticklabels(["Healthy", "CAD"])

    plt.suptitle("Confusion Matrices (Cleveland Track A @ Fixed Threshold 0.50)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "internal_confusion_matrices.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    print(f"[OK] Saved publication-quality figures to: {figures_dir}")


if __name__ == "__main__":
    run_internal_validation()
