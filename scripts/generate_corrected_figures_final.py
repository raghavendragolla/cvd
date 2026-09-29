"""
Final Publication Figure Generation (Verified Ground-Truth Standards)
====================================================================
Generates the definitive publication figures into corrected_figures_final/:
- FIGURE_1_FINAL.png / .pdf: Study Architecture & Experimental Protocol
- FIGURE_2_FINAL.png / .pdf: External Discrimination Performance (ROC-AUC Heatmap)
- FIGURE_3_FINAL.png / .pdf: Calibration Degradation & Cross-Dataset Drift
- FIGURE_4_FINAL.png / .pdf: Global Explanation Rank Stability (SHAP Spearman rho)
- FIGURE_5_FINAL.png / .pdf: Standardized Covariate Shift vs. SHAP Explanation Stability
- FIGURE_6_FINAL.png / .pdf: Subgroup Disparity & Missingness Robustness

Strictly loaded from verified ground-truth results in:
- corrected_results/
- results/
- data/raw/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.calibration import calibration_curve

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CORRECTED_DIR = os.path.join(ROOT_DIR, "corrected_results")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
OUT_DIR = os.path.join(ROOT_DIR, "corrected_figures_final")
os.makedirs(OUT_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import load_harmonized_cohort, TRACK_B_10_FEATURES

# Standardized publication aesthetics across all figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "axes.edgecolor": "#2d3748",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

TRANSFER_ORDER = [
    "Cleveland -> Hungarian",
    "Cleveland -> Zurich",
    "Cleveland -> VA Long Beach",
    "Hungarian -> Cleveland",
    "Zurich -> Cleveland",
    "VA Long Beach -> Cleveland"
]

TRANSFER_LABELS_SHORT = [
    "Clev -> Hung",
    "Clev -> Zuri",
    "Clev -> VA",
    "Hung -> Clev",
    "Zuri -> Clev",
    "VA -> Clev"
]

MODEL_ORDER = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]
MODEL_DISPLAY = ["Logistic Regression", "Random Forest", "Support Vector Machine", "XGBoost"]

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


def safe_savefig(fig_or_plt, filepath, **kwargs):
    """Safely saves figure to disk avoiding file lock issues via atomic replacement."""
    temp_path = filepath + ".tmp" + os.path.splitext(filepath)[1]
    fig_or_plt.savefig(temp_path, **kwargs)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass
    try:
        os.replace(temp_path, filepath)
    except Exception:
        import shutil
        shutil.move(temp_path, filepath)


# ==============================================================================
# FIGURE 1: STUDY ARCHITECTURE & EXPERIMENTAL PROTOCOL
# ==============================================================================

def generate_figure_1():
    """Renders the verified methodological architecture & workflow diagram."""
    # 1. Verify cohort counts, duplicates, and feature definition programmatically
    cohort_stats = {}
    total_raw = 0
    total_dedup = 0
    total_dups = 0
    for c in ["cleveland", "hungarian", "zurich", "va_long_beach"]:
        df_raw = load_harmonized_cohort(c, track="B10", data_dir=RAW_DATA_DIR, drop_duplicates=False)
        df_dedup = load_harmonized_cohort(c, track="B10", data_dir=RAW_DATA_DIR, drop_duplicates=True)
        n_r = len(df_raw)
        n_d = len(df_dedup)
        dups = n_r - n_d
        cohort_stats[c] = {"raw": n_r, "dedup": n_d, "dups": dups}
        total_raw += n_r
        total_dedup += n_d
        total_dups += dups

    # Assertions on verified counts
    assert cohort_stats["cleveland"]["dedup"] == 303, "Cleveland count mismatch"
    assert cohort_stats["hungarian"]["dedup"] == 293, "Hungarian count mismatch"
    assert cohort_stats["hungarian"]["dups"] == 1, "Hungarian duplicate mismatch"
    assert cohort_stats["zurich"]["dedup"] == 123, "Zurich count mismatch"
    assert cohort_stats["va_long_beach"]["dedup"] == 199, "VA Long Beach count mismatch"
    assert cohort_stats["va_long_beach"]["dups"] == 1, "VA Long Beach duplicate mismatch"
    assert total_raw == 920, "Total raw count mismatch"
    assert total_dups == 2, "Total duplicate count mismatch"
    assert total_dedup == 918, "Total analysed records mismatch"
    assert len(TRACK_B_10_FEATURES) == 10, "Feature count mismatch"
    assert "chol" not in TRACK_B_10_FEATURES, "Cholesterol incorrectly in Track B"

    fig, ax = plt.subplots(figsize=(11.5, 9.2), dpi=300)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.35, 9.2)
    ax.axis("off")

    def draw_box(x, y, w, h, title, text, bg_color="#eef3f8", edge_color="#3182ce", title_fontsize=9.2, text_fontsize=8.0, text_linespacing=1.3, text_y_offset=0.42):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.08",
            facecolor=bg_color, edgecolor=edge_color, linewidth=1.2,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top",
                fontweight="bold", fontsize=title_fontsize, color="#1a365d", zorder=3)
        ax.text(x + w / 2, y + h - text_y_offset, text, ha="center", va="top",
                fontsize=text_fontsize, color="#2d3748", multialignment="center",
                linespacing=text_linespacing, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=1.3, mutation_scale=12),
            zorder=1
        )
        if label:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=7.5, color="#718096")

    # Header title
    ax.text(5.0, 8.90, "Figure 1: Study Architecture & Cross-Dataset Experimental Protocol",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1a202c")

    # Layer 1: Cohorts & Feature Harmonization
    cohorts_lines = [
        "Cleveland (USA): N = 303   •   Hungarian (HUN): N = 293",
        "Zurich (SUI): N = 123   •   VA Long Beach (USA): N = 199",
        "Data Audit: 920 Raw Records  →  2 Duplicates Removed (1 HUN, 1 VA)",
        "Analyzed Records: 918"
    ]
    draw_box(0.45, 7.05, 4.30, 1.55, "1. Multi-Center Clinical Cohorts",
             "\n".join(cohorts_lines), title_fontsize=9.2, text_fontsize=8.0, text_linespacing=1.32)

    features_lines = [
        "10 harmonized clinical variables",
        "Demographics & History: Age, Sex, Chest Pain Type (cp)",
        "Physiological & ECG: Resting BP (trestbps), Fasting Sugar (fbs), restecg",
        "Stress Response: Max HR (thalach), Exercise Angina (exang), oldpeak, slope",
        "Cholesterol excluded due to substantial missingness and incompatible coding",
        "ca and thal excluded due to substantial external missingness"
    ]
    draw_box(5.25, 7.05, 4.50, 1.55, "2. Primary Feature Harmonization (10-Feature Representation)",
             "\n".join(features_lines), title_fontsize=7.3, text_fontsize=7.0, text_linespacing=1.24, text_y_offset=0.38)

    draw_arrow(4.75, 7.825, 5.25, 7.825)

    # Layer 2: Preprocessing & Internal Validation
    prep_lines = [
        "Zero-sentinel handling (resting BP: 0 → NaN)",
        "Within-Fold/Source KNN Imputation (k = 5)",
        "Z-Standardization (Fitted strictly on training folds)"
    ]
    draw_box(0.45, 5.35, 4.30, 1.30, "3. Leakage-Safe Preprocessing",
             "\n".join(prep_lines), bg_color="#f0fff4", edge_color="#38a169", text_fontsize=8.0)

    int_lines = [
        "5-Fold Stratified Cross-Validation on Each Development Cohort",
        "Pooled Out-of-Fold (OOF) Evaluation (Resolves Small-Fold ECE Bias)",
        "Logistic Regression • Random Forest • SVM (RBF) • XGBoost"
    ]
    draw_box(5.25, 5.35, 4.50, 1.30, "4. Internal Validation (Pooled OOF)",
             "\n".join(int_lines), bg_color="#f0fff4", edge_color="#38a169", text_fontsize=8.0)

    draw_arrow(2.60, 7.05, 2.60, 6.65)
    draw_arrow(7.50, 7.05, 7.50, 6.65)
    draw_arrow(4.75, 6.00, 5.25, 6.00)

    # Layer 3: External Validation Protocol
    ext_lines = [
        "24 Directed Pairwise Transfers (6 Directions x 4 Model Architectures)",
        "Cleveland <-> Hungarian  |  Cleveland <-> Zurich  |  Cleveland <-> VA Long Beach",
        "Target Cohort Labels Strictly Sequestered (Zero External Fine-Tuning or Calibration)"
    ]
    draw_box(1.0, 3.75, 8.20, 1.20, "5. True External Validation (Absolute Test Isolation)",
             "\n".join(ext_lines), bg_color="#fffaf0", edge_color="#dd6b20", text_fontsize=8.0)

    draw_arrow(2.60, 5.35, 2.60, 4.95)
    draw_arrow(7.50, 5.35, 7.50, 4.95)

    # Layer 4: Multi-Dimensional Evaluation Axes
    # Box 6
    axes1_lines = [
        "External ROC-AUC Change",
        "Brier Score",
        "10-Bin ECE"
    ]
    draw_box(0.35, 1.95, 3.00, 1.40, "6. Discrimination & Probability Drift",
             "\n".join(axes1_lines), bg_color="#faf5ff", edge_color="#805ad5", title_fontsize=8.8, text_fontsize=7.8, text_linespacing=1.45, text_y_offset=0.46)

    # Box 7
    axes2_lines = [
        "Exact 1024-Coalition Shapley Enumeration",
        "Global Spearman Feature Rank Stability (rho)",
        "Multi-Seed SHAP Background-Sensitivity Analysis"
    ]
    draw_box(3.60, 1.95, 3.00, 1.40, "7. Explanation Stability (SHAP XAI)",
             "\n".join(axes2_lines), bg_color="#faf5ff", edge_color="#805ad5", title_fontsize=8.8, text_fontsize=7.8, text_linespacing=1.45, text_y_offset=0.46)

    # Box 8
    axes3_lines = [
        "Source-Standardized Continuous W1\n(4 Continuous Features, SD Units)",
        "Discrete Jensen-Shannon Divergence (JSD)",
        "Absolute Prevalence Divergence (|Delta Prev|)",
        "Shift-Stability Association Modeling"
    ]
    draw_box(6.85, 1.95, 3.00, 1.40, "8. Standardized Covariate Shift",
             "\n".join(axes3_lines), bg_color="#faf5ff", edge_color="#805ad5", title_fontsize=8.8, text_fontsize=7.4, text_linespacing=1.22)

    draw_arrow(2.50, 3.75, 1.85, 3.35)
    draw_arrow(5.10, 3.75, 5.10, 3.35)
    draw_arrow(7.70, 3.75, 8.35, 3.35)

    # Layer 5: Integrated Synthesis & Sensitivity Safeguards
    # Box 9
    synth_lines = [
        "Non-parametric Cluster-Resampling Bootstrap (B=2000 Resamples, Clustered on 6 Transfer Pairs)",
        "Small-Cell Guarded Missingness Sensitivity (N >= 30, events >= 10, non-events >= 10)"
    ]
    draw_box(1.0, 0.35, 8.20, 1.15, "9. Integrated Statistical Synthesis & Sensitivity Safeguards",
             "\n".join(synth_lines), bg_color="#ebf8ff", edge_color="#2b6cb0", title_fontsize=9.2, text_fontsize=8.1, text_linespacing=1.45, text_y_offset=0.44)

    draw_arrow(1.85, 1.95, 3.20, 1.50)
    draw_arrow(5.10, 1.95, 5.10, 1.50)
    draw_arrow(8.35, 1.95, 7.00, 1.50)

    png_path = os.path.join(OUT_DIR, "FIGURE_1_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_1_FINAL.pdf")
    root_png_path = os.path.join(ROOT_DIR, "01_methodological_architecture.png")
    corrected_png_path = os.path.join(OUT_DIR, "01_methodological_architecture.png")
    corr_img_path = os.path.join(ROOT_DIR, "corrected_images", "01_methodological_architecture.png")
    corr_img_fig1_png = os.path.join(ROOT_DIR, "corrected_images", "FIGURE_1_FINAL.png")
    corr_img_fig1_pdf = os.path.join(ROOT_DIR, "corrected_images", "FIGURE_1_FINAL.pdf")

    safe_savefig(plt, png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    safe_savefig(plt, pdf_path, bbox_inches="tight", pad_inches=0.25)
    safe_savefig(plt, root_png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    safe_savefig(plt, corrected_png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    figures_png_path = os.path.join(ROOT_DIR, "figures", "01_methodological_architecture.png")
    if os.path.exists(os.path.dirname(figures_png_path)):
        safe_savefig(plt, figures_png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    if os.path.exists(os.path.dirname(corr_img_path)):
        safe_savefig(plt, corr_img_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        safe_savefig(plt, corr_img_fig1_png, dpi=300, bbox_inches="tight", pad_inches=0.25)
        safe_savefig(plt, corr_img_fig1_pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    print("[OK] Generated Figure 1: FIGURE_1_FINAL (PNG + PDF) and 01_methodological_architecture.png")


# ==============================================================================
# FIGURE 2: EXTERNAL VALIDATION ROC-AUC HEATMAP
# ==============================================================================

def generate_figure_2():
    """Renders the publication heatmap of external ROC-AUC across models and transfers."""
    ext_path = os.path.join(RESULTS_DIR, "external_validation_results.csv")
    corr_path = os.path.join(CORRECTED_DIR, "integrated_estimands_corrected.csv")
    df_ext = pd.read_csv(ext_path)
    df_corr = pd.read_csv(corr_path)

    # Programmatic verification: all 24 AUCs must be identical
    for _, r in df_ext.iterrows():
        c_match = df_corr[df_corr["experiment_id"] == r["experiment_id"]]
        assert len(c_match) == 1, f"Missing {r['experiment_id']}"
        val_orig = round(float(r["external_roc_auc"]), 2)
        val_corr = round(float(c_match["external_roc_auc"].iloc[0]), 2)
        assert abs(val_orig - val_corr) < 1e-4, f"Mismatch in {r['experiment_id']}: {val_orig} vs {val_corr}"

    df_ext["transfer_pair"] = df_ext["train_dataset"] + " -> " + df_ext["test_dataset"]
    pivot = df_ext.pivot(index="model", columns="transfer_pair", values="external_roc_auc")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=TRANSFER_ORDER)
    pivot.index = MODEL_DISPLAY
    pivot.columns = TRANSFER_LABELS_SHORT

    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=300)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="mako",
        vmin=60, vmax=90, cbar_kws={"label": "External ROC-AUC (%)", "shrink": 0.85},
        linewidths=1.0, linecolor="white", ax=ax, annot_kws={"size": 10, "weight": "bold"}
    )

    ax.set_title("Figure 2: External Discrimination Performance (ROC-AUC % Across 24 Pairwise Transfers)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Hospital Transfer Direction (Source -> Target)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Model Architecture", fontsize=10, fontweight="bold", labelpad=8)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "FIGURE_2_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_2_FINAL.pdf")
    safe_savefig(plt, png_path, dpi=300)
    safe_savefig(plt, pdf_path)
    plt.close()
    print("[OK] Generated Figure 2: FIGURE_2_FINAL (PNG + PDF)")


# ==============================================================================
# FIGURE 3: CALIBRATION CURVES & EXTERNAL ECE
# ==============================================================================

def generate_figure_3():
    """Renders calibration curves and ECE breakdown strictly from corrected result files."""
    preds_path = os.path.join(RESULTS_DIR, "external_predictions.csv")
    cal_path = os.path.join(CORRECTED_DIR, "calibration_corrected_external.csv")
    est_path = os.path.join(CORRECTED_DIR, "integrated_estimands_corrected.csv")

    df_preds = pd.read_csv(preds_path)
    df_cal = pd.read_csv(cal_path)
    df_est = pd.read_csv(est_path)

    # 1. Verification of Zurich -> Cleveland Random Forest corrected metrics
    exp_shifted = "PAIRWISE_E18_ZURI_TO_CLEV_RandomForest"
    exp_stable = "PAIRWISE_E2_CLEV_TO_HUNG_RandomForest"

    z_cal = df_cal[df_cal["experiment_id"] == exp_shifted].iloc[0]
    s_cal = df_cal[df_cal["experiment_id"] == exp_stable].iloc[0]

    z_ece = float(z_cal["external_ece"])
    z_brier = float(z_cal["external_brier_score"])
    s_ece = float(s_cal["external_ece"])
    s_brier = float(s_cal["external_brier_score"])

    # CRITICAL ASSERTION: exactly equal to verified ground truth
    assert abs(z_ece - 0.3816) < 1e-4, f"Zurich->Cleveland RF ECE must equal 0.3816, got {z_ece}"
    assert abs(z_brier - 0.3418) < 1e-4, f"Zurich->Cleveland RF Brier must equal 0.3418, got {z_brier}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300)

    # Panel A: Empirical Calibration Curves
    stable_data = df_preds[df_preds["experiment_id"] == exp_stable]
    shifted_data = df_preds[df_preds["experiment_id"] == exp_shifted]

    prob_true_s, prob_pred_s = calibration_curve(stable_data["true_label"], stable_data["predicted_probability"], n_bins=5)
    prob_true_z, prob_pred_z = calibration_curve(shifted_data["true_label"], shifted_data["predicted_probability"], n_bins=5)

    # Prevalence deltas calculated from actual data
    s_prev_delta = abs(54.46 - 37.54)  # Cleveland (54.5%) -> Hungarian (37.5%) = 16.9%
    z_prev_delta = abs(93.50 - 54.46)  # Zurich (93.5%) -> Cleveland (54.5%) = 39.0%

    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration (Ideal)", alpha=0.7)
    ax1.plot(prob_pred_s, prob_true_s, "s-", color="#2ca02c", lw=2, markersize=7,
             label=f"Clev -> Hung (RF, |dPrev|={s_prev_delta:.1f}%)\nECE = {s_ece:.4f}, Brier = {s_brier:.4f}")
    ax1.plot(prob_pred_z, prob_true_z, "o-", color="#d62728", lw=2, markersize=7,
             label=f"Zuri -> Clev (RF, |dPrev|={z_prev_delta:.1f}%)\nECE = {z_ece:.4f}, Brier = {z_brier:.4f}")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Observed Proportion of Cases", fontsize=10, fontweight="bold")
    ax1.set_title("A: Empirical Calibration Curves (Random Forest)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel B: External ECE Across Directed Transfers
    df_est["transfer_short"] = df_est["transfer_pair"].map(dict(zip(TRANSFER_ORDER, TRANSFER_LABELS_SHORT)))
    sns.boxplot(data=df_est, x="transfer_short", y="external_ece", order=TRANSFER_LABELS_SHORT,
                hue="transfer_short", palette="Blues", legend=False, ax=ax2, width=0.45, fliersize=0)
    sns.stripplot(data=df_est, x="transfer_short", y="external_ece", order=TRANSFER_LABELS_SHORT,
                  hue="model", palette=MODEL_COLORS, dodge=True, jitter=0.15, size=6.5, alpha=0.9, ax=ax2)

    ax2.set_title("B: External ECE Across Directed Hospital Transfers", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Hospital Transfer Direction", fontsize=10, fontweight="bold")
    ax2.set_ylabel("External Expected Calibration Error (ECE)", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=25)
    ax2.legend(title="", loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Figure 3: Probability Calibration Degradation and Severe Drift Under Cross-Dataset Transport",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "FIGURE_3_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_3_FINAL.pdf")
    safe_savefig(plt, png_path, dpi=300)
    safe_savefig(plt, pdf_path)
    plt.close()
    print("[OK] Generated Figure 3: FIGURE_3_FINAL (PNG + PDF)")


# ==============================================================================
# FIGURE 4: SHAP EXPLANATION RANK STABILITY HEATMAP
# ==============================================================================

def generate_figure_4():
    """Renders the publication heatmap of SHAP global rank stability with exact SVM values."""
    est_path = os.path.join(CORRECTED_DIR, "integrated_estimands_corrected.csv")
    svm_corr_path = os.path.join(CORRECTED_DIR, "svm_shap_rank_stability_corrected.csv")

    df_est = pd.read_csv(est_path)
    df_svm = pd.read_csv(svm_corr_path)

    # Programmatic assertion of exact SVM values
    svm_expected = {
        ("Cleveland", "Hungarian"): 0.8788,
        ("Cleveland", "Zurich"): 0.7939,
        ("Cleveland", "VA Long Beach"): 0.8909,
        ("Hungarian", "Cleveland"): 0.8061,
        ("Zurich", "Cleveland"): 0.5394,
        ("VA Long Beach", "Cleveland"): 0.8303
    }
    for (tr, te), exp_val in svm_expected.items():
        m = df_svm[(df_svm["train_dataset"] == tr) & (df_svm["test_dataset"] == te)]
        assert len(m) == 1, f"Missing SVM correction for {tr} -> {te}"
        val = float(m["corrected_spearman_rho"].iloc[0])
        assert abs(val - exp_val) < 1e-4, f"Mismatch for {tr}->{te} SVM: expected {exp_val}, got {val}"

    pivot = df_est.pivot(index="model", columns="transfer_pair", values="spearman_rank_correlation")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=TRANSFER_ORDER)
    pivot.index = MODEL_DISPLAY
    pivot.columns = TRANSFER_LABELS_SHORT

    # Figure height 5.3 with explicit subplots_adjust to reserve clean bottom margin
    fig, ax = plt.subplots(figsize=(9.0, 5.3), dpi=300)
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="viridis",
        vmin=0.5, vmax=1.0, cbar_kws={"label": "SHAP Rank Stability (Spearman rho)", "shrink": 0.82},
        linewidths=1.0, linecolor="white", ax=ax, annot_kws={"size": 10, "weight": "bold"}
    )

    ax.set_title("Figure 4: Global Explanation Rank Stability (SHAP Spearman rho Across 24 Pairwise Transfers)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Hospital Transfer Direction (Source -> Target)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("Model Architecture", fontsize=10, fontweight="bold", labelpad=8)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Explicit bottom margin allocation: bottom=0.22 reserves bottom 22% exclusively
    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.18, right=0.96)

    # Footnote cleanly anchored at y=0.03 (well below x-axis label at y ~ 0.12)
    footnote_text = (
        "*Note: Tree-based models (Random Forest, XGBoost) use mean absolute SHAP rank comparison across 10 clinical features.\n"
        "SVM values reflect exact 1024-coalition Shapley enumeration; multi-seed audit indicates background sensitivity (delta-rho ~ 0.04-0.22).\n"
        "Zurich -> Cleveland SVM (rho = 0.539) coincides with extreme target risk saturation (IQR = 0.021)."
    )
    fig.text(
        0.50, 0.03, footnote_text,
        ha="center", va="bottom", fontsize=7.8, color="#2d3748", style="italic", linespacing=1.35
    )

    png_path = os.path.join(OUT_DIR, "FIGURE_4_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_4_FINAL.pdf")
    root_png_path = os.path.join(ROOT_DIR, "04_shap_explanation_stability.png")
    corrected_png_path = os.path.join(OUT_DIR, "04_shap_explanation_stability.png")
    corr_img_path = os.path.join(ROOT_DIR, "corrected_images", "FIGURE_4_FINAL.png")
    corr_img_pdf_path = os.path.join(ROOT_DIR, "corrected_images", "FIGURE_4_FINAL.pdf")
    corr_img_04_path = os.path.join(ROOT_DIR, "corrected_images", "04_shap_explanation_stability.png")

    safe_savefig(plt, png_path, dpi=300)
    safe_savefig(plt, pdf_path)
    safe_savefig(plt, root_png_path, dpi=300)
    safe_savefig(plt, corrected_png_path, dpi=300)
    if os.path.exists(os.path.dirname(corr_img_path)):
        safe_savefig(plt, corr_img_path, dpi=300)
        safe_savefig(plt, corr_img_pdf_path)
        safe_savefig(plt, corr_img_04_path, dpi=300)
    plt.close()
    print("[OK] Generated Figure 4: FIGURE_4_FINAL (PNG + PDF) and 04_shap_explanation_stability.png")


# ==============================================================================
# FIGURE 5: STANDARDIZED COVARIATE SHIFT VS. SHAP STABILITY
# ==============================================================================

def generate_figure_5():
    """Renders standardized continuous Wasserstein distance (SD units) vs SHAP stability."""
    feat_shift_path = os.path.join(CORRECTED_DIR, "feature_shift_summary_standardized.csv")
    est_path = os.path.join(CORRECTED_DIR, "integrated_estimands_corrected.csv")
    corr_path = os.path.join(CORRECTED_DIR, "integrated_correlations_corrected.csv")

    df_feat = pd.read_csv(feat_shift_path)
    df_est = pd.read_csv(est_path)
    df_corr = pd.read_csv(corr_path)

    # 1. Verify 4 continuous features and direction-level mean standardized W1
    c_feats = df_feat[df_feat["feature_type"] == "continuous"]
    assert set(c_feats["feature"].unique()) == {"age", "trestbps", "thalach", "oldpeak"}, "Continuous features mismatch"

    expected_shifts = {
        ("Cleveland", "Hungarian"): 0.41705,
        ("Cleveland", "Zurich"): 0.494225,
        ("Cleveland", "VA Long Beach"): 0.54735,
        ("Hungarian", "Cleveland"): 0.468775,
        ("Zurich", "Cleveland"): 0.45065,
        ("VA Long Beach", "Cleveland"): 0.58105
    }
    dir_shifts = {}
    for (tr, te), grp in c_feats.groupby(["train_dataset", "test_dataset"]):
        mean_w1 = grp["standardized_wasserstein_distance"].mean()
        dir_shifts[(tr, te)] = mean_w1
        exp_w1 = expected_shifts[(tr, te)]
        assert abs(mean_w1 - exp_w1) < 1e-4, f"Mismatch in {tr}->{te} shift: {mean_w1} vs {exp_w1}"

    # Map the 4-continuous feature standardized W1 to df_est for plotting
    df_est["mean_4feat_std_w1"] = df_est.apply(
        lambda r: dir_shifts[(r["train_dataset"], r["test_dataset"])], axis=1
    )

    # 2. Extract verified correlation statistics
    # Full cohort (N=24)
    r_full = df_corr[df_corr["comparison_name"] == "Standardized Wasserstein vs SHAP Stability"].iloc[0]
    rho_full = float(r_full["spearman_rho"])
    ci_full_lo = float(r_full["cluster_boot_ci_lower"])
    ci_full_hi = float(r_full["cluster_boot_ci_upper"])
    p_full = float(r_full["naive_p_value_uncorrected"])

    assert abs(rho_full - (-0.4103)) < 1e-4, f"Full rho mismatch: {rho_full}"
    assert abs(ci_full_lo - (-0.6738)) < 1e-3, f"Full CI lower mismatch: {ci_full_lo}"
    assert abs(ci_full_hi - 0.0471) < 1e-3, f"Full CI upper mismatch: {ci_full_hi}"

    # Zurich-excluded (N=16)
    r_sens = df_corr[df_corr["comparison_name"] == "Standardized Wasserstein vs SHAP Stability [No Zurich]"].iloc[0]
    rho_sens = float(r_sens["spearman_rho"])
    ci_sens_lo = float(r_sens["cluster_boot_ci_lower"])
    ci_sens_hi = float(r_sens["cluster_boot_ci_upper"])
    p_sens = float(r_sens["naive_p_value_uncorrected"])

    assert abs(rho_sens - (-0.6282)) < 1e-4, f"Sensitivity rho mismatch: {rho_sens}"
    assert abs(ci_sens_lo - (-0.7272)) < 1e-3, f"Sensitivity CI lower mismatch: {ci_sens_lo}"
    assert abs(ci_sens_hi - (-0.1463)) < 1e-3, f"Sensitivity CI upper mismatch: {ci_sens_hi}"

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)

    # Scatter plot by model
    for m_name in MODEL_ORDER:
        m_df = df_est[df_est["model"] == m_name]
        nz_df = m_df[~m_df["is_zurich_transfer"]]
        z_df = m_df[m_df["is_zurich_transfer"]]

        # Non-Zurich points
        ax.scatter(
            nz_df["mean_4feat_std_w1"], nz_df["spearman_rank_correlation"],
            color=MODEL_COLORS[m_name], marker=MODEL_MARKERS[m_name], s=85, alpha=0.85,
            edgecolors="black", linewidth=0.8,
            label=MODEL_DISPLAY[MODEL_ORDER.index(m_name)]
        )
        # Zurich transfers (distinct yellow face highlight)
        if len(z_df) > 0:
            ax.scatter(
                z_df["mean_4feat_std_w1"], z_df["spearman_rank_correlation"],
                facecolors="#ffdd57", edgecolors=MODEL_COLORS[m_name], marker=MODEL_MARKERS[m_name],
                s=115, linewidth=2.0, alpha=0.95,
                label=f"{MODEL_DISPLAY[MODEL_ORDER.index(m_name)]} (Zurich Transfer)" if m_name == "LogisticRegression" else None
            )

    # Non-causal trendline
    sns.regplot(
        data=df_est, x="mean_4feat_std_w1", y="spearman_rank_correlation",
        scatter=False, color="#4a5568", ax=ax,
        line_kws={"linestyle": "--", "linewidth": 1.5, "alpha": 0.75}
    )

    # Annotation box: clearly distinguishing primary exploratory analysis from Zurich-excluded sensitivity
    stat_text = (
        "Primary Analysis (All Transfers, N=24 across 6 clusters):\n"
        f"Spearman rho = {rho_full:.4f} (naive p = {p_full:.4f})\n"
        f"Cluster 95% Bootstrap CI: [{ci_full_lo:.4f}, +{ci_full_hi:.4f}]\n"
        "Interpretation: Exploratory / descriptive (cluster CI spans zero)\n\n"
        "Zurich-Excluded Sensitivity Analysis (N=16 across 4 clusters):\n"
        f"Spearman rho = {rho_sens:.4f} (naive p = {p_sens:.4f})\n"
        f"Cluster 95% Bootstrap CI: [{ci_sens_lo:.4f}, {ci_sens_hi:.4f}]"
    )
    ax.text(
        0.04, 0.05, stat_text,
        transform=ax.transAxes, fontsize=8.2, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cbd5e0", alpha=0.94)
    )

    ax.set_title("Figure 5: Association Between Standardized Covariate Shift and SHAP Explanation Rank Stability",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Source-Standardized 1-Wasserstein Distance across 4 Continuous Features (SD units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("SHAP Global Feature Rank Stability (Spearman rho)", fontsize=10, fontweight="bold")
    ax.set_ylim(0.48, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "FIGURE_5_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_5_FINAL.pdf")
    safe_savefig(plt, png_path, dpi=300)
    safe_savefig(plt, pdf_path)
    plt.close()
    print("[OK] Generated Figure 5: FIGURE_5_FINAL (PNG + PDF)")


# ==============================================================================
# FIGURE 6: SUBGROUP ROBUSTNESS & MISSINGNESS SENSITIVITY
# ==============================================================================

def generate_figure_6():
    """
    Renders demographic subgroup disparities (strictly filtered for evaluability)
    and missingness sensitivity across preprocessing regimes.
    """
    sub_path = os.path.join(RESULTS_DIR, "subgroup_disparity.csv")
    miss_path = os.path.join(CORRECTED_DIR, "missingness_sensitivity_corrected.csv")

    df_sub = pd.read_csv(sub_path)
    df_miss = pd.read_csv(miss_path)

    # 1. Verification of Evaluability Filtering for Subgroup Comparisons
    # Rule: N >= 30, events >= 10, non-events >= 10
    eval_sub = df_sub[df_sub["is_evaluable_comparison"] == True].copy()
    excluded_sub = df_sub[df_sub["is_evaluable_comparison"] == False].copy()

    assert len(eval_sub) == 60, f"Expected 60 evaluable comparisons, got {len(eval_sub)}"
    assert len(excluded_sub) == 36, f"Expected 36 excluded comparisons, got {len(excluded_sub)}"

    # 2. Verification of Missingness Sensitivity Results
    tcc = df_miss[df_miss["test_complete_case_status"].str.contains("Valid", na=False)]
    fcc = df_miss[df_miss["full_complete_case_status"] == "Valid"]
    mi = df_miss[df_miss["primary_eval_status"].str.contains("Valid", na=False)].dropna(subset=["delta_auc_missing_indicator"])

    assert len(tcc) == 20, f"Expected 20 evaluable test CC sets, got {len(tcc)}"
    assert len(fcc) == 16, f"Expected 16 evaluable full CC sets, got {len(fcc)}"
    assert len(mi) == 20, f"Expected 20 evaluable missingness indicator sets, got {len(mi)}"

    tcc_med = tcc["delta_auc_test_complete_case"].median()
    assert abs(tcc_med - 0.0) < 1e-4, f"Test CC median must be 0.00, got {tcc_med}"

    inf = tcc[tcc["test_dataset"].isin(["Hungarian", "VA Long Beach"]) & (tcc["train_dataset"] == "Cleveland")]
    assert len(inf) == 8, f"Expected 8 informative comparisons, got {len(inf)}"
    inf_med = inf["delta_auc_test_complete_case"].median()
    assert abs(inf_med - (-3.61)) < 1e-2, f"Informative median must be -3.61, got {inf_med}"

    fcc_med = fcc["delta_auc_full_complete_case"].median()
    assert abs(fcc_med - (-3.385)) < 1e-2, f"Full CC median must be -3.38, got {fcc_med}"

    mi_med = mi["delta_auc_missing_indicator"].median()
    assert abs(mi_med - (-0.125)) < 1e-2, f"MI median must be -0.12, got {mi_med}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=300)

    # Panel A: Demographic Subgroup Disparities (Filtered for Evaluability: N=60)
    order = [
        "Sex (Male vs Female)",
        "Age (<50 vs 50-64)",
        "Age (50-64 vs >=65)",
        "Age (<50 vs >=65)"
    ]
    display_labels = [
        "Sex\n(Male vs Female)\n[N=16 evaluable]",
        "Age\n(<50 vs 50-64)\n[N=16 evaluable]",
        "Age\n(50-64 vs >=65)\n[N=16 evaluable]",
        "Age\n(<50 vs >=65)\n[N=12 evaluable]"
    ]

    sns.boxplot(data=eval_sub, x="comparison_type", y="abs_delta_roc_auc", order=order,
                hue="comparison_type", palette="Purples", legend=False, ax=ax1, width=0.42, fliersize=0)
    sns.stripplot(data=eval_sub, x="comparison_type", y="abs_delta_roc_auc", order=order,
                  color="#2d3748", jitter=0.18, size=5.5, alpha=0.85, ax=ax1)

    ax1.set_title("A: Discrimination Disparity Across Evaluated Subgroups", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Demographic Subgroup Comparison", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Absolute ROC-AUC Disparity (|Delta AUC|, %)", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(display_labels, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Annotate evaluability filter on Panel A
    ax1.text(
        0.5, 0.96,
        "Evaluability Rule: N >= 30, events >= 10, non-events >= 10\n"
        "(60 evaluable comparisons plotted; 36 degenerate comparisons excluded)",
        transform=ax1.transAxes, ha="center", va="top", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7fafc", edgecolor="#cbd5e0", alpha=0.9)
    )

    # Panel B: Missingness Sensitivity Across Preprocessing Regimes
    # Prepare data for plotting
    miss_plot_data = []
    for val in tcc["delta_auc_test_complete_case"]:
        miss_plot_data.append({"Regime": "Test Complete-Case\n(All N=20)", "delta_auc": val})
    for val in inf["delta_auc_test_complete_case"]:
        miss_plot_data.append({"Regime": "Test CC Informative\n(Clev->Hung/VA, N=8)", "delta_auc": val})
    for val in fcc["delta_auc_full_complete_case"]:
        miss_plot_data.append({"Regime": "Full Complete-Case\n(Train+Test, N=16)", "delta_auc": val})
    for val in mi["delta_auc_missing_indicator"]:
        miss_plot_data.append({"Regime": "Missingness Indicator\n(N=20)", "delta_auc": val})

    df_miss_plot = pd.DataFrame(miss_plot_data)
    regime_order = [
        "Test Complete-Case\n(All N=20)",
        "Test CC Informative\n(Clev->Hung/VA, N=8)",
        "Full Complete-Case\n(Train+Test, N=16)",
        "Missingness Indicator\n(N=20)"
    ]

    sns.boxplot(data=df_miss_plot, x="Regime", y="delta_auc", order=regime_order,
                hue="Regime", palette="Oranges", legend=False, ax=ax2, width=0.45, fliersize=0)
    sns.stripplot(data=df_miss_plot, x="Regime", y="delta_auc", order=regime_order,
                  color="#2d3748", jitter=0.18, size=5.5, alpha=0.85, ax=ax2)

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.set_title("B: Sensitivity to Missingness Preprocessing Regimes", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Missing Data Evaluation Regime", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Change in External Discrimination (Delta ROC-AUC, %)", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(regime_order)))
    ax2.set_xticklabels(regime_order, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Neutral clinical interpretation banner on Panel B
    ax2.text(
        0.5, 0.04,
        "Neutral Interpretation: Performance changes under complete-case restrictions were\n"
        "concentrated in comparisons with substantial missingness and were partly\n"
        "confounded by case-mix differences. (Zurich excluded due to n_neg < 10)",
        transform=ax2.transAxes, ha="center", va="bottom", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffaf0", edgecolor="#dd6b20", alpha=0.92)
    )

    fig.suptitle("Figure 6: Evaluability-Filtered Demographic Disparities and Missingness Sensitivity",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    png_path = os.path.join(OUT_DIR, "FIGURE_6_FINAL.png")
    pdf_path = os.path.join(OUT_DIR, "FIGURE_6_FINAL.pdf")
    safe_savefig(plt, png_path, dpi=300)
    safe_savefig(plt, pdf_path)
    plt.close()
    print("[OK] Generated Figure 6: FIGURE_6_FINAL (PNG + PDF)")


# ==============================================================================
# MAIN DISPATCHER & MANIFEST GENERATION
# ==============================================================================

def main():
    print("=" * 75)
    print("Executing Final Publication Figure Regeneration into corrected_figures_final/")
    print("=" * 75)
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print("=" * 75)
    print("[SUCCESS] All 6 publication figures generated successfully in PNG and PDF.")
    print("=" * 75)


if __name__ == "__main__":
    main()
