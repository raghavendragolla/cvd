"""
Phase 7: Dataset Shift, Calibration Drift & Post-Hoc Recalibration Engine.
Implements:
  - Quantitative feature-level dataset shift metrics (continuous: SMD, Wasserstein, KS; discrete: TVD, JSD, Chi-square).
  - Missingness shift analysis across cohorts before imputation.
  - Prevalence / target shift analysis across the 4 cohorts.
  - Prediction distribution shift analysis using Phase 5 predictions.
  - Development-only post-hoc recalibration: Platt scaling and Isotonic regression fitted strictly on 5-fold OOF development predictions.
  - Out-of-sample external evaluation of recalibration (comparing Uncalibrated, Platt, Isotonic).
  - Cross-phase synthesis connecting dataset shift to Phase 5 performance degradation and Phase 6 explanation stability.
  - 11 publication figures saved in figures/dataset_shift/ and figures/calibration/.
  - Full execution manifest saved in results/phase7_manifest.json.
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
import scipy.spatial.distance as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_B_10_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor
from src.evaluate import evaluate_classifier, compute_expected_calibration_error, compute_calibration_intercept_slope
from experiments.exp_internal_validation import load_config, build_model_pipeline


CONTINUOUS_FEATURES = ["age", "trestbps", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]


def compute_continuous_shift(x_train: np.ndarray, x_test: np.ndarray) -> dict:
    """Computes distribution statistics, SMD, Wasserstein distance, and KS statistic."""
    mean_tr, std_tr = float(np.mean(x_train)), float(np.std(x_train))
    mean_te, std_te = float(np.mean(x_test)), float(np.std(x_test))
    med_tr, iqr_tr = float(np.median(x_train)), float(stats.iqr(x_train))
    med_te, iqr_te = float(np.median(x_test)), float(stats.iqr(x_test))

    # Standardized Mean Difference (Cohen's d)
    pooled_std = np.sqrt((std_tr**2 + std_te**2) / 2.0)
    smd = (mean_tr - mean_te) / pooled_std if pooled_std > 0 else 0.0

    # Wasserstein Distance (Earth Mover's Distance)
    wass_dist = stats.wasserstein_distance(x_train, x_test)

    # Kolmogorov-Smirnov 2-sample test
    ks_res = stats.ks_2samp(x_train, x_test)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    return {
        "train_mean": round(mean_tr, 3),
        "train_std": round(std_tr, 3),
        "train_median": round(med_tr, 3),
        "train_iqr": round(iqr_tr, 3),
        "test_mean": round(mean_te, 3),
        "test_std": round(std_te, 3),
        "test_median": round(med_te, 3),
        "test_iqr": round(iqr_te, 3),
        "smd": round(float(smd), 4),
        "wasserstein_distance": round(float(wass_dist), 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pval, 6)
    }


def compute_categorical_shift(s_train: pd.Series, s_test: pd.Series) -> dict:
    """Computes category proportions, Total Variation Distance (L1), Jensen-Shannon Divergence, and Chi-Square."""
    all_cats = sorted(list(set(s_train.dropna().unique()).union(set(s_test.dropna().unique()))))

    p_tr = s_train.value_counts(normalize=True).reindex(all_cats, fill_value=0.0).to_numpy()
    p_te = s_test.value_counts(normalize=True).reindex(all_cats, fill_value=0.0).to_numpy()

    # Total Variation Distance (L1 / 2)
    tvd = 0.5 * np.sum(np.abs(p_tr - p_te))

    # Jensen-Shannon Divergence (bounded in [0, 1])
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

    prop_summary_tr = "; ".join([f"cat_{c}:{p_tr[i]:.2f}" for i, c in enumerate(all_cats)])
    prop_summary_te = "; ".join([f"cat_{c}:{p_te[i]:.2f}" for i, c in enumerate(all_cats)])

    return {
        "tvd": round(float(tvd), 4),
        "jensen_shannon_divergence": round(float(jsd), 4),
        "chi2_statistic": round(chi2_stat, 3),
        "chi2_pvalue": round(chi2_pval, 6),
        "train_proportions": prop_summary_tr,
        "test_proportions": prop_summary_te
    }


def compute_raw_missingness_shift(raw_dir: str) -> pd.DataFrame:
    """Calculates pre-imputation raw missingness percentages per feature for every cohort."""
    raw_files = {
        "Cleveland": "cleveland_raw.csv",
        "Hungarian": "hungarian_raw.csv",
        "Zurich": "zurich_raw.csv",
        "VA Long Beach": "va_raw.csv"
    }

    records = []
    cohort_missing = {}
    for cohort_name, fname in raw_files.items():
        fpath = os.path.join(raw_dir, fname)
        df = pd.read_csv(fpath)
        cohort_missing[cohort_name] = {}
        for feat in TRACK_B_10_FEATURES:
            col_data = df[feat]
            # Account for VA blood pressure zero
            if cohort_name == "VA Long Beach" and feat == "trestbps":
                n_miss = col_data.isna().sum() + (col_data <= 0).sum()
            else:
                n_miss = col_data.isna().sum()
            rate = round(float(n_miss / len(df)) * 100, 2)
            cohort_missing[cohort_name][feat] = rate

    pairwise_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland"),
    ]

    for tr, te in pairwise_directions:
        for feat in TRACK_B_10_FEATURES:
            rate_tr = cohort_missing[tr][feat]
            rate_te = cohort_missing[te][feat]
            diff = round(rate_te - rate_tr, 2)
            records.append({
                "train_dataset": tr,
                "test_dataset": te,
                "feature": feat,
                "train_missingness_pct": rate_tr,
                "test_missingness_pct": rate_te,
                "missingness_delta_pct": diff
            })

    return pd.DataFrame(records)


def compute_prevalence_shift(raw_dir: str) -> pd.DataFrame:
    """Calculates disease prevalence per cohort and absolute pairwise prevalence gaps."""
    cohort_stats = {}
    for c_name in ["cleveland", "hungarian", "zurich", "va_long_beach"]:
        df = load_harmonized_cohort(c_name, track="B10", data_dir=raw_dir)
        disp_name = "VA Long Beach" if c_name == "va_long_beach" else c_name.capitalize()
        y = df["target"].to_numpy()
        n = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        prev = round(float(n_pos / n) * 100, 2)
        cohort_stats[disp_name] = {
            "n": n, "n_positive": n_pos, "n_negative": n_neg, "prevalence_pct": prev
        }

    records = []
    pairwise_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland"),
    ]

    for tr, te in pairwise_directions:
        p_tr = cohort_stats[tr]["prevalence_pct"]
        p_te = cohort_stats[te]["prevalence_pct"]
        delta = round(p_te - p_tr, 2)
        records.append({
            "train_dataset": tr,
            "test_dataset": te,
            "train_n": cohort_stats[tr]["n"],
            "train_prevalence_pct": p_tr,
            "test_n": cohort_stats[te]["n"],
            "test_prevalence_pct": p_te,
            "prevalence_delta_pct": delta,
            "abs_prevalence_delta_pct": round(abs(delta), 2)
        })

    return pd.DataFrame(records)


def run_development_recalibration(
    X_dev_raw: pd.DataFrame,
    y_dev: np.ndarray,
    X_ext_raw: pd.DataFrame,
    y_ext: np.ndarray,
    model_name: str,
    config: dict
) -> tuple:
    """
    Fits Platt scaling and Isotonic regression strictly on 5-fold cross-validated out-of-fold (OOF)
    development predictions. Applies frozen mappings to external test data. Zero external labels accessed.
    """
    seed = config["experiment"]["random_seed"]
    eps = 1e-5

    # 1. Generate 5-Fold OOF Predictions on Development Cohort
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y_dev))

    for tr_idx, va_idx in skf.split(X_dev_raw, y_dev):
        fold_prep = HarmonizedClinicalPreprocessor(
            imputer_strategy=config["preprocessing"]["imputer_strategy"],
            n_neighbors=config["preprocessing"]["n_neighbors"],
            scale_features=False,
            clamp_negative_oldpeak=config["preprocessing"]["clamp_negative_oldpeak"]
        )
        X_tr = fold_prep.fit_transform(X_dev_raw.iloc[tr_idx])
        X_va = fold_prep.transform(X_dev_raw.iloc[va_idx])

        fold_model = build_model_pipeline(model_name, config)
        fold_model.fit(X_tr, y_dev[tr_idx])
        oof_probs[va_idx] = fold_model.predict_proba(X_va)[:, 1]

    # 2. Fit Platt Scaling on OOF Development Predictions
    clipped_oof = np.clip(oof_probs, eps, 1.0 - eps)
    logits_oof = np.log(clipped_oof / (1.0 - clipped_oof)).reshape(-1, 1)
    platt_calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
    platt_calibrator.fit(logits_oof, y_dev)

    # 3. Fit Isotonic Regression on OOF Development Predictions
    iso_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_calibrator.fit(oof_probs, y_dev)

    # 4. Fit Final Development Model on Full Development Cohort (Phase 5 Model)
    full_prep = HarmonizedClinicalPreprocessor(
        imputer_strategy=config["preprocessing"]["imputer_strategy"],
        n_neighbors=config["preprocessing"]["n_neighbors"],
        scale_features=False,
        clamp_negative_oldpeak=config["preprocessing"]["clamp_negative_oldpeak"]
    )
    X_dev_proc = full_prep.fit_transform(X_dev_raw)
    X_ext_proc = full_prep.transform(X_ext_raw)

    final_model = build_model_pipeline(model_name, config)
    final_model.fit(X_dev_proc, y_dev)

    # 5. Predict External Uncalibrated Probabilities
    ext_raw_prob = final_model.predict_proba(X_ext_proc)[:, 1]

    # 6. Apply Frozen Recalibration Mappings to External Probabilities
    ext_clipped = np.clip(ext_raw_prob, eps, 1.0 - eps)
    ext_logits = np.log(ext_clipped / (1.0 - ext_clipped)).reshape(-1, 1)
    ext_platt_prob = platt_calibrator.predict_proba(ext_logits)[:, 1]
    ext_iso_prob = iso_calibrator.predict(ext_raw_prob)

    # Evaluate all 3 variants on external test cohort
    thresh = config["experiment"]["classification_threshold"]
    metrics_raw = evaluate_classifier(y_ext, (ext_raw_prob >= thresh).astype(int), ext_raw_prob)
    metrics_platt = evaluate_classifier(y_ext, (ext_platt_prob >= thresh).astype(int), ext_platt_prob)
    metrics_iso = evaluate_classifier(y_ext, (ext_iso_prob >= thresh).astype(int), ext_iso_prob)

    return (
        metrics_raw, metrics_platt, metrics_iso,
        ext_raw_prob, ext_platt_prob, ext_iso_prob, oof_probs
    )


def run_all_phase7_experiments():
    config = load_config("config.yaml")
    raw_dir = os.path.join(ROOT_DIR, config["paths"]["raw_data_dir"])
    results_dir = os.path.join(ROOT_DIR, config["paths"]["results_dir"])
    figures_shift_dir = os.path.join(ROOT_DIR, "figures", "dataset_shift")
    figures_cal_dir = os.path.join(ROOT_DIR, "figures", "calibration")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_shift_dir, exist_ok=True)
    os.makedirs(figures_cal_dir, exist_ok=True)

    pairwise_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland"),
    ]
    models = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]

    print("=== Starting Phase 7: Dataset Shift, Calibration Drift & Recalibration ===")

    # 1. Compute Missingness Shift Table
    print("\n--- 1. Computing Raw Pre-Imputation Missingness Shift ---")
    missing_shift_df = compute_raw_missingness_shift(raw_dir)
    missing_path = os.path.join(results_dir, "missingness_shift.csv")
    missing_shift_df.to_csv(missing_path, index=False)
    print(f"[OK] Saved missingness shift to: {missing_path}")

    # 2. Compute Prevalence / Target Shift Table
    print("\n--- 2. Computing Cohort Prevalence and Base Rate Shifts ---")
    prev_shift_df = compute_prevalence_shift(raw_dir)
    prev_path = os.path.join(results_dir, "prevalence_shift.csv")
    prev_shift_df.to_csv(prev_path, index=False)
    print(f"[OK] Saved prevalence shift to: {prev_path}")

    # 3. Compute Distributional Covariate Shifts (Continuous + Categorical)
    print("\n--- 3. Computing Feature-Level Distributional Shifts ---")
    feature_shift_records = []

    # Cache harmonized data
    cohort_data = {}
    for c_name in ["cleveland", "hungarian", "zurich", "va_long_beach"]:
        df = load_harmonized_cohort(c_name, track="B10", data_dir=raw_dir)
        disp_name = "VA Long Beach" if c_name == "va_long_beach" else c_name.capitalize()
        cohort_data[disp_name] = df

    for tr, te in pairwise_directions:
        df_tr = cohort_data[tr]
        df_te = cohort_data[te]

        # Continuous shifts
        for f in CONTINUOUS_FEATURES:
            cont_res = compute_continuous_shift(df_tr[f].dropna().to_numpy(), df_te[f].dropna().to_numpy())
            feature_shift_records.append({
                "train_dataset": tr,
                "test_dataset": te,
                "feature": f,
                "feature_type": "continuous",
                "smd": cont_res["smd"],
                "wasserstein_distance": cont_res["wasserstein_distance"],
                "ks_statistic": cont_res["ks_statistic"],
                "ks_pvalue": cont_res["ks_pvalue"],
                "tvd": np.nan,
                "jensen_shannon_divergence": np.nan,
                "chi2_statistic": np.nan,
                "chi2_pvalue": np.nan,
                "train_summary": f"mean={cont_res['train_mean']}, std={cont_res['train_std']}, iqr={cont_res['train_iqr']}",
                "test_summary": f"mean={cont_res['test_mean']}, std={cont_res['test_std']}, iqr={cont_res['test_iqr']}"
            })

        # Categorical shifts
        for f in CATEGORICAL_FEATURES:
            cat_res = compute_categorical_shift(df_tr[f], df_te[f])
            feature_shift_records.append({
                "train_dataset": tr,
                "test_dataset": te,
                "feature": f,
                "feature_type": "categorical",
                "smd": np.nan,
                "wasserstein_distance": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "tvd": cat_res["tvd"],
                "jensen_shannon_divergence": cat_res["jensen_shannon_divergence"],
                "chi2_statistic": cat_res["chi2_statistic"],
                "chi2_pvalue": cat_res["chi2_pvalue"],
                "train_summary": cat_res["train_proportions"],
                "test_summary": cat_res["test_proportions"]
            })

    feat_shift_df = pd.DataFrame(feature_shift_records)
    feat_shift_path = os.path.join(results_dir, "feature_shift_summary.csv")
    feat_shift_df.to_csv(feat_shift_path, index=False)
    print(f"[OK] Saved feature shift summary ({len(feat_shift_df)} rows) to: {feat_shift_path}")

    # 4. Compute Prediction Distribution Shifts
    print("\n--- 4. Computing Prediction Distribution Shift (Phase 5 Model Predictions) ---")
    p5_preds_path = os.path.join(results_dir, "external_predictions.csv")
    p4_oof_path = os.path.join(results_dir, "internal_oof_predictions.csv")
    p5_preds_df = pd.read_csv(p5_preds_path)
    p4_oof_df = pd.read_csv(p4_oof_path)

    pred_shift_records = []
    for (tr, te), grp in p5_preds_df.groupby(["train_dataset", "test_dataset"]):
        for m_name in models:
            ext_m_preds = grp[grp["model"] == m_name]["predicted_probability"].to_numpy()

            # Dev OOF predictions from Phase 4 Track B
            dev_sub = p4_oof_df[
                (p4_oof_df["cohort"] == tr) &
                (p4_oof_df["model"] == m_name) &
                (p4_oof_df["track"] == "Track_B")
            ]
            dev_oof_probs = dev_sub["predicted_probability"].to_numpy() if len(dev_sub) > 0 else np.array([])

            pred_shift_records.append({
                "train_dataset": tr,
                "test_dataset": te,
                "model": m_name,
                "dev_pred_mean": round(float(np.mean(dev_oof_probs)), 4) if len(dev_oof_probs) > 0 else np.nan,
                "dev_pred_median": round(float(np.median(dev_oof_probs)), 4) if len(dev_oof_probs) > 0 else np.nan,
                "dev_pred_std": round(float(np.std(dev_oof_probs)), 4) if len(dev_oof_probs) > 0 else np.nan,
                "dev_pred_pos_rate_50": round(float(np.mean(dev_oof_probs >= 0.50)), 4) if len(dev_oof_probs) > 0 else np.nan,
                "ext_pred_mean": round(float(np.mean(ext_m_preds)), 4),
                "ext_pred_median": round(float(np.median(ext_m_preds)), 4),
                "ext_pred_std": round(float(np.std(ext_m_preds)), 4),
                "ext_pred_pos_rate_50": round(float(np.mean(ext_m_preds >= 0.50)), 4),
                "pred_mean_delta": round(float(np.mean(ext_m_preds) - np.mean(dev_oof_probs)), 4) if len(dev_oof_probs) > 0 else np.nan,
                "pred_pos_rate_delta": round(float(np.mean(ext_m_preds >= 0.50) - np.mean(dev_oof_probs >= 0.50)), 4) if len(dev_oof_probs) > 0 else np.nan
            })

    pred_shift_df = pd.DataFrame(pred_shift_records)
    pred_shift_path = os.path.join(results_dir, "prediction_distribution_shift.csv")
    pred_shift_df.to_csv(pred_shift_path, index=False)
    print(f"[OK] Saved prediction distribution shift ({len(pred_shift_df)} rows) to: {pred_shift_path}")

    # 5. Execute Development-Only Post-Hoc Recalibration (24 Experiments)
    print("\n--- 5. Executing Development-Only Post-Hoc Recalibration (Platt & Isotonic) ---")
    recal_records = []
    cal_drift_records = []
    cached_predictions = {}

    exp_counter = 1
    for tr, te in pairwise_directions:
        df_tr = cohort_data[tr]
        df_te = cohort_data[te]

        X_tr = df_tr[TRACK_B_10_FEATURES].copy()
        y_tr = df_tr["target"].to_numpy()
        X_te = df_te[TRACK_B_10_FEATURES].copy()
        y_te = df_te["target"].to_numpy()

        for m_name in models:
            exp_id = f"RECAL_E{exp_counter}_{tr[:4].upper()}_TO_{te[:4].upper()}_{m_name}"
            print(f"  [{exp_id}] Recalibrating {tr} -> {te} | Model: {m_name}...")

            m_raw, m_platt, m_iso, p_raw, p_platt, p_iso, p_oof = run_development_recalibration(
                X_tr, y_tr, X_te, y_te, m_name, config
            )
            cached_predictions[exp_id] = {
                "y_true": y_te, "p_raw": p_raw, "p_platt": p_platt, "p_iso": p_iso
            }

            # Recalibration comparative row
            recal_records.append({
                "experiment_id": exp_id,
                "train_dataset": tr,
                "test_dataset": te,
                "model": m_name,
                "raw_brier": m_raw["brier_score"],
                "platt_brier": m_platt["brier_score"],
                "iso_brier": m_iso["brier_score"],
                "platt_brier_delta": round(m_platt["brier_score"] - m_raw["brier_score"], 4),
                "iso_brier_delta": round(m_iso["brier_score"] - m_raw["brier_score"], 4),
                "raw_ece": m_raw["ece"],
                "platt_ece": m_platt["ece"],
                "iso_ece": m_iso["ece"],
                "platt_ece_delta": round(m_platt["ece"] - m_raw["ece"], 4),
                "iso_ece_delta": round(m_iso["ece"] - m_raw["ece"], 4),
                "raw_slope": m_raw["calibration_slope"],
                "platt_slope": m_platt["calibration_slope"],
                "iso_slope": m_iso["calibration_slope"],
                "raw_intercept": m_raw["calibration_intercept"],
                "platt_intercept": m_platt["calibration_intercept"],
                "iso_intercept": m_iso["calibration_intercept"],
                "raw_roc_auc": m_raw["roc_auc"],
                "platt_roc_auc": m_platt["roc_auc"],
                "iso_roc_auc": m_iso["roc_auc"],
                "raw_sensitivity": m_raw["sensitivity"],
                "platt_sensitivity": m_platt["sensitivity"],
                "iso_sensitivity": m_iso["sensitivity"],
                "raw_specificity": m_raw["specificity"],
                "platt_specificity": m_platt["specificity"],
                "iso_specificity": m_iso["specificity"]
            })

            # Granular calibration drift records (3 methods per experiment = 72 rows)
            for meth_name, m_dict in [("Raw_Uncalibrated", m_raw), ("Platt_Recalibrated", m_platt), ("Isotonic_Recalibrated", m_iso)]:
                cal_drift_records.append({
                    "experiment_id": exp_id,
                    "train_dataset": tr,
                    "test_dataset": te,
                    "model": m_name,
                    "calibration_method": meth_name,
                    "brier_score": m_dict["brier_score"],
                    "ece": m_dict["ece"],
                    "calibration_slope": m_dict["calibration_slope"],
                    "calibration_intercept": m_dict["calibration_intercept"],
                    "roc_auc": m_dict["roc_auc"],
                    "pr_auc": m_dict["pr_auc"],
                    "accuracy": m_dict["accuracy"],
                    "sensitivity": m_dict["sensitivity"],
                    "specificity": m_dict["specificity"],
                    "f1_score": m_dict["f1_score"],
                    "mcc": m_dict["mcc"]
                })

            exp_counter += 1

    recal_df = pd.DataFrame(recal_records)
    recal_path = os.path.join(results_dir, "recalibration_results.csv")
    recal_df.to_csv(recal_path, index=False)
    print(f"\n[OK] Saved recalibration results ({len(recal_df)} rows) to: {recal_path}")

    cal_drift_df = pd.DataFrame(cal_drift_records)
    cal_drift_path = os.path.join(results_dir, "calibration_drift.csv")
    cal_drift_df.to_csv(cal_drift_path, index=False)
    print(f"[OK] Saved calibration drift registry ({len(cal_drift_df)} rows) to: {cal_drift_path}")

    # 6. Cross-Phase Synthesis: Dataset Shift vs Degradation & Explanation Stability
    print("\n--- 6. Generating Cross-Phase Synthesis Dataset (Shift -> Performance -> XAI) ---")
    p5_res = pd.read_csv(os.path.join(results_dir, "external_validation_results.csv"))
    xai_res = pd.read_csv(os.path.join(results_dir, "xai_performance_relationships.csv"))

    # Compute aggregate mean Wasserstein distance and mean JSD per transfer direction
    cont_shifts = feat_shift_df[feat_shift_df["feature_type"] == "continuous"].groupby(["train_dataset", "test_dataset"])["wasserstein_distance"].mean().reset_index()
    cont_shifts.columns = ["train_dataset", "test_dataset", "mean_wasserstein_distance"]

    cat_shifts = feat_shift_df[feat_shift_df["feature_type"] == "categorical"].groupby(["train_dataset", "test_dataset"])["jensen_shannon_divergence"].mean().reset_index()
    cat_shifts.columns = ["train_dataset", "test_dataset", "mean_jsd"]

    smd_shifts = feat_shift_df[feat_shift_df["feature_type"] == "continuous"].groupby(["train_dataset", "test_dataset"])["smd"].apply(lambda s: round(float(np.mean(np.abs(s))), 4)).reset_index()
    smd_shifts.columns = ["train_dataset", "test_dataset", "mean_abs_smd"]

    dir_shifts = pd.merge(cont_shifts, cat_shifts, on=["train_dataset", "test_dataset"])
    dir_shifts = pd.merge(dir_shifts, smd_shifts, on=["train_dataset", "test_dataset"])
    dir_shifts = pd.merge(dir_shifts, prev_shift_df[["train_dataset", "test_dataset", "abs_prevalence_delta_pct"]], on=["train_dataset", "test_dataset"])

    # Merge with Phase 5 & 6 metrics
    cross_phase_synthesis = pd.merge(
        xai_res[[
            "experiment_id", "train_dataset", "test_dataset", "model",
            "auc_change", "brier_change", "sensitivity_change", "specificity_change",
            "external_calibration_slope", "external_calibration_intercept", "external_ece",
            "spearman_rank_correlation", "top3_overlap", "mean_abs_delta_shap"
        ]],
        dir_shifts,
        on=["train_dataset", "test_dataset"]
    )
    # Add recalibration delta
    cross_phase_synthesis = pd.merge(
        cross_phase_synthesis,
        recal_df[["train_dataset", "test_dataset", "model", "platt_brier_delta", "iso_brier_delta", "platt_ece_delta", "iso_ece_delta"]],
        on=["train_dataset", "test_dataset", "model"]
    )

    synthesis_path = os.path.join(results_dir, "dataset_shift_relationships.csv")
    cross_phase_synthesis.to_csv(synthesis_path, index=False)
    print(f"[OK] Saved cross-phase synthesis ({len(cross_phase_synthesis)} rows) to: {synthesis_path}")

    # 7. Execution Manifest
    manifest = {
        "metadata": {
            "title": "Phase 7 Dataset Shift, Calibration Drift & Post-Hoc Recalibration Manifest",
            "project": "Beyond Accuracy: Cross-Dataset Generalization, Calibration and Explanation Stability",
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "random_seed": config["experiment"]["random_seed"],
            "feature_schema": TRACK_B_10_FEATURES,
            "models_evaluated": models,
            "pairwise_transfer_directions": [f"{tr} -> {te}" for tr, te in pairwise_directions],
            "total_recalibration_experiments": len(recal_df),
            "files_generated": [
                "results/feature_shift_summary.csv",
                "results/missingness_shift.csv",
                "results/prevalence_shift.csv",
                "results/prediction_distribution_shift.csv",
                "results/calibration_drift.csv",
                "results/recalibration_results.csv",
                "results/dataset_shift_relationships.csv"
            ],
            "isolation_rule": "Recalibration mappings fit strictly on 5-fold OOF development predictions; zero external labels accessed"
        }
    }
    manifest_path = os.path.join(results_dir, "phase7_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Saved Phase 7 manifest to: {manifest_path}")

    # 8. Generate 11 Publication Figures
    print("\n--- 8. Generating 11 Publication Figures (300 DPI) ---")
    generate_phase7_figures(
        feat_shift_df,
        missing_shift_df,
        prev_shift_df,
        pred_shift_df,
        recal_df,
        cross_phase_synthesis,
        cached_predictions,
        cohort_data,
        figures_shift_dir,
        figures_cal_dir
    )

    return recal_df, cross_phase_synthesis


def generate_phase7_figures(
    feat_shift_df, missing_shift_df, prev_shift_df, pred_shift_df,
    recal_df, cross_phase_synthesis, cached_preds, cohort_data,
    shift_dir, cal_dir
):
    """Generates 11 publication-grade figures documenting clinical dataset shifts and post-hoc recalibration."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"LogisticRegression": "#1f77b4", "RandomForest": "#2ca02c", "SVM": "#ff7f0e", "XGBoost": "#d62728"}

    # 1. Figure: Feature Distribution Comparisons (Age, ST Depression, Max HR, Blood Pressure)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    continuous_plots = [("age", "Age (years)"), ("trestbps", "Resting Blood Pressure (mm Hg)"),
                        ("thalach", "Max Heart Rate (bpm)"), ("oldpeak", "ST Depression (mm)")]
    for idx, (feat, label) in enumerate(continuous_plots):
        ax = axes[idx // 2, idx % 2]
        plot_data = []
        for c_name, c_df in cohort_data.items():
            s = c_df[feat].dropna()
            for v in s:
                plot_data.append({"Cohort": c_name, feat: v})
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x="Cohort", y=feat, ax=ax, palette="Set2", showmeans=True)
        ax.set_title(f"Cross-Cohort Distribution: {label}", fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "1_feature_distributions.png"), dpi=300)
    plt.close()

    # 2. Figure: Standardized Feature Shift Heatmap (Wasserstein Distance for Continuous; TVD for Categorical)
    plt.figure(figsize=(12, 6))
    feat_shift_df["direction"] = feat_shift_df["train_dataset"] + " -> " + feat_shift_df["test_dataset"]
    # Combine Wasserstein for continuous and TVD for categorical into a normalized divergence measure
    feat_shift_df["divergence"] = feat_shift_df["wasserstein_distance"].fillna(0) + feat_shift_df["tvd"].fillna(0)
    pivot_shift = feat_shift_df.pivot(index="direction", columns="feature", values="divergence")
    sns.heatmap(pivot_shift, annot=True, cmap="YlOrRd", fmt=".2f", cbar_kws={"label": "Divergence (Wasserstein / TVD)"})
    plt.title("Biomarker Distributional Divergence Across Hospital Transfer Directions", fontsize=13, fontweight="bold")
    plt.xlabel("Harmonized Track B Clinical Biomarker", fontsize=11)
    plt.ylabel("Hospital Transfer Direction", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "2_feature_shift_heatmap.png"), dpi=300)
    plt.close()

    # 3. Figure: Missingness Shift Heatmap
    plt.figure(figsize=(12, 6))
    missing_shift_df["direction"] = missing_shift_df["train_dataset"] + " -> " + missing_shift_df["test_dataset"]
    pivot_miss = missing_shift_df.pivot(index="direction", columns="feature", values="missingness_delta_pct")
    sns.heatmap(pivot_miss, annot=True, cmap="coolwarm", fmt="+.1f", cbar_kws={"label": "Delta Missingness (%) [Ext - Dev]"})
    plt.title("Pre-Imputation Missingness Pattern Shift Across Transfer Directions", fontsize=13, fontweight="bold")
    plt.xlabel("Clinical Biomarker", fontsize=11)
    plt.ylabel("Hospital Transfer Direction", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "3_missingness_shift_heatmap.png"), dpi=300)
    plt.close()

    # 4. Figure: Prevalence Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    prev_plot_df = pd.DataFrame([
        {"Cohort": "Cleveland", "Prevalence": 45.87, "N": 303},
        {"Cohort": "Hungarian", "Prevalence": 36.18, "N": 293},
        {"Cohort": "Zurich", "Prevalence": 93.50, "N": 123},
        {"Cohort": "VA Long Beach", "Prevalence": 74.37, "N": 199}
    ])
    sns.barplot(data=prev_plot_df, x="Cohort", y="Prevalence", palette="Blues_d")
    for idx, r in prev_plot_df.iterrows():
        plt.text(idx, r["Prevalence"] + 2, f"{r['Prevalence']:.1f}% (N={r['N']})", ha="center", fontweight="bold")
    plt.title("Baseline CAD Disease Prevalence Across International Hospital Cohorts", fontsize=12, fontweight="bold")
    plt.ylabel("Angiographic CAD Prevalence (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "4_prevalence_comparison.png"), dpi=300)
    plt.close()

    # 5. Figure: Prediction Distribution Shift Comparison (Dev OOF vs External)
    plt.figure(figsize=(11, 5))
    pred_melt = pd.melt(
        pred_shift_df,
        id_vars=["train_dataset", "test_dataset", "model"],
        value_vars=["dev_pred_mean", "ext_pred_mean"],
        var_name="cohort_scope",
        value_name="mean_probability"
    )
    pred_melt["direction"] = pred_melt["train_dataset"] + " -> " + pred_melt["test_dataset"]
    pred_melt["cohort_scope"] = pred_melt["cohort_scope"].map({"dev_pred_mean": "Dev (OOF)", "ext_pred_mean": "Ext (Observed)"})
    sns.barplot(data=pred_melt, x="direction", y="mean_probability", hue="cohort_scope", palette="muted")
    plt.title("Mean Predicted Probability Shift: Development OOF vs External Test", fontsize=12, fontweight="bold")
    plt.xlabel("Hospital Transfer Direction", fontsize=10)
    plt.ylabel("Mean Predicted CAD Probability", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "5_prediction_distribution_comparison.png"), dpi=300)
    plt.close()

    # 6. Figure: External Calibration Curves Before Recalibration (Cleveland -> Hungarian & Cleveland -> VA)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    exp_hung = "RECAL_E2_CLEV_TO_HUNG_RandomForest"
    exp_va = "RECAL_E10_CLEV_TO_VA L_RandomForest"

    for ax, exp_key, title in zip(axes, [exp_hung, exp_va], ["Cleveland -> Hungarian (RF)", "Cleveland -> VA Long Beach (RF)"]):
        d = cached_preds[exp_key]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        # Empirical bins
        frac_pos, mean_pred = stats.binned_statistic(d["p_raw"], d["y_true"], statistic="mean", bins=6)[:2]
        bin_centers = 0.5 * (mean_pred[:-1] + mean_pred[1:])
        ax.plot(bin_centers, frac_pos, "s-", color="#d62728", label="Uncalibrated Raw")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Risk", fontsize=10)
        ax.set_ylabel("Observed Proportion", fontsize=10)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cal_dir, "6_calibration_curves_before.png"), dpi=300)
    plt.close()

    # 7. Figure: External Calibration Curves After Recalibration (Uncalibrated vs Platt vs Isotonic)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, exp_key, title in zip(axes, [exp_hung, exp_va], ["Cleveland -> Hungarian (RF)", "Cleveland -> VA Long Beach (RF)"]):
        d = cached_preds[exp_key]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        for p_arr, col, lbl in [(d["p_raw"], "#7f7f7f", "Raw Uncalibrated"),
                                (d["p_platt"], "#1f77b4", "Platt Recalibrated"),
                                (d["p_iso"], "#2ca02c", "Isotonic Recalibrated")]:
            frac_pos, mean_pred = stats.binned_statistic(p_arr, d["y_true"], statistic="mean", bins=6)[:2]
            bin_centers = 0.5 * (mean_pred[:-1] + mean_pred[1:])
            ax.plot(bin_centers, frac_pos, "o-", color=col, label=lbl, alpha=0.85)
        ax.set_title(f"Recalibrated: {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Event Rate", fontsize=10)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cal_dir, "7_calibration_curves_after.png"), dpi=300)
    plt.close()

    # 8. Figure: Brier Change Before vs After Recalibration
    plt.figure(figsize=(12, 5))
    recal_df["direction"] = recal_df["train_dataset"] + " -> " + recal_df["test_dataset"]
    brier_melt = pd.melt(
        recal_df,
        id_vars=["direction", "model"],
        value_vars=["raw_brier", "platt_brier", "iso_brier"],
        var_name="cal_variant",
        value_name="brier_score"
    )
    brier_melt["cal_variant"] = brier_melt["cal_variant"].map({
        "raw_brier": "Raw Uncalibrated", "platt_brier": "Platt Recalibrated", "iso_brier": "Isotonic Recalibrated"
    })
    sns.barplot(data=brier_melt, x="direction", y="brier_score", hue="cal_variant", palette="Set1")
    plt.title("Brier Score Comparison: Raw vs Development-Recalibrated (Lower is Better)", fontsize=12, fontweight="bold")
    plt.xlabel("Hospital Transfer Direction", fontsize=10)
    plt.ylabel("External Brier Score Loss", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(cal_dir, "8_brier_before_after.png"), dpi=300)
    plt.close()

    # 9. Figure: ECE Before vs After Recalibration
    plt.figure(figsize=(12, 5))
    ece_melt = pd.melt(
        recal_df,
        id_vars=["direction", "model"],
        value_vars=["raw_ece", "platt_ece", "iso_ece"],
        var_name="cal_variant",
        value_name="ece_val"
    )
    ece_melt["cal_variant"] = ece_melt["cal_variant"].map({
        "raw_ece": "Raw Uncalibrated", "platt_ece": "Platt Recalibrated", "iso_ece": "Isotonic Recalibrated"
    })
    sns.barplot(data=ece_melt, x="direction", y="ece_val", hue="cal_variant", palette="Set2")
    plt.title("Expected Calibration Error (ECE): Raw vs Development-Recalibrated (Lower is Better)", fontsize=12, fontweight="bold")
    plt.xlabel("Hospital Transfer Direction", fontsize=10)
    plt.ylabel("External ECE", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(os.path.join(cal_dir, "9_ece_before_after.png"), dpi=300)
    plt.close()

    # 10. Figure: Dataset Shift vs Calibration Degradation Scatter
    plt.figure(figsize=(9, 6))
    cross_phase_synthesis["direction"] = cross_phase_synthesis["train_dataset"] + " -> " + cross_phase_synthesis["test_dataset"]
    sns.scatterplot(
        data=cross_phase_synthesis,
        x="abs_prevalence_delta_pct",
        y="brier_change",
        hue="model",
        palette=palette,
        s=130,
        alpha=0.9
    )
    sns.regplot(
        data=cross_phase_synthesis,
        x="abs_prevalence_delta_pct",
        y="brier_change",
        scatter=False,
        color="gray",
        line_kws={"linestyle": "--", "alpha": 0.6}
    )
    plt.title("Target Prevalence Divergence vs Calibration Degradation (Delta Brier)", fontsize=12, fontweight="bold")
    plt.xlabel("Absolute Prevalence Gap (%) [|Ext Base Rate - Dev Base Rate|]", fontsize=11)
    plt.ylabel("Delta Brier Score [Positive = Degraded Calibration]", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "10_shift_vs_calibration.png"), dpi=300)
    plt.close()

    # 11. Figure: Dataset Shift vs SHAP Stability Scatter
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=cross_phase_synthesis,
        x="mean_wasserstein_distance",
        y="spearman_rank_correlation",
        hue="model",
        palette=palette,
        s=130,
        alpha=0.9
    )
    sns.regplot(
        data=cross_phase_synthesis,
        x="mean_wasserstein_distance",
        y="spearman_rank_correlation",
        scatter=False,
        color="gray",
        line_kws={"linestyle": "--", "alpha": 0.6}
    )
    plt.title("Biomarker Covariate Shift (Mean Wasserstein) vs Explanation Stability (rho)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Wasserstein Distance Across Continuous Biomarkers", fontsize=11)
    plt.ylabel("SHAP Feature Rank Correlation (rho)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(shift_dir, "11_shift_vs_shap_stability.png"), dpi=300)
    plt.close()

    print(f"[OK] Generated all 11 publication figures in {shift_dir} and {cal_dir}")


if __name__ == "__main__":
    run_all_phase7_experiments()
