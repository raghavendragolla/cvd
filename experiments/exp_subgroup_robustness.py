"""
Phase 8: Subgroup Robustness, Fairness & Missingness Sensitivity Engine.
Project: Beyond Accuracy: Cross-Dataset Generalization, Calibration and Explanation Stability
         for Explainable Heart Disease Prediction

Implements:
  1. Subgroup performance & calibration evaluation (Sex: 0 vs 1; Age: <50, 50-64, >=65).
  2. Subgroup performance disparity quantification (neutral comparative metrics, no winners).
  3. Stratified bootstrap confidence intervals (B=1000, seed=42) for evaluable cells.
  4. Missingness mechanism & co-occurrence pattern analysis across all 4 cohorts.
  5. Missingness sensitivity: Primary Imputation vs Complete-Case vs Missingness-Indicator pipelines.
  6. Explanation robustness: Subgroup SHAP feature rankings, rank correlation, and Top-K overlap.
  7. Publication figures in figures/subgroup_robustness/ and figures/missingness/.
  8. Full execution manifest saved in results/phase8_manifest.json.
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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_B_10_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor, mask_clinical_zero_sentinels
from src.evaluate import (
    compute_expected_calibration_error,
    compute_calibration_intercept_slope
)
from experiments.exp_internal_validation import load_config

# Predefined Age Groups
AGE_BINS = [-np.inf, 49.99, 64.99, np.inf]
AGE_LABELS = ["<50", "50-64", ">=65"]
MIN_EVALUABLE_N = 30


def get_subgroup_annotations(df_test: pd.DataFrame) -> pd.DataFrame:
    """Adds predefined sex and age subgroup columns to a test cohort dataframe."""
    annotated = df_test.copy()
    annotated["sex_str"] = annotated["sex"].map({0.0: "Female (sex=0)", 1.0: "Male (sex=1)", 0: "Female (sex=0)", 1: "Male (sex=1)"})
    annotated["age_group"] = pd.cut(annotated["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return annotated


def fast_ece(y_true, y_proba, n_bins=10):
    """Vectorized ECE calculation for fast bootstrap resampling."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_proba, bin_boundaries) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = (bin_idx == b)
        b_size = np.sum(mask)
        if b_size > 0:
            b_acc = np.mean(y_true[mask])
            b_conf = np.mean(y_proba[mask])
            ece += (b_size / n) * abs(b_acc - b_conf)
    return float(ece)


def fast_auc(y_true, y_proba):
    """Fast Mann-Whitney U AUC calculation in pure numpy."""
    ranks = stats.rankdata(y_proba)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    rank_pos = np.sum(ranks[y_true == 1])
    u = rank_pos - (n_pos * (n_pos + 1)) / 2.0
    return (u / (n_pos * n_neg)) * 100.0


def fast_f1(y_true, y_pred):
    """Fast F1 score calculation in pure numpy."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    denom = 2 * tp + np.sum((y_true == 0) & (y_pred == 1)) + np.sum((y_true == 1) & (y_pred == 0))
    return (2.0 * tp / denom * 100.0) if denom > 0 else 0.0


def compute_subgroup_bootstrap_cis(y_true, y_proba, y_pred, n_bootstraps=1000, seed=42):
    """Computes 95% stratified bootstrap confidence intervals across AUC, Brier, ECE, F1 in a single pass."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = np.array(y_pred)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) < 3 or len(neg_idx) < 3:
        return ("NA", "NA"), ("NA", "NA"), ("NA", "NA"), ("NA", "NA")

    b_pos_all = rng.choice(pos_idx, size=(n_bootstraps, len(pos_idx)), replace=True)
    b_neg_all = rng.choice(neg_idx, size=(n_bootstraps, len(neg_idx)), replace=True)

    auc_scores = []
    brier_scores = []
    ece_scores = []
    f1_scores = []

    for i in range(n_bootstraps):
        b_idx = np.concatenate([b_pos_all[i], b_neg_all[i]])
        yt_b = y_true[b_idx]
        yp_b = y_proba[b_idx]
        ypr_b = y_pred[b_idx]

        try:
            auc_scores.append(fast_auc(yt_b, yp_b))
            brier_scores.append(np.mean((yt_b - yp_b)**2))
            ece_scores.append(fast_ece(yt_b, yp_b))
            f1_scores.append(fast_f1(yt_b, ypr_b))
        except Exception:
            continue

    if len(auc_scores) < 100:
        return ("NA", "NA"), ("NA", "NA"), ("NA", "NA"), ("NA", "NA")

    auc_ci = (round(float(np.percentile(auc_scores, 2.5)), 2), round(float(np.percentile(auc_scores, 97.5)), 2))
    brier_ci = (round(float(np.percentile(brier_scores, 2.5)), 4), round(float(np.percentile(brier_scores, 97.5)), 4))
    ece_ci = (round(float(np.percentile(ece_scores, 2.5)), 4), round(float(np.percentile(ece_scores, 97.5)), 4))
    f1_ci = (round(float(np.percentile(f1_scores, 2.5)), 2), round(float(np.percentile(f1_scores, 97.5)), 2))

    return auc_ci, brier_ci, ece_ci, f1_ci



def run_subgroup_evaluation(ext_preds: pd.DataFrame, cohorts: dict) -> tuple:
    """
    Evaluates subgroup performance and calibration across all 24 pairwise transfer experiments.
    Returns (subgroup_perf_df, subgroup_cal_df, subgroup_disparity_df).
    """
    perf_records = []
    cal_records = []

    exp_groups = ext_preds.groupby(["experiment_id", "train_dataset", "test_dataset", "model"])

    for (exp_id, tr, te, model_name), df_exp in exp_groups:
        df_test_raw = cohorts[te]
        annotated_test = get_subgroup_annotations(df_test_raw)

        # Merge predictions with test annotations on sample_index
        df_merged = df_exp.merge(
            annotated_test[["sex", "sex_str", "age", "age_group"]].reset_index().rename(columns={"index": "sample_index"}),
            on="sample_index",
            how="left"
        )

        # Subgroup evaluations: Sex and Age
        subgroup_definitions = [
            ("sex", "0", df_merged[df_merged["sex"] == 0]),
            ("sex", "1", df_merged[df_merged["sex"] == 1]),
            ("age_group", "<50", df_merged[df_merged["age_group"] == "<50"]),
            ("age_group", "50-64", df_merged[df_merged["age_group"] == "50-64"]),
            ("age_group", ">=65", df_merged[df_merged["age_group"] == ">=65"])
        ]

        for var_name, val_name, sub in subgroup_definitions:
            n_tot = len(sub)
            n_pos = int((sub["true_label"] == 1).sum())
            n_neg = int((sub["true_label"] == 0).sum())
            prev = round(float(n_pos / n_tot * 100), 2) if n_tot > 0 else 0.0

            y_t = sub["true_label"].values
            y_p = sub["predicted_probability"].values
            y_pred = sub["predicted_label"].values

            # Rule for evaluability
            if n_tot < MIN_EVALUABLE_N:
                is_eval = False
                status = f"Small cell (N={n_tot} < {MIN_EVALUABLE_N})"
            elif n_pos == 0 or n_neg == 0:
                is_eval = False
                status = "Single class (ROC-AUC undefined)"
            else:
                is_eval = True
                status = "Valid"

            # Metric computations
            if n_tot > 0:
                brier = round(float(brier_score_loss(y_t, y_p)), 4)
                ece = round(float(fast_ece(y_t, y_p)), 4)
                mean_proba = round(float(np.mean(y_p)), 4)
                pos_rate_50 = round(float(np.mean(y_p >= 0.50)), 4)
                acc = round(float(accuracy_score(y_t, y_pred)), 4)
            else:
                brier, ece, mean_proba, pos_rate_50, acc = np.nan, np.nan, np.nan, np.nan, np.nan

            if is_eval:
                auc = round(float(roc_auc_score(y_t, y_p) * 100), 2)
                pr_auc = round(float(average_precision_score(y_t, y_p) * 100), 2)
                sens = round(float(recall_score(y_t, y_pred, zero_division=0) * 100), 2)
                spec = round(float(recall_score(1 - y_t, 1 - y_pred, zero_division=0) * 100), 2)
                prec = round(float(precision_score(y_t, y_pred, zero_division=0) * 100), 2)
                f1 = round(float(f1_score(y_t, y_pred, zero_division=0) * 100), 2)
                mcc = round(float(matthews_corrcoef(y_t, y_pred)), 4)
                intercept, slope = compute_calibration_intercept_slope(y_t, y_p)

                # Single-pass stratified bootstrap CIs
                (auc_l, auc_h), (brier_l, brier_h), (ece_l, ece_h), (f1_l, f1_h) = compute_subgroup_bootstrap_cis(
                    y_t, y_p, y_pred, n_bootstraps=1000, seed=42
                )
            else:
                auc = np.nan
                pr_auc = np.nan
                sens = round(float(recall_score(y_t, y_pred, zero_division=0) * 100), 2) if n_tot > 0 else np.nan
                spec = round(float(recall_score(1 - y_t, 1 - y_pred, zero_division=0) * 100), 2) if n_tot > 0 and n_neg > 0 else np.nan
                prec = round(float(precision_score(y_t, y_pred, zero_division=0) * 100), 2) if n_tot > 0 else np.nan
                f1 = round(float(f1_score(y_t, y_pred, zero_division=0) * 100), 2) if n_tot > 0 else np.nan
                mcc = np.nan
                intercept, slope = np.nan, np.nan
                auc_l, auc_h = "NA", "NA"
                brier_l, brier_h = "NA", "NA"
                ece_l, ece_h = "NA", "NA"
                f1_l, f1_h = "NA", "NA"


            perf_record = {
                "experiment_id": exp_id,
                "train_dataset": tr,
                "test_dataset": te,
                "model": model_name,
                "subgroup_variable": var_name,
                "subgroup_value": val_name,
                "n_total": n_tot,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "prevalence_pct": prev,
                "is_evaluable": is_eval,
                "status_note": status,
                "roc_auc": auc,
                "roc_auc_ci_lower": auc_l,
                "roc_auc_ci_upper": auc_h,
                "pr_auc": pr_auc,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": f1,
                "f1_ci_lower": f1_l,
                "f1_ci_upper": f1_h,
                "mcc": mcc,
                "brier_score": brier,
                "brier_ci_lower": brier_l,
                "brier_ci_upper": brier_h,
                "ece": ece,
                "ece_ci_lower": ece_l,
                "ece_ci_upper": ece_h,
                "calibration_slope": slope,
                "calibration_intercept": intercept
            }
            perf_records.append(perf_record)

            cal_records.append({
                "experiment_id": exp_id,
                "train_dataset": tr,
                "test_dataset": te,
                "model": model_name,
                "subgroup_variable": var_name,
                "subgroup_value": val_name,
                "n_total": n_tot,
                "prevalence_pct": prev,
                "is_evaluable": is_eval,
                "status_note": status,
                "mean_pred_proba": mean_proba,
                "pred_pos_rate_50": pos_rate_50,
                "brier_score": brier,
                "ece": ece,
                "calibration_slope": slope,
                "calibration_intercept": intercept
            })

    subgroup_perf_df = pd.DataFrame(perf_records)
    subgroup_cal_df = pd.DataFrame(cal_records)

    # Compute Subgroup Disparities
    disparity_records = []
    for (exp_id, tr, te, model_name), sub_group in subgroup_perf_df.groupby(["experiment_id", "train_dataset", "test_dataset", "model"]):
        # Sex Disparity: Male (1) vs Female (0)
        s0 = sub_group[(sub_group["subgroup_variable"] == "sex") & (sub_group["subgroup_value"] == "0")].iloc[0]
        s1 = sub_group[(sub_group["subgroup_variable"] == "sex") & (sub_group["subgroup_value"] == "1")].iloc[0]

        eval_sex = s0["is_evaluable"] and s1["is_evaluable"]
        auc_diff = round(abs(s1["roc_auc"] - s0["roc_auc"]), 2) if eval_sex else np.nan
        brier_diff = round(abs(s1["brier_score"] - s0["brier_score"]), 4) if not np.isnan(s0["brier_score"]) and not np.isnan(s1["brier_score"]) else np.nan
        ece_diff = round(abs(s1["ece"] - s0["ece"]), 4) if not np.isnan(s0["ece"]) and not np.isnan(s1["ece"]) else np.nan
        sens_diff = round(abs(s1["sensitivity"] - s0["sensitivity"]), 2) if not np.isnan(s0["sensitivity"]) and not np.isnan(s1["sensitivity"]) else np.nan
        spec_diff = round(abs(s1["specificity"] - s0["specificity"]), 2) if eval_sex else np.nan
        prev_diff = round(abs(s1["prevalence_pct"] - s0["prevalence_pct"]), 2)

        disparity_records.append({
            "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
            "comparison_type": "Sex (Male vs Female)",
            "subgroup_a": "Male (sex=1)", "subgroup_b": "Female (sex=0)",
            "n_a": s1["n_total"], "n_b": s0["n_total"],
            "is_evaluable_comparison": eval_sex,
            "comparison_note": "Both evaluable (N>=30, 2 classes)" if eval_sex else f"Suppressed: {s0['status_note']} | {s1['status_note']}",
            "abs_delta_prevalence_pct": prev_diff,
            "abs_delta_roc_auc": auc_diff,
            "abs_delta_brier": brier_diff,
            "abs_delta_ece": ece_diff,
            "abs_delta_sensitivity": sens_diff,
            "abs_delta_specificity": spec_diff
        })

        # Age Disparities: <50 vs 50-64, 50-64 vs >=65, <50 vs >=65
        a_young = sub_group[(sub_group["subgroup_variable"] == "age_group") & (sub_group["subgroup_value"] == "<50")].iloc[0]
        a_mid = sub_group[(sub_group["subgroup_variable"] == "age_group") & (sub_group["subgroup_value"] == "50-64")].iloc[0]
        a_old = sub_group[(sub_group["subgroup_variable"] == "age_group") & (sub_group["subgroup_value"] == ">=65")].iloc[0]

        age_pairs = [
            ("<50", "50-64", a_young, a_mid),
            ("50-64", ">=65", a_mid, a_old),
            ("<50", ">=65", a_young, a_old)
        ]
        for name_a, name_b, row_a, row_b in age_pairs:
            eval_age = row_a["is_evaluable"] and row_b["is_evaluable"]
            auc_d = round(abs(row_a["roc_auc"] - row_b["roc_auc"]), 2) if eval_age else np.nan
            brier_d = round(abs(row_a["brier_score"] - row_b["brier_score"]), 4) if not np.isnan(row_a["brier_score"]) and not np.isnan(row_b["brier_score"]) else np.nan
            ece_d = round(abs(row_a["ece"] - row_b["ece"]), 4) if not np.isnan(row_a["ece"]) and not np.isnan(row_b["ece"]) else np.nan
            sens_d = round(abs(row_a["sensitivity"] - row_b["sensitivity"]), 2) if not np.isnan(row_a["sensitivity"]) and not np.isnan(row_b["sensitivity"]) else np.nan
            spec_d = round(abs(row_a["specificity"] - row_b["specificity"]), 2) if eval_age else np.nan
            prev_d = round(abs(row_a["prevalence_pct"] - row_b["prevalence_pct"]), 2)

            disparity_records.append({
                "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
                "comparison_type": f"Age ({name_a} vs {name_b})",
                "subgroup_a": f"Age {name_a}", "subgroup_b": f"Age {name_b}",
                "n_a": row_a["n_total"], "n_b": row_b["n_total"],
                "is_evaluable_comparison": eval_age,
                "comparison_note": "Both evaluable (N>=30, 2 classes)" if eval_age else f"Suppressed: {row_a['status_note']} | {row_b['status_note']}",
                "abs_delta_prevalence_pct": prev_d,
                "abs_delta_roc_auc": auc_d,
                "abs_delta_brier": brier_d,
                "abs_delta_ece": ece_d,
                "abs_delta_sensitivity": sens_d,
                "abs_delta_specificity": spec_d
            })

    subgroup_disparity_df = pd.DataFrame(disparity_records)
    return subgroup_perf_df, subgroup_cal_df, subgroup_disparity_df


def run_missingness_mechanism_analysis(cohorts: dict) -> pd.DataFrame:
    """
    Computes feature missingness rates, co-occurrence patterns, and statistical associations
    with observed age, sex, and target CAD status across all 4 cohorts.
    """
    records = []
    for cohort_name, df in cohorts.items():
        n_total = len(df)
        df_masked = mask_clinical_zero_sentinels(df.copy())

        for feat in TRACK_B_10_FEATURES:
            is_miss = df_masked[feat].isna()
            n_miss = int(is_miss.sum())
            pct_miss = round(float(n_miss / n_total * 100), 2)

            # Association with target (odds ratio / correlation)
            if 0 < n_miss < n_total:
                y = (df_masked["target"] > 0).astype(int)
                contingency = pd.crosstab(is_miss, y)
                if contingency.shape == (2, 2):
                    try:
                        odds_ratio, p_val = stats.fisher_exact(contingency)
                    except Exception:
                        odds_ratio, p_val = np.nan, 1.0
                else:
                    odds_ratio, p_val = np.nan, 1.0

                # Correlation with age and sex
                age_corr, age_p = stats.pointbiserialr(is_miss, df_masked["age"].fillna(df_masked["age"].median()))
                sex_corr, sex_p = stats.pointbiserialr(is_miss, df_masked["sex"].fillna(1.0))
            else:
                odds_ratio, p_val = np.nan, np.nan
                age_corr, age_p = np.nan, np.nan
                sex_corr, sex_p = np.nan, np.nan

            records.append({
                "cohort": cohort_name,
                "feature": feat,
                "total_n": n_total,
                "n_missing": n_miss,
                "pct_missing": pct_miss,
                "missingness_present": n_miss > 0,
                "target_odds_ratio": round(float(odds_ratio), 3) if not np.isnan(odds_ratio) else np.nan,
                "target_association_pvalue": round(float(p_val), 5) if not np.isnan(p_val) else np.nan,
                "age_correlation": round(float(age_corr), 3) if not np.isnan(age_corr) else np.nan,
                "sex_correlation": round(float(sex_corr), 3) if not np.isnan(sex_corr) else np.nan,
                "mechanism_classification": "Descriptive statistical association (observational; non-causal)"
            })

    return pd.DataFrame(records)


def run_missingness_sensitivity_experiments(cohorts: dict, config: dict) -> pd.DataFrame:
    """
    Evaluates sensitivity of transportability performance across 3 distinct missingness handling regimes:
      1. Primary Development-Fitted Imputation (Phase 5 baseline)
      2. Complete-Case Sensitivity (evaluated on complete test cases)
      3. Complete-Case Trained & Evaluated (train complete cases -> test complete cases)
      4. Missingness-Indicator Sensitivity (train-only fitted SimpleImputer + MissingIndicator)
    """
    transfer_directions = [
        ("Cleveland", "Hungarian"),
        ("Cleveland", "Zurich"),
        ("Cleveland", "VA Long Beach"),
        ("Hungarian", "Cleveland"),
        ("Zurich", "Cleveland"),
        ("VA Long Beach", "Cleveland")
    ]
    model_types = ["LogisticRegression", "RandomForest", "SVM", "XGBoost"]

    records = []

    for tr_name, te_name in transfer_directions:
        df_tr_raw = cohorts[tr_name].copy()
        df_te_raw = cohorts[te_name].copy()

        y_tr = (df_tr_raw["target"] > 0).astype(int).values
        y_te = (df_te_raw["target"] > 0).astype(int).values

        # Determine complete-case masks for Track B
        df_tr_masked = mask_clinical_zero_sentinels(df_tr_raw[TRACK_B_10_FEATURES].copy())
        df_te_masked = mask_clinical_zero_sentinels(df_te_raw[TRACK_B_10_FEATURES].copy())

        tr_comp_mask = ~df_tr_masked.isna().any(axis=1).values
        te_comp_mask = ~df_te_masked.isna().any(axis=1).values

        n_te_comp = int(te_comp_mask.sum())
        n_tr_comp = int(tr_comp_mask.sum())

        for model_name in model_types:
            exp_id = f"EXP_PAIRWISE_{tr_name.upper()[:4]}_TO_{te_name.upper()[:4]}_{model_name}"

            # --- 1. Primary Pipeline (Phase 5) ---
            prep_prim = HarmonizedClinicalPreprocessor(scale_features=True)
            prep_prim.fit(df_tr_raw[TRACK_B_10_FEATURES])
            X_tr_prim = prep_prim.transform(df_tr_raw[TRACK_B_10_FEATURES])
            X_te_prim = prep_prim.transform(df_te_raw[TRACK_B_10_FEATURES])

            clf_prim = _build_model(model_name, config)
            clf_prim.fit(X_tr_prim, y_tr)
            p_te_prim = clf_prim.predict_proba(X_te_prim)[:, 1]

            auc_prim = round(float(roc_auc_score(y_te, p_te_prim) * 100), 2)
            brier_prim = round(float(brier_score_loss(y_te, p_te_prim)), 4)
            ece_prim = round(float(compute_expected_calibration_error(y_te, p_te_prim)), 4)

            # --- 2. Primary Pipeline Evaluated on Test Complete-Cases ---
            if n_te_comp >= MIN_EVALUABLE_N and len(np.unique(y_te[te_comp_mask])) > 1:
                p_te_cc = p_te_prim[te_comp_mask]
                y_te_cc = y_te[te_comp_mask]
                auc_te_cc = round(float(roc_auc_score(y_te_cc, p_te_cc) * 100), 2)
                brier_te_cc = round(float(brier_score_loss(y_te_cc, p_te_cc)), 4)
                ece_te_cc = round(float(compute_expected_calibration_error(y_te_cc, p_te_cc)), 4)
                note_te_cc = f"Valid (N_test_cc={n_te_comp})"
            else:
                auc_te_cc, brier_te_cc, ece_te_cc = np.nan, np.nan, np.nan
                note_te_cc = f"Insufficient test complete-cases (N={n_te_comp})"

            # --- 3. Complete-Case Trained & Evaluated (Train CC -> Test CC) ---
            if n_tr_comp >= MIN_EVALUABLE_N and n_te_comp >= MIN_EVALUABLE_N and len(np.unique(y_te[te_comp_mask])) > 1:
                df_tr_cc = df_tr_raw.iloc[tr_comp_mask]
                y_tr_cc = y_tr[tr_comp_mask]
                df_te_cc = df_te_raw.iloc[te_comp_mask]
                y_te_cc = y_te[te_comp_mask]

                prep_cc = HarmonizedClinicalPreprocessor(scale_features=True)
                prep_cc.fit(df_tr_cc[TRACK_B_10_FEATURES])
                X_tr_cc = prep_cc.transform(df_tr_cc[TRACK_B_10_FEATURES])
                X_te_cc = prep_cc.transform(df_te_cc[TRACK_B_10_FEATURES])

                clf_cc = _build_model(model_name, config)
                clf_cc.fit(X_tr_cc, y_tr_cc)
                p_te_both_cc = clf_cc.predict_proba(X_te_cc)[:, 1]

                auc_both_cc = round(float(roc_auc_score(y_te_cc, p_te_both_cc) * 100), 2)
                brier_both_cc = round(float(brier_score_loss(y_te_cc, p_te_both_cc)), 4)
                ece_both_cc = round(float(compute_expected_calibration_error(y_te_cc, p_te_both_cc)), 4)
                note_both_cc = f"Valid (N_tr_cc={n_tr_comp}, N_te_cc={n_te_comp})"
            else:
                auc_both_cc, brier_both_cc, ece_both_cc = np.nan, np.nan, np.nan
                note_both_cc = f"Insufficient CC sample size (N_tr={n_tr_comp}, N_te={n_te_comp})"

            # --- 4. Missingness-Indicator Pipeline ---
            # Train-only fitted SimpleImputer + MissingIndicator
            imp_ind = SimpleImputer(strategy="median")
            ind_gen = MissingIndicator(features="missing-only")

            X_tr_m = mask_clinical_zero_sentinels(df_tr_raw[TRACK_B_10_FEATURES].copy())
            X_te_m = mask_clinical_zero_sentinels(df_te_raw[TRACK_B_10_FEATURES].copy())

            X_tr_imp = imp_ind.fit_transform(X_tr_m)
            X_te_imp = imp_ind.transform(X_te_m)

            # Missing indicator features on train
            ind_tr = ind_gen.fit_transform(X_tr_m)
            ind_te = ind_gen.transform(X_te_m) if ind_tr.shape[1] > 0 else np.zeros((len(X_te_m), 0))

            if ind_tr.shape[1] > 0:
                X_tr_comb = np.hstack([X_tr_imp, ind_tr])
                X_te_comb = np.hstack([X_te_imp, ind_te])
            else:
                X_tr_comb = X_tr_imp
                X_te_comb = X_te_imp

            scaler_ind = StandardScaler()
            X_tr_scaled = scaler_ind.fit_transform(X_tr_comb)
            X_te_scaled = scaler_ind.transform(X_te_comb)

            clf_ind = _build_model(model_name, config)
            clf_ind.fit(X_tr_scaled, y_tr)
            p_te_ind = clf_ind.predict_proba(X_te_scaled)[:, 1]

            auc_ind = round(float(roc_auc_score(y_te, p_te_ind) * 100), 2)
            brier_ind = round(float(brier_score_loss(y_te, p_te_ind)), 4)
            ece_ind = round(float(compute_expected_calibration_error(y_te, p_te_ind)), 4)

            records.append({
                "experiment_id": exp_id,
                "train_dataset": tr_name,
                "test_dataset": te_name,
                "model": model_name,
                "primary_auc": auc_prim,
                "primary_brier": brier_prim,
                "primary_ece": ece_prim,
                "test_cc_auc": auc_te_cc,
                "test_cc_brier": brier_te_cc,
                "test_cc_ece": ece_te_cc,
                "test_cc_status": note_te_cc,
                "full_cc_auc": auc_both_cc,
                "full_cc_brier": brier_both_cc,
                "full_cc_ece": ece_both_cc,
                "full_cc_status": note_both_cc,
                "indicator_auc": auc_ind,
                "indicator_brier": brier_ind,
                "indicator_ece": ece_ind,
                "n_train_total": len(df_tr_raw),
                "n_train_cc": n_tr_comp,
                "n_test_total": len(df_te_raw),
                "n_test_cc": n_te_comp
            })

    return pd.DataFrame(records)


def run_subgroup_shap_analysis(cohorts: dict) -> pd.DataFrame:
    """
    Computes subgroup-level SHAP rankings, rank correlations, and Top-K overlap across sex and age groups.
    """
    shap_df = pd.read_csv(os.path.join(ROOT_DIR, "results", "shap_values_external.csv"))

    records = []
    exp_groups = shap_df.groupby(["experiment_id", "train_dataset", "test_dataset", "model"])

    for (exp_id, tr, te, model_name), df_exp in exp_groups:
        df_test_raw = cohorts[te]
        annotated_test = get_subgroup_annotations(df_test_raw)
        subgroup_meta = pd.DataFrame({
            "sample_index": annotated_test.index,
            "patient_sex": annotated_test["sex"].values,
            "patient_age": annotated_test["age"].values,
            "age_group": annotated_test["age_group"].values
        })

        df_merged = df_exp.merge(subgroup_meta, on="sample_index", how="left")

        # 1. Sex SHAP Comparison: Male vs Female
        sub_m = df_merged[df_merged["patient_sex"] == 1]
        sub_f = df_merged[df_merged["patient_sex"] == 0]


        if len(sub_m) >= MIN_EVALUABLE_N and len(sub_f) >= MIN_EVALUABLE_N:
            imp_m = sub_m[TRACK_B_10_FEATURES].abs().mean()
            imp_f = sub_f[TRACK_B_10_FEATURES].abs().mean()

            rho_sex, _ = stats.spearmanr(imp_m, imp_f)

            top3_m = set(imp_m.nlargest(3).index)
            top3_f = set(imp_f.nlargest(3).index)
            top3_jaccard = round(len(top3_m.intersection(top3_f)) / len(top3_m.union(top3_f)), 4)

            top5_m = set(imp_m.nlargest(5).index)
            top5_f = set(imp_f.nlargest(5).index)
            top5_jaccard = round(len(top5_m.intersection(top5_f)) / len(top5_m.union(top5_f)), 4)

            records.append({
                "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
                "comparison": "Sex (Male vs Female)",
                "subgroup_a": "Male (sex=1)", "subgroup_b": "Female (sex=0)",
                "n_a": len(sub_m), "n_b": len(sub_f),
                "spearman_rank_correlation": round(float(rho_sex), 4),
                "top3_overlap": top3_jaccard,
                "top5_overlap": top5_jaccard,
                "top3_features_a": "; ".join(imp_m.nlargest(3).index),
                "top3_features_b": "; ".join(imp_f.nlargest(3).index),
                "evaluation_status": "Valid"
            })
        else:
            records.append({
                "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
                "comparison": "Sex (Male vs Female)",
                "subgroup_a": "Male (sex=1)", "subgroup_b": "Female (sex=0)",
                "n_a": len(sub_m), "n_b": len(sub_f),
                "spearman_rank_correlation": np.nan, "top3_overlap": np.nan, "top5_overlap": np.nan,
                "top3_features_a": "NA", "top3_features_b": "NA",
                "evaluation_status": f"Cell size below threshold (Male N={len(sub_m)}, Female N={len(sub_f)})"
            })

        # 2. Age SHAP Comparison: <50 vs 50-64
        sub_young = df_merged[df_merged["age_group"] == "<50"]
        sub_mid = df_merged[df_merged["age_group"] == "50-64"]

        if len(sub_young) >= MIN_EVALUABLE_N and len(sub_mid) >= MIN_EVALUABLE_N:
            imp_y = sub_young[TRACK_B_10_FEATURES].abs().mean()
            imp_mid = sub_mid[TRACK_B_10_FEATURES].abs().mean()

            rho_age, _ = stats.spearmanr(imp_y, imp_mid)
            top3_y = set(imp_y.nlargest(3).index)
            top3_mid = set(imp_mid.nlargest(3).index)
            top3_age_jaccard = round(len(top3_y.intersection(top3_mid)) / len(top3_y.union(top3_mid)), 4)

            top5_y = set(imp_y.nlargest(5).index)
            top5_mid = set(imp_mid.nlargest(5).index)
            top5_age_jaccard = round(len(top5_y.intersection(top5_mid)) / len(top5_y.union(top5_mid)), 4)

            records.append({
                "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
                "comparison": "Age (<50 vs 50-64)",
                "subgroup_a": "Age <50", "subgroup_b": "Age 50-64",
                "n_a": len(sub_young), "n_b": len(sub_mid),
                "spearman_rank_correlation": round(float(rho_age), 4),
                "top3_overlap": top3_age_jaccard,
                "top5_overlap": top5_age_jaccard,
                "top3_features_a": "; ".join(imp_y.nlargest(3).index),
                "top3_features_b": "; ".join(imp_mid.nlargest(3).index),
                "evaluation_status": "Valid"
            })
        else:
            records.append({
                "experiment_id": exp_id, "train_dataset": tr, "test_dataset": te, "model": model_name,
                "comparison": "Age (<50 vs 50-64)",
                "subgroup_a": "Age <50", "subgroup_b": "Age 50-64",
                "n_a": len(sub_young), "n_b": len(sub_mid),
                "spearman_rank_correlation": np.nan, "top3_overlap": np.nan, "top5_overlap": np.nan,
                "top3_features_a": "NA", "top3_features_b": "NA",
                "evaluation_status": f"Cell size below threshold (<50 N={len(sub_young)}, 50-64 N={len(sub_mid)})"
            })

    return pd.DataFrame(records)


def _build_model(model_name: str, config: dict):
    """Instantiates a pristine model classifier from config."""
    seed = config.get("experiment", {}).get("random_seed", 42)
    model_params = {}
    for m in config.get("models", []):
        if m["name"] == model_name:
            model_params = m.get("params", {})
            break

    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=model_params.get("C", 1.0),
            solver=model_params.get("solver", "liblinear"),
            max_iter=model_params.get("max_iter", 1000),
            random_state=seed
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 5),
            min_samples_split=model_params.get("min_samples_split", 4),
            min_samples_leaf=model_params.get("min_samples_leaf", 2),
            random_state=seed,
            n_jobs=-1
        )
    elif model_name == "SVM":
        return SVC(
            C=model_params.get("C", 1.0),
            kernel=model_params.get("kernel", "rbf"),
            probability=True,
            random_state=seed
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 3),
            learning_rate=model_params.get("learning_rate", 0.05),
            subsample=model_params.get("subsample", 0.8),
            eval_metric="logloss",
            random_state=seed
        )
    raise ValueError(f"Unknown model name: {model_name}")



def generate_publication_figures(subgroup_perf_df: pd.DataFrame,
                                 disparity_df: pd.DataFrame,
                                 missingness_df: pd.DataFrame,
                                 sens_df: pd.DataFrame,
                                 shap_sub_df: pd.DataFrame):
    """Generates publication-quality figures at 300 DPI."""
    sub_dir = os.path.join(ROOT_DIR, "figures", "subgroup_robustness")
    miss_dir = os.path.join(ROOT_DIR, "figures", "missingness")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(miss_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.05)

    # 1. Subgroup ROC-AUC Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    eval_auc = subgroup_perf_df[subgroup_perf_df["is_evaluable"]].copy()
    eval_auc["subgroup_label"] = eval_auc["subgroup_variable"] + ": " + eval_auc["subgroup_value"]
    sns.boxplot(data=eval_auc, x="subgroup_label", y="roc_auc", hue="model", palette="Set2", ax=ax)
    ax.set_title("Subgroup ROC-AUC Across External Validation Experiments (N >= 30)", fontsize=13, weight="bold")
    ax.set_ylabel("External ROC-AUC (%)")
    ax.set_xlabel("Subgroup")
    ax.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "1_subgroup_roc_auc.png"), dpi=300)
    plt.close()

    # 2. Subgroup Brier Score Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=eval_auc, x="subgroup_label", y="brier_score", hue="model", palette="Spectral", ax=ax)
    ax.set_title("Subgroup Brier Score Across External Validation Experiments (N >= 30)", fontsize=13, weight="bold")
    ax.set_ylabel("External Brier Score (Lower is Better)")
    ax.set_xlabel("Subgroup")
    ax.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "2_subgroup_brier.png"), dpi=300)
    plt.close()

    # 3. Subgroup ECE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=eval_auc, x="subgroup_label", y="ece", hue="model", palette="coolwarm", ax=ax)
    ax.set_title("Subgroup Expected Calibration Error (ECE) Across External Experiments", fontsize=13, weight="bold")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_xlabel("Subgroup")
    ax.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "3_subgroup_ece.png"), dpi=300)
    plt.close()

    # 4. Sex Disparity Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sex_disp = disparity_df[disparity_df["comparison_type"].str.contains("Sex") & disparity_df["is_evaluable_comparison"]]

    sns.barplot(data=sex_disp, x="model", y="abs_delta_roc_auc", hue="test_dataset", ax=axes[0], palette="tab10")
    axes[0].set_title("|Δ ROC-AUC| (Male vs Female)", weight="bold")
    axes[0].set_ylabel("Absolute ROC-AUC Difference (pp)")
    axes[0].set_xlabel("")
    axes[0].legend_.remove()

    sns.barplot(data=sex_disp, x="model", y="abs_delta_brier", hue="test_dataset", ax=axes[1], palette="tab10")
    axes[1].set_title("|Δ Brier| (Male vs Female)", weight="bold")
    axes[1].set_ylabel("Absolute Brier Difference")
    axes[1].set_xlabel("")
    axes[1].legend_.remove()

    sns.barplot(data=sex_disp, x="model", y="abs_delta_ece", hue="test_dataset", ax=axes[2], palette="tab10")
    axes[2].set_title("|Δ ECE| (Male vs Female)", weight="bold")
    axes[2].set_ylabel("Absolute ECE Difference")
    axes[2].set_xlabel("")
    axes[2].legend(title="Target Cohort", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "4_sex_disparity.png"), dpi=300)
    plt.close()

    # 5. Age Disparity Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    age_disp = disparity_df[disparity_df["comparison_type"].str.contains("Age") & disparity_df["is_evaluable_comparison"]]
    sns.boxplot(data=age_disp, x="comparison_type", y="abs_delta_roc_auc", hue="model", palette="Set1", ax=ax)
    ax.set_title("Age Group Discrimination Disparities (|Δ ROC-AUC|)", fontsize=13, weight="bold")
    ax.set_ylabel("Absolute Difference in ROC-AUC (pp)")
    ax.set_xlabel("Age Group Pair Comparison")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "5_age_disparity.png"), dpi=300)
    plt.close()

    # 6. Missingness Rates Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    miss_pivot = missingness_df.pivot(index="cohort", columns="feature", values="pct_missing")
    sns.heatmap(miss_pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Missingness Rate (%)'}, ax=ax)
    ax.set_title("Raw Pre-Imputation Missingness Rates by Clinical Cohort (Track B)", fontsize=13, weight="bold")
    ax.set_ylabel("Cohort")
    ax.set_xlabel("Harmonized Track B Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(miss_dir, "6_missingness_rates.png"), dpi=300)
    plt.close()

    # 7. Complete-Case vs Imputed Sensitivity Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Filter to cases where test CC was evaluable
    sens_eval = sens_df.dropna(subset=["test_cc_auc"]).copy()

    sns.scatterplot(data=sens_eval, x="primary_auc", y="test_cc_auc", hue="test_dataset", style="model", s=90, ax=axes[0])
    axes[0].plot([50, 100], [50, 100], 'k--', alpha=0.6, label="Identity (1:1)")
    axes[0].set_title("External ROC-AUC: Primary Imputation vs Test Complete-Cases", weight="bold")
    axes[0].set_xlabel("Primary Pipeline ROC-AUC (%)")
    axes[0].set_ylabel("Test Complete-Case ROC-AUC (%)")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(data=sens_eval, x="primary_brier", y="test_cc_brier", hue="test_dataset", style="model", s=90, ax=axes[1])
    axes[1].plot([0.05, 0.45], [0.05, 0.45], 'k--', alpha=0.6, label="Identity (1:1)")
    axes[1].set_title("External Brier Score: Primary Imputation vs Test Complete-Cases", weight="bold")
    axes[1].set_xlabel("Primary Pipeline Brier")
    axes[1].set_ylabel("Test Complete-Case Brier")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(miss_dir, "7_complete_case_sensitivity.png"), dpi=300)
    plt.close()

    # 8. Subgroup SHAP Rank Stability
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_shap = shap_sub_df[shap_sub_df["evaluation_status"] == "Valid"].copy()
    sns.barplot(data=valid_shap, x="test_dataset", y="spearman_rank_correlation", hue="comparison", palette="Paired", ax=ax)
    ax.set_title("Subgroup Explanation Rank Consistency (SHAP Spearman ρ)", fontsize=13, weight="bold")
    ax.set_ylabel("Spearman Rank Correlation (ρ)")
    ax.set_xlabel("Target Cohort")
    ax.set_ylim([0.0, 1.05])
    ax.legend(title="Subgroup Comparison", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "8_subgroup_shap_ranks.png"), dpi=300)
    plt.close()

    print("[SUCCESS] All 8 publication figures generated successfully.")



def main():
    print("==========================================================")
    print("PHASE 8: SUBGROUP ROBUSTNESS, FAIRNESS & MISSINGNESS ENGINE")
    print("==========================================================")

    config = load_config()
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    cohort_names = ["Cleveland", "Hungarian", "Zurich", "VA Long Beach"]
    cohorts = {name: load_harmonized_cohort(name) for name in cohort_names}

    # 1. Load Phase 5 External Predictions
    ext_preds_path = os.path.join(results_dir, "external_predictions.csv")
    if not os.path.exists(ext_preds_path):
        raise FileNotFoundError(f"Missing Phase 5 predictions file: {ext_preds_path}")
    ext_preds = pd.read_csv(ext_preds_path)
    print(f"Loaded {len(ext_preds)} Phase 5 external predictions across 24 experiments.")

    # 2. Subgroup Performance & Calibration
    print("\n--- Running Subgroup Performance & Calibration Analysis ---")
    sub_perf_df, sub_cal_df, sub_disp_df = run_subgroup_evaluation(ext_preds, cohorts)

    sub_perf_path = os.path.join(results_dir, "subgroup_performance.csv")
    sub_cal_path = os.path.join(results_dir, "subgroup_calibration.csv")
    sub_disp_path = os.path.join(results_dir, "subgroup_disparity.csv")

    sub_perf_df.to_csv(sub_perf_path, index=False)
    sub_cal_df.to_csv(sub_cal_path, index=False)
    sub_disp_df.to_csv(sub_disp_path, index=False)
    print(f"Saved {len(sub_perf_df)} subgroup performance records to {sub_perf_path}")
    print(f"Saved {len(sub_cal_df)} subgroup calibration records to {sub_cal_path}")
    print(f"Saved {len(sub_disp_df)} subgroup disparity records to {sub_disp_path}")

    # 3. Missingness Mechanism & Co-Occurrence Analysis
    print("\n--- Running Missingness Mechanism & Pattern Analysis ---")
    miss_patterns_df = run_missingness_mechanism_analysis(cohorts)
    miss_patterns_path = os.path.join(results_dir, "missingness_patterns.csv")
    miss_patterns_df.to_csv(miss_patterns_path, index=False)
    print(f"Saved {len(miss_patterns_df)} missingness pattern records to {miss_patterns_path}")

    # 4. Missingness Sensitivity Experiments
    print("\n--- Running Missingness Sensitivity Experiments ---")
    sens_df = run_missingness_sensitivity_experiments(cohorts, config)
    sens_path = os.path.join(results_dir, "missingness_sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    print(f"Saved {len(sens_df)} missingness sensitivity records to {sens_path}")

    # 5. Subgroup SHAP Stability Analysis
    print("\n--- Running Subgroup SHAP Explanation Stability Analysis ---")
    shap_sub_df = run_subgroup_shap_analysis(cohorts)
    shap_sub_path = os.path.join(results_dir, "subgroup_shap_stability.csv")
    shap_sub_df.to_csv(shap_sub_path, index=False)
    print(f"Saved {len(shap_sub_df)} subgroup SHAP records to {shap_sub_path}")

    # 6. Generate Figures
    print("\n--- Generating Publication Figures ---")
    generate_publication_figures(sub_perf_df, sub_disp_df, miss_patterns_df, sens_df, shap_sub_df)

    # 7. Write Manifest
    manifest = {
        "phase": 8,
        "phase_name": "Subgroup Robustness, Fairness & Missingness Sensitivity",
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": config.get("experiment", {}).get("random_seed", 42),
        "bootstrap_replicates": 1000,

        "minimum_evaluable_cell_size": MIN_EVALUABLE_N,
        "subgroup_definitions": {
            "sex": ["0 (Female)", "1 (Male)"],
            "age_groups": ["<50", "50-64", ">=65"]
        },
        "datasets": cohort_names,
        "features": TRACK_B_10_FEATURES,
        "output_files": [
            "results/subgroup_performance.csv",
            "results/subgroup_calibration.csv",
            "results/subgroup_disparity.csv",
            "results/missingness_sensitivity.csv",
            "results/missingness_patterns.csv",
            "results/subgroup_shap_stability.csv",
            "results/phase8_manifest.json"
        ],
        "figures": [
            "figures/subgroup_robustness/1_subgroup_roc_auc.png",
            "figures/subgroup_robustness/2_subgroup_brier.png",
            "figures/subgroup_robustness/3_subgroup_ece.png",
            "figures/subgroup_robustness/4_sex_disparity.png",
            "figures/subgroup_robustness/5_age_disparity.png",
            "figures/missingness/6_missingness_rates.png",
            "figures/missingness/7_complete_case_sensitivity.png",
            "figures/subgroup_robustness/8_subgroup_shap_ranks.png"
        ]
    }
    manifest_path = os.path.join(results_dir, "phase8_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Phase 8 manifest to {manifest_path}")
    print("\n[SUCCESS] Phase 8 Execution Completed Successfully.")



if __name__ == "__main__":
    main()
