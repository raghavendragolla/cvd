"""
R1 & R2: Calibration & Internal Validation Correction Engine
============================================================
Implements:
1. Standard clinical calibration-in-the-large (intercept with slope fixed to 1 via offset)
   using root finding on the exact score equation: sum(y_i - sigma(alpha + logit(p_i))) = 0.
2. Separate calibration slope estimation via logistic regression: logit(y) ~ alpha' + beta * logit(p).
3. Recomputation of internal validation metrics from pooled out-of-fold (OOF) predictions.
4. Stratified bootstrap 95% confidence intervals (B=1000, seed=42).
5. Comprehensive comparison of new pooled-OOF metrics vs. old fold-mean metrics.
6. Recomputation of external validation calibration metrics on saved external predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, root_scalar
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CORRECTED_DIR = os.path.join(ROOT_DIR, "corrected_results")
os.makedirs(CORRECTED_DIR, exist_ok=True)


def compute_fast_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Vectorized ECE calculation."""
    n = len(y_true)
    if n == 0:
        return 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba, bin_boundaries) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (bin_idx == b)
        b_size = np.sum(mask)
        if b_size > 0:
            b_acc = np.mean(y_true[mask])
            b_conf = np.mean(y_proba[mask])
            ece += (b_size / n) * abs(b_acc - b_conf)
    return float(ece)


def fast_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Fast Mann-Whitney U ROC-AUC in pure numpy."""
    pos = y_proba[y_true == 1]
    neg = y_proba[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = np.sum(ranks[:len(pos)]) - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg))) * 100.0


def compute_standard_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-6) -> dict:
    """
    Computes gold-standard clinical calibration metrics:
    1. Calibration Intercept (alpha, slope fixed to 1):
       Solves sum(y_i - sigma(alpha + logit(p_i))) = 0.
    2. Calibration Slope (beta):
       Fits logit(P(Y=1)) = alpha' + beta * logit(p).
    """
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.clip(np.asarray(y_proba, dtype=float), eps, 1.0 - eps)

    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    n_tot = len(y_arr)

    if n_pos == 0 or n_neg == 0:
        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_status": "Degenerate (single class)"
        }

    logits = np.log(p_arr / (1.0 - p_arr))

    # 1. Intercept with slope=1 via root_scalar (Brentq method on [-25, 25])
    def score_func(alpha):
        z = np.clip(alpha + logits, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.sum(y_arr - p)

    try:
        res = root_scalar(score_func, bracket=[-25.0, 25.0], method="brentq")
        alpha = float(res.root)
    except Exception:
        alpha = 0.0

    # 2. Calibration Slope via L-BFGS-B with analytical gradient
    def fun_grad(th):
        arg = np.clip(th[0] + th[1] * logits, -35.0, 35.0)
        exp_arg = np.exp(arg)
        ph = exp_arg / (1.0 + exp_arg)
        # Log-loss with mild L2 regularization (1e-4) on slope
        val = np.sum(np.log1p(exp_arg) - y_arr * arg) + 0.5 * 1e-4 * (th[1] - 1.0)**2
        grad = np.array([
            np.sum(ph - y_arr),
            np.sum((ph - y_arr) * logits) + 1e-4 * (th[1] - 1.0)
        ])
        return val, grad

    res = minimize(fun_grad, [0.0, 1.0], jac=True, method="L-BFGS-B", bounds=[(-20.0, 20.0), (-10.0, 15.0)])
    cal_slope = float(res.x[1])

    return {
        "calibration_intercept": round(float(alpha), 4),
        "calibration_slope": round(float(cal_slope), 4),
        "calibration_status": "Valid"
    }


def evaluate_prediction_set(y_true, y_proba, threshold: float = 0.50) -> dict:
    """Calculates complete metric set for a prediction vector."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(y_true) > 0 and y_true[0] == 0:
            tn = len(y_true)
        else:
            tp = len(y_true)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    acc = float(accuracy_score(y_true, y_pred)) * 100.0

    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true, y_proba)) * 100.0
        pr_auc = float(average_precision_score(y_true, y_proba)) * 100.0
        mcc = float(matthews_corrcoef(y_true, y_pred))
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        mcc = np.nan

    sens = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
    spec = float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)) * 100.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0

    brier = float(brier_score_loss(y_true, y_proba))
    ece = compute_fast_ece(y_true, y_proba, n_bins=10)

    cal_res = compute_standard_calibration_metrics(y_true, y_proba)

    return {
        "n_samples": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence_pct": round(float(n_pos / len(y_true)) * 100.0, 2),
        "accuracy": round(acc, 2),
        "roc_auc": round(roc_auc, 2) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": round(pr_auc, 2) if not np.isnan(pr_auc) else np.nan,
        "sensitivity": round(sens, 2),
        "specificity": round(spec, 2),
        "precision": round(prec, 2),
        "f1_score": round(f1, 2),
        "mcc": round(mcc, 4) if not np.isnan(mcc) else np.nan,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "calibration_intercept": cal_res["calibration_intercept"],
        "calibration_slope": cal_res["calibration_slope"],
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def compute_stratified_bootstrap_cis(
    y_true, y_proba, threshold: float = 0.50, n_boot: int = 1000, seed: int = 42
) -> dict:
    """Fast bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(pos_idx) < 5 or len(neg_idx) < 5:
        return {}

    boot_acc = np.empty(n_boot)
    boot_auc = np.empty(n_boot)
    boot_brier = np.empty(n_boot)
    boot_ece = np.empty(n_boot)
    boot_cal_int = np.empty(n_boot)
    boot_cal_slope = np.empty(n_boot)

    valid_count = 0
    for i in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        b_y = y_arr[b_idx]
        b_p = p_arr[b_idx]

        auc_val = fast_auc(b_y, b_p)
        if np.isnan(auc_val):
            continue

        brier_val = float(np.mean((b_y - b_p)**2))
        ece_val = compute_fast_ece(b_y, b_p, n_bins=10)
        cal_res = compute_standard_calibration_metrics(b_y, b_p)
        acc_val = float(np.mean((b_p >= threshold) == b_y)) * 100.0

        boot_acc[valid_count] = acc_val
        boot_auc[valid_count] = auc_val
        boot_brier[valid_count] = brier_val
        boot_ece[valid_count] = ece_val
        boot_cal_int[valid_count] = cal_res["calibration_intercept"]
        boot_cal_slope[valid_count] = cal_res["calibration_slope"]
        valid_count += 1

    if valid_count < 50:
        return {}

    return {
        "accuracy_ci_lower": round(float(np.percentile(boot_acc[:valid_count], 2.5)), 2),
        "accuracy_ci_upper": round(float(np.percentile(boot_acc[:valid_count], 97.5)), 2),
        "roc_auc_ci_lower": round(float(np.percentile(boot_auc[:valid_count], 2.5)), 2),
        "roc_auc_ci_upper": round(float(np.percentile(boot_auc[:valid_count], 97.5)), 2),
        "brier_ci_lower": round(float(np.percentile(boot_brier[:valid_count], 2.5)), 4),
        "brier_ci_upper": round(float(np.percentile(boot_brier[:valid_count], 97.5)), 4),
        "ece_ci_lower": round(float(np.percentile(boot_ece[:valid_count], 2.5)), 4),
        "ece_ci_upper": round(float(np.percentile(boot_ece[:valid_count], 97.5)), 4),
        "cal_intercept_ci_lower": round(float(np.percentile(boot_cal_int[:valid_count], 2.5)), 4),
        "cal_intercept_ci_upper": round(float(np.percentile(boot_cal_int[:valid_count], 97.5)), 4),
        "cal_slope_ci_lower": round(float(np.percentile(boot_cal_slope[:valid_count], 2.5)), 4),
        "cal_slope_ci_upper": round(float(np.percentile(boot_cal_slope[:valid_count], 97.5)), 4),
    }


def run_r2_internal_validation_correction():
    """Executes R2: Pooled Out-Of-Fold internal validation metrics calculation."""
    print("\n=======================================================")
    print("Executing R2: Internal Validation Pooled-OOF Correction")
    print("=======================================================")

    oof_csv = os.path.join(RESULTS_DIR, "internal_oof_predictions.csv")
    if not os.path.exists(oof_csv):
        raise FileNotFoundError(f"Missing internal OOF predictions file: {oof_csv}")

    oof_df = pd.read_csv(oof_csv)

    grouped = oof_df.groupby(["cohort", "track", "model"])
    pooled_records = []

    for (cohort, track, model), grp in grouped:
        y_true = grp["true_label"].to_numpy()
        y_proba = grp["predicted_probability"].to_numpy()

        metrics = evaluate_prediction_set(y_true, y_proba, threshold=0.50)
        ci_metrics = compute_stratified_bootstrap_cis(y_true, y_proba, threshold=0.50, n_boot=1000, seed=42)

        rec = {
            "cohort": cohort,
            "track": track,
            "model": model,
            "evaluation_type": "Pooled_OOF",
            **metrics,
            **ci_metrics
        }
        pooled_records.append(rec)

    pooled_df = pd.DataFrame(pooled_records)
    out_pooled_path = os.path.join(CORRECTED_DIR, "internal_validation_pooled_oof.csv")
    pooled_df.to_csv(out_pooled_path, index=False)
    print(f"[OK] Saved corrected pooled-OOF internal validation to: {out_pooled_path}")

    # Compare with old fold-mean summary
    old_summary_path = os.path.join(RESULTS_DIR, "internal_validation_summary.csv")
    old_sum_df = pd.read_csv(old_summary_path)

    comparison_rows = []

    for _, prow in pooled_df.iterrows():
        c = prow["cohort"]
        t = prow["track"]
        m = prow["model"]

        sub_old = old_sum_df[(old_sum_df["cohort"] == c) & (old_sum_df["track"] == t) & (old_sum_df["model"] == m)]
        old_metric_dict = dict(zip(sub_old["metric"], sub_old["mean"]))
        old_sd_dict = dict(zip(sub_old["metric"], sub_old["std"]))

        for metric_name in [
            "accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity",
            "precision", "f1_score", "mcc", "brier_score", "ece",
            "calibration_intercept", "calibration_slope"
        ]:
            old_val = old_metric_dict.get(metric_name, np.nan)
            old_sd = old_sd_dict.get(metric_name, np.nan)
            new_val = prow.get(metric_name, np.nan)

            ci_l_key = f"{metric_name}_ci_lower" if f"{metric_name}_ci_lower" in prow else (
                "cal_intercept_ci_lower" if metric_name == "calibration_intercept" else (
                    "cal_slope_ci_lower" if metric_name == "calibration_slope" else (
                        "brier_ci_lower" if metric_name == "brier_score" else np.nan
                    )
                )
            )
            ci_u_key = f"{metric_name}_ci_upper" if f"{metric_name}_ci_upper" in prow else (
                "cal_intercept_ci_upper" if metric_name == "calibration_intercept" else (
                    "cal_slope_ci_upper" if metric_name == "calibration_slope" else (
                        "brier_ci_upper" if metric_name == "brier_score" else np.nan
                    )
                )
            )
            new_ci_l = prow.get(ci_l_key, np.nan)
            new_ci_u = prow.get(ci_u_key, np.nan)

            delta = round(new_val - old_val, 4) if (not np.isnan(new_val) and not np.isnan(old_val)) else np.nan

            comparison_rows.append({
                "cohort": c,
                "track": t,
                "model": m,
                "metric": metric_name,
                "old_fold_mean": old_val,
                "old_fold_sd": old_sd,
                "new_pooled_oof": new_val,
                "new_ci_lower": new_ci_l,
                "new_ci_upper": new_ci_u,
                "absolute_delta": delta,
                "evaluation_method_contrast": "Pooled-OOF vs. Unweighted Fold-Mean"
            })

    comp_df = pd.DataFrame(comparison_rows)
    out_comp_path = os.path.join(CORRECTED_DIR, "internal_validation_comparison.csv")
    comp_df.to_csv(out_comp_path, index=False)
    print(f"[OK] Saved internal validation comparison to: {out_comp_path}")

    return pooled_df, comp_df


def run_r1_calibration_correction():
    """Executes R1: Calibration correction on internal and external validation predictions."""
    print("\n=======================================================")
    print("Executing R1: Calibration Correction (Offset Intercept & Slope)")
    print("=======================================================")

    ext_preds_path = os.path.join(RESULTS_DIR, "external_predictions.csv")
    df_ext_preds = pd.read_csv(ext_preds_path)

    ext_records = []
    grouped = df_ext_preds.groupby("experiment_id")

    for exp_id, grp in grouped:
        tr_dataset = grp["train_dataset"].iloc[0]
        te_dataset = grp["test_dataset"].iloc[0]
        model_name = grp["model"].iloc[0]
        y_true = grp["true_label"].to_numpy()
        y_proba = grp["predicted_probability"].to_numpy()

        cal_res = compute_standard_calibration_metrics(y_true, y_proba)
        ece_val = compute_fast_ece(y_true, y_proba, n_bins=10)
        brier_val = float(brier_score_loss(y_true, y_proba))

        ci_metrics = compute_stratified_bootstrap_cis(y_true, y_proba, threshold=0.50, n_boot=1000, seed=42)

        ext_records.append({
            "experiment_id": exp_id,
            "train_dataset": tr_dataset,
            "test_dataset": te_dataset,
            "model": model_name,
            "n_samples": len(y_true),
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "prevalence_pct": round(float(np.mean(y_true)) * 100.0, 2),
            "external_brier_score": round(brier_val, 4),
            "external_ece": round(ece_val, 4),
            "external_calibration_intercept": cal_res["calibration_intercept"],
            "external_calibration_slope": cal_res["calibration_slope"],
            "cal_intercept_ci_lower": ci_metrics.get("cal_intercept_ci_lower", np.nan),
            "cal_intercept_ci_upper": ci_metrics.get("cal_intercept_ci_upper", np.nan),
            "cal_slope_ci_lower": ci_metrics.get("cal_slope_ci_lower", np.nan),
            "cal_slope_ci_upper": ci_metrics.get("cal_slope_ci_upper", np.nan),
            "brier_ci_lower": ci_metrics.get("brier_ci_lower", np.nan),
            "brier_ci_upper": ci_metrics.get("brier_ci_upper", np.nan),
            "ece_ci_lower": ci_metrics.get("ece_ci_lower", np.nan),
            "ece_ci_upper": ci_metrics.get("ece_ci_upper", np.nan)
        })

    ext_cal_df = pd.DataFrame(ext_records)
    out_ext_cal = os.path.join(CORRECTED_DIR, "calibration_corrected_external.csv")
    ext_cal_df.to_csv(out_ext_cal, index=False)
    print(f"[OK] Saved corrected external calibration to: {out_ext_cal}")

    return ext_cal_df


if __name__ == "__main__":
    run_r2_internal_validation_correction()
    run_r1_calibration_correction()
