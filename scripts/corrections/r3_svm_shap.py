"""
R3: SVM SHAP Reanalysis Engine (Exact 1024-Coalition Mathematical Vectorization)
================================================================================
Implements:
1. Isolated rerun of ONLY the SVM SHAP analysis across all 6 transfer directions.
2. Exhaustive enumeration of all 1024 coalitions (2^10 = 1024) using exact combinatorial
   Shapley weights: phi = A @ v, where A is the precomputed (10 x 1024) Shapley transformation matrix.
3. 5-seed reproducibility audit (seeds: 42, 100, 2024, 777, 999) for background variance.
4. Near-constant prediction diagnostic & flagging for trained SVM models.
5. Recalculation of SHAP feature importance, ranks, and source-target rank stability
   (Spearman rho, Kendall tau, Top-3 and Top-5 Jaccard overlap).
6. Preservation of old heuristic sampling results with side-by-side comparison.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import shap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CORRECTED_DIR = os.path.join(ROOT_DIR, "corrected_results")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(CORRECTED_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import load_harmonized_cohort, TRACK_B_10_FEATURES
from src.preprocessing import HarmonizedClinicalPreprocessor

TRANSFER_DIRECTIONS = [
    ("Cleveland", "Hungarian"),
    ("Cleveland", "Zurich"),
    ("Cleveland", "VA Long Beach"),
    ("Hungarian", "Cleveland"),
    ("Zurich", "Cleveland"),
    ("VA Long Beach", "Cleveland"),
]

SEEDS_LIST = [42, 100, 2024, 777, 999]


def build_exact_shapley_matrix(n_features: int = 10) -> tuple:
    """
    Builds the binary coalition matrix Z (1024 x 10) and the exact
    combinatorial Shapley operator matrix A (10 x 1024) such that:
    phi = A @ v, where v is the vector of model evaluations on all 1024 coalitions.
    """
    n_subsets = 2**n_features
    Z = np.array([[int(b) for b in format(k, f"0{n_features}b")] for k in range(n_subsets)], dtype=float)

    A = np.zeros((n_features, n_subsets), dtype=float)

    for i in range(n_features):
        mask0 = (Z[:, i] == 0)
        idx0 = np.where(mask0)[0]
        # Flip i-th bit to 1: since bits are big-endian (0 to 9)
        idx1 = idx0 + (2**(n_features - 1 - i))
        # Size of coalition excluding feature i
        sizes = np.sum(Z[idx0], axis=1)
        weights = 1.0 / (float(n_features) * comb(n_features - 1, sizes))

        A[i, idx1] += weights
        A[i, idx0] -= weights

    return Z, A


Z_COALITIONS, A_SHAPLEY = build_exact_shapley_matrix(10)


def build_trained_svm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    """Builds and fits the exact SVM pipeline used in the primary experiment."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=seed))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def check_near_constant_predictions(model, X_eval: pd.DataFrame, threshold_std: float = 0.02) -> dict:
    """Evaluates whether the model produces near-constant predictions across evaluation instances."""
    proba = model.predict_proba(X_eval)[:, 1]
    std_val = float(np.std(proba))
    range_val = float(np.max(proba) - np.min(proba))
    iqr_val = float(stats.iqr(proba))
    mean_val = float(np.mean(proba))

    is_near_constant = (std_val < threshold_std) or (range_val < 0.05)
    return {
        "mean_proba": round(mean_val, 4),
        "std_proba": round(std_val, 4),
        "min_proba": round(float(np.min(proba)), 4),
        "max_proba": round(float(np.max(proba)), 4),
        "range_proba": round(range_val, 4),
        "iqr_proba": round(iqr_val, 4),
        "is_near_constant": bool(is_near_constant),
        "diagnostic_note": "Near-constant risk predictions across evaluation instances" if is_near_constant else "Normal prediction spread"
    }


from joblib import Parallel, delayed

def _compute_single_patient_shapley(model, x: np.ndarray, X_bg: np.ndarray) -> np.ndarray:
    """Computes exact 1024-coalition Shapley attributions for a single patient against background."""
    n_bg = len(X_bg)
    phi_patient = np.zeros(10, dtype=float)
    H = np.empty((1024, 10), dtype=float)
    for bg_j in range(n_bg):
        b = X_bg[bg_j]
        np.multiply(Z_COALITIONS, x, out=H)
        H += (1.0 - Z_COALITIONS) * b
        v = model.predict_proba(H)[:, 1]
        phi_patient += A_SHAPLEY @ v
    return phi_patient / float(n_bg)


def compute_exact_shapley_attributions(model, X_bg: np.ndarray, X_eval: np.ndarray, n_jobs: int = 8) -> np.ndarray:
    """
    Computes exact 1024-coalition Shapley attributions for all instances in X_eval
    averaged over background instances X_bg using multi-threaded execution.
    Returns:
        np.ndarray of shape (len(X_eval), 10)
    """
    res = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_single_patient_shapley)(model, X_eval[i], X_bg)
        for i in range(len(X_eval))
    )
    return np.array(res, dtype=float)


def compute_rank_stability(ranks_src: np.ndarray, ranks_tgt: np.ndarray, top_k_list=(3, 5)) -> dict:
    """Computes Spearman rho, Kendall tau, and Top-K Jaccard overlap between two rankings."""
    rho, rho_p = stats.spearmanr(ranks_src, ranks_tgt)
    tau, tau_p = stats.kendalltau(ranks_src, ranks_tgt)

    res = {
        "spearman_rank_correlation": round(float(rho), 4),
        "spearman_pvalue": round(float(rho_p), 6),
        "kendall_rank_correlation": round(float(tau), 4),
        "kendall_pvalue": round(float(tau_p), 6)
    }

    for k in top_k_list:
        top_src = set(np.where(ranks_src <= k)[0])
        top_tgt = set(np.where(ranks_tgt <= k)[0])
        intersection = len(top_src.intersection(top_tgt))
        union = len(top_src.union(top_tgt))
        jaccard = intersection / union if union > 0 else 1.0
        res[f"top{k}_overlap"] = round(float(jaccard), 4)

    return res


def run_r3_svm_shap_correction():
    print("\n=======================================================")
    print("Executing R3: SVM SHAP Reanalysis (Exact 1024 Coalitions)")
    print("=======================================================")

    cohorts = {}
    for c_name in ["cleveland", "hungarian", "zurich", "va_long_beach"]:
        disp_name = "VA Long Beach" if c_name == "va_long_beach" else c_name.capitalize()
        cohorts[disp_name] = load_harmonized_cohort(
            c_name, track="B10", data_dir=RAW_DATA_DIR, drop_duplicates=True, mask_zero_sentinels=True
        )

    pred_dist_records = []
    shap_importance_records = []
    rank_stability_records = []
    seed_reproducibility_records = []

    for tr_name, te_name in TRANSFER_DIRECTIONS:
        exp_id = f"EXP_PAIRWISE_{tr_name.upper()[:4]}_TO_{te_name.upper()[:4]}_SVM"
        print(f"\nEvaluating SVM SHAP for {tr_name} -> {te_name}...")

        df_tr = cohorts[tr_name]
        df_te = cohorts[te_name]

        X_tr_raw = df_tr[TRACK_B_10_FEATURES].copy()
        y_tr = df_tr["target"].to_numpy()
        X_te_raw = df_te[TRACK_B_10_FEATURES].copy()
        y_te = df_te["target"].to_numpy()

        # 1. Fit Preprocessor strictly on training cohort
        prep = HarmonizedClinicalPreprocessor(
            imputer_strategy="knn",
            n_neighbors=5,
            scale_features=False,
            clamp_negative_oldpeak=False
        )
        X_tr_proc = prep.fit_transform(X_tr_raw)
        X_te_proc = prep.transform(X_te_raw)

        # 2. Fit primary SVM model (seed=42)
        model = build_trained_svm(X_tr_proc, y_tr, seed=42)

        # 3. Check near-constant predictions
        diag_tgt = check_near_constant_predictions(model, X_te_proc)

        pred_dist_records.append({
            "experiment_id": exp_id,
            "train_dataset": tr_name,
            "test_dataset": te_name,
            "target_n": len(df_te),
            "target_mean_proba": diag_tgt["mean_proba"],
            "target_std_proba": diag_tgt["std_proba"],
            "target_min_proba": diag_tgt["min_proba"],
            "target_max_proba": diag_tgt["max_proba"],
            "target_range_proba": diag_tgt["range_proba"],
            "target_iqr_proba": diag_tgt["iqr_proba"],
            "is_target_near_constant": diag_tgt["is_near_constant"],
            "diagnostic_note": diag_tgt["diagnostic_note"]
        })

        # 4. Exact 1024-coalition SHAP evaluation
        # Background sample of 20 points from source cohort
        bg_sample_42 = shap.sample(X_tr_proc, min(len(X_tr_proc), 20), random_state=42).to_numpy()

        # Use full evaluation sets
        X_tr_eval = X_tr_proc.to_numpy()
        X_te_eval = X_te_proc.to_numpy()

        print("  Computing exact 1024-coalition SHAP on source cohort...")
        shap_src = compute_exact_shapley_attributions(model, bg_sample_42, X_tr_eval)
        print("  Computing exact 1024-coalition SHAP on target cohort...")
        shap_tgt = compute_exact_shapley_attributions(model, bg_sample_42, X_te_eval)

        mean_abs_src = np.mean(np.abs(shap_src), axis=0)
        mean_abs_tgt = np.mean(np.abs(shap_tgt), axis=0)

        ranks_src = stats.rankdata(-mean_abs_src, method="min")
        ranks_tgt = stats.rankdata(-mean_abs_tgt, method="min")

        for i, feat in enumerate(TRACK_B_10_FEATURES):
            shap_importance_records.append({
                "experiment_id": exp_id,
                "train_dataset": tr_name,
                "test_dataset": te_name,
                "feature": feat,
                "source_mean_abs_shap": round(float(mean_abs_src[i]), 6),
                "source_feature_rank": int(ranks_src[i]),
                "target_mean_abs_shap": round(float(mean_abs_tgt[i]), 6),
                "target_feature_rank": int(ranks_tgt[i]),
                "delta_abs_shap": round(float(mean_abs_tgt[i] - mean_abs_src[i]), 6),
                "rank_displacement": int(abs(ranks_tgt[i] - ranks_src[i]))
            })

        stability = compute_rank_stability(ranks_src, ranks_tgt, top_k_list=(3, 5))

        old_shap_csv = os.path.join(RESULTS_DIR, "shap_rank_stability.csv")
        old_rho, old_tau = np.nan, np.nan
        if os.path.exists(old_shap_csv):
            old_df = pd.read_csv(old_shap_csv)
            old_match = old_df[(old_df["train_dataset"] == tr_name) & (old_df["test_dataset"] == te_name) & (old_df["model"] == "SVM")]
            if len(old_match) > 0:
                old_rho = float(old_match["spearman_rank_correlation"].iloc[0])
                old_tau = float(old_match["kendall_rank_correlation"].iloc[0])

        rank_stability_records.append({
            "experiment_id": exp_id,
            "train_dataset": tr_name,
            "test_dataset": te_name,
            "model": "SVM",
            "coalition_method": "Exact_1024_Coalitions",
            "old_heuristic_spearman_rho": old_rho,
            "corrected_spearman_rho": stability["spearman_rank_correlation"],
            "spearman_delta": round(stability["spearman_rank_correlation"] - old_rho, 4) if not np.isnan(old_rho) else np.nan,
            "old_heuristic_kendall_tau": old_tau,
            "corrected_kendall_tau": stability["kendall_rank_correlation"],
            "corrected_top3_overlap": stability["top3_overlap"],
            "corrected_top5_overlap": stability["top5_overlap"],
            "is_near_constant": diag_tgt["is_near_constant"]
        })

        # 5. 5-Seed Reproducibility Check across background seeds
        print("  Running 5-seed background reproducibility audit...")
        seed_shap_ranks = {}
        sub_eval = X_te_eval[:min(len(X_te_eval), 50)]
        for s in SEEDS_LIST:
            bg_s = shap.sample(X_tr_proc, min(len(X_tr_proc), 20), random_state=s).to_numpy()
            shap_s = compute_exact_shapley_attributions(model, bg_s, sub_eval)
            m_abs_s = np.mean(np.abs(shap_s), axis=0)
            r_s = stats.rankdata(-m_abs_s, method="min")
            seed_shap_ranks[s] = r_s

        rhos_across_seeds = []
        for i in range(len(SEEDS_LIST)):
            for j in range(i + 1, len(SEEDS_LIST)):
                r_ij, _ = stats.spearmanr(seed_shap_ranks[SEEDS_LIST[i]], seed_shap_ranks[SEEDS_LIST[j]])
                rhos_across_seeds.append(r_ij)

        seed_reproducibility_records.append({
            "experiment_id": exp_id,
            "train_dataset": tr_name,
            "test_dataset": te_name,
            "model": "SVM",
            "evaluated_seeds": str(SEEDS_LIST),
            "min_cross_seed_spearman_rho": round(float(np.min(rhos_across_seeds)), 4),
            "mean_cross_seed_spearman_rho": round(float(np.mean(rhos_across_seeds)), 4),
            "max_cross_seed_spearman_rho": round(float(np.max(rhos_across_seeds)), 4),
            "is_reproducible": bool(np.mean(rhos_across_seeds) >= 0.85)
        })

    pd.DataFrame(pred_dist_records).to_csv(os.path.join(CORRECTED_DIR, "svm_model_prediction_distributions.csv"), index=False)
    pd.DataFrame(shap_importance_records).to_csv(os.path.join(CORRECTED_DIR, "svm_shap_exact_coalitions.csv"), index=False)
    pd.DataFrame(rank_stability_records).to_csv(os.path.join(CORRECTED_DIR, "svm_shap_rank_stability_corrected.csv"), index=False)
    pd.DataFrame(seed_reproducibility_records).to_csv(os.path.join(CORRECTED_DIR, "svm_shap_seed_reproducibility.csv"), index=False)

    print("\n[OK] R3 Execution Complete. Saved 4 corrected SVM SHAP tables to corrected_results/.")


if __name__ == "__main__":
    run_r3_svm_shap_correction()
