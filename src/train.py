"""
Unified Model Training and Cleveland-Cohort Benchmarking,
Stratified 5-Fold CV, GridSearchCV Hyperparameter Optimization across 10 Models,
Model Selection, Probability Calibration (Platt & Isotonic), and Serialization Pipeline.
"""

import os
import sys

# Safe UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

from src.dataset import (
    load_cleveland_track_a,
    create_sample_patient_cohort,
    TRACK_A_FEATURES,
    FEATURE_COLUMNS,
)
from src.preprocessing import (
    prepare_train_test_split,
    compare_imbalance_strategies
)
from src.models import (
    get_model_zoo,
    get_model_grid_search_configs,
    build_super_ensemble,
    build_calibrated_model
)
from src.evaluate import (
    evaluate_classifier,
    compute_calibration_data,
    compute_expected_calibration_error,
    compute_decision_curve_analysis
)


def train_and_benchmark(
    data_path: str = None,
    output_dir: str = "models",
    data_dir: str = "data",
    random_state: int = 42
):
    """
    Executes end-to-end training on the UCI Cleveland cohort (303 records):
    1. Leakage-safe train/test split and train-fitted imputation
    2. Stratified 5-Fold CV + GridSearchCV hyperparameter optimization for 10 ML models
    3. Performance comparison (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    4. Automated Best Model Selection & Serialization
    5. Probability calibration (Platt & Isotonic)
    6. Serialization of reproducible Cleveland-cohort artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    
    print("=" * 80)
    print("🫀 CARDIOPULSE CLINICAL AI SUITE: 10-MODEL GRIDSEARCHCV & VALIDATION PIPELINE")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # PHASE 1 & 2: CLEVELAND DATA ACQUISITION & LEAKAGE-SAFE PREPROCESSING
    # --------------------------------------------------------------------------
    print("\n📂 [PHASE 1 & 2] Ingesting the UCI Cleveland cohort (303 records)...")
    
    cleveland_df = load_cleveland_track_a(os.path.join(data_dir, "raw"))
    print(f"  ✅ UCI Cleveland Cohort (13 Features): {cleveland_df.shape[0]} patients")
    
    # Generate curated sample patient cohort for testing
    sample_patients_path = os.path.join(data_dir, "raw", "sample_patients.csv")
    create_sample_patient_cohort(sample_patients_path)
    
    # Leakage-Safe Splits (Split first, imputer fit on train only)
    X_train_a, X_test_a, y_train_a, y_test_a = prepare_train_test_split(
        cleveland_df, feature_set=TRACK_A_FEATURES, test_size=0.2, random_state=random_state
    )
    print(f"  ✅ Stratified Split Complete: Train={len(X_train_a)} patients, Test={len(X_test_a)} patients (Holdout 20%)")

    processed_cohort = pd.concat(
        [X_train_a.assign(target=y_train_a), X_test_a.assign(target=y_test_a)],
        ignore_index=True,
    )
    processed_cohort.to_csv(
        os.path.join(data_dir, "processed", "cleveland_303_prepared.csv"), index=False
    )
    
    # Imbalance Strategy Audit
    print("\n🔬 [PHASE 2] Auditing Class Imbalance Strategies & Synthetic Physiological Validity...")
    imbalance_audit = compare_imbalance_strategies(X_train_a, y_train_a, random_state=random_state)
    with open(os.path.join(output_dir, "imbalance_audit.json"), "w", encoding="utf-8") as f:
        json.dump(imbalance_audit, f, indent=4)
    print(f"  ⚖️ Imbalance Audit Complete. SMOTE Clinical Violation Rate: {imbalance_audit['smote_audit']['audit_results']['clinical_violation_rate_pct']}%")

    # --------------------------------------------------------------------------
    # PHASE 3: STRATIFIED 5-FOLD CV & GRIDSEARCHCV OPTIMIZATION (10 ML MODELS)
    # --------------------------------------------------------------------------
    print("\n🚀 [PHASE 3] Stratified 5-Fold Cross-Validation & GridSearchCV Hyperparameter Tuning...")
    
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_configs = get_model_grid_search_configs(random_state=random_state, use_class_weights=True)
    
    tuned_estimators = {}
    best_params_map = {}
    cv_scores_map = {}
    
    # Run GridSearchCV across 9 base models
    for m_name, cfg in grid_configs.items():
        print(f"  🔍 Optimizing: {m_name} (Stratified 5-Fold CV)...")
        grid_search = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cv_kfold,
            scoring="roc_auc",
            refit=True,
            # A single worker keeps the small-cohort benchmark stable on Windows.
            n_jobs=1
        )
        grid_search.fit(X_train_a, y_train_a)
        
        best_est = grid_search.best_estimator_
        tuned_estimators[m_name] = best_est
        best_params_map[m_name] = grid_search.best_params_
        cv_scores_map[m_name] = {
            "mean_auc": round(float(grid_search.best_score_) * 100, 2),
            "std_auc": round(float(grid_search.cv_results_["std_test_score"][grid_search.best_index_]) * 100, 2)
        }
        print(f"     -> Best CV ROC-AUC: {cv_scores_map[m_name]['mean_auc']}% (±{cv_scores_map[m_name]['std_auc']}%)")
        
    # Model 10: Super-Ensemble Soft Voting Meta-Classifier
    ensemble_name = "🫀 Super-Ensemble (Top Precision)"
    print(f"\n  🔍 Assembling Model 10: {ensemble_name} from top tuned estimators...")
    super_ensemble = build_super_ensemble(tuned_estimators)
    super_ensemble_cv = cross_val_score(super_ensemble, X_train_a, y_train_a, cv=cv_kfold, scoring="roc_auc")
    super_ensemble.fit(X_train_a, y_train_a)
    
    tuned_estimators[ensemble_name] = super_ensemble
    best_params_map[ensemble_name] = {"voting": "soft", "weights": "equal"}
    cv_scores_map[ensemble_name] = {
        "mean_auc": round(float(np.mean(super_ensemble_cv)) * 100, 2),
        "std_auc": round(float(np.std(super_ensemble_cv)) * 100, 2)
    }
    print(f"     -> Best CV ROC-AUC: {cv_scores_map[ensemble_name]['mean_auc']}% (±{cv_scores_map[ensemble_name]['std_auc']}%)")

    # --------------------------------------------------------------------------
    # PHASE 4: HOLDOUT PERFORMANCE COMPARISON ACROSS ALL 10 MODELS
    # --------------------------------------------------------------------------
    print("\n📊 [PHASE 4] Comprehensive Evaluation on Holdout Test Set (n=61)...")
    
    metrics_summary = {}
    roc_curves_data = {}
    calibration_data = {}
    calibration_comparison = {}
    dca_data = {}
    
    primary_test_proba = None
    best_calibrated_xgb = None
    
    for model_name, model in tuned_estimators.items():
        y_pred = model.predict(X_test_a)
        y_proba = model.predict_proba(X_test_a)[:, 1]
        
        eval_metrics = evaluate_classifier(y_test_a, y_pred, y_proba)
        eval_metrics["cv_roc_auc_mean"] = cv_scores_map[model_name]["mean_auc"]
        eval_metrics["cv_roc_auc_std"] = cv_scores_map[model_name]["std_auc"]
        eval_metrics["best_params"] = best_params_map[model_name]
        
        # Calibration Evaluation for XGBoost and Primary Ensemble
        if "XGBoost" in model_name or "Super-Ensemble" in model_name:
            platt_model = build_calibrated_model(model, method="sigmoid", cv=5)
            platt_model.fit(X_train_a, y_train_a)
            platt_proba = platt_model.predict_proba(X_test_a)[:, 1]
            platt_pred = platt_model.predict(X_test_a)
            platt_metrics = evaluate_classifier(y_test_a, platt_pred, platt_proba)
            
            iso_model = build_calibrated_model(model, method="isotonic", cv=5)
            iso_model.fit(X_train_a, y_train_a)
            iso_proba = iso_model.predict_proba(X_test_a)[:, 1]
            iso_pred = iso_model.predict(X_test_a)
            iso_metrics = evaluate_classifier(y_test_a, iso_pred, iso_proba)
            
            calibration_comparison[model_name] = {
                "uncalibrated": {"brier": eval_metrics["brier_score"], "ece": eval_metrics["ece"]},
                "platt_sigmoid": {"brier": platt_metrics["brier_score"], "ece": platt_metrics["ece"]},
                "isotonic": {"brier": iso_metrics["brier_score"], "ece": iso_metrics["ece"]}
            }
            
            if "XGBoost" in model_name:
                best_calibrated_xgb = platt_model
                joblib.dump(platt_model, os.path.join(output_dir, "calibrated_xgb.joblib"))
                
        if primary_test_proba is None:
            primary_test_proba = y_proba.tolist()
            dca_data["y_true"] = y_test_a.tolist()
            dca_data["y_proba"] = primary_test_proba
            
        fpr, tpr, _ = roc_curve(y_test_a, y_proba)
        roc_curves_data[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": eval_metrics["roc_auc"]
        }
        
        calibration_data[model_name] = compute_calibration_data(y_test_a, y_proba, n_bins=10)
        metrics_summary[model_name] = eval_metrics
        
        # Serialize model safely
        safe_fname = "".join([c if c.isalnum() else "_" for c in model_name.split("(")[0]]).strip("_").lower() + ".joblib"
        joblib.dump(model, os.path.join(output_dir, safe_fname))
        
        if "XGBoost" in model_name:
            joblib.dump(model, os.path.join(output_dir, "xgb_model.joblib"))
            joblib.dump(model, os.path.join(output_dir, "heart_disease_model.joblib"))

    # --------------------------------------------------------------------------
    # SELECT BEST MODEL USING CROSS-VALIDATION ONLY
    # --------------------------------------------------------------------------
    # The holdout set remains for reporting only; do not use it for model selection.
    ranked_models = sorted(
        cv_scores_map.items(),
        key=lambda item: (item[1]["mean_auc"], -item[1]["std_auc"]),
        reverse=True
    )
    best_model_name, _ = ranked_models[0]
    best_model_metrics = metrics_summary[best_model_name]
    best_model_obj = tuned_estimators[best_model_name]
    
    joblib.dump(best_model_obj, os.path.join(output_dir, "best_model.joblib"))
    
    best_model_metadata = {
        "best_model_name": best_model_name,
        "test_roc_auc": best_model_metrics["roc_auc"],
        "test_accuracy": best_model_metrics["accuracy"],
        "test_f1_score": best_model_metrics["f1_score"],
        "test_sensitivity": best_model_metrics["sensitivity"],
        "test_specificity": best_model_metrics["specificity"],
        "confusion_matrix": {
            "tp": best_model_metrics["tp"],
            "tn": best_model_metrics["tn"],
            "fp": best_model_metrics["fp"],
            "fn": best_model_metrics["fn"]
        },
        "best_hyperparameters": best_model_metrics.get("best_params", {}),
        "cv_roc_auc_5fold": f"{best_model_metrics['cv_roc_auc_mean']}% ± {best_model_metrics['cv_roc_auc_std']}%"
    }
    
    with open(os.path.join(output_dir, "best_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(best_model_metadata, f, indent=4)
        
    print(f"\n🏆 BEST PERFORMING MODEL SELECTED: {best_model_name}")
    print(f"   • Test ROC-AUC: {best_model_metrics['roc_auc']}% | Test Accuracy: {best_model_metrics['accuracy']}% | F1: {best_model_metrics['f1_score']}%")
    print(f"   • Confusion Matrix: TP={best_model_metrics['tp']}, TN={best_model_metrics['tn']}, FP={best_model_metrics['fp']}, FN={best_model_metrics['fn']}")
    print(f"   • Serialized to: {os.path.join(output_dir, 'best_model.joblib')}")

    # Serialize Cleveland 13-feature schema.
    joblib.dump(TRACK_A_FEATURES, os.path.join(output_dir, "features.joblib"))
    joblib.dump(TRACK_A_FEATURES, os.path.join(output_dir, "track_a_features.joblib"))

    # Save all telemetry JSON files
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
        
    with open(os.path.join(output_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
        
    with open(os.path.join(output_dir, "roc_curves.json"), "w", encoding="utf-8") as f:
        json.dump(roc_curves_data, f, indent=4)
        
    with open(os.path.join(output_dir, "calibration_data.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=4)
        
    with open(os.path.join(output_dir, "calibration_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_comparison, f, indent=4)
        
    with open(os.path.join(output_dir, "dca_data.json"), "w", encoding="utf-8") as f:
        json.dump(dca_data, f, indent=4)
        
    training_manifest = {
        "cohort": "UCI Cleveland",
        "records": int(len(cleveland_df)),
        "features": TRACK_A_FEATURES,
        "holdout_size": int(len(X_test_a)),
        "selection_method": "stratified 5-fold cross-validation on training data",
    }
    with open(os.path.join(output_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=4)
        
    print("\n" + "=" * 80)
    print(f"🏆 Research Benchmark complete! All artifacts saved to: {output_dir}/")
    print("=" * 80)
    return metrics_summary


if __name__ == "__main__":
    train_and_benchmark()
