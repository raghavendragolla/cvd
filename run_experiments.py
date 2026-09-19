"""
Master Experiment Runner & Scientific Validation Suite for CardioPulse Clinical AI Framework.

Executes the End-to-End Clinical AI & Dual-XAI Pipeline:
1. UCI Cleveland Dataset Ingestion
2. Exploratory Data Analysis & Diagnostic Telemetry (EDA)
3. Leakage-Safe Preprocessing & Stratified 80/20 Train/Test Split
4. Stratified 5-Fold CV & GridSearchCV Hyperparameter Optimization (10 ML Models)
5. Performance Comparison (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix)
6. Automated Best Model Selection
7. SHAP + LIME Global Explanations (Beeswarm, Mean |SHAP|, Global LIME profile)
8. SHAP + LIME Individual Patient Explanations (Waterfall, Rules)
9. SHAP vs LIME Concordance & Discordance Analysis (Spearman rho, Jaccard, Concordance Index C_i)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Safe UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dataset import (
    load_cleveland_dataset,
    load_cleveland_track_a,
    create_sample_patient_cohort,
    TRACK_A_FEATURES,
    FEATURE_DESCRIPTIONS
)
from src.eda import perform_data_analysis
from src.preprocessing import (
    clean_and_prepare_data,
    prepare_train_test_split,
    compare_imbalance_strategies
)
from src.train import train_and_benchmark
from src.explainability import ClinicalExplainabilitySuite
from src.evaluate import evaluate_classifier


def run_full_research_pipeline():
    print("=" * 85)
    print("🫀 CARDIOPULSE: CLINICAL MACHINE LEARNING & DUAL-XAI RESEARCH PIPELINE")
    print("=" * 85)
    
    output_dir = "models"
    data_dir = "data"
    assets_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    
    # --------------------------------------------------------------------------
    # STEP 1: UCI CLEVELAND DATASET INGESTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📥 STEP 1: INGESTING UCI CLEVELAND DATASET (13 BIOMARKERS)")
    print("=" * 80)
    raw_cleveland_df = load_cleveland_dataset(os.path.join(data_dir, "raw"))
    print(f"  ✅ Raw UCI Cleveland Ingested: {raw_cleveland_df.shape[0]} patients, {raw_cleveland_df.shape[1]} columns")
    print(f"  Biomarkers: {', '.join(TRACK_A_FEATURES)}")

    # --------------------------------------------------------------------------
    # STEP 2: DATA ANALYSIS (EDA) & COHORT DIAGNOSTICS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 STEP 2: CLINICAL DATA ANALYSIS & EXPLORATORY PROFILING (EDA)")
    print("=" * 80)
    eda_summary = perform_data_analysis(raw_cleveland_df, output_dir=os.path.join(data_dir, "processed"))

    # --------------------------------------------------------------------------
    # STEP 3: LEAKAGE-SAFE PREPROCESSING & STRATIFIED TRAIN/TEST SPLIT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🛡️ STEP 3: LEAKAGE-SAFE PREPROCESSING & STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 80)
    clean_cleveland_df = clean_and_prepare_data(raw_cleveland_df, feature_set=TRACK_A_FEATURES)
    clean_cleveland_df.to_csv(os.path.join(data_dir, "processed", "track_a_cleveland_clean.csv"), index=False)
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        clean_cleveland_df,
        feature_set=TRACK_A_FEATURES,
        test_size=0.2,
        random_state=42,
        impute_missing=True
    )
    print(f"  ✅ Stratified 80/20 Partition: Train N={len(X_train)} | Holdout Test N={len(X_test)}")
    print(f"  ✅ Leakage-safe guarantee: KNNImputer & Scalers fitted strictly on training cohort.")

    # --------------------------------------------------------------------------
    # STEP 4: STRATIFIED 5-FOLD CV & GRIDSEARCHCV OPTIMIZATION (10 ML MODELS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("⚡ STEP 4: STRATIFIED 5-FOLD CV & GRIDSEARCHCV ACROSS 10 ML MODELS")
    print("=" * 80)
    metrics_summary = train_and_benchmark(output_dir=output_dir, data_dir=data_dir, random_state=42)

    # --------------------------------------------------------------------------
    # STEP 5: PERFORMANCE COMPARISON TABLE & BEST MODEL SELECTION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏆 STEP 5: PERFORMANCE COMPARISON TABLE (HOLDOUT TEST SET N=61)")
    print("=" * 80)
    print(f"{'Classifier Model':<35} | {'Acc':<6} | {'ROC-AUC':<8} | {'F1':<6} | {'Sens':<6} | {'Spec':<6} | {'Prec':<6} | {'Confusion (TP/TN/FP/FN)'}")
    print("-" * 105)
    for m_name, m_data in metrics_summary.items():
        cm_str = f"{m_data['tp']}/{m_data['tn']}/{m_data['fp']}/{m_data['fn']}"
        print(f"{m_name[:34]:<35} | {m_data['accuracy']:>5.1f}% | {m_data['roc_auc']:>6.2f}% | {m_data['f1_score']:>5.1f}% | {m_data['sensitivity']:>5.1f}% | {m_data['specificity']:>5.1f}% | {m_data['precision']:>5.1f}% | {cm_str:<15}")
    print("=" * 105)

    # Load metadata of best model
    best_meta_path = os.path.join(output_dir, "best_model_metadata.json")
    with open(best_meta_path, "r", encoding="utf-8") as f:
        best_meta = json.load(f)
    print(f"\n🌟 BEST MODEL SELECTED: {best_meta['best_model_name']}")
    print(f"   • Test ROC-AUC: {best_meta['test_roc_auc']}% | Test Accuracy: {best_meta['test_accuracy']}% | F1: {best_meta['test_f1_score']}%")
    print(f"   • 5-Fold Cross-Validation: {best_meta['cv_roc_auc_5fold']}")
    print(f"   • Confusion Matrix: TP={best_meta['confusion_matrix']['tp']}, TN={best_meta['confusion_matrix']['tn']}, FP={best_meta['confusion_matrix']['fp']}, FN={best_meta['confusion_matrix']['fn']}")

    # Load best model for explainability
    primary_model = joblib.load(os.path.join(output_dir, "best_model.joblib"))

    # --------------------------------------------------------------------------
    # STEP 6: SHAP + LIME GLOBAL EXPLANATIONS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🌐 STEP 6: DUAL-XAI SUITE: GLOBAL COHORT EXPLANATIONS (SHAP & LIME)")
    print("=" * 80)
    xai_suite = ClinicalExplainabilitySuite(primary_model, X_train)
    
    print("  Calculating Global Mean |SHAP| Feature Importances...")
    global_shap_fig = xai_suite.plot_global_shap_importance(X_train)
    print("  Calculating Global LIME Aggregated Biomarker Weights...")
    global_lime_df = xai_suite.compute_global_lime_importance(X_train, n_samples=30)
    print("\n  Top 5 Global Biomarkers (LIME Surrogate Weights):")
    for _, r in global_lime_df.head(5).iterrows():
        print(f"    - {r['feature'].upper()} ({r['feature_name']}): Mean |Weight| = {r['mean_abs_weight']:.4f}")

    # --------------------------------------------------------------------------
    # STEP 7: SHAP + LIME INDIVIDUAL PATIENT EXPLANATIONS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("👤 STEP 7: INDIVIDUAL PATIENT EXPLANATIONS (SHAP WATERFALL + LIME RULES)")
    print("=" * 80)
    
    # Analyze a high-risk cardiac patient from holdout test set
    high_risk_idx = y_test[y_test == 1].index[0]
    patient_df = X_test.loc[[high_risk_idx]]
    patient_y = y_test.loc[high_risk_idx]
    patient_pred = primary_model.predict(patient_df)[0]
    patient_proba = primary_model.predict_proba(patient_df)[0][1]
    
    print(f"  Analyzing Patient ID #{high_risk_idx} [Actual Target: {patient_y} | Predicted: {patient_pred} (Risk: {patient_proba*100:.1f}%)]")
    
    # Individual SHAP
    shap_ind = xai_suite.explain_patient_shap(patient_df)
    print("\n  [Individual SHAP Breakdown - Top Risk Drivers]:")
    for _, r in shap_ind["contributions_df"].head(4).iterrows():
        print(f"    • {r['feature'].upper()}: {r['actual_value']} -> SHAP {r['shap_value']:+.3f} ({r['impact']})")
        
    # Individual LIME
    lime_ind = xai_suite.explain_patient_lime(patient_df, num_features=5)
    print("\n  [Individual LIME Local Surrogate Rules]:")
    for _, r in lime_ind["rules_df"].head(4).iterrows():
        print(f"    • Rule: '{r['rule']}' -> Weight {r['weight']:+.3f} ({r['impact']})")

    # --------------------------------------------------------------------------
    # STEP 8: SHAP vs LIME CONCORDANCE & DISCORDANCE ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("⚖️ STEP 8: SHAP vs LIME CONCORDANCE & DISCORDANCE AUDIT")
    print("=" * 80)
    
    single_conc = xai_suite.compute_explainer_concordance(shap_ind, lime_ind)
    print(f"  Individual Patient Concordance Index (C_i): {single_conc['concordance_index']} ({single_conc['concordance_score']}%)")
    print(f"  • Spearman Rank Correlation (rho): {single_conc['spearman_rho']}")
    print(f"  • Top-3 Jaccard Overlap: {single_conc['jaccard_top3']}%")
    print(f"  • Directional Sign Concordance: {single_conc['sign_concordance_pct']}%")
    print(f"  • Clinical Audit Tag: {single_conc['consensus_status']}")

    # Cohort-wide concordance audit across holdout test set
    print(f"\n  Running Cohort-wide Concordance Audit across {len(X_test)} holdout test patients...")
    y_test_pred = pd.Series(primary_model.predict(X_test), index=y_test.index)
    cohort_audit_df = xai_suite.audit_cohort_concordance(X_test, y_true=y_test, y_pred=y_test_pred)
    cohort_audit_df.to_csv(os.path.join(data_dir, "processed", "cohort_concordance_audit.csv"), index=False)
    
    overall_mean_ci = float(cohort_audit_df["concordance_index"].mean())
    review_trigger_rate = float(cohort_audit_df["flag_review"].mean() * 100)
    
    print(f"  ✅ Cohort Audit Complete.")
    print(f"  • Mean Concordance Index C_i: {overall_mean_ci:.3f} ({overall_mean_ci * 100:.1f}%)")
    print(f"  • Automated Recommendation Rate: {100.0 - review_trigger_rate:.1f}%")
    print(f"  • Discordance / Secondary Review Flag Rate (C_i < 0.80): {review_trigger_rate:.1f}%")
    
    # Save full concordance summary
    concordance_summary = {
        "total_audited_patients": len(cohort_audit_df),
        "mean_concordance_index": round(overall_mean_ci, 3),
        "mean_concordance_score_pct": round(overall_mean_ci * 100, 1),
        "automated_recommendation_rate_pct": round(100.0 - review_trigger_rate, 1),
        "flagged_secondary_review_rate_pct": round(review_trigger_rate, 1),
        "single_patient_case": single_conc
    }
    with open(os.path.join(output_dir, "concordance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(concordance_summary, f, indent=4)

    # --------------------------------------------------------------------------
    # GENERATE PRESENTATION FIGURES & SCIENTIFIC ASSETS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🎨 STEP 9: GENERATING PRESENTATION FIGURES & RESEARCH ASSETS")
    print("=" * 80)
    try:
        import generate_report_assets
        print("  ✅ Research figures and publication assets generated.")
    except Exception as e:
        print(f"  Notice generating report assets: {e}")

    print("\n" + "=" * 85)
    print("🎉 FULL WORKFLOW EXECUTION COMPLETED SUCCESSFULLY!")
    print("   All models, telemetry, calibration tables, and XAI concordance audits saved.")
    print("=" * 85)


if __name__ == "__main__":
    run_full_research_pipeline()
