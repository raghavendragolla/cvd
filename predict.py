"""
CLI Clinical Prediction, Concordance, and Counterfactual Recourse Diagnostic Tool.
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

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dataset import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from src.preprocessing import clean_and_prepare_data, validate_patient_input
from src.utils import get_risk_tier, load_artifacts
from src.explainability import ClinicalExplainabilitySuite


def run_prediction_cli(patient_data: dict = None):
    """Executes full diagnostic pipeline, SHAP/LIME concordance, and actionable recourse in CLI."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    models, features, metrics = load_artifacts(models_dir)
    
    if not models:
        print("❌ No trained models found in 'models/' directory. Run `python src/train.py` first.")
        return
        
    primary_model_name = "⚡ XGBoost (Primary Tree)" if "⚡ XGBoost (Primary Tree)" in models else list(models.keys())[0]
    model = models[primary_model_name]
    
    # Default high-risk test profile if none provided
    if patient_data is None:
        patient_data = {
            "age": 58.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 260.0,
            "fbs": 0.0, "restecg": 2.0, "thalach": 130.0, "exang": 1.0, "oldpeak": 2.2,
            "slope": 2.0, "ca": 2.0, "thal": 7.0
        }
        
    patient_df = pd.DataFrame([patient_data])
    
    pred_class = int(model.predict(patient_df)[0])
    pred_proba = float(model.predict_proba(patient_df)[0][1])
    risk_info = get_risk_tier(pred_proba)
    
    print("=" * 70)
    print("🫀 CARDIOPULSE CLINICAL AI — COMPREHENSIVE RESEARCH DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Active Model       : {primary_model_name}")
    print(f"Predicted Diagnosis: {'🚨 HEART DISEASE DETECTED (CAD POSITIVE)' if pred_class == 1 else '✅ OPTIMAL / HEALTHY (CAD NEGATIVE)'}")
    print(f"Cardiac Risk Score : {pred_proba * 100:.2f}%")
    print(f"Clinical Risk Tier : {risk_info['tier']}")
    print(f"Clinical Directive : {risk_info['recommendation']}")
    print("-" * 70)
    
    # Explainability & Concordance
    raw_data_path = os.path.join(os.path.dirname(__file__), "data", "raw", "heart_disease.csv")
    if os.path.exists(raw_data_path):
        raw_df = pd.read_csv(raw_data_path)
        clean_df = clean_and_prepare_data(raw_df)
        xai_suite = ClinicalExplainabilitySuite(model, clean_df[FEATURE_COLUMNS])
        
        shap_res = xai_suite.explain_patient_shap(patient_df)
        lime_res = xai_suite.explain_patient_lime(patient_df)
        concordance = xai_suite.compute_explainer_concordance(shap_res, lime_res)
        
        print("🔬 NOVELTY 1: INTER-EXPLAINER CONCORDANCE TELEMETRY (SHAP vs LIME)")
        print(f"  • Consensus Score           : {concordance['concordance_score']}% ({concordance['consensus_status']})")
        print(f"  • Spearman Rank Correlation  : ρ = {concordance['spearman_rho']}")
        print(f"  • Shared Key Biomarkers     : {', '.join([f.upper() for f in concordance['overlapping_features']])}")
        print("-" * 70)
        
        recourse = xai_suite.compute_counterfactual_recourse(patient_data, target_risk=0.28)
        print("🎯 NOVELTY 2: COUNTERFACTUAL ACTIONABLE CLINICAL RECOURSE")
        print(f"  • Observed Risk: {recourse['current_risk']:.1f}%  -->  Target Risk: {recourse['optimized_risk']:.1f}%")
        if recourse["prescription"]:
            for idx, p in enumerate(recourse["prescription"]):
                print(f"  [{idx+1}] {p['biomarker']}: {p['current']} --> {p['recommended']} ({p['action']})")
        else:
            print("  • Patient is already in the optimal cardiovascular baseline zone.")
    print("=" * 70)


if __name__ == "__main__":
    run_prediction_cli()
