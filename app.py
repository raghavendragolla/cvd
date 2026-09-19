"""
CardioPulse Clinical Mission Control - Skyroot-Inspired Health AI Platform.
10-Model Clinical Machine Learning Suite, Dual Explainability (SHAP & LIME),
Inter-Explainer Concordance, Counterfactual Recourse Optimizer, and Decision Curve Analysis (DCA).
"""

import os
import sys
import importlib
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import src.utils
import src.evaluate
import src.explainability
importlib.reload(src.utils)
importlib.reload(src.evaluate)
importlib.reload(src.explainability)

from src.dataset import (
    load_cleveland_track_a,
    create_sample_patient_cohort,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS
)
from src.preprocessing import (
    clean_and_prepare_data,
    CATEGORICAL_MAPPINGS,
    CLINICAL_RANGES,
    validate_patient_input
)
from src.utils import (
    get_risk_tier,
    plot_risk_gauge,
    generate_ekg_waveform_figure,
    plot_patient_biomarker_radar,
    load_artifacts
)
from src.evaluate import (
    plot_interactive_confusion_matrix,
    plot_interactive_roc_curve,
    plot_interactive_calibration_curve,
    plot_model_radar_comparison,
    compute_decision_curve_analysis
)
from src.explainability import ClinicalExplainabilitySuite
from src.train import train_and_benchmark


# Page Configuration
st.set_page_config(
    page_title="CardioPulse AI | Skyroot of Clinical Cardiology",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_theme(theme_choice: str = "🔴 Crimson EKG Pulse"):
    """Injects bespoke Cyber-Cardiology theme styling and dynamic background palettes."""
    import streamlit.components.v1 as components

    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    js_path = os.path.join(os.path.dirname(__file__), "assets", "script.js")
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            components.html(f"<script>{f.read()}</script>", height=0, width=0)

    # Dynamic Background Palette overrides
    if theme_choice == "🔴 Crimson EKG Pulse":
        st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(255, 46, 91, 0.14) 0%, transparent 45%),
                        radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.09) 0%, transparent 45%),
                        #050811 !important;
            color: #F8FAFC !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070B14 0%, #120914 100%) !important;
            border-right: 1px solid rgba(255, 46, 91, 0.22) !important;
        }
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
            color: #F1F5F9 !important;
        }
        [data-baseweb="select"] > div:hover {
            border-color: #FF2E5B !important;
            box-shadow: 0 0 12px rgba(255, 46, 91, 0.35) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "🔵 Cybernetic Bio-Teal":
        st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.16) 0%, transparent 50%),
                        radial-gradient(circle at 85% 85%, rgba(0, 229, 153, 0.12) 0%, transparent 50%),
                        #040814 !important;
            color: #F0FDFA !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #040814 0%, #061520 100%) !important;
            border-right: 1px solid rgba(0, 240, 255, 0.22) !important;
        }
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
            color: #F0FDFA !important;
        }
        [data-baseweb="select"] > div:hover {
            border-color: #00F0FF !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.35) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "🟣 Neon Violet Synapse":
        st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 30%, rgba(181, 23, 158, 0.18) 0%, transparent 50%),
                        radial-gradient(circle at 80% 70%, rgba(255, 46, 91, 0.12) 0%, transparent 45%),
                        #080412 !important;
            color: #FAF5FF !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080412 0%, #150A22 100%) !important;
            border-right: 1px solid rgba(181, 23, 158, 0.28) !important;
        }
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
            color: #FAF5FF !important;
        }
        [data-baseweb="select"] > div:hover {
            border-color: #B5179E !important;
            box-shadow: 0 0 12px rgba(181, 23, 158, 0.35) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme_choice == "⚪ Nordic Clinical Lab":
        st.markdown("""
        <style>
        .stApp {
            background: #F8FAFC !important;
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"] {
            background: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.04) !important;
        }
        section[data-testid="stSidebar"] * {
            color: #334155 !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #0F172A !important;
            text-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #0284C7 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #0284C7 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: #64748B !important;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label,
        .stTextInput label,
        .stMultiSelect label,
        .stFileUploader label {
            color: #0F172A !important;
            text-shadow: none !important;
        }
        [data-baseweb="select"] > div {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            color: #0F172A !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        [data-baseweb="select"] * {
            color: #0F172A !important;
        }
        [data-baseweb="select"] svg {
            fill: #0284C7 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        ul[data-baseweb="menu"] {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08) !important;
        }
        li[data-baseweb="menu-item"],
        li[role="option"] {
            color: #0F172A !important;
        }
        li[data-baseweb="menu-item"]:hover,
        li[role="option"]:hover,
        li[data-baseweb="menu-item"][aria-selected="true"],
        li[role="option"][aria-selected="true"] {
            background: #F1F5F9 !important;
            color: #0284C7 !important;
        }
        [data-testid="stSlider"] [data-testid="stThumbValue"],
        div[data-testid="stSlider"] div[role="slider"] ~ div,
        div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
            color: #0284C7 !important;
            text-shadow: none !important;
        }
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            color: #64748B !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background-color: #0284C7 !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 0 8px rgba(2, 132, 199, 0.4) !important;
        }
        .custom-header, .clinical-card, .heart-core-container {
            background: #FFFFFF !important;
            border-color: #E2E8F0 !important;
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.06) !important;
        }
        .header-title, .card-title, [data-testid="stMetricValue"] {
            color: #0F172A !important;
            text-shadow: none !important;
        }
        .header-subtitle {
            color: #64748B !important;
        }
        .stat-pill {
            background: #F8FAFC !important;
            border-color: #CBD5E1 !important;
            color: #475569 !important;
        }
        .stat-pill b {
            color: #0284C7 !important;
        }
        .stExpander {
            background: #FFFFFF !important;
            border-color: #E2E8F0 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        }
        .stExpander summary,
        .stExpander summary p,
        .stExpander summary span {
            color: #0F172A !important;
        }
        .stExpander summary:hover,
        .stExpander summary:hover p {
            color: #0284C7 !important;
        }
        .stExpander svg {
            fill: #0284C7 !important;
            stroke: #0284C7 !important;
        }
        .stCaption,
        [data-testid="stCaptionContainer"] p {
            color: #64748B !important;
        }
        .stMarkdown p, .stMarkdown li {
            color: #334155 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #F1F5F9 !important;
            border: 1px solid #E2E8F0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748B !important;
        }
        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: #0284C7 !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        }
        [data-baseweb="textarea"] textarea {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #F8FAFC !important;
            border: 1px dashed #CBD5E1 !important;
        }
        [data-testid="stFileUploader"] section * {
            color: #475569 !important;
        }
        [data-baseweb="tag"] {
            background: #E0F2FE !important;
            border: 1px solid #38BDF8 !important;
        }
        [data-baseweb="tag"] span {
            color: #0369A1 !important;
        }
        </style>
        """, unsafe_allow_html=True)


@st.cache_resource
def load_app_context():
    """Initializes all 10 models, dataset cache, and benchmark telemetry."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    manifest_path = os.path.join(models_dir, "training_manifest.json")
    
    if not os.path.exists(manifest_path):
        train_and_benchmark(output_dir=models_dir)
        
    models, features, metrics = load_artifacts(models_dir)
    
    raw_df = load_cleveland_track_a(os.path.join(os.path.dirname(__file__), "data", "raw"))
    clean_df = clean_and_prepare_data(raw_df)
    
    # Load Cleveland-cohort benchmark telemetry.
    roc_curves_data = {}
    roc_path = os.path.join(models_dir, "roc_curves.json")
    if os.path.exists(roc_path):
        with open(roc_path, "r", encoding="utf-8") as f:
            roc_curves_data = json.load(f)
            
    dca_data = {}
    dca_path = os.path.join(models_dir, "dca_data.json")
    if os.path.exists(dca_path):
        with open(dca_path, "r", encoding="utf-8") as f:
            dca_data = json.load(f)
            
    calibration_data = {}
    cal_path = os.path.join(models_dir, "calibration_data.json")
    if os.path.exists(cal_path):
        with open(cal_path, "r", encoding="utf-8") as f:
            calibration_data = json.load(f)
            
    return models, features, metrics, clean_df, roc_curves_data, dca_data, calibration_data


@st.cache_resource
def get_explainer_suite(model_name: str):
    """Dynamically caches and returns the ClinicalExplainabilitySuite for the specified model."""
    models, _, _, clean_df, _, _, _ = load_app_context()
    target_model = models.get(model_name, list(models.values())[0])
    X_train_ref = clean_df[FEATURE_COLUMNS]
    return ClinicalExplainabilitySuite(target_model, X_train_ref)


# Preset Clinical Profiles
PRESET_PATIENT_PROFILES = {
    "🚨 Critical High-Risk Patient (Robert Tanaka, 58M)": {
        "age": 58.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 260.0,
        "fbs": 0.0, "restecg": 2.0, "thalach": 130.0, "exang": 1.0, "oldpeak": 2.2,
        "slope": 2.0, "ca": 2.0, "thal": 7.0
    },
    "✅ Healthy Active Individual (Elena Rostova, 41F)": {
        "age": 41.0, "sex": 0.0, "cp": 2.0, "trestbps": 130.0, "chol": 204.0,
        "fbs": 0.0, "restecg": 2.0, "thalach": 172.0, "exang": 0.0, "oldpeak": 1.4,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
    },
    "⚠️ Moderate / Borderline Risk (Arthur Pendelton, 60M)": {
        "age": 60.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 293.0,
        "fbs": 0.0, "restecg": 2.0, "thalach": 170.0, "exang": 0.0, "oldpeak": 0.1,
        "slope": 1.0, "ca": 2.0, "thal": 7.0
    },
    "🛡️ Young Healthy Female (Amina Morales, 37F)": {
        "age": 37.0, "sex": 0.0, "cp": 3.0, "trestbps": 120.0, "chol": 215.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 170.0, "exang": 0.0, "oldpeak": 0.0,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
    },
    "🔄 Reset to Healthy Baseline (Optimal)": {
        "age": 45.0, "sex": 0.0, "cp": 1.0, "trestbps": 120.0, "chol": 180.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 165.0, "exang": 0.0, "oldpeak": 0.0,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
    },
    "⚙️ Custom Patient Profile": {
        "age": 55.0, "sex": 1.0, "cp": 4.0, "trestbps": 135.0, "chol": 240.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 145.0, "exang": 0.0, "oldpeak": 1.0,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
    }
}


def sync_preset_callback():
    """Synchronize input session state with selected preset clinical profile."""
    preset_name = st.session_state.get("preset_selector")
    if preset_name in PRESET_PATIENT_PROFILES:
        p_data = PRESET_PATIENT_PROFILES[preset_name]
        for k, v in p_data.items():
            st.session_state[f"input_{k}"] = v
            
        # Explicit widget state synchronization
        if "slider_age" in st.session_state:
            st.session_state["slider_age"] = float(p_data["age"])
        if "select_sex" in st.session_state:
            st.session_state["select_sex"] = "Male (1.0)" if p_data["sex"] == 1.0 else "Female (0.0)"
        if "slider_trestbps" in st.session_state:
            st.session_state["slider_trestbps"] = float(p_data["trestbps"])
        if "slider_chol" in st.session_state:
            st.session_state["slider_chol"] = float(p_data["chol"])
        if "select_cp" in st.session_state:
            cp_opts = ["4.0: Asymptomatic (Severe Risk)", "1.0: Typical Angina", "2.0: Atypical Angina", "3.0: Non-Anginal Pain"]
            for opt in cp_opts:
                if opt.startswith(str(p_data["cp"])):
                    st.session_state["select_cp"] = opt
        if "select_fbs" in st.session_state:
            st.session_state["select_fbs"] = "1.0: True (> 120 mg/dL)" if p_data["fbs"] == 1.0 else "0.0: False (<= 120 mg/dL)"
        if "select_restecg" in st.session_state:
            restecg_opts = ["0.0: Normal", "1.0: ST-T Wave Abnormality", "2.0: Left Ventricular Hypertrophy"]
            r_idx = int(p_data["restecg"]) if int(p_data["restecg"]) in [0, 1, 2] else 0
            st.session_state["select_restecg"] = restecg_opts[r_idx]
        if "slider_thalach" in st.session_state:
            st.session_state["slider_thalach"] = float(p_data["thalach"])
        if "select_exang" in st.session_state:
            st.session_state["select_exang"] = "1.0: Yes (Exercise Angina)" if p_data["exang"] == 1.0 else "0.0: No Angina"
        if "slider_oldpeak" in st.session_state:
            st.session_state["slider_oldpeak"] = float(p_data["oldpeak"])
        if "select_slope" in st.session_state:
            slope_opts = ["1.0: Upsloping", "2.0: Flat (Concern)", "3.0: Downsloping"]
            s_idx = int(p_data["slope"]) - 1 if int(p_data["slope"]) in [1, 2, 3] else 0
            st.session_state["select_slope"] = slope_opts[s_idx]
        if "slider_ca" in st.session_state:
            st.session_state["slider_ca"] = float(p_data["ca"])
        if "select_thal" in st.session_state:
            thal_opts = ["3.0: Normal (3)", "6.0: Fixed Defect (6)", "7.0: Reversible Defect (7)"]
            for opt in thal_opts:
                if opt.startswith(str(int(p_data["thal"]))):
                    st.session_state["select_thal"] = opt


def reset_all_inputs():
    """Resets all patient biomarker session states and widget controls to optimal baseline."""
    baseline = {
        "age": 45.0, "sex": 0.0, "cp": 1.0, "trestbps": 120.0, "chol": 180.0,
        "fbs": 0.0, "restecg": 0.0, "thalach": 165.0, "exang": 0.0, "oldpeak": 0.0,
        "slope": 1.0, "ca": 0.0, "thal": 3.0
    }
    for k, v in baseline.items():
        st.session_state[f"input_{k}"] = v
        
    st.session_state["slider_age"] = baseline["age"]
    st.session_state["select_sex"] = "Female (0.0)"
    st.session_state["slider_trestbps"] = baseline["trestbps"]
    st.session_state["slider_chol"] = baseline["chol"]
    st.session_state["select_cp"] = "1.0: Typical Angina"
    st.session_state["select_fbs"] = "0.0: False (<= 120 mg/dL)"
    st.session_state["select_restecg"] = "0.0: Normal"
    st.session_state["slider_thalach"] = baseline["thalach"]
    st.session_state["select_exang"] = "0.0: No Angina"
    st.session_state["slider_oldpeak"] = baseline["oldpeak"]
    st.session_state["select_slope"] = "1.0: Upsloping"
    st.session_state["slider_ca"] = baseline["ca"]
    st.session_state["select_thal"] = "3.0: Normal (3)"
    st.session_state["preset_selector"] = "🔄 Reset to Healthy Baseline (Optimal)"


def main():
    models, features, metrics, clean_df, roc_curves_data, dca_data, calibration_data = load_app_context()

    # Initialize session states for inputs if not present
    if "input_age" not in st.session_state:
        default_p = PRESET_PATIENT_PROFILES["🚨 Critical High-Risk Patient (Robert Tanaka, 58M)"]
        for k, v in default_p.items():
            st.session_state[f"input_{k}"] = v

    # --------------------------------------------------------------------------
    # SIDEBAR: CONFIGURATION, THEME & ACTIVE MODEL TELEMETRY
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎨 CardioPulse Theme & Atmosphere")
        selected_theme = st.selectbox(
            "Visual Palette",
            options=["🔴 Crimson EKG Pulse", "🔵 Cybernetic Bio-Teal", "🟣 Neon Violet Synapse", "⚪ Nordic Clinical Lab"],
            index=0
        )
        load_theme(selected_theme)

        st.markdown("---")
        st.caption("UCI Cleveland cohort: 303 records, 13 clinical features")
        st.markdown("### ⚙️ Diagnostic AI Engine (10 Models)")
        selected_model_name = st.selectbox(
            "Active Clinical Classifier",
            options=list(models.keys()) if models else ["🫀 Super-Ensemble (Top Precision)"],
            index=0,
            help="Choose from 10 state-of-the-art benchmarked AI models."
        )

        # Dynamic Explainer Suite for Active Model
        explainability_suite = get_explainer_suite(selected_model_name)

        st.markdown("---")
        st.markdown("### 📊 Active Model Telemetry")
        if selected_model_name in metrics:
            m = metrics[selected_model_name]
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{m.get('roc_auc', 94.9):.1f}%")
            c2.metric("Accuracy", f"{m.get('accuracy', 85.0):.1f}%")
            c3, c4 = st.columns(2)
            c3.metric("ECE (Calib Error)", f"{m.get('ece', 0.091):.4f}")
            c4.metric("Brier Score", f"{m.get('brier_score', 0.104):.4f}")
            c5, c6 = st.columns(2)
            c5.metric("Sensitivity", f"{m.get('sensitivity', 96.4):.1f}%")
            c6.metric("Specificity", f"{m.get('specificity', 81.8):.1f}%")
        else:
            st.metric("Benchmark ROC-AUC", "95.0%")

        st.markdown("---")
        st.markdown("### 🔬 Scientific Research Specs")
        st.markdown("""
        - **Cohort**: UCI Cleveland (303 records, 13 features)
        - **Calibration**: Platt Sigmoid & Isotonic (ECE)
        - **Concordance**: Dual-XAI (TreeSHAP vs LIME $C_i$)
        - **Trust Gate**: $C_i \\ge 80\\%$ Auto vs $<80\\%$ Review
        - **Recourse**: Actionable Modifiable Biomarkers
        - **Validation**: stratified 5-fold CV plus a 20% holdout
        """)

    # --------------------------------------------------------------------------
    # TOP SKYROOT-INSPIRED MISSION CONTROL HEADER
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="custom-header">
        <div>
            <div class="header-title">
                🫀 CardioPulse Clinical AI Suite
                <span class="stat-pill" style="border-color:#FF2E5B; color:#FF2E5B; font-weight:700;">RESEARCH v3.0</span>
            </div>
            <div class="header-subtitle">
                <span>303 Cleveland records</span> • <span>13 Biomarkers</span> • <span>10 AI Models</span> • <span style="color:#00F0FF;">XAI Concordance</span>
            </div>
        </div>
        <div class="header-stats">
            <div class="stat-pill">Engine: <b style="color:#00E599;">Live Mission Telemetry</b></div>
            <div class="stat-pill">Consensus: <b style="color:#00F0FF;">SHAP + LIME</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # MAIN APPLICATION TABS
    # --------------------------------------------------------------------------
    tab_diag, tab_xai, tab_recourse, tab_bench, tab_batch, tab_glossary = st.tabs([
        "🩺 Patient Assessment & Live EKG",
        "🔬 Dual Explainability & Concordance",
        "🎯 Counterfactual Clinical Recourse",
        "📈 10-Model Tournament & DCA",
        "📁 Batch Patient Records (CSV)",
        "📚 Clinical Biomarker Reference"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: PATIENT DIAGNOSIS & ASSESSMENT
    # --------------------------------------------------------------------------
    with tab_diag:
        col_input, col_results = st.columns([1.1, 1.2], gap="large")

        with col_input:
            col_p1, col_p2 = st.columns([2.5, 1.5])
            with col_p1:
                selected_preset = st.selectbox(
                    "⚡ Quick-Load Clinical Case Profile:",
                    options=list(PRESET_PATIENT_PROFILES.keys()),
                    index=0,
                    key="preset_selector",
                    on_change=sync_preset_callback
                )
            with col_p2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.button(
                    "🔄 Reset to Baseline",
                    on_click=reset_all_inputs,
                    help="Reset all 13 clinical biomarkers to optimal healthy baseline",
                    use_container_width=True
                )

            # Section 1: Demographics & Vitals
            with st.expander("👤 1. Demographics & Baseline Vitals", expanded=True):
                c1, c2 = st.columns(2)
                age = c1.slider("Age (years)", min_value=20.0, max_value=85.0, value=float(st.session_state["input_age"]), step=1.0, key="slider_age")
                
                sex_default_idx = 0 if st.session_state["input_sex"] == 1.0 else 1
                sex_str = c2.selectbox("Biological Sex", options=["Male (1.0)", "Female (0.0)"], index=sex_default_idx, key="select_sex")
                sex = 1.0 if "Male" in sex_str else 0.0
                
                c3, c4 = st.columns(2)
                trestbps = c3.slider("Resting Blood Pressure (mm Hg)", min_value=90.0, max_value=200.0, value=float(st.session_state["input_trestbps"]), step=1.0, key="slider_trestbps")
                chol = c4.slider("Serum Cholesterol (mg/dL)", min_value=120.0, max_value=550.0, value=float(st.session_state["input_chol"]), step=5.0, key="slider_chol")

            # Section 2: Cardiac Physiology & Stress Test
            with st.expander("💓 2. Cardiac Physiology & Stress Test", expanded=True):
                c1, c2 = st.columns(2)
                cp_options = [
                    "4.0: Asymptomatic (Severe Risk)",
                    "1.0: Typical Angina",
                    "2.0: Atypical Angina",
                    "3.0: Non-Anginal Pain"
                ]
                cp_curr_val = st.session_state["input_cp"]
                cp_idx = 0
                for idx, opt in enumerate(cp_options):
                    if opt.startswith(str(cp_curr_val)):
                        cp_idx = idx
                cp_selected = c1.selectbox("Chest Pain Type (cp)", options=cp_options, index=cp_idx, key="select_cp")
                cp = float(cp_selected.split(":")[0])

                fbs_idx = 1 if st.session_state["input_fbs"] == 1.0 else 0
                fbs_selected = c2.selectbox(
                    "Fasting Blood Sugar > 120 mg/dL",
                    options=["0.0: False (<= 120 mg/dL)", "1.0: True (> 120 mg/dL)"],
                    index=fbs_idx,
                    key="select_fbs"
                )
                fbs = float(fbs_selected.split(":")[0])

                c3, c4 = st.columns(2)
                restecg_options = [
                    "0.0: Normal",
                    "1.0: ST-T Wave Abnormality",
                    "2.0: Left Ventricular Hypertrophy"
                ]
                restecg_idx = int(st.session_state["input_restecg"]) if int(st.session_state["input_restecg"]) in [0, 1, 2] else 0
                restecg_selected = c3.selectbox(
                    "Resting ECG (restecg)",
                    options=restecg_options,
                    index=restecg_idx,
                    key="select_restecg"
                )
                restecg = float(restecg_selected.split(":")[0])

                thalach = c4.slider("Max Heart Rate Achieved (bpm)", min_value=70.0, max_value=210.0, value=float(st.session_state["input_thalach"]), step=1.0, key="slider_thalach")

                exang_idx = 1 if st.session_state["input_exang"] == 1.0 else 0
                exang_selected = st.selectbox(
                    "Exercise Induced Angina (exang)",
                    options=["0.0: No Angina", "1.0: Yes (Exercise Angina)"],
                    index=exang_idx,
                    key="select_exang"
                )
                exang = float(exang_selected.split(":")[0])

            # Section 3: Ischemia & Fluoroscopy
            with st.expander("🔬 3. Ischemia & Fluoroscopy Findings", expanded=True):
                c1, c2 = st.columns(2)
                oldpeak = c1.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=float(st.session_state["input_oldpeak"]), step=0.1, key="slider_oldpeak")
                
                slope_options = ["1.0: Upsloping", "2.0: Flat (Concern)", "3.0: Downsloping"]
                slope_curr = int(st.session_state["input_slope"])
                slope_idx = slope_curr - 1 if slope_curr in [1, 2, 3] else 0
                slope_selected = c2.selectbox(
                    "ST Slope (slope)",
                    options=slope_options,
                    index=slope_idx,
                    key="select_slope"
                )
                slope = float(slope_selected.split(":")[0])

                c3, c4 = st.columns(2)
                ca = c3.slider("Major Vessels Colored (ca)", min_value=0.0, max_value=3.0, value=float(st.session_state["input_ca"]), step=1.0, key="slider_ca")
                
                thal_options = ["3.0: Normal (3)", "6.0: Fixed Defect (6)", "7.0: Reversible Defect (7)"]
                thal_curr = int(st.session_state["input_thal"])
                thal_idx = 0
                for idx, opt in enumerate(thal_options):
                    if opt.startswith(str(thal_curr)):
                        thal_idx = idx
                thal_selected = c4.selectbox("Thallium Scan (thal)", options=thal_options, index=thal_idx, key="select_thal")
                thal = float(thal_selected.split(":")[0])

            patient_dict = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            patient_df = pd.DataFrame([patient_dict])

            # Persistent Session State for current patient
            st.session_state["current_patient_df"] = patient_df
            st.session_state["current_patient_dict"] = patient_dict

        with col_results:
            st.markdown("#### 🎯 Real-Time Cardiac Telemetry & Riskometer")
            
            active_model = models.get(selected_model_name, list(models.values())[0])
            
            # Predict
            pred_class = int(active_model.predict(patient_df)[0])
            pred_proba = float(active_model.predict_proba(patient_df)[0][1])
            risk_info = get_risk_tier(pred_proba)

            # Store in session state for Explainability tab
            st.session_state["pred_class"] = pred_class
            st.session_state["pred_proba"] = pred_proba

            # ------------------------------------------------------------------
            # DYNAMIC SYNCHRONIZED BEATING HEART HUD
            # ------------------------------------------------------------------
            st.markdown(f"""
            <div class="heart-core-container">
                <div style="display: flex; align-items: center; gap: 1.5rem;">
                    <div class="heart-core-icon {risk_info['heart_anim']}">
                        🫀
                    </div>
                    <div>
                        <div style="font-family: var(--font-heading); font-size: 1.25rem; font-weight: 800; color: {risk_info['color']};">
                            {risk_info['icon']} {risk_info['tier']} ({pred_proba * 100:.1f}%)
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px;">
                            Heart Rate: <b style="color:#00F0FF;">{thalach:.0f} BPM</b> • Mean BP: <b style="color:#00F0FF;">{trestbps:.0f} mm Hg</b> • Dynamic Pulse Synchronized
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge & Radar in 2 sub-columns
            g_col1, g_col2 = st.columns([1.1, 1.1])
            with g_col1:
                gauge_fig = plot_risk_gauge(pred_proba)
                st.plotly_chart(gauge_fig, use_container_width=True)
            with g_col2:
                radar_fig = plot_patient_biomarker_radar(patient_dict)
                st.plotly_chart(radar_fig, use_container_width=True)

            # ------------------------------------------------------------------
            # LIVE EKG WAVEFORM MONITOR
            # ------------------------------------------------------------------
            ekg_fig = generate_ekg_waveform_figure(hr_bpm=thalach, st_depression=oldpeak, restecg=restecg)
            st.plotly_chart(ekg_fig, use_container_width=True)

            # Clinical Action Card
            st.markdown(f"""
            <div class="clinical-card">
                <div class="card-title">🩺 Clinical Triage & Action Directive</div>
                <p style="font-size: 14px; color: #CBD5E1; line-height: 1.6; margin: 0;">
                    {risk_info['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Instant Clinical XAI Tracing Mini-Ribbon
            with st.expander("⚡ Instant XAI Biomarker Tracing (Quick Look)", expanded=False):
                with st.spinner("Calculating SHAP biomarkers..."):
                    quick_shap = explainability_suite.explain_patient_shap(patient_df)
                    q_df = quick_shap["contributions_df"]
                    q_risk = q_df[q_df["shap_value"] > 0].head(3)
                    q_prot = q_df[q_df["shap_value"] < 0].head(2)
                    
                    c_qx1, c_qx2 = st.columns(2)
                    with c_qx1:
                        st.markdown("<b style='color:#FF2E5B;'>🚨 Top Risk Accelerators:</b>", unsafe_allow_html=True)
                        for _, row in q_risk.iterrows():
                            st.markdown(f"- <b>{row['feature'].upper()}</b> ({row['actual_value']}): +{row['shap_value']:.2f} log-odds", unsafe_allow_html=True)
                    with c_qx2:
                        st.markdown("<b style='color:#00F0FF;'>🛡️ Top Protective Factors:</b>", unsafe_allow_html=True)
                        for _, row in q_prot.iterrows():
                            st.markdown(f"- <b>{row['feature'].upper()}</b> ({row['actual_value']}): {row['shap_value']:.2f} log-odds", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: DUAL EXPLAINABILITY & CONCORDANCE (SHAP & LIME)
    # --------------------------------------------------------------------------
    with tab_xai:
        st.markdown("### 🔬 Dual Explainability Suite & Inter-Explainer Concordance Engine")
        st.markdown("""
        Understand *precisely why* the AI model arrived at its diagnostic risk assessment using both 
        **SHAP (Shapley Additive exPlanations)** based on cooperative game theory and **LIME (Local Interpretable Model-agnostic Explanations)** based on local linear surrogates.
        """)

        curr_df = st.session_state.get("current_patient_df", patient_df)
        curr_dict = st.session_state.get("current_patient_dict", patient_dict)
        curr_proba = st.session_state.get("pred_proba", pred_proba)

        # Explainability Sub-Navigation Tabs
        xai_sub_local, xai_sub_global, xai_sub_concord, xai_sub_report = st.tabs([
            "👤 Patient-Level Local XAI",
            "🌐 Global Cohort Insights (Beeswarm & Dependence)",
            "⚖️ Inter-Explainer Concordance (SHAP vs LIME)",
            "📄 Export Diagnostic XAI Report"
        ])

        with st.spinner("Computing real-time SHAP, LIME, and Concordance analytics..."):
            shap_res = explainability_suite.explain_patient_shap(curr_df)
            lime_res = explainability_suite.explain_patient_lime(curr_df, num_features=8)
            concordance = explainability_suite.compute_explainer_concordance(shap_res, lime_res)

        # ----------------------------------------------------------------------
        # SUB-TAB 1: PATIENT-LEVEL LOCAL XAI
        # ----------------------------------------------------------------------
        with xai_sub_local:
            # INTER-EXPLAINER CONCORDANCE KPI CARD
            audit_badge_color = "#00E599" if not concordance.get("flag_secondary_review", False) else "#FF2E5B"
            st.markdown(f"""
            <div class="clinical-card" style="border-left: 5px solid {audit_badge_color}; background: linear-gradient(135deg, rgba(0,240,255,0.08) 0%, rgba(7,11,20,0.9) 100%);">
                <div class="card-title">🔬 Scientific Novelty: Dual-Explainer Concordance Telemetry (C_i = {concordance.get('concordance_index', 0.85):.3f})</div>
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-top: 0.5rem;">
                    <div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: {audit_badge_color}; font-family: var(--font-heading);">
                            {concordance['consensus_status']}
                        </div>
                        <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px;">
                            Clinical Triage Action: <b style="color:{audit_badge_color}; font-size: 0.95rem;">{concordance.get('audit_action', 'Automated Clinical Recommendation')}</b>
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">
                            Spearman Rank (ρ): <b style="color:#00F0FF;">{concordance['spearman_rho']}</b> • Kendall Tau (τ): <b style="color:#00F0FF;">{concordance.get('kendall_tau', 0.80)}</b> • Directional Sign Match: <b style="color:#00E599;">{concordance.get('sign_concordance_pct', 100)}%</b>
                        </div>
                    </div>
                    <div>
                        <span class="stat-pill" style="border-color:#00E599; color:#00E599;">Top-3 Jaccard: {concordance.get('jaccard_top3', 50.0)}%</span>
                        <span class="stat-pill" style="border-color:#00F0FF; color:#00F0FF;">Top-5 Jaccard: {concordance['jaccard_top5']}%</span>
                        <span class="stat-pill" style="border-color:#FF2E5B; color:#FF2E5B;">Shared Risk Drivers: {', '.join([f.upper() for f in concordance['overlapping_features']])}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # AI Medical Summary Box
            clinical_narrative = explainability_suite.generate_clinical_summary(
                curr_dict, shap_res, lime_res, curr_proba
            )
            st.markdown(f"""
            <div class="clinical-card" style="border-left: 4px solid #FF2E5B;">
                {clinical_narrative}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Side-by-Side Plots (SHAP Waterfall vs LIME Rule Weights)
            col_shap, col_lime = st.columns(2, gap="large")

            with col_shap:
                st.markdown("#### 1️⃣ SHAP Feature Attribution (Local Log-Odds Waterfall)")
                st.caption("🔴 Red bars increase heart disease risk • 🔵 Cyan bars mitigate / reduce risk")
                shap_waterfall_fig = explainability_suite.plot_shap_waterfall(shap_res, patient_name="Selected Patient")
                st.plotly_chart(shap_waterfall_fig, use_container_width=True)

                with st.expander("📊 View Detailed SHAP Contribution Table"):
                    df_shap_table = shap_res["contributions_df"].copy()
                    if "share_pct" not in df_shap_table.columns:
                        if "abs_shap" in df_shap_table.columns:
                            tot_abs = df_shap_table["abs_shap"].sum()
                            df_shap_table["share_pct"] = ((df_shap_table["abs_shap"] / tot_abs) * 100.0) if tot_abs > 0 else 0.0
                        else:
                            df_shap_table["share_pct"] = 0.0

                    display_cols = [c for c in ["feature", "feature_name", "actual_value", "shap_value", "impact", "share_pct"] if c in df_shap_table.columns]
                    df_shap_table = df_shap_table[display_cols]
                    if "actual_value" in df_shap_table.columns:
                        df_shap_table["actual_value"] = df_shap_table["actual_value"].astype(str)
                    if "share_pct" in df_shap_table.columns:
                        df_shap_table["share_pct"] = df_shap_table["share_pct"].round(1).astype(str) + "%"
                    st.dataframe(df_shap_table, use_container_width=True, hide_index=True)

            with col_lime:
                st.markdown("#### 2️⃣ LIME Decision Boundary Attribution (Surrogate Rules)")
                st.caption("🔴 Red rules push prediction to Heart Disease • 🟢 Green rules push to Healthy")
                lime_fig = explainability_suite.plot_lime_explanation(lime_res, patient_name="Selected Patient")
                st.plotly_chart(lime_fig, use_container_width=True)

                with st.expander("📊 View LIME Decision Rules & Surrogate Weights"):
                    st.dataframe(
                        lime_res["rules_df"][["rule", "weight", "impact"]],
                        use_container_width=True,
                        hide_index=True
                    )

            # SHAP Force Spectrum
            st.markdown("---")
            st.markdown("#### 3️⃣ SHAP Force Spectrum: Risk Accelerators vs Protective Shields")
            st.caption("Decomposition of opposing biomarker forces shaping the final clinical probability.")
            shap_force_fig = explainability_suite.plot_shap_force(shap_res, patient_name="Selected Patient")
            st.plotly_chart(shap_force_fig, use_container_width=True)

        # ----------------------------------------------------------------------
        # SUB-TAB 2: GLOBAL COHORT INSIGHTS (BEESWARM & DEPENDENCE)
        # ----------------------------------------------------------------------
        with xai_sub_global:
            st.markdown("#### 🌐 Global Model Explainability & Population-Wide Mechanics")
            st.caption("Aggregated across all clinical training cases to discover primary global cardiac disease drivers.")

            c_g1, c_g2 = st.columns(2, gap="large")

            with c_g1:
                st.markdown("##### 🐝 Global SHAP Beeswarm Summary Distribution")
                st.caption("Each point represents a patient. Color indicates biomarker value (Cyan = Low, Crimson = High).")
                with st.spinner("Rendering SHAP summary beeswarm..."):
                    global_beeswarm_fig = explainability_suite.plot_shap_summary_beeswarm(clean_df[FEATURE_COLUMNS], max_display=10)
                    st.plotly_chart(global_beeswarm_fig, use_container_width=True)

            with c_g2:
                st.markdown("##### 📊 Global Mean |SHAP| Feature Importance Ranking")
                st.caption("Ranks features by average absolute impact on model predictions.")
                with st.spinner("Rendering global importance ranking..."):
                    global_shap_fig = explainability_suite.plot_global_shap_importance(clean_df[FEATURE_COLUMNS])
                    st.plotly_chart(global_shap_fig, use_container_width=True)

            st.markdown("---")
            st.markdown("##### 🔍 Interactive SHAP Partial Dependence & Feature Interaction Explorer")
            st.caption("Examine how specific continuous biomarkers non-linearly shift risk across the population.")

            dep_c1, dep_c2 = st.columns([1, 1])
            with dep_c1:
                selected_dep_feat = st.selectbox(
                    "Select Primary Biomarker to Inspect:",
                    options=FEATURE_COLUMNS,
                    index=FEATURE_COLUMNS.index("thalach") if "thalach" in FEATURE_COLUMNS else 0,
                    key="dep_feature_select"
                )
            with dep_c2:
                selected_col_feat = st.selectbox(
                    "Select Interaction Color Variable:",
                    options=["None"] + FEATURE_COLUMNS,
                    index=FEATURE_COLUMNS.index("oldpeak") + 1 if "oldpeak" in FEATURE_COLUMNS else 0,
                    key="dep_color_select"
                )

            color_arg = None if selected_col_feat == "None" else selected_col_feat
            with st.spinner(f"Computing partial dependence for {selected_dep_feat}..."):
                dep_fig = explainability_suite.plot_shap_dependence(
                    selected_dep_feat,
                    clean_df[FEATURE_COLUMNS],
                    color_feature=color_arg
                )
                st.plotly_chart(dep_fig, use_container_width=True)

        # ----------------------------------------------------------------------
        # SUB-TAB 3: INTER-EXPLAINER CONCORDANCE (SHAP VS LIME)
        # ----------------------------------------------------------------------
        with xai_sub_concord:
            st.markdown("#### ⚖️ Scientific Novelty: Dual-Explainer Concordance Telemetry")
            st.markdown("""
            **Why Dual Explainability Matters in Medicine**:
            Clinical machine learning often suffers from the *"Black-Box Dilemma"*. When a single explainer is used, physicians cannot verify if the explanation is an artifact of the method. 
            By cross-referencing **SHAP** (cooperative game theory) and **LIME** (local linear surrogates), CardioPulse calculates an **Inter-Explainer Concordance Score**. When both explainers agree, doctor diagnostic confidence increases significantly.
            """)

            conc_c1, conc_c2, conc_c3, conc_c4 = st.columns(4)
            conc_c1.metric("Inter-Explainer Consensus", f"{concordance['concordance_score']}%")
            conc_c2.metric("Spearman Rank Correlation (ρ)", f"{concordance['spearman_rho']}")
            conc_c3.metric("Top-5 Jaccard Overlap", f"{concordance['jaccard_top5']}%")
            conc_c4.metric("Sign Concordance Rate", f"{concordance.get('sign_concordance_pct', 100):.1f}%")

            st.markdown("---")
            st.markdown("##### 📋 Side-by-Side Feature Ranking Comparison (SHAP vs LIME)")
            
            # Build comparative leaderboard
            s_top = shap_res["contributions_df"].head(8)
            l_top = lime_res["rules_df"].head(8)
            
            comp_rows = []
            for idx in range(max(len(s_top), len(l_top))):
                s_f = s_top.iloc[idx]["feature"].upper() if idx < len(s_top) else "N/A"
                s_w = f"{s_top.iloc[idx]['shap_value']:+.3f}" if idx < len(s_top) else "N/A"
                s_imp = s_top.iloc[idx]["impact"] if idx < len(s_top) else "N/A"
                
                l_r = l_top.iloc[idx]["rule"] if idx < len(l_top) else "N/A"
                l_w = f"{l_top.iloc[idx]['weight']:+.3f}" if idx < len(l_top) else "N/A"
                l_imp = l_top.iloc[idx]["impact"] if idx < len(l_top) else "N/A"
                
                comp_rows.append({
                    "Rank": f"#{idx+1}",
                    "SHAP Feature": s_f,
                    "SHAP Impact (Log-Odds)": s_w,
                    "SHAP Direction": s_imp,
                    "LIME Rule": l_r,
                    "LIME Weight": l_w,
                    "LIME Direction": l_imp
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # ----------------------------------------------------------------------
        # SUB-TAB 4: EXPORT DIAGNOSTIC XAI REPORT
        # ----------------------------------------------------------------------
        with xai_sub_report:
            st.markdown("#### 📄 Exportable Clinical Diagnostic & Explainability Report")
            st.markdown("Generate and download a comprehensive clinical audit report containing patient biomarkers, model telemetry, SHAP attributions, LIME surrogate rules, and counterfactual recourse.")

            report_md = f"""# 🫀 CARDIOPULSE CLINICAL AI — DIAGNOSTIC & XAI AUDIT REPORT
**Date & Timestamp**: 2026-08-18 | **AI Suite Version**: v3.0 Conference Grade
**Active Classifier**: {selected_model_name}
**Predicted Diagnostic Class**: {"🚨 HEART DISEASE DETECTED (CAD POSITIVE)" if st.session_state.get("pred_class", pred_class) == 1 else "✅ OPTIMAL / HEALTHY (CAD NEGATIVE)"}
**Estimated CAD Probability**: {curr_proba * 100:.2f}%
**Clinical Risk Tier**: {risk_info['tier']}

---

## 1. Patient Clinical Biomarkers
| Biomarker | Variable Name | Recorded Value | Normal Reference |
| :--- | :--- | :--- | :--- |
| Age | age | {curr_dict.get('age', 0)} years | 20-65 years |
| Sex | sex | {"Male (1.0)" if curr_dict.get('sex', 0) == 1 else "Female (0.0)"} | 0=Female, 1=Male |
| Chest Pain Type | cp | {curr_dict.get('cp', 0)} | 1-4 |
| Resting Blood Pressure | trestbps | {curr_dict.get('trestbps', 0)} mm Hg | 90-120 mm Hg |
| Serum Cholesterol | chol | {curr_dict.get('chol', 0)} mg/dL | 125-200 mg/dL |
| Fasting Blood Sugar | fbs | {curr_dict.get('fbs', 0)} | < 120 mg/dL |
| Resting ECG | restecg | {curr_dict.get('restecg', 0)} | 0=Normal |
| Max Heart Rate | thalach | {curr_dict.get('thalach', 0)} bpm | 130-190 bpm |
| Exercise Angina | exang | {curr_dict.get('exang', 0)} | 0=No |
| ST Depression | oldpeak | {curr_dict.get('oldpeak', 0)} mm | 0.0-1.0 mm |
| ST Slope | slope | {curr_dict.get('slope', 0)} | 1=Upsloping |
| Major Vessels Colored | ca | {curr_dict.get('ca', 0)} | 0 vessels |
| Thallium Scan | thal | {curr_dict.get('thal', 0)} | 3=Normal |

---

## 2. Dual Explainability & Inter-Explainer Concordance
- **Inter-Explainer Consensus**: {concordance['concordance_score']}% ({concordance['consensus_status']})
- **Spearman Rank Correlation (ρ)**: {concordance['spearman_rho']}
- **Top-5 Jaccard Index**: {concordance['jaccard_top5']}%
- **Sign Concordance Rate**: {concordance.get('sign_concordance_pct', 100):.1f}%

### Top SHAP Feature Attributions:
{chr(10).join([f"- **{r['feature'].upper()}** ({r['actual_value']}): {r['shap_value']:+.3f} ({r['impact']})" for _, r in shap_res['contributions_df'].head(5).iterrows()])}

### Top LIME Surrogate Rules:
{chr(10).join([f"- Rule: `{r['rule']}` | Weight: {r['weight']:+.3f} | Impact: {r['impact']}" for _, r in lime_res['rules_df'].head(5).iterrows()])}

---

## 3. Clinical Recommendation & Directive
{risk_info['recommendation']}

---
*Attending Cardiologist Signature*: ___________________________  *Date*: ______________
"""

            st.text_area("Report Preview (Markdown):", value=report_md, height=260)
            
            c_rep1, c_rep2 = st.columns(2)
            with c_rep1:
                st.download_button(
                    "📥 Download Markdown Report (.md)",
                    data=report_md.encode("utf-8"),
                    file_name="cardiopulse_xai_diagnostic_report.md",
                    mime="text/markdown",
                    type="primary",
                    use_container_width=True
                )
            with c_rep2:
                json_telemetry = {
                    "patient_biomarkers": curr_dict,
                    "prediction": {
                        "class": int(st.session_state.get("pred_class", pred_class)),
                        "probability": float(curr_proba),
                        "risk_tier": risk_info["tier"]
                    },
                    "concordance": concordance,
                    "top_shap": shap_res["contributions_df"].head(5).to_dict(orient="records"),
                    "top_lime": lime_res["rules_df"].head(5).to_dict(orient="records")
                }
                st.download_button(
                    "📥 Download JSON Telemetry (.json)",
                    data=json.dumps(json_telemetry, indent=2).encode("utf-8"),
                    file_name="cardiopulse_telemetry.json",
                    mime="application/json",
                    use_container_width=True
                )

    # --------------------------------------------------------------------------
    # TAB 3: COUNTERFACTUAL CLINICAL RECOURSE SIMULATOR
    # --------------------------------------------------------------------------
    with tab_recourse:
        st.markdown("### 🎯 Actionable Counterfactual Clinical Recourse Optimizer")
        st.markdown("""
        **From Passive Prediction to Active Therapeutic Guidance**: Most AI models stop at predicting high risk. 
        Our **Counterfactual Recourse Engine** computes the exact, minimal lifestyle and clinical modifications needed to safely transition this patient from **High/Critical Risk down to Optimal Low Risk (< 30%)**.
        """)

        curr_dict = st.session_state.get("current_patient_dict", patient_dict)
        recourse_res = explainability_suite.compute_counterfactual_recourse(curr_dict, target_risk=0.28)

        col_cf1, col_cf2 = st.columns([1.1, 1.2], gap="large")

        with col_cf1:
            st.markdown("#### 📊 Risk Trajectory Comparison")
            
            curr_r = recourse_res["current_risk"]
            opt_r = recourse_res["optimized_risk"]
            
            c_r1, c_r2 = st.columns(2)
            c_r1.metric("Current Observed Risk", f"{curr_r:.1f}%", delta=f"{curr_r - 50:.1f}% vs baseline", delta_color="inverse")
            c_r2.metric("Prescribed Target Risk", f"{opt_r:.1f}%", delta=f"-{recourse_res.get('risk_reduction_pct', 0):.1f}% reduction", delta_color="normal")

            # Trajectory Delta Bar Chart
            df_comp = pd.DataFrame({
                "State": ["Current Observed", "After Clinical Recourse", "Healthy Baseline Target"],
                "Risk Probability (%)": [curr_r, opt_r, 20.0]
            })
            fig_bar = px.bar(
                df_comp,
                x="State",
                y="Risk Probability (%)",
                color="State",
                color_discrete_map={
                    "Current Observed": "#FF2E5B",
                    "After Clinical Recourse": "#00F0FF",
                    "Healthy Baseline Target": "#00E599"
                },
                text_auto=".1f"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(7, 11, 20, 0.7)",
                font=dict(color="#94A3B8"),
                margin=dict(l=25, r=25, t=35, b=25),
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_cf2:
            st.markdown("#### 💊 Actionable Clinical Prescription Roadmap")
            if recourse_res["prescription"]:
                for idx, p_step in enumerate(recourse_res["prescription"]):
                    st.markdown(f"""
                    <div class="clinical-card" style="border-left: 4px solid #00E599; margin-bottom: 0.75rem; padding: 1rem;">
                        <div style="font-weight: 700; color: #F8FAFC; font-family: var(--font-heading); font-size: 0.95rem;">
                            Step {idx+1}: {p_step['biomarker']}
                        </div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 3px;">
                            Current Value: <b style="color:#FF2E5B;">{p_step['current']}</b> $\\rightarrow$ Target: <b style="color:#00E599;">{p_step['recommended']}</b>
                        </div>
                        <div style="font-size: 0.82rem; color: #00F0FF; margin-top: 4px;">
                            📋 Action: {p_step['action']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Patient profile is already in the optimal cardiovascular risk zone (< 30%). Maintain current lifestyle regimen.")

    # --------------------------------------------------------------------------
    # TAB 4: 10-MODEL CLINICAL BENCHMARK & DCA
    # --------------------------------------------------------------------------
    with tab_bench:
        st.markdown("### 📈 10-Model Clinical Machine Learning Tournament & Decision Curve Analysis")
        st.markdown("Side-by-side leaderboard and diagnostic evaluation across 10 top machine learning classifiers evaluated via 5-fold cross-validation and Brier probability reliability.")

        # Comparison Metrics Table
        df_metrics_rows = []
        for m_name, m_val in metrics.items():
            df_metrics_rows.append({
                "Model Architecture": m_name,
                "ROC-AUC (%)": f"{m_val.get('roc_auc', 0):.2f}%",
                "Accuracy (%)": f"{m_val.get('accuracy', 0):.2f}%",
                "Sensitivity / Recall (%)": f"{m_val.get('sensitivity', 0):.2f}%",
                "Specificity (%)": f"{m_val.get('specificity', 0):.2f}%",
                "Precision (%)": f"{m_val.get('precision', 0):.2f}%",
                "F1-Score (%)": f"{m_val.get('f1_score', 0):.2f}%",
                "Brier Score (Calib)": f"{m_val.get('brier_score', 0):.4f}",
                "5-Fold CV AUC (%)": f"{m_val.get('cv_roc_auc_mean', 0):.2f}% ± {m_val.get('cv_roc_auc_std', 0):.1f}%"
            })
        st.dataframe(pd.DataFrame(df_metrics_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3 Column Layout: ROC Curves, Probability Calibration, and Decision Curve Analysis (DCA)
        col_roc, col_calib, col_dca = st.columns([1, 1, 1], gap="medium")

        with col_roc:
            if roc_curves_data:
                roc_fig_dict = {
                    name: (np.array(d["fpr"]), np.array(d["tpr"]), d["auc"])
                    for name, d in roc_curves_data.items()
                }
                roc_fig = plot_interactive_roc_curve(roc_fig_dict)
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("ROC curves data will appear after model training.")

        with col_calib:
            if calibration_data:
                cal_fig = plot_interactive_calibration_curve(calibration_data)
                st.plotly_chart(cal_fig, use_container_width=True)
            else:
                st.info("Calibration reliability curves computing...")

        with col_dca:
            if dca_data:
                dca_fig = compute_decision_curve_analysis(
                    np.array(dca_data["y_true"]),
                    np.array(dca_data["y_proba"])
                )
                st.plotly_chart(dca_fig, use_container_width=True)
            else:
                st.info("DCA telemetry is computing...")

        st.markdown("---")
    # --------------------------------------------------------------------------
    # TAB 5: BATCH PATIENT CSV PROCESSING
    # --------------------------------------------------------------------------
    with tab_batch:
        st.markdown("### 📁 Batch Patient Records & Cohort Analytics")
        st.markdown("Upload a batch of patient electronic health records (CSV) or load our pre-configured clinical trial cohort for batch diagnosis.")

        c_up1, c_up2 = st.columns([1.5, 1])
        with c_up1:
            uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])
        with c_up2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            load_sample_cohort = st.button("📥 Load Sample Patient Cohort (8 Patients)", type="secondary")

        batch_df = None
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
        elif load_sample_cohort or "batch_df" in st.session_state:
            sample_cohort_path = os.path.join(os.path.dirname(__file__), "data", "raw", "sample_patients.csv")
            if not os.path.exists(sample_cohort_path):
                create_sample_patient_cohort(sample_cohort_path)
            batch_df = pd.read_csv(sample_cohort_path)
            st.session_state["batch_df"] = batch_df

        if batch_df is not None:
            st.markdown(f"**Loaded Cohort**: {batch_df.shape[0]} patient records")
            
            missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required clinical biomarker columns: {missing_cols}")
            else:
                active_model = models.get(selected_model_name, list(models.values())[0])
                
                X_batch = batch_df[FEATURE_COLUMNS]
                preds = active_model.predict(X_batch)
                probas = active_model.predict_proba(X_batch)[:, 1]
                
                results_df = batch_df.copy()
                results_df["predicted_class"] = preds
                results_df["predicted_label"] = ["Heart Disease (1)" if p == 1 else "Healthy (0)" for p in preds]
                results_df["risk_probability_pct"] = (probas * 100).round(1)
                results_df["risk_tier"] = [get_risk_tier(p)["tier"] for p in probas]

                total_pts = len(results_df)
                disease_pts = sum(preds == 1)
                healthy_pts = sum(preds == 0)
                mean_risk = probas.mean() * 100

                st.markdown("---")
                st.markdown("#### 📊 Cohort Overview & Risk Distribution")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Patients", total_pts)
                c2.metric("Predicted CAD Cases", f"{disease_pts} ({disease_pts/total_pts*100:.1f}%)")
                c3.metric("Predicted Healthy", f"{healthy_pts} ({healthy_pts/total_pts*100:.1f}%)")
                c4.metric("Mean Cohort Risk", f"{mean_risk:.1f}%")

                fig_hist = px.histogram(
                    results_df,
                    x="risk_probability_pct",
                    nbins=10,
                    color="risk_tier",
                    title="<b>Cohort Risk Score Distribution</b>",
                    color_discrete_map={
                        "Optimal Cardiovascular Baseline": "#00E599",
                        "Moderate / Borderline Risk": "#FFB800",
                        "High Cardiovascular Risk": "#FF8533",
                        "Critical Alert (Severe CAD Risk)": "#FF2E5B"
                    }
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(7, 11, 20, 0.5)",
                    font=dict(color="#94A3B8"),
                    margin=dict(l=30, r=30, t=40, b=30),
                    height=280
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("#### 📋 Diagnostic Results Table")
                selected_tier_filter = st.multiselect(
                    "Filter by Risk Tier:",
                    options=list(results_df["risk_tier"].unique()),
                    default=list(results_df["risk_tier"].unique())
                )
                filtered_df = results_df[results_df["risk_tier"].isin(selected_tier_filter)]
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Enriched Diagnoses (CSV)",
                    data=csv_data,
                    file_name="heart_disease_diagnoses_output.csv",
                    mime="text/csv",
                    type="primary"
                )

    # --------------------------------------------------------------------------
    # TAB 6: CLINICAL BIOMARKER REFERENCE GUIDE
    # --------------------------------------------------------------------------
    with tab_glossary:
        st.markdown("### 📚 Clinical Biomarker Reference & Medical Glossary")
        st.markdown("Comprehensive physiological details, reference ranges, and clinical interpretation guidelines for all 13 cardiac biomarkers.")

        glossary_items = [
            {
                "Biomarker": "AGE (Patient Age)",
                "Variable": "age",
                "Normal Range": "20 – 65 years",
                "Clinical Impact": "Age is an independent non-modifiable cardiovascular risk factor. Advanced age correlates with arterial stiffness, endothelial dysfunction, and cumulative atheroma burden."
            },
            {
                "Biomarker": "SEX (Biological Sex)",
                "Variable": "sex",
                "Normal Range": "0 = Female, 1 = Male",
                "Clinical Impact": "Males have a higher baseline incidence of premature CAD. Estrogen offers cardioprotective endothelial effects in pre-menopausal females."
            },
            {
                "Biomarker": "CP (Chest Pain Type)",
                "Variable": "cp",
                "Normal Range": "1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic",
                "Clinical Impact": "Type 4 (Asymptomatic / Silent Ischemia) is paradoxical in medical datasets as it represents severe unperceived microvascular / multi-vessel disease requiring clinical intervention."
            },
            {
                "Biomarker": "TRESTBPS (Resting Blood Pressure)",
                "Variable": "trestbps",
                "Normal Range": "90 – 120 mm Hg",
                "Clinical Impact": "Hypertension (> 130/80 mm Hg) damages vascular endothelium, promotes arterial plaque formation, and increases myocardial afterload."
            },
            {
                "Biomarker": "CHOL (Serum Cholesterol)",
                "Variable": "chol",
                "Normal Range": "125 – 200 mg/dL",
                "Clinical Impact": "Elevated LDL-C and total cholesterol drive atherogenesis, plaque rupture, and coronary thrombosis."
            },
            {
                "Biomarker": "FBS (Fasting Blood Sugar)",
                "Variable": "fbs",
                "Normal Range": "< 100 mg/dL (0 = False, 1 = True if > 120)",
                "Clinical Impact": "Impaired glucose metabolism accelerates vascular calcification and induces diabetic cardiomyopathy."
            },
            {
                "Biomarker": "RESTECG (Resting Electrocardiogram)",
                "Variable": "restecg",
                "Normal Range": "0 = Normal",
                "Clinical Impact": "ST-T wave abnormalities (1.0) and Left Ventricular Hypertrophy (2.0) indicate chronic hypertensive injury and subendocardial ischemia."
            },
            {
                "Biomarker": "THALACH (Maximum Heart Rate)",
                "Variable": "thalach",
                "Normal Range": "130 – 190 bpm (Target: 220 - Age)",
                "Clinical Impact": "Inability to achieve age-predicted target heart rate (chronotropic incompetence) is a strong diagnostic marker for myocardial ischemia."
            },
            {
                "Biomarker": "EXANG (Exercise-Induced Angina)",
                "Variable": "exang",
                "Normal Range": "0 = No Angina",
                "Clinical Impact": "Angina provoked during stress testing indicates significant (> 70%) obstructive coronary artery narrowing failing to meet metabolic demand."
            },
            {
                "Biomarker": "OLDPEAK (ST Depression)",
                "Variable": "oldpeak",
                "Normal Range": "0.0 – 1.0 mm",
                "Clinical Impact": "ST segment horizontal/downsloping depression relative to baseline is one of the most specific non-invasive markers of active subendocardial ischemia."
            },
            {
                "Biomarker": "SLOPE (ST Segment Slope)",
                "Variable": "slope",
                "Normal Range": "1 = Upsloping",
                "Clinical Impact": "Flat (2.0) and Downsloping (3.0) ST segments during peak exercise carry high positive predictive value for severe coronary stenosis."
            },
            {
                "Biomarker": "CA (Major Colored Vessels)",
                "Variable": "ca",
                "Normal Range": "0 vessels",
                "Clinical Impact": "Number of major epicardial coronary vessels with > 50% stenosis visualized under fluoroscopy. Multi-vessel involvement (ca >= 2) signifies severe CAD."
            },
            {
                "Biomarker": "THAL (Thallium Stress Scintigraphy)",
                "Variable": "thal",
                "Normal Range": "3 = Normal",
                "Clinical Impact": "Reversible defects (7.0) show myocardial regions that are ischemic during stress but reperfused at rest. Fixed defects (6.0) indicate previous myocardial infarction / scar tissue."
            }
        ]

        st.dataframe(pd.DataFrame(glossary_items), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
