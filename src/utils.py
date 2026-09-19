"""
Clinical Utility Helpers, Plotly Gauge Visualization, EKG Rhythm Generator, and Artifact Serializers.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.dataset import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS
from src.preprocessing import CLINICAL_RANGES


def get_risk_tier(probability: float) -> dict:
    """
    Classifies heart disease risk probability into standardized clinical severity tiers.
    """
    prob_pct = probability * 100
    
    if prob_pct <= 30.0:
        return {
            "tier": "Optimal Cardiovascular Baseline",
            "tier_short": "Low Risk",
            "badge_class": "badge-low",
            "color": "#00E599",
            "heart_anim": "heart-beat-calm",
            "icon": "💚",
            "recommendation": "Optimal cardiovascular state. Maintain aerobic conditioning (150 min/wk), balanced Mediterranean diet, and routine annual wellness monitoring."
        }
    elif prob_pct <= 60.0:
        return {
            "tier": "Moderate / Borderline Risk",
            "tier_short": "Moderate",
            "badge_class": "badge-moderate",
            "color": "#FFB800",
            "heart_anim": "heart-beat-moderate",
            "icon": "💛",
            "recommendation": "Intermediate cardiac concern. Recommend ambulatory blood pressure monitoring, advanced lipid panel (ApoB/Lp(a)), stress echocardiography, and lifestyle intervention."
        }
    elif prob_pct <= 80.0:
        return {
            "tier": "High Cardiovascular Risk",
            "tier_short": "High Risk",
            "badge_class": "badge-high",
            "color": "#FF8533",
            "heart_anim": "heart-beat-fast",
            "icon": "🧡",
            "recommendation": "Substantial evidence of coronary artery disease. Urgent cardiologist consultation, coronary CT angiography (CCTA), and initiation of guideline-directed medical therapy."
        }
    else:
        return {
            "tier": "Critical Alert (Severe CAD Risk)",
            "tier_short": "Critical",
            "badge_class": "badge-critical",
            "color": "#FF2E5B",
            "heart_anim": "heart-beat-emergency",
            "icon": "🚨",
            "recommendation": "Critical probability of multi-vessel obstructive coronary stenosis. Immediate invasive coronary angiography triage, continuous telemetry monitoring, and emergency cardiology evaluation."
        }


def plot_risk_gauge(probability: float) -> go.Figure:
    """
    Renders an ultra-futuristic glowing radial speedometer gauge for patient cardiac risk score.
    """
    prob_pct = round(probability * 100, 1)
    
    if prob_pct <= 30:
        bar_color = "#00E599"
    elif prob_pct <= 60:
        bar_color = "#FFB800"
    elif prob_pct <= 80:
        bar_color = "#FF8533"
    else:
        bar_color = "#FF2E5B"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={
            "suffix": "%",
            "font": {"size": 44, "color": "#FFFFFF", "family": "'Chakra Petch', sans-serif"}
        },
        title={
            "text": "<b>⚡ CARDIAC RISK PROBABILITY</b>",
            "font": {"size": 13, "color": "#00F0FF", "family": "'Space Grotesk', sans-serif"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "#64748B",
                "tickfont": {"color": "#94A3B8", "size": 10}
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(7, 11, 20, 0.7)",
            "borderwidth": 1,
            "bordercolor": "rgba(0, 240, 255, 0.2)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 229, 153, 0.12)"},
                {"range": [30, 60], "color": "rgba(255, 184, 0, 0.12)"},
                {"range": [60, 80], "color": "rgba(255, 133, 51, 0.15)"},
                {"range": [80, 100], "color": "rgba(255, 46, 91, 0.22)"}
            ],
            "threshold": {
                "line": {"color": "#FF2E5B", "width": 3},
                "thickness": 0.85,
                "value": 80
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC", "family": "sans-serif"},
        height=260,
        margin=dict(l=25, r=25, t=45, b=20)
    )
    return fig


def generate_ekg_waveform_figure(hr_bpm: float = 75.0, st_depression: float = 0.0, restecg: float = 0.0) -> go.Figure:
    """
    Generates a realistic, dynamic P-Q-R-S-T electrocardiogram (EKG) waveform trace matching patient biomarkers.
    """
    # 3-cycle simulated EKG waveform
    n_points = 300
    t = np.linspace(0, 3, n_points)
    
    # Calculate cycle frequency based on bpm
    freq = hr_bpm / 60.0
    
    # Build P-Q-R-S-T synthetic trace
    voltage = np.zeros(n_points)
    
    for cycle in range(int(freq * 3) + 1):
        t_center = cycle / freq + 0.2
        
        # P wave
        voltage += 0.15 * np.exp(-((t - (t_center - 0.12)) ** 2) / (2 * 0.02 ** 2))
        # Q wave
        voltage -= 0.18 * np.exp(-((t - (t_center - 0.04)) ** 2) / (2 * 0.01 ** 2))
        # R peak
        r_height = 1.6 if restecg == 2.0 else 1.2
        voltage += r_height * np.exp(-((t - t_center) ** 2) / (2 * 0.012 ** 2))
        # S wave
        voltage -= 0.35 * np.exp(-((t - (t_center + 0.04)) ** 2) / (2 * 0.015 ** 2))
        # ST depression & T wave
        st_shift = - (st_depression * 0.12)
        voltage += st_shift * np.exp(-((t - (t_center + 0.10)) ** 2) / (2 * 0.04 ** 2))
        voltage += 0.28 * np.exp(-((t - (t_center + 0.22)) ** 2) / (2 * 0.045 ** 2))
        
    # Baseline jitter
    np.random.seed(42)
    voltage += np.random.normal(0, 0.015, n_points)
    
    # Waveform color: Cyan if normal, Crimson if significant ST depression
    line_color = "#FF2E5B" if st_depression > 1.0 else "#00F0FF"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=voltage,
        mode="lines",
        line=dict(color=line_color, width=2.5),
        name="Lead II ECG"
    ))
    
    fig.update_layout(
        title=f"<b>LIVE EKG TELEMETRY MONITOR (Lead II • {hr_bpm:.0f} BPM)</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)", zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)", zeroline=True, zerolinecolor="rgba(255, 46, 91, 0.3)", showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.8)",
        margin=dict(l=10, r=10, t=35, b=10),
        height=180,
        showlegend=False
    )
    return fig


def plot_patient_biomarker_radar(patient_dict: dict) -> go.Figure:
    """
    Renders radar comparison between patient normalized metrics and normal clinical baselines.
    """
    categories = [
        "Blood<br>Pressure",
        "Cholesterol",
        "Heart Rate<br>Reserve",
        "ST<br>Depression",
        "Vessels<br>(Ca)"
    ]
    
    # Normalize patient metrics to 0-100 scale
    bp_norm = np.clip((patient_dict.get("trestbps", 120) - 90) / (180 - 90) * 100, 0, 100)
    chol_norm = np.clip((patient_dict.get("chol", 200) - 150) / (350 - 150) * 100, 0, 100)
    hr_risk_norm = np.clip((200 - patient_dict.get("thalach", 150)) / (200 - 80) * 100, 0, 100)
    st_norm = np.clip(patient_dict.get("oldpeak", 0.0) / 4.0 * 100, 0, 100)
    ca_norm = np.clip(patient_dict.get("ca", 0.0) / 3.0 * 100, 0, 100)
    
    patient_vals = [bp_norm, chol_norm, hr_risk_norm, st_norm, ca_norm]
    patient_vals.append(patient_vals[0])
    
    ideal_vals = [30, 30, 20, 10, 0]
    ideal_vals.append(ideal_vals[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ideal_vals,
        theta=categories + [categories[0]],
        fill="toself",
        name="Healthy Baseline",
        line=dict(color="#00E599", dash="dash"),
        opacity=0.25
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_vals,
        theta=categories + [categories[0]],
        fill="toself",
        name="Patient Observed",
        line=dict(color="#00F0FF", width=2.5),
        opacity=0.45
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#94A3B8", size=9),
                gridcolor="rgba(0, 240, 255, 0.2)"
            ),
            angularaxis=dict(
                tickfont=dict(color="#F8FAFC", size=10, family="'Space Grotesk', -apple-system, sans-serif"),
                gridcolor="rgba(0, 240, 255, 0.2)"
            ),
            bgcolor="rgba(7, 11, 20, 0.75)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title="<b>BIOMETRIC DEVIATION RADAR</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=10)
        ),
        height=265,
        margin=dict(l=45, r=45, t=40, b=45)
    )
    return fig


def load_artifacts(models_dir: str):
    """
    Loads all trained model artifacts, feature lists, and metrics.
    """
    models = {}
    
    # Priority ordered models
    preferred_order = [
        "🫀 Super-Ensemble (Top Precision)",
        "⚡ XGBoost (Primary Tree)",
        "🐱 CatBoost (Clinical Booster)",
        "💡 LightGBM (Fast Booster)",
        "🌲 Random Forest Ensemble",
        "🌳 Extra Trees Classifier",
        "🎯 Support Vector Machine (SVC)",
        "📐 Logistic Regression (Linear)",
        "🧠 Multi-Layer Perceptron (Neural Net)",
        "📈 Gradient Boosting (GBDT)"
    ]
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    # Load all models found in directory
    for m_name in preferred_order:
        safe_fname = "".join([c if c.isalnum() else "_" for c in m_name.split("(")[0]]).strip("_").lower() + ".joblib"
        f_path = os.path.join(models_dir, safe_fname)
        if os.path.exists(f_path):
            try:
                models[m_name] = joblib.load(f_path)
            except Exception:
                pass
                
    # Fallback to standard names if needed
    if not models:
        xgb_path = os.path.join(models_dir, "xgb_model.joblib")
        if os.path.exists(xgb_path):
            models["⚡ XGBoost (Primary Tree)"] = joblib.load(xgb_path)
            
    features_path = os.path.join(models_dir, "features.joblib")
    features = joblib.load(features_path) if os.path.exists(features_path) else FEATURE_COLUMNS
    
    return models, features, metrics
