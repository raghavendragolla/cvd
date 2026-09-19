"""
Evaluation Telemetry, Clinical Performance Metrics, Probability Calibration (ECE & Brier),
Decision Curve Analysis (DCA), and Interactive Visualizations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve


def compute_expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) with uniform probability binning.
    ECE = Sum (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select items in bin
        if i == n_bins - 1:
            in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
        else:
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_proba[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def evaluate_classifier(y_true, y_pred, y_proba) -> dict:
    """
    Computes comprehensive clinical diagnostic metrics including Brier calibration score and ECE.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    ece_val = compute_expected_calibration_error(y_true, y_proba, n_bins=10)
    pr_auc = float(average_precision_score(y_true, y_proba)) * 100
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)) * 100, 2),
        "pr_auc": round(pr_auc, 2),
        "sensitivity": round(float(recall_score(y_true, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_true, y_pred)) * 100, 2),
        "specificity": round(float(specificity) * 100, 2),
        "precision": round(float(precision_score(y_true, y_pred)) * 100, 2),
        "f1_score": round(float(f1_score(y_true, y_pred)) * 100, 2),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
        "ece": round(ece_val, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
    return metrics


def compute_calibration_data(y_true, y_proba, n_bins: int = 10) -> dict:
    """Computes empirical vs predicted probability calibration bins and ECE."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    ece_score = compute_expected_calibration_error(y_true, y_proba, n_bins=n_bins)
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
        "ece": round(float(ece_score), 4)
    }


def compute_decision_curve_analysis(y_true, y_proba, thresholds=None) -> go.Figure:
    """
    Computes Decision Curve Analysis (DCA) measuring Clinical Net Benefit vs Threshold Probability.
    Crucial gold-standard medical research metric proving clinical triage utility.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.90, 50)
        
    n = len(y_true)
    
    net_benefits_model = []
    net_benefits_all = []
    net_benefits_none = []
    
    for pt in thresholds:
        # Model predictions at threshold pt
        y_pred_thresh = (y_proba >= pt).astype(int)
        tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
        fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
        
        # Net Benefit = (TP/N) - (FP/N) * (pt / (1 - pt))
        nb_model = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefits_model.append(nb_model)
        
        # Treat All Strategy: TP = all positives, FP = all negatives
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all = (tp_all / n) - (fp_all / n) * (pt / (1.0 - pt))
        net_benefits_all.append(nb_all)
        
        # Treat None Strategy: Net Benefit = 0
        net_benefits_none.append(0.0)
        
    fig = go.Figure()
    
    # Model Net Benefit
    fig.add_trace(go.Scatter(
        x=thresholds * 100,
        y=net_benefits_model,
        mode="lines",
        name="⚡ CardioPulse AI Model",
        line=dict(color="#00F0FF", width=3)
    ))
    
    # Treat All Strategy
    fig.add_trace(go.Scatter(
        x=thresholds * 100,
        y=net_benefits_all,
        mode="lines",
        name="🏥 Treat All (Universal Angiography)",
        line=dict(color="#FF2E5B", dash="dash", width=2)
    ))
    
    # Treat None Strategy
    fig.add_trace(go.Scatter(
        x=thresholds * 100,
        y=net_benefits_none,
        mode="lines",
        name="⚪ Treat None (No Intervention)",
        line=dict(color="#94A3B8", dash="dot", width=1.5)
    ))
    
    fig.update_layout(
        title="<b>DECISION CURVE ANALYSIS (CLINICAL NET BENEFIT)</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        xaxis_title="Threshold Probability (Pt %)",
        yaxis_title="Clinical Net Benefit",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.7)",
        font=dict(color="#94A3B8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(7, 11, 20, 0.85)",
            font=dict(color="#F8FAFC", size=10)
        ),
        margin=dict(l=35, r=35, t=55, b=35),
        height=360
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)", range=[-0.1, max(net_benefits_model) + 0.1])
    return fig


def plot_interactive_confusion_matrix(cm_data: dict, model_name: str = "Classifier") -> go.Figure:
    """
    Renders styled Plotly heatmap for Confusion Matrix.
    """
    z = [[cm_data.get("tn", 0), cm_data.get("fp", 0)],
         [cm_data.get("fn", 0), cm_data.get("tp", 0)]]
    
    x = ["Predicted Healthy (0)", "Predicted CAD (1)"]
    y = ["Actual Healthy (0)", "Actual CAD (1)"]
    
    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale=[[0, "#070B14"], [0.5, "#0A2540"], [1, "#00F0FF"]],
        text_auto=True,
        aspect="auto"
    )
    
    fig.update_layout(
        title=f"<b>CONFUSION MATRIX ({model_name.upper()})</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"),
        margin=dict(l=35, r=35, t=45, b=35),
        height=320
    )
    fig.update_xaxes(side="bottom", tickfont=dict(color="#E2E8F0"))
    fig.update_yaxes(tickfont=dict(color="#E2E8F0"))
    return fig


def plot_interactive_roc_curve(roc_curves_dict: dict) -> go.Figure:
    """
    Renders multi-model ROC Curves with AUC comparison.
    """
    fig = go.Figure()
    colors = ["#00F0FF", "#FF2E5B", "#00E599", "#FFB800", "#9D4EDD", "#3B82F6", "#EC4899", "#F97316", "#14B8A6", "#8B5CF6"]
    
    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_curves_dict.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{m_name[:22]} (AUC: {auc_val:.1f}%)",
            line=dict(color=color, width=2)
        ))
        
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (50%)",
        line=dict(color="#64748B", dash="dash", width=1.5)
    ))
    
    fig.update_layout(
        title="<b>MULTI-MODEL RECEIVER OPERATING CHARACTERISTIC (ROC)</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.7)",
        font=dict(color="#94A3B8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(7, 11, 20, 0.85)",
            font=dict(color="#F8FAFC", size=9)
        ),
        margin=dict(l=35, r=35, t=55, b=35),
        height=360
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)")
    return fig


def plot_model_radar_comparison(metrics_dict: dict) -> go.Figure:
    """
    Renders Radar Chart comparing top models across clinical performance dimensions.
    """
    categories = ["Accuracy", "ROC-AUC", "Sensitivity", "Specificity", "Precision", "F1-Score"]
    
    fig = go.Figure()
    colors = ["#00F0FF", "#FF2E5B", "#00E599", "#FFB800", "#9D4EDD"]
    
    top_models = list(metrics_dict.items())[:4]
    
    for idx, (m_name, m_data) in enumerate(top_models):
        values = [
            m_data.get("accuracy", 0),
            m_data.get("roc_auc", 0),
            m_data.get("sensitivity", 0),
            m_data.get("specificity", 0),
            m_data.get("precision", 0),
            m_data.get("f1_score", 0)
        ]
        values.append(values[0])
        
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=m_name[:20],
            line=dict(color=color, width=2),
            opacity=0.35
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 100], tickfont=dict(color="#64748B", size=8), gridcolor="rgba(0, 240, 255, 0.15)"),
            angularaxis=dict(tickfont=dict(color="#F8FAFC", size=10), gridcolor="rgba(0, 240, 255, 0.15)"),
            bgcolor="rgba(7, 11, 20, 0.6)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title="<b>MULTI-MODEL PERFORMANCE RADAR</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#94A3B8", size=9)),
        height=360,
        margin=dict(l=30, r=30, t=45, b=45)
    )
    return fig


def plot_interactive_calibration_curve(calibration_dict: dict) -> go.Figure:
    """Renders interactive Plotly reliability calibration curve."""
    fig = go.Figure()
    colors = ["#00F0FF", "#FF2E5B", "#00E599", "#FFB800", "#9D4EDD", "#3B82F6"]
    
    for idx, (m_name, cal_data) in enumerate(calibration_dict.items()):
        color = colors[idx % len(colors)]
        brier = cal_data.get("brier_score", 0.0)
        ece = cal_data.get("ece", 0.0)
        fig.add_trace(go.Scatter(
            x=[p * 100 for p in cal_data["prob_pred"]],
            y=[p * 100 for p in cal_data["prob_true"]],
            mode="lines+markers",
            name=f"{m_name[:20]} (Brier: {brier:.3f}, ECE: {ece:.3f})",
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
        
    # Perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode="lines",
        name="Perfect Calibration (45°)",
        line=dict(color="#64748B", dash="dash", width=1.5)
    ))
    
    fig.update_layout(
        title="<b>PROBABILITY CALIBRATION RELIABILITY CURVE (ECE & BRIER)</b>",
        title_font=dict(color="#00F0FF", size=13, family="'Chakra Petch', sans-serif"),
        xaxis_title="Mean Predicted Probability (%)",
        yaxis_title="Observed Empirical Frequency (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 11, 20, 0.7)",
        font=dict(color="#94A3B8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(7, 11, 20, 0.85)",
            font=dict(color="#F8FAFC", size=9)
        ),
        margin=dict(l=35, r=35, t=55, b=35),
        height=360
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 240, 255, 0.1)")
    return fig

