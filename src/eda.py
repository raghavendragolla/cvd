"""
Exploratory Data Analysis (EDA) & Clinical Diagnostic Telemetry for UCI Cleveland Dataset.
Performs demographic profiling, class prevalence, biomarker missingness analysis,
feature-target correlations, and physiological distribution checks.
"""

import os
import json
import pandas as pd
import numpy as np
from src.dataset import (
    TRACK_A_FEATURES,
    FEATURE_DESCRIPTIONS,
    MODIFIABLE_FEATURES,
    NON_MODIFIABLE_FEATURES
)
from src.preprocessing import CLINICAL_RANGES, CATEGORICAL_MAPPINGS


def perform_data_analysis(df: pd.DataFrame, output_dir: str = "data/processed") -> dict:
    """
    Executes comprehensive clinical exploratory data analysis on raw/clean UCI Cleveland data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df_analysis = df.copy()
    df_analysis = df_analysis.replace("?", np.nan)
    for col in df_analysis.columns:
        if col != "cohort":
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors="coerce")
            
    n_total = len(df_analysis)
    
    # 1. Target Distribution & Prevalence
    target_series = df_analysis["target"] if "target" in df_analysis.columns else None
    if target_series is not None:
        binary_target = (target_series > 0).astype(int)
        cad_count = int(binary_target.sum())
        healthy_count = int(n_total - cad_count)
        cad_prevalence = round((cad_count / n_total) * 100, 2)
    else:
        cad_count, healthy_count, cad_prevalence = 0, 0, 0.0

    # 2. Demographic Breakdown
    age_mean = round(float(df_analysis["age"].mean()), 1) if "age" in df_analysis.columns else 0.0
    age_std = round(float(df_analysis["age"].std()), 1) if "age" in df_analysis.columns else 0.0
    age_min = float(df_analysis["age"].min()) if "age" in df_analysis.columns else 0.0
    age_max = float(df_analysis["age"].max()) if "age" in df_analysis.columns else 0.0
    
    male_count = int((df_analysis["sex"] == 1.0).sum()) if "sex" in df_analysis.columns else 0
    female_count = int((df_analysis["sex"] == 0.0).sum()) if "sex" in df_analysis.columns else 0
    male_pct = round((male_count / max(n_total, 1)) * 100, 1)
    
    # 3. Missing Value Analysis
    missing_counts = {}
    for col in df_analysis.columns:
        n_missing = int(df_analysis[col].isna().sum())
        if n_missing > 0:
            missing_counts[col] = {
                "missing_samples": n_missing,
                "missing_percentage": round((n_missing / n_total) * 100, 2)
            }
            
    # 4. Feature Summary & Target Correlation (Pearson & Spearman)
    feature_stats = {}
    correlations = {}
    
    features_to_analyze = [c for c in TRACK_A_FEATURES if c in df_analysis.columns]
    for feat in features_to_analyze:
        s = df_analysis[feat].dropna()
        f_type = "Continuous" if feat in ["age", "trestbps", "chol", "thalach", "oldpeak"] else "Categorical/Discrete"
        
        stat_entry = {
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            "type": f_type,
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "median": round(float(s.median()), 2),
            "max": round(float(s.max()), 2),
            "missing": int(df_analysis[feat].isna().sum())
        }
        feature_stats[feat] = stat_entry
        
        if target_series is not None:
            pearson_r = df_analysis[[feat, "target"]].dropna().corr(method="pearson").iloc[0, 1]
            spearman_rho = df_analysis[[feat, "target"]].dropna().corr(method="spearman").iloc[0, 1]
            correlations[feat] = {
                "pearson_r": round(float(pearson_r), 3) if not np.isnan(pearson_r) else 0.0,
                "spearman_rho": round(float(spearman_rho), 3) if not np.isnan(spearman_rho) else 0.0,
                "relationship": "Positive Risk Factor" if pearson_r > 0 else "Protective/Inverse Factor"
            }

    eda_summary = {
        "dataset_name": "UCI Cleveland Clinic Foundation (Heart Disease)",
        "total_cohort_size": n_total,
        "feature_count": len(features_to_analyze),
        "target_distribution": {
            "healthy_n": healthy_count,
            "cad_positive_n": cad_count,
            "cad_prevalence_pct": cad_prevalence
        },
        "demographics": {
            "age_mean": age_mean,
            "age_std": age_std,
            "age_range": [age_min, age_max],
            "male_count": male_count,
            "female_count": female_count,
            "male_percentage": male_pct
        },
        "missing_biomarkers": missing_counts,
        "correlations_with_cad": correlations,
        "feature_statistics": feature_stats
    }

    # Save summary to JSON
    summary_path = os.path.join(output_dir, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=4)
        
    print("=" * 80)
    print("🩺 UCI CLEVELAND CLINICAL DATA ANALYSIS (EDA) COMPLETE")
    print("=" * 80)
    print(f"  • Total Patients: {n_total} | Features: {len(features_to_analyze)} biomarkers")
    print(f"  • CAD Prevalence: {cad_prevalence}% ({cad_count} Positive vs {healthy_count} Healthy)")
    print(f"  • Demographics: Age {age_mean} ± {age_std} yrs | {male_pct}% Male ({male_count} M / {female_count} F)")
    if missing_counts:
        missing_str = ", ".join([f"{k}: {v['missing_samples']} ({v['missing_percentage']}%)" for k, v in missing_counts.items()])
        print(f"  • Missing Biomarkers Detected: {missing_str}")
    else:
        print("  • Missing Biomarkers: 0 missing values detected")
    print("-" * 80)
    print("  Top Correlated Biomarkers with CAD:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True)[:5]
    for feat, c_info in sorted_corr:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        print(f"    - {feat.upper()} ({desc}): Spearman ρ = {c_info['spearman_rho']:+.3f} [{c_info['relationship']}]")
    print("=" * 80)
    
    return eda_summary
