"""
Clinical Preprocessing, Imputation, Transformation, Scaler Management,
and Synthetic Data Clinical Validity Auditing.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src.dataset import (
    TRACK_A_FEATURES,
    TRACK_B_FEATURES,
    FEATURE_COLUMNS,
    ALL_COLUMNS,
    FEATURE_DESCRIPTIONS
)

# Clinical categorical mappings
CATEGORICAL_MAPPINGS = {
    "sex": {0.0: "Female", 1.0: "Male"},
    "cp": {
        1.0: "Typical Angina",
        2.0: "Atypical Angina",
        3.0: "Non-Anginal Pain",
        4.0: "Asymptomatic (High Risk)"
    },
    "fbs": {0.0: "False (<= 120 mg/dL)", 1.0: "True (> 120 mg/dL)"},
    "restecg": {
        0.0: "Normal",
        1.0: "ST-T Wave Abnormality",
        2.0: "Left Ventricular Hypertrophy"
    },
    "exang": {0.0: "No Angina on Exertion", 1.0: "Yes (Exercise-Induced Angina)"},
    "slope": {
        1.0: "Upsloping",
        2.0: "Flat (Ischemic Concern)",
        3.0: "Downsloping"
    },
    "ca": {
        0.0: "0 Vessels Colored",
        1.0: "1 Vessel Colored",
        2.0: "2 Vessels Colored",
        3.0: "3 Vessels Colored"
    },
    "thal": {
        3.0: "Normal (3.0)",
        6.0: "Fixed Defect (6.0)",
        7.0: "Reversible Defect (7.0)"
    }
}

# Normal clinical physiological reference ranges
CLINICAL_RANGES = {
    "age": {"min": 18.0, "max": 100.0, "normal_min": 20.0, "normal_max": 65.0, "unit": "years"},
    "trestbps": {"min": 80.0, "max": 220.0, "normal_min": 90.0, "normal_max": 120.0, "unit": "mm Hg"},
    "chol": {"min": 100.0, "max": 600.0, "normal_min": 125.0, "normal_max": 200.0, "unit": "mg/dL"},
    "thalach": {"min": 60.0, "max": 220.0, "normal_min": 130.0, "normal_max": 190.0, "unit": "bpm"},
    "oldpeak": {"min": 0.0, "max": 7.0, "normal_min": 0.0, "normal_max": 1.0, "unit": "mm"}
}


def clean_and_prepare_data(df: pd.DataFrame, feature_set: list = None, impute_missing: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataset, imputes missing clinical biomarkers using KNN imputation,
    and transforms target to binary (0=Healthy, 1=Heart Disease).
    """
    if feature_set is None:
        feature_set = FEATURE_COLUMNS
        
    df_clean = df.copy()
    
    # Replace strings or missing marks
    df_clean = df_clean.replace("?", np.nan)
    
    # Coerce columns to numeric
    for col in df_clean.columns:
        if col != "cohort":
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
    # Drop records where target is missing if target column exists
    if "target" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["target"]).reset_index(drop=True)
        df_clean["target"] = (df_clean["target"] > 0).astype(int)
        
    if not impute_missing:
        return df_clean.dropna(subset=[c for c in feature_set if c in df_clean.columns]).reset_index(drop=True)
        
    # Multi-variable Clinical Feature Imputation
    cols_to_impute = [c for c in feature_set if c in df_clean.columns]
    if cols_to_impute and df_clean[cols_to_impute].isna().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        imputed_array = imputer.fit_transform(df_clean[cols_to_impute])
        df_clean[cols_to_impute] = imputed_array
        
        # Post-process discrete/categorical features to valid integers / categories
        if "sex" in df_clean.columns:
            df_clean["sex"] = df_clean["sex"].round().clip(0, 1)
        if "cp" in df_clean.columns:
            df_clean["cp"] = df_clean["cp"].round().clip(1, 4)
        if "fbs" in df_clean.columns:
            df_clean["fbs"] = df_clean["fbs"].round().clip(0, 1)
        if "restecg" in df_clean.columns:
            df_clean["restecg"] = df_clean["restecg"].round().clip(0, 2)
        if "exang" in df_clean.columns:
            df_clean["exang"] = df_clean["exang"].round().clip(0, 1)
        if "slope" in df_clean.columns:
            df_clean["slope"] = df_clean["slope"].round().clip(1, 3)
        if "ca" in df_clean.columns:
            df_clean["ca"] = df_clean["ca"].round().clip(0, 3)
        if "thal" in df_clean.columns:
            def map_thal(v):
                if np.isnan(v): return 3.0
                opts = [3.0, 6.0, 7.0]
                return float(min(opts, key=lambda x: abs(x - v)))
            df_clean["thal"] = df_clean["thal"].apply(map_thal)
            
        # Bound continuous clinical variables within physiological ranges
        if "trestbps" in df_clean.columns:
            df_clean["trestbps"] = df_clean["trestbps"].clip(80, 220)
        if "chol" in df_clean.columns:
            chol_valid = df_clean.loc[df_clean["chol"] > 50, "chol"]
            med_chol = float(chol_valid.median()) if not chol_valid.empty else 240.0
            df_clean.loc[df_clean["chol"] <= 50, "chol"] = med_chol
            df_clean["chol"] = df_clean["chol"].clip(100, 600)
        if "thalach" in df_clean.columns:
            df_clean["thalach"] = df_clean["thalach"].clip(60, 220)
        if "oldpeak" in df_clean.columns:
            df_clean["oldpeak"] = df_clean["oldpeak"].clip(0.0, 7.0)
            
    cols_to_check = [c for c in feature_set if c in df_clean.columns]
    return df_clean.dropna(subset=cols_to_check).reset_index(drop=True)


def prepare_train_test_split(
    df: pd.DataFrame,
    feature_set: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
    impute_missing: bool = True,
):
    """
    Splits cleaned data into stratified train/test subsets and imputes missing values
    using train-only statistics to prevent information leakage.
    """
    if feature_set is None:
        feature_set = TRACK_A_FEATURES if "ca" in df.columns and "thal" in df.columns else TRACK_B_FEATURES

    df_clean = df.copy()
    df_clean = df_clean.replace("?", np.nan)

    for col in df_clean.columns:
        if col != "cohort":
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    if "target" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["target"]).reset_index(drop=True)
        df_clean["target"] = (df_clean["target"] > 0).astype(int)

    available_features = [c for c in feature_set if c in df_clean.columns]
    X = df_clean[available_features].copy()
    y = df_clean["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if impute_missing:
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train[available_features]),
            columns=available_features,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test[available_features]),
            columns=available_features,
            index=X_test.index,
        )

        for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
            if col in X_train.columns:
                max_val = {"sex": 1, "cp": 4, "fbs": 1, "restecg": 2, "exang": 1, "slope": 3, "ca": 3, "thal": 7}[col]
                min_val = 0 if col not in ["cp", "slope", "ca", "thal"] else 1
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").round().clip(min_val, max_val)
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce").round().clip(min_val, max_val)

        for col, bounds in {"trestbps": (80, 220), "chol": (100, 600), "thalach": (60, 220), "oldpeak": (0.0, 7.0)}.items():
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").clip(bounds[0], bounds[1])
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce").clip(bounds[0], bounds[1])

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# IMBALANCE STRATEGY & SYNTHETIC DATA CLINICAL VALIDITY AUDIT
# --------------------------------------------------------------------------
def audit_synthetic_samples(X_synthetic: pd.DataFrame) -> dict:
    """
    Audits synthetic cardiac data (e.g. from SMOTE) against clinical physiological boundaries.
    Detects impossible clinical interactions:
    1. Blood pressure < 80 or > 220 mmHg
    2. Maximum heart rate > (220 - age) + 15 bpm (impossible exercise physiology)
    3. Oldpeak (ST depression) < 0 or > 7 mm
    4. Non-integer categorical values (interpolated category artifacts)
    """
    n_total = len(X_synthetic)
    if n_total == 0:
        return {"total_samples": 0, "violation_rate": 0.0, "violations_by_type": {}}
        
    violations = {
        "trestbps_out_of_bounds": 0,
        "chol_out_of_bounds": 0,
        "impossible_hr_for_age": 0,
        "oldpeak_negative_or_extreme": 0,
        "fractional_discrete_category": 0
    }
    
    for _, row in X_synthetic.iterrows():
        # 1. Trestbps check
        if "trestbps" in row and (row["trestbps"] < 80 or row["trestbps"] > 220):
            violations["trestbps_out_of_bounds"] += 1
            
        # 2. Chol check
        if "chol" in row and (row["chol"] < 100 or row["chol"] > 600):
            violations["chol_out_of_bounds"] += 1
            
        # 3. Maximum HR physiology check: max achievable HR approx 220 - age
        if "thalach" in row and "age" in row:
            max_physio_hr = (220 - row["age"]) + 15
            if row["thalach"] > max_physio_hr or row["thalach"] < 50:
                violations["impossible_hr_for_age"] += 1
                
        # 4. Oldpeak bounds
        if "oldpeak" in row and (row["oldpeak"] < 0.0 or row["oldpeak"] > 7.0):
            violations["oldpeak_negative_or_extreme"] += 1
            
        # 5. Discrete category interpolation check
        for cat_col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
            if cat_col in row:
                val = row[cat_col]
                if abs(val - round(val)) > 0.05:
                    violations["fractional_discrete_category"] += 1
                    break

    any_violation_count = sum(1 for _, row in X_synthetic.iterrows() if (
        ("trestbps" in row and (row["trestbps"] < 80 or row["trestbps"] > 220)) or
        ("chol" in row and (row["chol"] < 100 or row["chol"] > 600)) or
        ("thalach" in row and "age" in row and (row["thalach"] > (220 - row["age"]) + 15 or row["thalach"] < 50)) or
        ("oldpeak" in row and (row["oldpeak"] < 0.0 or row["oldpeak"] > 7.0))
    ))
    
    violation_rate = (any_violation_count / n_total) * 100.0
    
    return {
        "total_synthetic_samples": n_total,
        "samples_with_violations": any_violation_count,
        "clinical_violation_rate_pct": round(violation_rate, 2),
        "violations_breakdown": violations
    }


def compare_imbalance_strategies(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> dict:
    """
    Compares standard SMOTE oversampling against Cost-Sensitive Class Weighting.
    Performs a physiological validity audit on SMOTE-generated instances.
    """
    results = {}
    
    # 1. Baseline class distribution
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    imbalance_ratio = pos_count / max(neg_count, 1)
    results["class_distribution"] = {
        "negative_healthy": neg_count,
        "positive_disease": pos_count,
        "ratio": round(imbalance_ratio, 3)
    }
    
    # 2. Cost-sensitive weights (balanced)
    total_samples = len(y_train)
    weight_neg = total_samples / (2.0 * neg_count)
    weight_pos = total_samples / (2.0 * pos_count)
    results["cost_sensitive_weights"] = {
        0: round(float(weight_neg), 4),
        1: round(float(weight_pos), 4)
    }
    
    # 3. SMOTE Generation & Physiological Validity Audit
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Extract purely synthetic samples
        n_orig = len(X_train)
        X_synthetic = X_resampled.iloc[n_orig:] if isinstance(X_resampled, pd.DataFrame) else pd.DataFrame(X_resampled[n_orig:], columns=X_train.columns)
        
        audit_res = audit_synthetic_samples(X_synthetic)
        results["smote_audit"] = {
            "synthetic_samples_generated": len(X_synthetic),
            "audit_results": audit_res
        }
    except Exception as e:
        # Fallback simulation of SMOTE interpolation for auditing if imblearn not installed
        # Interpolate between random pairs of positive cases
        pos_df = X_train[y_train == 1]
        n_to_gen = max(0, neg_count - pos_count)
        if n_to_gen > 0 and len(pos_df) > 1:
            syn_rows = []
            for _ in range(n_to_gen):
                i1, i2 = np.random.choice(len(pos_df), 2, replace=False)
                lam = np.random.uniform(0.1, 0.9)
                row_syn = pos_df.iloc[i1] * lam + pos_df.iloc[i2] * (1 - lam)
                syn_rows.append(row_syn)
            X_synthetic = pd.DataFrame(syn_rows)
            audit_res = audit_synthetic_samples(X_synthetic)
        else:
            audit_res = {"total_synthetic_samples": 0, "clinical_violation_rate_pct": 0.0, "violations_breakdown": {}}
            
        results["smote_audit"] = {
            "synthetic_samples_generated": n_to_gen,
            "audit_results": audit_res,
            "note": "Evaluated via linear feature space interpolation audit"
        }
        
    return results


def fit_scaler(X_train: pd.DataFrame, output_path: str = None) -> StandardScaler:
    """Fits and optionally serializes StandardScaler for linear/SVM models."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(scaler, output_path)
    return scaler


def validate_patient_input(patient_dict: dict, feature_set: list = None) -> dict:
    """
    Ensures all required clinical biomarkers are present and correctly typed as floats.
    """
    if feature_set is None:
        feature_set = TRACK_A_FEATURES if "ca" in patient_dict and "thal" in patient_dict else TRACK_B_FEATURES
        
    validated = {}
    for col in feature_set:
        if col not in patient_dict:
            raise ValueError(f"Missing required clinical biomarker: '{col}'")
        try:
            validated[col] = float(patient_dict[col])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value for clinical biomarker '{col}': {patient_dict[col]}")
    return validated

