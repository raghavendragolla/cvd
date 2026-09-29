"""
Harmonization Automation Script: Phase 3
Standardizes the 4 cardiology cohorts (Cleveland, Hungarian, Zurich, VA Long Beach)
under Track A (13 features) and Track B (10 features).
Applies zero-sentinel masking, deduplication, target binarization, and leak-free imputation.
Exports clean harmonized datasets to data/processed/ and generates HARMONIZATION_MANIFEST.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import (
    load_harmonized_cohort,
    TRACK_A_FEATURES,
    TRACK_B_10_FEATURES,
    TRACK_B_11_FEATURES
)
from src.preprocessing import HarmonizedClinicalPreprocessor


def harmonize_and_export_cohorts(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed"
):
    os.makedirs(processed_dir, exist_ok=True)
    manifest = {
        "metadata": {
            "title": "Clinical Cohort Harmonization Manifest (Phase 3)",
            "research_project": "Beyond Accuracy: Cross-Dataset Generalization, Calibration and Explanation Stability",
            "tracks": {
                "Track A": "13 diagnostic clinical & imaging biomarkers (Cleveland single-cohort benchmark)",
                "Track B": "10 universally recorded clinical biomarkers (Cross-cohort transportability across 4 centers)"
            }
        },
        "cohorts": {}
    }

    # 1. Cleveland Track A (13 Features)
    print("Processing Cleveland Track A (13 features)...")
    clev_a = load_harmonized_cohort("cleveland", track="A", data_dir=raw_dir, drop_duplicates=True)
    clev_a_path = os.path.join(processed_dir, "cleveland_track_a.csv")
    clev_a.to_csv(clev_a_path, index=False)

    # 2. Cleveland Track B (10 Features)
    print("Processing Cleveland Track B (10 features)...")
    clev_b = load_harmonized_cohort("cleveland", track="B10", data_dir=raw_dir, drop_duplicates=True)
    clev_b_path = os.path.join(processed_dir, "cleveland_track_b.csv")
    clev_b.to_csv(clev_b_path, index=False)

    # 3. Hungarian Track B (10 Features)
    print("Processing Hungarian Track B (10 features)...")
    hung_b = load_harmonized_cohort("hungarian", track="B10", data_dir=raw_dir, drop_duplicates=True)
    hung_b_path = os.path.join(processed_dir, "hungarian_track_b.csv")
    hung_b.to_csv(hung_b_path, index=False)

    # 4. Zurich Track B (10 Features)
    print("Processing Zurich Track B (10 features)...")
    zur_b = load_harmonized_cohort("zurich", track="B10", data_dir=raw_dir, drop_duplicates=True)
    zur_b_path = os.path.join(processed_dir, "zurich_track_b.csv")
    zur_b.to_csv(zur_b_path, index=False)

    # 5. VA Long Beach Track B (10 Features)
    print("Processing VA Long Beach Track B (10 features)...")
    va_b = load_harmonized_cohort("va_long_beach", track="B10", data_dir=raw_dir, drop_duplicates=True)
    va_b_path = os.path.join(processed_dir, "va_long_beach_track_b.csv")
    va_b.to_csv(va_b_path, index=False)

    cohort_dfs = {
        "Cleveland_Track_A": (clev_a, TRACK_A_FEATURES, clev_a_path),
        "Cleveland_Track_B": (clev_b, TRACK_B_10_FEATURES, clev_b_path),
        "Hungarian_Track_B": (hung_b, TRACK_B_10_FEATURES, hung_b_path),
        "Zurich_Track_B": (zur_b, TRACK_B_10_FEATURES, zur_b_path),
        "VA_Long_Beach_Track_B": (va_b, TRACK_B_10_FEATURES, va_b_path)
    }

    for key, (df, features, path) in cohort_dfs.items():
        n_pos = int(df["target"].sum())
        n_neg = int(len(df) - n_pos)
        manifest["cohorts"][key] = {
            "file": os.path.relpath(path, ROOT_DIR).replace("\\", "/"),
            "n_samples": int(len(df)),
            "n_features": len(features),
            "features": features,
            "cad_positive_count": n_pos,
            "cad_negative_count": n_neg,
            "prevalence_rate": round(float(n_pos / len(df)), 4),
            "missing_cells": int(df[features].isna().sum().sum()),
            "missing_cell_percentage": round(float(df[features].isna().sum().sum() / (len(df) * len(features)) * 100), 2)
        }

    manifest_path = os.path.join(processed_dir, "HARMONIZATION_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Exported 5 harmonized cohort datasets to {processed_dir}")
    print(f"[OK] Generated Harmonization Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    harmonize_and_export_cohorts()
