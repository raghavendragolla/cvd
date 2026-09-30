# Cross-Dataset Evaluation of Machine Learning Models for Heart Disease Prediction: Generalization, Calibration, and Explainability

This repository contains the research code, experimental pipeline, configuration, and evaluation artifacts for the empirical study:

> **"Cross-Dataset Evaluation of Machine Learning Models for Heart Disease Prediction: Generalization, Calibration, and Explainability"**

---

## 1. Research Overview

While machine learning models for coronary artery disease (CAD) risk assessment frequently report strong discrimination under conventional single-cohort internal validation (cross-validation or random train-test splits), performance can degrade substantially when models are transferred across distinct clinical environments with differing patient demographics, disease severity, and diagnostic protocols.

This study conducts a multi-center benchmark to quantify the breakdown in **predictive discrimination**, **probabilistic calibration**, and **post-hoc feature attribution stability (SHAP)** across four international clinical cohorts under standardized feature harmonization.

---

## 2. Research Objectives & Questions

The experimental framework addresses four core questions:

1. **Generalization Decay:** How severely do discrimination metrics (ROC-AUC, PR-AUC) and probabilistic scores (Brier score, Expected Calibration Error) degrade when models trained on one clinical center are evaluated zero-shot on external hospital cohorts?
2. **Probability Calibration Drift:** Does discrimination protect against probabilistic miscalibration under cross-cohort deployment, and to what extent does Platt recalibration restore reliability?
3. **Explanation Instability:** Do post-hoc feature attributions (SHAP) remain consistent across cohorts, or does covariate shift induce feature rank displacement?
4. **Subgroup & Missingness Robustness:** How do demographic factors (age $\le 55$ vs. $> 55$, biological sex) and feature missingness patterns modulate cross-dataset transferability?

---

## 3. Study Design & Cohorts

The investigation benchmarks four international cardiology cohorts from the UCI Heart Disease repository ($N = 920$ total patients):

| Cohort | Clinical Center | Sample Size ($N$) | CAD Prevalence |
| :--- | :--- | :---: | :---: |
| **Cleveland** | Cleveland Clinic Foundation (Cleveland, OH, USA) | 303 | 45.87% |
| **Hungarian** | Hungarian Institute of Cardiology (Budapest, Hungary) | 294 | 36.05% |
| **Zurich** | University Hospital Zurich (Zurich, Switzerland) | 123 | 79.67% |
| **VA Long Beach** | V.A. Medical Center (Long Beach, CA, USA) | 200 | 74.50% |

### Harmonized Track B 10-Feature Representation
To evaluate transportability without data leakage or reliance on center-specific invasive diagnostic imaging, the study defines a universal 10-feature non-invasive clinical representation:

- `age`: Patient age in years
- `sex`: Biological sex ($1 = \text{male}$, $0 = \text{female}$)
- `cp`: Chest pain type (typical angina, atypical, non-anginal, asymptomatic)
- `trestbps`: Resting blood pressure on admission ($\text{mm Hg}$)
- `fbs`: Fasting blood sugar $> 120\text{ mg/dL}$ ($1 = \text{true}$, $0 = \text{false}$)
- `restecg`: Resting electrocardiographic abnormalities (normal, ST-T wave abnormality, LV hypertrophy)
- `thalach`: Maximum heart rate achieved during exercise stress testing ($\text{bpm}$)
- `exang`: Exercise-induced angina ($1 = \text{yes}$, $0 = \text{no}$)
- `oldpeak`: ST depression induced by exercise relative to rest ($\text{mm}$)
- `slope`: Slope of the peak exercise ST segment (upsloping, flat, downsloping)

*Note on Serum Cholesterol (`chol`):* In the Zurich and VA Long Beach cohorts, resting serum cholesterol contains substantial non-random missingness (recorded as non-physiological zero sentinels). The core Track B protocol accounts for this variable systematically as documented in the manuscript.

---

## 4. Evaluation Framework

The experimental architecture evaluates models across seven primary dimensions:

1. **Internal Validation:** 5-fold stratified cross-validation evaluated separately within each of the 4 cohorts across 4 classifier architectures (Table I).
2. **Cross-Dataset External Validation:** 24 directional transfer experiments evaluated across all 6 directed cohort pairs, with 1,000-iteration stratified bootstrap confidence intervals (Table II, Figure 2).
3. **Probability Calibration Analysis:** Brier score loss, Expected Calibration Error (ECE), calibration-in-the-large (intercept $\alpha$), and calibration slope ($\beta$) using logistic calibration before and after Platt scaling (Figure 3).
4. **Dataset & Covariate Shift:** Kolmogorov-Smirnov (KS) two-sample tests, Wasserstein distances, and Population Stability Index (PSI) per feature (Table IV, Figure 5).
5. **SHAP Explanation Stability:** TreeExplainer and KernelExplainer attributions evaluated for Spearman rank correlation ($\rho$), Top-5 Jaccard overlap, and explanation variance displacement across transfer pairs (Figure 4).
6. **Subgroup Robustness & Missingness:** Stratified performance across age ($\le 55$ vs. $> 55$) and biological sex, accompanied by complete-case vs. physiological KNN imputation sensitivity analysis (Figure 6).
7. **Integrated Statistical Synthesis:** 2,000-resample bootstrap meta-synthesis evaluating 14 bivariate correlations between covariate shift, calibration drift, and explanation displacement (Table V).

---

## 5. Machine Learning Models

The benchmark evaluates four primary predictive models defined in `config.yaml`:

- **Logistic Regression (L2 Regularization):** Baseline linear classifier with liblinear solver and standardization scaling.
- **Random Forest Ensemble:** 100 estimators, maximum depth of 5, minimum samples split of 4.
- **Support Vector Machine (RBF Kernel):** Radial Basis Function kernel machine ($C = 1.0$) with Platt probability scaling.
- **XGBoost (Gradient Boosted Decision Trees):** 100 estimators, max depth of 3, learning rate $\eta = 0.05$, subsample ratio of 0.8.

---

## 6. Repository Structure

```text
Research_Project/
├── README.md                           # This repository documentation
├── requirements.txt                    # Validated Python environment dependencies
├── config.yaml                         # Central experiment parameters, seeds, models, and paths
│
├── main.tex                            # Publication manuscript LaTeX source
├── references.bib                      # BibLaTeX bibliography database
├── main.pdf                            # Compiled manuscript PDF
│
├── figures/                            # 6 Canonical Publication Figures
│   ├── 01_methodological_architecture.png
│   ├── 02_external_validation_auc_heatmap.png
│   ├── 03_calibration_dataset_shift.png
│   ├── 04_shap_explanation_stability.png
│   ├── 05_dataset_shift_vs_shap_stability.png
│   └── 06_subgroup_or_missingness_robustness.png
│
├── src/                                # Core Modular Library
│   ├── __init__.py                     # Package initializer
│   ├── dataset.py                      # Multi-cohort ingestion, Track B schemas, and loaders
│   ├── preprocessing.py               # HarmonizedClinicalPreprocessor, zero masking, imputers
│   └── evaluate.py                     # Brier, ECE, calibration intercept/slope, diagnostic metrics
│
├── experiments/                        # Core Experiment Drivers
│   ├── exp_internal_validation.py      # Internal 5-fold cross-validation (Table I)
│   ├── exp_external_validation.py      # Cross-dataset transfer evaluations (Table II)
│   ├── exp_dataset_shift.py            # Covariate shift and distribution drift (Table IV)
│   ├── exp_xai_stability.py            # SHAP explanation stability analysis (Figure 4)
│   ├── exp_subgroup_robustness.py      # Age, sex, and missingness sensitivity (Figure 6)
│   └── exp_integrated_synthesis.py     # 2,000-resample integrated synthesis (Table V)
│
├── scripts/                            # Figure Generation & Numerical Audit
│   ├── generate_corrected_figures_final.py # Generates figures/ 01 to 06
│   ├── validate_corrected_figures_final.py # Programmatic audit validating figure values
│   ├── harmonize_datasets.py           # Standardizes raw cohorts into data/processed/
│   ├── profile_datasets.py             # Generates missingness and clinical baseline tables
│   └── corrections/                    # Audited numerical verification scripts
│       ├── r1_r2_calibration_internal_validation.py
│       ├── r3_svm_shap.py
│       ├── r4_dataset_shift.py
│       ├── r5_integrated_synthesis.py
│       └── r6_missingness.py
│
├── tests/                              # Automated Methodological Tests
│   ├── test_preprocessing.py
│   ├── test_internal_validation.py
│   ├── test_external_validation.py
│   ├── test_dataset_shift.py
│   ├── test_xai_stability.py
│   ├── test_subgroup_robustness.py
│   ├── test_integrated_synthesis.py
│   └── test_corrections.py
│
├── data/                               # Data Directory
│   ├── raw/                            # Target directory for user-provided raw UCI datasets
│   └── processed/                      # Harmonized 10-feature cohorts and manifest
│       ├── cleveland_track_b.csv
│       ├── hungarian_track_b.csv
│       ├── zurich_track_b.csv
│       ├── va_long_beach_track_b.csv
│       └── HARMONIZATION_MANIFEST.json
│
├── results/                            # Primary empirical results (39 benchmark outputs)
└── corrected_results/                  # Audited empirical outputs (14 verification files)
```

---

## 7. Data Availability & Raw Datasets

The reported results in the manuscript were generated exclusively from the publicly available UCI Heart Disease datasets. The synthetic heart-dataset generator is retained only as an offline fallback and was not used in the reported experiments.

**Raw UCI source datasets are NOT redistributed in this repository.**

To comply with academic distribution considerations, users wishing to execute the ingestion pipeline from raw source files should acquire the four original clinical data files directly from the official **UCI Machine Learning Repository Heart Disease study** (Janosi et al., 1988):

- `processed.cleveland.data` (or `cleveland_raw.csv`)
- `processed.hungarian.data` (or `hungarian_raw.csv`)
- `processed.switzerland.data` (or `zurich_raw.csv`)
- `processed.va.data` (or `va_raw.csv`)

Downloaded files should be placed into the `data/raw/` directory.

### Pre-Harmonized Datasets
For researchers wishing to reproduce the downstream statistical modeling directly without rebuilding the raw text parser, the preprocessed, leak-free harmonized Track B cohorts are provided in `data/processed/` along with the complete mapping manifest:
- `data/processed/cleveland_track_b.csv`
- `data/processed/hungarian_track_b.csv`
- `data/processed/zurich_track_b.csv`
- `data/processed/va_long_beach_track_b.csv`
- `data/processed/HARMONIZATION_MANIFEST.json`

---

## 8. Reproducibility Instructions

### Environment Setup
Create a virtual environment (Python 3.10 or higher recommended) and install dependencies:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Configuration
All experiment parameters, hyperparameter grids, random seeds, and relative directories are declared in `config.yaml`. No source code modifications are necessary to change cross-validation folds, seeds, or classifier parameters.

### Running the Experiments
Once dependencies are installed and raw data files are in place:

```bash
# 1. Dataset Harmonization (transforms data/raw/ -> data/processed/)
python scripts/harmonize_datasets.py

# 2. Internal Validation (Table I)
python experiments/exp_internal_validation.py

# 3. External Validation (Table II)
python experiments/exp_external_validation.py

# 4. Dataset Shift Analysis (Table IV)
python experiments/exp_dataset_shift.py

# 5. SHAP Explanation Stability (Figure 4)
python experiments/exp_xai_stability.py

# 6. Subgroup Robustness & Missingness Analysis (Figure 6)
python experiments/exp_subgroup_robustness.py

# 7. Integrated Statistical Synthesis (Table V)
python experiments/exp_integrated_synthesis.py
```

### Reproducing Figures & Validation
To regenerate the six publication figures and execute the programmatic numerical validation:

```bash
# Generate the 6 canonical publication figures into figures/
python scripts/generate_corrected_figures_final.py

# Execute the programmatic figure validation audit
python scripts/validate_corrected_figures_final.py
```

---

## 9. Automated Testing Suite

The repository includes automated unit and methodological assertion tests verifying data integrity, metric calculations, and reproducibility:

```bash
python -m unittest discover tests/
```

Test coverage encompasses:
- **Preprocessing:** Sentinel zero masking, valid Track B feature counts, and KNN imputer bounds.
- **Internal Validation:** 5-fold stratification, zero data leakage, metric calculations.
- **External Validation:** 24 transfer pairings, bootstrap confidence interval coverage.
- **Dataset Shift:** Kolmogorov-Smirnov statistics and Wasserstein distances.
- **XAI Stability:** SHAP kernel explainer reproducibility, Spearman $\rho$, and Top-$k$ Jaccard overlap.
- **Subgroup Robustness:** Age and sex partitioning consistency.
- **Integrated Synthesis:** 14 bivariate correlation estimands and cluster structures.
- **Numerical Corrections:** Calibration intercept root-finding and Platt probability monotonicity.

---

## 10. Citation

If you use this codebase, methodology, or harmonized data specifications in your research, please cite the manuscript:

```bibtex
@article{heart_disease_cross_dataset_2026,
  title   = {Cross-Dataset Evaluation of Machine Learning Models for Heart Disease Prediction: Generalization, Calibration, and Explainability},
  author  = {Raghavendra, Golla and Mani, Naveen Kumar},
  year    = {2026}
}
```

---

## 11. License

License information will be added before public release.

---

## 12. Disclaimer

This software, experimental pipeline, and documentation are provided strictly for academic and scientific research purposes. The models and outputs evaluated herein do not constitute a clinical diagnostic tool or medical device. Predictions should not be interpreted as medical advice or used to guide patient care.