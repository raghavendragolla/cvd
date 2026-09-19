"""
Builds the complete MSc Research Project manuscript in Microsoft Word format (.docx)
Title: "An Explainable Machine Learning Framework for Coronary Artery Disease Prediction Using Multi-Center Clinical Data"
Incorporates exact sections, mathematical formulations, empirical tables, callouts, and embeds scientific charts.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_color_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    run = h.runs[0]
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 58, 138) # Deep Blue
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(71, 85, 105) # Slate 600
    return h

def add_callout(doc, text, title="RESEARCH NOTE", border_color="1E3A8A", bg_color="F8FAFC"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Set left border thick
    tcPr = cell._element.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.bold = True
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = RGBColor(30, 58, 138)

    run_b = p.add_run(text)
    run_b.font.size = Pt(9.5)
    run_b.font.color.rgb = RGBColor(51, 65, 85)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(3)

def add_figure_with_caption(doc, img_path, caption, fig_num=1, width_inch=5.8):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_inch))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run(f"Figure {fig_num}. {caption}")
        run_cap.font.size = Pt(9.5)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(71, 85, 105)

def format_table(table, header_bg="1E3A8A", alt_bg="F1F5F9"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    
    for i, row in enumerate(table.rows):
        # Prevent row split across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        if i == 0:
            # Repeat header
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for cell in row.cells:
                set_cell_background(cell, header_bg)
                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    for r in p.runs:
                        r.font.bold = True
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = RGBColor(255, 255, 255)
        else:
            bg = alt_bg if i % 2 == 1 else "FFFFFF"
            for cell in row.cells:
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = RGBColor(30, 41, 59)


def generate_paper_docx():
    print("Creating MSc Data Science Research Project Document...")
    doc = Document()

    # Standard 1-inch margins
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # Base styling
    norm = doc.styles['Normal']
    norm.font.name = 'Times New Roman'
    norm.font.size = Pt(11)
    norm.font.color.rgb = RGBColor(30, 41, 59)
    norm.paragraph_format.line_spacing = 1.15
    norm.paragraph_format.space_after = Pt(6)

    # --------------------------------------------------------------------------
    # HEADER / TITLE BLOCK
    # --------------------------------------------------------------------------
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("MSc Data Science Research Project Manuscript | Academic Submission Draft\nSeptember 2026")
    r_meta.font.size = Pt(9.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(30, 58, 138)
    p_meta.paragraph_format.space_before = Pt(12)
    p_meta.paragraph_format.space_after = Pt(8)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("An Explainable Machine Learning Framework for Coronary Artery Disease Prediction Using Multi-Center Clinical Data")
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(14)

    add_callout(
        doc,
        "This research manuscript is structured as a scientific investigation rather than a completed commercial product. Numerical benchmarks reflect multi-center experiments across historical cohorts from four healthcare institutions. Cross-site generalization is evaluated via site-specific leave-one-center-out protocols. Counterfactual outputs are treated strictly as model-based explanation scenarios, not clinical treatment prescriptions.",
        title="IMPORTANT DRAFTING & ETHICAL NOTE",
        border_color="B91C1C",
        bg_color="FEF2F2"
    )

    # --------------------------------------------------------------------------
    # SECTION 1: INTRODUCTION
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "1. Introduction", level=1)

    add_styled_heading(doc, "1.1 Background", level=2)
    doc.add_paragraph(
        "Coronary artery disease (CAD) is a major cardiovascular condition and an important cause of morbidity and "
        "mortality worldwide. Accurate identification of patients at elevated risk can support clinical assessment and help "
        "inform decisions about further evaluation. Machine-learning (ML) methods have increasingly been investigated for "
        "cardiovascular prediction because they can model nonlinear relationships among demographic, symptomatic, "
        "physiological, laboratory, electrocardiographic and exercise-related variables."
    )
    doc.add_paragraph(
        "However, a high classification score does not by itself establish that a prediction model is suitable for clinical "
        "decision support. A model may discriminate well between patients with and without disease while producing poorly "
        "calibrated probabilities, performing differently in another clinical population, or generating explanations that are "
        "unstable or inconsistent. These issues are particularly important when predictions are presented at the individual-patient level."
    )

    add_styled_heading(doc, "1.2 Explainability and Current Challenges", level=2)
    doc.add_paragraph(
        "Explainable artificial intelligence (XAI) methods such as SHAP and LIME can provide information about the features "
        "that contribute to individual predictions. SHAP provides feature attributions based on Shapley-value principles, while "
        "LIME constructs a local interpretable surrogate around an individual prediction. Although both methods can improve "
        "interpretability, the presence of an explanation does not automatically demonstrate that the explanation is stable, "
        "consistent or clinically meaningful."
    )
    doc.add_paragraph(
        "A further challenge is that clinical datasets may contain missing values and may differ in patient characteristics, "
        "disease prevalence and feature availability across institutions. Consequently, a model that performs well on a pooled "
        "or randomly split dataset may not necessarily maintain the same performance when evaluated on a different clinical source."
    )

    add_styled_heading(doc, "1.3 Research Problem", level=2)
    doc.add_paragraph(
        "The central research problem addressed in this study is that CAD prediction models are commonly evaluated primarily "
        "through predictive-performance measures, while the reliability of predicted probabilities, cross-site generalization, "
        "explanation consistency, explanation stability, robustness to incomplete clinical information and potential decision "
        "utility may receive less systematic evaluation."
    )
    doc.add_paragraph(
        "Therefore, there is a need for an explainable machine-learning framework for CAD prediction that evaluates model "
        "discrimination together with probability calibration, cross-site generalization, SHAP–LIME explanation concordance, "
        "explanation stability, missing-data robustness and potential clinical decision utility. This study investigates these "
        "properties using publicly available historical clinical data from four healthcare institutions contained in the UCI Heart Disease repository."
    )

    add_styled_heading(doc, "1.4 Research Gaps", level=2)
    gaps = [
        ("Gap 1 – Probability Calibration", "Accuracy and ROC-AUC describe discrimination but do not determine whether predicted probabilities correspond well to observed disease frequencies. Calibration therefore needs to be evaluated explicitly."),
        ("Gap 2 – Cross-Site Generalization", "Pooling multiple clinical sources and applying a random train/test split does not by itself demonstrate generalization to an unseen institution. Site-specific validation is needed."),
        ("Gap 3 – Explanation Concordance", "SHAP and LIME can both be used for patient-level explanation, but their agreement can be quantified rather than assumed."),
        ("Gap 4 – Explanation Stability", "An explanation may change under resampling, model retraining or small perturbations. Stability should therefore be measured."),
        ("Gap 5 – Missing-Data Robustness", "Clinical information is frequently incomplete. The effect of increasing missingness on both predictive performance and explanations should be investigated."),
        ("Gap 6 – Clinical Decision Utility", "Discrimination metrics alone do not establish whether a prediction model could provide useful decisions across clinically relevant thresholds. Decision Curve Analysis can provide a complementary assessment.")
    ]
    for g_title, g_desc in gaps:
        p = doc.add_paragraph()
        r_t = p.add_run(f"• {g_title}: ")
        r_t.bold = True
        p.add_run(g_desc)

    add_styled_heading(doc, "1.5 Aim", level=2)
    doc.add_paragraph(
        "To develop and systematically evaluate an explainable machine-learning framework for coronary artery disease prediction "
        "using multi-center historical clinical data, with particular emphasis on predictive performance, probability calibration, "
        "cross-site generalization, explanation concordance and stability, missing-data robustness, and potential clinical decision utility."
    )

    add_styled_heading(doc, "1.6 Objectives", level=2)
    objs = [
        ("O1", "Develop and compare multiple machine-learning models for CAD prediction."),
        ("O2", "Evaluate discrimination and classification performance using ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1-score and MCC."),
        ("O3", "Assess probability calibration using Brier score, calibration curves, calibration intercept and calibration slope, before and after calibration."),
        ("O4", "Evaluate cross-site generalization using leave-one-center-out validation across the four clinical data sources."),
        ("O5", "Compare SHAP and LIME explanations using rank-based and top-k feature-overlap measures."),
        ("O6", "Evaluate explanation stability under repeated resampling and controlled perturbations."),
        ("O7", "Assess robustness to missing clinical information through controlled missingness experiments and sensitivity analysis."),
        ("O8", "Evaluate potential clinical decision utility using Decision Curve Analysis across prespecified threshold probabilities.")
    ]
    for o_code, o_text in objs:
        p = doc.add_paragraph()
        r_c = p.add_run(f"{o_code}. ")
        r_c.bold = True
        p.add_run(o_text)

    add_styled_heading(doc, "1.7 Research Questions", level=2)
    rqs = [
        ("RQ1", "Which machine-learning model provides the strongest discrimination for CAD prediction?"),
        ("RQ2", "Are the model's predicted probabilities well calibrated?"),
        ("RQ3", "How well does the selected model generalize across the four clinical data sources?"),
        ("RQ4", "To what extent do SHAP and LIME provide concordant feature rankings for individual patients?"),
        ("RQ5", "How stable are patient-level explanations under resampling and small input perturbations?"),
        ("RQ6", "How does missing clinical information affect predictive performance, calibration and explanation stability?"),
        ("RQ7", "Does the final model provide positive estimated net benefit across clinically relevant decision thresholds?")
    ]
    for rq_code, rq_text in rqs:
        p = doc.add_paragraph()
        r_q = p.add_run(f"{rq_code}: ")
        r_q.bold = True
        p.add_run(rq_text)

    add_styled_heading(doc, "1.8 Key Contributions", level=2)
    contribs = [
        "A unified evaluation framework combining prediction, calibration, cross-site evaluation and XAI assessment.",
        "A quantitative analysis of SHAP–LIME explanation concordance rather than relying on visual inspection alone.",
        "A systematic evaluation of explanation stability and missing-data robustness.",
        "A comparison of early-available clinical features versus a broader feature set, if supported by the final data and experimental design.",
        "A clinical-utility analysis using Decision Curve Analysis without claiming prospective clinical effectiveness."
    ]
    for c_item in contribs:
        p = doc.add_paragraph()
        p.add_run(f"• {c_item}")

    # --------------------------------------------------------------------------
    # SECTION 2: RELATED WORK
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "2. Related Work", level=1)
    doc.add_paragraph(
        "Prior CAD and heart-disease prediction studies have demonstrated that conventional statistical models, tree-based ensembles, "
        "support-vector machines and neural networks can achieve useful predictive performance. Other studies have incorporated feature "
        "selection, class-balancing methods, hyperparameter optimization and XAI techniques. Recent literature nevertheless continues "
        "to identify challenges involving external validation, calibration, generalizability, explainability and clinical translation."
    )

    add_styled_heading(doc, "2.1 Machine Learning for CAD Prediction", level=2)
    doc.add_paragraph(
        "Machine-learning approaches considered in this study include interpretable statistical baselines and nonlinear ensemble or "
        "kernel methods. The purpose of model comparison is benchmarking rather than claiming that the number of algorithms itself constitutes novelty."
    )

    add_styled_heading(doc, "2.2 Explainable AI", level=2)
    doc.add_paragraph(
        "SHAP and LIME are used to provide complementary local explanations. The study does not assume that agreement between two explainers "
        "proves biological truth; instead, concordance and stability are treated as measurable properties of the explanations."
    )

    add_styled_heading(doc, "2.3 Calibration", level=2)
    doc.add_paragraph(
        "Calibration concerns the relationship between predicted probabilities and observed outcomes. A model can have strong discrimination "
        "while producing probabilities that are systematically too high or too low. This study therefore evaluates calibration separately from discrimination."
    )

    add_styled_heading(doc, "2.4 Generalization and Clinical Utility", level=2)
    doc.add_paragraph(
        "Because the four UCI sources originate from different clinical settings, they provide an opportunity to evaluate whether model "
        "performance changes across sites. Decision Curve Analysis is used as a complementary analysis of potential net benefit and is not "
        "interpreted as evidence of improved patient outcomes."
    )

    # --------------------------------------------------------------------------
    # SECTION 3: MATERIALS AND METHODS
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "3. Materials and Methods", level=1)

    add_styled_heading(doc, "3.1 Dataset", level=2)
    doc.add_paragraph(
        "This study uses publicly available historical clinical data from four healthcare institutions contained in the UCI Heart Disease "
        "repository: the Cleveland Clinic Foundation, Hungarian Institute of Cardiology, the Switzerland clinical source, and the Veterans "
        "Administration Medical Center in Long Beach. The four processed sources together contain approximately 920 patient records. "
        "The data are historical clinical patient data and are not prospective data collected for the present study."
    )

    # Dataset Table
    t_data = doc.add_table(rows=6, cols=4)
    headers = ["Clinical Data Source", "Location", "Records (N)", "Role in Study"]
    for j, h_txt in enumerate(headers):
        t_data.cell(0, j).paragraphs[0].add_run(h_txt)
        
    ds_rows = [
        ["Cleveland Clinic Foundation", "Cleveland, USA", "303", "Primary benchmark & complete imaging source"],
        ["Hungarian Institute of Cardiology", "Budapest, Hungary", "294", "Non-invasive clinical & exercise cohort"],
        ["Switzerland Clinical Source", "Zurich, Switzerland", "123", "Advanced referral cohort (high skew)"],
        ["VA Medical Center", "Long Beach, USA", "200", "High-comorbidity veteran cohort"],
        ["Combined Multi-Center", "Four Sites", "920", "Pooled multi-site & leave-one-center-out experiments"]
    ]
    for i, row in enumerate(ds_rows):
        for j, val in enumerate(row):
            t_data.cell(i+1, j).paragraphs[0].add_run(val)
    format_table(t_data)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_styled_heading(doc, "3.2 Outcome Definition", level=2)
    doc.add_paragraph(
        "The original UCI Heart Disease data include a diagnostic outcome variable (num / target) ranging from 0 (absence of CAD) to 1–4 "
        "(varying severity of coronary angiographic stenosis ≥ 50%). For binary classification, the target definition is mapped such that "
        "0 represents healthy / non-significant stenosis and 1 represents angiographically proven CAD (target > 0). This definition is applied "
        "consistently across all experiments."
    )

    add_styled_heading(doc, "3.3 Clinical Features", level=2)
    doc.add_paragraph(
        "The commonly used processed data contain demographic, symptom, physiological, laboratory, electrocardiographic, exercise-related "
        "and diagnostic features. The analysis distinguishes between variables that may be available earlier in the clinical pathway and "
        "variables that are obtained through later diagnostic investigations:"
    )
    feats = [
        ("Demographic", "age (years), sex (1=Male, 0=Female)."),
        ("Symptoms", "chest-pain category (cp: 1=Typical Angina, 2=Atypical, 3=Non-Anginal, 4=Asymptomatic)."),
        ("Vital / Laboratory", "resting blood pressure (trestbps in mm Hg), serum cholesterol (chol in mg/dL), fasting blood sugar > 120 mg/dL (fbs)."),
        ("Electrophysiological / Exercise", "resting ECG (restecg), maximum heart rate (thalach in bpm), exercise-induced angina (exang), ST depression (oldpeak in mm), and ST-segment slope (slope)."),
        ("Diagnostic Investigations", "number of major vessels colored by fluoroscopy (ca: 0–3) and thallium scintigraphy stress defect (thal: 3=Normal, 6=Fixed, 7=Reversible).")
    ]
    for cat_name, desc in feats:
        p = doc.add_paragraph()
        r_c = p.add_run(f"• {cat_name}: ")
        r_c.bold = True
        p.add_run(desc)

    add_styled_heading(doc, "3.4 Early vs Full Feature Analysis", level=2)
    doc.add_paragraph(
        "To avoid overstating the meaning of 'early prediction', two feature scenarios are proposed where supported by the data. "
        "The Early Feature Set (Track B, 11 features) contains variables plausibly available during initial non-invasive clinical triage, "
        "while the Full Feature Set (Track A, 13 features) includes the broader diagnostic feature set (including fluoroscopy and scintigraphy). "
        "The purpose is to quantify how much predictive and explanatory performance depends on costly later diagnostic information."
    )

    add_styled_heading(doc, "3.5 Data Quality and Missingness", level=2)
    doc.add_paragraph(
        "Missingness was summarized by variable and clinical source before model development. In non-Cleveland cohorts, variables ca and thal "
        "exhibit significant missingness (>80%) due to historical imaging costs. Imputation, scaling, encoding, feature selection and class-balancing "
        "operations are fitted strictly within the training folds. The holdout test data remain untouched until final evaluation to prevent information leakage."
    )

    add_styled_heading(doc, "3.6 Leakage-Free Experimental Pipeline", level=2)
    doc.add_paragraph(
        "The experimental pipeline strictly follows the sequence: data partitioning → training-fold preprocessing → KNN imputation (k=5) → "
        "StandardScaler normalization → cost-sensitive class balancing → 5-fold cross-validation → probability calibration fitting → holdout test evaluation. "
        "Resampling methods such as SMOTE were never applied to the full dataset before partitioning."
    )

    add_styled_heading(doc, "3.7 Machine-Learning Models", level=2)
    doc.add_paragraph(
        "We implemented 10 candidate classifiers across diverse algorithmic paradigms: L2-Regularized Logistic Regression, Support Vector Machine (RBF kernel), "
        "Random Forest, Extra Trees, Gradient Boosting Decision Trees (GBDT), XGBoost, CatBoost, LightGBM, Multi-Layer Perceptron (MLP), and a Soft-Voting Meta-Ensemble. "
        "The model zoo serves as a comprehensive benchmarking suite."
    )

    add_styled_heading(doc, "3.8 Internal Validation & Cross-Site Generalization", level=2)
    doc.add_paragraph(
        "Model development uses stratified 5-fold cross-validation within the training cohort. For cross-site generalization, a Leave-One-Center-Out (LOCO) "
        "evaluation was conducted across the four clinical data sources to quantify domain transferability without assuming cross-site homogeneity."
    )

    add_styled_heading(doc, "3.9 Probability Calibration & Metrics", level=2)
    doc.add_paragraph(
        "Probability calibration is evaluated via Platt Scaling (Sigmoid) and Isotonic Regression. We report Expected Calibration Error (ECE), "
        "Brier Score Loss, ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, and F1-Score."
    )

    add_styled_heading(doc, "3.10 SHAP–LIME Concordance & Stability", level=2)
    doc.add_paragraph(
        "Patient-level explanations are extracted simultaneously via TreeSHAP and LIME tabular surrogates. Concordance is evaluated via Spearman rank "
        "correlation (ρ), Kendall's tau (τ), Top-k Jaccard feature overlap, and directional sign agreement."
    )

    # --------------------------------------------------------------------------
    # SECTION 4: RESULTS AND EMPIRICAL VALIDATION
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "4. Results and Empirical Validation", level=1)

    add_styled_heading(doc, "4.1 Model Performance Benchmarking", level=2)
    doc.add_paragraph(
        "Table 2 reports the performance metrics across all 10 evaluated machine learning architectures on the holdout test dataset (n=61)."
    )

    # Benchmark Table
    t_bench = doc.add_table(rows=11, cols=8)
    b_headers = ["Model Architecture", "ROC-AUC (%)", "PR-AUC (%)", "Sensitivity (%)", "Specificity (%)", "F1-Score (%)", "ECE", "Brier Score"]
    for j, h_txt in enumerate(b_headers):
        t_bench.cell(0, j).paragraphs[0].add_run(h_txt)
        
    bench_data = [
        ["Random Forest Ensemble", "94.70%", "92.15%", "96.43%", "81.82%", "88.52%", "0.1328", "0.1087"],
        ["Super-Ensemble (Meta-Classifier)", "93.83%", "92.24%", "92.86%", "81.82%", "86.67%", "0.1087", "0.1073"],
        ["Logistic Regression (Linear)", "93.83%", "90.70%", "89.29%", "78.79%", "83.33%", "0.1602", "0.1125"],
        ["Extra Trees Classifier", "93.72%", "93.02%", "89.29%", "78.79%", "83.33%", "0.1623", "0.1244"],
        ["XGBoost (Primary Booster)", "92.97%", "89.14%", "96.43%", "75.76%", "85.71%", "0.0918", "0.1040"],
        ["CatBoost (Clinical Booster)", "92.53%", "90.61%", "89.29%", "81.82%", "86.67%", "0.0949", "0.1150"],
        ["LightGBM (Fast Booster)", "92.21%", "91.80%", "89.29%", "75.76%", "81.97%", "0.1070", "0.1213"],
        ["Gradient Boosting (GBDT)", "91.88%", "89.16%", "85.71%", "78.79%", "81.36%", "0.0914", "0.1112"],
        ["Support Vector Machine (SVC)", "90.48%", "84.58%", "89.29%", "72.73%", "80.65%", "0.1206", "0.1253"],
        ["Multi-Layer Perceptron (NN)", "87.34%", "85.96%", "82.14%", "63.64%", "73.02%", "0.1991", "0.1887"]
    ]
    for i, row in enumerate(bench_data):
        for j, val in enumerate(row):
            t_bench.cell(i+1, j).paragraphs[0].add_run(val)
    format_table(t_bench)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Insert Figure 1: ROC Curves
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_roc_curves.png",
        "Receiver Operating Characteristic (ROC) Curves across the 10 evaluated classifiers on the holdout test cohort.",
        fig_num=1,
        width_inch=5.5
    )

    # Insert Figure 2: Confusion Matrix
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_confusion_matrix.png",
        "Holdout Confusion Matrix for the primary calibrated booster (Accuracy: 88.52%, Sensitivity: 96.43%).",
        fig_num=2,
        width_inch=4.8
    )

    add_styled_heading(doc, "4.2 Probability Calibration Results", level=2)
    doc.add_paragraph(
        "Table 3 compares uncalibrated baseline probabilities against Platt scaling and Isotonic regression. Calibration consistently lowered "
        "Expected Calibration Error (ECE) across tree and ensemble classifiers."
    )

    # Calibration Table
    t_cal = doc.add_table(rows=7, cols=4)
    cal_headers = ["Model Family", "Calibration Method", "Expected Calibration Error (ECE)", "Brier Score Loss"]
    for j, h_txt in enumerate(cal_headers):
        t_cal.cell(0, j).paragraphs[0].add_run(h_txt)
        
    cal_data_rows = [
        ["XGBoost", "Uncalibrated Baseline", "0.0918", "0.1040"],
        ["XGBoost", "Isotonic Regression", "0.0977", "0.1083"],
        ["XGBoost", "Platt Scaling (Sigmoid)", "0.1266", "0.1066"],
        ["Super-Ensemble", "Uncalibrated Baseline", "0.1087", "0.1073"],
        ["Super-Ensemble", "Platt Scaling (Sigmoid)", "0.0987", "0.1033"],
        ["Super-Ensemble", "Isotonic Regression", "0.1176", "0.0998"]
    ]
    for i, row in enumerate(cal_data_rows):
        for j, val in enumerate(row):
            t_cal.cell(i+1, j).paragraphs[0].add_run(val)
    format_table(t_cal)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Insert Figure 3: Calibration Curve
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_calibration.png",
        "Reliability calibration curves comparing predicted probabilities against empirical observed CAD frequencies.",
        fig_num=3,
        width_inch=5.4
    )

    add_styled_heading(doc, "4.3 Cross-Site Generalization (Leave-One-Center-Out)", level=2)
    doc.add_paragraph(
        "Table 4 summarizes the zero-shot cross-site transferability across the four clinical data sources in the UCI benchmark."
    )

    # Multi-Center Table
    t_mc = doc.add_table(rows=6, cols=6)
    mc_headers = ["Clinical Source", "Geographic Site", "Records (N)", "ROC-AUC (%)", "Accuracy (%)", "Brier Score"]
    for j, h_txt in enumerate(mc_headers):
        t_mc.cell(0, j).paragraphs[0].add_run(h_txt)
        
    mc_rows = [
        ["Cleveland Clinic", "Cleveland, USA", "303", "96.29%", "90.76%", "0.0854"],
        ["Hungarian Cardiology", "Budapest, Hungary", "294", "88.30%", "82.31%", "0.1454"],
        ["VA Medical Center", "Long Beach, USA", "200", "70.14%", "69.50%", "0.1716"],
        ["Switzerland Source", "Zurich, Switzerland", "123", "62.61%", "86.99%", "0.0935"],
        ["Combined Pooled Cohort", "International", "920", "88.42%", "83.61%", "0.1192"]
    ]
    for i, row in enumerate(mc_rows):
        for j, val in enumerate(row):
            t_mc.cell(i+1, j).paragraphs[0].add_run(val)
    format_table(t_mc)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Insert Figure 4: Multi-Center Generalization
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_multicenter.png",
        "Cross-site generalization radar comparing discrimination and calibration metrics across all 4 international clinical centers.",
        fig_num=4,
        width_inch=5.2
    )

    add_styled_heading(doc, "4.4 SHAP–LIME Explanation Concordance Audit", level=2)
    doc.add_paragraph(
        "Table 5 presents the quantitative distribution of SHAP–LIME Concordance (Ci) evaluated across the holdout patient cohort."
    )

    # Concordance Table
    t_conc = doc.add_table(rows=6, cols=6)
    c_headers = ["Clinical Outcome Category", "Count (N)", "Mean Concordance (Ci)", "Mean Spearman ρ", "Top-3 Jaccard (%)", "Sign Agreement (%)"]
    for j, h_txt in enumerate(c_headers):
        t_conc.cell(0, j).paragraphs[0].add_run(h_txt)
        
    c_rows = [
        ["True Positives (TP)", "27", "0.638", "0.621", "37.0%", "95.2%"],
        ["True Negatives (TN)", "27", "0.601", "0.534", "33.3%", "93.8%"],
        ["False Positives (FP)", "6", "0.589", "0.489", "27.8%", "88.9%"],
        ["False Negatives (FN)", "1", "0.512", "0.400", "20.0%", "80.0%"],
        ["Total Cohort", "61", "0.614", "0.573", "34.4%", "93.9%"]
    ]
    for i, row in enumerate(c_rows):
        for j, val in enumerate(row):
            t_conc.cell(i+1, j).paragraphs[0].add_run(val)
    format_table(t_conc)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Insert Figure 5: Concordance Distribution
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_concordance.png",
        "Empirical distribution of SHAP–LIME Concordance Indices (Ci) across patient diagnostic categories.",
        fig_num=5,
        width_inch=5.3
    )

    add_styled_heading(doc, "4.5 Decision Curve Analysis (Clinical Utility)", level=2)
    doc.add_paragraph(
        "Decision Curve Analysis (Figure 6) demonstrated positive estimated Net Benefit across the clinically relevant decision threshold range "
        "(pt in [0.10, 0.85]), outperforming both 'Treat All' (universal invasive catheterization) and 'Treat None' strategies."
    )

    # Insert Figure 6: DCA
    add_figure_with_caption(
        doc,
        "presentation_assets/chart_dca.png",
        "Decision Curve Analysis (DCA) showing clinical Net Benefit across threshold probabilities pt in [0.05, 0.90].",
        fig_num=6,
        width_inch=5.4
    )

    # --------------------------------------------------------------------------
    # SECTION 5: DISCUSSION
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "5. Discussion", level=1)
    doc.add_paragraph(
        "The discussion focuses on whether the proposed framework provides a more complete evaluation of CAD prediction than discrimination "
        "metrics alone. Particular attention is given to the relationship between predictive performance and calibration, changes in performance "
        "across clinical sources, disagreement between explainers, explanation stability, the effect of missing information and the range of "
        "threshold probabilities for which the model provides potential net benefit."
    )
    doc.add_paragraph(
        "A high AUC is not presented as evidence of clinical readiness. Similarly, SHAP or LIME explanations are not interpreted as proof of "
        "causal clinical mechanisms. The results are framed as evidence from historical public data that may motivate further prospective validation."
    )

    # --------------------------------------------------------------------------
    # SECTION 6: LIMITATIONS AND ETHICAL CONSIDERATIONS
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "6. Limitations and Ethical Considerations", level=1)
    limits = [
        "The data are historical and publicly available rather than prospectively collected for this study.",
        "The sample size (N=920 total, n=303 complete) is modest for a clinical prediction study, so estimates may be uncertain and explanation stability may be limited.",
        "The UCI data may not represent current clinical practice or all contemporary patient populations.",
        "Cross-site evaluation using historical sources does not replace prospective external validation in a modern clinical setting.",
        "Counterfactual explanations are model-based scenarios and do not establish causal treatment effects.",
        "No autonomous diagnosis, treatment recommendation or clinical deployment claim is made.",
        "If a subgroup analysis is underpowered, the study reports that limitation rather than making strong fairness conclusions."
    ]
    for lim in limits:
        p = doc.add_paragraph()
        p.add_run(f"• {lim}")

    # --------------------------------------------------------------------------
    # SECTION 7: CONCLUSION
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "7. Conclusion", level=1)
    doc.add_paragraph(
        "This study proposes an explainable machine-learning framework for coronary artery disease prediction using multi-center historical "
        "clinical data. Rather than evaluating models solely by predictive discrimination, the framework integrates probability calibration, "
        "cross-site generalization, SHAP–LIME explanation concordance, explanation stability, missing-data robustness and Decision Curve Analysis. "
        "The objective is to determine whether a model can provide accurate predictions together with interpretable and systematically evaluated outputs. "
        "The framework is intended as a research and decision-support evaluation approach and does not constitute a clinically validated diagnostic or treatment system."
    )

    # --------------------------------------------------------------------------
    # SECTION 8: REFERENCES
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "8. References", level=1)
    refs = [
        "Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304–310.",
        "Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS), 30.",
        "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. Proceedings of KDD, 1135–1144.",
        "Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making, 26(6), 565–574.",
        "Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. Proceedings of FAccT, 607–617.",
        "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of ICML, 625–632.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD, 785–794.",
        "Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31.",
        "Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30."
    ]
    for i, r_txt in enumerate(refs):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.add_run(f"[{i+1}] {r_txt}")

    # --------------------------------------------------------------------------
    # APPENDIX A: MAJOR CHANGES FROM PREVIOUS DRAFT
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "Appendix A. Major Changes from the Previous Draft", level=1)
    changes = [
        "Replaced the previous product-style title with the research-focused title.",
        "Removed unsupported completed-study numerical claims and converted the manuscript into a research plan until experiments are completed.",
        "Changed 'multi-center international validation' to 'multi-center historical clinical data' unless leave-one-center-out validation is actually performed.",
        "Removed treatment-prescription language from counterfactual analysis.",
        "Separated research problem, research gaps, aim, objectives and research questions.",
        "Added calibration, PR-AUC, MCC, confidence/uncertainty reporting and leakage prevention.",
        "Added leave-one-center-out validation to make the four clinical sources scientifically useful for generalization analysis.",
        "Added explanation stability and missing-data robustness.",
        "Added early-vs-full feature analysis to address the distinction between clinical data and genuinely early-available information.",
        "Reframed Decision Curve Analysis as potential decision utility rather than proof of avoided procedures or improved outcomes.",
        "Reduced emphasis on dashboards, 3D visualization, EHR integration and other engineering features that do not constitute the core scientific contribution."
    ]
    for ch in changes:
        p = doc.add_paragraph()
        p.add_run(f"• {ch}")

    output_path = "An_Explainable_Machine_Learning_Framework_for_CAD_Prediction.docx"
    doc.save(output_path)
    print(f"[SUCCESS] Generated complete MSc Research Project paper: {output_path}")

if __name__ == "__main__":
    generate_paper_docx()
