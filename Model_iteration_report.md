# Model Iteration Report — COPD Risk Finder

This document tracks each model iteration: approach, data, features, performance, known issues, and decisions made.
It is updated each time a new model is trained and replaces the previous one in the Streamlit application.

---

## Iteration 1 — Current Deployed Model

**Status**: `DEPLOYED` (active in Streamlit app as of March 2026)
**Artifact**: `models/xgb_model.pkl` *(note: filename is misleading — this is a Logistic Regression model)*
**Branch**: `main`
**Training script**: `train_model.py`

---

### 1.1 Objective

Predict whether a primary care patient is at risk of undiagnosed COPD, using EHR-derived features, to flag patients who should be referred for spirometry.

---

### 1.2 Algorithm

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| Regularisation | L2, C = 0.1 |
| Solver | lbfgs |
| Max iterations | 1,000 |
| Random state | 42 |
| Class balancing | SMOTE (k_neighbors = 5) applied to training set |
| Preprocessing | Median imputation → StandardScaler |
| Decision threshold | Optimised for ≥ 90% recall (stored in `models/threshold.pkl`) |

---

### 1.3 Training Data

| Property | Value |
|---|---|
| Primary dataset | GSK simulated EHR — ~8,000 synthetic patients |
| Augmentation dataset | Kaggle COPD dataset — 101 patients (all COPD-positive, used to enrich minority class) |
| Train / test split | 80 / 20 stratified, random_state = 42 |
| Test set size | ~1,600 patients (reported metrics computed on this set) |
| Paths (hardcoded) | `/Users/elsaleksandra/Downloads/dataset_epoc_advanced1` and `/Users/elsaleksandra/GSKIE/dataset.csv` — **not reproducible on other machines** |

---

### 1.4 Target Variable

| Property | Value |
|---|---|
| Column | `epoc_diagnostico` (from `target.csv`) |
| Definition | Administrative COPD diagnosis as recorded by a physician in the EHR |
| Derivation | Binary: 1 = COPD diagnosed, 0 = not diagnosed |
| Label strategy for spirometry | FEV1/FVC < 0.70 across **ALL** spirometry tests per patient (Strategy A — over-strict) |

> **Known problem**: `epoc_diagnostico` encodes physician-recorded diagnoses, which reflects the underdiagnosis bias that this tool is designed to correct. Training on it means the model learns to replicate flawed physician behaviour, not to identify true COPD cases.

---

### 1.5 Feature Set — 56 Features

Features engineered in `utils/preprocess.py`. Categories:

| Category | Features (n) | Key variables |
|---|---|---|
| Demographics | 3 | `edad`, `sexo_num`, `imc` |
| Smoking | 6 | `fumador_actual`, `exfumador`, `paquetes_ano`, `ever_smoked`, `age_x_packyears`, `high_risk_smoker` |
| Physical activity | 1 | `actividad_num` |
| Lab results | 7 | `pcr_mg_l`, `crp_elevated`, `vitamina_d_ng_ml`, `vitamin_d_deficient`, `colesterol_total_mg_dl`, `ferritina_ng_ml`, `tsh_ui_ml` |
| Exacerbations | 4 | `exacerbaciones_ultimo_anio`, `n_exacerbaciones_moderadas`, `n_exacerbaciones_graves`, `exacerbation_severity_score` |
| Medications | 9 | `n_medications`, `respiratory_med_count`, `has_copd_medication`, `med_lama`, `med_laba`, `med_saba`, `med_laba_lama`, `med_laba_lama_ics`, `med_cs_systemic` |
| Comorbidities | 7 | `n_diagnoses`, `n_unique_diagnoses`, `comorbidity_index`, **`has_copd_diagnosis`** ⚠️, `has_asthma`, `has_hypertension`, `has_diabetes` |
| NLP (clinical notes) | 4 | `note_has_dyspnea`, `note_has_exacerbations`, `note_no_symptoms`, `note_respiratory_keyword_score` |
| Spirometry aggregates | ~15 | Various FEV1/FVC derived features |

---

### 1.6 Reported Performance Metrics

> ⚠️ **These metrics are NOT reliable.** See Known Issues §1.7 below. They are reported here for historical reference only.

Evaluated on the held-out test set (~1,600 patients) at the optimised sensitivity threshold.

**Confusion matrix**:

|  | Predicted No COPD | Predicted COPD |
|---|---|---|
| **Actual No COPD** | TN = 195 | FP = 1,088 |
| **Actual COPD** | FN = 65 | TP = 652 |

| Metric | Value | Note |
|---|---|---|
| Sensitivity (Recall) | **90.9%** | Primary optimisation target |
| Specificity | 15.2% | Very low — high false positive rate |
| Precision (PPV) | 37.5% | 1 in 2.7 positive predictions is correct |
| NPV | 75.0% | |
| ROC-AUC | 0.621 | Low — barely better than random (0.50) |
| F1 Score | 0.53 | |
| Optimised threshold | Stored in `models/threshold.pkl` | |

---

### 1.7 Known Issues

These issues were identified during the EDA and review phase (March 2026). They are the primary motivation for Iteration 2.

| # | Severity | Issue | Location | Impact |
|---|---|---|---|---|
| I-01 | 🔴 Critical | **Feature leakage**: `has_copd_diagnosis` (ICD-10 J44 flag) is included as a predictor. This is an administrative encoding of the target label. | `utils/preprocess.py` line 48, `FEATURE_COLS` | All reported metrics are inflated and unreliable. True performance is unknown. |
| I-02 | 🔴 Critical | **Wrong target variable**: `epoc_diagnostico` encodes physician behaviour, not disease presence. Training on it teaches the model to replicate underdiagnosis, not to find undiagnosed cases. | `train_model.py`, target derivation | Model optimises for the wrong outcome. Fundamentally misaligned with the tool's purpose. |
| I-03 | 🟠 High | **GOLD label over-strict**: FEV1/FVC < 0.70 required across ALL spirometry tests. A patient with one obstructive and one normal test is labelled negative — contradicts GOLD 2024. | `train_model.py`, label derivation | Some COPD-positive patients are mislabelled as negative in training. |
| I-04 | 🟠 High | **Misleading model filename**: artifact is `xgb_model.pkl` but contains a Logistic Regression model. | `models/xgb_model.pkl`, `app.py` | Documentation / trust issue. |
| I-05 | 🟠 High | **Hardcoded paths**: training scripts reference `/Users/elsaleksandra/...` — not reproducible on any other machine. | `train_model.py`, `train_severity_model.py` | Model cannot be retrained without manual path changes. |
| I-06 | 🟡 Medium | **Risk update heuristic**: post-questionnaire CAT/MRC blending uses hard-coded additive deltas, not a Bayesian likelihood-ratio update. | `utils/risk_update.py` | Risk updates are not statistically grounded. |
| I-07 | 🟡 Medium | **Severity model underpowered**: GOLD stage classifier trained on only 101 patients for a 4-class problem. | `train_severity_model.py`, `models/severity_model.pkl` | Severity predictions are unreliable. |
| I-08 | 🟡 Medium | **No calibration**: predicted probabilities are not calibrated. A score of 0.7 does not mean 70% probability. | Model Card, `pages/3_Model_Card.py` | Risk scores cannot be interpreted as probabilities. |
| I-09 | 🟡 Medium | **No confidence intervals**: all reported metrics are point estimates with no uncertainty bounds. | `pages/3_Model_Card.py` | Metrics appear more precise than they are. |
| I-10 | 🟢 Low | **No model comparison**: only Logistic Regression was trained. XGBoost and Random Forest were not evaluated. | `train_model.py` | Cannot confirm LR is the best algorithm for this data. |

---

### 1.8 Additional Components (Streamlit App)

| Component | Description | Status |
|---|---|---|
| Doctor worklist (`app.py`) | Patient list with risk-stratified colour badges | Functional |
| Patient chart (`pages/1_Patient_Chart.py`) | EHR view with SHAP explainability widget | Functional |
| Patient questionnaire (`pages/2_Patient_Form.py`) | CAT + MRC symptom form (patient-facing) | Functional |
| Model Card (`pages/3_Model_Card.py`) | Mitchell et al. (2019) transparency documentation | Functional |
| Risk update (`utils/risk_update.py`) | Post-questionnaire probability adjustment | Functional (heuristic — see I-06) |
| Demo patients (`utils/fake_patients.py`) | 6 synthetic patients for demo | Functional |

---

---

## Iteration 2 — Next Model (In Development)

**Status**: `IN DEVELOPMENT`
**Branch**: `feature/model-refinement`
**Training script**: TBD (`train_model_v2.py`)

---

### 2.1 Objective

Same clinical goal as Iteration 1. The fundamental change is correcting the target variable and feature set so the model learns to identify patients **whose spirometry would show obstruction**, not patients **who were previously diagnosed**.

---

### 2.2 Key Changes from Iteration 1

| Dimension | Iteration 1 | Iteration 2 |
|---|---|---|
| **Target variable** | `epoc_diagnostico` (admin diagnosis) | `copd_label` = `min(FEV1/FVC) < 0.70` (spirometry gold standard) |
| **Label strategy** | ALL tests FEV1/FVC < 0.70 | MIN ratio across all clean tests < 0.70 |
| **Feature leakage** | `has_copd_diagnosis` (J44) included | J44, `epoc_diagnostico` both excluded |
| **Training population** | All 10,000 patients | Patients with spirometry records only (characterise selection bias) |
| **Spirometry features** | Aggregates included but label-basis variable not separated | `spiro_min_ratio` excluded; `spiro_slope`, `spiro_latest_ratio`, `spiro_n_tests` as features |
| **Algorithm** | Logistic Regression only | Compare LR, XGBoost, Random Forest with nested CV |
| **Calibration** | None | Platt scaling or isotonic regression |
| **Paths** | Hardcoded (developer machine) | Relative paths from `data/raw/` |

---

### 2.3 Target Variable

| Property | Value |
|---|---|
| Column | `copd_label` (derived, not from a raw table) |
| Definition | `min(FEV1/FVC) < 0.70` across all clean spirometry tests per patient |
| Exclusions | Rows with FEV1/FVC > 1.0 removed before aggregation (physically impossible values) |
| Clinical basis | GOLD 2024 airflow obstruction criterion (pre-bronchodilator approximation) |
| Limitation | Pre-bronchodilator values used (post-BD unavailable); some patients without spirometry cannot be labelled and are excluded from training |

---

### 2.4 Planned Feature Set (~28 features)

See `notebooks/01_EDA.ipynb` — Section "Feature Plan — Variable Reference" for full specification.

**Original variables** (12): `edad`, `sexo`, `imc`, `fumador_actual`, `exfumador`, `paquetes_ano`, `actividad_fisica`, `pcr_mg_l`, `vitamina_d_ng_ml`, `colesterol_total_mg_dl`, `ferritina_ng_ml`, `tsh_ui_ml`

**Engineered features** (~16): spirometry longitudinal aggregates (`spiro_latest_ratio`, `spiro_slope`, `spiro_n_tests`, `spiro_years_span`, `spiro_mean_fev1`, `spiro_mean_fvc`), habit flags (`ever_smoked`, `age_x_packyears`, `crp_elevated`, `vitamin_d_deficient`), ICD-10 comorbidity flags (`has_asthma`, `has_hypertension`, `has_diabetes`, `has_ihd`, `has_depression`, `n_unique_diag`), NLP keywords (`kw_dyspnea`, `kw_cough`, `kw_sputum`, `kw_exacerbation`, `kw_breathless`, `kw_no_symptoms`, `kw_wheezing`, `keyword_score`)

**Excluded**: `nivel_socioeconomico`, `zona_residencia`, `grupo_sanguineo`, `spiro_min_ratio`, `spiro_max_ratio`, `spiro_mean_ratio`, `epoc_diagnostico`, `has_copd_diagnosis` (J44), `kw_copd_mention`, `kw_smoking`, `id_paciente`, raw `fecha` columns

---

### 2.5 Planned Methodology

| Step | Approach |
|---|---|
| Data splitting | Stratified 80/20, random_state = 42 |
| Class imbalance | SMOTE on training fold only (inside CV loop) |
| Preprocessing | Median imputation → StandardScaler (inside CV pipeline) |
| Model comparison | Logistic Regression, XGBoost, Random Forest evaluated under identical nested CV |
| Validation | 5-fold stratified cross-validation; test set held out until final evaluation |
| Threshold | Optimise for ≥ 90% sensitivity on validation folds |
| Calibration | Platt scaling on validation set |
| Explainability | SHAP values (TreeExplainer for tree models, LinearExplainer for LR) |
| Fairness | Stratified performance by sex, age band (< 60 / ≥ 60), smoking status |

---

### 2.6 Performance Metrics — Target

Primary metric: **Sensitivity ≥ 90%** (same as Iteration 1 — screening tool rationale unchanged).

Additional metrics to report (not tracked in Iteration 1):

| Metric | Rationale |
|---|---|
| Sensitivity with 95% CI (bootstrap) | Uncertainty quantification |
| Specificity with 95% CI | How many unnecessary spirometries |
| ROC-AUC with 95% CI | Overall discrimination |
| Precision-Recall AUC | More informative than ROC under class imbalance |
| Brier Score | Calibration quality |
| Calibration curve | Visual check: does predicted probability = observed frequency? |
| Stratified sensitivity by sex, age band | Fairness audit |

---

### 2.7 Performance Metrics — Actual

> *To be filled in after training.*

| Metric | Logistic Regression | XGBoost | Random Forest | Selected model |
|---|---|---|---|---|
| Sensitivity | — | — | — | — |
| Specificity | — | — | — | — |
| Precision (PPV) | — | — | — | — |
| ROC-AUC | — | — | — | — |
| PR-AUC | — | — | — | — |
| Brier Score | — | — | — | — |
| F1 | — | — | — | — |
| Threshold | — | — | — | — |

**Selected model**: TBD
**Rationale for selection**: TBD
**Comparison vs Iteration 1**: TBD (note: direct comparison is not valid — different target variable, different population)

---

### 2.8 Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-30 | Replace `epoc_diagnostico` with spirometry-derived `copd_label` as target | Admin diagnosis encodes underdiagnosis bias. Spirometry FEV1/FVC < 0.70 is the GOLD 2024 gold standard. |
| 2026-03-30 | Use `min(FEV1/FVC)` across all tests as label basis | COPD is irreversible. Any valid obstructive reading confirms disease. Strategy A (all tests < 0.70) was over-strict and contradicted clinical guidelines. |
| 2026-03-30 | Restrict training to patients with spirometry records | Only patients with spirometry can receive an objective label. Patients without spirometry cannot be labelled without introducing systematic bias. |
| 2026-03-30 | Exclude `spiro_min_ratio` from features | It is the exact value used to derive the label. Using it as a predictor is direct data leakage. |
| 2026-03-30 | Exclude `nivel_socioeconomico` and `zona_residencia` | Healthcare access proxies. Including them would encode systemic inequity into predictions rather than measuring disease risk. |

---

*Report maintained by the project team. Update §2.7 and add a new section for each subsequent iteration.*
