"""
Feature engineering pipeline for COPD prediction.
Label: GOLD spirometry standard (reproducible FEV1/FVC < 0.70 across ALL tests).

Bias note: socioeconomic level and residence zone removed as direct features —
they are healthcare access proxies, not clinical COPD risk factors.
"""

import pandas as pd
import re

DATASET_PATH  = "/Users/elsaleksandra/Downloads/dataset_epoc_advanced1"
KAGGLE_PATH   = "/Users/elsaleksandra/GSKIE/dataset.csv"

# ── Feature column order — matches copd_model_v2.pkl feature_names exactly ────
FEATURE_COLS = [
    # Demographics
    "edad", "sexo", "imc",

    # Smoking
    "fumador_actual", "exfumador", "paquetes_ano",

    # Physical activity
    "actividad_fisica",

    # Lab results
    "pcr_mg_l", "vitamina_d_ng_ml", "colesterol_total_mg_dl",
    "ferritina_ng_ml", "tsh_ui_ml",

    # Engineered
    "ever_smoked", "crp_elevated", "vitamin_d_deficient", "age_x_packyears",

    # Comorbidities
    "has_asthma", "has_hypertension", "has_diabetes", "has_ihd", "has_depression",
    "n_unique_diag",

    # NLP from clinical notes
    "kw_dyspnea", "kw_cough", "kw_sputum", "kw_exacerbation",
    "kw_breathless", "kw_no_symptoms", "kw_wheezing", "keyword_score",
]

FEATURE_LABELS = {
    "edad": "Age",
    "sexo": "Sex (Male)",
    "imc": "BMI",
    "fumador_actual": "Current Smoker",
    "exfumador": "Ex-smoker",
    "paquetes_ano": "Pack-Years",
    "actividad_fisica": "Physical Activity Level",
    "pcr_mg_l": "C-Reactive Protein (mg/L)",
    "vitamina_d_ng_ml": "Vitamin D (ng/mL)",
    "colesterol_total_mg_dl": "Total Cholesterol (mg/dL)",
    "ferritina_ng_ml": "Ferritin (ng/mL)",
    "tsh_ui_ml": "TSH (mIU/mL)",
    "ever_smoked": "Ever Smoked",
    "crp_elevated": "Elevated CRP (>10 mg/L)",
    "vitamin_d_deficient": "Vitamin D Deficiency (<20 ng/mL)",
    "age_x_packyears": "Age × Pack-Years",
    "has_asthma": "Asthma (J45)",
    "has_hypertension": "Hypertension (I10)",
    "has_diabetes": "Type 2 Diabetes (E11)",
    "has_ihd": "Ischaemic Heart Disease (I25)",
    "has_depression": "Depression (F32)",
    "n_unique_diag": "Unique Diagnoses",
    "kw_dyspnea": "Clinical Note: Dyspnoea",
    "kw_cough": "Clinical Note: Cough",
    "kw_sputum": "Clinical Note: Sputum",
    "kw_exacerbation": "Clinical Note: Exacerbation",
    "kw_breathless": "Clinical Note: Breathlessness",
    "kw_no_symptoms": "Clinical Note: No Symptoms",
    "kw_wheezing": "Clinical Note: Wheezing",
    "keyword_score": "Clinical Note: Keyword Score",
}

# ── NLP helpers ───────────────────────────────────────────────────────────────
_RESP_KEYWORDS = [
    r"disnea", r"tos", r"sibilancias", r"expectoraci[oó]n",
    r"exacerbaci[oó]n", r"bronquitis", r"neumon[ií]a",
    r"inhalador", r"tabaco", r"fumador", r"jadeo", r"ahogo",
    r"fatiga", r"hipoxia", r"cianosis", r"hemoptisis",
]
_KEYWORD_RE = re.compile("|".join(_RESP_KEYWORDS), re.IGNORECASE)



def _extract_note_features(note: str) -> dict:
    note = str(note).lower()
    kw_count = len(_KEYWORD_RE.findall(note))
    return {
        "kw_dyspnea":    int("disnea" in note),
        "kw_cough":      int("tos" in note),
        "kw_sputum":     int("expectoraci" in note or "esputo" in note),
        "kw_exacerbation": int("exacerbaci" in note),
        "kw_breathless": int("ahogo" in note or "jadeo" in note or "falta de aire" in note),
        "kw_no_symptoms": int("sin s" in note and "ntomas" in note),
        "kw_wheezing":   int("sibilancias" in note or "pitos" in note),
        "keyword_score": min(kw_count, 10),
    }


def _extract_note_features_series(notes: pd.Series) -> pd.DataFrame:
    records = [_extract_note_features(n) for n in notes]
    return pd.DataFrame(records)


# ── Label derivation ──────────────────────────────────────────────────────────
def derive_copd_label(spirometry: pd.DataFrame) -> pd.DataFrame:
    """
    GOLD standard: persistent (reproducible) airflow limitation.
    COPD = ALL spirometry measurements for a patient show FEV1/FVC < 0.70.
    Requires consistent obstruction across every recorded test.
    """
    agg = spirometry.groupby("id_paciente")["fev1_fvc_ratio"].agg(
        min_ratio="min", n_tests="count"
    ).reset_index()

    all_below = spirometry.groupby("id_paciente")["fev1_fvc_ratio"].apply(
        lambda x: (x < 0.70).all()
    ).reset_index(name="all_below_threshold")

    worst = agg.merge(all_below, on="id_paciente")
    worst["copd"] = worst["all_below_threshold"].astype(int)

    # GOLD severity stage (using raw FEV1 as proxy for % predicted)
    fev1_at_worst = (
        spirometry.sort_values("fev1_fvc_ratio")
        .groupby("id_paciente")
        .first()[["fev1"]]
        .reset_index()
    )
    worst = worst.merge(fev1_at_worst, on="id_paciente", how="left")

    def gold_stage(row):
        if not row["all_below_threshold"]:
            return 0
        fev1 = row["fev1"]
        if fev1 >= 2.5:   return 1
        elif fev1 >= 1.5: return 2
        elif fev1 >= 0.8: return 3
        else:             return 4

    worst["gold_stage"] = worst.apply(gold_stage, axis=1)
    return worst[["id_paciente", "copd", "gold_stage", "min_ratio"]]


# ── Medication features ───────────────────────────────────────────────────────
def build_medication_features(medications: pd.DataFrame) -> pd.DataFrame:
    COPD_CLASSES = {"LAMA", "LABA", "LABA+LAMA", "LABA+LAMA+ICS"}
    RESP_CLASSES  = {"LAMA", "LABA", "SABA", "LABA+LAMA", "LABA+LAMA+ICS",
                     "CS_systemic", "Resp_antibiotic"}

    med_list = medications.groupby("id_paciente")["clase_farmacologica"].apply(list).reset_index()
    med_list["n_medications"]        = med_list["clase_farmacologica"].apply(len)
    med_list["respiratory_med_count"] = med_list["clase_farmacologica"].apply(
        lambda x: len(set(x) & RESP_CLASSES)
    )
    med_list["has_copd_medication"]  = med_list["clase_farmacologica"].apply(
        lambda x: int(bool(set(x) & COPD_CLASSES))
    )

    class_map = {
        "LAMA":         "med_lama",
        "LABA":         "med_laba",
        "SABA":         "med_saba",
        "LABA+LAMA":    "med_laba_lama",
        "LABA+LAMA+ICS":"med_laba_lama_ics",
        "CS_systemic":  "med_cs_systemic",
        "Resp_antibiotic":"med_resp_antibiotic",
    }
    for cls, col in class_map.items():
        med_list[col] = med_list["clase_farmacologica"].apply(lambda x: int(cls in x))

    return med_list.drop(columns=["clase_farmacologica"])


# ── Clinical event features ───────────────────────────────────────────────────
def build_event_features(clinical_events: pd.DataFrame) -> pd.DataFrame:
    grp = clinical_events.groupby("id_paciente")["cie10"].apply(list).reset_index()
    grp["n_diagnoses"]        = grp["cie10"].apply(len)
    grp["n_unique_diagnoses"] = grp["cie10"].apply(lambda x: len(set(x)))

    icd_map = {
        "J44": "has_copd_diagnosis",
        "J45": "has_asthma",
        "I10": "has_hypertension",
        "E11": "has_diabetes",
        "I25": "has_heart_disease",
        "F32": "has_depression",
    }
    for code, col in icd_map.items():
        grp[col] = grp["cie10"].apply(lambda x: int(code in x))

    return grp.drop(columns=["cie10"])


# ── Categorical encoding ──────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sexo_num"]    = (df["sexo"] == "M").astype(int)
    # NOTE: socioeconomic level and residence zone intentionally excluded —
    # they are healthcare access proxies, not COPD clinical risk factors.
    df["actividad_num"] = df["actividad_fisica"].map(
        {"baja": 0, "media": 1, "alta": 2}
    ).fillna(1)
    return df


# ── Engineered features ───────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interaction: age × pack-years (core COPD risk product)
    df["age_x_packyears"] = df["edad"] * df["paquetes_ano"].fillna(0)

    # Ever smoked flag
    df["ever_smoked"] = ((df["fumador_actual"].fillna(0) == 1) |
                         (df["exfumador"].fillna(0) == 1)).astype(int)

    # High-risk profile: older smoker with significant exposure
    df["high_risk_smoker"] = (
        (df["edad"] > 55) &
        (df["ever_smoked"] == 1) &
        (df["paquetes_ano"].fillna(0) > 10)
    ).astype(int)

    # Exacerbation severity score (moderate = 2×, severe = 4× weight)
    df["exacerbation_severity_score"] = (
        df["exacerbaciones_ultimo_anio"].fillna(0) +
        2 * df["n_exacerbaciones_moderadas"].fillna(0) +
        4 * df["n_exacerbaciones_graves"].fillna(0)
    )

    # Inflammation flags
    df["crp_elevated"]       = (df["pcr_mg_l"].fillna(0) > 10).astype(int)
    df["vitamin_d_deficient"] = (
        (df["vitamina_d_ng_ml"].fillna(99) < 20) & (df["vitamina_d_ng_ml"].fillna(99) > 0)
    ).astype(int)

    # Comorbidity burden (sum of binary comorbidity flags)
    comorbidity_cols = ["has_copd_diagnosis", "has_asthma", "has_hypertension",
                        "has_diabetes", "has_heart_disease", "has_depression"]
    df["comorbidity_index"] = df[comorbidity_cols].fillna(0).sum(axis=1)

    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────
def load_all_data():
    p = DATASET_PATH
    return (
        pd.read_csv(f"{p}/patients.csv"),
        pd.read_csv(f"{p}/spirometry.csv"),
        pd.read_csv(f"{p}/habits.csv"),
        pd.read_csv(f"{p}/lab_results.csv"),
        pd.read_csv(f"{p}/medications.csv"),
        pd.read_csv(f"{p}/clinical_events.csv"),
        pd.read_csv(f"{p}/exacerbations.csv"),
        pd.read_csv(f"{p}/clinical_notes.csv"),
    )


def build_feature_matrix() -> tuple:
    """
    Full pipeline: load → engineer → encode → return (X, y, meta_df).
    """
    (patients, spirometry, habits, lab_results,
     medications, clinical_events, exacerbations, clinical_notes) = load_all_data()

    labels = derive_copd_label(spirometry)

    df = patients.copy()
    df = df.merge(habits, on="id_paciente", how="left")
    df = df.merge(lab_results, on="id_paciente", how="left")
    df = df.merge(exacerbations, on="id_paciente", how="left")
    df = df.merge(build_medication_features(medications), on="id_paciente", how="left")
    df = df.merge(build_event_features(clinical_events), on="id_paciente", how="left")
    df = df.merge(labels, on="id_paciente", how="left")

    # NLP from clinical notes
    note_feats = _extract_note_features_series(clinical_notes["nota_clinica"])
    note_feats["id_paciente"] = clinical_notes["id_paciente"]
    df = df.merge(note_feats, on="id_paciente", how="left")

    df = encode_categoricals(df)

    # Fill flags
    flag_cols = [c for c in df.columns if c.startswith(("med_", "has_", "n_", "note_"))]
    df[flag_cols] = df[flag_cols].fillna(0)

    df = add_engineered_features(df)

    X = df[FEATURE_COLS].copy()
    y = df["copd"].fillna(0).astype(int)
    meta = df[["id_paciente", "gold_stage"]].copy()

    return X, y, meta


# ── Single-patient inference ──────────────────────────────────────────────────
def build_single_patient_row(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row feature dataframe for real-time inference.
    Column names match copd_model_v2.pkl feature_names exactly.
    """
    d = dict(inputs)

    edad      = d.get("edad", 50)
    pack_years = d.get("paquetes_ano", 0) or 0
    fumador   = d.get("fumador_actual", 0) or 0
    exfumador = d.get("exfumador", 0) or 0
    pcr       = d.get("pcr_mg_l") or 0
    vitd      = d.get("vitamina_d_ng_ml") or 0

    # Engineered features
    d["ever_smoked"]      = int(fumador or exfumador)
    d["crp_elevated"]     = int(pcr > 10)
    d["vitamin_d_deficient"] = int(0 < vitd < 20)
    d["age_x_packyears"]  = edad * pack_years

    # Input key → model key remapping
    d["sexo"]           = d.get("sexo_num", 0)
    d["actividad_fisica"] = d.get("actividad_num", 1)
    d["has_ihd"]        = d.get("has_heart_disease", 0)
    d["n_unique_diag"]  = d.get("n_unique_diagnoses", 0)

    # NLP from free-text note
    note = d.pop("clinical_note", "") or ""
    d.update(_extract_note_features(note))

    row = {col: d.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([row])
