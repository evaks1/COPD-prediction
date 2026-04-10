"""
Patient Chart — Doctor EHR view with embedded COPD AI risk widget (light theme).
"""

import os, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.fake_patients import FAKE_PATIENTS, PATIENT_BY_ID
from utils.preprocess import build_single_patient_row, FEATURE_LABELS
from utils.risk_update import updated_risk_score, severity_label

st.set_page_config(
    page_title="Patient Chart — MedView EHR",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base     = os.path.dirname(os.path.dirname(__file__))
    artifact = joblib.load(os.path.join(base, "models", "copd_model_v2.pkl"))
    return artifact["model"], artifact["threshold"], artifact["feature_names"]

@st.cache_resource
def load_severity_model():
    base = os.path.dirname(os.path.dirname(__file__))
    sev_path = os.path.join(base, "models", "severity_model.pkl")
    if not os.path.exists(sev_path):
        return None, None, None
    return (
        joblib.load(sev_path),
        joblib.load(os.path.join(base, "models", "severity_preprocessor.pkl")),
        joblib.load(os.path.join(base, "models", "severity_feature_cols.pkl")),
    )

model, threshold, feature_names = load_model()
sev_model, sev_preprocessor, sev_feat_cols = load_severity_model()

# ── CSS (light doctor theme) ─────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] { background: #f1f5f9; }

  [data-testid="stSidebar"] { background: #1e3a5c; border-right: none; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] hr { border-color: #2d5282 !important; }
  [data-testid="stSidebarNavItems"] a { color: #93c5fd !important; font-size: 0.87rem; }

  /* Patient header card */
  .pt-header {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex; align-items: flex-start; gap: 18px;
  }
  .big-avatar {
    width: 60px; height: 60px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1.3rem; color: white;
  }
  .pt-name-big { color: #0f172a; font-size: 1.45rem; font-weight: 700; margin: 0; }
  .pt-meta-line { color: #64748b; font-size: 0.83rem; margin-top: 4px; }
  .nhs-chip {
    display: inline-block; background: #eff6ff; color: #1d4ed8;
    border: 1px solid #bfdbfe; border-radius: 5px;
    padding: 2px 8px; font-family: monospace; font-size: 0.78rem;
  }

  /* Vital tiles */
  .vital-tile {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 14px; text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .v-val { font-size: 1.1rem; font-weight: 700; color: #0f172a; }
  .v-lbl { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase;
            letter-spacing: 0.05em; margin-top: 2px; }

  /* Section panels */
  .panel {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .panel-title {
    color: #1e3a5c; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 700; margin: 0 0 12px;
    padding-bottom: 8px; border-bottom: 1px solid #f1f5f9;
  }

  /* AI Risk widget */
  .risk-widget {
    border-radius: 12px; padding: 20px 22px; margin-bottom: 12px;
    border: 1px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .risk-widget .rw-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin-bottom: 8px; display: block;
  }
  .risk-big { font-size: 3.2rem; font-weight: 900; line-height: 1; }
  .risk-name { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }

  /* Progress bar */
  .prob-bar-bg {
    background: #f1f5f9; border-radius: 6px; height: 8px; margin: 10px 0;
  }
  .prob-bar-fill { height: 8px; border-radius: 6px; }

  /* Recommendation box */
  .rec-box {
    border-radius: 8px; padding: 14px 16px;
    font-size: 0.85rem; line-height: 1.6;
    border: 1px solid; margin-bottom: 12px;
  }

  /* Medication rows */
  .med-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #f8fafc; font-size: 0.85rem;
  }
  .med-row:last-child { border-bottom: none; }
  .med-name { color: #0f172a; font-weight: 500; }
  .med-class-chip {
    background: #eff6ff; color: #1d4ed8; border-radius: 10px;
    padding: 2px 9px; font-size: 0.71rem; border: 1px solid #bfdbfe;
  }
  .med-freq { color: #94a3b8; font-size: 0.76rem; }

  /* Lab rows */
  .lab-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f8fafc; font-size: 0.85rem;
  }
  .lab-row:last-child { border-bottom: none; }
  .lab-name { color: #475569; }
  .lab-val  { color: #0f172a; font-weight: 600; }

  /* Diagnosis chips */
  .dx-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px;
    padding: 6px 12px; margin: 4px 4px 4px 0; font-size: 0.81rem;
  }
  .dx-code { font-family: monospace; color: #d97706; font-weight: 700; }
  .dx-desc { color: #334155; }
  .dx-date { color: #94a3b8; font-size: 0.73rem; }

  /* Alert banner */
  .alert-strip {
    background: #fff1f2; border: 1px solid #fecaca; border-left: 4px solid #dc2626;
    border-radius: 8px; padding: 12px 18px; margin-bottom: 14px;
    color: #991b1b; font-size: 0.87rem;
  }
  .alert-strip strong { color: #7f1d1d; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #64748b; font-size: 0.88rem; }
  .stTabs [aria-selected="true"] { color: #1e3a5c !important; font-weight: 600; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }

  /* Buttons */
  .stButton button {
    border-radius: 7px; font-weight: 600; font-size: 0.85rem;
    background: #1e3a5c; color: white; border: none;
  }
  .stButton button:hover { background: #1d4ed8; }

  h1, h2, h3, h4 { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px;'>
      <div style='font-size:2rem;'>🏥</div>
      <div style='color:#f1f5f9; font-weight:700; font-size:1rem; margin-top:4px;'>MedView EHR</div>
      <div style='color:#94a3b8; font-size:0.72rem;'>Primary Care Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="color:#94a3b8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;">Doctor Navigation</div>', unsafe_allow_html=True)
    st.page_link("app.py",                   label="← Patient Worklist",    use_container_width=True)
    st.page_link("pages/1_Patient_Chart.py", label="📋 Patient Chart",      use_container_width=True)
    st.page_link("pages/2_Patient_Form.py",  label="📝 Patient Form",       use_container_width=True)
    st.page_link("pages/3_Model_Card.py",    label="🃏 Model Card",          use_container_width=True)

    st.markdown("---")
    st.markdown('<div style="color:#94a3b8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;">Switch Patient</div>', unsafe_allow_html=True)
    for p in FAKE_PATIENTS:
        if st.button(f"{p['photo_initials']}  {p['name']}", key=f"sw_{p['id']}", use_container_width=True):
            st.session_state["selected_patient_id"] = p["id"]
            st.rerun()

    st.markdown("---")
    st.markdown('<div style="text-align:center; font-size:0.7rem; color:#64748b; line-height:1.8;">Powered by<br><span style="color:#f97316; font-weight:800; font-size:0.9rem;">GSK</span> <span style="color:#94a3b8;">COPD AI Programme</span></div>', unsafe_allow_html=True)

# ── Select patient ─────────────────────────────────────────────────────
if "selected_patient_id" not in st.session_state:
    st.session_state["selected_patient_id"] = FAKE_PATIENTS[0]["id"]

patient  = PATIENT_BY_ID[st.session_state["selected_patient_id"]]
confirmed_copd = patient.get("confirmed_copd", False)

row        = build_single_patient_row(patient["model_inputs"])
base_proba = float(model.predict_proba(row[feature_names].values)[0, 1])

# Post-questionnaire risk update (if patient has completed the CAT form)
pid = st.session_state["selected_patient_id"]
q_key = f"questionnaire_{pid}"
questionnaire_result = st.session_state.get(q_key)

if questionnaire_result:
    risk_update = updated_risk_score(
        base_proba,
        cat_score=questionnaire_result.get("cat_total"),
        mrc_grade=questionnaire_result.get("mrc_grade"),
    )
    proba = risk_update["updated_prob"]
    questionnaire_delta = risk_update["delta"]
else:
    proba = base_proba
    questionnaire_delta = None
    risk_update = None

# Confirmed COPD patients bypass the screening probability entirely
if confirmed_copd:
    proba = 1.0
    rl = "Confirmed COPD"

predicted = int(proba >= threshold)

# GOLD severity estimate for high-risk patients
gold_severity = None
if sev_model is not None and proba >= threshold:
    inp = patient["model_inputs"]
    sev_row = {
        "AGE":          inp.get("edad", 60),
        "PackHistory":  inp.get("paquetes_ano", 0),
        "FEV1":         inp.get("fev1", np.nan),
        "FEV1PRED":     inp.get("fev1pred", np.nan),
        "FVC":          inp.get("fvc", np.nan),
        "CAT":          questionnaire_result.get("cat_total", np.nan) if questionnaire_result else np.nan,
        "gender":       inp.get("sexo_num", 1),
        "smoking":      1 if inp.get("fumador_actual") else 2,
        "Diabetes":     inp.get("has_diabetes", 0),
        "hypertension": inp.get("has_hypertension", 0),
        "IHD":          inp.get("has_heart_disease", 0),
        "MWT1Best":     np.nan,
    }
    sev_df = pd.DataFrame([[sev_row.get(c, np.nan) for c in sev_feat_cols]], columns=sev_feat_cols)
    sev_prep = sev_preprocessor.transform(sev_df)
    sev_pred = int(sev_model.predict(sev_prep)[0])
    gold_severity = severity_label(sev_pred)

# Risk colours (all light-theme)
if confirmed_copd:
    rl, rc, rbg, rborder, rec_bg, rec_border, rec_text = \
        "Confirmed COPD", "#7c3aed", "#faf5ff", "#e9d5ff", "#faf5ff", "#e9d5ff", "#4c1d95"
elif proba >= threshold:
    rl, rc, rbg, rborder, rec_bg, rec_border, rec_text = \
        "Refer", "#dc2626", "#fff1f2", "#fecaca", "#fff1f2", "#fecaca", "#7f1d1d"
elif proba >= 0.35:
    rl, rc, rbg, rborder, rec_bg, rec_border, rec_text = \
        "Monitor", "#b45309", "#fefce8", "#fde68a", "#fefce8", "#fde68a", "#713f12"
else:
    rl, rc, rbg, rborder, rec_bg, rec_border, rec_text = \
        "Low Risk", "#16a34a", "#f0fdf4", "#bbf7d0", "#f0fdf4", "#bbf7d0", "#14532d"

# ── Alert strip ────────────────────────────────────────────────────────
if confirmed_copd:
    st.markdown(f"""
    <div class="alert-strip" style="background:#faf5ff; border-color:#e9d5ff; color:#4c1d95;">
      ✅ <strong>Confirmed COPD:</strong> {patient['name']} has a documented COPD diagnosis (J44).
      Management per GOLD guidelines. Screening model not applicable.
    </div>
    """, unsafe_allow_html=True)
elif predicted:
    st.markdown(f"""
    <div class="alert-strip">
      ⚠️ <strong>COPD AI Alert:</strong> {patient['name']} is flagged for spirometry referral —
      <strong>{rl}</strong> ({proba*100:.0f}% probability).
      Spirometry referral is recommended.
    </div>
    """, unsafe_allow_html=True)

# ── Patient header ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="pt-header">
  <div class="big-avatar" style="background:{patient['photo_color']}">{patient['photo_initials']}</div>
  <div style="flex:1;">
    <p class="pt-name-big">{patient['name']}</p>
    <div class="pt-meta-line">
      {patient['age']}y · {patient['sex']} · DOB {patient['dob']} ·
      <span class="nhs-chip">{patient['nhs_number']}</span> · {patient['id']}
    </div>
    <div class="pt-meta-line" style="margin-top:5px;">
      👨‍⚕️ {patient['gp']} &nbsp;|&nbsp; 📞 {patient['phone']} &nbsp;|&nbsp; 📍 {patient['address']}
    </div>
    <div class="pt-meta-line" style="margin-top:4px; color:#d97706;">
      🩺 <em>{patient['reason_for_visit']}</em>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Vitals ──────────────────────────────────────────────────────────────
v1,v2,v3,v4,v5,v6 = st.columns(6)
for col,(lbl,val) in zip([v1,v2,v3,v4,v5,v6],[
    ("BMI", f"{patient['bmi']} kg/m²"), ("SpO₂", patient["spo2"]),
    ("BP",  patient["bp"]), ("RR", patient["rr"]),
    ("HR",  patient["hr"]), ("Last visit", patient["last_visit"]),
]):
    with col:
        st.markdown(f"""
        <div class="vital-tile">
          <div class="v-val">{val}</div><div class="v-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Two-column layout ───────────────────────────────────────────────────
chart_col, ai_col = st.columns([3, 2], gap="medium")

# ══ LEFT: Clinical chart ══════════════════════════════════════════════
with chart_col:
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "💊 Medications", "🧪 Lab Results", "📜 History"])

    with tab1:
        inp = patient["model_inputs"]
        smoke_str = (
            f"🚬 Current smoker · {inp.get('paquetes_ano',0):.0f} pack-years"
            if inp.get("fumador_actual") else
            f"Ex-smoker · {inp.get('paquetes_ano',0):.0f} pack-years"
            if inp.get("exfumador") else "Non-smoker"
        )
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">Clinical Note</div>
          <div style="color:#334155; font-size:0.9rem; line-height:1.7; font-style:italic; padding:4px 0;">
            "{patient['clinical_note']}"
          </div>
        </div>
        <div class="panel">
          <div class="panel-title">Respiratory Summary</div>
          <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <div style="font-size:0.87rem; color:#334155;">{smoke_str}</div>
            <div style="font-size:0.87rem; color:#334155;">⚡ {inp.get('exacerbaciones_ultimo_anio',0)} exacerbation(s) last year</div>
          </div>
          <div style="color:#64748b; font-size:0.8rem; margin-top:8px;">
            🔬 <em>Spirometry: {patient['spirometry_note']}</em>
          </div>
        </div>
        <div class="panel">
          <div class="panel-title">Occupation & Exposure</div>
          <div style="color:#334155; font-size:0.87rem;">{patient['occupation']}</div>
        </div>
        """, unsafe_allow_html=True)

        if patient["diagnoses"]:
            dx_html = "".join(
                f'<div class="dx-chip"><span class="dx-code">{d["code"]}</span>'
                f'<span class="dx-desc">{d["description"]}</span>'
                f'<span class="dx-date">{d["date"]}</span></div>'
                for d in patient["diagnoses"]
            )
            st.markdown(f'<div class="panel"><div class="panel-title">Active Diagnoses</div>{dx_html}</div>',
                        unsafe_allow_html=True)

    with tab2:
        if patient["medications"]:
            meds_html = "".join(
                f'<div class="med-row"><div><div class="med-name">{m["name"]}</div>'
                f'<div class="med-freq">{m["frequency"]}</div></div>'
                f'<span class="med-class-chip">{m["class"]}</span></div>'
                for m in patient["medications"]
            )
            st.markdown(f'<div class="panel"><div class="panel-title">Current Medications ({len(patient["medications"])} items)</div>{meds_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel"><div class="panel-title">Medications</div>'
                        '<div style="color:#94a3b8; font-size:0.85rem; font-style:italic;">No active medications on record</div></div>',
                        unsafe_allow_html=True)

    with tab3:
        labs_html = "".join(
            f'<div class="lab-row"><span class="lab-name">{n}</span>'
            f'<div><span class="lab-val">{v}</span> '
            f'<span style="color:#94a3b8; font-size:0.76rem;">{s}</span></div></div>'
            for n, (v, s) in patient["lab_results"].items()
        )
        st.markdown(f'<div class="panel"><div class="panel-title">Latest Laboratory Results</div>{labs_html}</div>',
                    unsafe_allow_html=True)

    with tab4:
        events = [{"date": patient["last_visit"], "type": "GP Visit",
                   "desc": patient["reason_for_visit"], "icon": "🩺"}]
        if "FEV1" in patient["spirometry_note"]:
            events.append({"date": "2024-06-12", "type": "Spirometry",
                           "desc": patient["spirometry_note"], "icon": "🫁"})
        for d in patient.get("diagnoses", []):
            events.append({"date": d["date"], "type": "Diagnosis",
                           "desc": f"{d['code']} — {d['description']}", "icon": "📋"})
        events.sort(key=lambda x: x["date"], reverse=True)

        tl_html = "".join(
            f'<div style="display:flex; gap:12px; padding:10px 0; border-bottom:1px solid #f1f5f9;">'
            f'<div style="font-size:1.2rem; padding-top:2px; flex-shrink:0;">{e["icon"]}</div>'
            f'<div><div style="color:#0f172a; font-size:0.85rem; font-weight:600;">{e["type"]}</div>'
            f'<div style="color:#475569; font-size:0.8rem;">{e["desc"]}</div>'
            f'<div style="color:#94a3b8; font-size:0.74rem; margin-top:2px;">{e["date"]}</div>'
            f'</div></div>'
            for e in events
        )
        st.markdown(f'<div class="panel"><div class="panel-title">Clinical Timeline</div>{tl_html}</div>',
                    unsafe_allow_html=True)

# ══ RIGHT: AI Risk Widget ═════════════════════════════════════════════
with ai_col:

    # Risk score card
    if confirmed_copd:
        st.markdown(f"""
        <div class="risk-widget" style="background:{rbg}; border-color:{rborder};">
          <span class="rw-label" style="color:{rc};">🩺 COPD Status</span>
          <div style="display:flex; align-items:flex-end; gap:14px;">
            <div class="risk-big" style="color:{rc}; font-size:2rem;">✅</div>
            <div>
              <div class="risk-name" style="color:{rc};">Confirmed COPD</div>
              <div style="font-size:0.74rem; color:#64748b;">Diagnosed J44 · GOLD Stage 2 (Moderate)</div>
            </div>
          </div>
          <div style="font-size:0.72rem; color:#94a3b8; margin-top:8px;">
            Screening model not applicable — patient has confirmed diagnosis.
            Manage per current GOLD guidelines.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-widget" style="background:{rbg}; border-color:{rborder};">
          <span class="rw-label" style="color:{rc};">🤖 COPD AI Risk Assessment</span>
          <div style="display:flex; align-items:flex-end; gap:14px;">
            <div class="risk-big" style="color:{rc};">{proba*100:.0f}%</div>
            <div>
              <div class="risk-name" style="color:{rc};">{rl} Risk</div>
              <div style="font-size:0.74rem; color:#64748b;">COPD probability</div>
            </div>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="background:{rc}; width:{proba*100:.1f}%;"></div>
          </div>
          <div style="font-size:0.72rem; color:#94a3b8;">
            Threshold {threshold:.2f} · LR · Sensitivity ≥91% · GOLD Standard
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Gauge (hidden for confirmed COPD patients)
    if not confirmed_copd:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix":"%", "font":{"size":28, "color":rc}},
            gauge={
                "axis": {"range":[0,100], "tickwidth":1, "tickcolor":"#cbd5e1",
                         "tickfont":{"color":"#94a3b8","size":9}},
                "bar": {"color":rc, "thickness":0.22},
                "bgcolor": "#ffffff",
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range":[0,25],  "color":"#f0fdf4"},
                    {"range":[25,45], "color":"#fefce8"},
                    {"range":[45,65], "color":"#fff1f2"},
                    {"range":[65,100],"color":"#faf5ff"},
                ],
                "threshold": {"line":{"color":"#1e3a5c","width":2},
                              "thickness":0.8, "value":threshold*100},
            },
        ))
        fig.update_layout(height=190, margin=dict(t=20,b=0,l=20,r=20),
                          paper_bgcolor="#ffffff", font={"color":"#64748b"})
        st.plotly_chart(fig, use_container_width=True)

    # Post-questionnaire update notice
    if risk_update and questionnaire_delta and questionnaire_delta > 0:
        st.markdown(f"""
        <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
                    padding:10px 14px; margin:8px 0; font-size:0.82rem; color:#1e40af;">
          📋 <strong>Questionnaire submitted</strong> — Risk updated from
          {base_proba*100:.0f}% → <strong>{proba*100:.0f}%</strong>
          (+{questionnaire_delta*100:.0f}pp from CAT/MRC results)<br>
          <span style="color:#64748b;">{risk_update['cat_interpretation']} &nbsp;·&nbsp; {risk_update['mrc_interpretation']}</span>
        </div>
        """, unsafe_allow_html=True)

    # GOLD severity estimate
    if gold_severity:
        sev_colors = {"MILD": "#16a34a", "MODERATE": "#b45309", "SEVERE": "#dc2626", "VERY SEVERE": "#9333ea"}
        sev_c = sev_colors.get(gold_severity, "#64748b")
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                    padding:10px 14px; margin:8px 0; font-size:0.82rem; color:#475569;">
          🫁 <strong>Estimated GOLD severity:</strong>
          <span style="color:{sev_c}; font-weight:700;">{gold_severity}</span>
          <span style="color:#94a3b8; font-size:0.74rem;">&nbsp;(if confirmed COPD)</span>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation
    rec_texts = {
        "Low Risk":       "✅ No immediate action needed. Reassess in 12 months or if symptoms develop.",
        "Monitor":        "⚠️ Consider spirometry at next visit. Direct patient to complete the symptom questionnaire.",
        "Refer":          "🔴 Spirometry referral recommended. Discuss with patient today and send questionnaire link.",
        "Confirmed COPD": "📋 Continue GOLD-guided management. Review inhaler technique, exacerbation action plan, and pulmonary rehabilitation eligibility.",
    }
    st.markdown(f"""
    <div class="rec-box" style="background:{rec_bg}; border-color:{rec_border}; color:{rec_text};">
      <strong>Recommendation</strong><br>{rec_texts[rl]}
    </div>
    """, unsafe_allow_html=True)

    # SHAP
    with st.expander("🔍 Key Contributing Risk Factors", expanded=(rl in ("Refer", "Confirmed COPD"))):
        try:
            import shap
            if hasattr(model, "coef_"):
                # Logistic Regression — use coefficient × feature value as attribution
                coefs = model.coef_[0]
                sv = coefs * row_prep[0]
            else:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(row_prep)[0]
            shap_df = pd.DataFrame({
                "feature": feat_cols,
                "shap": sv,
                "label": [FEATURE_LABELS.get(f, f) for f in feat_cols],
            }).sort_values("shap", key=abs, ascending=False).head(10)

            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                y=shap_df["label"][::-1],
                x=shap_df["shap"][::-1],
                orientation="h",
                marker_color=["#dc2626" if v>0 else "#16a34a" for v in shap_df["shap"][::-1]],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            fig_s.add_vline(x=0, line_width=1, line_color="#e2e8f0")
            fig_s.update_layout(
                height=300, margin=dict(l=8,r=8,t=8,b=24),
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font={"color":"#64748b","size":10},
                xaxis={"gridcolor":"#f1f5f9", "title":""},
                yaxis={"title":""},
            )
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("🔴 Increases risk &nbsp;&nbsp; 🟢 Decreases risk")
        except Exception:
            if hasattr(model, "coef_"):
                fi = pd.Series(abs(model.coef_[0]), index=feat_cols)
            else:
                fi = pd.Series(model.feature_importances_, index=feat_cols)
            fi.index = [FEATURE_LABELS.get(f,f) for f in fi.index]
            st.bar_chart(fi.sort_values(ascending=False).head(10)[::-1])

    # Actions
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">Actions</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        if st.button("🫁 Refer Spirometry", use_container_width=True):
            st.success(f"Referral created for {patient['name']}.")
    with a2:
        if st.button("📩 Patient Form", use_container_width=True):
            st.session_state["prefill_patient_id"] = patient["id"]
            st.switch_page("pages/2_Patient_Form.py")
    if st.button("← Back to Worklist", use_container_width=True):
        st.switch_page("app.py")

    st.markdown("""
    <div style="text-align:center; margin-top:14px; padding-top:12px; border-top:1px solid #f1f5f9;
                color:#94a3b8; font-size:0.7rem; line-height:1.7;">
      <span style="color:#f97316; font-weight:800;">GSK</span> COPD AI Programme<br>
      Research Prototype · Clinical decision support only
    </div>
    """, unsafe_allow_html=True)
