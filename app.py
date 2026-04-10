"""
COPD Screener — Doctor EHR Worklist (Light Theme)
"""

import os, sys
import joblib
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from utils.fake_patients import FAKE_PATIENTS
from utils.preprocess import build_single_patient_row

st.set_page_config(
    page_title="MedView EHR — COPD Screener",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base     = os.path.dirname(__file__)
    artifact = joblib.load(os.path.join(base, "models", "copd_model_v2.pkl"))
    return artifact["model"], artifact["threshold"], artifact["feature_names"]

model, threshold, feature_names = load_model()

def get_risk(patient):
    # Confirmed COPD patients bypass the screening model entirely
    if patient.get("confirmed_copd"):
        return None, "COPD Previously Diagnosed", "#7c3aed", "#f3e8ff", "#e9d5ff"
    row   = build_single_patient_row(patient["model_inputs"])
    proba = float(model.predict_proba(row[feature_names].values)[0, 1])
    if proba >= threshold: return proba, "High — Physician Review Needed", "#dc2626", "#fee2e2", "#fecaca"
    elif proba >= 0.35:    return proba, "Moderate — Monitor Closely",     "#b45309", "#fef9c3", "#fde68a"
    else:                  return proba, "Low",                            "#16a34a", "#dcfce7", "#bbf7d0"

if "patient_risks" not in st.session_state:
    st.session_state["patient_risks"] = {p["id"]: get_risk(p) for p in FAKE_PATIENTS}

# ── CSS (light doctor theme) ─────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] { background: #f1f5f9; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #1e3a5c;
    border-right: none;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] hr { border-color: #2d5282 !important; }
  [data-testid="stSidebar"] a { color: #93c5fd !important; }
  /* keep nav visible but styled */
  [data-testid="stSidebarNavItems"] a { color: #93c5fd !important; font-size: 0.87rem; }

  /* ── Top bar ── */
  .topbar {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .topbar-brand { display: flex; align-items: center; gap: 12px; }
  .topbar-brand .logo {
    background: #1e3a5c; color: white; border-radius: 8px;
    width: 38px; height: 38px; display: flex; align-items: center;
    justify-content: center; font-size: 1.2rem;
  }
  .topbar-brand h2 { color: #0f172a; font-size: 1.1rem; margin: 0; font-weight: 700; }
  .topbar-brand span { color: #64748b; font-size: 0.78rem; display: block; }
  .topbar-pill {
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 20px;
    padding: 5px 14px; color: #475569; font-size: 0.82rem; font-weight: 500;
  }
  .gsk-badge {
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 20px;
    padding: 5px 14px; font-size: 0.82rem; font-weight: 700; color: #c2410c;
  }

  /* ── Alert banner ── */
  .alert-banner {
    background: #fff1f2; border: 1px solid #fecaca; border-left: 4px solid #dc2626;
    border-radius: 8px; padding: 12px 18px; display: flex;
    align-items: center; gap: 12px; margin-bottom: 16px; color: #7f1d1d;
    font-size: 0.88rem;
  }
  .alert-banner strong { color: #991b1b; }

  /* ── Stat cards ── */
  .stat-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .stat-val { font-size: 2rem; font-weight: 800; color: #0f172a; }
  .stat-lbl { color: #64748b; font-size: 0.78rem; margin-top: 2px; }

  /* ── Section label ── */
  .section-lbl {
    color: #94a3b8; font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 20px 0 10px; padding-left: 2px;
  }

  /* ── Patient row ── */
  .pt-row {
    background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid transparent;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s, border-color 0.15s;
  }
  .pt-row:hover { box-shadow: 0 4px 14px rgba(30,58,92,0.1); border-color: #93c5fd; }
  .pt-row-refer   { border-left-color: #dc2626 !important; }
  .pt-row-monitor { border-left-color: #b45309 !important; }
  .pt-row-low     { border-left-color: #16a34a !important; }
  .pt-row-copd    { border-left-color: #7c3aed !important; }

  .avatar {
    width: 42px; height: 42px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; color: white;
  }
  .pt-name  { color: #0f172a; font-weight: 600; font-size: 0.95rem; }
  .pt-meta  { color: #64748b; font-size: 0.79rem; margin-top: 2px; }
  .pt-nhs   { color: #94a3b8; font-size: 0.74rem; font-family: monospace; }

  /* ── Risk badges ── */
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 700;
  }
  .badge-low     { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
  .badge-monitor { background: #fef9c3; color: #a16207; border: 1px solid #fde68a; }
  .badge-refer   { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
  .badge-copd    { background: #f3e8ff; color: #7e22ce; border: 1px solid #e9d5ff; }

  .tag-pill {
    display: inline-block; background: #f0f9ff; color: #0369a1;
    border: 1px solid #bae6fd; border-radius: 10px;
    padding: 2px 9px; font-size: 0.73rem;
  }

  /* ── Table header ── */
  .tbl-head {
    padding: 0 18px 6px;
    color: #94a3b8; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Footer ── */
  .ehr-footer {
    text-align: center; color: #94a3b8; font-size: 0.72rem;
    padding: 20px 0 8px; border-top: 1px solid #e2e8f0; margin-top: 28px;
  }

  /* ── Buttons ── */
  .stButton button {
    background: #1e3a5c; color: white !important; border: none; border-radius: 7px;
    font-weight: 600; font-size: 0.83rem; padding: 8px 16px;
  }
  .stButton button:hover { background: #1d4ed8; color: white !important; }
  .stButton button p { color: white !important; }

  h1, h2, h3, h4 { color: #0f172a; }
  p, div { color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar (Doctor only — no patient form link) ──────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px;'>
      <div style='font-size:2.2rem;'>🏥</div>
      <div style='color:#f1f5f9; font-weight:700; font-size:1.1rem; margin-top:4px;'>MedView EHR</div>
      <div style='color:#94a3b8; font-size:0.72rem;'>Primary Care Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='color:#94a3b8; font-size:0.7rem; font-weight:700;
      text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;'>Doctor Navigation</div>""",
      unsafe_allow_html=True)

    st.page_link("app.py",                   label="🏠  Patient Worklist",  use_container_width=True)
    st.page_link("pages/1_Patient_Chart.py", label="📋  Patient Chart",     use_container_width=True)

    st.markdown("---")
    st.markdown("""<div style='color:#94a3b8; font-size:0.7rem; font-weight:700;
      text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;'>Patient Portal</div>""",
      unsafe_allow_html=True)
    st.page_link("pages/2_Patient_Form.py",  label="📝  Patient Questionnaire", use_container_width=True)
    st.page_link("pages/3_Model_Card.py",    label="🃏  Model Card",             use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#94a3b8; line-height:2;'>
      👤 Dr. A. Patel<br>
      🏥 Northside Family Practice<br>
      📅 Monday, 23 March 2026
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; font-size:0.7rem; color:#64748b; line-height:1.8;'>
      Powered by<br>
      <span style='color:#f97316; font-weight:800; font-size:0.95rem;'>GSK</span>
      <span style='color:#94a3b8;'> COPD AI Programme</span><br>
      <span style='color:#64748b;'>Sensitivity 88.9% · GOLD Standard</span>
    </div>
    """, unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="logo">🏥</div>
    <div>
      <h2 style="margin:0;">MedView EHR</h2>
      <span>Northside Family Practice · Dr. A. Patel</span>
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <span class="topbar-pill">📅 Mon 23 Mar 2026</span>
    <span class="topbar-pill">👤 Dr. A. Patel</span>
    <span class="gsk-badge">GSK COPD AI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Alert banner ──────────────────────────────────────────────────────
high_risk = [p for p in FAKE_PATIENTS
             if st.session_state["patient_risks"][p["id"]][1] == "High — Physician Review Needed"
             and not p.get("confirmed_copd")]
if high_risk:
    names = ", ".join(p["name"] for p in high_risk)
    st.markdown(f"""
    <div class="alert-banner">
      <span style="font-size:1.3rem;">⚠️</span>
      <div><strong>{len(high_risk)} patient(s) flagged today:</strong>
      {names} — physician review advised.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────
risks = [st.session_state["patient_risks"][p["id"]][1] for p in FAKE_PATIENTS]
sc1, sc2, sc3, sc4, sc5 = st.columns(5)
for col, (val, lbl, color) in zip(
    [sc1, sc2, sc3, sc4, sc5],
    [
        (str(len(FAKE_PATIENTS)), "Today's Patients", "#1e3a5c"),
        (str(risks.count("Low")), "Low", "#16a34a"),
        (str(risks.count("Moderate — Monitor Closely")), "Moderate — Monitor Closely", "#b45309"),
        (str(risks.count("High — Physician Review Needed")), "High — Physician Review Needed", "#dc2626"),
        (str(risks.count("COPD Previously Diagnosed")), "COPD Previously Diagnosed", "#7c3aed"),
    ]
):
    with col:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-val" style="color:{color}">{val}</div>
          <div class="stat-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Patient list ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-lbl">Today\'s Patient List — Monday, 23 March 2026</div>',
            unsafe_allow_html=True)

fc1, fc2, _ = st.columns([1, 1, 3])
filter_risk = fc1.selectbox("COPD Risk Indicators", ["All", "High — Physician Review Needed", "Moderate — Monitor Closely", "Low", "COPD Previously Diagnosed"])
filter_gp   = fc2.selectbox("GP",   ["All GPs", "Dr. A. Patel", "Dr. S. Thompson"],
                             label_visibility="collapsed")

st.markdown("""
<div class="tbl-head" style="display:grid; grid-template-columns:46px 2fr 2fr 1fr 1fr;">
  <div></div><div>Patient</div><div>Visit reason</div><div>COPD Risk Indicators</div><div>Status</div>
</div>
""", unsafe_allow_html=True)

for p in FAKE_PATIENTS:
    proba, risk_level, risk_color, risk_bg, risk_border = \
        st.session_state["patient_risks"][p["id"]]
    if filter_risk != "All" and risk_level != filter_risk:
        continue
    if filter_gp != "All GPs" and p["gp"] != filter_gp:
        continue

    confirmed = p.get("confirmed_copd", False)
    badge_class = {
        "Low": "badge-low",
        "Moderate — Monitor Closely": "badge-monitor",
        "High — Physician Review Needed": "badge-refer",
        "COPD Previously Diagnosed": "badge-copd",
    }[risk_level]
    risk_icon = {
        "Low": "🟢",
        "Moderate — Monitor Closely": "🟡",
        "High — Physician Review Needed": "🔴",
        "COPD Previously Diagnosed": "✅",
    }[risk_level]
    prob_line = (
        '<div style="color:#7c3aed; font-size:0.72rem; margin-top:3px; font-weight:600;">COPD diagnosed</div>'
        if confirmed else
        f'<div style="color:#94a3b8; font-size:0.72rem; margin-top:3px; text-align:left;">{proba*100:.0f}% probability</div>'
    )

    row_col, btn_col = st.columns([7, 1])
    accent_class = {
        "Low": "pt-row-low",
        "Moderate — Monitor Closely": "pt-row-monitor",
        "High — Physician Review Needed": "pt-row-refer",
        "COPD Previously Diagnosed": "pt-row-copd",
    }[risk_level]
    with row_col:
        st.markdown(f"""
        <div class="pt-row {accent_class}" style="display:grid; grid-template-columns:46px 2fr 2fr 1fr 1fr; align-items:center; gap:14px;">
          <div class="avatar" style="background:{p['photo_color']}">{p['photo_initials']}</div>
          <div>
            <div class="pt-name">{p['name']}</div>
            <div class="pt-meta">{p['age']}y · {p['sex']} · {p['gp']}</div>
            <div class="pt-nhs">{p['nhs_number']} · {p['id']}</div>
          </div>
          <div style="color:#475569; font-size:0.83rem;">{p['reason_for_visit']}</div>
          <div>
            <span class="badge {badge_class}">{risk_icon} {risk_level}</span>
            {prob_line}
          </div>
          <div><span class="tag-pill">{p['tag']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Open Chart", key=f"open_{p['id']}", use_container_width=True):
            st.session_state["selected_patient_id"] = p["id"]
            st.switch_page("pages/1_Patient_Chart.py")

st.markdown("""
<div class="ehr-footer">
  MedView EHR v4.2 &nbsp;·&nbsp; GSK COPD AI Programme (Research Prototype) &nbsp;·&nbsp;
  GOLD Spirometry Standard &nbsp;·&nbsp;
  <em>Clinical decision support only — not a replacement for spirometry</em>
</div>
""", unsafe_allow_html=True)
