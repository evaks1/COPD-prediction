"""
Model Card — COPD Screening AI
Follows the Model Card standard (Mitchell et al., 2019).
"""

import os, sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocess import FEATURE_LABELS

st.set_page_config(
    page_title="Model Card — COPD Screener AI",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base     = os.path.dirname(os.path.dirname(__file__))
    artifact = joblib.load(os.path.join(base, "models", "copd_model_v2.pkl"))
    return artifact["model"], artifact["threshold"], artifact["feature_names"]

model, threshold, feature_names = load_model()

# ── Pre-computed test-set metrics (Iteration 2 — copd_model_v2.pkl) ──
# Test set: 1,990 patients (stratified 20% of 9,948) — 76.4% COPD+
# Confusion matrix at optimised threshold (0.667):
TP, FP, TN, FN = 1351, 388, 82, 169
METRICS = {
    "Sensitivity (Recall)": TP / (TP + FN),      # 88.9 %
    "Specificity":           TN / (TN + FP),      # 17.5 %
    "Precision (PPV)":       TP / (TP + FP),      # 77.7 %
    "NPV":                   TN / (TN + FN),      # 32.7 %
    "ROC-AUC":               0.586,
    "F1 Score":              2*TP / (2*TP + FP + FN),  # 0.829
    "Accuracy":              (TP + TN) / (TP + TN + FP + FN),
}
TEST_N      = 1990
TRAIN_N     = 7958   # 9,948 synthetic patients, 80% train split
COPD_PREV   = 76.4   # % in derived label (Strategy B, GOLD 2024)
THRESHOLD   = threshold

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] { background: #f1f5f9; }
  [data-testid="stSidebar"] { background: #1e3a5c; border-right: none; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] hr { border-color: #2d5282 !important; }
  [data-testid="stSidebarNavItems"] a { color: #93c5fd !important; }

  .mc-hero {
    background: linear-gradient(135deg, #1e3a5c 0%, #1d4ed8 100%);
    border-radius: 14px; padding: 32px 36px; margin-bottom: 20px; color: white;
  }
  .mc-hero h1 { color: white; font-size: 1.8rem; margin: 0 0 6px; font-weight: 800; }
  .mc-hero p  { color: #bfdbfe; font-size: 0.95rem; margin: 0; }

  .mc-badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 700; margin: 4px 4px 0 0;
  }
  .badge-blue   { background: #dbeafe; color: #1d4ed8; }
  .badge-green  { background: #dcfce7; color: #15803d; }
  .badge-orange { background: #fff7ed; color: #c2410c; }
  .badge-red    { background: #fee2e2; color: #b91c1c; }
  .badge-purple { background: #f3e8ff; color: #7e22ce; }

  .card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .card-title {
    color: #1e3a5c; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 700; margin: 0 0 14px;
    padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
    display: flex; align-items: center; gap: 8px;
  }

  .metric-tile {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px 14px; text-align: center;
  }
  .metric-val { font-size: 1.9rem; font-weight: 800; line-height: 1; }
  .metric-lbl { color: #64748b; font-size: 0.75rem; margin-top: 5px; font-weight: 500; }

  .highlight-metric {
    background: #eff6ff; border: 2px solid #bfdbfe; border-radius: 10px;
    padding: 16px 14px; text-align: center;
  }
  .highlight-metric .metric-val { color: #1d4ed8; }

  .param-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f8fafc; font-size: 0.87rem;
  }
  .param-row:last-child { border-bottom: none; }
  .param-key { color: #64748b; font-weight: 500; }
  .param-val { color: #0f172a; font-weight: 600; font-family: monospace; font-size: 0.82rem; }

  .limitation-item {
    display: flex; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid #f8fafc; font-size: 0.87rem;
  }
  .limitation-item:last-child { border-bottom: none; }
  .lim-icon { font-size: 1.1rem; flex-shrink: 0; padding-top: 1px; }
  .lim-title { color: #0f172a; font-weight: 600; font-size: 0.87rem; }
  .lim-desc  { color: #64748b; font-size: 0.82rem; margin-top: 2px; line-height: 1.5; }

  .use-case-row {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 0; border-bottom: 1px solid #f8fafc; font-size: 0.87rem;
  }
  .use-case-row:last-child { border-bottom: none; }

  h1,h2,h3,h4 { color: #0f172a; }
  .stButton button {
    background: #1e3a5c; color: white; border: none; border-radius: 7px;
    font-weight: 600; font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px;'>
      <div style='font-size:2rem;'>🏥</div>
      <div style='color:#f1f5f9; font-weight:700; font-size:1rem; margin-top:4px;'>MedView EHR</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.page_link("app.py",                   label="← Patient Worklist",    use_container_width=True)
    st.page_link("pages/1_Patient_Chart.py", label="📋 Patient Chart",      use_container_width=True)
    st.page_link("pages/2_Patient_Form.py",  label="📝 Patient Form",       use_container_width=True)
    st.page_link("pages/3_Model_Card.py",    label="🃏 Model Card",          use_container_width=True)
    st.markdown("---")
    st.markdown('<div style="text-align:center; font-size:0.7rem; color:#64748b; line-height:1.8;">Powered by<br><span style="color:#f97316; font-weight:800; font-size:0.9rem;">GSK</span> <span style="color:#94a3b8;">COPD AI Programme</span></div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="mc-hero">
  <div style="display:flex; align-items:center; gap:16px; margin-bottom:14px;">
    <div style="font-size:2.8rem;">🃏</div>
    <div>
      <h1>COPD Screening AI — Model Card</h1>
      <p>Transparency documentation for the GSK COPD pre-spirometry risk model</p>
    </div>
  </div>
  <div>
    <span class="mc-badge badge-blue">v2.1 · March 2026</span>
    <span class="mc-badge badge-green">Logistic Regression + Kaggle Augmentation</span>
    <span class="mc-badge badge-orange">GOLD Spirometry Label</span>
    <span class="mc-badge badge-purple">Research Prototype</span>
    <span class="mc-badge badge-red">Not for clinical deployment</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROW 1: Key metrics
# ══════════════════════════════════════════════════════════════
st.markdown("### Key Performance Metrics")
st.caption("Evaluated on a held-out test set of 2,000 patients (20% stratified split)")

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
metric_display = [
    ("Sensitivity", METRICS["Sensitivity (Recall)"], "#1d4ed8", True),
    ("Specificity",  METRICS["Specificity"],          "#64748b", False),
    ("Precision (PPV)", METRICS["Precision (PPV)"],   "#64748b", False),
    ("NPV",          METRICS["NPV"],                  "#64748b", False),
    ("ROC-AUC",      METRICS["ROC-AUC"],               "#64748b", False),
    ("F1 Score",     METRICS["F1 Score"],              "#64748b", False),
    ("Accuracy",     METRICS["Accuracy"],              "#64748b", False),
]
for col, (lbl, val, color, highlight) in zip(
    [m1, m2, m3, m4, m5, m6, m7], metric_display
):
    with col:
        css_class = "highlight-metric" if highlight else "metric-tile"
        st.markdown(f"""
        <div class="{css_class}">
          <div class="metric-val" style="color:{color}">{val*100:.1f}%</div>
          <div class="metric-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.info("⭐ **Sensitivity (recall) is the primary optimisation target.** For a screening tool, missing a true COPD case (false negative) is more harmful than a false alarm (false positive) which leads to an unnecessary but harmless spirometry test.", icon=None)

# ══════════════════════════════════════════════════════════════
# ROW 2: Confusion matrix + ROC area + threshold
# ══════════════════════════════════════════════════════════════
col_cm, col_gauge, col_thresh = st.columns([1.4, 1, 1.2])

with col_cm:
    st.markdown('<div class="card"><div class="card-title">🔢 Confusion Matrix (test set, n=2,000)</div>', unsafe_allow_html=True)
    cm_fig = go.Figure(go.Heatmap(
        z=[[TN, FP], [FN, TP]],
        x=["Predicted: No COPD", "Predicted: COPD"],
        y=["Actual: No COPD", "Actual: COPD"],
        text=[[f"TN\n{TN}", f"FP\n{FP}"], [f"FN\n{FN}", f"TP\n{TP}"]],
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[[0,"#f1f5f9"],[0.3,"#93c5fd"],[1,"#1d4ed8"]],
        showscale=False,
    ))
    cm_fig.update_layout(
        height=220, margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font={"size": 11, "color": "#475569"},
        xaxis={"side": "bottom"},
    )
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.6; padding:0 4px 8px;">
      <b>TN</b> = correctly cleared &nbsp;·&nbsp; <b>TP</b> = correctly flagged<br>
      <b>FP</b> = unnecessary spirometry &nbsp;·&nbsp; <b>FN</b> = missed cases
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_gauge:
    st.markdown('<div class="card"><div class="card-title">🎯 Sensitivity Gauge</div>', unsafe_allow_html=True)
    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=METRICS["Sensitivity (Recall)"] * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#1d4ed8"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#94a3b8", "size": 9}},
            "bar": {"color": "#1d4ed8", "thickness": 0.25},
            "bgcolor": "white", "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 70],  "color": "#fee2e2"},
                {"range": [70, 85], "color": "#fef9c3"},
                {"range": [85, 100],"color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "#16a34a", "width": 3},
                          "thickness": 0.8, "value": 90},
        },
    ))
    g.update_layout(height=200, margin=dict(t=20, b=0, l=10, r=10),
                    paper_bgcolor="white", font={"color": "#64748b"})
    st.plotly_chart(g, use_container_width=True)
    st.markdown('<div style="text-align:center; font-size:0.75rem; color:#64748b; padding-bottom:8px;">Target ≥90% (green line)</div></div>', unsafe_allow_html=True)

with col_thresh:
    st.markdown('<div class="card"><div class="card-title">⚙️ Decision Threshold</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; padding:12px 0 8px;">
      <div style="font-size:3rem; font-weight:900; color:#1d4ed8; line-height:1;">{THRESHOLD:.3f}</div>
      <div style="color:#64748b; font-size:0.8rem; margin-top:4px;">probability cut-off</div>
    </div>
    <div style="font-size:0.82rem; color:#475569; line-height:1.7; border-top:1px solid #f1f5f9; padding-top:10px;">
      Default classification threshold is 0.50.<br>
      This model uses <strong>{THRESHOLD:.3f}</strong>, deliberately
      lowered to maximise sensitivity — trading precision for recall in a
      screening context where false negatives (missed COPD) are more harmful
      than false positives (unnecessary spirometry).
    </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROW 3: Feature importance + model params
# ══════════════════════════════════════════════════════════════
fi_col, param_col = st.columns([2, 1], gap="medium")

with fi_col:
    st.markdown('<div class="card"><div class="card-title">📊 Feature Importance (|LR coefficient|)</div>', unsafe_allow_html=True)
    # copd_model_v2 is CalibratedClassifierCV(ImbPipeline(LR)); average coefs across folds
    try:
        coefs = np.mean([
            c.estimator.named_steps["classifier"].coef_[0]
            for c in model.calibrated_classifiers_
        ], axis=0)
        fi_vals = abs(coefs)
    except Exception:
        fi_vals = np.ones(len(feature_names))
    fi_series = pd.Series(fi_vals, index=feature_names)
    fi_series.index = [FEATURE_LABELS.get(f, f) for f in fi_series.index]
    fi_df = fi_series.sort_values().tail(15)

    colors = []
    for feat in fi_df.index:
        if any(k in feat for k in ["Pack-Years", "Smoker", "Age"]):
            colors.append("#1d4ed8")   # blue = primary clinical
        elif any(k in feat for k in ["NLP", "Clinical Note"]):
            colors.append("#7c3aed")   # purple = NLP
        elif any(k in feat for k in ["LAMA", "LABA", "SABA", "Medication", "Med"]):
            colors.append("#0891b2")   # teal = medications
        elif any(k in feat for k in ["Vitamin", "CRP", "Ferritin", "Cholesterol", "TSH"]):
            colors.append("#059669")   # green = labs
        else:
            colors.append("#64748b")   # grey = other

    fi_fig = go.Figure(go.Bar(
        y=fi_df.index,
        x=fi_df.values,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fi_fig.update_layout(
        height=420, margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font={"color": "#475569", "size": 11},
        xaxis={"title": "Importance (|coefficient|)", "gridcolor": "#f1f5f9", "title_font": {"size": 10}},
        yaxis={"title": ""},
    )
    st.plotly_chart(fi_fig, use_container_width=True)

    # Legend
    legend_items = [
        ("#1d4ed8", "Smoking / Age"), ("#7c3aed", "NLP (clinical notes)"),
        ("#0891b2", "Medications"), ("#059669", "Lab results"), ("#64748b", "Other"),
    ]
    legend_html = "".join(
        f'<span style="display:inline-flex; align-items:center; gap:5px; margin-right:14px; font-size:0.75rem; color:#64748b;">'
        f'<span style="width:10px; height:10px; border-radius:2px; background:{c}; display:inline-block;"></span>{l}</span>'
        for c, l in legend_items
    )
    st.markdown(f"<div style='padding:0 4px 8px;'>{legend_html}</div></div>", unsafe_allow_html=True)

with param_col:
    st.markdown('<div class="card"><div class="card-title">🔧 Model Parameters</div>', unsafe_allow_html=True)
    params = [
        ("Algorithm",       "Logistic Regression (L2, lbfgs)"),
        ("C (regularisation)", "0.1"),
        ("max_iter",        "1000"),
        ("Training data",   "7,958 synthetic patients (80% split)"),
        ("Augmentation",    "None (Iteration 2)"),
        ("Imbalance handling", "SMOTE (k=5)"),
        ("Threshold",       f"{THRESHOLD:.3f}"),
        ("Optimised for",   "Recall ≥ 90%"),
    ]
    rows = "".join(
        f'<div class="param-row"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>'
        for k, v in params
    )
    st.markdown(f"{rows}</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📦 Features Used</div>', unsafe_allow_html=True)
    feat_groups = [
        ("Demographics", "Age, Sex, BMI"),
        ("Smoking history", "Current/ex-smoker, pack-years, age×pack-years, high-risk composite"),
        ("Physical activity", "Low / medium / high"),
        ("Lab results", "CRP, Vitamin D, cholesterol, ferritin, TSH + binary flags"),
        ("Exacerbations", "Last year count, moderate, severe, severity score"),
        ("Medications", "7 respiratory drug classes + count"),
        ("Comorbidities", "6 ICD-10 flags + burden index"),
        ("NLP", "4 features from Spanish clinical notes"),
    ]
    rows2 = "".join(
        f'<div class="param-row"><span class="param-key">{g}</span></div>'
        f'<div style="color:#94a3b8; font-size:0.76rem; padding:0 0 8px 0; line-height:1.5;">{d}</div>'
        for g, d in feat_groups
    )
    st.markdown(f"{rows2}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROW 4: Training data + label derivation
# ══════════════════════════════════════════════════════════════
data_col, label_col = st.columns(2, gap="medium")

with data_col:
    st.markdown('<div class="card"><div class="card-title">📂 Training Dataset</div>', unsafe_allow_html=True)

    # 2-slice donut: training COPD- and COPD+ (Iteration 2 — no external augmentation)
    donut = go.Figure(go.Pie(
        labels=["Train — No COPD", "Train — COPD+"],
        values=[1878, 6080],
        hole=0.58,
        marker_colors=["#dbeafe", "#1d4ed8"],
        textinfo="percent",
        textfont={"size": 10},
        hovertemplate="%{label}<br>%{value} patients (%{percent})<extra></extra>",
    ))
    donut.update_layout(
        height=200, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15, font={"size": 9, "color": "#64748b"}),
        font={"color": "#475569"},
        annotations=[{"text": "7,958<br>train", "x": 0.5, "y": 0.5,
                      "font": {"size": 13, "color": "#0f172a"}, "showarrow": False}],
    )
    st.plotly_chart(donut, use_container_width=True)

    data_facts = [
        ("Patients with spirometry (total)", "9,948"),
        ("  → Train split", "7,958  (80%)"),
        ("  → Test split", "1,990  (20%, held out)"),
        ("COPD+ in training", "~6,080  (76.4%)"),
        ("COPD− in training", "~1,878  (23.6%)"),
        ("After SMOTE (per fold)", "6,080 vs 6,080  (balanced)"),
        ("Test set", "~1,520 COPD+  / ~470 COPD−"),
        ("External augmentation", "None (Iteration 2)"),
        ("Language", "Spanish (clinical notes)"),
        ("Features engineered", "30 (FEATURES_MAIN)"),
    ]
    rows3 = "".join(
        f'<div class="param-row"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>'
        for k, v in data_facts
    )
    st.markdown(rows3, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-top:1px solid #f1f5f9; margin:14px 0 10px;"></div>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                padding:10px 12px; margin-top:10px; font-size:0.81rem; color:#166534;">
      <strong>Iteration 2 — no external augmentation.</strong> All 9,948 patients come from the
      synthetic EHR dataset. Patients without clean spirometry records were excluded (52 removed).
      The high COPD+ prevalence (76.4%) reflects Strategy B labelling: any patient with at least
      one obstructive spirometry test (min FEV₁/FVC &lt; 0.70) is labelled positive.
      SMOTE rebalances to 1:1 <em>inside each CV fold</em>; Platt scaling recalibrates
      predicted probabilities back to the 76.4% training distribution.
    </div>
    </div>
    """, unsafe_allow_html=True)

with label_col:
    st.markdown('<div class="card"><div class="card-title">🏷️ Label Derivation — GOLD Standard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.87rem; line-height:1.8; color:#334155;">

    <p style="margin:0 0 12px;">
      The <code>target.csv</code> clinical labels were <strong>not used</strong>.
      Instead, COPD status was derived from spirometry measurements using the
      <strong>GOLD (Global Initiative for Chronic Obstructive Lung Disease)</strong> standard:
    </p>

    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
                padding:14px 16px; margin-bottom:14px;">
      <div style="font-size:1rem; font-weight:700; color:#1d4ed8; text-align:center; margin-bottom:8px;">
        FEV₁ / FVC &lt; 0.70
      </div>
      <div style="font-size:0.8rem; color:#475569; text-align:center;">
        Forced Expiratory Volume₁ / Forced Vital Capacity
      </div>
    </div>

    <p style="margin:0 0 8px;"><strong>Label rule (GOLD 2024 — Iteration 2):</strong></p>
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
                padding:12px 14px; margin-bottom:14px; font-size:0.83rem; color:#166534;">
      COPD = <strong>minimum</strong> FEV₁/FVC ratio across all clean spirometry tests &lt; 0.70.
      A patient with at least one confirmed obstructive measurement is labelled positive,
      consistent with the GOLD 2024 definition of persistent airflow limitation.
    </div>

    <p style="margin:0 0 8px;"><strong>Training prevalence & class balancing:</strong></p>
    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
                padding:12px 14px; margin-bottom:14px; font-size:0.83rem; color:#1e40af;">
      Applying the minimum-ratio rule to a cohort where all patients have at least one
      spirometry record yields a high COPD+ prevalence (<strong>76.4%</strong> of training patients).
      This reflects the selection bias of the cohort — patients who undergo spirometry are
      already suspected of having airway disease — not the prevalence in a general unscreened
      population.<br><br>
      During training, <strong>SMOTE</strong> resamples each fold to a 50/50 class ratio so the
      model learns discriminative signal rather than predicting the majority class.
      <strong>Platt scaling</strong> (sigmoid calibration, 5-fold CV) then recalibrates the output
      probabilities back to the observed training distribution. The reported threshold (0.667)
      and predicted probabilities reflect the 76.4% training prevalence and should be
      interpreted accordingly when applied to populations with a different base rate.
    </div>

    <p style="margin:0 0 6px;"><strong>Why not use target.csv?</strong></p>
    <ul style="color:#64748b; font-size:0.82rem; line-height:1.8; padding-left:18px; margin:0;">
      <li>Only 29% agreement between target.csv and spirometry-derived labels</li>
      <li>target.csv prevalence (6.4%) inconsistent with the spirometry measurements</li>
      <li>Spirometry is the recognised gold standard for COPD diagnosis</li>
    </ul>

    </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROW 5: Limitations + bias + intended use
# ══════════════════════════════════════════════════════════════
lim_col, use_col = st.columns(2, gap="medium")

with lim_col:
    st.markdown('<div class="card"><div class="card-title">⚠️ Limitations & Known Issues</div>', unsafe_allow_html=True)
    limitations = [
        ("🧪", "Synthetic dataset",
         "The model was trained entirely on synthetically generated EHR data. Clinical features are not strongly correlated with spirometry outcomes in the synthetic generation process, resulting in a lower ROC-AUC (0.586) than expected for real-world data. Performance on real patient populations is unknown."),
        ("📉", "Moderate ROC-AUC",
         "AUC of 0.586 reflects the synthetic data limitation — not the clinical validity of the features used. Real-world COPD screening studies typically achieve AUC 0.75–0.90. No spirometry features are used (by design), as they would constitute label leakage."),
        ("⚖️", "Low specificity",
         f"At the chosen threshold ({THRESHOLD:.3f}), specificity is ~17.5%, meaning many patients without COPD are flagged. This is intentional for a screening tool but increases spirometry workload."),
        ("🌍", "Population generalisability",
         "Trained on a single synthetic cohort. Performance may differ across ethnicities, regions, or healthcare systems not represented in training data."),
        ("📅", "Static snapshot",
         "The model uses a single point-in-time feature snapshot. It does not model disease progression or longitudinal trends beyond the engineered features."),
        ("🔬", "No post-bronchodilator data",
         "True GOLD standard requires post-bronchodilator FEV₁/FVC. This dataset does not distinguish pre/post bronchodilator measurements."),
        ("📊", "Training prevalence mismatch",
         "The training cohort has a 76.4% COPD+ prevalence because all patients in the dataset underwent spirometry — a pre-selected, high-suspicion group. SMOTE rebalances training folds to 50/50; Platt scaling recalibrates output probabilities to the observed training distribution. Both the optimised threshold and predicted probabilities reflect this prevalence. In a real unscreened primary care population (estimated COPD prevalence 8–15%), predicted probabilities would need recalibration against local base-rate data before clinical use."),
    ]
    for icon, title, desc in limitations:
        st.markdown(f"""
        <div class="limitation-item">
          <div class="lim-icon">{icon}</div>
          <div>
            <div class="lim-title">{title}</div>
            <div class="lim-desc">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with use_col:
    st.markdown('<div class="card"><div class="card-title">✅ Intended Use & Bias Mitigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#334155;">

    <p style="font-weight:700; color:#0f172a; margin:0 0 10px;">Intended use</p>
    <div style="line-height:1.8; margin-bottom:14px;">
      This model is intended as a <strong>clinical decision support tool</strong> for
      primary care physicians to identify patients who may benefit from spirometry referral.
      It is <strong>not</strong> intended to:
      <ul style="padding-left:18px; color:#64748b; margin:6px 0;">
        <li>Replace spirometry or clinical assessment</li>
        <li>Provide a definitive COPD diagnosis</li>
        <li>Be used without clinician oversight</li>
        <li>Be deployed in a clinical setting without prospective validation</li>
      </ul>
    </div>

    <p style="font-weight:700; color:#0f172a; margin:0 0 10px;">Target users</p>
    <div style="line-height:1.8; margin-bottom:14px; color:#64748b;">
      General practitioners / family physicians in primary care settings
      with access to basic EHR data.
    </div>

    <p style="font-weight:700; color:#0f172a; margin:0 0 10px;">Bias mitigation</p>
    """, unsafe_allow_html=True)

    bias_items = [
        ("🚫", "Socioeconomic level removed",
         "nivel_socioeconomico was excluded as a direct model feature. It is a healthcare access proxy, not a clinical COPD risk factor, and including it could lead to differential treatment by economic status."),
        ("🚫", "Residence zone removed",
         "zona_residencia was excluded for the same reason — it correlates with socioeconomic status rather than reflecting a direct clinical pathway to COPD."),
        ("✅", "Clinical features prioritised",
         "The final feature set focuses on clinically validated COPD risk factors: smoking history, age, exacerbations, respiratory medications, and inflammatory markers."),
        ("✅", "SMOTE applied",
         "Synthetic minority oversampling was used to prevent the model from simply predicting the majority class, ensuring the model learns COPD risk patterns."),
        ("✅", "SHAP explanations",
         "Every prediction includes a SHAP explanation visible to the clinician, enabling transparency and detection of unexpected feature influences."),
    ]
    for icon, title, desc in bias_items:
        st.markdown(f"""
        <div class="use-case-row">
          <div style="font-size:1rem; flex-shrink:0;">{icon}</div>
          <div>
            <div style="font-weight:600; color:#0f172a; font-size:0.85rem;">{title}</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:2px; line-height:1.5;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
            padding:18px 24px; margin-top:8px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div style="font-size:0.82rem; color:#475569;">
      <strong style="color:#0f172a;">Model Card format:</strong>
      Mitchell et al., "Model Cards for Model Reporting" (FAccT 2019)
    </div>
    <div style="font-size:0.82rem; color:#475569;">
      <strong style="color:#0f172a;">GOLD reference:</strong>
      Global Strategy for the Diagnosis, Management, and Prevention of COPD 2024 Report
    </div>
    <div style="font-size:0.82rem; color:#475569;">
      <span style="color:#f97316; font-weight:800;">GSK</span> COPD AI Programme ·
      Research Prototype · v2.1 · March 2026
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
