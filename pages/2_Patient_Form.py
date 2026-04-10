"""
Patient Pre-Appointment Questionnaire
————————————————————————————————————
This page is designed for PATIENTS only.
It has no EHR navigation, no clinical jargon, and no model scores shown.
In production this would live on a separate patient-portal URL.
"""

import os, sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.fake_patients import PATIENT_BY_ID

st.set_page_config(
    page_title="Before Your Breathing Test — Health Questionnaire",
    page_icon="🫁",
    layout="centered",   # centred — works well on mobile
    initial_sidebar_state="collapsed",
)

# ── CSS: patient portal — warm, friendly, no EHR chrome ──────────────
st.markdown("""
<style>
  /* ── Hide Streamlit chrome ── */
  [data-testid="stSidebar"]      { display: none; }
  [data-testid="collapsedControl"]{ display: none; }
  #MainMenu, footer, header       { display: none; }

  /* ── Page background ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] {
    background: #f8faff;
  }

  /* ── Top branding bar ── */
  .patient-topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 0 12px;
    text-align: center;
    margin-bottom: 0;
  }
  .patient-topbar .org {
    font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 2px;
  }
  .patient-topbar .title {
    font-size: 1rem; color: #1e293b; font-weight: 700;
  }

  /* ── Patient identification banner ── */
  .patient-id-bar {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 20px 0 24px;
  }
  .patient-id-bar .av {
    width: 48px; height: 48px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1rem; color: white; flex-shrink: 0;
  }
  .patient-id-bar .greeting { color: #1e40af; font-size: 1rem; font-weight: 700; }
  .patient-id-bar .subtext  { color: #3b82f6; font-size: 0.82rem; margin-top: 2px; }

  /* ── Hero ── */
  .hero {
    text-align: center;
    padding: 36px 24px 28px;
  }
  .hero .icon { font-size: 3.5rem; margin-bottom: 10px; }
  .hero h1   { color: #1e293b; font-size: 1.9rem; font-weight: 800; margin: 0 0 8px; }
  .hero p    { color: #64748b; font-size: 1rem; max-width: 480px; margin: 0 auto; line-height: 1.6; }

  /* ── Progress bar ── */
  .progress-wrap {
    margin: 0 0 28px;
  }
  .progress-label {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;
  }
  .progress-track {
    background: #e2e8f0; border-radius: 6px; height: 7px;
  }
  .progress-fill {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 6px; height: 7px;
    transition: width 0.3s;
  }

  /* ── Step pill ── */
  .step-pill {
    display: inline-block;
    background: #ede9fe; color: #6d28d9;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 700;
    margin-bottom: 10px;
  }

  /* ── Form sections ── */
  .q-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .q-section h3 {
    color: #1e293b; font-size: 1rem; font-weight: 700; margin: 0 0 16px;
    padding-bottom: 10px; border-bottom: 1px solid #f1f5f9;
  }

  /* ── Submit button ── */
  .stButton button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 700; padding: 12px 28px;
    width: 100%;
  }
  .stButton button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  }

  /* ── Results ── */
  .result-hero {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 14px; padding: 28px 24px; text-align: center; margin-bottom: 20px;
  }
  .result-hero h2 { color: #15803d; font-size: 1.4rem; margin: 8px 0 6px; }
  .result-hero p  { color: #166534; font-size: 0.9rem; }

  .score-tile {
    background: #ffffff; border-radius: 12px; border: 2px solid;
    padding: 18px 12px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .score-tile .sv { font-size: 2.4rem; font-weight: 900; line-height: 1; }
  .score-tile .sl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }

  .summary-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 14px;
  }
  .summary-box h4 { color: #334155; font-size: 0.85rem; font-weight: 700;
                    margin: 0 0 10px; }
  .summary-item { font-size: 0.87rem; color: #475569; padding: 4px 0;
                  border-bottom: 1px solid #f8fafc; }
  .summary-item:last-child { border-bottom: none; }

  .tip-box {
    background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 12px;
    padding: 20px 22px; margin-top: 12px;
  }
  .tip-box h4 { color: #6d28d9; font-size: 0.9rem; font-weight: 700; margin: 0 0 10px; }

  /* General type */
  label, p, div, span { color: #334155; }
  h1, h2, h3 { color: #1e293b; }

  /* Footer */
  .patient-footer {
    text-align: center; color: #94a3b8; font-size: 0.72rem;
    padding: 24px 0 12px; border-top: 1px solid #e2e8f0; margin-top: 32px;
  }

  /* ── Consent gate ── */
  .consent-gate {
    background: #ffffff;
    border: 2px solid #cbd5e1;
    border-radius: 16px;
    padding: 28px 30px;
    margin: 24px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }
  .consent-gate h2 {
    color: #1e293b; font-size: 1.15rem; font-weight: 800;
    margin: 0 0 16px; display: flex; align-items: center; gap: 8px;
  }
  .consent-gate ul {
    margin: 0 0 18px; padding-left: 20px;
  }
  .consent-gate ul li {
    color: #475569; font-size: 0.9rem; margin-bottom: 8px; line-height: 1.5;
  }
  .consent-disclaimer {
    background: #fff7ed;
    border: 1.5px solid #fed7aa;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
  }
  .consent-disclaimer p {
    color: #9a3412; font-size: 0.85rem; margin: 0; line-height: 1.6;
  }
  .consent-disclaimer strong {
    color: #7c2d12;
  }
</style>
""", unsafe_allow_html=True)

# ── Top branding bar ──────────────────────────────────────────────────
st.markdown("""
<div class="patient-topbar">
  <div class="org">Northside Family Practice · Patient Portal</div>
  <div class="title">🫁 Breathing Health Questionnaire</div>
</div>
""", unsafe_allow_html=True)

# ── Pre-fill from session state (when doctor sends form) ──────────────
prefill_id = st.session_state.get("prefill_patient_id") or \
             st.session_state.get("selected_patient_id")

if prefill_id and prefill_id in PATIENT_BY_ID:
    patient     = PATIENT_BY_ID[prefill_id]
    inp         = patient["model_inputs"]
    pf_name     = patient["name"].split()[0]      # first name only
    pf_age      = patient["age"]
    pf_color    = patient["photo_color"]
    pf_initials = patient["photo_initials"]
    pf_smoke    = ("Current smoker" if inp.get("fumador_actual")
                   else "Ex-smoker (quit > 1 year ago)" if inp.get("exfumador")
                   else "Never smoked")
    pf_pack_yrs = int(inp.get("paquetes_ano", 0))

    st.markdown(f"""
    <div class="patient-id-bar">
      <div class="av" style="background:{pf_color}">{pf_initials}</div>
      <div>
        <div class="greeting">Hello, {pf_name} 👋</div>
        <div class="subtext">Your doctor has asked you to complete this short questionnaire
        before your breathing test appointment. It takes about 5 minutes.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    patient     = None
    pf_name     = ""
    pf_age      = 50
    pf_color    = "#6366f1"
    pf_initials = "?"
    pf_smoke    = "Never smoked"
    pf_pack_yrs = 0

    st.markdown("""
    <div class="hero">
      <div class="icon">🫁</div>
      <h1>Breathing Health Questionnaire</h1>
      <p>Your doctor has asked you to complete this short form before your
      breathing test. It should take about 5 minutes.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Consent gate ─────────────────────────────────────────────────────
consent_key = f"consent_given_{prefill_id or 'anon'}"
if consent_key not in st.session_state:
    st.session_state[consent_key] = False

if not st.session_state[consent_key]:
    st.markdown("""
    <div class="consent-gate">
      <h2>🔒 Before you begin — your data &amp; privacy</h2>
      <ul>
        <li><strong>What we collect:</strong> Symptoms, breathing difficulties, smoking history,
        and lifestyle information you enter in this questionnaire.</li>
        <li><strong>Who sees it:</strong> Only your doctor and care team at this practice.
        Your information is not shared with third parties.</li>
        <li><strong>Why we ask:</strong> To help your doctor prepare for your appointment.
        This questionnaire <em>does not</em> produce a diagnosis.</li>
      </ul>
      <div class="consent-disclaimer">
        <p><strong>Important:</strong> This is an informational screening tool only.
        It does not provide a medical diagnosis or replace a clinical consultation.
        All medical decisions are made exclusively by your doctor.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    agreed = st.checkbox(
        "I understand and agree to share my health information with my care team for this appointment.",
        value=False,
        key="consent_checkbox",
    )
    proceed = st.button(
        "Continue to questionnaire →",
        disabled=not agreed,
        use_container_width=True,
        type="primary",
    )
    if proceed and agreed:
        st.session_state[consent_key] = True
        st.rerun()

    st.markdown("""
    <div class="patient-footer">
      Northside Family Practice · Powered by
      <span style="color:#f97316; font-weight:700;">GSK</span> COPD AI Programme ·
      Your data is handled securely and used only by your care team.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Progress tracker (static — updates on submit) ─────────────────────
st.markdown("""
<div class="progress-wrap">
  <div class="progress-label"><span>Step 1 of 4</span><span>25%</span></div>
  <div class="progress-track"><div class="progress-fill" style="width:25%"></div></div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# FORM
# ════════════════════════════════════════════════════════
with st.form("patient_q_form"):

    # ── STEP 1: About you ─────────────────────────────────
    st.markdown('<span class="step-pill">Step 1 of 4 — About You</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="q-section"><h3>Tell us a little about yourself</h3>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        name_val = c1.text_input("Your first name (optional)", value=pf_name if pf_name else "")
        age_val  = c2.number_input("Your age", 18, 110, pf_age)

        c1, c2  = st.columns(2)
        height  = c1.number_input("Your height (cm)", 130, 220, 170)
        weight  = c2.number_input("Your weight (kg)", 30.0, 250.0, 75.0, step=0.5)

        smoke_opts = ["Never smoked", "Current smoker",
                      "Ex-smoker (quit > 1 year ago)", "Ex-smoker (quit < 1 year ago)"]
        smoke = st.selectbox(
            "Smoking status",
            smoke_opts,
            index=smoke_opts.index(pf_smoke) if pf_smoke in smoke_opts else 0,
        )
        pack_yrs = (
            st.slider("How many pack-years have you smoked?", 0, 80, pf_pack_yrs,
                      help="Pack-years = packs per day × years smoked")
            if smoke != "Never smoked" else 0
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="progress-wrap">
      <div class="progress-label"><span>Step 2 of 4</span><span>50%</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:50%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 2: Symptoms ──────────────────────────────────
    st.markdown('<span class="step-pill">Step 2 of 4 — Your Symptoms</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="q-section"><h3>How has your breathing been?</h3>', unsafe_allow_html=True)

        duration = st.selectbox("How long have you had breathing difficulties?", [
            "No breathing difficulties",
            "Less than 3 months",
            "3 to 12 months",
            "1 to 3 years",
            "More than 3 years",
        ])

        st.markdown("**Which of these do you experience?** *(tick everything that applies)*")
        sc1, sc2 = st.columns(2)
        with sc1:
            sym_breath = st.checkbox("Feeling short of breath")
            sym_cough  = st.checkbox("A cough that won't go away")
            sym_phlegm = st.checkbox("Bringing up phlegm or mucus")
            sym_wheeze = st.checkbox("Wheezing or a whistling chest")
        with sc2:
            sym_tight  = st.checkbox("Tightness in the chest")
            sym_tired  = st.checkbox("Feeling more tired than usual")
            sym_weight = st.checkbox("Losing weight without trying")
            sym_infect = st.checkbox("Getting chest infections often")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="progress-wrap">
      <div class="progress-label"><span>Step 3 of 4</span><span>75%</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:75%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 3: Breathlessness scale ──────────────────────
    st.markdown('<span class="step-pill">Step 3 of 4 — Breathlessness Scale</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="q-section"><h3>How breathless do you get day-to-day?</h3>', unsafe_allow_html=True)
        st.caption("Choose the description that fits you best right now.")
        mrc = st.radio("", [1, 2, 3, 4, 5], format_func=lambda x: {
            1: "1 — I only get breathless during hard exercise (running, heavy lifting)",
            2: "2 — I get breathless when hurrying on flat ground, or walking up a hill",
            3: "3 — I walk slower than people my age on flat ground, or need to stop for breath",
            4: "4 — I have to stop for breath after walking about 100 metres, or after a few minutes",
            5: "5 — I am too breathless to leave the house, or I get breathless getting dressed",
        }[x], index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="progress-wrap">
      <div class="progress-label"><span>Step 4 of 4</span><span>100%</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:100%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 4: CAT questions ─────────────────────────────
    st.markdown('<span class="step-pill">Step 4 of 4 — Day-to-Day Impact</span>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="q-section"><h3>How much does your breathing affect your daily life?</h3>', unsafe_allow_html=True)
        st.caption("For each question, slide from 0 (not at all) to 5 (as bad as it could be).")

        cat_items = [
            ("Coughing",           "I never cough",                      "I cough all the time"),
            ("Phlegm",             "I have no phlegm",                   "My chest is full of phlegm"),
            ("Chest tightness",    "My chest doesn't feel tight",        "My chest feels very tight"),
            ("Climbing stairs",    "Not breathless on stairs",           "Very breathless on stairs"),
            ("Home life",          "I do everything I want at home",     "I can't do much at home"),
            ("Leaving home",       "I feel confident leaving the house", "I'm afraid to leave the house"),
            ("Sleep",              "I sleep soundly",                    "I sleep badly because of breathing"),
            ("Energy",             "I feel full of energy",              "I have no energy"),
        ]
        cat_scores = []
        for i, (topic, left_lbl, right_lbl) in enumerate(cat_items):
            st.markdown(f"**{topic}**")
            cl, cs, cr = st.columns([2, 3, 2])
            cl.markdown(f"<small style='color:#94a3b8'>{left_lbl}</small>", unsafe_allow_html=True)
            v = cs.slider("", 0, 5, 0, key=f"cat_{i}", label_visibility="collapsed")
            cr.markdown(f"<small style='color:#94a3b8; text-align:right; display:block;'>{right_lbl}</small>",
                        unsafe_allow_html=True)
            cat_scores.append(v)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Extra info ────────────────────────────────────────
    with st.container():
        st.markdown('<div class="q-section"><h3>A few more quick questions</h3>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            fam_copd  = st.checkbox("Someone in my family has had COPD or emphysema")
            dust_exp  = st.checkbox("My job involves dust, fumes or chemicals")
        with r2:
            fam_asthm = st.checkbox("Someone in my family has asthma")
            had_asthm = st.checkbox("I have been told I have asthma")
        extra = st.text_area(
            "Is there anything else you'd like your doctor to know about your breathing?",
            placeholder="Optional — leave blank if nothing else to add",
            height=80,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Submit my answers →", use_container_width=True)

# ════════════════════════════════════════════════════════
# RESULTS (patient-facing — no scores, just reassurance)
# ════════════════════════════════════════════════════════
if submitted:
    cat_total = sum(cat_scores)
    n_syms    = sum([sym_breath, sym_cough, sym_phlegm, sym_wheeze,
                     sym_tight, sym_tired, sym_weight, sym_infect])
    bmi_val   = round(weight / ((height / 100) ** 2), 1)

    # Save results to session state so doctor chart can update risk score
    if prefill_id:
        st.session_state[f"questionnaire_{prefill_id}"] = {
            "cat_total":  cat_total,
            "mrc_grade":  mrc,
            "n_symptoms": n_syms,
            "bmi":        bmi_val,
        }

    # Progress to 100%
    st.markdown("""
    <div class="progress-wrap">
      <div class="progress-label"><span>Done!</span><span>✓ Submitted</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:100%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # Confirmation — friendly, no clinical jargon, no model scores
    st.markdown(f"""
    <div class="result-hero">
      <div style="font-size:2.5rem;">✅</div>
      <h2>Thank you{', ' + name_val if name_val.strip() else ''}!</h2>
      <p>Your answers have been securely sent to your doctor at Northside Family Practice.<br>
      They will review them before your appointment.</p>
    </div>
    """, unsafe_allow_html=True)

    # Simple summary for the patient
    st.markdown("### What you told us")

    with st.container():
        st.markdown('<div class="summary-box"><h4>Your answers at a glance</h4>', unsafe_allow_html=True)
        summary_rows = []
        if name_val.strip():
            summary_rows.append(f"👤 Name: <strong>{name_val}</strong>, age {age_val}")
        summary_rows.append(f"📏 BMI: <strong>{bmi_val} kg/m²</strong>")
        summary_rows.append(f"🚬 Smoking: <strong>{smoke}</strong>" + (f" · {pack_yrs} pack-years" if pack_yrs else ""))
        summary_rows.append(f"🕐 Breathing difficulties for: <strong>{duration}</strong>")
        sym_list = [s for s, v in [
            ("shortness of breath", sym_breath), ("persistent cough", sym_cough),
            ("phlegm", sym_phlegm), ("wheezing", sym_wheeze), ("chest tightness", sym_tight),
            ("unusual tiredness", sym_tired), ("unexplained weight loss", sym_weight),
            ("frequent chest infections", sym_infect)] if v]
        if sym_list:
            summary_rows.append(f"🩺 Symptoms: <strong>{', '.join(sym_list)}</strong>")
        extra_flags = [f for f, v in [
            ("family history of COPD", fam_copd), ("family history of asthma", fam_asthm),
            ("occupational dust/fume exposure", dust_exp), ("previous asthma diagnosis", had_asthm)] if v]
        if extra_flags:
            summary_rows.append(f"⚠️ Additional factors: <strong>{', '.join(extra_flags)}</strong>")
        if extra.strip():
            summary_rows.append(f"💬 Your notes: <em>{extra.strip()}</em>")

        for row in summary_rows:
            st.markdown(f'<div class="summary-item">{row}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tips while waiting
    with st.container():
        st.markdown("""
        <div class="tip-box">
          <h4>💜 While you wait for your appointment</h4>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("See helpful tips before your breathing test"):
        st.markdown("""
        **🚭 If you smoke — consider stopping now**
        It's the single most effective thing you can do to slow lung decline.
        Ask your practice nurse about free NHS support.

        **🚶 Keep moving, even gently**
        A short walk each day helps maintain lung capacity and energy.

        **💉 Book your vaccinations**
        Annual flu jab and one-off pneumonia vaccine are recommended —
        they reduce the risk of serious chest infections.

        **📓 Keep a breathing diary**
        Note days when you feel worse (cold weather, exercise, infections) —
        this helps your doctor understand your triggers.

        **💊 Use your inhalers as prescribed**
        Don't stop using them without speaking to your doctor first,
        even if you feel better.

        **❓ Have questions?**
        Write them down and bring them to your appointment —
        no question is too small.
        """)

    st.markdown("""
    <div class="patient-footer">
      Northside Family Practice · Powered by
      <span style="color:#f97316; font-weight:700;">GSK</span> COPD AI Programme ·
      Your data is handled securely and used only by your care team.
    </div>
    """, unsafe_allow_html=True)
