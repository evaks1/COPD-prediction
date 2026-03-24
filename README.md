# COPD Screener — GSK Demo

An AI-powered COPD screening tool for family doctors, built for the GSK demo.
It flags patients at risk of COPD before spirometry and includes a separate patient questionnaire.

---

## What you need before starting

- A Mac or Windows computer
- **Python 3.10 or newer** — check by opening Terminal and typing `python3 --version`
  - If you don't have it: download from [python.org](https://www.python.org/downloads/)
- The project folder on your computer (the one containing `app.py`)

---

## First-time setup (do this once)

Open **Terminal** (Mac) or **Command Prompt** (Windows), then run these commands one at a time.

**1. Go to the project folder**
```
cd /Users/elsaleksandra/GSKIE
```
> ⚠️ Change the path above to wherever you saved the project folder.

**2. Install the required Python packages**
```
pip3 install -r requirements.txt
```
This takes 1–2 minutes and only needs to be done once.

**3. Train the model** (also only once — or after any code changes)
```
python3 train_model.py
python3 train_severity_model.py
```
You should see output ending in `Done.` for each. This creates the model files in the `models/` folder.

---

## Running the app

Every time you want to open the demo, run:

```
python3 -m streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.

If it doesn't open automatically, copy that address into your browser manually.

To stop the app, press `Ctrl + C` in the Terminal.

---

## What's in the app

| Page | Who it's for | What it does |
|------|-------------|--------------|
| **Worklist** (main page) | Doctor | List of today's patients with AI risk badges |
| **Patient Chart** | Doctor | Full EHR view with AI risk score, SHAP explanations, GOLD severity estimate |
| **Patient Form** | Patient | Mobile-friendly questionnaire (CAT score + symptoms) — no clinical jargon |
| **Model Card** | Anyone | Full transparency on how the model works, its metrics, and its limitations |

---

## Navigating the demo

1. Open the app — you'll see the **doctor worklist** with 6 demo patients
2. Click **Open Chart** on any patient to see their full EHR + AI risk widget
3. For flagged patients, click **📩 Send Patient Form** to open the patient questionnaire
4. After the patient submits the form, the doctor's risk score updates automatically
5. Click **Model Card** in the sidebar to see full model statistics

---

## Troubleshooting

**"Port 8501 is already in use"**
Run this first, then try again:
```
pkill -f streamlit
```

**"Module not found" error**
Re-run the install step:
```
pip3 install -r requirements.txt
```

**"No such file or directory: models/xgb_model.pkl"**
You need to train the model first:
```
python3 train_model.py
python3 train_severity_model.py
```

**The browser opened but shows an error**
Try refreshing the page. If it persists, stop the app (`Ctrl+C`) and restart it.

---

## Project structure (for reference)

```
GSKIE/
├── app.py                        ← Doctor worklist (main page)
├── pages/
│   ├── 1_Patient_Chart.py        ← Doctor EHR chart view
│   ├── 2_Patient_Form.py         ← Patient questionnaire
│   └── 3_Model_Card.py           ← Model transparency page
├── utils/
│   ├── fake_patients.py          ← Demo patient records
│   ├── preprocess.py             ← Feature engineering pipeline
│   └── risk_update.py            ← Post-questionnaire risk update
├── models/                       ← Saved model files (created after training)
├── train_model.py                ← Main model training script
├── train_severity_model.py       ← GOLD severity model training script
├── requirements.txt              ← Python package list
└── dataset.csv                   ← Kaggle COPD dataset (101 real patients)
```

---

## About the model

The AI model uses **Logistic Regression** trained on:
- 8,000 synthetic EHR patients (Spanish primary care simulation)
- 101 real COPD patients from a public Kaggle dataset

It is optimised for **high sensitivity (≥90%)** — meaning it is deliberately set to catch as many COPD cases as possible, even if that means some false alarms. False alarms lead to an unnecessary (but harmless) spirometry test; missed cases are more harmful.

**This is a research prototype. It is not approved for clinical use.**
