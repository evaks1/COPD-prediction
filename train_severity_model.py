"""
Train GOLD severity classifier from Kaggle COPD dataset (101 patients).
Classes: 1=MILD, 2=MODERATE, 3=SEVERE, 4=VERY SEVERE

Outputs: models/severity_model.pkl, models/severity_preprocessor.pkl
Run: python train_severity_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)

KAGGLE_PATH = "/Users/elsaleksandra/GSKIE/dataset.csv"

SEVERITY_MAP = {
    "MILD":       1,
    "MODERATE":   2,
    "SEVERE":     3,
    "VERY SEVERE": 4,
}
SEVERITY_LABEL = {1: "MILD", 2: "MODERATE", 3: "SEVERE", 4: "VERY SEVERE"}

# ─────────────────────────────────────────────
# 1. Load Kaggle data
# ─────────────────────────────────────────────
df = pd.read_csv(KAGGLE_PATH)
print(f"Loaded {len(df)} patients from Kaggle dataset")
print(df["COPDSEVERITY"].value_counts())

# Encode severity label
df["severity_num"] = df["COPDSEVERITY"].str.upper().str.strip().map(SEVERITY_MAP)
df = df.dropna(subset=["severity_num"])
df["severity_num"] = df["severity_num"].astype(int)

# ─────────────────────────────────────────────
# 2. Select features available at inference time
# ─────────────────────────────────────────────
# Features we can derive from the GSKIE patient records
SEVERITY_FEATURES = [
    "AGE",
    "PackHistory",
    "FEV1",
    "FEV1PRED",
    "FVC",
    "CAT",
    "gender",
    "smoking",
    "Diabetes",
    "hypertension",
    "IHD",
    "MWT1Best",
]

available = [c for c in SEVERITY_FEATURES if c in df.columns]
X = df[available].copy()
y = df["severity_num"].copy()

print(f"\nFeatures used: {available}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# ─────────────────────────────────────────────
# 3. Preprocessing + model
# ─────────────────────────────────────────────
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_prep = preprocessor.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced",
)

# Cross-validation (small dataset — use CV instead of hold-out)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_prep, y, cv=cv, scoring="accuracy")
print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train on full dataset
model.fit(X_prep, y)
y_pred = model.predict(X_prep)
print("\n── Full-dataset classification report ──")
print(classification_report(y, y_pred, target_names=["MILD", "MODERATE", "SEVERE", "VERY SEVERE"]))

# ─────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────
joblib.dump(model,        "models/severity_model.pkl")
joblib.dump(preprocessor, "models/severity_preprocessor.pkl")
joblib.dump(available,    "models/severity_feature_cols.pkl")

print("\nSaved: models/severity_model.pkl, severity_preprocessor.pkl, severity_feature_cols.pkl")
print("Done.")
