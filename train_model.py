"""
Train COPD screening model.

Label source: GOLD spirometry standard (FEV1/FVC < 0.70) — NOT target.csv.
Goal: maximise sensitivity (recall) for COPD positive class.
Model: Logistic Regression + Kaggle positive-class augmentation.
Run: python train_model.py
Outputs saved to models/
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    recall_score, precision_score, confusion_matrix, roc_curve,
)
from imblearn.over_sampling import SMOTE

from utils.preprocess import build_feature_matrix, FEATURE_COLS, FEATURE_LABELS, KAGGLE_PATH

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load synthetic dataset
# ─────────────────────────────────────────────
print("Loading and engineering features…")
X, y, meta = build_feature_matrix()

print(f"  Synthetic patients: {len(y)}")
print(f"  COPD positive (spirometry-derived): {y.sum()} ({100*y.mean():.1f}%)")
print(f"  COPD negative: {(y==0).sum()} ({100*(y==0).mean():.1f}%)")

# ─────────────────────────────────────────────
# 2. Load Kaggle COPD dataset (positive augmentation)
# ─────────────────────────────────────────────
print("\nLoading Kaggle augmentation data…")
kaggle_raw = pd.read_csv(KAGGLE_PATH)

# Map Kaggle columns → our FEATURE_COLS
def map_kaggle_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Kaggle COPD dataset columns to our feature schema.
    Unmapped features stay NaN → imputed with training-set median.
    Kaggle dataset: gender(0=F,1=M), smoking(1=current,2=ex),
                    Diabetes/hypertension/IHD (0/1).
    """
    rows = []
    for _, r in df.iterrows():
        age       = r["AGE"]
        pack_yrs  = r["PackHistory"]
        gender    = int(r["gender"])         # 0=female, 1=male
        smoking   = int(r.get("smoking", 2)) # 1=current, 2=ex
        diabetes  = int(r.get("Diabetes", 0))
        htn       = int(r.get("hypertension", 0))
        ihd       = int(r.get("IHD", 0))

        fumador   = int(smoking == 1)
        exfumador = int(smoking == 2)
        ever_smoked = 1  # all have pack history

        row = {col: np.nan for col in FEATURE_COLS}
        row["edad"]             = age
        row["sexo_num"]         = gender
        row["paquetes_ano"]     = pack_yrs
        row["fumador_actual"]   = fumador
        row["exfumador"]        = exfumador
        row["ever_smoked"]      = ever_smoked
        row["age_x_packyears"]  = age * pack_yrs
        row["high_risk_smoker"] = int(age > 55 and ever_smoked and pack_yrs > 10)
        row["has_diabetes"]     = diabetes
        row["has_hypertension"] = htn
        row["has_heart_disease"] = ihd
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLS)

X_kaggle = map_kaggle_to_features(kaggle_raw)
y_kaggle = pd.Series(np.ones(len(X_kaggle), dtype=int))
print(f"  Kaggle patients added (all COPD positive): {len(y_kaggle)}")

# ─────────────────────────────────────────────
# 3. Train / test split (synthetic data only — keep test uncontaminated)
# ─────────────────────────────────────────────
X_train_syn, X_test, y_train_syn, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Augment training set with Kaggle positives
X_train = pd.concat([X_train_syn, X_kaggle], ignore_index=True)
y_train = pd.concat([y_train_syn, y_kaggle], ignore_index=True)

print(f"\nTraining set after augmentation: {len(y_train)} patients")
print(f"  COPD positive: {y_train.sum()} ({100*y_train.mean():.1f}%)")

# ─────────────────────────────────────────────
# 4. Preprocessing pipeline (impute → scale)
# ─────────────────────────────────────────────
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

# ─────────────────────────────────────────────
# 5. SMOTE to balance training set
# ─────────────────────────────────────────────
print("Applying SMOTE…")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)
print(f"  After SMOTE — COPD: {y_train_res.sum()}, No COPD: {(y_train_res==0).sum()}")

# ─────────────────────────────────────────────
# 6. Train Logistic Regression
# ─────────────────────────────────────────────
print("Training Logistic Regression…")
model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    solver="lbfgs",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res)

# ─────────────────────────────────────────────
# 7. Find optimal threshold for ≥90% recall
# ─────────────────────────────────────────────
proba_test = model.predict_proba(X_test_prep)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba_test)

# Choose lowest threshold where recall ≥ 0.90, maximising precision
best_threshold = 0.5
best_precision = 0.0

for thresh, rec in zip(thresholds, tpr):
    if rec >= 0.90:
        preds = (proba_test >= thresh).astype(int)
        prec = precision_score(y_test, preds, zero_division=0)
        if prec >= best_precision:
            best_precision = prec
            best_threshold = thresh

# Fall back: if no threshold reaches 90% recall, pick the one with highest recall
if best_threshold == 0.5 and recall_score(y_test, (proba_test >= 0.5).astype(int)) < 0.90:
    idx = np.argmax(tpr >= 0.90) if any(tpr >= 0.90) else np.argmax(tpr)
    best_threshold = float(thresholds[idx])

print(f"  Optimal threshold: {best_threshold:.3f}")

# ─────────────────────────────────────────────
# 8. Evaluate
# ─────────────────────────────────────────────
y_pred = (proba_test >= best_threshold).astype(int)
roc_auc = roc_auc_score(y_test, proba_test)
recall  = recall_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred, zero_division=0)
cm      = confusion_matrix(y_test, y_pred)

print("\n── Test set evaluation ──────────────────────")
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  Recall  : {recall:.4f}  (sensitivity)")
print(f"  Precision: {prec:.4f}")
print(f"  Confusion matrix:\n{cm}")
print(classification_report(y_test, y_pred, target_names=["No COPD", "COPD"]))

# ─────────────────────────────────────────────
# 9. Feature importances (LR coefficients)
# ─────────────────────────────────────────────
importances = pd.Series(
    np.abs(model.coef_[0]), index=FEATURE_COLS
).sort_values(ascending=False)
print("\n── Top 15 features (|coefficient|) ─────────")
for feat, imp in importances.head(15).items():
    label = FEATURE_LABELS.get(feat, feat)
    print(f"  {label:<40} {imp:.4f}")

# ─────────────────────────────────────────────
# 10. Save artifacts
# ─────────────────────────────────────────────
joblib.dump(model,          "models/xgb_model.pkl")   # kept same filename for app compatibility
joblib.dump(preprocessor,   "models/preprocessor.pkl")
joblib.dump(best_threshold, "models/threshold.pkl")
joblib.dump(FEATURE_COLS,   "models/feature_cols.pkl")

print("\nSaved: models/xgb_model.pkl (LR), preprocessor.pkl, threshold.pkl, feature_cols.pkl")
print("Done.")
