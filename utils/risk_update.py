"""
Post-questionnaire risk update.

When a patient completes the CAT questionnaire and MRC dyspnoea grade,
use these validated clinical instruments to adjust the model's base risk score.

CAT  (COPD Assessment Test):   0–10=low, 11–20=medium, 21–30=high, >30=very high
MRC dyspnoea scale:            1=mild, 2=moderate, 3=moderately severe, 4=severe, 5=very severe

The update is a Bayesian-style blending: the questionnaire provides additional
clinical signal that re-weights the model probability.
"""

import numpy as np


# CAT score impact on COPD probability
_CAT_IMPACT = {
    (0,  10):  0.00,   # minimal symptoms → no upward adjustment
    (11, 20):  0.05,   # medium symptoms  → +5pp
    (21, 30):  0.12,   # high symptoms    → +12pp
    (31, 40):  0.20,   # very high        → +20pp
}

# MRC dyspnoea impact
_MRC_IMPACT = {
    1: 0.00,
    2: 0.04,
    3: 0.10,
    4: 0.16,
    5: 0.22,
}


def _cat_delta(cat_score: int) -> float:
    for (lo, hi), delta in _CAT_IMPACT.items():
        if lo <= cat_score <= hi:
            return delta
    return 0.0


def updated_risk_score(
    base_prob: float,
    cat_score: int | None = None,
    mrc_grade: int | None = None,
) -> dict:
    """
    Blend model probability with CAT + MRC signal.

    Parameters
    ----------
    base_prob : float  — model output probability (0–1)
    cat_score : int    — CAT total score (0–40), or None if not collected
    mrc_grade : int    — MRC dyspnoea grade (1–5), or None if not collected

    Returns
    -------
    dict with keys:
        updated_prob   : float  — adjusted probability
        risk_level     : str    — "Low" / "Moderate" / "High" / "Very High"
        delta          : float  — change from base probability
        cat_interpretation : str
        mrc_interpretation : str
    """
    delta = 0.0
    cat_text = "Not collected"
    mrc_text = "Not collected"

    if cat_score is not None:
        cat_d = _cat_delta(int(cat_score))
        delta += cat_d
        if cat_score <= 10:
            cat_text = f"CAT {cat_score} — Low symptom burden"
        elif cat_score <= 20:
            cat_text = f"CAT {cat_score} — Medium symptom burden"
        elif cat_score <= 30:
            cat_text = f"CAT {cat_score} — High symptom burden"
        else:
            cat_text = f"CAT {cat_score} — Very high symptom burden"

    if mrc_grade is not None:
        mrc_d = _MRC_IMPACT.get(int(mrc_grade), 0.0)
        delta += mrc_d
        mrc_labels = {
            1: "MRC 1 — Breathless only with strenuous exercise",
            2: "MRC 2 — Short of breath when hurrying on level ground",
            3: "MRC 3 — Walks slower than peers due to breathlessness",
            4: "MRC 4 — Stops for breath after walking ~100m on level ground",
            5: "MRC 5 — Too breathless to leave house / dress independently",
        }
        mrc_text = mrc_labels.get(int(mrc_grade), f"MRC {mrc_grade}")

    # Clamp to [0, 1]
    updated = float(np.clip(base_prob + delta, 0.0, 1.0))

    # Risk level thresholds (same as main app)
    if updated < 0.30:
        risk_level = "Low"
    elif updated < 0.50:
        risk_level = "Moderate"
    elif updated < 0.70:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "updated_prob":       updated,
        "risk_level":         risk_level,
        "delta":              delta,
        "cat_interpretation": cat_text,
        "mrc_interpretation": mrc_text,
    }


def severity_label(gold_stage: int) -> str:
    """Human-readable GOLD severity label."""
    return {0: "None", 1: "MILD", 2: "MODERATE", 3: "SEVERE", 4: "VERY SEVERE"}.get(gold_stage, "Unknown")
