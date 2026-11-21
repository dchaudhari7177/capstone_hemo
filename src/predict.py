"""Load the saved final model and explain a single patient report.

    from src.predict import load_final_model, predict_patient
    bundle = load_final_model()
    predict_patient(bundle, raw_record)      # raw_record: dict / one-row DataFrame with MMC3 + MMC2 columns

Research decision-support only: the model sees genotype and baseline clinical data, not treatment exposure,
product type or HLA, which also drive inhibitor development.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .explain import EnsembleShap
from .pipeline import PROJECT_ROOT, build_features

MODEL_PATH = PROJECT_ROOT / "models" / "final_inhibitor_ensemble.joblib"


def load_final_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    bundle["shap"] = EnsembleShap(bundle["model"], bundle["background"])
    return bundle


def predict_patient(bundle, raw_record, top=5):
    """Predicted inhibitor probability, the decision at the saved threshold, and the top SHAP reasons."""
    raw = pd.DataFrame([raw_record]) if isinstance(raw_record, dict) else raw_record.reset_index(drop=True)
    num, cat = build_features(raw)
    X = pd.concat([num, cat], axis=1)
    model, thr = bundle["model"], bundle["threshold"]
    p = float(model.predict_proba(X)[0])
    exp = bundle["shap"](X)
    contrib = pd.Series(exp.values[0], index=exp.feature_names)
    order = contrib.abs().sort_values(ascending=False).index[:top]
    def readable(v):
        if pd.isna(v):
            return None
        try:
            return round(float(v), 3)
        except (TypeError, ValueError):
            return str(v)

    reasons = [{"input": f, "value": readable(X[f].iloc[0]),
                "effect_on_risk": round(float(contrib[f]), 4),
                "direction": "raises risk" if contrib[f] > 0 else "lowers risk"} for f in order]
    return {"probability": round(p, 4), "threshold": round(float(thr), 4),
            "flag_inhibitor_risk": bool(p >= thr),
            "risk_vs_cohort": round(p / float(bundle["prevalence"]), 2),
            "shap_base_value": round(float(exp.base_values[0]), 4), "top_reasons": reasons}
