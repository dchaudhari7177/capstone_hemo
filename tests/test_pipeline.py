"""Automated tests for the final pipeline (run from the project root:  python -m pytest -q).

Unit tests      - parsing rules for the messy source columns
Data-integrity  - the leakage rules the whole evaluation depends on
Model tests     - the ensemble, the SHAP explanation and the saved model behave as documented
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.pipeline import (InhibitorEnsemble, build_features, effect, exon_number, grantham,  # noqa: E402
                          load_reports, parse_level, severity)


# ---------------------------------------------------------------- unit tests: parsing
@pytest.mark.parametrize("raw, value, kind", [
    ("<1", 0.5, "upper_bound"),          # censored: half the bound
    ("23 to 40", 31.5, "range_mid"),     # range: midpoint
    ("12", 12.0, "exact"),
    ("9|<1?", 9.0, "exact"),             # multi-value cell: first value
    ("", math.nan, "missing"),
    ("Not reported", math.nan, "missing"),
])
def test_parse_level(raw, value, kind):
    v, k = parse_level(raw)
    assert k == kind
    assert (math.isnan(v) and math.isnan(value)) or v == pytest.approx(value)


@pytest.mark.parametrize("raw, expected", [
    ("Severe", "severe"), ("svere", "severe"), ("Moderare", "moderate"), ("Mild", "mild"),
    ("mild/moderate", "mild_moderate"), ("Non-severe", "moderate"), ("Not reported", "unknown"), ("", "unknown"),
])
def test_severity(raw, expected):
    assert severity(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Large deletion", "large_deletion"), ("Nonsense", "nonsense"), ("Frameshift", "frameshift"),
    ("Missense", "missense"), ("Splice site", "splice"), ("", "unknown"),
])
def test_effect(raw, expected):
    assert effect(raw) == expected


def test_exon_number_rejects_spreadsheet_dates():
    assert exon_number("14") == 14.0
    assert math.isnan(exon_number("41306"))      # Excel date serial, not an exon
    assert math.isnan(exon_number("Jul-13"))


def test_grantham_known_values():
    assert grantham("A", "A") == pytest.approx(0.0)
    assert grantham("C", "W") > grantham("L", "I")   # very different vs. very similar residues
    assert math.isnan(grantham("A", "*"))


# ---------------------------------------------------------------- data-integrity tests
@pytest.fixture(scope="module")
def data():
    df, flow = load_reports()
    num, cat = build_features(df)
    return df, flow, num, cat


def test_record_counts(data):
    df, flow, num, cat = data
    assert len(df) == 4962 and int(df["y"].sum()) == 836
    assert num.shape[1] + cat.shape[1] == 51


def test_unknown_outcomes_are_never_negative(data):
    df, flow, *_ = data
    assert df["y"].isin([0, 1]).all()
    assert set(df["Inhibitors"].str.strip().str.lower()) == {"yes", "no"}
    with_unknown, _ = load_reports(keep_unknown=True)
    assert with_unknown["y"].isna().sum() == 3804           # kept as missing, not 0


def test_no_variant_in_both_train_and_test(data):
    df, _, num, _ = data
    y, g = df["y"].astype(int).to_numpy(), df["variant_key"].to_numpy(dtype=object)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(num, y, g):
        assert not set(g[tr]) & set(g[te])


def test_no_outcome_derived_column_is_a_feature(data):
    *_, num, cat = data
    forbidden = {"Inhibitors", "y", "uinhibitor", "inhibitor_positive_rate", "mut_id", "mut_syn", "aa_syn", "ref_id"}
    assert not forbidden & (set(num.columns) | set(cat.columns))


def test_features_are_row_wise(data):
    df, _, num, cat = data
    sub = df.sample(25, random_state=1)
    n2, c2 = build_features(sub.reset_index(drop=True))
    pd.testing.assert_frame_equal(n2, num.loc[sub.index].reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(c2, cat.loc[sub.index].reset_index(drop=True), check_dtype=False)


# ---------------------------------------------------------------- model tests
@pytest.fixture(scope="module")
def small_model(data):
    df, _, num, cat = data
    y, g = df["y"].astype(int).to_numpy(), df["variant_key"].to_numpy(dtype=object)
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(num, y, g))
    m = InhibitorEnsemble(["LightGBM", "XGBoost"]).fit(num.iloc[tr], cat.iloc[tr], y[tr])
    return m, pd.concat([num, cat], axis=1), y, tr, te


def test_ensemble_beats_chance_on_unseen_variants(small_model):
    from sklearn.metrics import roc_auc_score
    m, X, y, tr, te = small_model
    p = m.predict_proba(X.iloc[te])
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y[te], p) > 0.70


def test_shap_values_add_up_to_prediction(small_model):
    from src.explain import EnsembleShap
    m, X, y, tr, te = small_model
    exp = EnsembleShap(m, X.iloc[tr].sample(50, random_state=0))(X.iloc[te[:40]])
    gap = np.abs(exp.base_values + exp.values.sum(1) - m.predict_proba(X.iloc[te[:40]]))
    assert np.median(gap) < 1e-3 and gap.max() < 0.05


def test_lime_returns_weights_for_known_inputs(small_model):
    from src.explain import LimeView
    m, X, y, tr, te = small_model
    lv = LimeView(m, X.iloc[tr], X)
    w = lv.weights(lv.explain(X.iloc[[te[0]]], num_features=5, num_samples=500))
    assert (w != 0).sum() == 5 and set(w.index) == set(m.feature_names)


MODEL_FILE = ROOT / "models" / "final_inhibitor_ensemble.joblib"


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="run notebook 03 first to create the saved model")
def test_saved_model_predicts_one_patient(data):
    from src.predict import load_final_model, predict_patient
    df, *_ = data
    bundle = load_final_model()
    out = predict_patient(bundle, df.iloc[[0]].drop(columns=["y"]))
    assert 0 <= out["probability"] <= 1
    assert out["flag_inhibitor_risk"] == (out["probability"] >= out["threshold"])
    assert len(out["top_reasons"]) == 5
