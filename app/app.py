"""FVIII inhibitor risk explorer - demo application for PW26_DN_03.

Run from the project root:   streamlit run app/app.py
Uses the final model saved by notebook 03 (models/final_inhibitor_ensemble.joblib).
Research decision-support prototype - not a clinical tool.
"""
import os
import sys
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.explain import LimeView, row_explanation
from src.pipeline import base_columns, build_features, load_reports
from src.predict import load_final_model

st.set_page_config(page_title="FVIII Inhibitor Risk Explorer", page_icon="🩸", layout="wide")

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
EDITABLE = [  # (raw column, label, kind)
    ("mut_effect", "Variant effect", "choice"), ("mut_type", "Variant type", "choice"),
    ("d_id", "Protein domain", "choice"), ("location", "Location", "choice"),
    ("cli_phe", "Clinical severity", "choice"), ("clotting", "FVIII activity (%) e.g. <1, 3, 5 to 10", "text"),
    ("antigen", "FVIII antigen (%)", "text"), ("type", "CRM type (I / II)", "text"),
    ("e_i_numb", "Exon number", "text"), ("aa_numb", "Amino-acid position", "text"),
    ("nuc_numb", "cDNA position", "text"), ("ntchange", "Nucleotide change (e.g. C>T)", "text"),
    ("aa_first", "Reference amino acid (e.g. Arg)", "text"), ("aa_last", "New amino acid (e.g. Cys, Ter)", "text"),
    ("n_bp", "Length of variant (bp)", "text"),
]


@st.cache_resource(show_spinner="Loading the final model and data ...")
def load_everything():
    bundle = load_final_model()
    df, flow = load_reports()
    num, cat = build_features(df)
    X = pd.concat([num, cat], axis=1)
    lime_view = LimeView(bundle["model"], X, X)
    p_all = bundle["model"].predict_proba(X)
    return bundle, df, flow, X, lime_view, base_columns(df), p_all


bundle, df, flow, X, lime_view, BASE, P_ALL = load_everything()
model, THR = bundle["model"], bundle["threshold"]
PREV = bundle["prevalence"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### PW26_DN_03")
    st.markdown("**Explainable Inhibitor Risk Classification for Haemophilia A Using Fused Genomic–Clinical Data**")
    st.caption("PES University · Dept. of CSE · Capstone Phase 3")
    st.divider()
    st.markdown("**Final model**")
    st.markdown("Mean of LightGBM, XGBoost, CatBoost and random forest on 51 genomic + clinical inputs")
    st.metric("Trained on", f"{len(df):,} reports")
    st.metric("Decision threshold", f"{THR:.3f}")
    st.metric("Grouped-CV ROC-AUC (unseen variants)", "0.768")
    st.divider()
    st.caption("Research prototype for decision support. Inhibitor risk also depends on treatment exposure, product "
               "type and immune genetics, which this model does not see.")

st.title("FVIII Inhibitor Risk Explorer")
st.caption("Predicts whether a person with haemophilia A develops FVIII inhibitors from their F8 variant and baseline "
           "clinical data, and explains every prediction with SHAP and LIME.")

tab_pred, tab_global, tab_eval, tab_data = st.tabs(
    ["1 · Predict and explain", "2 · What the model learnt", "3 · How well it works", "4 · Data and pipeline"])

# ---------------------------------------------------------------- tab 1
with tab_pred:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.subheader("Patient")
        b = BASE
        def pick(mask, highest=True):
            idx = np.where(mask.to_numpy())[0]
            return int(idx[np.argmax(P_ALL[idx])] if highest else idx[np.argmin(P_ALL[idx])])
        examples = {
            "Large deletion · severe · FVIII <1 %": pick((b["effect"] == "large_deletion") & (b["severity"] == "severe") & (df["y"] == 1)),
            "Nonsense · severe": pick((b["effect"] == "nonsense") & (b["severity"] == "severe")),
            "Missense · mild": pick((b["effect"] == "missense") & (b["severity"] == "mild") & (df["y"] == 0), highest=False),
            "Missense · severe · C2 domain": pick((b["effect"] == "missense") & (b["severity"] == "severe") & (b["domain"] == "c2")),
        }
        choice = st.selectbox("Start from an example report", list(examples) + ["Pick any report by row number"])
        if choice == "Pick any report by row number":
            idx = int(st.number_input("Row number (0 – 4961)", 0, len(df) - 1, 0))
        else:
            idx = int(examples[choice])
        base = df.iloc[idx]
        st.caption(f"Report {idx}: {base['mut_syn'] or 'variant name not given'} · recorded outcome: "
                   f"**{'inhibitor' if base['y'] == 1 else 'no inhibitor'}** (the model never sees this)")
        with st.form("patient"):
            st.markdown("Edit any value, then press **Predict**.")
            values = {}
            c1, c2 = st.columns(2)
            for i, (col, label, kind) in enumerate(EDITABLE):
                box = c1 if i % 2 == 0 else c2
                current = str(base[col]) if pd.notna(base[col]) else ""
                if kind == "choice":
                    opts = sorted(set(df[col].dropna().astype(str)) - {""})
                    values[col] = box.selectbox(label, opts, index=opts.index(current) if current in opts else 0, key=col)
                else:
                    values[col] = box.text_input(label, current, key=col)
            go = st.form_submit_button("Predict", type="primary", width="stretch")

    with right:
        record = base.drop(labels=["y"]).copy()
        for col, v in values.items():
            record[col] = v
        raw = pd.DataFrame([record])
        num1, cat1 = build_features(raw)
        Xi = pd.concat([num1, cat1], axis=1)
        p = float(model.predict_proba(Xi)[0])
        flagged = p >= THR
        st.subheader("Prediction")
        m1, m2, m3 = st.columns(3)
        m1.metric("Inhibitor probability", f"{p:.1%}")
        m2.metric("vs cohort average", f"{p / PREV:.1f}×", help=f"Cohort prevalence {PREV:.1%}")
        m3.metric("Decision", "High risk" if flagged else "Low risk", help=f"Flagged when probability ≥ {THR:.3f}")
        (st.error if flagged else st.success)(
            f"{'Flagged for closer monitoring' if flagged else 'Not flagged'}: predicted probability {p:.1%} is "
            f"{'above' if flagged else 'below'} the decision threshold {THR:.1%}.")
        fig, ax = plt.subplots(figsize=(7, 0.9))
        ax.barh([0], [1], color="#eeeeee"); ax.barh([0], [p], color="#c0392b" if flagged else "#2e86c1")
        ax.axvline(THR, color="k", ls="--", lw=1); ax.text(THR, 0.55, f" threshold {THR:.2f}", fontsize=8)
        ax.set(xlim=(0, 1), yticks=[]); ax.set_xticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, width="stretch"); plt.close(fig)

        st.subheader("Why — SHAP (exact contribution of each input)")
        exp = bundle["shap"](Xi)
        shap.plots.waterfall(row_explanation(exp, 0), max_display=10, show=False)
        fig = plt.gcf(); fig.set_size_inches(7, 4.6)
        st.pyplot(fig, width="stretch"); plt.close(fig)
        st.caption("Starts at the average predicted risk (bottom) and adds each input's contribution. Red raises the "
                   "risk, blue lowers it.")

        with st.expander("Cross-check with LIME (local surrogate model)", expanded=False):
            le = lime_view.explain(Xi, num_features=8, num_samples=3000, seed=0)
            lst = le.as_list(label=1)[::-1]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh([n[:55] for n, _ in lst], [w for _, w in lst], color=["#c0392b" if w > 0 else "#2e86c1" for _, w in lst])
            ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("LIME weight (red raises risk)")
            st.pyplot(fig, width="stretch"); plt.close(fig)
            st.caption(f"LIME fits a simple linear model to 3,000 perturbed copies of this patient "
                       f"(local fit R² = {le.score:.2f}). It should broadly agree with SHAP.")

# ---------------------------------------------------------------- tab 2
with tab_global:
    st.subheader("Global explanation — what drives the predictions")
    st.image(str(FIG / "02_shap_global_importance.png"), width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "03_shap_beeswarm.png"), width="stretch")
    with c2:
        st.markdown("**Does the model follow known biology?**")
        st.dataframe(pd.read_csv(TAB / "shap_biology_checks.csv"), hide_index=True, width="stretch")
        st.markdown("**SHAP vs LIME on 100 held-out patients**")
        ag = pd.read_csv(TAB / "shap_lime_agreement.csv")
        st.write(f"Shared top-5 inputs: **{ag['top5_overlap'].mean():.2f} of 5** (chance 0.49) · "
                 f"direction agreement **{ag['sign_agreement_top5'].mean():.1%}**")
    st.image(str(FIG / "04_shap_biology_check.png"), width="stretch")

# ---------------------------------------------------------------- tab 3
with tab_eval:
    st.subheader("Held-out evaluation on variants the model never saw")
    st.image(str(FIG / "01_holdout_performance.png"), width="stretch")
    st.dataframe(pd.read_csv(TAB / "final_holdout_metrics.csv").rename(columns={"Unnamed: 0": "metric"}).round(3),
                 hide_index=True, width="stretch")
    abl = FIG / "10_ablation_study.png"
    if abl.exists():
        st.subheader("Ablation study")
        st.image(str(abl), width="stretch")
    st.subheader("Comparison with published work")
    st.dataframe(pd.read_csv(TAB / "literature_comparison.csv"), hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 4
with tab_data:
    st.subheader("From two source tables to one modelling dataset")
    st.dataframe(flow, hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Labelled patient reports", f"{len(df):,}")
    c2.metric("With inhibitor", f"{int(df['y'].sum())} ({df['y'].mean():.1%})")
    c3.metric("Variant groups (never split)", f"{df['variant_key'].nunique():,}")
    st.markdown("**Pipeline modules**: `load_reports()` cleans MMC3 and joins MMC2 → `build_features()` makes 51 "
                "row-wise inputs → `InhibitorEnsemble` (4 tree models) → `EnsembleShap` / `LimeView` explain → "
                "this app shows the result.")
    st.dataframe(X.head(20), width="stretch")
