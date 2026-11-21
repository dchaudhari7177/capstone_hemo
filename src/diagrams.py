"""Draws the design diagrams used in the report and slides (python src/diagrams.py -> results/figures/arch_*.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY, ORANGE, GREY, LIGHT, GREEN, RED = "#1f3b73", "#e8710a", "#5f6b7a", "#f3f5f8", "#2e7d32", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, lines=(), color=NAVY, fc="white", title_size=11, line_size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=1.6, ec=color, fc=fc))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=title_size, weight="bold", color=color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.52 - i * 0.27, ln, ha="center", va="top", fontsize=line_size, color="#222")


def arrow(ax, a, b, color=GREY, text=None, rad=0.0, ts=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=ts, color=color, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1. system architecture
fig, ax = canvas(16, 9.4)
ax.text(8, 9.15, "System architecture — explainable FVIII inhibitor risk classification", ha="center", fontsize=15,
        weight="bold", color=NAVY)
# layer labels
for y, lab in [(7.25, "DATA\nLAYER"), (4.75, "PROCESSING\n& MODELLING"), (1.55, "EXPLANATION\n& INTERFACE")]:
    ax.text(0.02, y, lab, fontsize=8.2, weight="bold", color=GREY, va="center")
box(ax, 1.6, 6.6, 3.2, 1.45, "MMC2 — variant table", ["6,211 F8 variants × 34 columns", "type, effect, domain, position"], ORANGE)
box(ax, 5.2, 6.6, 3.2, 1.45, "MMC3 — patient reports", ["10,064 reports × 23 columns", "severity, FVIII, inhibitor outcome"], ORANGE)
box(ax, 8.9, 6.6, 3.4, 1.45, "Biology reference tables", ["UniProt P00451 domains, epitopes", "Grantham, BLOSUM62, aa properties"], ORANGE)
box(ax, 12.7, 6.6, 3.0, 1.45, "Saved artefacts", ["final_inhibitor_ensemble.joblib", "figures, CSV result tables"], GREY)

box(ax, 1.6, 3.7, 3.4, 2.25, "1  Data preparation", ["clean & label (unknown ≠ No)", "remove shifted / gnomAD / F9 rows",
    "join MMC3 → MMC2 on mut_id", "union-find variant_key", "4,962 labelled reports"])
box(ax, 5.4, 3.7, 3.4, 2.25, "2  Feature engineering", ["parse FVIII levels, severity, exon", "13 + 11 base inputs (v1)",
    "+ 26 biology features (v2)", "= 38 numeric + 13 categorical", "row-wise → leakage-free"])
box(ax, 9.2, 3.7, 3.2, 2.25, "3  Modelling", ["LightGBM · XGBoost", "CatBoost · random forest", "mean probability ensemble",
    "threshold = max MCC", "(inner grouped CV)"])
box(ax, 12.8, 3.7, 2.9, 2.25, "4  Evaluation", ["variant-grouped CV", "5 × 5 repeats", "PR-AUC, MCC, ROC-AUC,", "F1, Brier, bootstrap CI",
    "ablation study"])

box(ax, 1.6, 0.55, 4.4, 2.3, "5  Explainability", ["EnsembleShap — exact TreeSHAP per model,", "averaged in probability space",
    "LimeView — local surrogate with", "a 'missing' bin; agreement, stability,", "faithfulness, biology checks"], GREEN)
box(ax, 6.5, 0.55, 4.3, 2.3, "6  Prediction service", ["load_final_model()", "predict_patient(record) →", "probability, decision,",
    "top-5 SHAP reasons"], GREEN)
box(ax, 11.3, 0.55, 4.4, 2.3, "7  Streamlit application", ["predict & explain one patient", "SHAP waterfall + LIME cross-check",
    "global explanations, evaluation,", "data pipeline views"], GREEN)

arrow(ax, (3.2, 6.6), (3.2, 5.95)); arrow(ax, (6.8, 6.6), (3.9, 5.95), rad=0.15)
arrow(ax, (10.6, 6.6), (7.6, 5.95), rad=0.12, text="prior knowledge")
arrow(ax, (5.0, 4.8), (5.4, 4.8)); arrow(ax, (8.8, 4.8), (9.2, 4.8)); arrow(ax, (12.4, 4.8), (12.8, 4.8))
arrow(ax, (10.8, 3.7), (4.0, 2.85), rad=-0.12, text="trained model")
arrow(ax, (11.4, 3.7), (8.6, 2.85), rad=-0.1)
arrow(ax, (6.0, 1.7), (6.5, 1.7)); arrow(ax, (10.8, 1.7), (11.3, 1.7))
arrow(ax, (12.1, 5.95), (13.6, 6.6), rad=-0.2, text="save")
save(fig, "arch_system_architecture.png")

# ---------------------------------------------------------------- 2. record flow
fig, ax = canvas(13, 6.2)
ax.text(6.5, 5.95, "Data preparation — from 10,064 reports to 4,962 labelled reports", ha="center", fontsize=14,
        weight="bold", color=NAVY)
steps = [("Raw MMC3 reports", 10064, ""), ("Shifted-column rows", 10062, "−2"), ("gnomAD population rows", 8773, "−1,289"),
         ("mut_id missing in MMC2", 8772, "−1"), ("F9 (haemophilia B) variants", 8766, "−6"),
         ("Unknown outcome (kept out, never set to 'No')", 4962, "−3,804")]
for i, (lab, n, rm) in enumerate(steps):
    y = 5.0 - i * 0.82; w = 9.5 * n / 10064
    ax.add_patch(FancyBboxPatch((0.4, y), w, 0.58, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=ORANGE if i == len(steps) - 1 else NAVY, ec="none", alpha=0.9 if i else 0.75))
    ax.text(0.55, y + 0.29, f"{n:,}", va="center", color="white", weight="bold", fontsize=11)
    ax.text(0.4 + w + 0.15, y + 0.29, f"{lab}  {rm}", va="center", fontsize=10, color="#222")
ax.text(0.4, 0.05, "Result: 4,962 reports · 836 with inhibitor (16.9 %) · 2,639 variants merged into 2,474 variant groups",
        fontsize=10.5, weight="bold", color=ORANGE)
save(fig, "arch_record_flow.png")

# ---------------------------------------------------------------- 3. evaluation protocol
fig, ax = canvas(15, 6.6)
ax.text(7.5, 6.35, "Evaluation protocol — every number is measured on variants the model has never seen", ha="center",
        fontsize=14, weight="bold", color=NAVY)
box(ax, 0.3, 2.3, 3.2, 2.6, "All 4,962 reports", ["grouped by variant_key", "(2,474 groups)", "", "a variant is never on", "both sides of a split"], ORANGE)
box(ax, 4.2, 3.5, 4.3, 2.4, "Outer loop: 5 × 5 repeated", ["StratifiedGroupKFold", "train = 4 folds of variants", "test = 1 fold of unseen variants",
    "metrics per fold, then mean ± SD"])
box(ax, 4.2, 0.4, 4.3, 2.6, "Inner loop (training part only)", ["grouped 3-fold CV", "out-of-fold probabilities", "→ threshold that maximises MCC",
    "test fold is never looked at"], GREEN)
box(ax, 9.3, 3.5, 2.6, 2.4, "Fit ensemble", ["4 models on", "the full training", "part"])
box(ax, 12.4, 3.5, 2.4, 2.4, "Score test fold", ["once, at the", "inner-CV", "threshold"], RED)
box(ax, 9.3, 0.4, 5.5, 2.0, "Reported", ["PR-AUC · MCC · ROC-AUC · F1 · balanced accuracy", "sensitivity · specificity · precision · Brier",
    "+ no-skill reference for each", "+ held-out split with group bootstrap 95 % CI"], GREY)
arrow(ax, (3.5, 4.0), (4.2, 4.6)); arrow(ax, (6.35, 3.5), (6.35, 3.0)); arrow(ax, (8.5, 4.7), (9.3, 4.7))
arrow(ax, (8.5, 2.7), (12.9, 3.5), rad=0.12); ax.text(10.4, 3.25, "threshold", fontsize=8, color=GREY, style="italic")
arrow(ax, (11.9, 4.7), (12.4, 4.7)); arrow(ax, (13.6, 3.5), (13.0, 2.4))
save(fig, "arch_evaluation_protocol.png")

# ---------------------------------------------------------------- 4. class diagram (LLD)
fig, ax = canvas(16, 8.6)
ax.set_ylim(-0.6, 8.6); ax.set_xlim(0, 16.6)
ax.text(8, 8.35, "Low-level design — main modules, classes and functions", ha="center", fontsize=14, weight="bold", color=NAVY)


def uml(ax, x, y, w, name, attrs, methods, color=NAVY):
    h = 0.55 + 0.3 * (len(attrs) + len(methods)) + 0.25
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="square,pad=0", lw=1.5, ec=color, fc="white"))
    ax.add_patch(FancyBboxPatch((x, y - 0.55), w, 0.55, boxstyle="square,pad=0", lw=1.5, ec=color, fc=color))
    ax.text(x + w / 2, y - 0.28, name, ha="center", va="center", color="white", weight="bold", fontsize=10.5)
    yy = y - 0.8
    for a in attrs:
        ax.text(x + 0.12, yy, a, fontsize=8.6, va="center", family="DejaVu Sans Mono"); yy -= 0.3
    ax.plot([x, x + w], [yy + 0.15, yy + 0.15], color=color, lw=0.8)
    for m in methods:
        ax.text(x + 0.12, yy - 0.05, m, fontsize=8.6, va="center", family="DejaVu Sans Mono"); yy -= 0.3
    return (x, y - h, w, h)


uml(ax, 0.3, 7.9, 4.6, "pipeline.py  (data)", ["MMC2, MMC3 : Path", "MISSING : set", "INHIBITOR_EPITOPES"],
    ["load_reports(keep_unknown)", "variant_key(mmc2)  # union-find", "parse_level(v) / severity(v)", "effect(v) / exon_number(v)"], ORANGE)
uml(ax, 0.3, 4.1, 4.6, "pipeline.py  (features)", ["SIG=19, MATURE=2332", "BLOSUM62, GRANTHAM tables"],
    ["base_columns(df)", "features_v1(b) -> num, cat", "features_v2(b) -> num, cat", "build_features(df)"], ORANGE)
uml(ax, 5.7, 7.9, 4.6, "InhibitorEnsemble", ["names : list[str]", "pipes : dict[name, Pipeline]", "num_cols, cat_cols"],
    ["fit(num, cat, y)", "predict_each(X) -> dict", "predict_proba(X) -> array", "feature_names (property)"])
uml(ax, 5.7, 4.1, 4.6, "evaluation helpers", ["MODELS : dict[name, factory]"],
    ["encoder(num, cat, impute)", "best_threshold(y, p, 'mcc')", "metric_panel(y, p, thr)", "fit_predict(...)"])
uml(ax, 11.1, 7.9, 4.6, "EnsembleShap", ["parts : {name: (enc, TreeExplainer, map)}", "base_value : float"],
    ["__init__(model, background)", "values_each(X)", "__call__(X) -> shap.Explanation"], GREEN)
uml(ax, 11.1, 5.1, 4.6, "LimeView", ["labels, pools, edges per input", "train_codes : ndarray"],
    ["encode(X) / decode(A, instance)", "explain(instance, k, n, seed)", "weights(lime_exp) -> Series"], GREEN)
uml(ax, 11.1, 2.3, 4.6, "predict.py", ["MODEL_PATH"], ["load_final_model()", "predict_patient(bundle, record)"], GREY)
uml(ax, 5.7, 1.3, 4.6, "app/app.py  (Streamlit)", ["bundle, df, X, lime_view"], ["tab 1 predict & explain", "tabs 2-4 global / eval / data"], GREY)
for a, b in [((4.9, 6.4), (5.7, 6.6)), ((4.9, 2.8), (5.7, 6.0)), ((10.3, 6.9), (11.1, 6.9)), ((10.3, 6.2), (11.1, 4.5)),
             ((11.1, 1.2), (10.3, 0.8)), ((8.0, 4.1), (8.0, 4.9)), ((11.1, 3.3), (10.3, 0.4))]:
    arrow(ax, a, b)
arrow(ax, (15.7, 6.0), (15.7, 1.8), rad=-0.35)
ax.text(15.95, 3.9, "uses", fontsize=8, color=GREY, style="italic", rotation=90)
save(fig, "arch_class_diagram.png")

# ---------------------------------------------------------------- 5. sequence diagram (prediction)
fig, ax = canvas(15, 7.4)
ax.text(7.5, 7.15, "Sequence of one prediction in the application", ha="center", fontsize=14, weight="bold", color=NAVY)
actors = ["User", "Streamlit app", "build_features()", "InhibitorEnsemble", "EnsembleShap", "LimeView"]
xs = [1.0, 3.6, 6.2, 8.8, 11.4, 14.0]
for x, a in zip(xs, actors):
    ax.add_patch(FancyBboxPatch((x - 1.05, 6.2), 2.1, 0.55, boxstyle="round,pad=0.02", fc=NAVY if a != "User" else ORANGE, ec="none"))
    ax.text(x, 6.47, a, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.plot([x, x], [0.3, 6.2], color="#aab", lw=1, ls="--")
msgs = [(0, 1, 5.7, "choose / edit patient, press Predict"), (1, 2, 5.05, "raw record (MMC3 + MMC2 fields)"),
        (2, 1, 4.45, "51 inputs (38 numeric + 13 categorical)"), (1, 3, 3.85, "predict_proba(X)"),
        (3, 1, 3.25, "probability → compare with threshold 0.258"), (1, 4, 2.65, "explain(X)"),
        (4, 1, 2.05, "SHAP values (sum = probability − base)"), (1, 5, 1.45, "explain(X) on request"),
        (1, 0, 0.8, "risk %, decision, waterfall, LIME bars")]
for a, b, y, t in msgs:
    arrow(ax, (xs[a], y), (xs[b], y), color=GREEN if b < a else NAVY)
    ax.text((xs[a] + xs[b]) / 2, y + 0.1, t, ha="center", fontsize=8.6, color="#222")
save(fig, "arch_sequence_prediction.png")

# ---------------------------------------------------------------- 6. use cases
fig, ax = canvas(13, 6.6)
ax.text(6.5, 6.35, "Use-case diagram", ha="center", fontsize=14, weight="bold", color=NAVY)
def actor(x, y, who):
    ax.add_patch(plt.Circle((x, y + 0.75), 0.22, fc="white", ec=NAVY, lw=1.6))
    ax.plot([x, x], [y + 0.53, y + 0.05], color=NAVY, lw=1.6); ax.plot([x - 0.3, x + 0.3], [y + 0.35, y + 0.35], color=NAVY, lw=1.6)
    ax.plot([x, x - 0.25], [y + 0.05, y - 0.35], color=NAVY, lw=1.6); ax.plot([x, x + 0.25], [y + 0.05, y - 0.35], color=NAVY, lw=1.6)
    ax.text(x, y - 0.75, who, ha="center", fontsize=9.5)
actor(1.2, 3.1, "Clinician /\nhaematologist"); actor(11.8, 3.1, "Researcher /\ndata scientist")
ax.add_patch(FancyBboxPatch((3.4, 0.2), 6.2, 5.75, boxstyle="round,pad=0.02", fc=LIGHT, ec=GREY, lw=1.2))
ax.text(6.5, 5.68, "Inhibitor risk explorer", ha="center", fontsize=11, weight="bold", color=GREY)
cases = ["Enter / edit a patient's variant and clinical data", "Get inhibitor probability and risk flag",
         "See why: SHAP waterfall of top inputs", "Cross-check the explanation with LIME",
         "Inspect global explanations and biology checks", "Review evaluation, ablation and data pipeline"]
for i, t in enumerate(cases):
    y = 5.0 - i * 0.86
    ax.add_patch(plt.matplotlib.patches.Ellipse((6.5, y), 5.4, 0.68, fc="white", ec=NAVY, lw=1.4))
    ax.text(6.5, y, t, ha="center", va="center", fontsize=9.2)
    if i <= 3:
        ax.plot([1.55, 3.8], [3.5, y], color=GREY, lw=0.9)
    if i >= 2:
        ax.plot([11.45, 9.2], [3.5, y], color=GREY, lw=0.9)
save(fig, "arch_use_cases.png")
print("diagrams written to", OUT)
