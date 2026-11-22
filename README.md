# PW26_DN_03 — Explainable Inhibitor Risk Classification for Haemophilia A Using Fused Genomic–Clinical Data

PES University · Department of CSE · UE23CS441A Capstone Project Phase 3 (Review 3, Nov 2026)
Guide: Dr. Divyashree N · Team: Dipak Chaudhari (PES1UG24CS804), Tejas Nagmote (PES1UG23CS646),
Sneha A (PES1UG23CS582), Varsha V P (PES1UG23CS670)

## For Review 3 — open these first

> The review documents in `reports/` (report, presentation, study guide, briefing) are kept on the team's machines
> and are **not published in this repository** (see `.gitignore`). Everything needed to reproduce the results —
> data, notebooks, code, tests, app and saved model — is here.

| File | What it is |
|---|---|
| `reports/PW26_DN_03_Team_Study_Guide.pdf` | **Start here (teammates):** 18-page plain-language guide — notebooks, how it works, every app page, how to run |
| `reports/PW26_DN_03_Capstone_Phase3_Report.pdf` | Project report, 59 pages, PES format (print this one) |
| `reports/PW26_DN_03_Capstone_Phase3_Report.docx` | Same report, editable in Word |
| `reports/PW26_DN_03_Review3_Presentation.pptx` | Review-3 deck on the department template, 24 slides, speaker notes on every slide |
| `reports/Review3_Team_Briefing.md` | What every team member should know: project, numbers, slide script, demo, viva Q&A |
| `app/app.py` | Demo application (`streamlit run app/app.py`) |

## Problem

People with haemophilia A lack clotting factor VIII (FVIII) because of a variant in the *F8* gene and are treated with
replacement FVIII. In 15–30 % of severe patients the immune system produces **inhibitors** — antibodies that neutralise
the replacement FVIII. This project predicts, from a patient's **F8 variant** and **baseline clinical data**, whether an
inhibitor is reported, and **explains every prediction with SHAP and LIME**.

## Data

EAHAD / HADB F8 variant database supplement (Print et al., *Blood Vessels, Thrombosis & Hemostasis* 2025):
`mmc2` = one row per variant (6,211), `mmc3` = one row per published patient report (10,064). After cleaning — unknown
outcomes removed, **never counted as "no inhibitor"** — **4,962 labelled reports, 836 with inhibitor (16.8 %),
2,639 variants in 2,474 variant groups**.

## Headline results (every test patient carries a variant the model never saw)

| | ROC-AUC | PR-AUC (no-skill 0.168) | MCC | Balanced acc. | F1 |
|---|---|---|---|---|---|
| Before — report's CatBoost, v1 features (5×5 CV) | 0.760 ± 0.042 | 0.445 | 0.312 | 0.663 | 0.422 |
| **After — v2 features + 4-model ensemble (5×5 CV)** | **0.768 ± 0.045** | **0.457** | **0.337** | **0.687** | **0.452** |
| After — held-out split, 505 unseen variants | 0.790 [0.739–0.835] | 0.469 | 0.335 | 0.705 | 0.455 |
| After — known-variant setting | 0.797 ± 0.017 | 0.493 | 0.366 | 0.703 | 0.479 |

* **Ablation (notebook 04):** genomic only 0.746, clinical only 0.695, fused **0.765** (fused better in 15/15 folds);
  position / protein-structure features matter most; the ensemble ties LightGBM with the best Brier score.
* **Literature:** the base paper's protocol re-run on our data gives 94.0 % / AUC 0.985, but 65 % of its test rows are
  copies of training rows; with the leakage removed it scores 79.8 % / 0.727.
* **SHAP:** exact for the ensemble; genomic families 68 %, clinical 25 %, interactions 7 % of the attribution;
  5/5 published risk factors reproduced. **LIME** shares 2.75 of SHAP's top-5 inputs (chance 0.49) with 99.6 % direction
  agreement; removing the top-5 SHAP inputs moves the prediction 4× more than random inputs.

## Folder layout

```
notebooks/
  01_dataset_cleaning_and_feature_selection.ipynb      data audit + cleaning rules (team pipeline; pre-executed)
  02_model_improvement_and_literature_comparison.ipynb guide feedback, bias–variance, v2 features + ensemble,
                                                       before/after, literature comparison          (~27 min)
  03_final_model_explainability_SHAP_LIME.ipynb         held-out evaluation, SHAP, LIME, agreement, stability,
                                                       faithfulness, final model saved                (~10 min)
  04_ablation_study.ipynb                              data blocks, leave-one-family-out, models     (~18 min)
src/
  pipeline.py   cleaning, biology tables, features, models, InhibitorEnsemble
  explain.py    EnsembleShap (exact ensemble SHAP), LimeView (LIME with a 'missing' bin)
  predict.py    load_final_model() / predict_patient()
  diagrams.py   architecture, record flow, evaluation protocol, class, sequence and use-case diagrams
app/app.py      Streamlit application
tests/          31 pytest tests (parsing, leakage rules, model, SHAP, LIME, saved model)
data/raw/       the two EAHAD supplementary CSVs (the original published data)
data/processed/ final_modelling_dataset.csv / .xlsx — the exact table the model is trained on (4,962 rows:
                identifiers + raw fields + 51 model inputs + target), plus its column dictionary, and
                final_modelling_dataset_encoded.csv / .xlsx — the same rows as the model sees them (13 categorical
                inputs one-hot encoded into 147 0/1 columns + 38 numeric inputs = 188 columns) (written by notebook 03)
models/         final_inhibitor_ensemble.joblib (model + threshold 0.258 + SHAP background)
results/        figures (PNG) and tables (CSV/JSON) used by the report and slides
reports/        Review-3 report and deck, team briefing, earlier review documents
archive/        earlier team notebook (v4 stack) and viva Q&A
```

## How to run

```powershell
pip install -r requirements.txt
```

```powershell
python -m pytest -q
```

```powershell
streamlit run app/app.py
```

Notebooks run top to bottom from the `notebooks/` folder (02, then 03, then 04). From Python:

```python
from src.predict import load_final_model, predict_patient
predict_patient(load_final_model(), raw_report_row)   # MMC3 + MMC2 columns of one report
```

## Limitations

* No treatment exposure, product type, treatment intensity or HLA data — the strongest known inhibitor drivers.
  12.3 % of reports share identical inputs with a report of the opposite outcome, so part of the error is irreducible.
* SHAP and LIME explain the model, not causation.
* Research decision-support only; not validated for clinical use.
