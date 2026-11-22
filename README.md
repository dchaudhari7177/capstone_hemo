# Explainable Inhibitor Risk Classification for Haemophilia A Using Fused Genomic–Clinical Data

Capstone project PW26_DN_03 · PES University, Department of CSE
Guide: Dr. Divyashree N · Team: Dipak Chaudhari, Tejas Nagmote, Sneha A, Varsha V P

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

| File | Contents |
|---|---|
| `data/raw/BVTH_VTH-2024-000215-mmc2.csv`, `...-mmc3.csv` | The original published tables |
| `data/processed/final_modelling_dataset.csv` / `.xlsx` | The exact table the model is trained on: 4,962 rows × 73 columns (identifiers, original fields, the 51 model inputs, target `inhibitor`); the `.xlsx` has a column-dictionary sheet |
| `data/processed/final_modelling_dataset_dictionary.csv` | Role, family and meaning of every column |
| `data/processed/final_modelling_dataset_encoded.csv` / `.xlsx` | The same rows as the model sees them: 38 numeric inputs + 147 one-hot columns from the 13 categorical inputs |

Missing inputs are kept as missing: LightGBM, XGBoost and CatBoost read them directly; the random forest fills them with
the training median and adds a "was missing" flag. Categorical inputs are one-hot encoded inside the model pipeline,
fitted on the training data of each fold only.

## Results (every test patient carries a variant the model never saw)

| | ROC-AUC | PR-AUC (no-skill 0.168) | MCC | Balanced acc. | F1 |
|---|---|---|---|---|---|
| Baseline — CatBoost, original 24 inputs (5×5 CV) | 0.760 ± 0.042 | 0.445 | 0.312 | 0.663 | 0.422 |
| **Final — 51 inputs + 4-model ensemble (5×5 CV)** | **0.768 ± 0.045** | **0.457** | **0.337** | **0.687** | **0.452** |
| Final — held-out split, 505 unseen variants | 0.790 [0.739–0.835] | 0.469 | 0.335 | 0.705 | 0.455 |
| Final — known-variant setting | 0.797 ± 0.017 | 0.493 | 0.366 | 0.703 | 0.479 |

* **Ablation:** genomic only 0.746, clinical only 0.695, fused **0.765** (fused better in 15/15 folds); position /
  protein-structure features matter most; the ensemble ties LightGBM with the best Brier score.
* **Literature protocol:** the published over-sampling protocol re-run on this data gives 94.0 % / AUC 0.985, but 65 % of
  its test rows are copies of training rows; with the leakage removed it scores 79.8 % / 0.727.
* **Explainability:** exact SHAP for the ensemble; genomic families 68 %, clinical 25 %, interactions 7 % of the
  attribution; 5/5 published risk factors reproduced. LIME shares 2.75 of SHAP's top-5 inputs (chance 0.49) with
  99.6 % direction agreement; removing the top-5 SHAP inputs moves the prediction 4× more than random inputs.

## Repository layout

```
notebooks/
  01_dataset_cleaning_and_feature_selection.ipynb      data audit and cleaning rules
  02_model_improvement_and_literature_comparison.ipynb metric panel, bias–variance diagnosis, biology features +
                                                       ensemble, before/after, literature comparison  (~27 min)
  03_final_model_explainability_SHAP_LIME.ipynb         FINAL MODEL: held-out evaluation, SHAP, LIME, agreement,
                                                       stability, faithfulness; saves the model and the datasets (~8 min)
  04_ablation_study.ipynb                              data blocks, leave-one-family-out, models     (~18 min)
src/
  pipeline.py   cleaning, biology reference tables, features, models, InhibitorEnsemble
  explain.py    EnsembleShap (exact ensemble SHAP), LimeView (LIME with a 'missing' bin)
  predict.py    load_final_model() / predict_patient()
  diagrams.py   architecture, data-flow, evaluation, class, sequence and use-case diagrams
app/app.py      Streamlit application: predict and explain one patient, global explanations, evaluation, data pipeline
tests/          31 pytest tests (parsing, leakage rules, model, SHAP, LIME, saved model)
data/           raw and processed datasets (see above)
models/         final_inhibitor_ensemble.joblib (model + decision threshold 0.258 + SHAP background)
results/        figures (PNG) and result tables (CSV/JSON) written by the notebooks
archive/        earlier modelling notebook
```

## How to run

Requires Python 3.12.

```powershell
pip install -r requirements.txt
```

```powershell
python -m pytest -q
```

```powershell
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Notebooks run top to bottom from the `notebooks/` folder (02, then 03, then 04);
notebook 03 re-creates `models/` and `data/processed/`. From Python:

```python
from src.predict import load_final_model, predict_patient
predict_patient(load_final_model(), raw_report_row)   # MMC3 + MMC2 columns of one report
```

## Limitations

* No treatment exposure, product type, treatment intensity or HLA data — the strongest known inhibitor drivers.
  12.3 % of reports share identical inputs with a report of the opposite outcome, so part of the error is irreducible.
* SHAP and LIME explain the model, not causation.
* Research decision-support only; not validated for clinical use.
