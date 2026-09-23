# Temporal EHR Intelligence Platform

**[→ Visual walkthrough](https://akshit9162.github.io/Temporal-EHR-Intelligence-Platform/)**

A Streamlit application that predicts whether a patient has a chosen medical condition from multi-table electronic health records. It then asks what happens to those models when the data shifts over time. The same pipeline covers EDA, feature engineering, training, evaluation, temporal-shift analysis, continual learning and feature importance.

## Data

The data is a Synthea-style synthetic EHR dataset, stored as 7 relational CSV tables under `Dataset/`:

| Table | Used for |
|---|---|
| `patients.csv` | Demographics, income, coverage; first-encounter year for the temporal split |
| `encounters.csv` | Visit count, cost mean and std, claim mean and std, payer coverage |
| `conditions.csv` | The target label, plus a comorbidity count |
| `medications.csv`, `procedures.csv` | Medication and procedure counts |
| `observations.csv` | Mean and std of vitals and labs (BMI, blood pressure, heart rate, glucose, hemoglobin, …) |
| `immunizations.csv` | EDA |

**Target:** 1 if the patient was ever diagnosed with the selected condition. You choose it from the 15 most prevalent conditions.

## Temporal split

Patients are split on the year of their first recorded encounter:

- **D1, historical:** first encounter before 2020. Used for training and in-distribution evaluation.
- **D2, current:** first encounter in 2020 or later. Used to measure temporal shift and for continual learning.

The cutoff marks the COVID-19 onset, a well-documented break in healthcare utilisation. Every evaluation uses held-out test splits, never the full datasets.

## Models

| Model | Configuration | Scaling |
|---|---|---|
| Decision Tree | `max_depth=10`, `min_samples_split=10` | none |
| SVM (RBF) | `C=1.0`, `gamma='scale'`, probability outputs | `StandardScaler` |
| MLP | hidden layers `(100, 50)`, early stopping | `StandardScaler` |

## Continual learning

- **MLP:** deep-copied and fine-tuned on D2 with `warm_start=True`, so training resumes from the D1 weights instead of starting over.
- **Decision Tree and SVM:** retrained on D1 + D2 combined, since neither supports incremental updates.

The dashboard reports accuracy and F1 on D2 before and after the update.

## Leakage controls

- `HEALTHCARE_EXPENSES` is excluded from the features.
- The comorbidity count (`CONDITION_COUNT`) excludes the target diagnosis. Without that, every positive patient would have a count of at least 1, and the feature would encode the label.

**Known limitation:** medication, procedure and observation aggregates cover a patient's whole record. For positive patients that includes events after diagnosis, some of which (for example, treatment for the condition) are downstream of the label. A stricter version would cut each patient's history at the diagnosis date.

## Dashboard

| Tab | Contents |
|---|---|
| EDA | Class balance, top conditions, demographics, vitals, correlations for D1 and D2 |
| Preprocessing | Target definition, split justification, feature table, D1-vs-D2 drift statistics |
| Training | Decision Tree, SVM and MLP trained on D1 |
| Evaluation | Accuracy, precision, recall, F1, confusion matrices, ROC curves |
| Complexity | Tree depth, SVM `C` sweep, MLP loss curve |
| Temporal shift | D1-trained models on the D2 test set: accuracy and F1 drop |
| Continual learning | D2 performance before and after the update |
| Feature importance | Gini importance (tree) and permutation importance (all models) |

## Run

```bash
pip install streamlit pandas numpy scikit-learn plotly
streamlit run "Temporal EHR Intelligence Platform.py"
```

Place the seven CSVs in `Dataset/` next to the script. This is team coursework (BITS Pilani, Machine Learning).
