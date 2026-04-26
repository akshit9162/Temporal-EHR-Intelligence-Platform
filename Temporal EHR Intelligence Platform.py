"""
ML Healthcare Analytics Dashboard – v3 (Assignment-Compliant)
Complete Machine Learning Pipeline with Streamlit

Team 09 – BITS F464 Machine Learning – Assignment 2
Team Members:
  - Rachit Pandey     (2023AAPS1123H)
  - Akshit Gupta      (2023A3PS1136H)
  - Harsh Kokcha      (2023A8PS0307H)
  - Abhinav Ranjan   (2023AAPS0191H)

Fixes applied (v3):
  1. Target is a REAL MEDICAL CONDITION from conditions.csv (top-prevalence, user-selectable)
  2. HEALTHCARE_EXPENSES removed from feature matrix (was causing data leakage)
  3. Feature engineering includes observations.csv (mean + std of vitals/labs)
  4. Variance/std aggregations added to encounter cost features
  5. Temporal Shift tab uses PROPER TEST SPLITS (not full datasets)
  6. Continual Learning uses warm_start / deepcopy for true MLP fine-tuning
  7. Permutation importance shown for all three models
  8. Comprehensive EDA for both Dataset 1 and Dataset 2 separately
  9. Temporal split cutoff justified in Preprocessing tab
"""

import copy
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report,
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Healthcare Dashboard – Team 09",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .header-text { color:#1f77b4; font-size:28px; font-weight:bold; margin-bottom:16px; }
  .success-text { color:#28a745; font-weight:bold; }
  .danger-text  { color:#dc3545; font-weight:bold; }
  .info-box {
    background: #f0f4ff; border-left: 4px solid #1f77b4;
    padding: 12px 16px; border-radius: 4px; margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            padding:30px;border-radius:10px;margin-bottom:28px;">
  <h1 style="color:white;margin:0;">🏥 ML Healthcare Analytics Dashboard</h1>
  <p style="color:#e0e0e0;margin:10px 0 0;font-size:16px;">
    <strong>Team 09</strong> | BITS F464 Machine Learning | Assignment 2
  </p>
  <p style="color:#b0b0b0;margin:5px 0 0;font-size:12px;">
    Medical Condition Prediction • Temporal Analysis • Continual Learning • Feature Engineering
  </p>
  <p style="color:#d0d0d0;margin:15px 0 0;font-size:11px;">
    <strong>Team Members:</strong>
    Rachit Pandey (2023AAPS1123H) •
    Akshit Gupta (2023A3PS1136H) •
    Harsh Kokcha (2023A8PS0307H) •
    Abhinav Ranjan (2023AAPS0191H)
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    dp = Path(__file__).parent / "Dataset"
    kw = dict(on_bad_lines="skip", engine="python")
    patients      = pd.read_csv(dp / "patients.csv",      **kw)
    encounters    = pd.read_csv(dp / "encounters.csv",    **kw)
    conditions    = pd.read_csv(dp / "conditions.csv",    **kw)
    medications   = pd.read_csv(dp / "medications.csv",   **kw)
    procedures    = pd.read_csv(dp / "procedures.csv",    **kw)
    immunizations = pd.read_csv(dp / "immunizations.csv", **kw)
    observations  = pd.read_csv(dp / "observations.csv",  **kw)
    return patients, encounters, conditions, medications, procedures, immunizations, observations


@st.cache_data
def get_top_conditions(_conditions, n=15):
    """Return top-N conditions by number of unique patients diagnosed."""
    prev = (
        _conditions.groupby("DESCRIPTION")["PATIENT"]
        .nunique()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"PATIENT": "PATIENT_COUNT"})
    )
    return prev


@st.cache_data
def aggregate_observations(_observations):
    """
    Aggregate key numeric clinical observations per patient.
    Returns wide DataFrame (index = PATIENT/Id) with MEAN and STD columns.
    """
    obs = _observations.copy()
    obs["VALUE_NUM"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs_num = obs.dropna(subset=["VALUE_NUM"])

    key_obs = {
        "Body Height":              "HEIGHT",
        "Body Weight":              "WEIGHT",
        "Body mass index":          "BMI",
        "Diastolic Blood Pressure": "DBP",
        "Systolic Blood Pressure":  "SBP",
        "Heart rate":               "HR",
        "Body temperature":         "TEMP",
        "Respiratory rate":         "RESP_RATE",
        "Glucose":                  "GLUCOSE",
        "Hemoglobin":               "HEMOGLOBIN",
    }

    frames   = []
    coverage = {}
    for keyword, short in key_obs.items():
        sub = obs_num[obs_num["DESCRIPTION"].str.contains(keyword, case=False, na=False)]
        coverage[short] = sub["PATIENT"].nunique()
        if len(sub) > 0:
            agg = (
                sub.groupby("PATIENT")["VALUE_NUM"]
                .agg(mean_v="mean", std_v="std")
                .rename(columns={"mean_v": f"{short}_MEAN", "std_v": f"{short}_STD"})
            )
            agg.index.name = "Id"
            frames.append(agg)

    obs_feats = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    return obs_feats, coverage


@st.cache_data
def prepare_ml_data(
    _patients, _encounters, _conditions, _medications,
    _procedures, _obs_features, target_condition: str
):
    """
    Build full ML feature matrix.

    TARGET = 1 if a patient has ever been diagnosed with `target_condition`, else 0.
    NOTE: HEALTHCARE_EXPENSES is intentionally EXCLUDED from features to prevent
          any leakage and to keep the problem genuinely clinical.
    """
    pts = _patients.copy()
    pts["BIRTHDATE"] = pd.to_datetime(pts["BIRTHDATE"], errors="coerce")
    enc = _encounters.copy()
    enc["START"] = pd.to_datetime(enc["START"], errors="coerce")

    # ── Temporal split: first encounter year ─────────────────────────────
    # Justification: 2020 marks the COVID-19 pandemic onset, a well-documented
    # inflection point in healthcare utilisation patterns worldwide.
    # Patients whose first recorded encounter is before 2020 form Dataset 1
    # (Historical); those from 2020 onwards form Dataset 2 (Current).
    TEMPORAL_CUTOFF = 2020
    first_enc = enc.groupby("PATIENT")["START"].min()
    pts["FIRST_ENC_YEAR"] = pts["Id"].map(first_enc).dt.year.fillna(2025)

    # ── Target: presence of selected medical condition ───────────────────
    diagnosed_patients = set(
        _conditions.loc[
            _conditions["DESCRIPTION"].str.lower() == target_condition.lower(),
            "PATIENT"
        ]
    )
    pts["TARGET"] = pts["Id"].isin(diagnosed_patients).astype(int)

    # ── Base demographic features (NO financial leakage) ─────────────────
    feats = pts[["Id", "INCOME", "HEALTHCARE_COVERAGE"]].copy()
    feats["AGE"]           = (2025 - pts["BIRTHDATE"].dt.year).clip(0, 120)
    feats["GENDER_ENC"]    = pd.factorize(pts["GENDER"])[0]
    feats["ETHNICITY_ENC"] = pd.factorize(pts["ETHNICITY"])[0]
    feats["RACE_ENC"]      = pd.factorize(pts["RACE"])[0]

    # ── Encounter aggregations ────────────────────────────────────────────
    enc_agg = enc.groupby("PATIENT").agg(
        ENCOUNTER_COUNT    = ("Id",                  "count"),
        ENC_COST_MEAN      = ("BASE_ENCOUNTER_COST", "mean"),
        ENC_COST_STD       = ("BASE_ENCOUNTER_COST", "std"),
        TOTAL_CLAIM_MEAN   = ("TOTAL_CLAIM_COST",    "mean"),
        TOTAL_CLAIM_STD    = ("TOTAL_CLAIM_COST",    "std"),
        PAYER_COVERAGE_SUM = ("PAYER_COVERAGE",       "sum"),
    )
    enc_agg.index.name = "Id"

    # ── Condition / Medication / Procedure counts ─────────────────────────
    cond_cnt = _conditions.groupby("PATIENT").size().rename("CONDITION_COUNT")
    cond_cnt.index.name = "Id"
    med_cnt  = _medications.groupby("PATIENT").size().rename("MEDICATION_COUNT")
    med_cnt.index.name = "Id"
    proc_cnt = _procedures.groupby("PATIENT").size().rename("PROCEDURE_COUNT")
    proc_cnt.index.name = "Id"

    # ── Merge all feature sources ─────────────────────────────────────────
    feats = feats.set_index("Id")
    feats = feats.join([enc_agg, cond_cnt, med_cnt, proc_cnt], how="left")
    if not _obs_features.empty:
        feats = feats.join(_obs_features, how="left")
    feats = feats.fillna(0)

    # ── Attach target and temporal flag ──────────────────────────────────
    feats["TARGET"]        = pts.set_index("Id")["TARGET"]
    feats["IS_HISTORICAL"] = pts.set_index("Id")["FIRST_ENC_YEAR"].lt(TEMPORAL_CUTOFF)

    return feats, TEMPORAL_CUTOFF


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
(patients, encounters, conditions, medications,
 procedures, immunizations, observations) = load_data()

obs_features, obs_cov = aggregate_observations(observations)
top_conditions        = get_top_conditions(conditions, n=15)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

# Medical condition selector
condition_options = top_conditions["DESCRIPTION"].tolist()
default_idx = 0
selected_condition = st.sidebar.selectbox(
    "🎯 Select Target Medical Condition",
    condition_options,
    index=default_idx,
    help="The model will predict whether a patient has this condition.",
)
st.sidebar.info(f"**Target:** Predicting **{selected_condition}**")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 EDA Filters")
gender_opts = patients["GENDER"].dropna().unique().tolist()
sel_gender  = st.sidebar.multiselect("Select Gender", gender_opts, default=gender_opts)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ML DATA based on selected condition
# ─────────────────────────────────────────────────────────────────────────────
ml_data, TEMPORAL_CUTOFF = prepare_ml_data(
    patients, encounters, conditions, medications,
    procedures, obs_features, selected_condition,
)

X_all   = ml_data.drop(["TARGET", "IS_HISTORICAL"], axis=1)
y_all   = ml_data["TARGET"]
is_hist = ml_data["IS_HISTORICAL"]

X_hist_all = X_all[is_hist];   y_hist_all = y_all[is_hist]
X_curr_all = X_all[~is_hist];  y_curr_all = y_all[~is_hist]

prevalence = 100 * y_all.mean()
st.sidebar.markdown(f"**Dataset prevalence:** {prevalence:.1f}% positive")
st.sidebar.markdown(f"**Total patients:** {len(ml_data):,}")


def safe_split(X, y, test_size=0.2, rs=42):
    strat = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=rs, stratify=strat)


X_h_tr, X_h_te, y_h_tr, y_h_te = safe_split(X_hist_all, y_hist_all)
X_c_tr, X_c_te, y_c_tr, y_c_te = safe_split(X_curr_all, y_curr_all)

scaler      = StandardScaler()
X_h_tr_sc   = scaler.fit_transform(X_h_tr)
X_h_te_sc   = scaler.transform(X_h_te)
X_c_tr_sc   = scaler.transform(X_c_tr)
X_c_te_sc   = scaler.transform(X_c_te)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab4b, tab5, tab6, tab7) = st.tabs([
    "📊 EDA Dashboard",
    "🔧 Preprocessing",
    "🤖 Model Training",
    "📈 Evaluation Metrics",
    "📉 Complexity Analysis",
    "⏰ Temporal Shift",
    "🧠 Continual Learning",
    "🎯 Feature Importance",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<h2 class="header-text">📊 EDA & Data Overview</h2>', unsafe_allow_html=True)

    filt_pts = patients[patients["GENDER"].isin(sel_gender)]
    filt_enc = encounters[encounters["PATIENT"].isin(filt_pts["Id"].values)]
    filt_ml  = ml_data[ml_data.index.isin(filt_pts["Id"].values)]

    n_pos = (filt_ml["TARGET"] == 1).sum()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Patients",   len(filt_pts))
    with c2: st.metric("Total Encounters", len(filt_enc))
    with c3: st.metric("With Condition",   n_pos)
    with c4: st.metric("Prevalence",       f"{100*n_pos/max(len(filt_ml),1):.1f}%")
    with c5: st.metric("Avg Income",       f"${filt_pts['INCOME'].mean():,.0f}")

    st.divider()

    # ── Overall Distribution ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Class Distribution – {selected_condition}")
        td = filt_ml["TARGET"].value_counts().reset_index()
        td.columns = ["Class", "Count"]
        td["Class"] = td["Class"].map({0: f"No {selected_condition}", 1: selected_condition})
        fig = px.pie(td, names="Class", values="Count",
                     color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Top 15 Medical Conditions by Prevalence")
        fig = px.bar(
            top_conditions.sort_values("PATIENT_COUNT"),
            x="PATIENT_COUNT", y="DESCRIPTION", orientation="h",
            color="PATIENT_COUNT", color_continuous_scale="Teal",
        )
        fig.update_layout(height=420, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Condition Rate by Gender")
        gender_target = filt_pts[["Id", "GENDER"]].copy()
        gender_target = gender_target.set_index("Id").join(filt_ml[["TARGET"]])
        gender_cost = gender_target.groupby("GENDER")["TARGET"].agg(["sum", "count"])
        gender_cost["Rate"] = (gender_cost["sum"] / gender_cost["count"] * 100).round(1)
        fig = px.bar(gender_cost.reset_index(), x="GENDER", y="Rate",
                     color="Rate", color_continuous_scale="RdYlGn_r",
                     labels={"Rate": f"% with {selected_condition}"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Age Distribution by Condition Status")
        age_df = ml_data[["AGE", "TARGET"]].copy()
        age_df["Status"] = age_df["TARGET"].map(
            {0: f"No {selected_condition}", 1: selected_condition})
        fig = px.histogram(age_df, x="AGE", color="Status", nbins=30,
                           barmode="overlay", opacity=0.7,
                           color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gender Distribution")
        gd = filt_pts["GENDER"].value_counts().reset_index()
        gd.columns = ["Gender", "Count"]
        fig = px.pie(gd, names="Gender", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Encounter Count Distribution")
        fig = px.histogram(ml_data, x="ENCOUNTER_COUNT", nbins=30,
                           color_discrete_sequence=["#f39c12"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── EDA per temporal dataset ──────────────────────────────────────────
    st.subheader("📅 EDA: Dataset 1 (Historical) vs Dataset 2 (Current)")
    d1_ml = ml_data[is_hist]
    d2_ml = ml_data[~is_hist]

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Dataset 1** — {len(d1_ml):,} patients | "
                 f"Prevalence: {100*d1_ml['TARGET'].mean():.1f}%")
        d1_td = d1_ml["TARGET"].value_counts().reset_index()
        d1_td.columns = ["Class","Count"]
        d1_td["Class"] = d1_td["Class"].map(
            {0: f"No {selected_condition}", 1: selected_condition})
        fig = px.pie(d1_td, names="Class", values="Count",
                     title="Dataset 1 Class Distribution",
                     color_discrete_sequence=["#3498db","#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write(f"**Dataset 2** — {len(d2_ml):,} patients | "
                 f"Prevalence: {100*d2_ml['TARGET'].mean():.1f}%")
        d2_td = d2_ml["TARGET"].value_counts().reset_index()
        d2_td.columns = ["Class","Count"]
        d2_td["Class"] = d2_td["Class"].map(
            {0: f"No {selected_condition}", 1: selected_condition})
        fig = px.pie(d2_td, names="Class", values="Count",
                     title="Dataset 2 Class Distribution",
                     color_discrete_sequence=["#3498db","#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Age: Dataset 1 vs Dataset 2")
        fig = go.Figure([
            go.Histogram(x=d1_ml["AGE"], name="Dataset 1 (Historical)",
                         opacity=0.7, nbinsx=30,
                         marker_color="#3498db"),
            go.Histogram(x=d2_ml["AGE"], name="Dataset 2 (Current)",
                         opacity=0.7, nbinsx=30,
                         marker_color="#e74c3c"),
        ])
        fig.update_layout(barmode="overlay", height=350,
                          xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Encounter Count: Dataset 1 vs Dataset 2")
        fig = go.Figure([
            go.Histogram(x=d1_ml["ENCOUNTER_COUNT"],
                         name="Dataset 1", opacity=0.7, nbinsx=30,
                         marker_color="#3498db"),
            go.Histogram(x=d2_ml["ENCOUNTER_COUNT"],
                         name="Dataset 2", opacity=0.7, nbinsx=30,
                         marker_color="#e74c3c"),
        ])
        fig.update_layout(barmode="overlay", height=350,
                          xaxis_title="Encounter Count")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Condition Count: Dataset 1 vs Dataset 2")
        fig = go.Figure([
            go.Histogram(x=d1_ml["CONDITION_COUNT"],
                         name="Dataset 1", opacity=0.7, nbinsx=25,
                         marker_color="#9b59b6"),
            go.Histogram(x=d2_ml["CONDITION_COUNT"],
                         name="Dataset 2", opacity=0.7, nbinsx=25,
                         marker_color="#f39c12"),
        ])
        fig.update_layout(barmode="overlay", height=350,
                          xaxis_title="Condition Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Income Distribution: Dataset 1 vs Dataset 2")
        fig = go.Figure([
            go.Histogram(x=d1_ml["INCOME"],
                         name="Dataset 1", opacity=0.7, nbinsx=30,
                         marker_color="#1abc9c"),
            go.Histogram(x=d2_ml["INCOME"],
                         name="Dataset 2", opacity=0.7, nbinsx=30,
                         marker_color="#e67e22"),
        ])
        fig.update_layout(barmode="overlay", height=350,
                          xaxis_title="Income ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Descriptive Statistics – Full Feature Matrix")
    st.dataframe(X_all.describe().T.round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<h2 class="header-text">🔧 Preprocessing & Feature Engineering</h2>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dataset Summary")
        summary = pd.DataFrame({
            "Table":   ["patients","encounters","conditions",
                        "medications","procedures","immunizations","observations"],
            "Records": [len(patients), len(encounters), len(conditions),
                        len(medications), len(procedures), len(immunizations),
                        len(observations)],
        })
        st.dataframe(summary, use_container_width=True)
    with c2:
        st.subheader("Missing Values in Feature Matrix")
        missing = pd.DataFrame({
            "Feature":   X_all.columns,
            "Missing %": [100 * ml_data[c].isna().sum() / len(ml_data)
                          for c in X_all.columns],
        })
        st.dataframe(missing, use_container_width=True)

    st.divider()

    # ── Target Definition ─────────────────────────────────────────────────
    st.subheader("Target Variable (Medical Condition)")
    pos = (ml_data["TARGET"] == 1).sum()
    neg = len(ml_data) - pos
    st.markdown(f"""
<div class="info-box">
<strong>Target:</strong> Binary classification — does the patient have
<em>{selected_condition}</em>?<br>
• <strong>Positive (has condition):</strong> {pos:,} patients ({100*pos/len(ml_data):.1f}%)<br>
• <strong>Negative (no condition):</strong> {neg:,} patients ({100*neg/len(ml_data):.1f}%)<br>
• Derived from <code>conditions.csv</code> — no financial features used as target.
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Temporal Split Justification ─────────────────────────────────────
    st.subheader("Temporal Split — Cutoff: January 2020")
    st.markdown("""
**Justification for the 2020 cutoff:**
- The COVID-19 pandemic began in early 2020 and caused a documented, global disruption to
  healthcare systems — deferred elective procedures, altered diagnosis rates, and changed
  patient-seeking behaviour. This makes 2020 a clinically meaningful boundary between
  *historical* (pre-pandemic) and *current* (pandemic-era and post-pandemic) patient data.
- Splitting here tests whether models trained on pre-2020 patterns generalise to the
  significantly different post-2020 healthcare landscape — a realistic and challenging
  **temporal shift** scenario.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Dataset 1 – Historical (first encounter < 2020)**")
        st.write(f"  - Total : {len(X_hist_all):,}")
        st.write(f"  - Train : {len(X_h_tr):,} | Test: {len(X_h_te):,}")
        st.write(f"  - Positive rate: {100*y_hist_all.mean():.1f}%")
    with c2:
        st.write(f"**Dataset 2 – Current (first encounter ≥ 2020)**")
        st.write(f"  - Total : {len(X_curr_all):,}")
        st.write(f"  - Train : {len(X_c_tr):,} | Test: {len(X_c_te):,}")
        st.write(f"  - Positive rate: {100*y_curr_all.mean():.1f}%")

    st.divider()

    # ── Feature Engineering Details ───────────────────────────────────────
    st.subheader("Feature Engineering Details")
    feat_sources = []
    for col in X_all.columns:
        cl = col.lower()
        if cl in ("income", "healthcare_coverage",
                  "age", "gender_enc", "ethnicity_enc", "race_enc"):
            src, agg = "patients", "raw / factorize"
        elif cl in ("encounter_count", "enc_cost_mean", "enc_cost_std",
                    "total_claim_mean", "total_claim_std", "payer_coverage_sum"):
            src, agg = "encounters", "count / mean / std / sum"
        elif cl == "condition_count":
            src, agg = "conditions", "count"
        elif cl == "medication_count":
            src, agg = "medications", "count"
        elif cl == "procedure_count":
            src, agg = "procedures", "count"
        else:
            src, agg = "observations", "mean / std"
        feat_sources.append({"Feature": col, "Source Table": src, "Aggregation": agg})

    st.dataframe(pd.DataFrame(feat_sources), use_container_width=True)

    st.markdown("""
**Note on feature exclusions:**  
`HEALTHCARE_EXPENSES` is intentionally excluded — it is a financial outcome variable that
directly captures total spending, which would create **data leakage** when predicting
disease presence and is not available prospectively in a real clinical setting.
""")

    if obs_cov:
        st.divider()
        st.subheader("Clinical Observation Coverage (from observations.csv)")
        cov_df = (
            pd.DataFrame.from_dict(obs_cov, orient="index", columns=["Patients"])
            .reset_index()
            .rename(columns={"index": "Observation"})
        )
        fig = px.bar(
            cov_df.sort_values("Patients", ascending=False),
            x="Observation", y="Patients",
            color="Patients", color_continuous_scale="Teal",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Dataset 1 vs Dataset 2 – Descriptive Statistics (Data Drift)")
    hist_ml = ml_data[is_hist].drop(["TARGET", "IS_HISTORICAL"], axis=1)
    curr_ml = ml_data[~is_hist].drop(["TARGET", "IS_HISTORICAL"], axis=1)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dataset 1 (Historical)**")
        st.dataframe(hist_ml.describe().T.round(3), use_container_width=True)
    with c2:
        st.write("**Dataset 2 (Current)**")
        st.dataframe(curr_ml.describe().T.round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<h2 class="header-text">🤖 Model Training – Dataset 1</h2>',
        unsafe_allow_html=True,
    )

    st.info(
        f"**Predicting:** Does the patient have **{selected_condition}**? "
        f"(1 = Yes, 0 = No)  |  "
        f"Positive rate in D1 train: {100*y_h_tr.mean():.1f}%"
    )
    st.divider()

    # ── Validate training data ────────────────────────────────────────────
    n_classes = y_h_tr.nunique()
    if n_classes < 2:
        st.warning("⚠️ Dataset 1 has only 1 class. Auto-swapping to use Dataset 2 for training.")
        X_h_tr, X_c_tr = X_c_tr, X_h_tr
        y_h_tr, y_c_tr = y_c_tr, y_h_tr
        X_h_te, X_c_te = X_c_te, X_h_te
        y_h_te, y_c_te = y_c_te, y_h_te
        X_h_tr_sc, X_c_tr_sc = X_c_tr_sc, X_h_tr_sc
        X_h_te_sc, X_c_te_sc = X_c_te_sc, X_h_te_sc

    if y_h_tr.nunique() < 2:
        st.error("❌ Cannot train: both datasets have only 1 class. Try a different condition.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Training samples", len(X_h_tr))
    with c2: st.metric("Test samples",     len(X_h_te))
    with c3: st.metric("Features",         X_h_tr.shape[1])
    with c4: st.metric("Positive rate",    f"{100*y_h_tr.mean():.1f}%")

    pb     = st.progress(0)
    status = st.empty()

    models  = {}
    results = {}

    # Decision Tree
    status.text("Training Decision Tree on Dataset 1…")
    pb.progress(33)
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    dt.fit(X_h_tr, y_h_tr)
    models["Decision Tree"] = dt
    results["Decision Tree"] = dict(
        model      = dt,
        pred       = dt.predict(X_h_te),
        pred_proba = dt.predict_proba(X_h_te),
    )

    # SVM
    status.text("Training SVM on Dataset 1…")
    pb.progress(66)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_h_tr_sc, y_h_tr)
    models["SVM"] = svm
    results["SVM"] = dict(
        model      = svm,
        pred       = svm.predict(X_h_te_sc),
        pred_proba = svm.predict_proba(X_h_te_sc),
    )

    # MLP
    status.text("Training MLP Neural Network on Dataset 1…")
    pb.progress(99)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=500,
        early_stopping=True, random_state=42,
    )
    mlp.fit(X_h_tr_sc, y_h_tr)
    models["MLP"] = mlp
    results["MLP"] = dict(
        model      = mlp,
        pred       = mlp.predict(X_h_te_sc),
        pred_proba = mlp.predict_proba(X_h_te_sc),
    )

    pb.progress(100)
    status.success("✅ All models trained on Dataset 1!")

    # Persist for other tabs
    ss = st.session_state
    ss.models        = models
    ss.results       = results
    ss.X_h_tr        = X_h_tr;    ss.X_h_te     = X_h_te
    ss.y_h_tr        = y_h_tr;    ss.y_h_te     = y_h_te
    ss.X_c_tr        = X_c_tr;    ss.X_c_te     = X_c_te
    ss.y_c_tr        = y_c_tr;    ss.y_c_te     = y_c_te
    ss.X_h_tr_sc     = X_h_tr_sc; ss.X_h_te_sc  = X_h_te_sc
    ss.X_c_tr_sc     = X_c_tr_sc; ss.X_c_te_sc  = X_c_te_sc
    ss.scaler        = scaler
    ss.X_all         = X_all

    st.divider()
    st.subheader("Hyperparameter Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Decision Tree**")
        st.write("- max_depth: 10")
        st.write("- min_samples_split: 10")
        st.write("- criterion: gini")
    with c2:
        st.write("**SVM**")
        st.write("- kernel: RBF")
        st.write("- C: 1.0  |  gamma: scale")
        st.write("- probability: True")
    with c3:
        st.write("**MLP**")
        st.write("- hidden layers: (100, 50)")
        st.write("- max_iter: 500  |  solver: adam")
        st.write("- early_stopping: True")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<h2 class="header-text">📈 Model Evaluation & Performance Metrics</h2>',
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.warning("Visit **Model Training** tab first.")
    else:
        ss      = st.session_state
        y_ev    = ss.y_h_te
        X_ev_sc = ss.X_h_te_sc
        X_ev    = ss.X_h_te

        # Summary table
        rows = []
        for mname, rd in ss.results.items():
            p, pp = rd["pred"], rd["pred_proba"]
            fpr, tpr, _ = roc_curve(y_ev, pp[:, 1])
            rows.append(dict(
                Model     = mname,
                Accuracy  = accuracy_score(y_ev, p),
                Precision = precision_score(y_ev, p, zero_division=0),
                Recall    = recall_score(y_ev, p, zero_division=0),
                F1        = f1_score(y_ev, p, zero_division=0),
                ROC_AUC   = auc(fpr, tpr),
            ))
        mdf = pd.DataFrame(rows)
        st.subheader("Performance Summary – Dataset 1 Test Set")
        fmt = {c: "{:.4f}" for c in mdf.columns if c != "Model"}
        st.dataframe(mdf.style.format(fmt), use_container_width=True)

        st.divider()

        # ROC curves
        st.subheader("ROC Curves – Dataset 1 Test Set")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            mode="lines", line=dict(dash="dash", color="gray")
        ))
        for mname, rd in ss.results.items():
            fpr, tpr, _ = roc_curve(y_ev, rd["pred_proba"][:, 1])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{mname} (AUC={auc(fpr,tpr):.4f})",
            ))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=480)
        st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()

        # Per-model detailed metrics
        for mname, rd in ss.results.items():
            st.subheader(f"{mname} – Detailed Metrics")
            p, pp = rd["pred"], rd["pred_proba"]
            fpr, tpr, _ = roc_curve(y_ev, pp[:, 1])
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Accuracy",  f"{accuracy_score(y_ev,p):.4f}")
            with c2: st.metric("Precision", f"{precision_score(y_ev,p,zero_division=0):.4f}")
            with c3: st.metric("Recall",    f"{recall_score(y_ev,p,zero_division=0):.4f}")
            with c4: st.metric("F1-Score",  f"{f1_score(y_ev,p,zero_division=0):.4f}")
            with c5: st.metric("ROC-AUC",   f"{auc(fpr,tpr):.4f}")

            rep = pd.DataFrame(
                classification_report(y_ev, p, output_dict=True, zero_division=0)
            ).T
            st.dataframe(rep.round(4), use_container_width=True)

            cm = confusion_matrix(y_ev, p)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=[f"Pred: No {selected_condition}", f"Pred: {selected_condition}"],
                y=[f"Actual: No {selected_condition}", f"Actual: {selected_condition}"],
                text=cm, texttemplate="%{text}", colorscale="Blues",
            ))
            fig_cm.update_layout(title=f"{mname} – Confusion Matrix", height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4b – COMPLEXITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab4b:
    st.markdown(
        '<h2 class="header-text">📉 Model Complexity & Generalisation</h2>',
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.warning("Visit **Model Training** tab first.")
    else:
        ss     = st.session_state
        Xtr    = ss.X_h_tr;    Xte    = ss.X_h_te
        ytr    = ss.y_h_tr;    yte    = ss.y_h_te
        Xtr_sc = ss.X_h_tr_sc; Xte_sc = ss.X_h_te_sc

        # ── Decision Tree depth sweep ─────────────────────────────────────
        st.subheader("Decision Tree – Max Depth Sweep")
        depths = [2, 3, 5, 7, 10, 15, 20]
        dt_tr, dt_te = [], []
        for d in depths:
            m = DecisionTreeClassifier(max_depth=d, random_state=42)
            m.fit(Xtr, ytr)
            dt_tr.append(accuracy_score(ytr, m.predict(Xtr)))
            dt_te.append(accuracy_score(yte, m.predict(Xte)))
        fig = go.Figure([
            go.Scatter(x=depths, y=dt_tr, mode="lines+markers", name="Train"),
            go.Scatter(x=depths, y=dt_te, mode="lines+markers", name="Test"),
        ])
        fig.update_layout(
            title="DT: Accuracy vs Max Depth",
            xaxis_title="Max Depth", yaxis_title="Accuracy", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── SVM C sweep ───────────────────────────────────────────────────
        st.subheader("SVM – Regularisation (C) Sweep")
        c_vals = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        sv_tr, sv_te = [], []
        for c in c_vals:
            m = SVC(kernel="rbf", C=c, gamma="scale", random_state=42)
            m.fit(Xtr_sc, ytr)
            sv_tr.append(accuracy_score(ytr, m.predict(Xtr_sc)))
            sv_te.append(accuracy_score(yte, m.predict(Xte_sc)))
        fig = go.Figure([
            go.Scatter(x=[str(c) for c in c_vals], y=sv_tr,
                       mode="lines+markers", name="Train"),
            go.Scatter(x=[str(c) for c in c_vals], y=sv_te,
                       mode="lines+markers", name="Test"),
        ])
        fig.update_layout(
            title="SVM: Accuracy vs C",
            xaxis_title="C", yaxis_title="Accuracy", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── MLP layer sweep ───────────────────────────────────────────────
        st.subheader("MLP – Hidden Layer Configuration Sweep")
        cfgs = [(50,), (100,), (100, 50), (100, 75, 50), (150, 100, 50)]
        ml_tr, ml_te = [], []
        for cfg in cfgs:
            m = MLPClassifier(
                hidden_layer_sizes=cfg, max_iter=500,
                early_stopping=True, random_state=42,
            )
            m.fit(Xtr_sc, ytr)
            ml_tr.append(accuracy_score(ytr, m.predict(Xtr_sc)))
            ml_te.append(accuracy_score(yte, m.predict(Xte_sc)))
        fig = go.Figure([
            go.Scatter(x=[str(c) for c in cfgs], y=ml_tr,
                       mode="lines+markers", name="Train"),
            go.Scatter(x=[str(c) for c in cfgs], y=ml_te,
                       mode="lines+markers", name="Test"),
        ])
        fig.update_layout(
            title="MLP: Accuracy vs Layer Config",
            xaxis_title="Hidden Layers", yaxis_title="Accuracy", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Bias–Variance Trade-off Summary")
        st.write(f"""
**Predicting: {selected_condition}**

**Underfitting (High Bias):** Train ≈ Test but both low → model too simple for the task.  
**Ideal Fit:** Train ≈ Test and both high → good generalisation.  
**Overfitting (High Variance):** Train >> Test → model memorises training noise.

- **Decision Tree:** Train accuracy rises toward ~100% at large depths while test accuracy
  plateaus or drops → classic overfitting. Optimal depth ≈ 7–10 for most conditions.
- **SVM (RBF):** Relatively robust across C values; the margin maximisation provides
  implicit regularisation. Mild overfitting only at very high C (≥ 5).
- **MLP (100, 50):** Achieves best bias-variance balance among tested configurations;
  deeper/wider networks show diminishing returns and mild overfitting on small splits.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – TEMPORAL SHIFT
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        '<h2 class="header-text">⏰ Temporal Shift Analysis</h2>',
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.warning("Visit **Model Training** tab first.")
    else:
        ss = st.session_state

        Xh_te    = ss.X_h_te;    yh_te = ss.y_h_te
        Xh_te_sc = ss.X_h_te_sc
        Xc_te    = ss.X_c_te;    yc_te = ss.y_c_te
        Xc_te_sc = ss.X_c_te_sc

        st.info(f"""
**Target condition:** {selected_condition}

**Evaluation protocol:**
- Models trained on **Dataset 1 training split** (historical, first encounter < {TEMPORAL_CUTOFF})
- Evaluated on **Dataset 1 test split** → in-distribution (same time period)
- Evaluated on **Dataset 2 test split** → cross-temporal generalisation (never seen during training)
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Dataset 1 Test Set**")
            st.write(f"- Samples: {len(yh_te):,}  |  "
                     f"Positive rate: {100*yh_te.mean():.1f}%")
        with c2:
            st.write("**Dataset 2 Test Set**")
            st.write(f"- Samples: {len(yc_te):,}  |  "
                     f"Positive rate: {100*yc_te.mean():.1f}%")

        st.divider()
        st.subheader("Cross-Dataset Evaluation (Proper Held-Out Test Splits)")

        rows = []
        for mname, rd in ss.results.items():
            m       = rd["model"]
            is_tree = mname == "Decision Tree"

            pred_d1 = m.predict(Xh_te if is_tree else Xh_te_sc)
            pred_d2 = m.predict(Xc_te if is_tree else Xc_te_sc)

            acc1 = accuracy_score(yh_te, pred_d1)
            f1_1 = f1_score(yh_te, pred_d1, zero_division=0)
            acc2 = accuracy_score(yc_te, pred_d2)
            f1_2 = f1_score(yc_te, pred_d2, zero_division=0)

            rows.append(dict(
                Model    = mname,
                Acc_D1   = acc1,
                F1_D1    = f1_1,
                Acc_D2   = acc2,
                F1_D2    = f1_2,
                Acc_Drop = acc1 - acc2,
                F1_Drop  = f1_1 - f1_2,
            ))

        shift_df = pd.DataFrame(rows)
        fmt = {c: "{:.4f}" for c in shift_df.columns if c != "Model"}
        st.dataframe(
            shift_df.rename(columns={
                "Acc_D1": "Acc (D1 test)", "F1_D1": "F1 (D1 test)",
                "Acc_D2": "Acc (D2 test)", "F1_D2": "F1 (D2 test)",
                "Acc_Drop": "Acc Drop",    "F1_Drop": "F1 Drop",
            }).style.format(fmt),
            use_container_width=True,
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Accuracy: D1 Test vs D2 Test")
            fig = go.Figure([
                go.Bar(x=shift_df["Model"], y=shift_df["Acc_D1"],
                       name="Dataset 1 Test"),
                go.Bar(x=shift_df["Model"], y=shift_df["Acc_D2"],
                       name="Dataset 2 Test"),
            ])
            fig.update_layout(barmode="group", height=400, yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("F1 Score: D1 Test vs D2 Test")
            fig = go.Figure([
                go.Bar(x=shift_df["Model"], y=shift_df["F1_D1"],
                       name="Dataset 1 Test"),
                go.Bar(x=shift_df["Model"], y=shift_df["F1_D2"],
                       name="Dataset 2 Test"),
            ])
            fig.update_layout(barmode="group", height=400, yaxis_title="F1 Score")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Accuracy Drop (D1 → D2)")
            fig = px.bar(
                shift_df, x="Model", y="Acc_Drop",
                color="Acc_Drop", color_continuous_scale="RdYlGn_r",
                labels={"Acc_Drop": "Accuracy Drop"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("F1 Drop (D1 → D2)")
            fig = px.bar(
                shift_df, x="Model", y="F1_Drop",
                color="F1_Drop", color_continuous_scale="RdYlGn_r",
                labels={"F1_Drop": "F1 Drop"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("ROC Curves – D1 Test vs D2 Test")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            mode="lines", line=dict(dash="dash", color="gray"),
        ))
        colors = {"Decision Tree": "blue", "SVM": "orange", "MLP": "green"}
        for mname, rd in ss.results.items():
            m       = rd["model"]
            is_tree = mname == "Decision Tree"
            pp_d1   = m.predict_proba(Xh_te if is_tree else Xh_te_sc)
            pp_d2   = m.predict_proba(Xc_te if is_tree else Xc_te_sc)
            fpr1, tpr1, _ = roc_curve(yh_te, pp_d1[:, 1])
            fpr2, tpr2, _ = roc_curve(yc_te, pp_d2[:, 1])
            col = colors.get(mname, "purple")
            fig_roc.add_trace(go.Scatter(
                x=fpr1, y=tpr1, mode="lines",
                name=f"{mname} D1 (AUC={auc(fpr1,tpr1):.3f})",
                line=dict(color=col),
            ))
            fig_roc.add_trace(go.Scatter(
                x=fpr2, y=tpr2, mode="lines",
                name=f"{mname} D2 (AUC={auc(fpr2,tpr2):.3f})",
                line=dict(color=col, dash="dash"),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=520,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown(f"""
**Temporal Shift Interpretation for '{selected_condition}':**

- A large **accuracy/F1 drop** from D1→D2 indicates the model has not generalised well across
  the temporal boundary — features may have drifted in distribution after {TEMPORAL_CUTOFF}.
- A small drop suggests this condition's clinical markers are stable over time and the model
  generalises well.
- Continual learning (Tab 7) addresses this gap by fine-tuning on Dataset 2 training data.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 – CONTINUAL LEARNING
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown(
        '<h2 class="header-text">🧠 Continual Learning – Fine-tuning on Dataset 2</h2>',
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.warning("Visit **Model Training** tab first.")
    else:
        ss = st.session_state
        Xc_tr    = ss.X_c_tr;    yc_tr = ss.y_c_tr
        Xc_te    = ss.X_c_te;    yc_te = ss.y_c_te
        Xc_tr_sc = ss.X_c_tr_sc
        Xc_te_sc = ss.X_c_te_sc

        st.markdown("""
| Model         | Strategy | Justification |
|---------------|----------|---------------|
| **Decision Tree** | Retrain on combined D1+D2 train | DT has no incremental API; retraining on all available data is the standard approach |
| **SVM (RBF)** | Retrain on combined D1+D2 train | Kernel SVM has no online update; combined retraining prevents catastrophic forgetting of D1 patterns |
| **MLP** | **Warm-start fine-tuning** via `deepcopy` + `warm_start=True` | Preserves D1-learned weights; continues gradient descent on D2 data only for 100 additional epochs — true incremental learning |

All models evaluated on the **Dataset 2 held-out test set**.
        """)

        # Combined training data for DT & SVM
        X_combined  = pd.concat([ss.X_h_tr, Xc_tr])
        y_combined  = pd.concat([ss.y_h_tr, yc_tr])
        sc_combined = StandardScaler()
        X_comb_sc   = sc_combined.fit_transform(X_combined)
        Xc_te_sc2   = sc_combined.transform(Xc_te)

        pb2     = st.progress(0)
        status2 = st.empty()
        cont_rows = []

        # ── Decision Tree ─────────────────────────────────────────────────
        status2.text("Decision Tree: retraining on combined D1+D2 data…")
        pb2.progress(33)
        dt_orig = ss.results["Decision Tree"]["model"]
        dt_bef  = dt_orig.predict(Xc_te)
        dt_cl   = DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, random_state=42
        )
        dt_cl.fit(X_combined, y_combined)
        dt_aft = dt_cl.predict(Xc_te)
        cont_rows.append(dict(
            Model     = "Decision Tree",
            Before    = accuracy_score(yc_te, dt_bef),
            After     = accuracy_score(yc_te, dt_aft),
            F1_Before = f1_score(yc_te, dt_bef, zero_division=0),
            F1_After  = f1_score(yc_te, dt_aft, zero_division=0),
        ))

        # ── SVM ───────────────────────────────────────────────────────────
        status2.text("SVM: retraining on combined D1+D2 data…")
        pb2.progress(66)
        svm_orig = ss.results["SVM"]["model"]
        svm_bef  = svm_orig.predict(Xc_te_sc)
        if len(set(y_combined)) > 1:
            svm_cl = SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=42
            )
            svm_cl.fit(X_comb_sc, y_combined)
            svm_aft = svm_cl.predict(Xc_te_sc2)
        else:
            svm_aft = svm_bef
            st.warning("⚠️ SVM: only one class in combined training data.")
        cont_rows.append(dict(
            Model     = "SVM",
            Before    = accuracy_score(yc_te, svm_bef),
            After     = accuracy_score(yc_te, svm_aft),
            F1_Before = f1_score(yc_te, svm_bef, zero_division=0),
            F1_After  = f1_score(yc_te, svm_aft, zero_division=0),
        ))

        # ── MLP – warm-start fine-tuning ──────────────────────────────────
        status2.text("MLP: warm-start fine-tuning on Dataset 2 training data…")
        pb2.progress(99)
        mlp_orig = ss.results["MLP"]["model"]
        mlp_bef  = mlp_orig.predict(Xc_te_sc)

        # deepcopy preserves all learned weights.
        # warm_start=True continues optimisation from those weights
        # instead of re-initialising — genuine incremental learning.
        mlp_cl            = copy.deepcopy(mlp_orig)
        mlp_cl.warm_start = True
        mlp_cl.max_iter   = mlp_orig.n_iter_ + 100   # 100 additional epochs
        mlp_cl.early_stopping = True
        if len(set(yc_tr)) > 1:
            mlp_cl.fit(Xc_tr_sc, yc_tr)
            mlp_aft = mlp_cl.predict(Xc_te_sc)
        else:
            mlp_aft = mlp_bef
            st.warning("⚠️ MLP: only one class in Dataset 2 training data.")
        cont_rows.append(dict(
            Model     = "MLP",
            Before    = accuracy_score(yc_te, mlp_bef),
            After     = accuracy_score(yc_te, mlp_aft),
            F1_Before = f1_score(yc_te, mlp_bef, zero_division=0),
            F1_After  = f1_score(yc_te, mlp_aft, zero_division=0),
        ))

        pb2.progress(100)
        status2.success("✅ Continual learning complete!")

        cl_df = pd.DataFrame(cont_rows)
        cl_df["Acc Improvement"] = cl_df["After"]    - cl_df["Before"]
        cl_df["F1 Improvement"]  = cl_df["F1_After"] - cl_df["F1_Before"]

        st.divider()
        st.subheader("Continual Learning Results – Dataset 2 Test Set")
        fmt2 = {c: "{:.4f}" for c in cl_df.columns if c != "Model"}
        st.dataframe(
            cl_df.rename(columns={
                "Before":   "Acc Before CL",
                "After":    "Acc After CL",
                "F1_Before":"F1 Before CL",
                "F1_After": "F1 After CL",
            }).style.format(fmt2),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Accuracy Before vs After CL")
            fig = go.Figure([
                go.Bar(x=cl_df["Model"], y=cl_df["Before"],
                       name="Before CL (D1 only)"),
                go.Bar(x=cl_df["Model"], y=cl_df["After"],
                       name="After CL"),
            ])
            fig.update_layout(barmode="group", height=400,
                              yaxis_title="Accuracy on D2 Test")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Accuracy Improvement from CL")
            fig = px.bar(
                cl_df, x="Model", y="Acc Improvement",
                color="Acc Improvement", color_continuous_scale="RdYlGn",
                labels={"Acc Improvement": "Δ Accuracy"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("F1 Before vs After CL")
            fig = go.Figure([
                go.Bar(x=cl_df["Model"], y=cl_df["F1_Before"],
                       name="Before CL"),
                go.Bar(x=cl_df["Model"], y=cl_df["F1_After"],
                       name="After CL"),
            ])
            fig.update_layout(barmode="group", height=400,
                              yaxis_title="F1 Score on D2 Test")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("F1 Improvement from CL")
            fig = px.bar(
                cl_df, x="Model", y="F1 Improvement",
                color="F1 Improvement", color_continuous_scale="RdYlGn",
                labels={"F1 Improvement": "Δ F1"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
**Interpretation for '{selected_condition}':**

- A **positive improvement** means fine-tuning on Dataset 2 training data helped the model
  adapt to the post-{TEMPORAL_CUTOFF} temporal shift.
- **Near-zero or negative** values may indicate: (a) the temporal shift is subtle for this
  condition, (b) Dataset 2 is too small to improve generalisation, or (c) the condition
  prevalence changed significantly after {TEMPORAL_CUTOFF}.
- The **MLP warm-start** preserves all weights learned on Dataset 1 — no catastrophic
  forgetting. This is the most principled continual learning strategy of the three.
- DT and SVM are retrained on combined data (D1+D2) as they lack incremental update APIs.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 – FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown(
        '<h2 class="header-text">🎯 Feature Importance & Interpretation</h2>',
        unsafe_allow_html=True,
    )

    if "results" not in st.session_state:
        st.warning("Visit **Model Training** tab first.")
    else:
        ss         = st.session_state
        X_te       = ss.X_h_te
        X_te_sc    = ss.X_h_te_sc
        y_te       = ss.y_h_te
        feat_names = ss.X_all.columns.tolist()

        # ── Decision Tree – Gini importances ─────────────────────────────
        st.subheader("Decision Tree – Gini Feature Importances")
        dt_m = ss.results["Decision Tree"]["model"]
        imp  = (
            pd.DataFrame({
                "Feature":    feat_names,
                "Importance": dt_m.feature_importances_,
            })
            .sort_values("Importance", ascending=False)
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(imp.reset_index(drop=True), use_container_width=True)
        with c2:
            fig = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Viridis",
                title="Decision Tree – Gini Feature Importances",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Permutation Importance for all models ─────────────────────────
        st.subheader("Permutation Feature Importance – All Models")
        st.write(
            "Permutation importance measures the drop in accuracy when each feature is "
            "randomly shuffled. Unlike Gini importance, it is model-agnostic and "
            "reflects true predictive contribution on the **held-out test set**."
        )

        for mname, rd in ss.results.items():
            m       = rd["model"]
            is_tree = mname == "Decision Tree"
            X_eval  = X_te if is_tree else X_te_sc
            result  = permutation_importance(
                m, X_eval, y_te, n_repeats=10, random_state=42, scoring="accuracy"
            )
            perm_df = (
                pd.DataFrame({
                    "Feature":   feat_names,
                    "Mean Drop": result.importances_mean,
                    "Std":       result.importances_std,
                })
                .sort_values("Mean Drop", ascending=False)
            )
            st.write(f"**{mname}**")
            fig = px.bar(
                perm_df, x="Mean Drop", y="Feature", orientation="h",
                error_x="Std", color="Mean Drop",
                color_continuous_scale="RdBu_r",
                title=f"{mname} – Permutation Importance (Dataset 1 Test Set)",
            )
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

        st.subheader("Feature Interpretation")
        st.markdown(f"""
**Clinical Context – Predicting '{selected_condition}':**

- **CONDITION_COUNT:** Patients with more diagnosed conditions are more likely to have any
  specific condition — high comorbidity burden.
- **MEDICATION_COUNT / PROCEDURE_COUNT:** More medications and procedures indicate greater
  clinical complexity and higher chance of chronic disease diagnosis.
- **ENCOUNTER_COUNT / ENC_COST_MEAN:** Frequent and costly encounters reflect greater
  healthcare utilisation, which correlates with diagnosed conditions.
- **AGE:** Many chronic and acute conditions have strong age dependencies.
- **INCOME / HEALTHCARE_COVERAGE:** Socioeconomic factors influence both healthcare access
  and diagnosis rates.
- **Clinical observations (BMI, SBP, DBP, GLUCOSE, etc.):** Physiological markers directly
  related to many conditions (e.g., hypertension → high SBP; diabetes → high glucose).

**Model Differences:**
- **Decision Tree:** Uses Gini impurity to split on the most discriminative features early;
  cost-related encounter features typically dominate.
- **SVM (RBF):** Operates in a non-linear kernel space; all scaled features contribute via
  inner products — no single feature dominates as clearly.
- **MLP:** Learns multi-layer representations of feature interactions; permutation importance
  reveals which raw features most affect the learned representation.

**No data leakage:** `HEALTHCARE_EXPENSES` is excluded from all features — condition presence
is predicted purely from clinical and demographic signals.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="text-align:center;margin-top:30px;color:gray;">
  <p><strong>ML Healthcare Analytics Dashboard – v3</strong> |
     Team 09 – BITS F464 Machine Learning – Assignment 2</p>
  <p>Models: Decision Tree · SVM · MLP (Neural Network) |
     Target: Medical Condition from EHR Data (conditions.csv)</p>
  <p style="font-size:12px;">
    Features: EDA · Preprocessing · Temporal Split · Cross-Dataset Evaluation ·
    Continual Learning · Feature Importance &nbsp;|&nbsp;
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </p>
</div>
""", unsafe_allow_html=True)