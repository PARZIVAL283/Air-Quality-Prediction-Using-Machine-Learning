
import streamlit as st
import joblib
import json
import os
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Air Quality Dashboard")

st.caption(
    "Overview of the trained machine learning model "
    "and its predictive performance."
)

# --------------------------------------------------
# LOAD MODEL AND METADATA
# --------------------------------------------------

MODEL_PATH = "models/best_model_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

if not os.path.exists(METADATA_PATH):
    st.error("Metadata file not found.")
    st.stop()

model = load_model()
metadata = load_metadata()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

metrics = metadata["test_metrics"]

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Mean Absolute Error",
    f"{metrics['MAE']:.4f} mg/m³"
)

col2.metric(
    "Root Mean Squared Error",
    f"{metrics['RMSE']:.4f} mg/m³"
)

col3.metric(
    "R² Score",
    f"{metrics['R2']:.4f}"
)

st.divider()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.info("**Algorithm:** Tuned Random Forest Regressor")
    st.info("**Prediction Target:** CO(GT)")
    st.info("**Problem Type:** Regression")

with col2:
    st.info("**Dataset:** UCI Air Quality Dataset")
    st.info(
        f"**Input Features:** "
        f"{metadata['number_of_features']}"
    )
    st.info("**Training Approach:** Chronological Split")

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("Feature Importance")

# Find the Random Forest estimator inside the pipeline
rf_model = None

for step_name, estimator in model.named_steps.items():
    if hasattr(estimator, "feature_importances_"):
        rf_model = estimator
        break

if rf_model is not None:

    feature_names = metadata["features"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        labels={
            "Importance": "Importance Score",
            "Feature": "Input Feature"
        }
    )

    fig.update_layout(
        height=550,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(
        "Feature importance could not be extracted "
        "from the saved model."
    )

st.divider()

# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

st.subheader("Model Stability")

cv = metadata["cross_validation"]

col1, col2, col3 = st.columns(3)

col1.metric(
    "CV Mean MAE",
    f"{cv['mean_MAE']:.4f}"
)

col2.metric(
    "CV Mean RMSE",
    f"{cv['mean_RMSE']:.4f}"
)

col3.metric(
    "CV Mean R²",
    f"{cv['mean_R2']:.4f}"
)

st.caption(
    "Cross-validation results are based on "
    "five chronological TimeSeriesSplit folds."
)

st.success("")
