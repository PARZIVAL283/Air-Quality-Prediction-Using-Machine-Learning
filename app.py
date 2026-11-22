
import streamlit as st
import joblib
import json
import os

st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/best_model_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


# Check required files
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check the models folder.")
    st.stop()

if not os.path.exists(METADATA_PATH):
    st.error("Metadata file not found.")
    st.stop()

model = load_model()
metadata = load_metadata()

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 750;
        color: #168B75;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #607080;
        margin-bottom: 25px;
    }

    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #E1E8ED;
        min-height: 125px;
    }

    .info-card h3 {
        font-size: 15px;
        color: #607080;
        margin: 0;
    }

    .info-card p {
        font-size: 25px;
        font-weight: bold;
        color: #168B75;
        margin: 10px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🌍 Air Quality Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Air Quality Analysis and CO Prediction</div>',
    unsafe_allow_html=True
)

st.divider()

# Introduction
st.header("Welcome")

st.write("""
This application uses machine learning to estimate carbon monoxide
(CO) concentration from environmental and air-quality sensor readings.

It combines historical-data-based machine learning with a separate
live air-quality monitoring feature.
""")

# Metrics
metrics = metadata["test_metrics"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="info-card">
        <h3>Model Test MAE</h3>
        <p>{metrics['MAE']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="info-card">
        <h3>Model Test RMSE</h3>
        <p>{metrics['RMSE']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="info-card">
        <h3>Model Test R²</h3>
        <p>{metrics['R2']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Application modules
st.header("Application Modules")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dashboard")
    st.write("Explore model performance and project information.")

    st.subheader("🌡️ Manual Prediction")
    st.write("Enter sensor readings to estimate CO concentration.")

    st.subheader("📂 Batch Prediction")
    st.write("Upload CSV data and generate multiple predictions.")

with col2:
    st.subheader("🌐 Live Air Quality")
    st.write("Explore current air-quality information from Open-Meteo.")

    st.subheader("📈 Model Insights")
    st.write("Understand feature importance and model evaluation.")

st.divider()

# Research information
st.header("Research Information")

info1, info2, info3 = st.columns(3)

info1.write("**Dataset**")
info1.write("UCI Air Quality Dataset")

info2.write("**Algorithm**")
info2.write("Tuned Random Forest Regressor")

info3.write("**Prediction Target**")
info3.write("CO(GT) — mg/m³")

st.warning("""
Research limitation: The machine learning model was trained on
historical UCI measurements from 2004–2005. Its predictions are
estimates based on supplied inputs, not verified current air-quality
measurements or future forecasts.
""")

st.caption("Developed as an academic Data Science project By the Team \"**Double Trouble**\".")
