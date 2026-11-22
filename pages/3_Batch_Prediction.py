
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

# -----------------------------
# Load model and metadata
# -----------------------------

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
    st.error("Model file not found. Check your models folder.")
    st.stop()

if not os.path.exists(METADATA_PATH):
    st.error("Model metadata not found.")
    st.stop()

model = load_model()
metadata = load_metadata()
FEATURES = metadata["features"]

# -----------------------------
# Page heading
# -----------------------------

st.title("📂 Batch CO Prediction")

st.write(
    "Upload a CSV file containing air-quality sensor and "
    "environmental readings to predict CO concentrations "
    "for multiple records."
)

st.info(
    "Your CSV must contain all 14 model input features. "
    "The model will generate one CO prediction per valid row."
)

# -----------------------------
# CSV template
# -----------------------------

st.subheader("1. Download CSV Template")

template_data = {
    "PT08.S1(CO)": [1360.0],
    "PT08.S2(NMHC)": [1046.0],
    "PT08.S3(NOx)": [1056.0],
    "PT08.S4(NO2)": [1692.0],
    "PT08.S5(O3)": [1268.0],
    "T": [13.6],
    "RH": [48.9],
    "AH": [0.7578],
    "hour_sin": [np.sin(2 * np.pi * 18 / 24)],
    "hour_cos": [np.cos(2 * np.pi * 18 / 24)],
    "month_sin": [np.sin(2 * np.pi * 3 / 12)],
    "month_cos": [np.cos(2 * np.pi * 3 / 12)],
    "weekday_sin": [np.sin(2 * np.pi * 2 / 7)],
    "weekday_cos": [np.cos(2 * np.pi * 2 / 7)]
}

template_df = pd.DataFrame(template_data)[FEATURES]

st.download_button(
    label="⬇️ Download CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="air_quality_input_template.csv",
    mime="text/csv"
)

st.caption(
    "The template contains one example row. Replace or add "
    "rows with your own readings while keeping the column names."
)

# -----------------------------
# Upload CSV
# -----------------------------

st.subheader("2. Upload Your CSV File")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        input_df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Check required columns
        missing_columns = [
            col for col in FEATURES if col not in input_df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Keep only model features and preserve their order
        prediction_df = input_df[FEATURES].copy()

        # Convert values to numeric
        for col in FEATURES:
            prediction_df[col] = pd.to_numeric(
                prediction_df[col],
                errors="coerce"
            )

        # Check missing or invalid values
        if prediction_df.isnull().any().any():
            st.error(
                "Your file contains missing or non-numeric "
                "values in the required feature columns. "
                "Please correct them and upload again."
            )
            st.stop()

        if not np.isfinite(prediction_df.to_numpy()).all():
            st.error(
                "Your file contains infinite values. "
                "Please correct them and upload again."
            )
            st.stop()

        if len(prediction_df) == 0:
            st.error("The uploaded CSV contains no data rows.")
            st.stop()

        # -----------------------------
        # Generate predictions
        # -----------------------------

        st.subheader("3. Prediction Results")

        predictions = model.predict(prediction_df)

        results_df = input_df.copy()
        results_df["Predicted_CO(GT)"] = predictions

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(results_df))

        with col2:
            st.metric(
                "Average Predicted CO",
                f"{np.mean(predictions):.4f} mg/m³"
            )

        with col3:
            st.metric(
                "Maximum Predicted CO",
                f"{np.max(predictions):.4f} mg/m³"
            )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # -----------------------------
        # Download predictions
        # -----------------------------

        csv_output = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_output,
            file_name="air_quality_predictions.csv",
            mime="text/csv"
        )

        st.success(
            "Predictions generated successfully!"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
