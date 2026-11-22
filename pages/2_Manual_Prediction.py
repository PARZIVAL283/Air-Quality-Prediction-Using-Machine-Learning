
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import date, time

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Manual CO Prediction",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Manual CO Concentration Prediction")

st.markdown("""
Enter sensor and environmental measurements below
to estimate carbon monoxide (CO) concentration.
""")

st.info(
    "This model was trained using historical UCI Air Quality "
    "measurements from 2004–2005. Predictions are model estimates, "
    "not current air-quality measurements."
)

# --------------------------------------------------
# LOAD MODEL
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
    st.error("Model metadata not found.")
    st.stop()

model = load_model()
metadata = load_metadata()

FEATURES = metadata["features"]

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("1. Sensor and Environmental Measurements")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pt08_s1 = st.number_input(
            "PT08.S1(CO)",
            value=1360.0,
            help="CO sensor response"
        )

        pt08_s2 = st.number_input(
            "PT08.S2(NMHC)",
            value=1046.0,
            help="NMHC sensor response"
        )

        pt08_s3 = st.number_input(
            "PT08.S3(NOx)",
            value=1056.0,
            help="NOx sensor response"
        )

        pt08_s4 = st.number_input(
            "PT08.S4(NO2)",
            value=1692.0,
            help="NO2 sensor response"
        )

    with col2:
        pt08_s5 = st.number_input(
            "PT08.S5(O3)",
            value=1268.0,
            help="O3 sensor response"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=13.6,
            format="%.2f"
        )

        humidity = st.number_input(
            "Relative Humidity (%)",
            value=48.9,
            format="%.2f"
        )

        absolute_humidity = st.number_input(
            "Absolute Humidity (AH)",
            value=0.7578,
            format="%.4f"
        )

    st.divider()

    st.subheader("2. Date and Time")

    col3, col4 = st.columns(2)

    with col3:
        selected_date = st.date_input(
            "Select Date",
            value=date(2004, 3, 10)
        )

    with col4:
        selected_time = st.time_input(
            "Select Time",
            value=time(18, 0)
        )

    submitted = st.form_submit_button(
        "Predict CO Concentration",
        use_container_width=True,
        type="primary"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    # Extract date/time components
    hour = selected_time.hour
    month = selected_date.month
    weekday = selected_date.weekday()

    # Generate cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    # Create input dictionary
    input_data = {
        "PT08.S1(CO)": pt08_s1,
        "PT08.S2(NMHC)": pt08_s2,
        "PT08.S3(NOx)": pt08_s3,
        "PT08.S4(NO2)": pt08_s4,
        "PT08.S5(O3)": pt08_s5,
        "T": temperature,
        "RH": humidity,
        "AH": absolute_humidity,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos
    }

    # Arrange features in exact training order
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Generate prediction
    try:
        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("3. Prediction Result")

        st.success("Prediction generated successfully!")

        st.metric(
            label="Predicted CO Concentration",
            value=f"{prediction:.4f} mg/m³"
        )

        st.caption(
            f"Selected date: {selected_date} | "
            f"Selected time: {selected_time}"
        )

        # Show submitted input
        with st.expander("View submitted input features"):
            st.dataframe(
                input_df.T.rename(
                    columns={0: "Input Value"}
                ),
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
