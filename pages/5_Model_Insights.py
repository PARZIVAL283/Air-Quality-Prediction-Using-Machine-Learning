
import streamlit as st
import joblib
import json
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide"
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


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
    st.stop()

if not os.path.exists(METADATA_PATH):
    st.error("Metadata file not found.")
    st.stop()

model = load_model()
metadata = load_metadata()

st.title("📊 Model Insights & Evaluation")

st.write(
    "Explore the performance, feature importance, "
    "and validation results of the trained air-quality model."
)

# -----------------------------
# Model Performance
# -----------------------------

st.header("1. Model Performance")

metrics = metadata["test_metrics"]

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:.4f} mg/m³")
col2.metric("RMSE", f"{metrics['RMSE']:.4f} mg/m³")
col3.metric("R² Score", f"{metrics['R2']:.4f}")

st.caption(
    "These metrics represent the final chronological test-set evaluation."
)

with st.expander("Understand the metrics"):

    st.markdown("""
    - **MAE:** Average absolute difference between actual and predicted CO.
    - **RMSE:** Penalizes larger prediction errors more heavily.
    - **R²:** Measures how much variation in the target is explained by the model.
    """)

# -----------------------------
# Model Information
# -----------------------------

st.header("2. Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write("**Model:**", metadata["model_name"])
    st.write("**Target:**", metadata["target"])
    st.write("**Target Unit:**", metadata["target_unit"])

with info_col2:
    st.write("**Number of Features:**", metadata["number_of_features"])
    st.write("**Dataset:** UCI Air Quality Dataset")
    st.write("**Task:** Regression")

# -----------------------------
# Feature Importance
# -----------------------------

st.header("3. Feature Importance")

feature_importance = None

for step in model.named_steps.values():
    if hasattr(step, "feature_importances_"):
        feature_importance = step.feature_importances_
        break

if feature_importance is not None:

    importance_df = pd.DataFrame({
        "Feature": metadata["features"],
        "Importance": feature_importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.caption(
        "Feature importance indicates the model's relative reliance "
        "on each feature. It does not establish causation."
    )

else:
    st.warning("Feature importance is unavailable for this model.")

# -----------------------------
# Cross Validation
# -----------------------------

st.header("4. Cross-Validation Results")

cv = metadata["cross_validation"]

cv_col1, cv_col2, cv_col3 = st.columns(3)

cv_col1.metric("Mean CV MAE", f"{cv['mean_MAE']:.4f}")
cv_col2.metric("Mean CV RMSE", f"{cv['mean_RMSE']:.4f}")
cv_col3.metric("Mean CV R²", f"{cv['mean_R2']:.4f}")

st.write(
    "The model was evaluated using five chronological "
    "TimeSeriesSplit folds on the training data."
)

# -----------------------------
# Model Configuration
# -----------------------------

st.header("5. Model Configuration")

st.write("**Best Hyperparameters:**")

best_parameters = {
    "Number of Trees": 300,
    "Maximum Depth": 20,
    "Minimum Samples Split": 5,
    "Minimum Samples Leaf": 2,
    "Maximum Features": 0.7
}

parameter_df = pd.DataFrame(
    list(best_parameters.items()),
    columns=["Parameter", "Value"]
)

st.table(parameter_df)

# -----------------------------
# Feature Engineering
# -----------------------------

st.header("6. Feature Engineering")

st.markdown("""
The model uses 14 input features:

**Sensor and Environmental Features**
- PT08.S1(CO)
- PT08.S2(NMHC)
- PT08.S3(NOx)
- PT08.S4(NO2)
- PT08.S5(O3)
- Temperature (T)
- Relative Humidity (RH)
- Absolute Humidity (AH)

**Time-Based Features**
- Hour sine and cosine
- Month sine and cosine
- Weekday sine and cosine

Cyclical encoding helps represent repeating time patterns
without treating the beginning and end of a cycle as far apart.
""")

# -----------------------------
# Conclusion
# -----------------------------

st.header("7. Model Evaluation Summary")

st.success(
    f"The tuned Random Forest achieved a test R² of "
    f"{metrics['R2']:.4f} and an RMSE of "
    f"{metrics['RMSE']:.4f} mg/m³."
)

st.write("""
The results indicate that the tuned Random Forest model
performs better than the baseline and Linear Regression
models in the evaluated chronological test period.

However, performance on historical data does not guarantee
the same accuracy under new environmental conditions.
""")
