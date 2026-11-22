# 🌍 Air Quality Prediction Using Machine Learning

A machine learning-based air quality prediction system that estimates Carbon Monoxide (CO) concentration using environmental and sensor measurements. The project combines a tuned Random Forest Regression model with an interactive Streamlit application for manual prediction, batch prediction, live air-quality monitoring, and model evaluation.

## 📌 Project Overview

Air pollution is a major environmental concern that affects human health and environmental sustainability. Monitoring air quality and estimating pollutant concentrations can help people better understand environmental conditions.

This project develops a machine learning pipeline to predict Carbon Monoxide (CO) concentration using the UCI Air Quality dataset. A web-based application built with Streamlit allows users to interact with the trained model and explore air-quality information.

## 🎯 Objectives

- Develop a machine learning model to predict CO concentration.
- Perform data cleaning, preprocessing, and feature engineering.
- Compare multiple regression algorithms.
- Optimize model performance using hyperparameter tuning.
- Develop an interactive web application for air-quality prediction.
- Provide live air-quality information through an external API.

## 📂 Dataset

**Dataset:** UCI Air Quality Dataset

**Source:** [UCI Machine Learning Repository – Air Quality](https://archive.ics.uci.edu/dataset/387/air+quality)

Dataset characteristics:

- 9,357 original records.
- 15 original columns.
- Measurements collected between March 2004 and April 2005.
- Contains air-quality sensor measurements and environmental variables.
- Target variable: `CO(GT)`.

After preprocessing and feature engineering, the final modeling dataset contains 7,344 records and 14 input features.

## ⚙️ Methodology

### 1. Data Preprocessing

- Converted special missing-value markers (`-200`) into missing values.
- Removed the highly incomplete `NMHC(GT)` variable.
- Removed records with missing target values.
- Used median imputation for missing sensor measurements.
- Converted date and time information into a datetime feature.
- Used chronological training and testing splits to preserve temporal order.

### 2. Feature Engineering

Created time-based features:

- Hour
- Month
- Weekday
- Hour sine and cosine
- Month sine and cosine
- Weekday sine and cosine

Cyclical encoding represents repeating time patterns more effectively.

### 3. Machine Learning Models

The following regression models were evaluated:

- Dummy Regressor (baseline)
- Linear Regression
- Random Forest Regressor

RandomizedSearchCV with TimeSeriesSplit was used for hyperparameter optimization.

### 4. Best Model

**Tuned Random Forest Regressor**

Best hyperparameters:

| Parameter | Value |
|---|---:|
| Number of estimators | 300 |
| Maximum depth | 20 |
| Minimum samples split | 5 |
| Minimum samples leaf | 2 |
| Maximum features | 0.7 |

## 📊 Model Performance

Final chronological test-set results:

| Metric | Result |
|---|---:|
| MAE | 0.3149 |
| MSE | 0.2177 |
| RMSE | 0.4666 |
| R² Score | 0.8713 |

The tuned Random Forest achieved an R² score of 0.8713 on the held-out chronological test set.

### Model Comparison

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Dummy Regressor | 1.0677 | 1.3197 | -0.0299 |
| Linear Regression | 0.6254 | 0.7411 | 0.6752 |
| Random Forest | 0.3292 | 0.4892 | 0.8585 |
| Tuned Random Forest | **0.3149** | **0.4666** | **0.8713** |

*CO concentration is measured in mg/m³. MSE is expressed in (mg/m³)².*

## 🖥️ Application Features

The project includes an interactive Streamlit web application with the following modules:

### 1. Dashboard
- Project overview.
- Model performance metrics.
- Model information and feature importance.

### 2. Manual Prediction
- Enter sensor and environmental readings.
- Automatically generate time-based features.
- Predict CO concentration.

### 3. Batch Prediction
- Upload CSV files containing the 14 required model features.
- Generate predictions for multiple records.
- Download prediction results.

### 4. Live Air Quality Monitoring
- Retrieve current air-quality information using the Open-Meteo API.
- Display AQI and pollutant measurements.
- Visualize hourly air-quality forecasts.
- Download API data.

### 5. Model Insights
- Model evaluation metrics.
- Feature importance visualization.
- Cross-validation results.
- Hyperparameter information.

### 6. Live CSV Analysis
- Upload CSV files downloaded from the live air-quality API.
- Explore pollutant statistics.
- Visualize pollutant trends.
- Download analyzed data.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| External API | Open-Meteo Air Quality API |
| Development Environment | Google Colab, Visual Studio Code |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
Air-Quality-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Manual_Prediction.py
│   ├── 3_Batch_Prediction.py
│   ├── 4_Live_Air_Quality.py
│   ├── 5_Model_Insights.py
│   └── 6_Live_CSV_Analysis.py
│
├── models/
│   ├── best_model_pipeline.joblib
│   └── model_metadata.json
│
├── data/
│
├── notebooks/
│   └── Air_Quality_Prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or compatible version.
- Git.
- Visual Studio Code (recommended).

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Air-Quality-Prediction-Using-Machine-Learning.git
```

### Step 2: Navigate to the project directory

```bash
cd Air-Quality-Prediction-Using-Machine-Learning
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

**Important:** The trained model was serialized using Scikit-learn 1.6.1. Keep this version when loading the provided Joblib model.

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Live Air Quality API

This project uses the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) to retrieve current and forecast air-quality information.

The live API monitoring module is independent of the UCI-trained machine learning model because their input features and measurement sources differ.

## ⚠️ Limitations

- The training dataset contains historical measurements from 2004–2005.
- The model was trained using measurements from a specific monitoring location.
- Predictions depend on the quality and compatibility of user-provided input features.
- The model should not be interpreted as a verified real-time monitoring or future forecasting system.
- Live API measurements are displayed separately and are not directly passed into the trained model.
- Model performance on historical test data does not guarantee equivalent performance under new environmental conditions.

## 🔮 Future Improvements

- Evaluate the model using more recent and geographically diverse air-quality datasets.
- Investigate time-series forecasting techniques.
- Explore additional regression algorithms and ensemble methods.
- Improve model explainability using SHAP.
- Integrate compatible sensor data from additional monitoring stations.
- Deploy the application using a cloud hosting platform.

## 📚 References

1. De Vito, S., et al. (2008). *On the use of neural networks for the prediction of benzene concentration in an urban environment*. Sensors and Actuators B: Chemical.
2. [UCI Machine Learning Repository – Air Quality Dataset](https://archive.ics.uci.edu/dataset/387/air+quality).
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/).
4. [Streamlit Documentation](https://docs.streamlit.io/).
5. [Open-Meteo Air Quality API Documentation](https://open-meteo.com/en/docs/air-quality-api).

---

## 👨‍💻 Author

**Sourodip Das Gupta**

Computer Science and Engineering Student, ML, NLP researcher

GitHub: https://github.com/PARZIVAL283

Email: sourodipdasgupta283@gmail.com

linkedIn: www.linkedin.com/in/sourodip-das-gupta-3b97a62a6

---

⭐ If you find this project interesting, consider giving the repository a star!

**Note:** This project was developed for academic and educational purposes.
