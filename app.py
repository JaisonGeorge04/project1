import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

st.title("Breast Cancer Diagnosis Predictor")
st.write("""
Provide the feature values below. The model predicts whether the tumor is:
- **Benign (1)**  
- **Malignant (0)**
""")

# -------------------------
# Load Model and Scaler
# -------------------------
model_path = "logistic.pkl"
scaler_path = "gbc.pkl"

if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
    st.error("Model files not found. Upload 'gradient_boosting_model.pkl' and 'scaler.pkl' to the same directory.")
    st.stop()

try:
    gbc_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Feature Names
# -------------------------
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# -------------------------
# Input Form
# -------------------------
st.subheader("Enter Feature Values")

input_data = {}

cols = st.columns(3)
for idx, feature in enumerate(feature_names):
    with cols[idx % 3]:
        input_data[feature] = st.number_input(
            label=feature.replace("_", " ").title(),
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f"
        )

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    try:
        # Convert dict → DataFrame
        input_df = pd.DataFrame([input_data])[feature_names]

        # Scale data
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = gbc_model.predict(input_scaled)[0]

        # Output
        if prediction == 1:
            st.success("Prediction: **Benign**")
        else:
            st.error("Prediction: **Malignant**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

