import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
 
    model = tf.keras.models.load_model("models/heart_failure_ann.h5")
    return model

model = load_model()

# ----------------------------
# App UI
# ----------------------------
st.title("❤️ Heart Failure Prediction (ANN)")
st.write("Enter patient data to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert categorical inputs to match training encoding
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "RestingECG": [resting_ecg],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})

# Same preprocessing as notebook
input_encoded = pd.get_dummies(input_data)
# Ensure same columns as training
# (You need to save training columns from notebook into pickle)
with open("models/columns1.pkl", "rb") as f:
    training_columns = pickle.load(f)
scaler = joblib.load("models/scaler.pkl")


for col in training_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[training_columns]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    pred = model.predict(input_encoded)
    result = (pred > 0.5).astype("int")[0][0]

    if result == 1:
        st.error("⚠️ The model predicts **Heart Disease**.")
    else:
        st.success("✅ The model predicts **No Heart Disease**.")





