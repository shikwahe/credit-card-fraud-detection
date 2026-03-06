import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details")

# Input fields
time = st.number_input("Transaction Time")
amount = st.number_input("Transaction Amount")

# V features
features = []

for i in range(1, 29):
    val = st.number_input(f"V{i}")
    features.append(val)

# Combine all features
input_data = np.array([[time] + features + [amount]])

# Prediction
if st.button("Predict Transaction"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")