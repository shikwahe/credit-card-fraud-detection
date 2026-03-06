import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection Dashboard")

st.write("Enter transaction details to check if the transaction is fraudulent.")

# Basic inputs
col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Transaction Time")

with col2:
    amount = st.number_input("Transaction Amount")

st.subheader("Transaction Features")

# Features layout
features = []
cols = st.columns(4)

for i in range(28):
    with cols[i % 4]:
        val = st.number_input(f"V{i+1}")
        features.append(val)

# Combine features
input_data = np.array([[time] + features + [amount]])

if st.button("Predict Transaction"):

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0]

    fraud_prob = prob[1]
    legit_prob = prob[0]

    # Result message
    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.subheader("Fraud Risk Meter")

    # Risk meter
    st.progress(float(fraud_prob))

    st.write(f"Fraud Probability: **{fraud_prob:.2f}**")

    st.subheader("Prediction Probability")

    # Probability bar chart
    labels = ["Legitimate", "Fraud"]
    values = [legit_prob, fraud_prob]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Fraud Prediction Confidence")

    st.pyplot(fig)

    st.subheader("Fraud vs Legitimate Distribution")

    # Distribution chart
    fig2, ax2 = plt.subplots()

    ax2.pie(
        [legit_prob, fraud_prob],
        labels=["Legitimate", "Fraud"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax2.set_title("Transaction Risk Distribution")

    st.pyplot(fig2)
