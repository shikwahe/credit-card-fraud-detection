import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details to predict if it is Fraud or Legitimate")

# Only show some important features
v1 = st.number_input("Feature V1")
v2 = st.number_input("Feature V2")
v3 = st.number_input("Feature V3")
v4 = st.number_input("Feature V4")
v5 = st.number_input("Feature V5")

amount = st.number_input("Transaction Amount")

# Prediction button
if st.button("Predict Fraud"):

    data = np.array([[v1,v2,v3,v4,v5,amount]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
        risk = 80
    else:
        st.success("✅ Legitimate Transaction")
        risk = 20

    st.subheader("Fraud Risk Meter")

    fig1, ax1 = plt.subplots(figsize=(3,2))
    ax1.barh(["Risk"], [risk])
    ax1.set_xlim(0,100)
    ax1.set_title("Fraud Risk Level")
    st.pyplot(fig1)

    st.subheader("Transaction Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(3,2))
        ax2.pie([80,20], labels=["Legit","Fraud"], autopct="%1.1f%%")
        ax2.set_title("Fraud vs Legit")
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(3,2))
        ax3.bar(["Amount"], [amount])
        ax3.set_title("Transaction Amount")
        st.pyplot(fig3)
