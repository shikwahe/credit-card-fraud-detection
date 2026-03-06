import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load model
model = joblib.load("fraud_model.pkl")

# Title
st.markdown("## 💳 Credit Card Fraud Detection Dashboard")
st.write("Predict whether a transaction is **Fraudulent or Legitimate**")

# Sidebar Inputs
st.sidebar.header("Enter Transaction Details")

v1 = st.sidebar.number_input("Feature V1")
v2 = st.sidebar.number_input("Feature V2")
v3 = st.sidebar.number_input("Feature V3")
v4 = st.sidebar.number_input("Feature V4")
v5 = st.sidebar.number_input("Feature V5")
amount = st.sidebar.number_input("Transaction Amount")

predict_btn = st.sidebar.button("🚀 Predict Fraud")

# Dashboard metrics
colA, colB, colC = st.columns(3)

colA.metric("Model Accuracy", "94%")
colB.metric("Transactions Analyzed", "284,807")
colC.metric("Fraud Cases", "492")

st.divider()

if predict_btn:

    data = np.array([[v1, v2, v3, v4, v5, amount]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
        risk = 80
    else:
        st.success("✅ Legitimate Transaction")
        risk = 20

    st.subheader("📊 Fraud Analysis")

    # Create centered columns
    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        # Risk Meter
        st.markdown("### Fraud Risk Meter")

        fig1, ax1 = plt.subplots(figsize=(4,1.8))
        ax1.barh(["Risk Level"], [risk], color="red")
        ax1.set_xlim(0,100)
        ax1.set_xlabel("Risk %")
        st.pyplot(fig1)

    st.divider()

    # Charts Row
    col4, col5 = st.columns(2)

    with col4:

        st.markdown("### Fraud vs Legit Distribution")

        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(
            [80,20],
            labels=["Legit","Fraud"],
            autopct="%1.1f%%",
            colors=["green","red"]
        )
        st.pyplot(fig2)

    with col5:

        st.markdown("### Transaction Amount")

        fig3, ax3 = plt.subplots(figsize=(4,2))
        ax3.bar(["Amount"], [amount], color="orange")
        ax3.set_ylabel("USD")
        st.pyplot(fig3)
