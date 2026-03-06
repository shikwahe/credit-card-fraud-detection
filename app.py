import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Background styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}
.title {
    font-size:42px;
    font-weight:bold;
    text-align:center;
    padding:12px;
    border-radius:10px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    margin-bottom:25px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("fraud_model.pkl")

# Title
st.markdown('<div class="title">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.write("### Enter transaction details to check if it is Fraud or Legitimate")

# Sidebar inputs
st.sidebar.header("Transaction Details")

v1 = st.sidebar.number_input("Feature V1")
v2 = st.sidebar.number_input("Feature V2")
v3 = st.sidebar.number_input("Feature V3")
v4 = st.sidebar.number_input("Feature V4")
v5 = st.sidebar.number_input("Feature V5")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)

predict_btn = st.sidebar.button("🚀 Predict Transaction")

st.divider()

if predict_btn:

    try:
        # Create full feature array
        data = np.zeros((1, model.n_features_in_))

        data[0][0] = v1
        data[0][1] = v2
        data[0][2] = v3
        data[0][3] = v4
        data[0][4] = v5
        data[0][-1] = amount

        # Prediction
        prediction = model.predict(data)

        # Probability
        prob = model.predict_proba(data)
        fraud_prob = prob[0][1] * 100

        # Result
        if fraud_prob > 30:
            st.error(f"⚠ Fraudulent Transaction Detected ({fraud_prob:.2f}% risk)")
        else:
            st.success(f"✅ Legitimate Transaction ({fraud_prob:.2f}% fraud risk)")

        st.subheader("Fraud Analysis Dashboard")

        col1, col2 = st.columns(2)

        # Risk Meter
        with col1:

            risk = fraud_prob
            color = "red" if risk > 30 else "green"

            fig1, ax1 = plt.subplots(figsize=(4,2))
            ax1.barh(["Risk Level"], [risk], color=color)
            ax1.set_xlim(0,100)
            ax1.set_xlabel("Fraud Risk %")
            ax1.set_title("Fraud Risk Meter")

            st.pyplot(fig1)

        # Fraud vs Legit Chart
        with col2:

            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.pie(
                [100-risk, risk],
                labels=["Legitimate","Fraud"],
                autopct="%1.1f%%",
                colors=["#00ff9f","#ff4b4b"]
            )
            ax2.set_title("Fraud vs Legit Distribution")

            st.pyplot(fig2)

    except Exception as e:
        st.error("⚠ Prediction failed.")
        st.write(e)
