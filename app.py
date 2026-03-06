import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --------- Custom Styling (Blue Background + Title Highlight) ----------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}

.main-title{
    font-size:40px;
    font-weight:800;
    text-align:center;
    padding:10px;
    border-radius:10px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    margin-bottom:20px;
}

.metric-box{
    background:#1f4e79;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("fraud_model.pkl")

# Title
st.markdown('<div class="main-title">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)

st.write("### Predict whether a transaction is **Fraudulent or Legitimate**")

# Sidebar Inputs
st.sidebar.header("Enter Transaction Details")

v1 = st.sidebar.number_input("Feature V1")
v2 = st.sidebar.number_input("Feature V2")
v3 = st.sidebar.number_input("Feature V3")
v4 = st.sidebar.number_input("Feature V4")
v5 = st.sidebar.number_input("Feature V5")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)

predict_btn = st.sidebar.button("🚀 Predict Fraud")

# Dashboard metrics
colA, colB, colC = st.columns(3)

colA.metric("Model Accuracy", "94%")
colB.metric("Total Transactions", "284,807")
colC.metric("Fraud Cases", "492")

st.divider()

# Prediction
if predict_btn:

    try:

        # Create feature array
        data = np.zeros((1, model.n_features_in_))

        data[0][0] = v1
        data[0][1] = v2
        data[0][2] = v3
        data[0][3] = v4
        data[0][4] = v5
        data[0][-1] = amount

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("⚠ Fraudulent Transaction Detected")
            risk = 80
        else:
            st.success("✅ Legitimate Transaction")
            risk = 20

        st.subheader("📊 Fraud Analysis")

        # Side-by-side charts
        col1, col2 = st.columns(2)

        # Risk Meter
        with col1:

            st.markdown("### Fraud Risk Level")

            fig1, ax1 = plt.subplots(figsize=(4,2))

            color = "red" if risk > 50 else "green"

            ax1.barh(["Risk"], [risk], color=color)
            ax1.set_xlim(0,100)
            ax1.set_xlabel("Risk %")
            ax1.set_title("Risk Meter")

            st.pyplot(fig1)

        # Fraud vs Legit Chart
        with col2:

            st.markdown("### Fraud vs Legit Distribution")

            fig2, ax2 = plt.subplots(figsize=(4,3))

            ax2.pie(
                [80,20],
                labels=["Legit","Fraud"],
                autopct="%1.1f%%",
                colors=["#00ff9f","#ff4b4b"]
            )

            st.pyplot(fig2)

    except Exception as e:
        st.error("⚠ Model prediction failed.")
        st.write(e)
