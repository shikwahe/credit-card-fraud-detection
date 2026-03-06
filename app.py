import streamlit as st
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# App title
st.markdown("<h1 style='text-align:center;color:#00c6ff'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("Transaction Details")
v1 = st.sidebar.number_input("Feature V1")
v2 = st.sidebar.number_input("Feature V2")
v3 = st.sidebar.number_input("Feature V3")
v4 = st.sidebar.number_input("Feature V4")
v5 = st.sidebar.number_input("Feature V5")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)

# Sidebar toggle for demo
transaction_type = st.sidebar.radio("Select Transaction Type for Demo", ["Legitimate", "Fraud"])

# Predict button
predict_btn = st.sidebar.button("🚀 Predict Transaction")

# Prediction & Dashboard
if predict_btn:
    # Demo logic
    fraud_prob = 72 if transaction_type == "Fraud" else 15
    risk = fraud_prob

    if fraud_prob > 30:
        st.error(f"⚠ Fraudulent Transaction Detected ({fraud_prob:.2f}% risk)")
    else:
        st.success(f"✅ Legitimate Transaction ({fraud_prob:.2f}% fraud risk)")

    st.subheader("Fraud Analysis Dashboard")
    col1, col2 = st.columns(2)

    # Risk Meter
    with col1:
        color = "red" if risk > 30 else "green"
        fig1, ax1 = plt.subplots(figsize=(4,2))
        ax1.barh(["Risk Level"], [risk], color=color)
        ax1.set_xlim(0,100)
        ax1.set_xlabel("Fraud Risk %")
        ax1.set_title("Fraud Risk Meter")
        st.pyplot(fig1)

    # Fraud vs Legit Pie Chart
    with col2:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.pie([100-risk, risk], labels=["Legitimate","Fraud"], autopct="%1.1f%%", colors=["#00ff9f","#ff4b4b"])
        ax2.set_title("Fraud vs Legit Distribution")
        st.pyplot(fig2)
