# Sidebar choice for demo
transaction_type = st.sidebar.radio("Select Transaction Type for Demo", ["Legitimate", "Fraud"])

predict_btn = st.sidebar.button("🚀 Predict Transaction")

if predict_btn:
    # -------------------
    # DEMO ONLY LOGIC
    # -------------------
    if transaction_type == "Fraud":
        fraud_prob = 72
    else:
        fraud_prob = 15

    risk = fraud_prob

    # Show result
    if fraud_prob > 30:
        st.error(f"⚠ Fraudulent Transaction Detected ({fraud_prob:.2f}% risk)")
    else:
        st.success(f"✅ Legitimate Transaction ({fraud_prob:.2f}% fraud risk)")

    # Fraud Analysis Dashboard
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
        ax2.pie(
            [100-risk, risk],
            labels=["Legitimate","Fraud"],
            autopct="%1.1f%%",
            colors=["#00ff9f","#ff4b4b"]
        )
        ax2.set_title("Fraud vs Legit Distribution")
        st.pyplot(fig2)
