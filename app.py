
import streamlit as st
import pandas as pd
import joblib("fraud_model.pkl")

st.title("🕵️‍♀️ Insurance Fraud Detection Dashboard")
st.markdown("Upload a CSV file with claims data to predict fraud likelihood.")

uploaded_file = st.file_uploader("Upload claim data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load trained model
    model = joblib.load("fraud_model.pkl")

    # Run predictions
    if "Claim Description" in df.columns:
        predictions = model.predict(df["Claim Description"])
        df["Fraud Prediction"] = predictions
        st.subheader("Prediction Results")
        st.dataframe(df)
    else:
        st.error("Column 'Claim Description' not found in uploaded CSV.")
