import streamlit as st
import pandas as pd
import joblib

st.title("🕵️‍♀️ Insurance Fraud Detection Dashboard")
st.markdown("Upload a CSV file with claims data to predict fraud likelihood.")

uploaded_file = st.file_uploader("Upload claim data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, nrows=500)

    required_columns = [
        "Claim Amount",
        "Previous Claims Count",
        "Claim Location",
        "Vehicle Make/Model",
        "Claim Description"
    ]

    if all(col in df.columns for col in required_columns):
        model = joblib.load("fraud_model.pkl")
        predictions = model.predict(df[required_columns])
        df["Fraud Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df)
    else:
        st.error("Uploaded file is missing one or more required columns:")
        st.code("\n".join(required_columns))
