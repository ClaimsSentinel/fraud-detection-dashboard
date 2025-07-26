import streamlit as st
import pandas as pd
import joblib

st.title("🕵️‍♀️ Insurance Fraud Detection Dashboard")
st.markdown("Upload a CSV file with claims data to predict fraud likelihood.")

# Upload CSV
uploaded_file = st.file_uploader("Upload claim data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, nrows=500)

    # Load trained model
    model = joblib.load("fraud_model.pkl")

    # Check if column exists
    if "Claim Description" in df.columns:
        # Apply model to the column (must be transformed into proper input format)
        predictions = model.predict(df["Claim Description"])
        df["Fraud Prediction"] = predictions

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(df)
    else:
        st.error("Column 'Claim Description' not found in uploaded CSV.")
