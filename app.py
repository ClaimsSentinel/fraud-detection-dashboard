
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fraud_model_pipeline import pipeline

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🔍 Insurance Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload an insurance claims CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load uploaded data
        df_input = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df_input.head())

        # Run predictions
        st.subheader("🚩 Predicted Fraud Results")
        predictions = pipeline.predict(df_input)
        df_input["Fraudulent_Prediction"] = predictions
        st.dataframe(df_input)

        # Show summary
        fraud_count = sum(predictions)
        total_count = len(predictions)
        st.markdown(f"**🔢 Total Claims:** {total_count}")
        st.markdown(f"**🚨 Predicted Fraudulent Claims:** {fraud_count}")

        # Visuals
        st.subheader("📊 Claim Amount Distribution")
        fig, ax = plt.subplots()
        df_input.boxplot(column="Claim Amount", by="Fraudulent_Prediction", ax=ax)
        plt.title("Claim Amount by Predicted Fraud")
        plt.suptitle("")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("👆 Upload a CSV file to get started.")
