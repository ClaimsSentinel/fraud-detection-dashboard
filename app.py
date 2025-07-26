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
    "Claim Description",
    "Claim ID",
    "Adjuster Notes",
    "Date of Claim",
    "Policyholder ID"
]

    if all(col in df.columns for col in required_columns):
        model = joblib.load("fraud_model.pkl")
        predictions = model.predict(df)
        df["Fraud Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df)
        st.download_button(
    label="💾 Download Predictions for Review",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="fraud_predictions_for_review.csv",
    mime="text/csv"
)
    else:
        st.error("Uploaded file is missing one or more required columns:")
        st.code("\n".join(required_columns))

import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

st.markdown("---")
st.header("🔁 Retrain Model with Human Feedback")

feedback_file = st.file_uploader("Upload labeled feedback (.csv)", type="csv", key="feedback")

if feedback_file:
    feedback_df = pd.read_csv(feedback_file)

    required_cols = [
        "Claim Amount", "Previous Claims Count", "Claim Location",
        "Vehicle Make/Model", "Claim Description", "True Fraud"
    ]

    if all(col in feedback_df.columns for col in required_cols):
        # Prepare X and y
        X = feedback_df.drop("True Fraud", axis=1)
        y = feedback_df["True Fraud"]

        # Rebuild same pipeline
        numeric_features = ["Claim Amount", "Previous Claims Count"]
        categorical_features = ["Claim Location", "Vehicle Make/Model"]
        text_feature = "Claim Description"

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("txt", TfidfVectorizer(max_features=50), text_feature)
        ])

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train/test split and fit
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        # Save model
        joblib.dump(pipeline, "fraud_model.pkl")

        # Show evaluation
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=False)
        st.success("✅ Model retrained successfully!")
        st.text(report)

    else:
        st.error("Missing one or more required columns:")
        st.code("\n".join(required_cols))
