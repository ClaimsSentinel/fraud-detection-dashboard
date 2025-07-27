import streamlit as st
import pandas as pd
import joblib
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up Streamlit page
st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
st.title("🕵️‍♀️ Insurance Fraud Detection Dashboard")
st.markdown("Upload your claim data (CSV or Excel) to detect potential fraud cases.")

# Define expected columns
expected_columns = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    """Maps uploaded columns to expected ones using fuzzy matching."""
    return {
        req: get_close_matches(req, uploaded_cols, n=1, cutoff=cutoff)[0]
        if get_close_matches(req, uploaded_cols, n=1, cutoff=cutoff) else None
        for req in required_cols
    }

def load_data(file):
    """Reads uploaded file as DataFrame."""
    try:
        if file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# --- Prediction Section ---
st.subheader("📊 Predict Fraud from New Claims")
uploaded_file = st.file_uploader("Upload a claim file", type=["csv", "xlsx"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        column_map = fuzzy_column_map(df_raw.columns.tolist(), expected_columns)
        missing = [col for col, match in column_map.items() if match is None]

        if missing:
            st.error("Missing required columns:")
            st.code("\n".join(missing))
        else:
            df = df_raw.rename(columns={v: k for k, v in column_map.items()})
            try:
                model = joblib.load("fraud_model.pkl")
                df_for_prediction = df[model.feature_names_in_].copy()
                predictions = model.predict(df_for_prediction)
                df["Fraud Prediction"] = predictions

                st.success("✅ Predictions completed!")
                st.dataframe(df)

                st.download_button(
                    label="💾 Download Results",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

# --- Retraining Section ---
st.markdown("---")
st.header("🔁 Retrain Model with Labeled Feedback")

feedback_file = st.file_uploader("Upload labeled feedback file", type=["csv", "xlsx"], key="feedback")

if feedback_file:
    feedback_df = load_data(feedback_file)
    feedback_required = [
        "Claim Amount", "Previous Claims Count", "Claim Location",
        "Vehicle Make/Model", "Claim Description", "True Fraud"
    ]

    if feedback_df is not None:
        if all(col in feedback_df.columns for col in feedback_required):
            X = feedback_df.drop("True Fraud", axis=1)
            y = feedback_df["True Fraud"]

            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), ["Claim Amount", "Previous Claims Count"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["Claim Location", "Vehicle Make/Model"]),
                ("txt", TfidfVectorizer(max_features=50), "Claim Description")
            ])

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)

            joblib.dump(pipeline, "fraud_model.pkl")

            report = classification_report(y_test, pipeline.predict(X_test), output_dict=False)
            st.success("✅ Model retrained and saved.")
            st.text(report)
        else:
            st.error("Missing required feedback columns:")
            st.code("\n".join(feedback_required))
