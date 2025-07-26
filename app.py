import streamlit as st
import pandas as pd
import joblib
from difflib import get_close_matches

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
st.title("🕵️‍♀️ Insurance Fraud Detection Dashboard")
st.markdown("Upload a CSV file with claims data to predict fraud likelihood.")

# Define required columns
expected_columns = [
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

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        if match:
            mapping[req_col] = match[0]
        else:
            mapping[req_col] = None
    return mapping

# Upload main claim file
uploaded_file = st.file_uploader("📂 Upload claim data (.csv)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    column_map = fuzzy_column_map(df_raw.columns.tolist(), expected_columns)

    # Check for unmatched columns
    unmatched = [k for k, v in column_map.items() if v is None]
    if unmatched:
        st.error("❌ Some required columns were not found in your file:")
        st.code("\n".join(unmatched))
    else:
        # Rename columns based on fuzzy match
        df = df_raw.rename(columns={v: k for k, v in column_map.items()})

        try:
            model = joblib.load("fraud_model.pkl")
            predictions = model.predict(df)
            df["Fraud Prediction"] = predictions

            st.subheader("✅ Prediction Results")
            st.dataframe(df)

            st.download_button(
                label="💾 Download Predictions for Review",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ---------------------------------------
# Retrain Section
# ---------------------------------------

st.markdown("---")
st.header("🔁 Retrain Model with Human Feedback")

feedback_file = st.file_uploader("📂 Upload labeled feedback (.csv)", type="csv", key="feedback")

if feedback_file:
    feedback_df = pd.read_csv(feedback_file)

    feedback_required = [
        "Claim Amount", "Previous Claims Count", "Claim Location",
        "Vehicle Make/Model", "Claim Description", "True Fraud"
    ]

    if all(col in feedback_df.columns for col in feedback_required):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        X = feedback_df.drop("True Fraud", axis=1)
        y = feedback_df["True Fraud"]

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, "fraud_model.pkl")

        report = classification_report(y_test, pipeline.predict(X_test), output_dict=False)
        st.success("✅ Model retrained successfully!")
        st.text(report)

    else:
        st.error("Missing one or more required columns:")
        st.code("\n".join(feedback_required))
