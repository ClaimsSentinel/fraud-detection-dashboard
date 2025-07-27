# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import re
from io import BytesIO
from datetime import datetime
from difflib import get_close_matches
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

st.set_page_config(page_title="ClaimsSentinel | Insurance Fraud Detection", layout="centered")
st.title("🛡️ ClaimsSentinel")
st.markdown("### Smart insights. Safer claims.")

# ---------------------------
# Configuration
# ---------------------------
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

# ---------------------------
# Helper Functions
# ---------------------------
def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

def clean_dataframe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'[,$%\n\r\t\xa0\u202f\u00a0\-]', '', regex=True)
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

@st.cache_data

def load_model():
    if Path("model.joblib").exists():
        return joblib.load("model.joblib")
    return None

def save_model(model):
    joblib.dump(model, "model.joblib")

def preprocess_input(df, model):
    preprocessor = model.named_steps["preprocessor"]
    X = df[expected_columns]
    X_transformed = preprocessor.transform(X)
    return X, X_transformed

# ---------------------------
# Upload Section
# ---------------------------
st.markdown("---")
st.header("📂 Upload CSV or Excel File")

uploaded_file = st.file_uploader("Upload claim data to predict fraud:", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = clean_dataframe(df)

        if "Fraud Label" in df.columns:
            df = df.drop(columns=["Fraud Label"])  # Remove label if accidentally included

        mapping = fuzzy_column_map(df.columns.tolist(), expected_columns)
        missing = [k for k, v in mapping.items() if v is None]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            df = df.rename(columns={v: k for k, v in mapping.items() if v})
            model = load_model()

            if model is None:
                st.warning("⚠️ No trained model found. Please retrain the model first.")
            else:
                X, X_transformed = preprocess_input(df, model)
                y_pred = model.predict(X_transformed)

                df["Fraud Prediction"] = y_pred
                fraud_df = df[df["Fraud Prediction"] == 1]

                st.markdown("---")
                st.header("🔍 Predictions")
                st.dataframe(fraud_df if not fraud_df.empty else df, use_container_width=True, height=300)

                st.markdown(f"""
                    <div style='font-size: 18px; padding-top: 10px;'>
                    📊 <b>Total claims:</b> {len(df)} &nbsp; | &nbsp; ⚠️ <b>Flagged as fraud:</b> {fraud_df.shape[0]}
                    </div>
                """, unsafe_allow_html=True)

                # Download
                def convert_df(df):
                    return df.to_csv(index=False).encode("utf-8")

                st.download_button("📥 Download Results", convert_df(df), "fraud_predictions.csv", "text/csv")

                # SHAP explanation (per-line selected)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("🧠 Explain Why This Is Fraud")

                selected_claim_id = st.selectbox("Select a flagged Claim ID:", fraud_df["Claim ID"].unique() if not fraud_df.empty else ["None"])
                if st.button("Explain Selected Fraud Claim"):
                    try:
                        index_to_explain = fraud_df[fraud_df["Claim ID"] == selected_claim_id].index[0]
                        preprocessor = model.named_steps["preprocessor"]
                        classifier = model.named_steps["classifier"]

                        X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
                        X_numeric = np.array(X_dense, dtype=np.float64)

                        explainer = shap.TreeExplainer(classifier)
                        shap_values = explainer.shap_values(X_numeric)

                        shap_df = pd.DataFrame({
                            "Feature": preprocessor.get_feature_names_out(),
                            "SHAP Value": shap_values[1][index_to_explain]
                        }).sort_values(by="SHAP Value", ascending=False)

                        st.markdown(f"**Explaining Claim ID:** `{selected_claim_id}`")
                        st.dataframe(shap_df.head(10), use_container_width=True)
                    except Exception as e:
                        st.error(f"SHAP error: {e}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------------------------
# Retraining Section
# ---------------------------
st.markdown("---")
st.header("🧠 Retrain Fraud Detection Model")

model_choice = st.selectbox("Choose model type:", ["Random Forest", "Logistic Regression"])
train_file = st.file_uploader("Upload labeled training data:", type=["csv", "xlsx"], key="train")

if train_file:
    try:
        train_df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
        train_df = clean_dataframe(train_df)

        if "Fraud Label" not in train_df.columns:
            st.error("Missing 'Fraud Label' column in training data.")
        else:
            mapping = fuzzy_column_map(train_df.columns.tolist(), expected_columns + ["Fraud Label"])
            missing = [k for k, v in mapping.items() if v is None]

            if missing:
                st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
            else:
                train_df = train_df.rename(columns={v: k for k, v in mapping.items() if v})
                X = train_df[expected_columns]
                y = train_df["Fraud Label"]

                numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categoricals = X.select_dtypes(include=["object", "datetime64"]).columns.tolist()

                preprocessor = ColumnTransformer([
                    ("num", StandardScaler(), numeric),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
                ])

                clf = LogisticRegression(max_iter=1000) if model_choice == "Logistic Regression" else RandomForestClassifier(
                    n_estimators=200, max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, random_state=42, class_weight='balanced')

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])

                pipeline.fit(X, y)
                save_model(pipeline)
                st.success("✅ Model retrained and saved successfully.")
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
