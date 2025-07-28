import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import base64
from io import BytesIO
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="ClaimsSentinel", layout="centered")

# ---------------------------
# ✅ LOGO
# ---------------------------
logo_path = "logo/claimsentinel_logo.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    st.markdown("""
        <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
            <img src='data:image/png;base64,""" + \
        base64.b64encode(open(logo_path, "rb").read()).decode() + """' style='width: 400px;' />
        </div>""", unsafe_allow_html=True)

REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

# ---------------------------
# ✅ FUNCTIONS
# ---------------------------
def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.6):
    mapping = {}
    for req in required_cols:
        match = get_close_matches(req, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req] = match[0] if match else None
    return mapping

def clean_dataframe(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'[$,]', '', regex=True)
    return df

def highlight_fraud(row):
    return ['background-color: MistyRose' if row.get("Potential Fraud") == 1 else '' for _ in row]

model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None
explainer = None

# ---------------------------
# ✅ PREDICTION
# ---------------------------
st.markdown("""<div style='width:600px;'>""", unsafe_allow_html=True)
st.subheader("📂 Upload Claims File")
file = st.file_uploader("", type=["csv", "xlsx"], key="predict")

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df = clean_dataframe(df)
    mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS)

    if any(v is None for v in mapping.values()):
        st.error("Missing columns: " + ", ".join([k for k, v in mapping.items() if v is None]))
    else:
        df.rename(columns={v: k for k, v in mapping.items() if v}, inplace=True)
        try:
            X = df[REQUIRED_COLUMNS]
            if model:
                preds = model.predict(X)
                df["Potential Fraud"] = preds
                st.success(f"{sum(preds)} claims flagged as potential fraud")

                st.dataframe(df.style.apply(highlight_fraud, axis=1), use_container_width=True)

                explain_row = st.selectbox("Select a claim to explain:", df.index[df["Potential Fraud"] == 1].tolist())
                if st.button("Explain why this is potential fraud"):
                    if explainer is None:
                        X_transformed = model.named_steps['preprocessor'].transform(X)
                        explainer = shap.Explainer(model.named_steps['classifier'], X_transformed)
                    shap_values = explainer(X_transformed)
                    shap.initjs()
                    st.pyplot(shap.plots.waterfall(shap_values[explain_row], show=False))

                # Save SHAP summary to Excel
                if 'shap_values' in locals():
                    top_features = shap_values[explain_row].values
                    feature_names = shap_values[explain_row].feature_names
                    sorted_idx = np.argsort(np.abs(top_features))[::-1][:3]
                    reasons = [f"{feature_names[i]}: {top_features[i]:.2f}" for i in sorted_idx]
                    df["Potential Fraud Factors"] = ""
                    df.loc[explain_row, "Potential Fraud Factors"] = ", ".join(reasons)

                out = BytesIO()
                df.to_excel(out, index=False)
                st.download_button("Download Results (Excel)", out.getvalue(), file_name="results.xlsx")
            else:
                st.warning("No trained model found. Retrain below.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# ✅ RETRAINING
# ---------------------------
st.markdown("---")
st.markdown("""<div style='width:600px;'>""", unsafe_allow_html=True)
st.subheader("🧠 Retrain Model")

with st.expander("📚 Upload labeled training data"):
    train_file = st.file_uploader("Upload training file with 'Fraud Label' column", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
        df = clean_dataframe(df)

        full_required = REQUIRED_COLUMNS + ["Fraud Label"]
        mapping = fuzzy_column_map(df.columns.tolist(), full_required)
        if any(v is None for k, v in mapping.items() if k in full_required):
            st.error("Missing columns: " + ", ".join([k for k, v in mapping.items() if v is None]))
        else:
            df.rename(columns={v: k for k, v in mapping.items() if v}, inplace=True)
            try:
                X = df[REQUIRED_COLUMNS]
                y = df["Fraud Label"]

                numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categoricals = X.select_dtypes(include=["object", "category"]).columns.tolist()

                preprocessor = ColumnTransformer([
                    ("num", StandardScaler(), numeric),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
                ])

                clf = LogisticRegression(max_iter=1000) if model_choice == "Logistic Regression" else RandomForestClassifier()
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                joblib.dump(pipeline, model_path)
                model = pipeline

                st.success("Model retrained and saved!")
                st.text(classification_report(y_test, y_pred))
            except Exception as e:
                st.error(f"Training error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
