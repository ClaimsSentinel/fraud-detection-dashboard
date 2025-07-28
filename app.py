import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
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

# Display high-def centered logo
logo_path = "logo/claimsentinel_logo.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=False, width=400)

REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

FUZZY_SYNONYMS = {
    "Claim Amount": ["claim_amt", "amount claimed", "total claim"],
    "Previous Claims Count": ["prev claims", "claim count"],
    "Claim Location": ["location", "incident location"],
    "Vehicle Make/Model": ["car model", "vehicle details"],
    "Claim Description": ["description", "incident description"],
    "Claim ID": ["claim number", "id", "ref id"],
    "Adjuster Notes": ["adjuster comment", "notes", "adjuster remarks"],
    "Date of Claim": ["claim date", "incident date"],
    "Policyholder ID": ["customer id", "client id"]
}

# Enhanced fuzzy column mapper
def fuzzy_column_map(uploaded_cols, required_cols):
    mapping = {}
    lower_uploaded = {col.lower(): col for col in uploaded_cols}
    for req in required_cols:
        synonyms = [req] + FUZZY_SYNONYMS.get(req, [])
        match = next((lower_uploaded[c] for c in lower_uploaded if any(s.lower() in c for s in synonyms)), None)
        mapping[req] = match
    return mapping

# Clean numbers
def clean_dataframe(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'[$,]', '', regex=True)
    return df

model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None
explainer = None

st.markdown("<div style='width:600px;'>", unsafe_allow_html=True)
st.subheader("📂 Upload Claims File")
file = st.file_uploader("", type=["csv", "xlsx"], key="predict")

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df = clean_dataframe(df)
    mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS)
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        st.warning("⚠️ Soft warning: fuzzy matched some fields. Check results.")
    df.rename(columns={v: k for k, v in mapping.items() if v}, inplace=True)
    try:
        X = df[REQUIRED_COLUMNS]
        if model:
            preds = model.predict(X)
            df["Potential Fraud"] = preds

            def highlight_fraud(row):
                return ['background-color: MistyRose' if row["Potential Fraud"] == 1 else '' for _ in row]

            st.dataframe(df.style.apply(highlight_fraud, axis=1), use_container_width=True)
            fraud_indexes = df.index[df["Potential Fraud"] == 1].tolist()
            if fraud_indexes:
                selected_row = st.selectbox("Select a claim to explain:", fraud_indexes)
                if st.button("Explain why this is potential fraud"):
                    if explainer is None:
                        explainer = shap.Explainer(model.named_steps['classifier'])
                        X_transformed = model.named_steps['preprocessor'].transform(X)
                    shap_values = explainer(X_transformed)
                    shap.initjs()
                    st.pyplot(shap.plots.waterfall(shap_values[selected_row], show=False), bbox_inches='tight')

            # Add top 2 SHAP values
            if explainer is None:
                explainer = shap.Explainer(model.named_steps['classifier'])
                X_transformed = model.named_steps['preprocessor'].transform(X)
            shap_vals = explainer(X_transformed)
            top_features = [", ".join([f.feature_names[i] for i in np.argsort(-abs(f.values))[:2]]) for f in shap_vals]
            df["Potential Fraud Factors"] = top_features

            out = BytesIO()
            df.to_excel(out, index=False)
            st.download_button("Download Results (Excel)", out.getvalue(), file_name="results.xlsx")
        else:
            st.warning("⚠️ No trained model found. Retrain below.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='width:600px;'>", unsafe_allow_html=True)
st.subheader("🧠 Retrain Model")
with st.expander("📚 Upload labeled training data"):
    train_file = st.file_uploader("Upload training file with 'Fraud Label' column", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
        df = clean_dataframe(df)
        mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS + ["Fraud Label"])
        missing = [k for k, v in mapping.items() if v is None and k != "Fraud Label"]
        if missing:
            st.warning("⚠️ Soft warning: fuzzy matched some training fields.")
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
            st.success("✅ Model retrained and saved!")
            st.text(classification_report(y_test, y_pred))
        except Exception as e:
            st.error(f"Training error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
