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

# Display centered, high-resolution logo only
logo_path = "logo/claimsentinel_logo.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    st.markdown("""
        <div style='display: flex; justify-content: center; margin-top: 1rem;'>
            <img src='data:image/png;base64,""" + BytesIO(image.tobytes()).getvalue().hex() + """' style='width: 400px;'/>
        </div>
    """, unsafe_allow_html=True)

REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

FALLBACK_COLUMN_MAP = {
    "Claim Amount": ["Total Claim Value"],
    "Previous Claims Count": ["Num Prev Claims"],
    "Claim Location": ["Incident Locale"],
    "Vehicle Make/Model": ["Vehicle Brand/Model"],
    "Claim Description": ["Incident Summary"],
    "Claim ID": ["Ref ID"],
    "Adjuster Notes": ["Adjuster Comments"],
    "Date of Claim": ["Filing Date"],
    "Policyholder ID": ["Insured Party ID"]
}

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.6):
    mapping = {}
    for req in required_cols:
        match = get_close_matches(req, uploaded_cols, n=1, cutoff=cutoff)
        if match:
            mapping[req] = match[0]
        else:
            for fallback in FALLBACK_COLUMN_MAP.get(req, []):
                if fallback in uploaded_cols:
                    mapping[req] = fallback
                    break
            else:
                mapping[req] = None
    return mapping

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
    if any(v is None for v in mapping.values()):
        st.error("Missing columns: " + ", ".join([k for k, v in mapping.items() if v is None]))
    else:
        df.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
        X = df[REQUIRED_COLUMNS]
        if model:
            try:
                preds = model.predict(X)
                df["Potential Fraud"] = preds
                st.success(f"{sum(preds)} claims flagged as potential fraud")

                def highlight_fraud(row):
                    return ['background-color: MistyRose' if row["Potential Fraud"] == 1 else '' for _ in row]

                st.dataframe(df.style.apply(highlight_fraud, axis=1), use_container_width=True)

                selected_row = st.selectbox("Select a claim to explain:", df.index[df["Potential Fraud"] == 1].tolist())
                if st.button("Explain why this is potential fraud"):
                    if explainer is None:
                        explainer = shap.Explainer(model.named_steps['classifier'])
                        X_transformed = model.named_steps['preprocessor'].transform(X)
                    shap_values = explainer(X_transformed)
                    shap.initjs()
                    st_shap = shap.plots.waterfall(shap_values[selected_row], show=False)
                    st.pyplot(bbox_inches='tight')

                out = BytesIO()
                df.to_excel(out, index=False)
                st.download_button("Download Results (Excel)", out.getvalue(), file_name="results.xlsx")

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("No trained model found. Retrain below.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='width:600px;'>", unsafe_allow_html=True)
st.subheader("🧠 Retrain Model")
with st.expander("Upload labeled training data"):
    train_file = st.file_uploader("Upload training file with 'Fraud Label' column", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
        df = clean_dataframe(df)
        mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS + ["Fraud Label"])
        if any(v is None for k, v in mapping.items() if k != "Fraud Label"):
            st.error("Missing columns: " + ", ".join([k for k, v in mapping.items() if v is None and k != "Fraud Label"]))
        else:
            df.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
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

st.markdown("</div>", unsafe_allow_html=True)
