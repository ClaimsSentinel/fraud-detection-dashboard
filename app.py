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

# Display high-def, centered logo
logo_path = "logo/claimsentinel_logo.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    from base64 import b64encode
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = b64encode(buf.getvalue()).decode()
    st.markdown(f"""
        <div style='display: flex; justify-content: center; margin-top: 1rem;'>
            <img src='data:image/png;base64,{encoded}' style='width: 400px;'/>
        </div>
    """, unsafe_allow_html=True)

REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

# Fuzzy mapping
fuzzy_synonyms = {
    "Total Claim Value": "Claim Amount",
    "Num Prev Claims": "Previous Claims Count",
    "Incident Locale": "Claim Location",
    "Vehicle Brand/Model": "Vehicle Make/Model",
    "Incident Summary": "Claim Description",
    "Ref ID": "Claim ID",
    "Adjuster Comments": "Adjuster Notes",
    "Filing Date": "Date of Claim",
    "Insured Party ID": "Policyholder ID",
    "Likely Fraud?": "Fraud Label"
}

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.6):
    mapping = {}
    for req in required_cols:
        if req in uploaded_cols:
            mapping[req] = req
        else:
            match = get_close_matches(req, uploaded_cols, n=1, cutoff=cutoff)
            if match:
                mapping[req] = match[0]
            else:
                fuzzy_match = [col for col in uploaded_cols if fuzzy_synonyms.get(col) == req]
                mapping[req] = fuzzy_match[0] if fuzzy_match else None
    return mapping

def clean_dataframe(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.replace(r'[$,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='ignore')
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
        X = df[REQUIRED_COLUMNS].copy()
        X = X.fillna(0)
        if model:
            try:
                preds = model.predict(X)
                df["Potential Fraud"] = preds

                # SHAP explainer setup
                X_transformed = model.named_steps['preprocessor'].transform(X)
                explainer = shap.Explainer(model.named_steps['classifier'], X_transformed)
                shap_values = explainer(X_transformed)

                def top_fraud_factors(shap_vals, feature_names):
                    result = []
                    for i in range(len(shap_vals)):
                        vals = shap_vals[i].values
                        top_idxs = np.argsort(np.abs(vals))[-2:][::-1]
                        signs = ["↑" if vals[j] > 0 else "↓" for j in top_idxs]
                        result.append(", ".join([f"{feature_names[j]} {signs[k]}" for k, j in enumerate(top_idxs)]))
                    return result

                df["Potential Fraud Factors"] = top_fraud_factors(shap_values, explainer.feature_names)

                def highlight_fraud(row):
                    return ['background-color: MistyRose' if row["Potential Fraud"] == 1 else '' for _ in row]

                st.dataframe(df.style.apply(highlight_fraud, axis=1), use_container_width=True)

                selected_row = st.selectbox("Select a claim to explain:", df.index[df["Potential Fraud"] == 1].tolist())
                if st.button("Explain why this is potential fraud"):
                    shap.plots.waterfall(shap_values[selected_row], show=False)
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
            X = df[REQUIRED_COLUMNS].copy()
            X = X.fillna(0)
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
