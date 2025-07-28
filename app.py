
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import base64
from io import BytesIO
from datetime import datetime
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from PIL import Image

st.set_page_config(page_title="ClaimsSentinel Dashboard", layout="centered")

MODEL_PATH = "model.pkl"
MAX_SHAP_ROWS = 1000
REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def show_logo():
    logo_path = "logo/claimsentinel_logo.png"
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        encoded = image_to_base64(image)
        st.markdown(f'''
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <img src="data:image/png;base64,{encoded}" style="width:300px;" />
            </div>
        ''', unsafe_allow_html=True)

def show_title():
    st.markdown('''
        <div style="text-align:center; margin-top: 10px;">
            <h1 style="font-size: 28px; font-weight: 700;">ClaimsSentinel Dashboard</h1>
            <p style="font-size: 16px; font-weight: 400; color: gray;">Smart insights. Safer claims.</p>
        </div>
    ''', unsafe_allow_html=True)

def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(r'[\$,\-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def preprocess_dates(df):
    if "Date of Claim" in df.columns:
        df["Date of Claim"] = pd.to_datetime(df["Date of Claim"], errors="coerce")
        df["Claim Month"] = df["Date of Claim"].dt.month
        df["Claim Day"] = df["Date of Claim"].dt.day
        df["Claim Year"] = df["Date of Claim"].dt.year
        df.drop(columns=["Date of Claim"], inplace=True)
    return df

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

def get_friendly_reason(feature_name):
    mapping = {
        "Claim Amount": "High claim amount",
        "Previous Claims Count": "Multiple prior claims",
        "Claim Location": "High-risk location",
        "Vehicle Make/Model": "High-risk vehicle type",
        "Claim Description": "Suspicious description",
        "Adjuster Notes": "Unusual adjuster notes",
        "Claim Month": "Unusual filing month",
        "Claim Day": "Odd filing day",
        "Claim Year": "Outlier year",
        "Policyholder ID": "Policyholder ID flagged"
    }
    return mapping.get(feature_name, feature_name)

show_logo()
show_title()

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

with st.container():
    st.markdown("<h4>📂 Upload Claims File</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key="predict")

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        df = clean_dataframe(df)
        mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS)
        if any(v is None for v in mapping.values()):
            st.error("Missing required columns: " + ", ".join([k for k, v in mapping.items() if v is None]))
        else:
            df.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
            df = preprocess_dates(df)

            if model:
                X = df.copy()
                df["Potential Fraud"] = model.predict(X)
                df["Top Reason 1"] = ""
                df["Top Reason 2"] = ""

                try:
                    classifier = model.named_steps["classifier"]
                    preprocessor = model.named_steps["preprocessor"]
                    X_shap = X.head(MAX_SHAP_ROWS)
                    explainer = shap.Explainer(classifier, preprocessor.transform(X_shap))
                    shap_vals = explainer(preprocessor.transform(X_shap))

                    for i in range(len(X_shap)):
                        vals = shap_vals[i].values
                        names = shap_vals[i].feature_names
                        top = np.argsort(-np.abs(vals))[:2]
                        df.loc[i, "Top Reason 1"] = get_friendly_reason(names[top[0]])
                        df.loc[i, "Top Reason 2"] = get_friendly_reason(names[top[1]])
                except Exception as e:
                    st.warning(f"SHAP explanation skipped: {e}")

                st.subheader("🔎 Potential Fraud Predictions")
                flagged = df[df["Potential Fraud"] == 1]
                for i, row in flagged.head(100).iterrows():
                    with st.expander(f"🚩 Claim ID: {row['Claim ID']}"):
                        st.write(row.drop("Potential Fraud"))
                        st.write(f"**Reason 1**: {row['Top Reason 1']}")
                        st.write(f"**Reason 2**: {row['Top Reason 2']}")

                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Results")
                    worksheet = writer.sheets["Results"]
                    highlight = writer.book.add_format({"bg_color": "#FFE4E1"})
                    worksheet.conditional_format("J2:J{}".format(len(df)+1), {
                        "type": "cell", "criteria": "==", "value": 1, "format": highlight
                    })
                st.download_button("📥 Download Excel", output.getvalue(), file_name="predictions.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error("Model not found. Please retrain.")

st.markdown("---")

with st.container():
    st.markdown("<h4>🧠 Retrain Model</h4>", unsafe_allow_html=True)
    train_file = st.file_uploader("Upload training data with `Fraud Label`", type=["csv", "xlsx"], key="train")
    model_type = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        train_df = pd.read_csv(train_file) if train_file.name.endswith("csv") else pd.read_excel(train_file)
        train_df = clean_dataframe(train_df)
        mapping = fuzzy_column_map(train_df.columns.tolist(), REQUIRED_COLUMNS + ["Fraud Label"])
        if any(v is None for v in mapping.values()):
            st.error("Missing required columns: " + ", ".join([k for k, v in mapping.items() if v is None]))
        else:
            train_df.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
            train_df = preprocess_dates(train_df)
            X = train_df.drop(columns=["Fraud Label"])
            y = train_df["Fraud Label"]
            numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categoricals = X.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
            ])

            model_cls = LogisticRegression(max_iter=1000) if model_type == "Logistic Regression" else RandomForestClassifier(n_estimators=100, random_state=42)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model_cls)
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, MODEL_PATH)
            st.success(f"{model_type} model retrained and saved.")
