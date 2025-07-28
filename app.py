import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
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

st.set_page_config(page_title="ClaimsSentinel: Potential Fraud Dashboard", layout="centered")

def show_logo():
    logo_path = "logo/claimsentinel_logo.png"
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        st.image(image, width=300)

show_logo()
st.title("🛡️ ClaimsSentinel Dashboard")
st.markdown("**Smart insights. Safer claims.**")

MAX_SHAP_ROWS = 1000
MODEL_PATH = "model.pkl"
REQUIRED_COLUMNS = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(r'[\$,\-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                continue
    return df

def preprocess_dates(df):
    if "Date of Claim" in df.columns:
        df["Date of Claim"] = pd.to_datetime(df["Date of Claim"], errors="coerce")
        df["Claim Month"] = df["Date of Claim"].dt.month
        df["Claim Day"] = df["Date of Claim"].dt.day
        df["Claim Year"] = df["Date of Claim"].dt.year
        df = df.drop(columns=["Date of Claim"])
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
        "Claim Location": "Filed in high-risk location",
        "Vehicle Make/Model": "High-risk vehicle type",
        "Claim Description": "Suspicious claim description",
        "Adjuster Notes": "Suspicious adjuster notes",
        "Claim Month": "Claim filed in unusual month",
        "Claim Day": "Unusual day of claim",
        "Claim Year": "Year of claim appears abnormal",
        "Policyholder ID": "Flagged policyholder"
    }
    return mapping.get(feature_name, feature_name)

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

st.header("📂 Upload Claims File")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
    df = clean_dataframe(df)
    mapping = fuzzy_column_map(df.columns.tolist(), REQUIRED_COLUMNS)
    unmapped = [k for k, v in mapping.items() if v is None]
    if unmapped:
        st.error(f"❌ Could not map required columns: {', '.join(unmapped)}")
    else:
        df = df.rename(columns={v: k for k, v in mapping.items() if v})
        df = preprocess_dates(df)
        if "Fraud Label" in df.columns:
            df = df.drop(columns=["Fraud Label"])

        if model:
            X = df.copy()
            df["Potential Fraud Prediction"] = model.predict(X)
            df["Top Reason 1"] = ""
            df["Top Reason 2"] = ""
            shap_success = False

            try:
                classifier = model.named_steps["classifier"]
                preprocessor = model.named_steps["preprocessor"]
                X_shap = X.head(MAX_SHAP_ROWS) if len(X) > MAX_SHAP_ROWS else X
                X_transformed = preprocessor.transform(X_shap)
                explainer = shap.Explainer(classifier, X_transformed)
                shap_values = explainer(X_transformed)

                for i in range(len(X_shap)):
                    row_vals = shap_values[i].values
                    feature_names = shap_values[i].feature_names
                    top_idx = np.argsort(-np.abs(row_vals))[:2]
                    df.loc[i, "Top Reason 1"] = get_friendly_reason(feature_names[top_idx[0]])
                    df.loc[i, "Top Reason 2"] = get_friendly_reason(feature_names[top_idx[1]])
                shap_success = True
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")

            st.subheader("🔎 Potential Fraud Predictions")
            for i, row in df[df["Potential Fraud Prediction"] == 1].head(100).iterrows():
                with st.expander(f"🚩 Claim ID: {row['Claim ID']}"):
                    st.write(row.drop("Potential Fraud Prediction"))
                    st.write(f"**Reason 1**: {row['Top Reason 1']}")
                    st.write(f"**Reason 2**: {row['Top Reason 2']}")

            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Predictions")
                workbook = writer.book
                worksheet = writer.sheets["Predictions"]
                red = workbook.add_format({"bg_color": "#FFE4E1", "font_color": "#000000"})
                worksheet.conditional_format("J2:J{}".format(len(df)+1), {
                    "type": "cell", "criteria": "==", "value": 1, "format": red
                })
                if not shap_success:
                    worksheet.write(len(df)+2, 0, "SHAP explanations were not available.")
                writer.close()

            st.download_button("📥 Download Report (.xlsx)",
                            data=output.getvalue(),
                            file_name=f"potential_fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.error("⚠️ No trained model found. Retrain below.")

st.markdown("---")
st.subheader("🧠 Retrain Model")
train_file = st.file_uploader("Upload labeled training data", type=["csv", "xlsx"], key="train")
model_choice = st.radio("Select model", ["Logistic Regression", "Random Forest"])

if train_file:
    try:
        train_df = pd.read_csv(train_file) if train_file.name.endswith("csv") else pd.read_excel(train_file)
        train_df = clean_dataframe(train_df)
        mapping = fuzzy_column_map(train_df.columns.tolist(), REQUIRED_COLUMNS + ["Fraud Label"])
        missing = [k for k, v in mapping.items() if v is None]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            train_df = train_df.rename(columns={v: k for k, v in mapping.items() if v})
            train_df = preprocess_dates(train_df)
            X = train_df.drop(columns=["Fraud Label"])
            y = train_df["Fraud Label"]
            numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categoricals = X.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
            ])

            clf = LogisticRegression(max_iter=1000) if model_choice == "Logistic Regression" else RandomForestClassifier(n_estimators=100, random_state=42)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, MODEL_PATH)
            st.success(f"{model_choice} trained and saved.")
    except Exception as e:
        st.error(f"Training failed: {e}")
