import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import io
import shap
from pathlib import Path
from datetime import datetime
from PIL import Image
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

# Logo and branding
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def show_logo():
    logo_path = "logo/claimsentinel_logo.png"
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        encoded = image_to_base64(image)
        st.markdown(f"""
            <style>
                .logo-container img:hover {{
                    transform: scale(1.07);
                    transition: transform 0.3s ease;
                }}
            </style>
            <div class='logo-container' style='display: flex; justify-content: center; margin: 2rem 0;'>
                <img src='data:image/png;base64,{encoded}' style='width:400px;' />
            </div>
        """, unsafe_allow_html=True)

show_logo()

# Expected input columns
required_columns = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

def clean_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
    return df

def get_preprocessor(X):
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categoricals = X.select_dtypes(include=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
    ])

model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# --- Upload for Prediction ---
st.markdown("<h4 style='font-size:22px; font-weight:600;'>📂 Upload CSV or Excel File</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = clean_dataframe(df)
        st.success("✅ File uploaded.")

        mapping = fuzzy_column_map(df.columns.tolist(), required_columns)
        df = df.rename(columns={v: k for k, v in mapping.items() if v})
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Missing required columns.")
        else:
            X = df[required_columns]
            if model:
                preds = model.predict(X)
                df["Fraud Prediction"] = preds

                st.subheader("🔎 Predictions")
                fraud_df = df[df["Fraud Prediction"] == 1].copy()

                if fraud_df.empty:
                    st.info("No fraudulent claims detected.")
                else:
                    for i, row in fraud_df.iterrows():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.write(f"**{row['Claim ID']}** | {row['Claim Description']} | {row['Claim Location']}")
                        with col2:
                            if st.button("Explain", key=f"explain_{i}"):
                                try:
                                    classifier = model.named_steps["classifier"]
                                    preprocessor = model.named_steps["preprocessor"]
                                    X_transformed = preprocessor.transform(X)
                                    explainer = shap.Explainer(classifier, X_transformed)
                                    shap_values = explainer(X_transformed)
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                    st.subheader(f"🧠 SHAP Explanation for Claim ID: {row['Claim ID']}")
                                    shap.plots.waterfall(shap_values[i], max_display=10)
                                    st.pyplot()
                                except Exception as e:
                                    st.error(f"SHAP error: {e}")

                st.markdown(f"""
                    <div style='padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
                        📊 <b>Total claims:</b> {len(df)} &nbsp;&nbsp;|&nbsp;&nbsp; ⚠️ <b>Flagged as fraud:</b> {df['Fraud Prediction'].sum()}
                    </div>
                """, unsafe_allow_html=True)

                st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            else:
                st.error("⚠️ No trained model found. Please retrain below.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Retrain Section ---
st.markdown("---")
st.markdown("<h4 style='font-size:22px; font-weight:600;'>🧠 Retrain Fraud Detection Model</h4>", unsafe_allow_html=True)

with st.expander("📚 Upload labeled data to retrain the model"):
    train_file = st.file_uploader("Upload training file with `Fraud Label`", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        try:
            train_df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
            train_df = clean_dataframe(train_df)
            if "Fraud Label" not in train_df.columns:
                st.error("Missing 'Fraud Label' column.")
            elif pd.api.types.is_datetime64_any_dtype(train_df["Fraud Label"]):
                st.error("❌ 'Fraud Label' column contains dates.")
            else:
                mapping = fuzzy_column_map(train_df.columns.tolist(), required_columns)
                train_df = train_df.rename(columns={v: k for k, v in mapping.items() if v})
                X = train_df[required_columns]
                y = train_df["Fraud Label"]

                pipeline = Pipeline([
                    ("preprocessor", get_preprocessor(X)),
                    ("classifier", LogisticRegression(max_iter=1000) if model_choice == "Logistic Regression" else RandomForestClassifier(n_estimators=100, random_state=42))
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                joblib.dump(pipeline, model_path)
                model = pipeline

                st.success("✅ Model trained and saved.")
                st.text("📊 Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.markdown(f"""
                    <div style='padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
                        ✅ <b>Model trained on:</b> {len(train_df)} claims
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Training failed: {e}")

