# app.py - ClaimsSentinel Full Version

import streamlit as st
import pandas as pd
import joblib
import os
import base64
import numpy as np
from PIL import Image
from datetime import datetime
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="ClaimsSentinel Dashboard", layout="centered")

# Load local CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/custom.css")

# Branding Logo and Slogan
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    buffered = st.sidebar if hasattr(st, 'sidebar') else st
    st.markdown("""
        <div style='display: flex; justify-content: center; margin-bottom: 1rem;'>
            <img src='data:image/png;base64,""" + base64.b64encode(open(logo_path, "rb").read()).decode() + """' style='height: 100px;'>
        </div>
        <h4 style='text-align:center;'>Smart insights. Safer claims.</h4>
    """, unsafe_allow_html=True)

# Required Columns
required_columns = [
    "Claim Amount", "Previous Claims Count", "Claim Location",
    "Vehicle Make/Model", "Claim Description", "Claim ID",
    "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

# Column Mapping Helper
def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

# Clean input formatting
def clean_dataframe(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.replace("$", "", regex=False)
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = df[col].str.replace("-", "", regex=False)
    return df

# Load model if exists
model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# Upload for Prediction
st.markdown("<h4 style='font-size:22px;'>📂 Upload Claims File</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = clean_dataframe(df)
        st.success("✅ File uploaded.")

        mapping = fuzzy_column_map(df.columns.tolist(), required_columns)
        df = df.rename(columns={v: k for k, v in mapping.items() if v})

        if not all(col in df.columns for col in required_columns):
            st.error("Missing required columns after mapping.")
        elif model is None:
            st.warning("⚠️ No trained model found. Please retrain the model first.")
        else:
            X = df[required_columns]
            preds = model.predict(X)
            df["Fraud Prediction"] = preds

            fraud_df = df[df["Fraud Prediction"] == 1]
            st.markdown("""
                <div style='background-color:#f5f5f5; padding:10px; border-radius:8px;'>
                    📊 <b>Total claims:</b> {0} &nbsp;&nbsp;|&nbsp;&nbsp; ⚠️ <b>Flagged as fraud:</b> {1}
                </div>
            """.format(len(df), len(fraud_df)), unsafe_allow_html=True)

            st.dataframe(fraud_df, use_container_width=True)

            # Export results
            st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"fraud_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            # SHAP per-row explanation
            st.markdown("<br><b>Need more insight?</b> Click below to explain fraud predictions:", unsafe_allow_html=True)
            selected_row = st.selectbox("Select a Claim ID", fraud_df["Claim ID"].tolist())
            if st.button("Explain Why This Is Fraud"):
                try:
                    import shap
                    shap.initjs()
                    preprocessor = model.named_steps["preprocessor"]
                    classifier = model.named_steps["classifier"]
                    X_transformed = preprocessor.transform(X)
                    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
                    X_numeric = np.array(X_dense, dtype=np.float64)
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(X_numeric)

                    idx = df[df["Claim ID"] == selected_row].index[0]
                    shap_df = pd.DataFrame({
                        "Feature": preprocessor.get_feature_names_out(),
                        "SHAP Value": shap_values[1][idx]
                    }).sort_values(by="SHAP Value", ascending=False)

                    st.markdown(f"**Explaining Claim ID:** `{selected_row}`")
                    st.dataframe(shap_df.head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"SHAP error: {e}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Retrain Section
st.markdown("---")
st.markdown("<h4 style='font-size:22px;'>🧠 Retrain Fraud Detection Model</h4>", unsafe_allow_html=True)
with st.expander("📚 Upload labeled training data"):
    train_file = st.file_uploader("Upload file with 'Fraud Label' column", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Random Forest", "Logistic Regression"])

    if train_file:
        try:
            train_df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
            train_df = clean_dataframe(train_df)

            if "Fraud Label" not in train_df.columns:
                st.error("Missing 'Fraud Label' column.")
            else:
                mapping = fuzzy_column_map(train_df.columns.tolist(), required_columns)
                train_df = train_df.rename(columns={v: k for k, v in mapping.items() if v})
                X = train_df[required_columns]
                y = train_df["Fraud Label"]

                numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categoricals = X.select_dtypes(include=["object"]).columns.tolist()

                preprocessor = ColumnTransformer([
                    ("num", StandardScaler(), numeric),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricals)
                ])

                if model_choice == "Logistic Regression":
                    clf = LogisticRegression(max_iter=1000)
                else:
                    clf = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        class_weight='balanced'
                    )

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipeline.fit(X_train, y_train)
                joblib.dump(pipeline, model_path)
                st.success(f"✅ {model_choice} model trained and saved.")
                st.text("📊 Classification Report")
                st.text(classification_report(y_test, pipeline.predict(X_test)))

                if model_choice == "Random Forest":
                    st.markdown("### 🧠 Feature Importance")
                    importances = clf.feature_importances_
                    feature_names = preprocessor.get_feature_names_out()
                    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(importance_df["Feature"], importance_df["Importance"], color="green")
                    ax.set_title("Top Features Impacting Fraud Detection")
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Training failed: {e}")
