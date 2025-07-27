# app.py - Streamlit Dashboard with Cleaned Prediction and SHAP Explanations

import streamlit as st
import pandas as pd
import joblib
import os
import base64
from difflib import get_close_matches
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image
import io
import shap

# Set up page
st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

# Load custom styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("assets/custom.css")

# Convert image to base64
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Show ClaimsSentinel logo
def show_logo():
    logo_path = "logo/claimsentinel_logo.png"
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

# Required columns
required_columns = [
    "Claim Amount", "Previous Claims Count", "Claim Location", "Vehicle Make/Model",
    "Claim Description", "Claim ID", "Adjuster Notes", "Date of Claim", "Policyholder ID"
]

# Clean uploaded dataframe
def clean_dataframe(df):
    df = df.dropna(subset=required_columns)
    df = df[required_columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(float)
            except:
                continue
    return df

# Fuzzy match columns
def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

# Load trained model
model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# File uploader section
st.markdown("<h4 style='font-size:22px; font-weight:600;'>📂 Upload CSV or Excel File</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("✅ File uploaded.")

        mapping = fuzzy_column_map(df.columns.tolist(), required_columns)
        unmapped = [k for k, v in mapping.items() if v is None]
        if unmapped:
            st.warning(f"⚠️ Could not map: {', '.join(unmapped)}")
        else:
            df = df.rename(columns={v: k for k, v in mapping.items() if v})
            if all(col in df.columns for col in required_columns):
                df = clean_dataframe(df)
                X = df[required_columns]

                if model:
                    preds = model.predict(X)
                    df["Fraud Prediction"] = preds

                    st.subheader("🔎 Predictions")
                    st.dataframe(df[["Claim ID", "Fraud Prediction"]].head(10))

                    st.markdown(f"""
                        <div style='padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
                            📊 <b>Total claims:</b> {len(df)} &nbsp;&nbsp;|&nbsp;&nbsp; ⚠️ <b>Flagged as fraud:</b> {df['Fraud Prediction'].sum()}
                        </div>
                    """, unsafe_allow_html=True)

                    # SHAP explain button
                    if st.button("🔍 Explain Why This is Fraud"):
                        explainer = shap.Explainer(model.named_steps['classifier'], model.named_steps['preprocessor'].transform(X))
                        shap_values = explainer(X)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot(shap.plots.beeswarm(shap_values, max_display=10))

                    # Download
                    st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                else:
                    st.error("⚠️ No trained model found. Please retrain below.")
            else:
                st.error("❌ Missing required columns after mapping.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Retrain section
st.markdown("---")
st.markdown("<h4 style='font-size:22px; font-weight:600;'>🧠 Retrain Fraud Detection Model</h4>", unsafe_allow_html=True)

with st.expander("📚 Upload labeled data to retrain the model"):
    train_file = st.file_uploader("Upload training file with `Fraud Label`", type=["csv", "xlsx"], key="train")
    model_choice = st.radio("Choose model", ["Logistic Regression", "Random Forest"])

    if train_file:
        try:
            train_df = pd.read_csv(train_file) if train_file.name.endswith(".csv") else pd.read_excel(train_file)
            if "Fraud Label" not in train_df.columns:
                st.error("Missing 'Fraud Label' column.")
            else:
                mapping = fuzzy_column_map(train_df.columns.tolist(), required_columns)
                missing = [k for k, v in mapping.items() if v is None]
                if missing:
                    st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
                else:
                    train_df = train_df.rename(columns={v: k for k, v in mapping.items() if v})
                    X = train_df[required_columns]
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
                    y_pred = pipeline.predict(X_test)

                    joblib.dump(pipeline, model_path)
                    model = pipeline

                    st.success(f"✅ {model_choice} model trained and saved.")
                    st.text("📊 Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    st.markdown(f"""
                        <div style='padding: 10px; background-color: #f5f5f5; border-radius: 10px;'>
                            ✅ <b>Model trained on:</b> {len(train_df)} claims
                        </div>
                    """, unsafe_allow_html=True)

                    if model_choice == "Random Forest":
                        st.markdown("### 🧠 Top Influential Features")
                        importances = clf.feature_importances_
                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                        importance_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False).head(10)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
                        ax.set_xlabel("Importance Score")
                        ax.set_title("Top Features Influencing Fraud Prediction")
                        st.pyplot(fig)

                        st.caption("🔎 This chart shows which features most influenced the fraud prediction model after training. Higher scores indicate more weight in fraud detection.")
                    else:
                        st.info("ℹ️ Feature importance is only available for Random Forest models.")
        except Exception as e:
            st.error(f"Training failed: {e}")
