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
from PIL import Image
import io

# Must be first Streamlit call
st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

# Inject custom CSS for colors, fonts, and hover animation
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/custom.css")

# Show logo only, centered with hover effect and larger size
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

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
            <img src='data:image/png;base64,{encoded}' style='width:260px;' />
        </div>
    """, unsafe_allow_html=True)

show_logo()

# Required columns
required_columns = [
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

# Fuzzy matching for column names
def fuzzy_column_map(uploaded_cols, required_cols, cutoff=0.7):
    mapping = {}
    for req_col in required_cols:
        match = get_close_matches(req_col, uploaded_cols, n=1, cutoff=cutoff)
        mapping[req_col] = match[0] if match else None
    return mapping

# Load model if available
model_path = "model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

# File uploader
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
                X = df[required_columns]

                if model:
                    preds = model.predict(X)
                    df["Fraud Prediction"] = preds

                    total = len(df)
                    fraud_count = df["Fraud Prediction"].sum()
                    fraud_percent = round((fraud_count / total) * 100, 2)

                    st.success(f"📊 **{total} total claims analyzed**")
                    st.warning(f"⚠️ **{fraud_count} flagged as potential fraud ({fraud_percent}%)**")

                    st.subheader("🔎 Predictions")
                    st.dataframe(df[["Claim ID", "Fraud Prediction"]].head(10))

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
st.markdown("<h2 style='font-size:28px; font-weight:700; color:#1A237E;'>🧠 Retrain Fraud Detection Model</h2>", unsafe_allow_html=True)

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

                    st.success(f"📊 {model_choice} model trained successfully on {len(train_df)} labeled claims.")
                    st.text("📊 Classification Report")
                    st.text(classification_report(y_test, y_pred))

        except Exception as e:
            st.error(f"Training failed: {e}")
