
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("synthetic_insurance_claims.csv")

# Preprocess
X = data["Claim Description"]
y = data["Fraudulent"]  # assuming this is the target column

# Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# Train model
pipeline.fit(X, y)

# Save model to file
joblib.dump(pipeline, "fraud_model.pkl")
