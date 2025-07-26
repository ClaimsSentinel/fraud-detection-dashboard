from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

def pipeline():
    data = pd.DataFrame({
        "Claim Description": [
            "Car accident on highway with multiple injuries",
            "Minor fender bender in parking lot",
            "Suspicious fire damage claim",
            "Water pipe burst due to freezing"
        ],
        "Fraudulent": [1, 0, 1, 0]
    })

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(data["Claim Description"], data["Fraudulent"])
    joblib.dump(model, "fraud_model.pkl")

    return model