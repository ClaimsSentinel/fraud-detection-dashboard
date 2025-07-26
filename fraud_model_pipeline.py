
import pandas as pd
import numpy as np
import random
from faker import Faker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load synthetic data
df = pd.read_csv("synthetic_insurance_claims.csv")

# Keep relevant columns
df_model = df[[
    "Claim Amount",
    "Previous Claims Count",
    "Claim Location",
    "Vehicle Make/Model",
    "Claim Description",
    "Fraudulent"
]].copy()

# Split features and label
X = df_model.drop("Fraudulent", axis=1)
y = df_model["Fraudulent"]

# Define transformers
numeric_features = ["Claim Amount", "Previous Claims Count"]
numeric_transformer = StandardScaler()

categorical_features = ["Claim Location", "Vehicle Make/Model"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

text_features = "Claim Description"
text_transformer = TfidfVectorizer(max_features=50)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("txt", text_transformer, text_features)
    ]
)

# Pipeline with classifier
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance visualization
classifier = pipeline.named_steps["classifier"]
preprocessor = pipeline.named_steps["preprocessor"]

# Extract feature names
num_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.transformers_[1][1]
cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
txt_features = preprocessor.transformers_[2][1].get_feature_names_out()

all_feature_names = list(num_features) + list(cat_features) + list(txt_features)
importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(15)

# Plot top 15
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Top 15 Most Important Features in Fraud Detection")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
