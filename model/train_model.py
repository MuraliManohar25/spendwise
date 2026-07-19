"""Train RandomForest overspend-risk classifier and save artifacts."""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spend_model.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_cols.json")

FEATURE_COLS = [
    "monthly_budget",
    "days_into_month",
    "rent_or_pg_spent_so_far",
    "avg_daily_spend",
    "std_daily_spend",
    "trend_slope",
    "max_daily_spend",
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS]
y = df["overspend_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_weight = {"SAFE": 1.0, "CAUTION": 1.2, "HIGH_RISK": 2.0}

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight=class_weight,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix (rows=true, cols=predicted):")
labels = sorted(y.unique())
print(confusion_matrix(y_test, y_pred, labels=labels))
print("Labels order:", labels)

print("\nFeature Importances:")
for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {col}: {imp:.4f}")

joblib.dump(model, MODEL_PATH)
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Feature columns saved to {FEATURES_PATH}")
