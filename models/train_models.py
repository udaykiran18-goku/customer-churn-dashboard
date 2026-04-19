import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(data):

    print("\n🚀 Starting Model Training...")

    # ✅ Required columns check
    required_columns = ["Recency", "Frequency", "Monetary", "Churn"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    # ✅ Handle missing values
    data = data.copy()
    data = data.fillna(data.median(numeric_only=True))

    # 🔍 CHECK DUPLICATES
    print("\n🔍 Checking duplicates:")
    print(data.duplicated().sum())

    # 📊 CHURN DISTRIBUTION
    print("\n📊 Churn Distribution:")
    print(data["Churn"].value_counts())

    # 🔥 ADD NOISE (VERY IMPORTANT FIX)
    noise_factor = 0.05

    data["Recency"] += np.random.normal(0, noise_factor * data["Recency"].std(), len(data))
    data["Frequency"] += np.random.normal(0, noise_factor * data["Frequency"].std(), len(data))
    data["Monetary"] += np.random.normal(0, noise_factor * data["Monetary"].std(), len(data))

    # Features & Target
    X = data[["Recency", "Frequency", "Monetary"]]
    y = data["Churn"]

    # 🔥 STRATIFIED SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 🔥 BALANCED MODEL
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.4f}")

    # 📊 Detailed Metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/churn_model.joblib"
    joblib.dump(pipeline, model_path)

    print(f"\n💾 Model saved at: {model_path}")

    return pipeline


# ✅ MAIN
if __name__ == "__main__":
    print("📂 Loading dataset...")

    try:
        df = pd.read_csv("datasets/processed_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ Dataset not found. Check path: datasets/processed_data.csv")

    train_model(df)