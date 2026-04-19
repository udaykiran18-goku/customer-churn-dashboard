from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("models/churn_model.joblib")


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running 🚀"}


@app.post("/predict")
def predict(data: dict):

    recency = data["recency"]
    frequency = data["frequency"]
    monetary = data["monetary"]

    df = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    # ML prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # 🔥 BUSINESS RULE OVERRIDE
    if recency > 90 and frequency <= 2:
        prediction = 1
        probability = max(probability, 0.7)

    # Final output
    result = "Churn Likely" if prediction == 1 else "Active Customer"
    confidence = round(probability * 100, 2)

    if confidence >= 70:
        risk_level = "High Risk"
    elif confidence >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "prediction": result,
        "confidence": confidence,
        "risk_level": risk_level
    }


@app.get("/feature-importance")
def feature_importance():

    try:
        rf_model = model.named_steps["model"]
        importance = rf_model.feature_importances_

        return {
            "Recency": float(importance[0]),
            "Frequency": float(importance[1]),
            "Monetary": float(importance[2])
        }

    except Exception as e:
        return {"error": str(e)}