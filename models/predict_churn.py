import joblib
import pandas as pd


# Load model
model = joblib.load("models/churn_model.pkl")


def predict_churn(recency, frequency, monetary):

    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "Churn Likely"
    else:
        return "Active Customer"


# Test
if __name__ == "__main__":
    result = predict_churn(300, 1, 200)
    print("Prediction:", result)