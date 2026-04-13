from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load("nephrolithiasis_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Kidney Stone Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to array
        features = np.array(list(data.values())).reshape(1, -1)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}
