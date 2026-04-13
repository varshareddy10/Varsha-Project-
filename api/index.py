from fastapi import FastAPI
import numpy as np
import joblib
import os

app = FastAPI()

# Safe loading
model = None
scaler = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "nephrolithiasis_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
except:
    print("Model files not found, running in dummy mode")

@app.get("/")
def home():
    return {"message": "Kidney Stone Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)

        # ✅ If model exists → real prediction
        if model is not None and scaler is not None:
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][1]

        # ⚠️ If model missing → dummy prediction
        else:
            pred = 1 if np.mean(features) > 0.5 else 0
            prob = float(np.mean(features))

        return {
            "prediction": int(pred),
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}
