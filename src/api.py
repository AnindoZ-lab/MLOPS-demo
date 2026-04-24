from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="MLOps API")

# Load model at startup
MODEL_PATH = Path("../models/model.joblib")
model = None

class PredictionInput(BaseModel):
    features: list

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        print("Model not found. Train first with src/train.py")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    
    return PredictionOutput(prediction=int(prediction), probability=float(probability))
