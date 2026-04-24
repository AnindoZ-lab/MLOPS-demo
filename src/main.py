from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from pathlib import Path
import json

# Define request/response models
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        min_items=4, 
        max_items=4,
        description="List of 4 float features: [sepal_length, sepal_width, petal_length, petal_width]"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="API for classifying Iris flowers using Random Forest model",
    version="1.0.0"
)

# Global variables
model = None
model_info = {}
class_names = ['setosa', 'versicolor', 'virginica']

@app.on_event("startup")
async def load_model():
    """Load the trained model and metrics on startup"""
    global model, model_info
    
    model_path = Path("../models/model.joblib")
    metrics_path = Path("../models/metrics.json")
    
    # Load model
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    else:
        print(f"✗ Model not found at {model_path}")
        print("  Run 'python src/train.py' first to train and save the model")
    
    # Load model info from metrics
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                model_info = metrics.get("model_info", {})
                print(f"✓ Model info loaded from {metrics_path}")
        except Exception as e:
            print(f"✗ Error loading metrics: {e}")
            model_info = {
                "model_type": "RandomForestClassifier",
                "classes": class_names,
                "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            }
    else:
        model_info = {
            "model_type": "RandomForestClassifier",
            "classes": class_names,
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API health status"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict Iris species from 4 features
    
    Features order:
    - sepal_length (cm): 4.0 - 8.0
    - sepal_width (cm): 2.0 - 4.5
    - petal_length (cm): 1.0 - 7.0
    - petal_width (cm): 0.1 - 2.5
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using 'python src/train.py'"
        )
    
    # Validate input features (optional validation)
    features = np.array(request.features).reshape(1, -1)
    
    # Validate feature dimensions
    if features.shape[1] != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 4 features, got {features.shape[1]}"
        )
    
    try:
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities
        probabilities = model.predict_proba(features)[0].tolist()
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        # Map prediction to class name
        class_name = class_names[prediction] if prediction < len(class_names) else "unknown"
        
        return PredictionResponse(
            prediction=int(prediction),
            class_name=class_name,
            probabilities=probabilities,
            confidence=round(confidence, 4)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Predict multiple Iris samples in batch
    
    Returns list of predictions for each input sample
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Extract features from all requests
        features_list = [req.features for req in requests]
        features_array = np.array(features_list)
        
        # Batch predictions
        predictions = model.predict(features_array).tolist()
        probabilities = model.predict_proba(features_array).tolist()
        
        # Prepare responses
        responses = []
        for i, pred in enumerate(predictions):
            class_name = class_names[pred] if pred < len(class_names) else "unknown"
            confidence = max(probabilities[i])
            
            responses.append({
                "prediction": pred,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "index": i
            })
        
        return {
            "batch_size": len(requests),
            "predictions": responses
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_loaded": True,
        "model_info": model_info,
        "class_names": class_names,
        "num_classes": len(class_names),
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "num_features": 4
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
