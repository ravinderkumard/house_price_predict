from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
import os


MODEL_PATH = "models/random_forest_model.joblib"
FEATURE_PATH = "models/feature_names.joblib"

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    print("Model loaded successfully")
except:
    print("Model not found. Please train the model first.")
    model = None
    feature_names = []

# Define input schema
class HouseFeatures(BaseModel):
    square_feet: int
    num_bedrooms: int
    num_bathrooms: int
    year_built: int
    location_quality: int  # 1-10
    
    class Config:
        schema_extra = {
            "example": {
                "square_feet": 2000,
                "num_bedrooms": 3,
                "num_bathrooms": 2,
                "year_built": 2010,
                "location_quality": 7
            }
        }

class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]


# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on features",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to dataframe
    input_data = pd.DataFrame([features.dict()])
    
    # Ensure column order matches training
    input_data = input_data[feature_names]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "USD",
        "features": features.dict()
    }

@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to list of dictionaries
    houses_data = [house.dict() for house in request.houses]
    
    # Create DataFrame
    input_data = pd.DataFrame(houses_data)
    input_data = input_data[feature_names]
    
    # Make predictions
    predictions = model.predict(input_data)
    
    results = []
    for i, (house, pred) in enumerate(zip(request.houses, predictions)):
        results.append({
            "house_id": i,
            "predicted_price": round(float(pred), 2),
            "features": house.dict()
        })
    
    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)