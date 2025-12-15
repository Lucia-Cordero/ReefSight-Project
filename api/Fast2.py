from project_logic.predict import (
    load_image_model_trained,
    load_tabular_model_trained,
    predict_tabular,
    predict_image
)
from project_logic.preprocessing import TabularInput
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
import json
import random
from datetime import datetime

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="ReefSight Centralized Prediction API",
    description="Unified endpoint for multi-modal coral bleaching prediction."
)
print("✅ Fast API initialized")

# -----------------------------
# Load pre-trained models
# -----------------------------
app.state.image_model = load_image_model_trained()
app.state.tabular_model = load_tabular_model_trained()
MODEL_READY = True

# -----------------------------
# Request schema
# -----------------------------
class PredictionRequest(BaseModel):
    prediction_type: str
    lat: float
    lon: float
    date: str
    override_data: Optional[Dict[str, float]] = None

# -----------------------------
# NOAA Data Fetch (backend)
# -----------------------------
def fetch_noaa_data(date_str: str, lat: float, lon: float) -> Dict[str, float]:
    """
    Backend-owned NOAA data fetch.
    Currently simulated with small deterministic variation for temperature.
    """
    temp_variation = random.uniform(-1.5, 1.5)
    return {
        "Distance_to_Shore": 10.0,
        "Turbidity": 2.5,
        "Cyclone_Frequency": 0.1,
        "Depth_m": 15.0,
        "ClimSST": 26.0,
        "Temperature_Kelvin": 300.0 + temp_variation,
        "Temperature_Kelvin_Standard_Deviation": 1.5,
        "Windspeed": 5.0,
    }

# -----------------------------
# Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Hi, The API is running! Welcome to ReefSight"}

# -----------------------------
# Unified Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict_unified(
    payload: str = Form(...),
    image_file: Optional[UploadFile] = File(None)
):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Models not ready")

    # -----------------------------
    # Parse payload
    # -----------------------------
    try:
        request_data = PredictionRequest.parse_obj(json.loads(payload))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    mode = request_data.prediction_type

    # -----------------------------
    # Image requirement enforcement
    # -----------------------------
    if "Image" in mode and image_file is None:
        raise HTTPException(
            status_code=400,
            detail="Image file required for selected prediction mode"
        )

    # -----------------------------
    # Tabular feature resolution
    # -----------------------------
    tabular_features = None
    if "Image-Only" not in mode:
        if request_data.override_data:
            tabular_features = request_data.override_data
        else:
            tabular_features = fetch_noaa_data(
                request_data.date,
                request_data.lat,
                request_data.lon
            )

    X_pred = pd.DataFrame([tabular_features]) if tabular_features else None

    # -----------------------------
    # Prediction routing
    # -----------------------------
    prediction = {}
    if mode == "Tabular-Only" or mode == "Manual Data Entry Only (No NOAA Pull)":
        prediction = predict_tabular(
            model=app.state.tabular_model,
            X_pred=X_pred
        )

    elif mode == "Image-Only (VGG Augmented)":
        image_bytes = await image_file.read()
        prediction = predict_image(
            model=app.state.image_model,
            image_bytes=image_bytes
        )

    elif mode == "Multi-Modal Fusion (Image + Data)":
        image_bytes = await image_file.read()
        tabular_pred = predict_tabular(app.state.tabular_model, X_pred)
        image_pred = predict_image(app.state.image_model, image_bytes)

        # Deterministic fusion
        risk_tab = tabular_pred.get("predicted_bleaching_risk", 50.0)
        risk_img = image_pred.get("predicted_bleaching_risk", 50.0)
        final_risk = (0.4 * risk_tab) + (0.6 * risk_img)

        prediction = {
            "predicted_bleaching_risk": round(final_risk, 1),
            "classification": "Bleached" if final_risk > 50 else "Unbleached",
            "fusion_detail": {
                "tabular_risk": risk_tab,
                "image_risk": risk_img,
            },
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown prediction type: {mode}")

    # -----------------------------
    # Response
    # -----------------------------
    return {
        "status": "success",
        "mode_used": mode,
        "input_data": {
            "latitude": request_data.lat,
            "longitude": request_data.lon,
            "date": request_data.date,
            "tabular_features_used": tabular_features,
        },
        "prediction": prediction,
    }

