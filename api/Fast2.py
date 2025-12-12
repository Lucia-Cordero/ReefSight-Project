
from project_logic.predict import load_image_model_trained, load_tabular_model_trained, predict_tabular, predict_image
from project_logic.preprocessing import TabularInput
from fastapi import FastAPI, UploadFile, File, HTTPException, Form    # --- ADD ON: Form needed for multi-modal uploads
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import traceback


# =========================================
# FASTAPI INIT
# =========================================
app = FastAPI()
print("✅ FastAPI initialized")


# =========================================
# CORS (Required for Streamlit frontend)
# =========================================
# --- ADD ON: Allow Streamlit → API requests ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# MODEL LOADING
# =========================================
# --- ADD ON: Safe model loading with debug prints ---
try:
    app.state.image_model = load_image_model_trained()
    print("✅ Image model loaded")
except Exception as e:
    app.state.image_model = None
    print("❌ Failed to load image model:", e)
    traceback.print_exc()

try:
    app.state.tabular_model = load_tabular_model_trained()
    print("✅ Tabular model loaded")
except Exception as e:
    app.state.tabular_model = None
    print("❌ Failed to load tabular model:", e)
    traceback.print_exc()

MODEL_READY = bool(app.state.image_model or app.state.tabular_model)


# =========================================
# ROOT ENDPOINT
# =========================================
@app.get("/")
def root():
    return {
        "message": "Hi, the API is running! Welcome to ReefSight.",
        "model_ready": MODEL_READY,
    }


# =========================================
# IMAGE-ONLY PREDICTION ENDPOINT
# =========================================
@app.post("/predict/image")
async def predict_image_api(image_file: UploadFile = File(...)):

    # Validate content type
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image_file.read()

    if not app.state.image_model:
        raise HTTPException(
            status_code=503,
            detail="Image model is not available",
        )

    prediction = predict_image(model=app.state.image_model, image_bytes=image_bytes)

    return {
        "prediction": prediction,
        "inputs": {"filename": image_file.filename},
        "model_ready": True,
    }


# =========================================
# TABULAR-ONLY PREDICTION ENDPOINT
# =========================================
@app.post("/predict/tabular")
def predict_tabular_api(payload: TabularInput):

    # Validate presence of tabular data
    if not payload.tabular_data:
        raise HTTPException(status_code=400, detail="Tabular data is required.")

    X_pred = pd.DataFrame([payload.tabular_data])

    required_columns = [
        "Distance_to_Shore",
        "Turbidity",
        "Cyclone_Frequency",
        "Depth_m",
        "ClimSST",
        "Temperature_Kelvin",
        "Temperature_Kelvin_Standard_Deviation",
        "Windspeed",
    ]

    # Check for missing fields
    missing = [c for c in required_columns if c not in X_pred.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required tabular features: {missing}",
        )

    if not app.state.tabular_model:
        raise HTTPException(
            status_code=503,
            detail="Tabular model is not available",
        )

    prediction = predict_tabular(app.state.tabular_model, X_pred)

    return {
        "prediction": prediction,
        "inputs": X_pred.to_dict(orient="records")[0],
        "model_ready": True,
    }


# =========================================
# UNIVERSAL MULTI-MODAL ENDPOINT
# =========================================
@app.post("/predict")
async def predict_multi_modal(
    payload: str = Form(...),        # JSON payload (string)
    image_file: UploadFile = File(None),  # optional image file
):

    # --- ADD ON: Safe JSON parsing ---
    try:
        payload = json.loads(payload)
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    prediction_type = payload.get("prediction_type")
    tabular_data = payload.get("tabular_data", {})

    # ---------------------------------
    # VALIDATION BY MODE
    # ---------------------------------
    if prediction_type == "Multi-Modal Fusion (Image + Data)":
        if not image_file:
            raise HTTPException(status_code=400, detail="Fusion requires an image.")
        if not tabular_data:
            raise HTTPException(status_code=400, detail="Fusion requires tabular data.")

    if prediction_type in ("Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"):
        if not tabular_data:
            raise HTTPException(status_code=400, detail="Tabular data is required.")

    if prediction_type == "Image-Only (VGG Augmented)" and not image_file:
        raise HTTPException(
            status_code=400,
            detail="Image-only prediction requires an image file.",
        )

    # ---------------------------------
    # IMAGE PROCESSING (if supplied)
    # ---------------------------------
    image_prediction = None

    if image_file:
        if not image_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        if not app.state.image_model:
            raise HTTPException(status_code=503, detail="Image model unavailable")

        img_bytes = await image_file.read()
        image_prediction = predict_image(app.state.image_model, img_bytes)

    # ---------------------------------
    # TABULAR PROCESSING (if supplied)
    # ---------------------------------
    tabular_prediction = None

    if tabular_data:
        if not app.state.tabular_model:
            raise HTTPException(status_code=503, detail="Tabular model unavailable")

        X_pred = pd.DataFrame([tabular_data])
        tabular_prediction = predict_tabular(app.state.tabular_model, X_pred)

    # ---------------------------------
    # PREDICTION FUSION LOGIC
    # ---------------------------------
    if prediction_type == "Multi-Modal Fusion (Image + Data)":
        combined = round((image_prediction + tabular_prediction) / 2, 3)

    elif prediction_type == "Image-Only (VGG Augmented)":
        combined = image_prediction

    elif prediction_type in ("Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"):
        combined = tabular_prediction

    else:
        combined = tabular_prediction or image_prediction

    # Return result
    return {
        "status": "success",
        "prediction_type": prediction_type,
        "predicted_bleaching_risk": combined,
        "tabular_data_used": tabular_data,
        "image_processed": bool(image_file),
        "model_ready": MODEL_READY,
    }
