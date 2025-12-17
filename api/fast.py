from project_logic.predict import load_image_model_trained, load_tabular_model_trained, predict_tabular, predict_image
from project_logic.preprocessing import FullTabularInput, MinimalTabularInput
from project_logic.tabular_preproc import *
from project_logic.tabular_fetch import build_X_pred

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
import pandas as pd



# -----------------------------------------------------------------------
#                           INITIALIZE FAST API
# -----------------------------------------------------------------------

app = FastAPI()
print('✅ Fast API initialized')

# -----------------------------------------------------------------------
#                           PRELOAD TRAINED MODELS
# -----------------------------------------------------------------------
app.state.image_model = load_image_model_trained()
app.state.tabular_model = load_tabular_model_trained()


# -----------------------------------------------------------------------
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
'''

# -----------------------------------------------------------------------
#                               ROOT ENDPOINT
# -----------------------------------------------------------------------

@app.get("/")
def root():
    return {
        'message': "Hi, The API  is running! Welcome to ReefSight"
    }


# -----------------------------------------------------------------------
#                          IMAGE PREDICT ENDPOINT
# -----------------------------------------------------------------------

@app.post("/predict/image")
async def predict_image_api(image_file: UploadFile= File(...)):

    # Make sure it's an image
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read uploaded image bytes
    image_bytes = await image_file.read()


    # Call prediction function "predict_image"
    model = app.state.image_model
    prediction = predict_image(model=model, image_bytes=image_bytes)

    return {
        "prediction": prediction,
        "inputs": {"filename": image_file.filename},
        "model_ready": True
    }

# -----------------------------------------------------------------------
#                          TABULAR PREDICT ENDPOINT
# -----------------------------------------------------------------------

@app.post("/predict/tabular")
def predict_tabular_api(
    payload: Union[FullTabularInput, MinimalTabularInput]
):

    model = app.state.tabular_model

    # CASE 1 — Full feature vector (16 features)
    if isinstance(payload, FullTabularInput):
        X_pred = pd.DataFrame([payload.dict()])
        source = "full_input"

    # CASE 2 — Minimal input (lat, lon, date)
    else:
        X_pred = build_X_pred(
            lat=payload.latitude,
            lon=payload.longitude,
            dt=payload.observation_date
        )
        source = "auto_enriched"

    prediction = predict_tabular(model=model, X_pred=X_pred)

    return {
        "prediction": prediction,
        "inputs": X_pred.to_dict(orient="records")[0],
        "feature_source": source,
        "model_ready": True
    }
