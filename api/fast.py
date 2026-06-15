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
        fetch_errors = {}
        fallback = {}

    # CASE 2 — Minimal input (lat, lon, date)
    else:
        try:
            X_pred, fetch_errors, fallback = build_X_pred(
                lat=payload.latitude,
                lon=payload.longitude,
                dt=payload.observation_date
            )
        except Exception as e:
            # build_X_pred should never raise — if it does it's a code-level bug
            # all other errors are absorbed internally by X_pred_build via fetch_errors
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Unexpected error during data enrichment.",
                    "message": str(e)
                }
            )

        # Any fetch failure → abort before prediction
        if fetch_errors:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Environmental data fetching failed. Prediction aborted.",
                    "fetch_errors": fetch_errors,
                    "fallback": fallback,
                    "suggestion": "Please try again later. The NOAA ERDDAP servers may be rate-limiting or temporarily unavailable."
                }
            )

        source = "auto_enriched"


    # Guard: check for missing values before prediction
    # is that necessary?
    # if fetch_errors, then this should not be reached
    missing_cols = X_pred.columns[X_pred.isnull().any()].tolist()
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "X_pred contains missing values. Prediction aborted.",
                "missing_columns": missing_cols,
                "suggestion": "Some environmental variables could not be resolved. Check fetch_errors for details."
            }
        )

    try:
        prediction = predict_tabular(model=model, X_pred=X_pred)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model prediction failed.",
                "message": str(e)
            }
        )

    return {
        "prediction": prediction,
        "inputs": X_pred.to_dict(orient="records")[0],
        "feature_source": source,
        "fetch_errors": fetch_errors,
        "fallback_sources": fallback,
        "model_ready": True
    }
