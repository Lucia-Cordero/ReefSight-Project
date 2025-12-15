from project_logic.predict import load_image_models_trained, load_tabular_model_trained, predict_tabular, predict_image
from project_logic.preprocessing import TabularInput
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from project_logic.tabular_preproc import *



# Initialize FastAPI app
app = FastAPI()
print('✅ Fast API initialized')

# Pre-load trained models (image, tabular) into app.state
app.state.image_models = {
    "baseline": load_image_models_trained("baseline"),
    "vgg16": load_image_models_trained("vgg16"),
}
app.state.tabular_model = load_tabular_model_trained()


'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
'''

# Root endpoint for https://our-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API  is running! Welcome to ReefSight"
    }


# Image predict endpoint for https://our-domain.com/predict/image

@app.post("/predict/image")
async def predict_image_api(
    image_file: UploadFile = File(...),
    model_name: str = Query("baseline")
):
    # Validate image
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # Validate model name
    if model_name not in app.state.image_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown image model '{model_name}'"
        )

    image_bytes = await image_file.read()

    model = app.state.image_models[model_name]
    prediction = predict_image(
        model=model,
        image_bytes=image_bytes
    )

    return {
        "prediction": prediction,
        "model_used": model_name,
        "inputs": {"filename": image_file.filename},
        "model_ready": True
    }

# Tabular predict endpoint for https://our-domain.com/predict/tabular
@app.post("/predict/tabular")
def predict_tabular_api(payload: TabularInput):


    # Convert payload → pandas DataFrame (1 row)
    X_pred = pd.DataFrame([payload. dict()])

    # Call prediction function "predict_tabular"
    model = app.state.tabular_model
    prediction = predict_tabular(model=model, X_pred=X_pred)

    return {
        "prediction": prediction,
        "inputs": X_pred.to_dict(orient="records")[0],
        "model_ready": True
}
