from reefsight.predict import load_image_model_trained, load_tabular_model_trained, predict_tabular, predict_image
from reefsight.preprocessing import TabularInput
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd



# Initialize FastAPI app
app = FastAPI()
print('✅ Fast API initialized')

# Pre-load trained models (image, tabular) into app.state
app.state.image_model = load_image_model_trained()
app.state.tabular_model = load_tabular_model_trained()

MODEL_READY=True
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
async def predict_image_api(image_file: UploadFile= File(...)):

    # Make sure it's an image
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read uploaded image bytes
    image_bytes = await image_file.read()


    # If model is not ready, return warning message
    if not MODEL_READY:
        return {
            "prediction": "model_not_ready",
            "inputs": {"filename": image_file.filename},
            "model_ready": False
        }

    # Call prediction function "predict_image"
    model = app.state.image_model
    prediction = predict_image(model=model, image_bytes=image_bytes)

    return {
        "prediction": prediction,
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
