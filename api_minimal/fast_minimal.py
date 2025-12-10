from reefsight.main import predict_tabular, predict_image_from_bytes
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware



# -------------------------------------------------------------
# Assumes  predict_image / predict_tabular functions exist in main.py file
# -------------------------------------------------------------


# Initialize FastAPI app
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Root endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API  is running! This is the root endpoint"
    }



# Predict endpoint DUMMY
def get_predict(input_one: float,
            input_two: float):

    # Dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }


# Predict endpoint for IMAGE PREDICTION
MODEL_READY = True

@app.post("/predict_image")
async def predict_image_endpoint(image_file: UploadFile= File(...)):

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

    # Otherwise, call real predict function
    prediction = predict_image_from_bytes(image_bytes)   #PLUG IN FUTURE PREDICT FUNCTION

    return {
        "prediction": prediction,
        "inputs": {"filename": image_file.filename},
        "model_ready": True
    }
