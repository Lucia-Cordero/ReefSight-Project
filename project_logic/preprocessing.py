import numpy as np
import pandas as pd
import dill
from pydantic import BaseModel
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os


class TabularInput(BaseModel):
    Longitude_Degrees: float
    year_norm: float
    Latitude_Degrees: float
    Depth_m: float
    Distance_to_Shore: float
    Temperature_Kelvin_Standard_Deviation: float
    Temperature_Kelvin: float
    TSA: float
    Cyclone_Frequency: float
    SSTA: float
    ClimSST: float
    Realm_Name: str
    SSTA_DHW: float
    month_cos: float
    TSA_DHW: float
    Exposure: float
    Ocean_Name: str
    Windspeed: float
    month_sin: float
    Turbidity: float

def load_img(img_bytes: bytes):

    img = Image.open(io.BytesIO(img_bytes))
    print("✅ Image successfully loaded as", type(img))

    img = img.convert('RGB')
    print("✅ Image converted to RGB")

    img = img.resize((224, 224))
    print("✅ Image resized")

    img = img_to_array(img) #shape = (224, 224, 3)
    print("✅ Image successfully converted to", type(img), img.shape)

    img = img.reshape((-1, 224, 224, 3))
    print("✅ Image successfully reshaped as ndarray of shape", img.shape)

    return img


def load_tabular_preproc():
    preprocessor_path = os.path.join("models", "preproc_tabular.dill")
    with open(preprocessor_path, "rb") as f:
            preprocessor = dill.load(f)

    return preprocessor

def preprocess_tabular(X: pd.DataFrame = None):

    preprocessor = load_tabular_preproc()
    X_preprocessed = preprocessor.transform(X)

    return X_preprocessed
