from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import dill
import os



from project_logic.preprocessing import load_img
from project_logic.preprocessing import preprocess_tabular


# -----------------------------------------------------------------------
#                              MODEL LOADING
# -----------------------------------------------------------------------

IMAGE_MODELS = {
    "baseline": "baseline_image_model.keras",
    "vgg16": "vgg16_best.keras",
}

def load_image_models_trained(model_name: str = "vgg16"):
    if model_name not in IMAGE_MODELS:
        raise ValueError(
            f"Unknown image model '{model_name}'. "
            f"Available models: {list(IMAGE_MODELS.keys())}"
        )

    model_path = os.path.join("models", IMAGE_MODELS[model_name])
    model = load_model(model_path)

    print(f"✅ Image model '{model_name}' loaded")
    return model


def load_tabular_model_trained():
    model_path = os.path.join("models", "RandomForestClassifier.dill")
    with open(model_path, "rb") as f:
        tabular_model = dill.load(f)
    print('✅ Tabular_Model_loaded')
    return tabular_model


# -----------------------------------------------------------------------
#                           PREDICTION: IMAGE
# -----------------------------------------------------------------------


def predict_image(model=None, image_bytes=None):
    """
    Make a bleaching prediction using the latest trained CNN/VGG16 model
    """
    #Load image with 'load_img' function
    preprocessed_image = load_img(image_bytes)

    #Predict using loaded model's .predict function
    pred = model.predict(preprocessed_image)[0][0]

    #Report classes & probabilities
    class_names = ['Bleached', 'Unbleached']

    prob_unbleached = float(pred)
    prob_bleached = 1 - prob_unbleached

    predicted_label = 1 if pred > 0.5 else 0
    predicted_class = class_names[predicted_label]

    print('✅ Image prediction ready')

    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_unbleached": prob_unbleached
    }


# -----------------------------------------------------------------------
#                           PREDICTION: TABULAR
# -----------------------------------------------------------------------

def predict_tabular(model=None, X_pred: pd.DataFrame = None):
    """
    Make a bleaching prediction using the latest trained tabular model
    """

    #Preprocess X_pred using preprocess_tabular function
    X_pred_preprocessed = preprocess_tabular (X_pred)

    #Predict using loaded model's .predict function
    pred = model.predict(X_pred_preprocessed)

    #Report classes & probabilities
    class_names = ['Bleached', 'Unbleached']

    prob_unbleached = float(pred)
    prob_bleached = 1 - prob_unbleached

    predicted_label = 1 if pred > 0.5 else 0
    predicted_class = class_names[predicted_label]


    print('✅ Tabular prediction ready')

    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_unbleached": prob_unbleached
    }
