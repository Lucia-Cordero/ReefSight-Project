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

def load_image_model_trained():
    model_path = os.path.join("models", "VGG16_image_model.keras")
    image_model = load_model(model_path)
    print('✅ Image_Model_VGG16_loaded')
    return image_model


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
    Make a bleaching prediction using the VGG16 model
    """
    #Load image with 'load_img' function
    preprocessed_image = load_img(image_bytes)

    #Predict using loaded model's .predict function
    pred = model.predict(preprocessed_image)[0][0]

    #Report classes & probabilities
    class_names = ['Bleached', 'Healthy']

    prob_healthy = float(pred)
    prob_bleached = 1 - prob_healthy

    predicted_label = 1 if pred > 0.5 else 0
    predicted_class = class_names[predicted_label]

    print('✅ Image prediction ready')

    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_healthy": prob_healthy
    }


# -----------------------------------------------------------------------
#                           PREDICTION: TABULAR
# -----------------------------------------------------------------------

def predict_tabular(model=None, X_pred: pd.DataFrame = None):
    """
    Make a bleaching prediction using the RandomFores model
    """

    #Preprocess X_pred using preprocess_tabular function
    X_pred_preprocessed = preprocess_tabular (X_pred)

    #Predict using loaded model's .predict function
    y_pred = model.predict(X_pred_preprocessed)
    y_proba = model.predict_proba(X_pred_preprocessed)[0]


    #Report classes & probabilities
    class_names = ['Healthy', 'Bleached']

    predicted_label = int(y_pred[0])
    predicted_class = class_names[predicted_label]

    prob_healthy = float(y_proba[0])
    prob_bleached = float(y_proba[1])

    print('✅ Tabular prediction ready')

    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_healthy": prob_healthy
    }
