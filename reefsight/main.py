import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image
import io
import tensorflow as tf
import numpy as np



#Loading the model once at start_up
#(avoid doing it inside predict_image_from_bytes function otherwise API too slow)
MODEL_PATH = "/home/lucia/code/Lucia-Cordero/baseline_model.keras"
MODEL = load_model(MODEL_PATH)




def predict_image_from_bytes(image_bytes):
    """
    Make a bleaching prediction using the latest trained computer vision model
    """

    print("\n⭐️ Use case: predict images")

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 3. Preprocess
    image_size = (224, 224)
    img = pil_img.resize(image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)


    # 5. Predict
    pred = MODEL.predict(img_array)[0][0]

    # 6. Convert to class name
    class_names = ['Bleached', 'Unbleached']

    prob_unbleached = float(pred)
    prob_bleached = 1 - prob_unbleached

    predicted_label = 1 if pred > 0.5 else 0
    predicted_class = class_names[predicted_label]

    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_unbleached": prob_unbleached
    }




def predict_tabular(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a bleaching prediction using the latest trained tabular model
    """

    print("\n⭐️ Use case: predict tabular")

    pass #CODE WILL GO HERE
