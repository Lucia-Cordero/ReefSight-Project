from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import dill
import os
import tensorflow as tf
import cv2
import base64
import matplotlib
from io import BytesIO
from PIL import Image

from project_logic.preprocessing import load_img_with_original, preprocess_tabular
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
    preprocessed_image, original_np = load_img_with_original(image_bytes)
    pred = model.predict(preprocessed_image)[0][0]

    class_names = ['Bleached', 'Healthy']
    prob_healthy = float(pred)
    prob_bleached = 1 - prob_healthy
    predicted_label = 1 if pred > 0.5 else 0
    predicted_class = class_names[predicted_label]

    cam = compute_gradcam(model, preprocessed_image)
    gradcam_b64 = render_gradcam_overlay(original_np, cam)

    print('✅ Image prediction + GradCAM ready')


    return {
        "predicted_class": predicted_class,
        "probability_bleached": prob_bleached,
        "probability_healthy": prob_healthy,
        "gradcam_image": gradcam_b64
    }

# -----------------------------------------------------------------------
#                           GRADCAM: HEATMAP
# -----------------------------------------------------------------------

def compute_gradcam(model, img_array: np.ndarray) -> np.ndarray:

    # The outer Sequential layers before VGG16 (augmentation + rescaling)
    # We need to pass the image through those first manually
    preprocessing_layers = model.layers[:4]   # RandomFlip, RandomRotation, RandomZoom, Rescaling
    vgg_base = model.layers[4]                # the nested VGG16
    post_layers = model.layers[5:]            # GlobalAveragePooling, Dense, Dropout, Dense

    last_conv_layer = vgg_base.get_layer("block5_conv3")

    # Build grad_model that works entirely within vgg_base's graph
    # Input: vgg_base's input; Outputs: block5_conv3 output + vgg_base output
    grad_model = tf.keras.Model(
        inputs=vgg_base.input,
        outputs=[last_conv_layer.output, vgg_base.output]
    )

    # Step 1: manually run through preprocessing layers
    x = img_array
    for layer in preprocessing_layers:
        x = layer(x, training=False)

    # Step 2: run through grad_model with GradientTape
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv_outputs, vgg_output = grad_model(x, training=False)

        # Step 3: manually run through post-VGG layers to get final prediction
        out = vgg_output
        for layer in post_layers:
            out = layer(out, training=False)

        loss = out[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
    conv_outputs = conv_outputs[0]
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam).numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    return cam


def render_gradcam_overlay(original_np: np.ndarray, cam: np.ndarray,
                           alpha: float = 0.55) -> str:
    # Heatmap + blend (unchanged)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)

    # --- Colorbar from the CV2 LUT directly ---
    img_w = overlay.shape[1]

    # 1-pixel-tall gradient strip spanning the full 0-255 JET range
    gradient = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colorbar_strip = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)
    colorbar_strip = cv2.cvtColor(colorbar_strip, cv2.COLOR_BGR2RGB)

    # Resize to match overlay width and give it some height
    colorbar_strip = cv2.resize(
        colorbar_strip, (img_w, 20), interpolation=cv2.INTER_LINEAR
    )

    # Simple white label bar underneath (optional but readable)
    combined = np.vstack([overlay, colorbar_strip])

    pil_img = Image.fromarray(combined)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

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
