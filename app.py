import streamlit as st
import requests
import pandas as pd
import json
import io
import os
from datetime import datetime

# --- CONFIGURATION ---
# IMPORTANT: Replace this with your actual deployed FastAPI URL
API_URL = "http://localhost:8000"
# Fallback to which endpoint?
DEFAULT_ENDPOINT = #"/predict/endpoint"#

# PAGE SETUP
st.set_page_config(
    page_title="ReefSight Bleaching Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 ReefSight: Multi-Modal Coral Bleaching Prediction")

st.image("https://www.cruiseexperts.com/news/wp-content/uploads/2015/04/Great-Barrier-Reef.jpg",
         caption="A healthy Great Barrier Reef", use_column_width=True
)

st.markdown("""
Welcome to the ReefSight prediction interface. You can analyze coral health using either
an **Image Upload** or **Structured Environmental Data**.
""")

# --- Prediction Type Selector ---
prediction_type = st.sidebar.selectbox(
    "Select Prediction Mode:",
    ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)", "Tabular-Only")
)


# --- Tabular Data Inputs (Needed for Fusion and Tabular-Only) ---
# NOTE: These inputs assume your tabular data is unscaled, and your API handles the scaling.
# Adjust the number and names of features (e.g., 10 features)
st.sidebar.header("Contextual Data Input")

with st.sidebar.form(key='tabular_form'):
    st.markdown("**Enter 10 Tabular Features**:")

    # Example Feature Inputs (Adjust these to your 10 actual feature names!)
    f1 = st.number_input('1. Latitude Degrees', value=15.0, format="%.4f", key='lat')
    f2 = st.number_input('2. Longitude Degrees', value=-80.0, format="%.4f", key='lon')
    f3 = st.number_input('3. Distance to Shore (km)', value=10.0, format="%.2f", key='dist')
    f4 = st.number_input('4. Turbidity (NTU)', value=2.5, format="%.1f", key='turbidity')
    f5 = st.number_input('5. Cyclone Frequency (per year)', value=0.1, format="%.2f", key='cyclone')
    f6 = st.number_input('6. Depth (m)', value=15.0, format="%.1f", key='depth')
    f7 = st.number_input('7. ClimSST (°C)', value=26.0, format="%.2f", key='clim_sst')
    f8 = st.number_input('8. Temperature (Kelvin)', value=300.0, format="%.2f", key='temp_k')
    f9 = st.number_input('9. Temp Kelvin Std Dev', value=1.5, format="%.2f", key='temp_std')
    f10 = st.number_input('10. Windspeed (m/s)', value=5.0, format="%.2f", key='windspeed')

    tabular_submitted = st.form_submit_button(label='Generate Prediction')

    # Package tabular data into a dictionary with the EXACT required keys
    tabular_data = {
        'Latitude_Degrees': f1,
        'Longitude_Degrees': f2,
        'Distance_to_Shore': f3,
        'Turbidity': f4,
        'Cyclone_Frequency': f5,
        'Depth_m': f6,
        'ClimSST': f7,
        'Temperature_Kelvin': f8,
        'Temperature_Kelvin_Standard_Deviation': f9,
        'Windspeed': f10
    }


uploaded_file = None
if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)"):
    st.header("Image Input")
    uploaded_file = st.file_uploader("Upload a coral image (JPG, PNG)", type=["jpg", "png", "jpeg"])

# --- API CALL FUNCTION ---
def make_api_request(endpoint, files=None, data=None):
    """Handles the request and displays prediction or error."""

    full_url = API_URL + endpoint

    if API_URL == "http://localhost:8000" and not os.getenv("TEST_MODE"):
        st.warning("⚠️ Using local API endpoint. Ensure your FastAPI server is running.")

    try:
        if files or data:
            # For Fusion and Tabular, data must be a JSON string inside 'data' field
            if data:
                payload = {'data': json.dumps(data)}
            else:
                payload = None

            response = requests.post(full_url, files=files, data=payload)
        else:
            # Should not happen in this app structure
            return None, "Error: No data or files provided."

        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json(), None

    except requests.exceptions.HTTPError as e:
        error_detail = response.json().get('detail', str(e))
        return None, f"API HTTP Error ({response.status_code}): {error_detail}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: Could not reach API at {API_URL}. Is it running?"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

# IMAGE UPLOAD AND API CALL
# --- EXECUTION LOGIC ---
if tabular_submitted or (uploaded_file is not None):

    files = None
    data_to_send = tabular_data if tabular_data else None

    # Determine endpoint and files based on prediction_type
    if prediction_type == "Tabular-Only":
        endpoint = "/predict/tabular"
        # No files needed

    elif prediction_type == "Image-Only (VGG Augmented)":
        endpoint = "/predict/image/vgg/augmented"
        if uploaded_file is None:
            st.error("Please upload an image for Image-Only prediction.")
            st.stop()
        files = {"image_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data_to_send = None

    elif prediction_type == "Multi-Modal Fusion (Image + Data)":
        endpoint = "/predict/fusion"
        if uploaded_file is None:
            st.error("Please upload an image for Fusion prediction.")
            st.stop()
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}


    # --- CALL API ---
    with st.spinner(f"Predicting using {endpoint}..."):
        result_json, error = make_api_request(endpoint, files=files, data=data_to_send)


    # --- DISPLAY RESULTS ---
    if error:
        st.error(f"Prediction Failed: {error}")
    elif result_json:
        # Check for model loading errors returned in the JSON payload
        if "error" in result_json.get("prediction", {}):
            st.error(f"Model Loading Error: {result_json['prediction']['error']}")
        else:
            st.success("Prediction Successful!")

            prediction = result_json.get('prediction', {})

            # Display uploaded image if available
            if uploaded_file:
                image_data = uploaded_file.getvalue()
                st.image(image_data, caption=uploaded_file.name, width=300)

            st.header("Results")

            col1, col2 = st.columns(2)

            # Display Prediction Metrics
            with col1:
                st.metric(
                    label="Predicted Class",
                    value=prediction.get("predicted_class", "N/A")
                )
            with col2:
                # Display Bleaching Probability as a percentage
                prob_bleached = prediction.get("probability_bleached", 0.0)
                st.metric(
                    label="Bleaching Probability",
                    value=f"{prob_bleached * 100:.2f} %"
                )

            st.subheader("Input Details")
            st.json(result_json.get("inputs"))

            st.info(f"Model Used: {prediction.get('model_used')}")
