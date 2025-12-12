import streamlit as st
import requests
import pandas as pd
import json
import io
import os
from datetime import datetime as dt
from datetime import date as dt_date
import altair as alt
from streamlit_folium import st_folium
import folium
from PIL import Image
import time
import streamlit.components.v1 as components

# --- CONFIGURATION ---
API_URL = "https://my-api-98532754363.europe-west1.run.app/"
NOAA_DATA_SOURCE_URL = "https://coralreefwatch.noaa.gov/product/5km/index.php#data_access"

# ==================================================================
# 1. SIMULATED DATA & API
# ==================================================================
def get_simulated_noaa_data(date: dt, lat: float, lon: float) -> dict:
    st.info(f"🔌 **Simulating NOAA data fetch** for {lat}°N, {lon}°E on {date}...")
    return {
        'Distance_to_Shore': 10.0 + (abs(lon) % 5),
        'Turbidity': 2.5 + (abs(lat) % 1),
        'Cyclone_Frequency': 0.1,
        'Depth_m': 15.0 + (abs(lon) % 10),
        'ClimSST': 26.0 + (abs(lat) % 2),
        'Temperature_Kelvin': 300.0 + (abs(lon) % 2),
        'Temperature_Kelvin_Standard_Deviation': 1.5,
        'Windspeed': 5.0 + (abs(lat) % 3),
    }

def make_api_request_simulation(prediction_type, data, image_present, api_url):
    if prediction_type == "Image-Only (VGG Augmented)":
        risk = 0.75
        model = "96 vgg16_best.keras (Image Model)"
        features = ["VGG16 Features (Image)"]
    elif prediction_type in ("Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"):
        risk = 0.55
        model = "coral_bleaching_tabular_pipe.ipynb (Tabular Model)"
        features = list(data.keys())
    elif prediction_type == "Multi-Modal Fusion (Image + Data)":
        risk = 0.89
        model = "Fusion Model (96 vgg16 + Tabular Pipeline)"
        features = list(data.keys()) + ["VGG16 Features (Image)"]
    else:
        risk = 0.40
        model = "Default/Unknown Model"
        features = []

    time.sleep(1)
    return {
        "status": "success",
        "predicted_bleaching_risk": risk * 100,
        "model_used": model,
        "input_features": features,
        "data_sent_to_api": data,
        "image_processed": image_present,
        "api_endpoint": api_url
    }, None

# ==================================================================
# 2. PAGE SETUP & CSS
# ==================================================================
st.set_page_config(
    page_title="🌊 ReefSight Bleaching Predictor",
    initial_sidebar_state="collapsed",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #e0f7fa 0%, #b2ebf2 40%, #80deea 100%);
    color: #004d40;
}
h1, h2, h3, h4, h5, h6 { color: #004d40 !important; }
button[data-testid*="stFormSubmitButton"] {
    background-color: darkorange !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
    border: none !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
}

/* Fish loader */
.fish-loader-container { width:100%; height:50px; overflow:hidden; position:relative; margin:20px 0; background:transparent; }
.fish-loader { width:50px; height:30px; background-color:#ff8f00; border-radius:50% 50% 50% 50%/60% 60% 40% 40%; position:absolute; left:-100px; animation:swim 3s linear infinite; transform:rotate(5deg);}
.fish-loader::after { content:''; position:absolute; top:5px; left:45px; width:20px; height:15px; background-color:#ff8f00; border-radius:50%/0 100% 0 100%; transform:rotate(45deg);}
@keyframes swim { 0% {left:-10%;} 100% {left:110%;} }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1,8,1])
with col2:
    st.title("🌊 ReefSight: Multi-Modal Coral Bleaching Prediction")
    st.image("Great-Barrier-Reef.jpg", caption="A healthy Great Barrier Reef", width=1050)
    st.markdown("Welcome! Analyze coral health using images, environmental data, or both.")
st.markdown("---")

# ==================================================================
# 3. MAP + SIDEBAR INPUTS
# ==================================================================
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None

col_map, col_inputs = st.columns([3,1])

with col_map:
    st.subheader("Select Location on Map")
    default_location = [0.0,0.0]
    map_center = [st.session_state.selected_location["lat"], st.session_state.selected_location["lon"]] if st.session_state.selected_location else default_location

    # Reef overlay map
    m = folium.Map(
        location=map_center, zoom_start=3, width="100%", height=550,
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="Reef Overlay"
    )

    if st.session_state.selected_location:
        folium.Marker(
            location=[st.session_state.selected_location["lat"], st.session_state.selected_location["lon"]],
            tooltip="Selected Location",
            icon=folium.Icon(color="darkblue", icon="fish", prefix="fa")
        ).add_to(m)

    map_data = st_folium(m, width="100%", height=550)

    if map_data and map_data.get("last_clicked"):
        st.session_state.selected_location = {"lat": map_data["last_clicked"]["lat"], "lon": map_data["last_clicked"]["lng"]}

# Form for inputs
with col_inputs:
    with st.form("prediction_input_form"):
        st.subheader("Required Prediction Inputs")
        input_date = st.date_input("Observation Date", dt_date.today())
        current_lat = st.session_state.selected_location["lat"] if st.session_state.selected_location else 0.0
        current_lon = st.session_state.selected_location["lon"] if st.session_state.selected_location else 0.0
        input_lat = st.number_input("Latitude", value=current_lat, format="%.6f")
        input_lon = st.number_input("Longitude", value=current_lon, format="%.6f")
        st.session_state.selected_location = {"lat": input_lat, "lon": input_lon}

        st.markdown("---")
        st.subheader("Prediction Mode Selection")
        prediction_type = st.radio(
            "Choose prediction mode:",
            ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)", "Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"),
            index=0, horizontal=True
        )

        st.markdown("---")
        override_features = {}
        override_data = False
        if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"):
            st.subheader("Optional Feature Overrides")
            with st.expander("Click to enter environmental data manually"):
                override_data = True
                c1,c2 = st.columns(2)
                with c1:
                    override_features["Distance_to_Shore"]=st.number_input("Distance to Shore (km)",10.0)
                    override_features["Turbidity"]=st.number_input("Turbidity (NTU)",2.5)
                    override_features["Cyclone_Frequency"]=st.number_input("Cyclone Frequency",0.1)
                    override_features["Depth_m"]=st.number_input("Depth (m)",15.0)
                with c2:
                    override_features["ClimSST"]=st.number_input("ClimSST (°C)",26.0)
                    override_features["Temperature_Kelvin"]=st.number_input("Temperature (K)",300.0)
                    override_features["Temperature_Kelvin_Standard_Deviation"]=st.number_input("Temp Std Dev",1.5)
                    override_features["Windspeed"]=st.number_input("Windspeed (m/s)",5.0)

        st.markdown("---")
        uploaded_file = None
        if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)"):
            st.subheader("Image Input")
            uploaded_file = st.file_uploader("Upload coral image", type=["jpg","png","jpeg"])

        form_submitted = st.form_submit_button("RUN PREDICTION", type="primary", help="Run bleaching prediction now!")

# ==================================================================
# 4. HANDLE SUBMISSION
# ==================================================================
if form_submitted:
    # Validation
    if not input_date or input_lat is None or input_lon is None or (prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)") and not uploaded_file):
        st.error("Please fill all required fields.")
        st.stop()

    # Fish loader
    loader_placeholder = st.empty()
    loader_placeholder.markdown("""
    <div class="fish-loader-container">
        <div class="fish-loader"></div>
        <p style="text-align:center; color:#004d40; font-weight:bold; margin-top:30px;">
            Running prediction...
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare data
    data_payload={}
    if prediction_type!="Image-Only (VGG Augmented)":
        aux_data = get_simulated_noaa_data(dt.combine(input_date, dt.min.time()), input_lat, input_lon)
        data_payload.update(aux_data)
        if override_data:
            data_payload.update(override_features)

    # Simulated API request
    api_response,error = make_api_request_simulation(prediction_type, data_payload, uploaded_file is not None, API_URL)
    loader_placeholder.empty()

    # Show bubbles
    def show_bubbles(num_bubbles=50, width="100%", height=400):
        components.html(f"""
<div id="particles-js" style="position:relative; width:{width}; height:{height}px;"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {{
  "particles": {{
    "number": {{ "value": {num_bubbles} }},
    "color": {{ "value": "#00c8ff" }},
    "shape": {{ "type": "circle" }},
    "opacity": {{ "value": 0.6 }},
    "size": {{ "value": 10, "random": true }},
    "line_linked": {{ "enable": false }},
    "move": {{ "enable": true, "speed": 2, "direction": "top", "random": true, "out_mode": "out" }}
  }},
  "interactivity": {{
    "events": {{ "onhover": {{ "enable": false }}, "onclick": {{ "enable": false }} }}
  }},
  "retina_detect": true
}});
</script>
""", height=height)

    if error:
        st.error(f"Prediction Error: {error}")
        st.stop()

    st.success("Prediction complete!")
    show_bubbles(num_bubbles=80,height=500)

    bleaching_percentage = api_response["predicted_bleaching_risk"]
    risk_level = "High Risk" if bleaching_percentage>70 else ("Moderate Risk" if bleaching_percentage>40 else "Low Risk")

    st.write(f"**Date:** {input_date}")
    st.write(f"**Latitude:** {input_lat}")
    st.write(f"**Longitude:** {input_lon}")
    st.write(f"**Prediction Type:** {prediction_type}")
    st.metric(label="Predicted Bleaching Risk", value=f"{bleaching_percentage:.1f}%", delta=risk_level)

    if uploaded_file:
        st.subheader("Uploaded Coral Image")
        st.image(uploaded_file, width=350)

    # Show model used
    st.subheader("Model Used")
    st.info(api_response.get("model_used","N/A"))

# ==================================================================
# 5. FOOTER
# ==================================================================
st.markdown("---")
colL,colM,colR = st.columns([1,8,1])
with colM:
    st.markdown("### Privacy and Data Security Policy")
    st.warning("NO DATA RETENTION POLICY")
    st.markdown("""
* **Input Data**: All user inputs (images, coordinates, environmental data) are used ONLY for the immediate prediction request.
* **Data Handling**: No images, coordinates, or predictions are stored, logged, or retained.
* **Security**: All processing occurs in-memory and is wiped immediately.
* **Usage**: Your data is never written to disk or saved in any persistent way.
""")

