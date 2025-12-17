import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime as dt
from datetime import date as dt_date
from streamlit_folium import st_folium
import folium
from PIL import Image
import streamlit.components.v1 as components
import io
import random
import base64
from io import BytesIO

def img_to_bytes(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- CONFIGURATION ---
API_URL = "https://reefsight-api-98532754363.europe-west1.run.app"
NOAA_DATA_SOURCE_URL = "https://coralreefwatch.noaa.gov/product/5km/index.php#data_access"

# --- CORAL_FACTS ---
CORAL_FACTS = [
    " Coral bleaching occurs when corals expel the algae (zooxanthellae) that live in their tissues, causing the coral to turn white.",
    " The primary cause of coral bleaching is rising sea temperatures, often linked to climate change.",
    " Bleached corals are not dead, but they are under more stress and are at a higher risk of mortality.",
    " Increased sea surface temperatures are the most common cause of coral bleaching.",
    " Pollution from agricultural runoff and sewage can also lead to coral bleaching.",
    " Overexposure to sunlight, especially during low tides, can cause corals to bleach.",
    " Ocean acidification, resulting from increased CO2 levels, weakens corals and makes them more susceptible to bleaching.",
    " Coral bleaching reduces the biodiversity of coral reefs, as many marine species depend on healthy corals for habitat.",
    " Bleached corals have reduced reproductive capabilities, affecting the regeneration of coral populations.",
    " Coral reefs provide coastal protection by reducing wave energy, and their degradation can lead to increased coastal erosion.",
    " The loss of coral reefs can negatively impact local economies that rely on tourism and fishing.",
    " The first global coral bleaching event was recorded in 1998, during a strong El Niño event.",
    " Another significant global bleaching event occurred in 2010, affecting reefs in the Caribbean, Indian Ocean, and Southeast Asia.",
    " The most severe global bleaching event to date happened between 2014 and 2017, impacting over 70% of the world's coral reefs.",
    " Rising global temperatures due to climate change are the primary driver of increased coral bleaching events.",
    " Climate models predict that if current trends continue, annual severe bleaching will occur on 99% of the world's reefs by the end of the century.",
    " Efforts to reduce greenhouse gas emissions can help mitigate the impact of climate change on coral reefs.",
    " Marine protected areas (MPAs) can help reduce local stressors on coral reefs, giving them a better chance to recover from bleaching events.",
    " Coral restoration projects involve growing corals in nurseries and transplanting them to degraded reefs.",
    " Researchers are exploring the potential of breeding heat-resistant coral species to withstand higher temperatures.",
    " Reducing pollution and improving water quality can help alleviate some of the stressors that contribute to coral bleaching.",
    " Coral reefs support over 25% of all marine species, despite covering less than 1% of the ocean floor.",
    " They provide food and livelihood for millions of people worldwide.",
    " Coral reefs are a source of new medicines, including treatments for cancer and other diseases.",
    " Healthy coral reefs contribute to the overall health of the ocean, which is essential for the planet's climate regulation.",
    " Protecting coral reefs is crucial for maintaining biodiversity and the well-being of human communities that depend on them."
]

def get_random_fact():
    return random.choice(CORAL_FACTS)

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🌊 ReefSight Bleaching Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS ---
st.markdown("""
<style>
.stApp { background-color: #ffffff; color:#004d40; }
h1,h2,h3,h4,h5,h6{color:#004d40 !important;}
button[data-testid*="stFormSubmitButton"] { background-color: darkorange !important; color: white !important; font-weight:bold !important; font-size:16px !important; padding:10px 22px !important; border-radius:8px !important; border:none !important; margin-left:auto !important; margin-right:auto !important; display:block !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown(
    "<h1 style='text-align:center; color:#004d40;'>🌊 ReefSight: Multi-Modal Coral Bleaching Prediction</h1>",
    unsafe_allow_html=True
)

# Centered image
img = Image.open("reefmix.jpg")
st.markdown(
    f"""
    <div style='text-align:center;'>
        <img src='data:image/png;base64,{img_to_bytes(img)}'
             style='max-width:90%; height:auto;' />
    </div>
    """,
    unsafe_allow_html=True
)

# Centered descriptive sentence
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Welcome to ReefSight. Above you can see a clear difference between a Healthy and Bleached South Florida Coral Reef. Use our model to analyze coral health using images, environmental data, or both.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# --- SESSION STATE ---
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None

# --- LAYOUT COLUMNS ---
col_map, col_inputs = st.columns([3,1])

# --- MAP COLUMN ---
with col_map:
    st.subheader("Select Location on Map")
    default_location = [0.0,0.0]
    map_center = [
        st.session_state.selected_location["lat"],
        st.session_state.selected_location["lon"]
    ] if st.session_state.selected_location else default_location

    m = folium.Map(
        location=map_center,
        zoom_start=3,
        width="100%",
        min_zoom=2,
        max_bounds=True,
        height=550,
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Reef Overlay"
    )
    m.fit_bounds([[-85, -180], [85, 180]])

    if st.session_state.selected_location:
        folium.Marker(
            location=[st.session_state.selected_location["lat"], st.session_state.selected_location["lon"]],
            tooltip="Selected Location",
            icon=folium.Icon(color="darkblue", icon="fish", prefix="fa")
        ).add_to(m)

    map_data = st_folium(m, width="100%", height=550)
    if map_data and map_data.get("last_clicked"):
        st.session_state.selected_location = {
            "lat": map_data["last_clicked"]["lat"],
            "lon": map_data["last_clicked"]["lng"]
        }

    # --- RESULTS ROW ---
    st.markdown("---")
    st.markdown("## Results:")

# --- INPUT FORM COLUMN ---
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
            ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)", "Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"),
            index=0,
            horizontal=True
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
                    override_features["Distance_to_Shore"] = st.number_input("Distance to Shore (km)", value=10.0)
                    override_features["Turbidity"] = st.number_input("Turbidity (NTU)", value=2.5)
                    override_features["Cyclone_Frequency"] = st.number_input("Cyclone Frequency", value=0.1)
                    override_features["Depth_m"] = st.number_input("Depth (m)", value=15.0)
                with c2:
                    override_features["ClimSST"] = st.number_input("ClimSST (°C)", value=26.0)
                    override_features["Temperature_Kelvin"] = st.number_input("Temperature (K)", value=300.0)
                    override_features["Temperature_Kelvin_Standard_Deviation"] = st.number_input("Temp Std Dev", value=1.5)
                    override_features["Windspeed"] = st.number_input("Windspeed (m/s)", value=5.0)

        st.markdown("---")
        uploaded_file = None
        if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)"):
            st.subheader("Image Input")
            uploaded_file = st.file_uploader("Upload coral image", type=["jpg","png","jpeg"])

        form_submitted = st.form_submit_button("RUN PREDICTION", type="primary", help="Run bleaching prediction now!")

# --- SUBMISSION HANDLER ---
if form_submitted:
    # Validation
    if input_lat is None or input_lon is None:
        st.error("Please provide a valid location.")
        st.stop()
    if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)") and not uploaded_file:
        st.error("Please upload an image for image-based prediction.")
        st.stop()

    loader_placeholder = st.empty()
    loader_placeholder.markdown("""
<div style="width:100%; height:60px; overflow:hidden; position:relative; background:transparent;">
    <div class="octopus" style="font-size:50px; position:absolute; left:-60px;">🐙</div>
    <p style="text-align:center; color:#004d40; font-weight:bold; margin-top:10px;">
        Running prediction and fetching environmental data...
    </p>
</div>

<style>
@keyframes swim {
    0% { left: -60px; }
    100% { left: 100%; }
}
.octopus {
    animation: swim 7s linear infinite;
    transform: scale(1.25); /* 25% larger */
    filter: hue-rotate(260deg) saturate(3) brightness(1); /* purple #7547d1 */
}
</style>
""", unsafe_allow_html=True)

    # --- Determine endpoint & payload ---
    if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)") and uploaded_file:
        # Image endpoint
        endpoint = f"{API_URL}/predict/image"
        files = {"image_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"model_name": "baseline"}  # optional
        json_payload = None
    else:
        # Tabular endpoint
        endpoint = f"{API_URL}/predict/tabular"
        final_override_data = override_features if override_data and override_features else {}
        json_payload = {
            "lat": input_lat,
            "lon": input_lon,
            "date": str(input_date),
            **final_override_data
        }
        files = None
        params = None

    # --- API Request ---
    try:
        if files:
            response = requests.post(endpoint, files=files, params=params, timeout=30)
        else:
            response = requests.post(endpoint, json=json_payload, timeout=30)
        response.raise_for_status()
        api_result = response.json()
    except requests.exceptions.HTTPError as e:
        loader_placeholder.empty()
        error_detail = response.json().get("detail", "Unknown API error.") if response.content else "No response body."
        st.error(f"Prediction API Error ({response.status_code}): {error_detail}")
        st.stop()
    except Exception as e:
        loader_placeholder.empty()
        st.error(f"Prediction API request failed. Check the endpoint URL: {endpoint}. Error: {e}")
        st.stop()

    loader_placeholder.empty()
    st.success("Prediction Complete!")

    # --- Display Results in col_map Results row ---
    with col_map:
        # Bubble Animation
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
                "move": {{
                  "enable": true,
                  "speed": 2,
                  "direction": "top",
                  "random": true,
                  "out_mode": "out"
                }}
              }},
              "interactivity": {{
                "events": {{
                  "onhover": {{ "enable": false }},
                  "onclick": {{ "enable": false }}
                }}
              }},
              "retina_detect": true
            }});
            </script>
            """, height=height)

        show_bubbles(num_bubbles=80, height=500)

        # --- Display Prediction ---
    if "prediction" in api_result:
        pred = api_result["prediction"]
        if isinstance(pred, dict) and "predicted_class" in pred:
            predicted_class = pred["predicted_class"]
            probability = pred.get(f"probability_{predicted_class.lower()}", None)
            prob_percent = f"{probability*100:.1f}%" if probability is not None else "N/A"
        else:
            # fallback for legacy format
            predicted_class = "Bleached" if pred else "Healthy"
            prob_percent = "N/A"

    st.subheader("Prediction Result")
    st.markdown(
        f"<h2 style='text-align:center; color:#d32f2f;'>{predicted_class} ({prob_percent})</h2>",
        unsafe_allow_html=True
    )

    if uploaded_file:
        uploaded_file.seek(0)
        st.subheader("Uploaded Coral Image")
        st.image(uploaded_file, width=350)

        fact = get_random_fact()
        st.info(f"Did you know: {fact}")
        st.markdown("---")
        st.subheader("Prediction Details (Raw API Response)")
        st.json(api_result)

# --- FOOTER ---
st.markdown("---")
colL, colM, colR = st.columns([1,8,1])
with colM:
    st.markdown("### Privacy and Data Security Policy")
    st.warning("NO DATA RETENTION POLICY")
    st.markdown("""
* All inputs (images, coordinates, environmental data) are used only for the immediate prediction request.
* No data is stored or logged.
* All processing occurs in-memory and is wiped after generating the prediction.
""")
