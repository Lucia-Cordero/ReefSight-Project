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
API_URL = "https://reef-sight-api-98532754363.europe-west1.run.app"
NOAA_DATA_SOURCE_URL = "https://coralreefwatch.noaa.gov/product/5km/index.php#data_access"

# --- CORAL_FACTS ---
CORAL_FACTS = [
    " Coral bleaching occurs when corals expel the algae (zooxanthellae) that live in their tissues, causing the coral to turn white.",
    " The primary cause of coral bleaching is rising sea temperatures, often linked to climate change.",
    " Bleached corals are not dead; but they are under more stress and are at a higher risk of mortality.",
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
    """Returns a random fact from the CORAL_FACTS list."""
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
.fish-loader-container { width:100%; height:50px; overflow:hidden; position:relative; margin:20px 0; background:transparent; }
.fish-loader { width:50px; height:30px; background-color:#ff8f00; border-radius:50% 50% 50% 50% / 60% 60% 40% 40%; position:absolute; left:-100px; animation:swim 3s linear infinite; transform:rotate(-5deg);}
.fish-loader::after { content:''; position:absolute; top:5px; left:45px; width:20px; height:15px; background-color:#ff8f00; border-radius:50% / 0 100% 0 100%; transform:rotate(45deg);}
@keyframes swim {0%{left:-10%;}100%{left:110%;}}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown(
    "<h1 style='text-align:center; color:#004d40;'>🌊 ReefSight: Multi-Modal Coral Bleaching Prediction</h1>",
    unsafe_allow_html=True
)

# Centered image (KEEPING ORIGINAL BASE64 RENDERING LOGIC)
try:
    # NOTE: This requires 'reefmix.jpg' to be in the local directory when run,
    # which is the user's intended implementation.
    img = Image.open("reefmix.jpg")
    img_src = f'data:image/png;base64,{img_to_bytes(img)}'
except FileNotFoundError:
    # Fallback if image file is not found in this environment
    img_src = "https://placehold.co/1050x300/4eb3e3/ffffff?text=Image+reefmix.jpg+Placeholder"

st.markdown(
    f"""
    <div style='text-align:center;'>
        <img src='{img_src}'
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
    # Initialize with a dictionary structure to avoid key errors later
    st.session_state.selected_location = {"lat": 0.0, "lon": 0.0}

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
    st.markdown("## Results:") # Page break + Results title

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
            horizontal=False # Changed to vertical for better UI flow in small column
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

# --- CUSTOM BUBBLE ANIMATION FUNCTION (Moved outside if block for clarity, but definition is identical) ---

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


# --- SUBMISSION HANDLER (FIXED ROUTING AND PARSING) ---
if form_submitted:
    # 1. Validation
    if input_lat is None or input_lon is None:
        st.error("Please provide a valid location.")
        st.stop()
    if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)") and not uploaded_file:
        st.error("Please upload an image for image-based prediction.")
        st.stop()
    if prediction_type == "Manual Data Entry Only (No NOAA Pull)" and not override_features:
        st.error("Please enter manual data for this prediction mode.")
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
    transform: scale(1.25);
    filter: hue-rotate(260deg) saturate(3) brightness(1);
}
</style>
""", unsafe_allow_html=True)

    # 2. Dynamic Endpoint and Request Construction
    api_call_kwargs = {}

    # Modes requiring Tabular Endpoint
    if prediction_type in ("Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"):
        endpoint = "/predict/tabular"

        # Prepare the request body for the /predict/tabular endpoint
        tabular_payload = {
            "latitude": input_lat,
            "longitude": input_lon,
            "date": str(input_date),
            **(override_features if override_features else {}) # Merge manual overrides
        }
        # Send as JSON body for the POST request
        api_call_kwargs = {"json": tabular_payload}

    # Modes requiring Image Endpoint
    elif prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (Baseline)"):
        endpoint = "/predict/image"

        # Prepare the image file for the /predict/image endpoint
        files = {"image_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        # Send as multipart/form-data (only the file)
        api_call_kwargs = {"files": files}

    else:
        loader_placeholder.empty()
        st.error(f"Internal routing error for prediction type: {prediction_type}")
        st.stop()

    # 3. Make API Request
    try:
        full_url = f"{API_URL}{endpoint}"

        response = requests.post(full_url, timeout=30, **api_call_kwargs)
        response.raise_for_status()
        api_result = response.json()

    except requests.exceptions.HTTPError as e:
        loader_placeholder.empty()
        error_detail = response.json().get("detail", "Unknown API error.") if response.content else "No response body."
        st.error(f"Prediction API Error ({response.status_code}): {error_detail}")
        st.stop()
    except Exception as e:
        loader_placeholder.empty()
        st.error(f"Prediction API request failed. Check the endpoint URL: {full_url}. Error: {e}")
        st.stop()

    loader_placeholder.empty()
    st.success("Prediction Complete!")

    # 4. Display Results in col_map Results row (FIXED DISPLAY LOGIC)
    with col_map:
        # Bubble Animation
        show_bubbles(num_bubbles=80, height=500)

        # Display Prediction Result Card
        if "prediction" in api_result:
            prediction_data = api_result["prediction"]

            # --- API Key Parsing for Binary Classification ---
            classification = prediction_data.get("predicted_class", "Error: No Class")
            probability_bleached = prediction_data.get("probability_bleached", 0.0)
            probability_unbleached = prediction_data.get("probability_unbleached", 0.0)
            risk = probability_bleached * 100 # Convert probability to percentage
            # ------------------------------------------------

            # Determine visual style
            if "Bleached" in classification:
                color = "#f44336" # Red
                icon = "🔥"
            else:
                color = "#4CAF50" # Green
                icon = "✅"

            # Set risk level text based on risk percentage
            level = "High Risk" if risk > 70 else ("Moderate Risk" if risk > 40 else "Low Risk")

            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">{icon} CLASSIFICATION: {classification.upper()} {icon}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Main Risk Metric
            st.metric("Predicted Bleaching Risk", f"{risk:.1f}%", level)

            st.markdown("---")
            st.subheader("Prediction Confidence Breakdown")

            prob_healthy = probability_unbleached * 100

            col_p1, col_p2 = st.columns(2)

            with col_p1:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #4CAF50;">
                    <p style="margin: 0; font-size: 14px; color: #388e3c; font-weight: bold;">Probability Healthy/Unbleached</p>
                    <h3 style="margin: 5px 0 0; color: #1b5e20;">{prob_healthy:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            with col_p2:
                st.markdown(f"""
                <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #f44336;">
                    <p style="margin: 0; font-size: 14px; color: #d32f2f; font-weight: bold;">Probability Bleached</p>
                    <h3 style="margin: 5px 0 0; color: #b71c1c;">{risk:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")


        st.subheader("Input & Metadata")

        # Use 'input_data' key for metadata display
        input_display_data = api_result.get("input_data", {})

        st.markdown(f"**Date:** `{input_display_data.get('date', str(input_date))}`")
        st.markdown(f"**Prediction Mode:** `{api_result.get('mode_used', prediction_type)}`")

        col_meta_1, col_meta_2 = st.columns(2)
        with col_meta_1:
            st.markdown(f"**Latitude:** `{input_display_data.get('latitude', input_lat):.4f}`")
        with col_meta_2:
            st.markdown(f"**Longitude:** `{input_display_data.get('longitude', input_lon):.4f}`")

        if uploaded_file:
            st.markdown("#### Uploaded Coral Image Preview")
            uploaded_file.seek(0)
            st.image(uploaded_file, width=350)

        # Display environmental data used, if any
        # Filter out geo/date metadata to only show environmental features
        environmental_features = {k: v for k, v in input_display_data.items() if k not in ["latitude", "longitude", "date"]}
        if environmental_features:
            st.markdown("#### Environmental Data Used:")
            df_env = pd.DataFrame(list(environmental_features.items()), columns=['Feature', 'Value'])
            st.table(df_env)

        fact = get_random_fact()
        st.info(f"Did you know: {fact}")
        st.markdown("---")
        st.subheader("Full API Response (Debug)")
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
