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

# --- CONFIGURATION ---
API_URL = "https://my-api-98532754363.europe-west1.run.app/"
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
.stApp { background: linear-gradient(to bottom, #e0f7fa 0%, #b2ebf2 40%, #80deea 100%); color:#004d40; }
h1,h2,h3,h4,h5,h6{color:#004d40 !important;}
button[data-testid*="stFormSubmitButton"] { background-color: darkorange !important; color: white !important; font-weight:bold !important; font-size:16px !important; padding:10px 22px !important; border-radius:8px !important; border:none !important; margin-left:auto !important; margin-right:auto !important; display:block !important; }
.fish-loader-container { width:100%; height:50px; overflow:hidden; position:relative; margin:20px 0; background:transparent; }
.fish-loader { width:50px; height:30px; background-color:#ff8f00; border-radius:50% 50% 50% 50% / 60% 60% 40% 40%; position:absolute; left:-100px; animation:swim 3s linear infinite; transform:rotate(-5deg);}
.fish-loader::after { content:''; position:absolute; top:5px; left:45px; width:20px; height:15px; background-color:#ff8f00; border-radius:50% / 0 100% 0 100%; transform:rotate(45deg);}
@keyframes swim {0%{left:-10%;}100%{left:110%;}}
</style>
""", unsafe_allow_html=True)

# --- TITLE + HEADER IMAGE ---
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.title("🌊 ReefSight: Multi-Modal Coral Bleaching Prediction")
    st.image("reef3.jpg", caption="A healthy Great Barrier Reef", width=1050)
    st.markdown(
        "<p style='text-align:center;'>Welcome to ReefSight. Analyze coral health using images, environmental data, or both.</p>",
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

    # Reef overlay
    m = folium.Map(
        location=map_center,
        zoom_start=3,
        width="100%",
        height=550,
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Reef Overlay"
    )
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
            ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)", "Tabular-Only", "Manual Data Entry Only (No NOAA Pull)"),
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
        if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)"):
            st.subheader("Image Input")
            uploaded_file = st.file_uploader("Upload coral image", type=["jpg","png","jpeg"])

        form_submitted = st.form_submit_button("RUN PREDICTION", type="primary", help="Run bleaching prediction now!")

# --- SUBMISSION HANDLER ---
if form_submitted:
    # Cosmetic validation
    if input_lat is None or input_lon is None:
        st.error("Please provide a valid location.")
        st.stop()
    if prediction_type in ("Multi-Modal Fusion (Image + Data)", "Image-Only (VGG Augmented)") and not uploaded_file:
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
    animation: swim 3s linear infinite;
    transform: scale(1.25); /* 25% larger */
    filter: hue-rotate(260deg) saturate(3) brightness(1); /* approximates purple #7547d1 */
}
</style>
""", unsafe_allow_html=True)



    # --- Build Payload for Backend ---
    final_override_data = override_features if override_data and override_features else None
    payload = {
        "prediction_type": prediction_type,
        "lat": input_lat,
        "lon": input_lon,
        "date": str(input_date),
        "override_data": final_override_data
    }
    files = {}
    if uploaded_file:
        files["image_file"] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)

    # --- API Request ---
    try:
        full_url = f"{API_URL}/predict"
        response = requests.post(full_url, data={"payload": json.dumps(payload)}, files=files, timeout=30)
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

    # --- Bubble Animation ---
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

    # --- Display Results ---
    st.write(f"**Date:** {api_result['input_data']['date']}")
    st.write(f"**Latitude:** {api_result['input_data']['latitude']}")
    st.write(f"**Longitude:** {api_result['input_data']['longitude']}")
    st.write(f"**Prediction Type:** {api_result['mode_used']}")

        # --- Multi-Modal Fusion Detail ---
    if api_result["mode_used"] == "Multi-Modal Fusion (Image + Data)":
        fusion = prediction_data.get("fusion_detail", {})
        if fusion:
            st.subheader("Fusion Details")
            st.write(f"**Tabular Risk:** {fusion.get('tabular_risk', 0.0)}%")
            st.write(f"**Image Risk:** {fusion.get('image_risk', 0.0)}%")

    if uploaded_file:
        st.subheader("Uploaded Coral Image")
        uploaded_file.seek(0)
        st.image(uploaded_file, width=350)

    st.markdown("---")
    fact = get_random_fact()
    st.info(f"Did you know: {fact}")
    st.markdown("---")
    st.subheader("Prediction Details")
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
