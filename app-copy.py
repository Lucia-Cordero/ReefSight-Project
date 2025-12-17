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

# --- UTILITY FUNCTIONS ---
def img_to_bytes(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- CONFIGURATION ---
API_URL = "https://reef-sight-api-98532754363.europe-west1.run.app"
NOAA_DATA_SOURCE_URL = "https://coralreefwatch.noaa.gov/product/5km/index.php#data_access"
img_url = "https://image2url.com/images/1765895634547-53b1795e-520b-477f-9fd4-7aa744291e4c.jpg"

# Define User options list early for validation
options_list = ("Image", "Data", "Image+Data")

# Header Image loader
def load_image(url: str):
    return Image.open(BytesIO(requests.get(url).content))

# --- CORAL FACTS complied using researched scientific websites material---
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
    " Coral reefs are a source of new medicines, including treatments for cancer and other medicines.",
    " Healthy coral reefs contribute to the overall health of the ocean, which is essential for the planet's climate regulation.",
    " Protecting coral reefs is crucial for maintaining biodiversity and the well-being of human communities that depend on them."
]

def get_random_fact():
    return random.choice(CORAL_FACTS)


# --- OCTOPUS LOADER ---
def show_octopus_loader():
    st.markdown("""
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

# --- CALLBACK FUNCTION FOR MODE SELECTION ---
def update_mode_selection():
    """Sets the chosen mode and enables rendering of the rest of the application."""
    # The radio widget's current value is accessible via its key in session_state
    st.session_state.prediction_type = st.session_state.mode_selector_radio_temp
    st.session_state.mode_chosen_flag = True

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

div[data-testid="stHorizontalRadio"] div[role="radiogroup"] {
    display: flex !important;
    justify-content: center !important;
    width: 100%;
    gap: 20px;
}

div[data-testid="stRadio"] > label {
    font-size: 1.1em;
    padding: 10px;
    border-radius: 8px;
    border: 2px solid #004d40;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "selected_location" not in st.session_state:
    st.session_state.selected_location = {"lat": 24.555, "lon": -81.782} # Key West Default
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "raw_api_response" not in st.session_state:
    st.session_state.raw_api_response = {}
if "input_lat" not in st.session_state:
    st.session_state.input_lat = st.session_state.selected_location["lat"]
if "input_lon" not in st.session_state:
    st.session_state.input_lon = st.session_state.selected_location["lon"]
if "input_date" not in st.session_state:
    st.session_state.input_date = dt_date.today()
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
# State for loading indicator
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# --- Validate and reset prediction_type if invalid ---
if "prediction_type" not in st.session_state or st.session_state.prediction_type not in options_list:
    st.session_state.prediction_type = "Image+Data"

if "has_manual_data" not in st.session_state:
    st.session_state.has_manual_data = "No (Pull NOAA Data)"

if "mode_chosen_flag" not in st.session_state:
    st.session_state.mode_chosen_flag = False

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🌊 ReefSight</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align:center; color:#004d40;'> Automatic Coral Health Insights Powered by AI</h1>",
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("<h5 style='text-align:center;'>Use our model to analyze coral health using images, environmental data, or both.", unsafe_allow_html=True)

# Centered image
img = load_image(img_url)
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
    "<p style='text-align:center; font-size:18px;'>Above you can see a clear difference between a Healthy and Bleached South Florida Coral Reef. </p>",
    unsafe_allow_html=True
)
st.markdown("---")

# --- SECTION 1: PREDICTION MODE SELECTION (Centered) ---
st.markdown("<h2 style='text-align:center;'> Select Your Data Type </h2>", unsafe_allow_html=True)

# Use columns to center the radio button group
col_l, col_c, col_r = st.columns([0.435, 0.455, 0.1])

with col_c:
    st.markdown(
        "<div style='display:flex; justify-content:center;'>",
        unsafe_allow_html=True
    )
    st.radio(
        "Choose prediction mode:",
        options_list,
        index=options_list.index(st.session_state.prediction_type),
        horizontal=True,
        key="mode_selector_radio_temp",
        label_visibility="collapsed",
        on_change=update_mode_selection
    )

st.markdown("</div>", unsafe_allow_html=True)

if "run_prediction_flag" not in st.session_state:
    st.session_state.run_prediction_flag = False

# --- PREDICTION LOGIC FUNCTION ---
def run_prediction(input_lat, input_lon, input_date, prediction_type, override_features, uploaded_file):

    # --- Internal Mode Handling for Tabular-Only Flow ---
    internal_mode = prediction_type
    if prediction_type == "Data":
        if st.session_state.has_manual_data == "Yes (Manual Data Entry)":
            internal_mode = "Tabular-Only (Manual)"
        else:
            internal_mode = "Tabular-Only (NOAA)"

    # 1. Validation
    requires_loc_date = internal_mode not in ("Image")
    if requires_loc_date:
        # Note: Lat/Lon/Date are required for Fusion and all Tabular modes
        if input_lat is None or input_lon is None or (abs(input_lat) < 0.001 and abs(input_lon) < 0.001):
            st.error("Please provide a valid location (Latitude and Longitude, ensure they are not zero).")
            return False
        if input_date is None:
            st.error("Please provide an Observation Date.")
            return False

    if internal_mode in ("Image+Data", "Image") and not uploaded_file:
        st.error("Please upload an image for image-based prediction.")
        return False

    if internal_mode == "Tabular-Only (Manual)":
        required_keys = ["Distance_to_Shore", "Turbidity", "Cyclone_Frequency", "Depth_m", "ClimSST", "Temperature_Kelvin", "Windspeed", "SSTA", "SSTA_DHW", "TSA", "TSA_DHW", "Exposure"]
        # Ensure all required keys exist in override_features and none of their values are None
        if not all(key in override_features and override_features[key] and override_features[key] != '' for key in required_keys):
            st.error("Please ensure ALL 12 required manual data fields have been entered for this prediction mode.")
            return False

    # Reset results view and diagnostics before running
    st.session_state.show_results = False
    st.session_state.api_result = None
    st.session_state.raw_api_response = {}

    # --- START LOADING ---
    st.session_state.is_loading = True

    # Date extraction for new API schema
    input_year = input_date.year
    input_month = input_date.month

    # 2. Dynamic Endpoint and Request Construction
    api_call_kwargs = {}

    # Tabular Modes (NOAA or Manual)
    if internal_mode in ("Tabular-Only (NOAA)", "Tabular-Only (Manual)"):
        endpoint = "/predict/tabular"

        # Prepare the request body for the /predict/tabular endpoint
        tabular_payload = {
            "Latitude_Degrees": input_lat,
            "Longitude_Degrees": input_lon,
            "Date_Year": input_year,
            "Date_Month": input_month,
        }

        if internal_mode == "Tabular-Only (Manual)":
             tabular_payload.update(override_features)

        api_call_kwargs = {"json": tabular_payload}

    # Image Modes (Fusion or Image-Only)
    elif internal_mode in ("Image+Data", "Image"):
        endpoint = "/predict/image"

        if uploaded_file:
            uploaded_file.seek(0)
            files = {"image_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        data = {}
        if internal_mode == "Image+Data":
            # Data should contain location/date and overrides if provided
            data = {
                "Latitude_Degrees": input_lat,
                "Longitude_Degrees": input_lon,
                "Date_Year": input_year,
                "Date_Month": input_month
            }
            if override_features:
                data.update(override_features)

            api_call_kwargs = {"files": files, "data": data}
        else:
            api_call_kwargs = {"files": files}

    else:
        st.session_state.is_loading = False
        st.error(f"Internal routing error for prediction mode: {internal_mode}")
        return False

    # 3. Make API Request
    try:
        full_url = f"{API_URL}{endpoint}"

        # Check if we are sending data as JSON or Multipart
        if "json" in api_call_kwargs:
            response = requests.post(full_url, timeout=30, json=api_call_kwargs["json"])
        else:
            response = requests.post(full_url, timeout=30, files=api_call_kwargs.get("files"), data=api_call_kwargs.get("data"))

        # Store raw response text for diagnostic purposes
        try:
            api_result = response.json()
            st.session_state.raw_api_response = api_result
        except json.JSONDecodeError:
            st.session_state.is_loading = False
            st.error("API response was not valid JSON.")
            st.exception(f"Raw response text: {response.text}")
            return False

        response.raise_for_status() # Raise HTTPError for bad status codes (4xx or 5xx)

    except requests.exceptions.HTTPError as e:
        st.session_state.is_loading = False
        # Attempt to extract API error detail if possible
        error_detail = st.session_state.raw_api_response.get("detail", "Unknown API error or status code.")
        st.error(f"Prediction API Error ({response.status_code}): {error_detail}")
        st.exception(e)
        return False
    except Exception as e:
        st.session_state.is_loading = False
        st.error(f"Prediction API request failed. Check the endpoint URL: {full_url}.")
        st.exception(e)
        return False

    # --- STOP LOADING ---
    st.session_state.is_loading = False
    st.success("Prediction Complete!")

    # Store results and enable display
    st.session_state.api_result = api_result
    st.session_state.show_results = True
    return True

# --- SECTION 2: DYNAMIC LAYOUT START (Conditional on mode selection) ---
if st.session_state.mode_chosen_flag:

    # Define current mode variables
    current_mode = st.session_state.prediction_type
    is_tabular_only = current_mode == "Data"
    # Re-evaluate is_manual_tabular_only based on current state
    is_manual_tabular_only = is_tabular_only and st.session_state.has_manual_data == "Yes (Manual Data Entry)"
    is_fusion = current_mode == "Image+Data"
    is_image_only = current_mode == "Image"

    # Define visibility gates for Section 2 components
    show_map = is_fusion or (is_tabular_only and st.session_state.has_manual_data == "No (Pull NOAA Data)")

    # Required Inputs (Lat/Lon/Date) are needed for all non-Image-Only modes.
    show_required_inputs = not is_image_only

    # Manual Data Override/Entry is needed for Tabular-Manual or Fusion (as override)
    show_manual_override = is_manual_tabular_only or is_fusion

    # Image Input is needed for Fusion or Image-Only
    show_image_uploader = is_fusion or is_image_only

    # Tabular Data Source Radio is needed for Tabular-Only and Fusion modes
    show_tabular_source_radio = is_tabular_only or is_fusion

    st.markdown("<h2 style='text-align:center;'>Input your data</h2>", unsafe_allow_html=True)
    st.markdown("---")

    if show_tabular_source_radio:
            st.markdown(f"### Data Source Selection")
            st.session_state.has_manual_data = st.radio(
                "Do you have your own environmental data?",
                ("No (Pull NOAA Data)", "Yes (Manual Data Entry)"),
                index=0 if st.session_state.has_manual_data == "No (Pull NOAA Data)" else 1,
                key="manual_data_radio"
            )
            st.info("⚠️ Note: Fetching and processing external NOAA data may take up to 10 minutes!")
            is_manual_tabular_only = st.session_state.has_manual_data == "Yes (Manual Data Entry)"
            show_manual_override = is_manual_tabular_only or is_fusion
            st.markdown("---")

    if show_map:
        map_col, input_col = st.columns([3, 2])
    else:
        map_col = None
        input_col = st.container()

    with input_col:

        current_override_features = {}
        current_uploaded_file = st.session_state.uploaded_file

        with st.form("prediction_input_form", clear_on_submit=False):

            # --- REQUIRED PREDICTION INPUTS (Date/Lat/Lon) - Shown for Fusion, Tabular-NOAA, and Tabular-Manual
            if show_required_inputs:
                st.subheader("Required Location and Time")

                input_date = st.date_input("Observation Date", value=st.session_state.input_date, key="date_input")

                # Use session state as the source of truth, updated by map click or manual entry
                input_lat = st.number_input("Latitude (Degrees)", value=st.session_state.input_lat, format="%.6f")
                input_lon = st.number_input("Longitude (Degrees)", value=st.session_state.input_lon, format="%.6f")

                st.session_state.input_lat = input_lat
                st.session_state.input_lon = input_lon
                st.session_state.input_date = input_date
                st.session_state.selected_location = {"lat": input_lat, "lon": input_lon}
                st.markdown("---")
            else:
                # Needed dummy inputs for Image-Only mode (Lat/Lon/Date are not sent to API in this mode)
                input_lat = st.session_state.input_lat
                input_lon = st.session_state.input_lon
                input_date = st.session_state.input_date

            # --- MANUAL OVERRIDE / FULL MANUAL ENTRY ---
            if show_manual_override:
                header_text = "Manual Feature Overrides (Will override NOAA data)"
                if is_manual_tabular_only:
                    header_text = "Complete Manual Data Entry (Required for prediction)"

                st.subheader(header_text)

                if is_fusion:
                    expander = st.expander("Click to manually input data (Highly Optional)")
                else: # Tabular-Only (Manual)
                    expander = st.container()

                with expander:

                    st.markdown("##### Base Environmental Features")
                    c1,c2 = st.columns(2)
                    with c1:
                        # NOTE: The key must be unique and consistent across reruns.
                        current_override_features["Distance_to_Shore"] = st.number_input("Distance to Shore (km)", value=50.0, key="dist_shore", format="%.2f")
                        current_override_features["Turbidity"] = st.number_input("Turbidity (NTU)", value=0.5, key="turbidity", format="%.2f")
                        current_override_features["Cyclone_Frequency"] = st.number_input("Cyclone Frequency", value=10.0, key="cyclone_freq", format="%.1f")
                        current_override_features["Depth_m"] = st.number_input("Depth (m)", value=25.0, key="depth_m", format="%.1f")
                    with c2:
                        current_override_features["ClimSST"] = st.number_input("ClimSST (°C)", value=295.0, help="Climatological Sea Surface Temperature", key="clim_sst", format="%.1f")
                        current_override_features["Temperature_Kelvin"] = st.number_input("Temperature (K)", value=302.0, key="temp_k", format="%.1f")
                        current_override_features["Windspeed"] = st.number_input("Windspeed (m/s)", value=5.0, key="windspeed", format="%.1f")
                        # Categorical feature
                        current_override_features["Exposure"] = st.selectbox("Reef Exposure", options=["Sheltered", "Exposed"], index=1, key="exposure")

                    st.markdown("##### NOAA Derived Features (Enter data if not pulling from NOAA)")
                    c3,c4 = st.columns(2)
                    with c3:
                        current_override_features["SSTA"] = st.number_input("SSTA", value=0.1, help="Sea Surface Temperature Anomaly", key="ssta", format="%.3f")
                        current_override_features["SSTA_DHW"] = st.number_input("SSTA DHW", value=0.1, help="SSTA Degree Heating Weeks", key="ssta_dhw", format="%.3f")
                    with c4:
                        current_override_features["TSA"] = st.number_input("TSA", value=0.5, help="Thermal Stress Anomaly", key="tsa", format="%.3f")
                        current_override_features["TSA_DHW"] = st.number_input("TSA DHW", value=0.5, help="Thermal Stress Anomaly Degree Heating Weeks", key="tsa_dhw", format="%.3f")

                st.markdown("---")

            # --- IMAGE UPLOADER ---
            if show_image_uploader:
                st.subheader("Image Input")
                current_uploaded_file = st.file_uploader("Upload coral image", type=["jpg","png","jpeg"], key="image_uploader")

            # Save uploaded file to session state
            st.session_state.uploaded_file = current_uploaded_file

            # --- SUBMIT BUTTON ---
            submitted = st.form_submit_button(
                "RUN PREDICTION",
                type="primary",
                help="Run bleaching prediction now!"
            )
# Trigger loader and flag to run prediction on next rerun
            if submitted:
                st.session_state.is_loading = True
                st.session_state.run_prediction_flag = True

        if st.session_state.run_prediction_flag:
            st.session_state.run_prediction_flag = False  # reset immediately

            run_prediction(
                input_lat=st.session_state.input_lat,
                input_lon=st.session_state.input_lon,
                input_date=st.session_state.input_date,
                prediction_type=st.session_state.prediction_type,
                override_features=current_override_features,
                uploaded_file=st.session_state.uploaded_file
            )


    # --- MAP COLUMN (Left Side) ---
    if show_map and map_col:
        with map_col:
            st.subheader("Choose location on map or input coordinates")
            map_center = [
                st.session_state.selected_location["lat"],
                st.session_state.selected_location["lon"]
            ]

            m = folium.Map(
                location=map_center,
                zoom_start=6,
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
                st.session_state.input_lat = map_data["last_clicked"]["lat"]
                st.session_state.input_lon = map_data["last_clicked"]["lng"]
                st.session_state.selected_location = {
                    "lat": st.session_state.input_lat,
                    "lon": st.session_state.input_lon
                }

    # --- RESULTS DISPLAY ---
    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>Results:</h2>", unsafe_allow_html=True)

    # Display Loader while thinking
    if st.session_state.is_loading:
        show_octopus_loader()

    if st.session_state.show_results and st.session_state.api_result:
        api_result = st.session_state.api_result

        # Default API result data structure for display if actual API is not called
        if not api_result.get("prediction"):
            api_result = {
                "prediction": {
                    "probability_bleached": 0.35,
                    "probability_unbleached": 0.65
                },
                "input_data": {
                    "Latitude_Degrees": st.session_state.input_lat,
                    "Longitude_Degrees": st.session_state.input_lon,
                    "Date_Year": st.session_state.input_date.year,
                    "Date_Month": st.session_state.input_date.month
                },
                "tabular_features": {
                    "Distance_to_Shore": 50.0, "Turbidity": 0.5, "Cyclone_Frequency": 10.0, "Depth_m": 25.0,
                    "ClimSST": 295.0, "Temperature_Kelvin": 302.0, "Windspeed": 5.0, "Exposure": "Exposed",
                    "SSTA": 0.1, "SSTA_DHW": 0.1, "TSA": 0.5, "TSA_DHW": 0.5
                },
                "data_source": "NOAA (Mocked)"
            }

        # Display Prediction Result Card
        if "prediction" in api_result:
            prediction_data = api_result["prediction"]

            # --- Robust Classification Logic using Probability ---
            probability_bleached = prediction_data.get("probability_bleached", 0.0)
            probability_unbleached = prediction_data.get("probability_unbleached", 0.0)

            risk = probability_bleached * 100

            if probability_bleached > 0.5:
                final_classification = "Bleached"
                color = "#f44336" # Red
                icon = "🔥"
            else:
                final_classification = "Healthy"
                color = "#4CAF50" # Green
                icon = "✅"

            level = "High Risk" if risk > 70 else ("Moderate Risk" if risk > 40 else "Low Risk")

            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">{icon} CLASSIFICATION: {final_classification.upper()} {icon}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Display Image and Metrics in a row
            img_col, metric_col = st.columns([1, 2])

            with img_col:
                st.subheader("Image Analyzed")
                if st.session_state.uploaded_file:
                    st.image(st.session_state.uploaded_file, caption="User Upload")
                else:
                    st.info("No image was uploaded for this prediction mode.")

            with metric_col:
                prob_healthy = probability_unbleached * 100

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #4CAF50;">
                        <p style="margin: 0; font-size: 14px; color: #388e3c; font-weight: bold;">Probability Healthy</p>
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

            input_display_data = api_result.get("input_data", {})

            # Only show location/date if they were used in the prediction
            if input_display_data.get("Latitude_Degrees") is not None:
                col_l1, col_l2, col_l3 = st.columns(3)
                col_l1.metric("Latitude", f"{input_display_data.get('Latitude_Degrees'):.4f} °")
                col_l2.metric("Longitude", f"{input_display_data.get('Longitude_Degrees'):.4f} °")
                col_l3.metric("Observation Date", f"{input_display_data.get('Date_Year')}-{input_display_data.get('Date_Month')}")

            tabular_features = api_result.get("tabular_features", {})

            df_data = {
                "Feature": [],
                "Value": []
            }

            # Determine the source of the data for the user
            source_note = ""
            if "data_source" in api_result:
                source = api_result["data_source"]
                if source == "NOAA":
                    source_note = f"Data was fetched from NOAA's Coral Reef Watch data based on location and date."
                elif source == "Manual":
                    source_note = "All environmental data was provided manually by the user."
                elif source == "Image_Only":
                    source_note = "Prediction was Image-Only. No environmental data was used."

            if source_note:
                st.info(source_note)

            for key, value in tabular_features.items():
                if key not in ["Latitude_Degrees", "Longitude_Degrees", "Date_Year", "Date_Month"]:
                    display_key = key.replace('_', ' ').title()
                    df_data["Feature"].append(display_key)
                    df_data["Value"].append(f"{value:.4f}" if isinstance(value, (float, int)) else str(value))

            if df_data["Feature"]:
                df = pd.DataFrame(df_data)
                df = df.set_index('Feature')
                st.dataframe(df, use_container_width=True)

            # Random Fact
            fact = get_random_fact()
            st.info(f"Did you know: {fact}")

# --- FOOTER ---
st.markdown("---")
colL, colM, colR = st.columns([1,8,1])
with colM:
    st.markdown("<h3 style='text-align: center;'>Privacy and Data Policy</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 0.25rem; padding: 10px; text-align: center; margin: 10px 0;">
        <strong>NO DATA RETENTION POLICY</strong>
    </div>
""", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>* All inputs (images, coordinates, environmental data) are used only for the immediate prediction request.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>* Predictions are generated by machine learning models using submitted data and may contain inaccuracies.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>* No data is stored or logged.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>* All processing occurs in-memory and is wiped after generating the prediction.</h5>", unsafe_allow_html=True)
