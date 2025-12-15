import streamlit as st
import requests
from datetime import date as dt_date
from streamlit_folium import st_folium
import folium
import random

# ======================
# CONFIG
# ======================
API_URL = "https://reef-sight-api-98532754363.europe-west1.run.app"

# ======================
# CORAL FACTS
# ======================
CORAL_FACTS = [
    "Coral bleaching occurs when corals expel the algae that live in their tissues.",
    "Rising sea temperatures are the primary cause of coral bleaching.",
    "Bleached corals are not dead but are under extreme stress.",
    "Coral reefs support over 25% of all marine species.",
    "Protecting coral reefs helps protect coastlines from erosion."
]

def get_random_fact():
    return random.choice(CORAL_FACTS)

# ======================
# PAGE SETUP
# ======================
st.set_page_config(
    page_title="🌊 ReefSight Bleaching Predictor",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.title("🌊 ReefSight: Coral Bleaching Predictor")
st.markdown("Analyze coral bleaching risk using **image** or **environmental data**.")
st.markdown("---")

# ======================
# SESSION STATE
# ======================
if "lat" not in st.session_state:
    st.session_state.lat = 0.0
if "lon" not in st.session_state:
    st.session_state.lon = 0.0

# ======================
# PREDICTION MODE (TOP LEVEL)
# ======================
prediction_type = st.radio(
    "Select prediction mode",
    ("Image-Only", "Tabular-Only"),
    horizontal=True
)

st.markdown("---")

# ============================================================
# IMAGE-ONLY MODE
# ============================================================
if prediction_type == "Image-Only":

    with st.form("image_form"):
        st.subheader("Upload Coral Image")

        uploaded_file = st.file_uploader(
            "Upload a coral reef image",
            type=["jpg", "jpeg", "png"]
        )

        submit = st.form_submit_button("RUN IMAGE PREDICTION")

    if submit:
        if not uploaded_file:
            st.error("Please upload an image.")
            st.stop()

        try:
            files = {
                "image_file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                f"{API_URL}/predict/image",
                files=files,
                timeout=30
            )

            response.raise_for_status()
            api_result = response.json()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        prediction = api_result["prediction"]

        st.success("Prediction Complete!")
        st.write(f"**Predicted Class:** {prediction['predicted_class']}")
        st.metric("Probability Bleached", f"{prediction['probability_bleached']:.2%}")
        st.metric("Probability Unbleached", f"{prediction['probability_unbleached']:.2%}")

        st.image(uploaded_file, width=350)
        st.info(f"Did you know? {get_random_fact()}")

# ============================================================
# TABULAR-ONLY MODE
# ============================================================
else:

    col_map, col_form = st.columns([3, 2])

    # -----------------------
    # MAP
    # -----------------------
    with col_map:
        st.subheader("Select Location")

        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=3
        )

        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            tooltip="Selected Location"
        ).add_to(m)

        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data.get("last_clicked"):
            st.session_state.lat = map_data["last_clicked"]["lat"]
            st.session_state.lon = map_data["last_clicked"]["lng"]

    # -----------------------
    # TABULAR FORM
    # -----------------------
    with col_form:
        with st.form("tabular_form"):
            st.subheader("Environmental Data")

            tabular_data = {
                "Latitude_Degrees": st.number_input("Latitude", st.session_state.lat),
                "Longitude_Degrees": st.number_input("Longitude", st.session_state.lon),
                "Date_Year": st.number_input("Year", dt_date.today().year),
                "Date_Month": st.number_input("Month", dt_date.today().month),
                "Distance_to_Shore": st.number_input("Distance to Shore (km)", 150.0),
                "Turbidity": st.number_input("Turbidity (NTU)", 0.03),
                "Cyclone_Frequency": st.number_input("Cyclone Frequency", 60.0),
                "Depth_m": st.number_input("Depth (m)", 10.0),
                "ClimSST": st.number_input("ClimSST (K)", 290.0),
                "Temperature_Kelvin": st.number_input("Temperature (K)", 300.0),
                "Windspeed": st.number_input("Windspeed (m/s)", 7.0),
                "SSTA": st.number_input("SSTA", 0.25),
                "SSTA_DHW": st.number_input("SSTA_DHW", 0.27),
                "TSA": st.number_input("TSA", 1.2),
                "TSA_DHW": st.number_input("TSA_DHW", 2.0),
                "Exposure": st.selectbox(
                    "Exposure",
                    ["Sheltered", "Exposed", "Moderate"]
                )
            }

            submit = st.form_submit_button("RUN TABULAR PREDICTION")

    if submit:
        try:
            response = requests.post(
                f"{API_URL}/predict/tabular",
                json=tabular_data,
                timeout=30
            )

            response.raise_for_status()
            api_result = response.json()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        prediction = api_result["prediction"]

        st.success("Prediction Complete!")
        st.write(f"**Predicted Class:** {prediction['predicted_class']}")
        st.metric("Probability Bleached", f"{prediction['probability_bleached']:.2%}")
        st.metric("Probability Unbleached", f"{prediction['probability_unbleached']:.2%}")

        #st.subheader("Model Inputs")
        #st.json(api_result["inputs"])
        st.info(f"Did you know? {get_random_fact()}")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("ReefSight – No data is stored. All predictions are processed in-memory.")
