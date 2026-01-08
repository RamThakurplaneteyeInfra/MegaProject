import streamlit as st
import requests
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
import pandas as pd
import time

API = "http://192.168.42.56:3002"

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("🌱 Agriculture Intelligence Platform")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key in [
    "district",
    "subdistrict",
    "village",
    "geojson",
    "tile_url",
    "summary",
    "layer_type",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# --------------------------------------------------
# API HELPERS
# --------------------------------------------------
def get_districts():
    r = requests.get(f"{API}/districts")
    return r.json().get("districts", [])

def get_subdistricts(district):
    r = requests.get(f"{API}/subdistricts", params={"district": district})
    return r.json().get("subdistricts", [])

def get_villages(district, subdistrict):
    r = requests.get(
        f"{API}/villages",
        params={"district": district, "subdistrict": subdistrict},
    )
    return r.json().get("villages", [])

# --------------------------------------------------
# SIDEBAR SELECTION
# --------------------------------------------------
st.sidebar.header("📍 Location Selection")

district = st.sidebar.selectbox(
    "Select District",
    [""] + get_districts()
)

subdistrict = None
village = None

if district:
    subdistrict = st.sidebar.selectbox(
        "Select Subdistrict",
        [""] + get_subdistricts(district)
    )

if district and subdistrict:
    village = st.sidebar.selectbox(
        "Select Village",
        [""] + get_villages(district, subdistrict)
    )

# --------------------------------------------------
# ANALYSIS TYPE
# --------------------------------------------------
analysis_type = st.sidebar.radio(
    "Select Analysis",
    ["Pest Detection", "Growth", "Soil Moisture", "Water Uptake"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if district:
    params = {
        "district": district,
        "subdistrict": subdistrict,
        "village": village,
    }

    endpoint_map = {
        "Pest Detection": "/pest-detection",
        "Growth": "/analyze_Growth1",
        "Soil Moisture": "/SoilMoisture",
        "Water Uptake": "/wateruptake",
    }

    endpoint = endpoint_map[analysis_type]

    try:
        response = requests.post(
            f"{API}{endpoint}",
            params={k: v for k, v in params.items() if v}
        ).json()

        st.session_state.summary = response.get("pixel_summary")

        # Geometry
        feature = response["features"][0]
        st.session_state.geojson = {
            "type": "FeatureCollection",
            "features": [feature],
        }

        st.session_state.tile_url = feature["properties"].get("tile_url")

    except Exception as e:
        st.error(f"API Error: {e}")
        st.stop()

# --------------------------------------------------
# MAP
# --------------------------------------------------
m = leafmap.Map(center=[19.5, 75.5], zoom=7)
m.add_basemap("HYBRID")

if st.session_state.geojson:
    m.add_geojson(
        st.session_state.geojson,
        layer_name="Boundary",
        style_function=lambda x: {
            "color": "yellow",
            "weight": 2,
            "fillOpacity": 0,
        },
    )

if st.session_state.tile_url:
    m.add_tile_layer(
        url=st.session_state.tile_url,
        name=analysis_type,
        attribution="Google Earth Engine",
        opacity=0.85,
    )

st_folium(m, height=650, width=1300)

# --------------------------------------------------
# SUMMARY TABLE
# --------------------------------------------------
if st.session_state.summary:
    st.subheader(f"📊 {analysis_type} Summary")
    st.dataframe(
        pd.DataFrame([st.session_state.summary]),
        use_container_width=True,
    )
