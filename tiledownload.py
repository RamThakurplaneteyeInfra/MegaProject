import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from io import BytesIO
from PIL import Image
import datetime

# ======================================
# CONFIG
# ======================================
API_URL = "http://192.168.42.56:3004/ndvi-sugarcane-detection"

st.set_page_config(layout="wide")
st.title("🌱 NDVI Sugarcane Detection")

# ======================================
# SIDEBAR INPUTS
# ======================================
with st.sidebar:
    st.header("Inputs")

    district = st.text_input("District", "Sangali")
    subdistrict = st.text_input("Subdistrict", "Tasgaon")
    village = st.text_input("Village (optional)", "")

    start_date = st.date_input(
        "Start date", datetime.date(2025, 5, 1)
    )
    end_date = st.date_input(
        "End date", datetime.date(2025, 5, 31)
    )

    ndvi_threshold = st.slider(
        "NDVI Threshold",
        0.1, 0.9, 0.45, 0.05
    )

    run_btn = st.button("Run Detection")

# ======================================
# MAIN LOGIC
# ======================================
if run_btn:
    params = {
        "district": district,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ndvi_threshold": ndvi_threshold
    }

    if subdistrict:
        params["subdistrict"] = subdistrict
    if village:
        params["village"] = village

    with st.spinner("Calling NDVI Sugarcane API..."):
        try:
            resp = requests.post(API_URL, params=params, timeout=120)
        except Exception as e:
            st.error(f"API connection failed: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    data = resp.json()
    feature = data["features"][0]

    tile_url = feature["properties"]["tile_url"]
    geometry = feature["geometry"]

    # ======================================
    # MAP
    # ======================================
    st.subheader("🗺️ Sugarcane Map")

    m = folium.Map(tiles=None)

    # Dark basemap (so black background looks correct)
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© CARTO",
        name="Basemap",
    ).add_to(m)

    # Earth Engine tile layer (cache busted)
    folium.TileLayer(
        tiles=tile_url + "?v=" + str(int(datetime.datetime.now().timestamp())),
        attr="Google Earth Engine",
        name="Sugarcane Mask",
        overlay=True,
    ).add_to(m)

    # AOI boundary
    aoi_layer = folium.GeoJson(
        geometry,
        name="AOI",
        style_function=lambda x: {
            "color": "yellow",
            "weight": 2,
            "fillOpacity": 0,
        },
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.fit_bounds(aoi_layer.get_bounds())

    st_folium(m, width=1300, height=650)

    # ======================================
    # DOWNLOAD SINGLE TILE AS PNG
    # ======================================
    st.subheader("⬇️ Download Tile (PNG)")

    col1, col2, col3 = st.columns(3)
    with col1:
        z = st.number_input("Zoom (z)", 8, 18, 14)
    with col2:
        x = st.number_input("Tile X", 0, value=8192)
    with col3:
        y = st.number_input("Tile Y", 0, value=5461)

    if st.button("Download PNG"):
        tile_fetch_url = (
            tile_url
            .replace("{z}", str(z))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
        )

        with st.spinner("Downloading tile..."):
            r = requests.get(tile_fetch_url)
            if r.status_code != 200:
                st.error("Failed to fetch tile")
                st.stop()

        img = Image.open(BytesIO(r.content))
        st.image(img, caption=f"Tile z={z}, x={x}, y={y}")

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="Download PNG",
            data=buf,
            file_name=f"sugarcane_tile_z{z}_x{x}_y{y}.png",
            mime="image/png",
        )
