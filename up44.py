from fastapi import FastAPI, HTTPException, Query, Depends,Body,Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import date, datetime, timedelta
from typing import Optional, List
import ee
import uvicorn
from db import get_db
from models import Village, VillageBoundary,Base
from pydantic import BaseModel
from typing import Any
from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.responses import FileResponse,JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import geopandas as gpd
from shapely.geometry import shape
import tempfile
import httpx
import zipfile
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from fastapi.responses import StreamingResponse
import requests
from io import BytesIO
from cachetools import TTLCache
from datetime import datetime
import calendar

manual_geometry: dict | None = Body(
    None,
    description="Optional GeoJSON Polygon. Send only if user draws."
)

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "palus_villages_farms.geojson"
gdf = gpd.read_file(DATA_FILE)

# ============================================
# EARTH ENGINE INIT
# ============================================
try:
    ee.Initialize(project="vast-torus-479210-u3")
except Exception as e:
    print("EE init warning:", e)

# ============================================
# APP SETUP
# ============================================
app = FastAPI(title="Village Boundary & Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DATE HELPER
# ============================================
def default_start_date(end_date: str = None):
    d = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()
    return (d - timedelta(days=15)).strftime("%Y-%m-%d")

# ============================================
# DATABASE FETCH HELPERS
# ============================================

def get_district_data(db: Session, district: str):
    geom_sql = text("""
        SELECT ST_AsGeoJSON(ST_Union(b.geom))::json AS geom
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.district) = LOWER(:district)
    """)

    geom = db.execute(geom_sql, {"district": district}).scalar()
    if not geom:
        raise HTTPException(404, "District not found")

    subs = db.execute(text("""
        SELECT DISTINCT sub_dist
        FROM village
        WHERE LOWER(district) = LOWER(:district)
        ORDER BY sub_dist
    """), {"district": district}).scalars().all()

    return {
        "district": district,
        "geometry": geom,
        "subdistricts": subs
    }


def get_subdistrict_villages(db: Session, district: str, subdistrict: str):
    rows = db.execute(text("""
        SELECT
            v.village_name,
            ST_AsGeoJSON(b.geom)::json AS geometry
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE
            LOWER(v.district) = LOWER(:district)
            AND LOWER(v.sub_dist) = LOWER(:subdistrict)
        ORDER BY v.village_name
    """), {
        "district": district,
        "subdistrict": subdistrict
    }).mappings().all()

    if not rows:
        raise HTTPException(404, "No villages found")

    return list(rows)


def get_single_village(db: Session, district: str, subdistrict: str, village: str):
    row = db.execute(text("""
        SELECT
            v.village_name,
            ST_AsGeoJSON(b.geom)::json AS geometry
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE
            LOWER(v.district) = LOWER(:district)
            AND LOWER(v.sub_dist) = LOWER(:subdistrict)
            AND LOWER(v.village_name) = LOWER(:village)
        LIMIT 1
    """), {
        "district": district,
        "subdistrict": subdistrict,
        "village": village
    }).mappings().first()

    if not row:
        raise HTTPException(404, "Village not found")

    return row

def filter_s1(collection, start, end, aoi):
    return (
        collection.filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .sort("system:time_start")
    )

def addIndices(image):
    vv = image.select("VV")
    vh = image.select("VH")
    ratio = vv.divide(vh.add(1e-6)).rename("VV_VH_ratio")
    swi = vv.subtract(vh).divide(vv.add(vh).add(1e-6)).rename("SWI")
    rvi = vh.multiply(4).divide(vv.add(vh).add(1e-6)).rename("RVI")
    return image.addBands([ratio, swi, rvi])


def safe_median(collection):
    size = collection.size()
    return ee.Image(ee.Algorithms.If(size.gt(0), collection.median(), ee.Image(0)))
# ============================================
# CORE FETCH ENDPOINT
# ============================================
@app.get("/fetch-boundaries")
def fetch_boundaries(
    district: str = Query(...),
    subdistrict: Optional[str] = None,
    village: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if district and not subdistrict:
        return get_district_data(db, district)

    if district and subdistrict and not village:
        return {
            "district": district,
            "subdistrict": subdistrict,
            "villages": get_subdistrict_villages(db, district, subdistrict)
        }

    if district and subdistrict and village:
        return get_single_village(db, district, subdistrict, village)

    raise HTTPException(400, "Invalid parameters")

# ============================================
# LIST ENDPOINTS (UI HELPERS)
# ============================================
@app.get("/districts")
def list_districts(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT
            d.district_name,
            ST_AsGeoJSON(d.geom)::json AS geometry
        FROM districts d
        ORDER BY d.district_name
    """)).mappings().all()

    return {
        "districts": [
            {
                "district": r["district_name"],
                "geometry": r["geometry"]
            }
            for r in rows
        ]
    }
@app.get("/subdistricts")
def list_subdistricts(
    district: str = Query(...),
    db: Session = Depends(get_db)
):
    rows = db.execute(text("""
        SELECT
            v.sub_dist,
            ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.district) = LOWER(:district)
        GROUP BY v.sub_dist
        ORDER BY v.sub_dist
    """), {"district": district}).mappings().all()

    return {
        "district": district,
        "subdistricts": [
            {
                "subdistrict": r["sub_dist"],
                "geometry": r["geometry"]
            }
            for r in rows if r["sub_dist"]
        ]
    }

@app.get("/villages")
def list_villages(
    subdistrict: str = Query(...),
    db: Session = Depends(get_db)
):
    rows = db.execute(text("""
        SELECT
            v.village_name,
            GeometryType(b.geom) AS geom_type,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coordinates
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.sub_dist) = LOWER(:subdistrict)
        ORDER BY v.village_name
    """), {"subdistrict": subdistrict}).mappings().all()

    return {
        "subdistrict": subdistrict,
        "villages": [
            {
                "village": r["village_name"],
                "geom_type": r["geom_type"].replace("ST_", ""),
                "coordinates": r["coordinates"],
            }
            for r in rows
        ]
    }

# ============================================
# EE HELPER
# ============================================
def ee_geometry_from_geojson(geojson):
    return ee.Geometry(geojson)
@app.get("/locations")
def get_districts_and_subdistricts(
    district: Optional[str] = Query(None, description="Optional district name"),
    db: Session = Depends(get_db),
):
    try:
        # --------------------------------------------------
        # CASE 1: All districts with their subdistricts
        # --------------------------------------------------
        if not district:
            rows = db.execute(
                text("""
                    SELECT
                        v.district AS district_name,
                        ARRAY_AGG(DISTINCT v.sub_dist ORDER BY v.sub_dist) AS subdistricts
                    FROM village v
                    GROUP BY v.district
                    ORDER BY v.district
                """)
            ).mappings().all()

            return {
                "count": len(rows),
                "data": [
                    {
                        "district": row["district_name"],
                        "subdistricts": row["subdistricts"]
                    }
                    for row in rows
                ]
            }

        # --------------------------------------------------
        # CASE 2: Single district → its subdistricts
        # --------------------------------------------------
        rows = db.execute(
            text("""
                SELECT DISTINCT sub_dist
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                ORDER BY sub_dist
            """),
            {"district": district},
        ).mappings().all()

        if not rows:
            raise HTTPException(status_code=404, detail="District not found")

        return {
            "district": district,
            "subdistricts": [row["sub_dist"] for row in rows]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch districts/subdistricts: {e}"
        )
# ============================================
# PEST DETECTION (SINGLE VILLAGE)
# ============================================
@app.post("/pest-detection4")
async def pest_detection4(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION
        # ==================================================
        extra_list = None
        geometry = None
        response_geometry = None
        plot_name = None

        
        # ---------- DISTRICT ONLY ----------
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "District boundary not found")

            subs = db.execute(
                text("""
                    SELECT DISTINCT sub_dist
                    FROM village
                    WHERE LOWER(district) = LOWER(:district)
                    ORDER BY sub_dist
                """),
                {"district": district},
            ).fetchall()

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]
            extra_list = {"subdistricts": [s[0] for s in subs if s[0]]}

        # ---------- DISTRICT + SUBDISTRICT ----------
        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Subdistrict boundary not found")

            villages = db.execute(
                text("""
                    SELECT village_name
                    FROM village
                    WHERE LOWER(district) = LOWER(:district)
                      AND LOWER(sub_dist) = LOWER(:subdistrict)
                    ORDER BY village_name
                """),
                {"district": district, "subdistrict": subdistrict},
            ).fetchall()

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]
            extra_list = {"villages": [v[0] for v in villages if v[0]]}

        # ---------- DISTRICT + SUBDISTRICT + VILLAGE ----------
        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Village boundary not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # DATE WINDOWS
        # ==================================================
        analysis_end = ee.Date(end_date)
        analysis_start = analysis_end.advance(-15, "day")

        baseline_year = analysis_end.get("year").subtract(1)
        baseline_start = ee.Date.fromYMD(baseline_year, 6, 1)
        baseline_end = ee.Date.fromYMD(baseline_year, 10, 30)

        # ==================================================
        # SENTINEL-1 BASE COLLECTION
        # ==================================================
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )

        if s1.size().getInfo() == 0:
            raise HTTPException(404, "No Sentinel-1 IW images found")

        s1_median = s1.median().clip(geometry)
        vv = s1_median.select("VV")
        vh = s1_median.select("VH")

        ratio = vv.divide(vh.add(1e-6))
        sar_mean = vv.addBands(vh).addBands(ratio).reduce(ee.Reducer.mean())

        def normalize01(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=10,
                bestEffort=True,
            )
            band = img.bandNames().get(0)
            return img.unitScale(
                ee.Number(stats.get(ee.String(band).cat("_min"))),
                ee.Number(stats.get(ee.String(band).cat("_max"))),
            ).clamp(0, 1)

        sar_norm = normalize01(sar_mean)
        water_norm = normalize01(vh.multiply(-1))
        avg = sar_norm.add(water_norm).multiply(0.5)

        low = avg.reduceRegion(
            ee.Reducer.percentile([2]), geometry, 10, bestEffort=True
        ).values().get(0)

        high = avg.reduceRegion(
            ee.Reducer.percentile([98]), geometry, 10, bestEffort=True
        ).values().get(0)

        chewing_mask = avg.unitScale(low, high).clamp(0, 1).lte(0.01)

        # ==================================================
        # FUNGI (DELTA VV/VH)
        # ==================================================
        def s1_comp(start, end):
            return (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
                .median()
                .clip(geometry)
            )

        baseline = s1_comp(baseline_start, baseline_end)
        analysis = s1_comp(analysis_start, analysis_end)

        delta_ratio = (
            analysis.select("VV").divide(analysis.select("VH"))
            .subtract(baseline.select("VV").divide(baseline.select("VH")))
        )

        fungi_mask = delta_ratio.gte(0.75)

        # ==================================================
        # PLACEHOLDER MASKS
        # ==================================================
        sucking_mask = avg.gte(1)
        wilt_mask = avg.gte(1)
        soilborne_mask = avg.lte(-1)

        # ==================================================
        # COMBINED CLASS
        # ==================================================
        combined = (
            ee.Image(0)
            .where(chewing_mask, 1)
            .where(fungi_mask, 2)
            .where(sucking_mask, 3)
            .where(wilt_mask, 4)
            .where(soilborne_mask, 5)
            .clip(geometry)
        )

        tile_url = combined.visualize(
            min=0,
            max=5,
            palette=[
                "#FF0000",  # healthy
                "#FFFFFF",  # chewing
                "#00DD88",  # fungi
                "#2600FF",  # sucking
                "#D60BFF",  # wilt
                "#FFE608",  # soilborne
            ],
        ).getMapId()["tile_fetcher"].url_format

        # ==================================================
        # PIXEL COUNTS
        # ==================================================
        ones = ee.Image.constant(1)

        def count(mask):
            return int(
                ones.updateMask(mask)
                .reduceRegion(
                    ee.Reducer.count(),
                    geometry,
                    10,
                    bestEffort=True,
                )
                .get("constant")
                .getInfo() or 0
            )

        chewing = count(chewing_mask)
        fungi = count(fungi_mask)
        sucking = count(sucking_mask)
        wilt = count(wilt_mask)
        soilborne = count(soilborne_mask)

        total = int(
            ones.reduceRegion(
                ee.Reducer.count(),
                geometry,
                10,
                bestEffort=True,
            ).get("constant").getInfo()
        )

        healthy = total - (chewing + fungi + sucking + wilt + soilborne)

        def pct(v):
            return round((v / total) * 100, 2) if total else 0

        response = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "data_source": "Sentinel-1 Pest Detection",
                    "last_updated": datetime.now().isoformat(),
                },
            }],
            "pixel_summary": {
                "total_pixel_count": total,
                "healthy_pixel_count": healthy,
                "healthy_pixel_percentage": pct(healthy),
                "chewing_pixel_count": chewing,
                "chewing_pixel_percentage": pct(chewing),
                "fungi_pixel_count": fungi,
                "fungi_pixel_percentage": pct(fungi),
                "sucking_pixel_count": sucking,
                "sucking_pixel_percentage": pct(sucking),
                "wilt_pixel_count": wilt,
                "wilt_pixel_percentage": pct(wilt),
                "soilborne_pixel_count": soilborne,
                "soilborne_pixel_percentage": pct(soilborne),
                "baseline_start": baseline_start.format("YYYY-MM-dd").getInfo(),
                "baseline_end": baseline_end.format("YYYY-MM-dd").getInfo(),
                "analysis_start": analysis_start.format("YYYY-MM-dd").getInfo(),
                "analysis_end": analysis_end.format("YYYY-MM-dd").getInfo(),
            },
        }

        if extra_list:
            response.update(extra_list)

        return response

    except Exception as e:
        raise HTTPException(500, f"Pest detection failed: {e}")


       


@app.post("/analyze_Growth1")
async def analyze_growth(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):

    # ==================================================
    # GEOMETRY SELECTION (UNCHANGED)
    # ==================================================
    if district and not subdistrict and not village:
        row = db.execute(
            text("""
                SELECT d.district_name,
                       ST_AsGeoJSON(d.geom)::json AS geometry
                FROM districts d
                WHERE LOWER(d.district_name) = LOWER(:district)
                LIMIT 1
            """),
            {"district": district},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "District boundary not found")

        subs = db.execute(
            text("""
                SELECT DISTINCT sub_dist
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                ORDER BY sub_dist
            """),
            {"district": district},
        ).fetchall()

        plot_name = row["district_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"subdistricts": [s[0] for s in subs if s[0]]}

    elif district and subdistrict and not village:
        row = db.execute(
            text("""
                SELECT v.sub_dist,
                       ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                GROUP BY v.sub_dist
            """),
            {"district": district, "subdistrict": subdistrict},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Subdistrict boundary not found")

        villages = db.execute(
            text("""
                SELECT village_name
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                  AND LOWER(sub_dist) = LOWER(:subdistrict)
                ORDER BY village_name
            """),
            {"district": district, "subdistrict": subdistrict},
        ).fetchall()

        plot_name = row["sub_dist"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"villages": [v[0] for v in villages if v[0]]}

    elif district and subdistrict and village:
        row = db.execute(
            text("""
                SELECT v.village_name,
                       ST_AsGeoJSON(b.geom)::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                  AND LOWER(v.village_name) = LOWER(:village)
                LIMIT 1
            """),
            {"district": district, "subdistrict": subdistrict, "village": village},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Village boundary not found")

        plot_name = row["village_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = None
    else:
        raise HTTPException(400, "Invalid input combination")

    # ==================================================
    # ANALYSIS
    # ==================================================
    try:
        area_hectares = geometry.area().divide(10000).getInfo()

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        # ---------- Sentinel-2 ----------
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
        )

        s2_size = s2_collection.size().getInfo()
        s2_latest_date = None

        if s2_size > 0:
            s2_latest_date = ee.Date(
                s2_collection.sort("system:time_start", False)
                .first()
                .get("system:time_start")
            )

        # ---------- Sentinel-1 ----------
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .map(lambda img: img.clip(geometry))
        )

        s1_size = s1_collection.size().getInfo()
        s1_latest_date = None

        if s1_size > 0:
            s1_latest_date = ee.Date(
                s1_collection.sort("system:time_start", False)
                .first()
                .get("system:time_start")
            )

        if s1_size == 0 and s2_size == 0:
            raise HTTPException(404, "No Sentinel-1 or Sentinel-2 images available")

        # ==================================================
        # SENSOR SELECTION (UNCHANGED LOGIC)
        # ==================================================
        if s2_latest_date and (not s1_latest_date or s2_latest_date.millis().getInfo() >= s1_latest_date.millis().getInfo()):
            data_source = "Sentinel-2 NDVI"
            latest_image_date = s2_latest_date.format("YYYY-MM-dd").getInfo()

            s2_composite = s2_collection.median().clip(geometry)
            ndvi = s2_composite.normalizedDifference(["B8", "B4"]).rename("NDVI")

            weak_mask = ndvi.lt(0.2)
            stress_mask = ndvi.gte(0.2).And(ndvi.lt(0.35))
            moderate_mask = ndvi.gte(0.35).And(ndvi.lt(0.5))
            healthy_mask = ndvi.gte(0.5)

        else:
            data_source = "Sentinel-1 VH"
            latest_image_date = s1_latest_date.format("YYYY-MM-dd").getInfo()

            vh = s1_collection.median().clip(geometry)

            weak_mask = vh.gte(-11)
            stress_mask = vh.lt(-11).And(vh.gt(-13))
            moderate_mask = vh.lte(-13).And(vh.gt(-15))
            healthy_mask = vh.lte(-15)

        # ==================================================
        # CLASSIFICATION (UNCHANGED)
        # ==================================================
        combined_class = (
            ee.Image(0)
            .where(weak_mask, 1)
            .where(stress_mask, 2)
            .where(moderate_mask, 3)
            .where(healthy_mask, 4)
            .updateMask(
                weak_mask.Or(stress_mask).Or(moderate_mask).Or(healthy_mask)
            )
            .clip(geometry)
        )

        combined_smooth = combined_class.focal_mean(radius=10, units="meters")

        vis_params = {
            "min": 1,
            "max": 4,
            "palette": ["#bc1e29", "#58cf54", "#28ae31", "#056c3e"],
        }

        tile_url = combined_smooth.visualize(**vis_params).getMapId()["tile_fetcher"].url_format

        # ==================================================
        # PIXEL SUMMARY (UNCHANGED)
        # ==================================================
        count_img = ee.Image.constant(1)

        total_pixel_count = int(
            count_img.reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant").getInfo()
        )

        def pixel_count(mask):
            return int(
                count_img.updateMask(mask)
                .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

        healthy_count = pixel_count(healthy_mask)
        moderate_count = pixel_count(moderate_mask)
        weak_count = pixel_count(weak_mask)
        stress_count = pixel_count(stress_mask)

        # ==================================================
        # RESPONSE (UNCHANGED)
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_id": plot_name,
                "area_acres": round(area_hectares * 2.471, 2),
                "start_date": start_date,
                "end_date": end_date,
                "tile_url": tile_url,
                "data_source": data_source,
                "last_updated": datetime.now().isoformat(),
            },
        }

        response = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,
                "healthy_pixel_count": healthy_count,
                "healthy_pixel_percentage": (healthy_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "moderate_pixel_count": moderate_count,
                "moderate_pixel_percentage": (moderate_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "weak_pixel_count": weak_count,
                "weak_pixel_percentage": (weak_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "stress_pixel_count": stress_count,
                "stress_pixel_percentage": (stress_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "analysis_start_date": analysis_start.format("YYYY-MM-dd").getInfo(),
                "analysis_end_date": analysis_end.format("YYYY-MM-dd").getInfo(),
                "latest_image_date": latest_image_date,
                "area_hectares": round(area_hectares, 2),
            },
        }

        if extra_list:
            response.update(extra_list)

        return response

    except Exception as e:
        raise HTTPException(500, f"Growth analysis failed: {e}")

    


@app.post("/wateruptake")
async def analyze_water_uptake(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):

    # ==================================================
    # GEOMETRY SELECTION (UNCHANGED)
    # ==================================================
    if district and not subdistrict and not village:
        row = db.execute(
            text("""
                SELECT d.district_name,
                       ST_AsGeoJSON(d.geom)::json AS geometry
                FROM districts d
                WHERE LOWER(d.district_name) = LOWER(:district)
                LIMIT 1
            """),
            {"district": district},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "District boundary not found")

        subs = db.execute(
            text("""
                SELECT DISTINCT sub_dist
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                ORDER BY sub_dist
            """),
            {"district": district},
        ).fetchall()

        plot_name = row["district_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"subdistricts": [s[0] for s in subs if s[0]]}

    elif district and subdistrict and not village:
        row = db.execute(
            text("""
                SELECT v.sub_dist,
                       ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                GROUP BY v.sub_dist
            """),
            {"district": district, "subdistrict": subdistrict},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Subdistrict boundary not found")

        villages = db.execute(
            text("""
                SELECT village_name
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                  AND LOWER(sub_dist) = LOWER(:subdistrict)
                ORDER BY village_name
            """),
            {"district": district, "subdistrict": subdistrict},
        ).fetchall()

        plot_name = row["sub_dist"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"villages": [v[0] for v in villages if v[0]]}

    elif district and subdistrict and village:
        row = db.execute(
            text("""
                SELECT v.village_name,
                       ST_AsGeoJSON(b.geom)::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                  AND LOWER(v.village_name) = LOWER(:village)
                LIMIT 1
            """),
            {"district": district, "subdistrict": subdistrict, "village": village},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Village boundary not found")

        plot_name = row["village_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = None
    else:
        raise HTTPException(400, "Invalid input combination")

    # ==================================================
    # WATER UPTAKE ANALYSIS (FIXED INTERNALLY)
    # ==================================================
    try:
        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        # ---------- Sentinel-2 NDMI ----------
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img: img.clip(geometry))
            .map(lambda img: img.addBands(
                img.normalizedDifference(["B8A", "B11"]).rename("NDMI")
            ))
            .select(["NDMI"])
        )

        s2_size = s2_collection.size().getInfo()
        s2_latest_date = None
        if s2_size > 0:
            s2_latest_date = ee.Date(
                s2_collection.sort("system:time_start", False)
                .first()
                .get("system:time_start")
            )

        # ---------- Sentinel-1 VH ----------
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .map(lambda img: img.clip(geometry))
        )

        s1_size = s1_collection.size().getInfo()
        s1_latest_date = None

        if s1_size >= 2:
            s1_sorted = s1_collection.sort("system:time_start", False)
            latest_img = ee.Image(s1_sorted.first())
            prev_img = ee.Image(s1_sorted.toList(2).get(1))
            s1_latest_date = ee.Date(latest_img.get("system:time_start"))

        if s1_size == 0 and s2_size == 0:
            raise HTTPException(404, "No Sentinel data available")

        # ==================================================
        # SENSOR SELECTION (UNCHANGED)
        # ==================================================
        if s2_latest_date and (not s1_latest_date or s2_latest_date.millis().getInfo() >= s1_latest_date.millis().getInfo()):
            # ---------- Sentinel-2 ----------
            ndmi = s2_collection.median().clip(geometry)

            deficient = ndmi.lt(-0.21)
            less = ndmi.gte(-0.21).And(ndmi.lt(-0.031))
            adequat = ndmi.gte(-0.031).And(ndmi.lt(0.142))
            excellent = ndmi.gte(0.14).And(ndmi.lt(0.22))
            excess = ndmi.gte(0.22)

        else:
            # ---------- Sentinel-1 ----------
            delta_vh = latest_img.subtract(prev_img).rename("deltaVH").clip(geometry)

            excess = delta_vh.gte(6)
            excellent = delta_vh.gte(4.0).And(delta_vh.lt(6))
            adequat = delta_vh.gt(-0.1).And(delta_vh.lt(4))
            less = delta_vh.gte(-3.13).And(delta_vh.lte(-0.1))
            deficient = delta_vh.lt(-3.13)

        # ==================================================
        # CLASSIFICATION (MASK FIX APPLIED)
        # ==================================================
        combined_class = (
            ee.Image(0)
            .where(deficient, 1)
            .where(less, 2)
            .where(adequat, 3)
            .where(excellent, 4)
            .where(excess, 5)
            .updateMask(
                deficient.Or(less).Or(adequat).Or(excellent).Or(excess)
            )
            .clip(geometry)
        )

        smoothed_class = combined_class.focal_mean(radius=7, units="meters")

        vis_params = {
            "min": 1,
            "max": 5,
            "palette": [
                "#EBFF34",
                "#CC8213AF",
                "#1348E88E",
                "#2E199ABD",
                "#0602178F",
            ],
        }

        tile_url = smoothed_class.visualize(**vis_params).getMapId()["tile_fetcher"].url_format

        # ==================================================
        # PIXEL SUMMARY (UNCHANGED)
        # ==================================================
        count_image = ee.Image.constant(1)

        def get_pixel_count(mask):
            return int(
                count_image.updateMask(mask)
                .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

        deficient_count = get_pixel_count(combined_class.eq(1))
        less_count = get_pixel_count(combined_class.eq(2))
        adequat_count = get_pixel_count(combined_class.eq(3))
        excellent_count = get_pixel_count(combined_class.eq(4))
        excess_count = get_pixel_count(combined_class.eq(5))
        total_pixel_count = get_pixel_count(count_image)

        # ==================================================
        # RESPONSE (UNCHANGED)
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_id": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }

        response = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,
                "deficient_pixel_count": deficient_count,
                "deficient_pixel_percentage": (deficient_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "less_pixel_count": less_count,
                "less_pixel_percentage": (less_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "adequat_pixel_count": adequat_count,
                "adequat_pixel_percentage": (adequat_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "excellent_pixel_count": excellent_count,
                "excellent_pixel_percentage": (excellent_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "excess_pixel_count": excess_count,
                "excess_pixel_percentage": (excess_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
            },
        }

        if extra_list:
            response.update(extra_list)

        return response

    except Exception as e:
        raise HTTPException(500, f"Water uptake analysis failed: {e}")

@app.post("/SoilMoisture")
async def analyze_soil_moisture(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):

    # ==================================================
    # GEOMETRY SELECTION (MATCHES GROWTH)
    # ==================================================
    if district and not subdistrict and not village:
        row = db.execute(
            text("""
                SELECT d.district_name,
                       ST_AsGeoJSON(d.geom)::json AS geometry
                FROM districts d
                WHERE LOWER(d.district_name) = LOWER(:district)
                LIMIT 1
            """),
            {"district": district},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "District boundary not found")

        subs = db.execute(
            text("""
                SELECT DISTINCT sub_dist
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                ORDER BY sub_dist
            """),
            {"district": district},
        ).fetchall()

        plot_name = row["district_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"subdistricts": [s[0] for s in subs if s[0]]}

    elif district and subdistrict and not village:
        row = db.execute(
            text("""
                SELECT v.sub_dist,
                       ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                GROUP BY v.sub_dist
            """),
            {"district": district, "subdistrict": subdistrict},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Subdistrict boundary not found")

        villages = db.execute(
            text("""
                SELECT village_name
                FROM village
                WHERE LOWER(district) = LOWER(:district)
                  AND LOWER(sub_dist) = LOWER(:subdistrict)
                ORDER BY village_name
            """),
            {"district": district, "subdistrict": subdistrict},
        ).fetchall()

        plot_name = row["sub_dist"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = {"villages": [v[0] for v in villages if v[0]]}

    elif district and subdistrict and village:
        row = db.execute(
            text("""
                SELECT v.village_name,
                       ST_AsGeoJSON(b.geom)::json AS geometry
                FROM village v
                JOIN village_boundaries b ON b.village_id = v.id
                WHERE LOWER(v.district) = LOWER(:district)
                  AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                  AND LOWER(v.village_name) = LOWER(:village)
                LIMIT 1
            """),
            {"district": district, "subdistrict": subdistrict, "village": village},
        ).mappings().first()

        if not row:
            raise HTTPException(404, "Village boundary not found")

        plot_name = row["village_name"]
        geometry = ee.Geometry(row["geometry"])
        extra_list = None
    else:
        raise HTTPException(400, "Invalid input combination")

    # ==================================================
    # SOIL MOISTURE ANALYSIS (UNCHANGED)
    # ==================================================
    try:
        # ---------- Sentinel-1 ----------
        s1_collection = (
            filter_s1(
                ee.ImageCollection("COPERNICUS/S1_GRD"),
                start_date,
                end_date,
                geometry,
            ).map(addIndices)
        )

        s1_size = s1_collection.size().getInfo()
        s1_latest_date = None

        if s1_size > 0:
            s1_sorted = s1_collection.sort("system:time_start", False)
            s1_latest_img = ee.Image(s1_sorted.first())
            s1_latest_date = ee.Date(
                s1_latest_img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        # ---------- Sentinel-2 ----------
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
        )

        s2_size = s2_collection.size().getInfo()
        s2_latest_date = None

        if s2_size > 0:
            s2_sorted = s2_collection.sort("system:time_start", False)
            s2_latest_img = ee.Image(s2_sorted.first())
            s2_latest_date = ee.Date(
                s2_latest_img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        if s1_size == 0 and s2_size == 0:
            raise HTTPException(404, "No satellite images available")

        # ---------- SENSOR SELECTION ----------
        if s2_latest_date and (not s1_latest_date or s2_latest_date >= s1_latest_date):
            sensor_used = "Sentinel-2"
            latest_date = s2_latest_date

            s2_composite = s2_collection.median().clip(geometry)
            ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

            classified = (
                ndwi.where(ndwi.lte(-0.4), 1)
                .where(ndwi.gt(-0.4).And(ndwi.lte(-0.3)), 2)
                .where(ndwi.gt(-0.3).And(ndwi.lte(0)), 3)
                .where(ndwi.gt(0).And(ndwi.lte(0.2)), 4)
                .where(ndwi.gt(0.2), 5)
            ).clip(geometry)

            image_count = s2_size
        else:
            sensor_used = "Sentinel-1"
            latest_date = s1_latest_date

            vv = safe_median(s1_collection.select(["VV"])).clip(geometry)

            classified = (
                vv.where(vv.gt(-6), 5)
                .where(vv.gt(-8).And(vv.lte(-6)), 4)
                .where(vv.gt(-10).And(vv.lte(-8)), 3)
                .where(vv.gt(-12).And(vv.lte(-10)), 2)
                .where(vv.lte(-12), 1)
            ).clip(geometry)

            image_count = s1_size

        # ---------- VISUAL ----------
        palette = ["#2FC0D3", "#4365D4", "#473CDF", "#2116BF", "#000475"]
        combined_smooth = classified.focal_mean(
        radius=10,
        units="meters"
        )

        visual = combined_smooth.visualize(min=1, max=5, palette=palette)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ---------- PIXEL SUMMARY ----------
        count_img = ee.Image.constant(1)
        total_pixel_count = int(
            count_img.reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant").getInfo()
        )

        labels = {
            1: "less",
            2: "adequate",
            3: "excellent",
            4: "excess",
            5: "shallow_water",
        }

        pixel_summary = {
            "total_pixel_count": total_pixel_count,
            "analysis_start_date": start_date,
            "analysis_end_date": end_date,
            "latest_image_date": latest_date,
            "sensor_used": sensor_used,
        }

        for cid, label in labels.items():
            cnt = int(
                count_img.updateMask(classified.eq(cid))
                .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

            pixel_summary[f"{label}_pixel_count"] = cnt
            pixel_summary[f"{label}_pixel_percentage"] = (
                round((cnt / total_pixel_count) * 100, 2) if total_pixel_count else 0
            )

        # ==================================================
        # RESPONSE (MATCHES GROWTH)
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_id": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "tile_url": tile_url,
                "data_source": f"{sensor_used} Soil Moisture",
                "last_updated": datetime.now().isoformat(),
            },
        }

        response = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": pixel_summary,
        }

        if extra_list:
            response.update(extra_list)

        return response

    except Exception as e:
        raise HTTPException(500, f"Soil moisture analysis failed: {e}")
    
    
def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    return ee.Image(img).log10().multiply(10.0)

def RefinedLee(img):
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0]
    ])
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs() \
        .addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs()) \
        .addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs()) \
        .addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(
        sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(
        sample_mean.select(6).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(
        sample_mean.select(3).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(
        sample_mean.select(0).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))

    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    directions = directions.updateMask(gradmask)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigmaV = sample_stats.toArray().arraySort().arraySlice(0, 0, 5) \
        .arrayReduce(ee.Reducer.mean(), [0])

    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3) \
        .cat(ee.List.repeat(ee.List.repeat(1,7),4))
    diag_weights = ee.List([
        [1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0],
        [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0],
        [1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1]
    ])

    rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    for i in range(1, 4):
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2*i+1)))
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2*i+1)))

        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2*i+2)))
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2*i+2)))

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)) \
        .divide(sigmaV.add(1.0))
    b = varX.divide(dir_var)

    result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
    return result.arrayFlatten([['sum']])



@app.post("/WaterDetection")
async def detect_water(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION (MATCHES GROWTH)
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "District boundary not found")

            subs = db.execute(
                text("""
                    SELECT DISTINCT sub_dist
                    FROM village
                    WHERE LOWER(district) = LOWER(:district)
                    ORDER BY sub_dist
                """),
                {"district": district},
            ).fetchall()

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            extra_list = {"subdistricts": [s[0] for s in subs if s[0]]}

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Subdistrict boundary not found")

            villages = db.execute(
                text("""
                    SELECT village_name
                    FROM village
                    WHERE LOWER(district) = LOWER(:district)
                      AND LOWER(sub_dist) = LOWER(:subdistrict)
                    ORDER BY village_name
                """),
                {"district": district, "subdistrict": subdistrict},
            ).fetchall()

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            extra_list = {"villages": [v[0] for v in villages if v[0]]}

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Village boundary not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            extra_list = None

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # Sentinel-1 VH Collection
        # ==================================================
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.Or(
                    ee.Filter.eq("orbitProperties_pass", "ASCENDING"),
                    ee.Filter.eq("orbitProperties_pass", "DESCENDING"),
                )
            )
        )

        size = collection.size().getInfo()
        if size == 0:
            raise HTTPException(status_code=404, detail="No Sentinel-1 VH images found")

        sorted_col = collection.sort("system:time_start", False)
        latest = ee.Image(sorted_col.first())
        latest_date = (
            ee.Date(latest.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )

        vh = collection.select("VH").mosaic().clip(geometry)

        # ==================================================
        # Refined Lee filtering
        # ==================================================
        vh_filtered = toDB(RefinedLee(toNatural(vh)))

        # ==================================================
        # Water mask: VH < -20 dB
        # ==================================================
        water = vh_filtered.lt(-20)
        water_mask = water.updateMask(water)

        # ==================================================
        # Visualization
        # ==================================================
        vis = {"min": -25, "max": 0, "palette": ["0000ff"]}
        visual = water_mask.visualize(**vis).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # Pixel counting
        # ==================================================
        scale = 10
        ones = ee.Image.constant(1)

        total_pixels = (
            ones.reduceRegion(
                ee.Reducer.count(), geometry, scale, bestEffort=True
            )
            .get("constant")
            .getInfo()
        )

        water_pixels = (
            ones.updateMask(water_mask)
            .reduceRegion(
                ee.Reducer.count(), geometry, scale, bestEffort=True
            )
            .get("constant")
            .getInfo()
            or 0
        )

        # ==================================================
        # Pixel coordinates
        # ==================================================
        points = (
            water_mask.selfMask()
            .addBands(ee.Image.pixelLonLat())
            .sample(
                region=geometry,
                scale=scale,
                geometries=True,
                tileScale=4,
            )
            .getInfo()
        )

        coords = [
            f["geometry"]["coordinates"]
            for f in points.get("features", [])
        ]
        coords = [list(x) for x in {tuple(c) for c in coords}]

        # ==================================================
        # GeoJSON Feature
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "latest_image_date": latest_date,
                "image_count": size,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,
                "water_pixel_count": water_pixels,
                "water_pixel_percentage": (
                    round((water_pixels / total_pixels) * 100, 2)
                    if total_pixels
                    else 0
                ),
                "water_pixel_coordinates": coords,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Water detection failed: {str(e)}",
        )
@app.post("/WaterDetection1")
async def detect_water(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "District boundary not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Subdistrict boundary not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Village boundary not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # Sentinel-1 VH Collection
        # ==================================================
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.Or(
                    ee.Filter.eq("orbitProperties_pass", "ASCENDING"),
                    ee.Filter.eq("orbitProperties_pass", "DESCENDING"),
                )
            )
        )

        size = collection.size().getInfo()
        if size == 0:
            raise HTTPException(404, "No Sentinel-1 VH images found")

        sorted_col = collection.sort("system:time_start", False)
        latest = ee.Image(sorted_col.first())
        latest_date = (
            ee.Date(latest.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )

        vh = collection.select("VH").mosaic().clip(geometry)

        # ==================================================
        # Refined Lee filtering
        # ==================================================
        vh_filtered = toDB(RefinedLee(toNatural(vh)))

        # ==================================================
        # Water mask: VH < -20 dB
        # ==================================================
        water_mask = vh_filtered.lt(-25)

        # ==================================================
        # Visualization
        # ==================================================
        vis = {"min": -25, "max": 0, "palette": ["0000ff"]}
        visual = water_mask.selfMask().visualize(**vis).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # Pixel counting
        # ==================================================
        scale = 10
        ones = ee.Image.constant(1)

        total_pixels = (
            ones.reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
            .get("constant")
            .getInfo()
        )

        water_pixels = (
            ones.updateMask(water_mask)
            .reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
            .get("constant")
            .getInfo()
            or 0
        )

        # ==================================================
        # GeoJSON Feature
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "latest_image_date": latest_date,
                "image_count": size,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,
                "water_pixel_count": water_pixels,
                "water_pixel_percentage": (
                    round((water_pixels / total_pixels) * 100, 2)
                    if total_pixels
                    else 0
                ),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Water detection failed: {str(e)}",
        )
    
@app.post("/ndvi-sugarcane-detection12")
async def ndvi_sugarcane_detection(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    start_date: str = Query(...),
    end_date: str = Query(...),
    ndvi_threshold: float = Query(0.72),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) NDVI COLLECTIONS
        # ==================================================
        def landsat_ndvi(collection_id):
            return (
                ee.ImageCollection(collection_id)
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .map(lambda img: (
                    img.select("SR.*")
                    .multiply(ee.Number(img.get("REFLECTANCE_MULT_BAND_3")))
                    .add(ee.Number(img.get("REFLECTANCE_ADD_BAND_2")))
                    .normalizedDifference(["SR_B5", "SR_B4"])
                    .rename("ndvi")
                    .clip(geometry)
                    .copyProperties(img, ["system:time_start"])
                ))
            )

        landsat8 = landsat_ndvi("LANDSAT/LC08/C02/T1_L2")
        landsat9 = landsat_ndvi("LANDSAT/LC09/C02/T1_L2")

        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(lambda img: (
                img.select("B.*")
                .multiply(0.0001)
                .normalizedDifference(["B8", "B4"])
                .rename("ndvi")
                .clip(geometry)
                .copyProperties(img, ["system:time_start"])
            ))
        )

        ndvi_collection = landsat8.merge(landsat9).merge(sentinel2)

        # ==================================================
        # 3) MONTHLY NDVI
        # ==================================================
        start = ee.Date(start_date)
        end = ee.Date(end_date)

        month_count = end.difference(start, "month").add(1)
        month_starts = ee.List.sequence(0, month_count.subtract(1)).map(
            lambda m: start.advance(m, "month")
        )

        def monthly_composite(date):
            date = ee.Date(date)
            col = ndvi_collection.filterDate(date, date.advance(1, "month"))

            img = ee.Image(
                ee.Algorithms.If(
                    col.size().gt(0),
                    col.median(),
                    ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))
                )
            ).clip(geometry)

            return img.set("system:time_start", date.millis())

        monthly_ndvi = ee.ImageCollection(month_starts.map(monthly_composite))

        # ==================================================
        # 4) NDVI AREA (SUGARCANE ONLY, NON-CANOPY)
        # ==================================================
        def area_calc(img):
            canopy = (
                ee.ImageCollection(
                    "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
                )
                .mosaic()
                .gt(2)
                .clip(geometry)
            )

            mask = img.gte(ndvi_threshold).And(canopy.Not())

            area = (
                ee.Image.pixelArea()
                .divide(10000)
                .updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1e13
                )
                .get("area")
            )

            return ee.Feature(None, {
                "month": ee.Date(img.get("system:time_start")).format("YYYY-MM"),
                "area_ha": area
            })

        ndvi_area_fc = ee.FeatureCollection(monthly_ndvi.map(area_calc))

        # ==================================================
        # 5) FINAL TILE IMAGE (CANOPY REMOVED)
        # ==================================================
        mean_ndvi = monthly_ndvi.mean()

        canopy = (
            ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
            )
            .mosaic()
            .rename("height")
            .clip(geometry)
        )

        non_canopy = canopy.lte(2)

        sugarcane_mask = (
            mean_ndvi.gte(ndvi_threshold)
            .And(non_canopy)
            .selfMask()
        )

        moderate_ndvi_mask = (
            mean_ndvi.gte(0.30)
            .And(mean_ndvi.lte(0.34))
            .And(non_canopy)
            .selfMask()
        )

        black_bg = ee.Image.constant(0).clip(geometry).visualize(
            palette=["000000"]
        )

        sugarcane_vis = sugarcane_mask.visualize(
            palette=["00FF00"]
        )

        moderate_ndvi_vis = moderate_ndvi_mask.visualize(
            palette=["0000FF"]
        )

        boundary = (
            ee.Image()
            .byte()
            .paint(
                ee.FeatureCollection([ee.Feature(geometry)]),
                1,
                2
            )
            .visualize(palette=["FFFFFF"])
        )

        final_viz = (
            black_bg
            .blend(moderate_ndvi_vis)
            .blend(sugarcane_vis)
            .blend(boundary)
        )

        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 6) EXPORT
        # ==================================================
        export_name = f"{plot_name}_{start_date}_{end_date}"

        task = ee.batch.Export.image.toDrive(
            image=final_viz,
            description=export_name,
            folder="sugarcane_ndvi_exports",
            fileNamePrefix=export_name,
            region=geometry,
            scale=30,
            maxPixels=1e13,
            fileFormat="GEO_TIFF"
        )
        task.start()

        # ==================================================
        # 7) RESPONSE
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "data_source": "Landsat 8/9 + Sentinel-2 NDVI (Canopy removed)",
                    "canopy_rule": "height > 2 m excluded",
                    "ndvi_threshold": ndvi_threshold,
                    "last_updated": datetime.now().isoformat()
                }
            }],
            "ndvi_area_summary": ndvi_area_fc.getInfo()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"NDVI detection failed: {e}")

@app.post("/ndvi-sugarcane-detection1")
async def ndvi_sugarcane_detection(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    start_date: str = Query(...),
    end_date: str = Query(...),
    ndvi_threshold: float = Query(0.72),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) NDVI COLLECTIONS
        # ==================================================
        def landsat_ndvi(collection_id):
            return (
                ee.ImageCollection(collection_id)
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .map(lambda img: (
                    img.select("SR.*")
                    .multiply(ee.Number(img.get("REFLECTANCE_MULT_BAND_3")))
                    .add(ee.Number(img.get("REFLECTANCE_ADD_BAND_2")))
                    .normalizedDifference(["SR_B5", "SR_B4"])
                    .rename("ndvi")
                    .clip(geometry)
                    .copyProperties(img, ["system:time_start"])
                ))
            )

        landsat8 = landsat_ndvi("LANDSAT/LC08/C02/T1_L2")
        landsat9 = landsat_ndvi("LANDSAT/LC09/C02/T1_L2")

        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(lambda img: (
                img.select("B.*")
                .multiply(0.0001)
                .normalizedDifference(["B8", "B4"])
                .rename("ndvi")
                .clip(geometry)
                .copyProperties(img, ["system:time_start"])
            ))
        )

        ndvi_collection = landsat8.merge(landsat9).merge(sentinel2)

        # ==================================================
        # 3) MONTHLY NDVI
        # ==================================================
        start = ee.Date(start_date)
        end = ee.Date(end_date)

        month_count = end.difference(start, "month").add(1)
        month_starts = ee.List.sequence(0, month_count.subtract(1)).map(
            lambda m: start.advance(m, "month")
        )

        def monthly_composite(date):
            date = ee.Date(date)
            col = ndvi_collection.filterDate(date, date.advance(1, "month"))

            img = ee.Image(
                ee.Algorithms.If(
                    col.size().gt(0),
                    col.median(),
                    ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))
                )
            ).clip(geometry)

            return img.set("system:time_start", date.millis())

        monthly_ndvi = ee.ImageCollection(month_starts.map(monthly_composite))

        # ==================================================
        # 4) NDVI AREA CALCULATION (SUGARCANE ONLY)
        # ==================================================
        def area_calc(img):
            mask = img.gte(ndvi_threshold)
            area = (
                ee.Image.pixelArea()
                .divide(10000)
                .updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1e13
                )
                .get("area")
            )

            return ee.Feature(None, {
                "month": ee.Date(img.get("system:time_start")).format("YYYY-MM"),
                "area_ha": area
            })

        ndvi_area_fc = ee.FeatureCollection(monthly_ndvi.map(area_calc))

        # ==================================================
        # 5) FINAL TILE IMAGE (VISUAL)
        # ==================================================
        mean_ndvi = monthly_ndvi.mean()

        # High NDVI (Sugarcane)
        sugarcane_mask = mean_ndvi.gte(ndvi_threshold).selfMask()

        # NDVI 0.30–0.34 (Blue)
        moderate_ndvi_mask = (
            mean_ndvi.gte(0.30)
            .And(mean_ndvi.lte(0.34))
            .selfMask()
        )

        black_bg = ee.Image.constant(0).clip(geometry).visualize(
            min=0,
            max=1,
            palette=["000000"]
        )

        sugarcane_vis = sugarcane_mask.visualize(
            palette=["00FF00"]  # Green
        )

        moderate_ndvi_vis = moderate_ndvi_mask.visualize(
            palette=["0000FF"]  # Blue
        )

        boundary = (
            ee.Image()
            .byte()
            .paint(
                featureCollection=ee.FeatureCollection([ee.Feature(geometry)]),
                color=1,
                width=2
            )
            .visualize(palette=["FFFFFF"])
        )

        final_viz = (
            black_bg
            .blend(moderate_ndvi_vis)
            .blend(sugarcane_vis)
            .blend(boundary)
        )

        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 6) EXPORT GEO-TIFF TO GOOGLE DRIVE
        # ==================================================
        export_name = f"{plot_name}_{start_date}_{end_date}"

        task = ee.batch.Export.image.toDrive(
            image=final_viz,
            description=export_name,
            folder="sugarcane_ndvi_exports",
            fileNamePrefix=export_name,
            region=geometry,
            scale=30,
            maxPixels=1e13,
            fileFormat="GEO_TIFF"
        )

        task.start()

        # ==================================================
        # 7) RESPONSE
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "data_source": "Landsat 8/9 + Sentinel-2 NDVI",
                    "start_date": start_date,
                    "end_date": end_date,
                    "ndvi_threshold": ndvi_threshold,
                    "ndvi_classes": {
                        "sugarcane": f">= {ndvi_threshold} (green)",
                        "moderate_vegetation": "0.30–0.34 (blue)"
                    },
                    "drive_export": {
                        "status": "STARTED",
                        "folder": "sugarcane_ndvi_exports",
                        "file_prefix": export_name,
                        "file_format": "GEO_TIFF",
                        "ee_account_note": "Exported to the same Google Drive account used for Earth Engine initialization"
                    },
                    "last_updated": datetime.now().isoformat(),
                }
            }],
            "ndvi_area_summary": ndvi_area_fc.getInfo()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"NDVI detection failed: {e}")

@app.post("/NDWIDetection1")
async def detect_ndwi(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    ),
    start_date: str = Depends(default_start_date),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION (same as your WaterDetection1)
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "District boundary not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Subdistrict boundary not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()

            if not row:
                raise HTTPException(404, "Village boundary not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # Sentinel-2 SR Collection
        # ==================================================
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
        )

        size = collection.size().getInfo()
        if size == 0:
            raise HTTPException(404, "No Sentinel-2 images found")

        sorted_col = collection.sort("system:time_start", False)
        latest = ee.Image(sorted_col.first())
        latest_date = (
            ee.Date(latest.get("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )

        image = collection.median().clip(geometry)

        # ==================================================
        # NDWI = (Green - NIR) / (Green + NIR)
        # Bands: B3 (Green), B8 (NIR)
        # ==================================================
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

        # ==================================================
        # Water mask: NDWI > 0.3
        # ==================================================
        water_mask = ndwi.gt(0.15)

        # ==================================================
        # Visualization
        # ==================================================
        vis = {"min": 0, "max": 1, "palette": ["0000ff"]}
        visual = water_mask.selfMask().visualize(**vis).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # Pixel counting
        # ==================================================
        scale = 10
        ones = ee.Image.constant(1)

        total_pixels = (
            ones.reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
            .get("constant")
            .getInfo()
        )

        water_pixels = (
            ones.updateMask(water_mask)
            .reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
            .get("constant")
            .getInfo()
            or 0
        )

        # ==================================================
        # GeoJSON Feature
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "latest_image_date": latest_date,
                "image_count": size,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,
                "water_pixel_count": water_pixels,
                "water_pixel_percentage": (
                    round((water_pixels / total_pixels) * 100, 2)
                    if total_pixels
                    else 0
                ),
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NDWI detection failed: {str(e)}",
        )
# @app.post("/NDWIDetection1")
# async def detect_ndmi(
#     district: str = Query(...),
#     subdistrict: str | None = Query(None),
#     village: str | None = Query(None),
#     end_date: str = Query(
#         default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
#     ),
#     start_date: str = Depends(default_start_date),
#     db: Session = Depends(get_db),
# ):
#     try:
#         # ==================================================
#         # GEOMETRY SELECTION (unchanged)
#         # ==================================================
#         if district and not subdistrict and not village:
#             row = db.execute(
#                 text("""
#                     SELECT d.district_name,
#                            ST_AsGeoJSON(d.geom)::json AS geometry
#                     FROM districts d
#                     WHERE LOWER(d.district_name) = LOWER(:district)
#                     LIMIT 1
#                 """),
#                 {"district": district},
#             ).mappings().first()

#             if not row:
#                 raise HTTPException(404, "District boundary not found")

#             plot_name = row["district_name"]
#             geometry = ee.Geometry(row["geometry"])

#         elif district and subdistrict and not village:
#             row = db.execute(
#                 text("""
#                     SELECT v.sub_dist,
#                            ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
#                     FROM village v
#                     JOIN village_boundaries b ON b.village_id = v.id
#                     WHERE LOWER(v.district) = LOWER(:district)
#                       AND LOWER(v.sub_dist) = LOWER(:subdistrict)
#                     GROUP BY v.sub_dist
#                 """),
#                 {"district": district, "subdistrict": subdistrict},
#             ).mappings().first()

#             if not row:
#                 raise HTTPException(404, "Subdistrict boundary not found")

#             plot_name = row["sub_dist"]
#             geometry = ee.Geometry(row["geometry"])

#         elif district and subdistrict and village:
#             row = db.execute(
#                 text("""
#                     SELECT v.village_name,
#                            ST_AsGeoJSON(b.geom)::json AS geometry
#                     FROM village v
#                     JOIN village_boundaries b ON b.village_id = v.id
#                     WHERE LOWER(v.district) = LOWER(:district)
#                       AND LOWER(v.sub_dist) = LOWER(:subdistrict)
#                       AND LOWER(v.village_name) = LOWER(:village)
#                     LIMIT 1
#                 """),
#                 {
#                     "district": district,
#                     "subdistrict": subdistrict,
#                     "village": village,
#                 },
#             ).mappings().first()

#             if not row:
#                 raise HTTPException(404, "Village boundary not found")

#             plot_name = row["village_name"]
#             geometry = ee.Geometry(row["geometry"])

#         else:
#             raise HTTPException(400, "Invalid input combination")

#         # ==================================================
#         # Sentinel-2 SR Collection
#         # ==================================================
#         collection = (
#             ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
#             .filterBounds(geometry)
#             .filterDate(start_date, end_date)
#             .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
#         )

#         size = collection.size().getInfo()
#         if size == 0:
#             raise HTTPException(404, "No Sentinel-2 images found")

#         sorted_col = collection.sort("system:time_start", False)
#         latest = ee.Image(sorted_col.first())
#         latest_date = (
#             ee.Date(latest.get("system:time_start"))
#             .format("YYYY-MM-dd")
#             .getInfo()
#         )

#         image = collection.median().clip(geometry)

#         # ==================================================
#         # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
#         # Bands: B8 (NIR), B11 (SWIR1)
#         # ==================================================
#         ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

#         # ==================================================
#         # Moisture mask: NDMI > 0.2
#         # ==================================================
#         moisture_mask = ndmi.gt(0.2)

#         # ==================================================
#         # Visualization
#         # ==================================================
#         vis = {"min": 0, "max": 1, "palette": ["00ffff"]}
#         visual = moisture_mask.selfMask().visualize(**vis).clip(geometry)
#         tile_url = visual.getMapId()["tile_fetcher"].url_format

#         # ==================================================
#         # Pixel counting
#         # ==================================================
#         scale = 10
#         ones = ee.Image.constant(1)

#         total_pixels = (
#             ones.reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
#             .get("constant")
#             .getInfo()
#         )

#         moisture_pixels = (
#             ones.updateMask(moisture_mask)
#             .reduceRegion(ee.Reducer.count(), geometry, scale, bestEffort=True)
#             .get("constant")
#             .getInfo()
#             or 0
#         )

#         # ==================================================
#         # GeoJSON Feature
#         # ==================================================
#         feature = {
#             "type": "Feature",
#             "geometry": {
#                 "type": "Polygon",
#                 "coordinates": row["geometry"]["coordinates"],
#             },
#             "properties": {
#                 "plot_name": plot_name,
#                 "start_date": start_date,
#                 "end_date": end_date,
#                 "latest_image_date": latest_date,
#                 "image_count": size,
#                 "tile_url": tile_url,
#                 "last_updated": datetime.now().isoformat(),
#             },
#         }

#         return {
#             "type": "FeatureCollection",
#             "features": [feature],
#             "pixel_summary": {
#                 "total_pixel_count": total_pixels,
#                 "moisture_pixel_count": moisture_pixels,
#                 "moisture_pixel_percentage": (
#                     round((moisture_pixels / total_pixels) * 100, 2)
#                     if total_pixels
#                     else 0
#                 ),
#             },
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"NDMI detection failed: {str(e)}",
#         )

@app.post("/CanopyHeightAgeStructure")
async def canopy_height_age_structure(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District boundary not found")
            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict boundary not found")
            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village boundary not found")
            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # Load Canopy Height
        # ==================================================
        canopy = (
            ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
            )
            .mosaic()
            .rename("height")
            .clip(geometry)
        )

        forest = canopy.updateMask(canopy.gt(2))  # ignore shrubs/grass

        # ==================================================
        # Height statistics
        # ==================================================
        scale = 100
        stats = forest.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), "", True)
            .combine(ee.Reducer.max(), "", True)
            .combine(ee.Reducer.percentile([99]), "", True),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
        )

        mean_h = stats.get("height_mean").getInfo()
        min_h = stats.get("height_min").getInfo()
        max_h = stats.get("height_max").getInfo()
        p99_h = stats.get("height_p99").getInfo()

        # ==================================================
        # Age-class masks (height → age proxy)
        # ==================================================
        age_classes = {
            "young": canopy.gt(2).And(canopy.lte(6)),
            "mid_age": canopy.gt(6).And(canopy.lte(12)),
            "mature": canopy.gt(12).And(canopy.lte(20)),
            "old_age": canopy.gt(20),
        }

        pixel_area = ee.Image.pixelArea()
        forest_mask = canopy.gt(2)

        total_area = (
            pixel_area.updateMask(forest_mask)
            .reduceRegion(
                ee.Reducer.sum(),
                geometry,
                scale=10,
                bestEffort=True,
                maxPixels=1e13,
            )
            .get("area")
        )

        total_area_val = ee.Number(total_area)

        age_percentages = {}
        for label, mask in age_classes.items():
            area = (
                pixel_area.updateMask(mask)
                .reduceRegion(
                    ee.Reducer.sum(),
                    geometry,
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e13,
                )
                .get("area")
            )
            pct = ee.Number(area).divide(total_area_val).multiply(100)
            age_percentages[label] = pct.getInfo()

        # ==================================================
        # Visualization (Green → Yellow → Orange → Red)
        # ==================================================
        vis = {
            "min": 2,
            "max": 30,
            "palette": [
                "#00FF00",  # Green  - low canopy
                "#FFFF00",  # Yellow - mid canopy
                "#FFA500",  # Orange - tall canopy
                "#FF0000",  # Red    - very tall canopy
            ],
        }

        visual = forest.visualize(**vis).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # GeoJSON Feature
        # ==================================================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": row["geometry"]["coordinates"],
            },
            "properties": {
                "plot_name": plot_name,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }

        # ==================================================
        # Final Response
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [feature],
            "canopy_summary": {
                "mean_height_m": mean_h,
                "min_height_m": min_h,
                "max_height_m": max_h,
                "p99_height_m": p99_h,
                "age_structure_percentages": {
                    "mid_age_percentage": round(age_percentages["mid_age"], 2),
                    "mature_percentage": round(age_percentages["mature"], 2),
                    "old_age_percentage": round(age_percentages["old_age"], 2),
                },
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Canopy height age-structure analysis failed: {str(e)}",
        )
@app.post("/CanopyHeightAgeStructureclasswise")
async def canopy_height_age_structure(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # GEOMETRY SELECTION
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            plot_name = row["district_name"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            plot_name = row["sub_dist"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()
            plot_name = row["village_name"]
        else:
            raise HTTPException(400, "Invalid input combination")

        if not row:
            raise HTTPException(404, "Boundary not found")

        geometry = ee.Geometry(row["geometry"])

        # ==================================================
        # CANOPY HEIGHT IMAGE
        # ==================================================
        canopy = (
            ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
            )
            .mosaic()
            .rename("height")
            .clip(geometry)
        )

        # ==================================================
        # HEIGHT STATISTICS
        # ==================================================
        forest = canopy.updateMask(canopy.gt(2))

        stats = forest.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), "", True)
            .combine(ee.Reducer.max(), "", True)
            .combine(ee.Reducer.percentile([99]), "", True),
            geometry=geometry,
            scale=100,
            bestEffort=True,
            maxPixels=1e13,
        )

        # ==================================================
        # AGE CLASSES (HEIGHT PROXY)
        # ==================================================
        age_classes = {
            "young": {
                "mask": canopy.gt(2).And(canopy.lte(6)),
                "color": "#00FF00",
            },
            "mid_age": {
                "mask": canopy.gt(6).And(canopy.lte(12)),
                "color": "#FFFF00",
            },
            "mature": {
                "mask": canopy.gt(12).And(canopy.lte(20)),
                "color": "#FFA500",
            },
            "old_age": {
                "mask": canopy.gt(20),
                "color": "#FF0000",
            },
        }

        pixel_area = ee.Image.pixelArea()
        class_results = {}

        for cls, cfg in age_classes.items():
            masked = canopy.updateMask(cfg["mask"])

            # ---- AREA (ha)
            area_m2 = (
                pixel_area.updateMask(cfg["mask"])
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometry,
                    scale=10,
                    bestEffort=True,
                    maxPixels=1e13,
                )
                .get("area")
            )

            area_ha = ee.Number(area_m2).divide(10000).getInfo()

            # ---- CLASS-ONLY TILE
            vis = {
                "min": 0,
                "max": 30,
                "palette": [cfg["color"]],
            }

            tile = masked.visualize(**vis).clip(geometry)
            tile_url = tile.getMapId()["tile_fetcher"].url_format

            class_results[cls] = {
                "tile_url": tile_url,
                "area_hectares": round(area_ha, 2),
            }

        # ==================================================
        # FINAL RESPONSE
        # ==================================================
        return {
            "plot_name": plot_name,
            "geometry": row["geometry"],
            "canopy_summary": {
                "mean_height_m": stats.get("height_mean").getInfo(),
                "min_height_m": stats.get("height_min").getInfo(),
                "max_height_m": stats.get("height_max").getInfo(),
                "p99_height_m": stats.get("height_p99").getInfo(),
            },
            "age_classes": class_results,
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Canopy height age-structure analysis failed: {str(e)}",
        )

@app.post("/boundary-to-shapefile")
async def boundary_to_shapefile(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # DETERMINE OUTPUT NAME (PRIORITY-BASED)
        # ==================================================
        raw_name = village or subdistrict or district
        output_name = (
            raw_name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        # ==================================================
        # GEOMETRY SELECTION FROM DATABASE
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name AS name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist AS name,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name AS name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()

        else:
            raise HTTPException(400, "Invalid input combination")

        if not row:
            raise HTTPException(404, "Boundary not found")

        # ==================================================
        # BUILD GEODATAFRAME
        # ==================================================
        geom = shape(row["geometry"])

        gdf = gpd.GeoDataFrame(
            [{
                "name": row["name"],
                "district": district,
                "subdistrict": subdistrict,
                "village": village,
                "created_at": datetime.utcnow().isoformat(),
            }],
            geometry=[geom],
            crs="EPSG:4326",
        )

        # ==================================================
        # WRITE SHAPEFILE (WINDOWS SAFE)
        # ==================================================
        tmp_dir = tempfile.mkdtemp()

        shp_path = os.path.join(tmp_dir, f"{output_name}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")

        zip_tmp = tempfile.NamedTemporaryFile(
            suffix=".zip",
            delete=False,
        )
        zip_path = zip_tmp.name
        zip_tmp.close()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                f = shp_path.replace(".shp", ext)
                if os.path.exists(f):
                    zipf.write(f, arcname=os.path.basename(f))

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{output_name}_boundary.zip",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Shapefile generation failed: {str(e)}",
        )
@app.post("/boundary-to-kml")
async def boundary_to_kml(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # DETERMINE OUTPUT NAME (PRIORITY-BASED)
        # village > subdistrict > district
        # ==================================================
        raw_name = village or subdistrict or district
        output_name = (
            raw_name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        # ==================================================
        # GEOMETRY SELECTION FROM DATABASE
        # ==================================================
        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT d.district_name AS name,
                           ST_AsGeoJSON(d.geom)::json AS geometry
                    FROM districts d
                    WHERE LOWER(d.district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist AS name,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name AS name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {
                    "district": district,
                    "subdistrict": subdistrict,
                    "village": village,
                },
            ).mappings().first()

        else:
            raise HTTPException(400, "Invalid input combination")

        if not row:
            raise HTTPException(404, "Boundary not found")

        # ==================================================
        # BUILD GEODATAFRAME
        # ==================================================
        geom = shape(row["geometry"])

        gdf = gpd.GeoDataFrame(
            [{
                "name": row["name"],
                "district": district,
                "subdistrict": subdistrict,
                "village": village,
                "created_at": datetime.utcnow().isoformat(),
            }],
            geometry=[geom],
            crs="EPSG:4326",
        )

        # ==================================================
        # WRITE KML (WINDOWS SAFE)
        # ==================================================
        tmp_kml = tempfile.NamedTemporaryFile(
            suffix=".kml",
            delete=False,
        )
        kml_path = tmp_kml.name
        tmp_kml.close()

        gdf.to_file(kml_path, driver="KML")

        return FileResponse(
            kml_path,
            media_type="application/vnd.google-earth.kml+xml",
            filename=f"{output_name}.kml",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"KML generation failed: {str(e)}",
        )

# @app.post("/ndvi-sugarcane-detection")
# async def ndvi_sugarcane_detection(
#     district: str = Query(...),
#     subdistrict: str | None = Query(None),
#     village: str | None = Query(None),
#     start_date: str = Query(...),
#     end_date: str = Query(...),
#     ndvi_threshold: float = Query(0.6),
#     db: Session = Depends(get_db),
# ):
#     try:
#         # ==================================================
#         # 1) GEOMETRY SELECTION (UNCHANGED)
#         # ==================================================
#         geometry = None
#         response_geometry = None
#         plot_name = None

#         if district and not subdistrict and not village:
#             row = db.execute(
#                 text("""
#                     SELECT district_name,
#                            ST_AsGeoJSON(geom)::json AS geometry
#                     FROM districts
#                     WHERE LOWER(district_name) = LOWER(:district)
#                     LIMIT 1
#                 """),
#                 {"district": district},
#             ).mappings().first()
#             if not row:
#                 raise HTTPException(404, "District not found")

#             plot_name = row["district_name"]
#             geometry = ee.Geometry(row["geometry"])
#             response_geometry = row["geometry"]

#         elif district and subdistrict and not village:
#             row = db.execute(
#                 text("""
#                     SELECT v.sub_dist,
#                            ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
#                     FROM village v
#                     JOIN village_boundaries b ON b.village_id = v.id
#                     WHERE LOWER(v.district) = LOWER(:district)
#                       AND LOWER(v.sub_dist) = LOWER(:subdistrict)
#                     GROUP BY v.sub_dist
#                 """),
#                 {"district": district, "subdistrict": subdistrict},
#             ).mappings().first()
#             if not row:
#                 raise HTTPException(404, "Subdistrict not found")

#             plot_name = row["sub_dist"]
#             geometry = ee.Geometry(row["geometry"])
#             response_geometry = row["geometry"]

#         elif district and subdistrict and village:
#             row = db.execute(
#                 text("""
#                     SELECT v.village_name,
#                            ST_AsGeoJSON(b.geom)::json AS geometry
#                     FROM village v
#                     JOIN village_boundaries b ON b.village_id = v.id
#                     WHERE LOWER(v.district) = LOWER(:district)
#                       AND LOWER(v.sub_dist) = LOWER(:subdistrict)
#                       AND LOWER(v.village_name) = LOWER(:village)
#                     LIMIT 1
#                 """),
#                 {"district": district, "subdistrict": subdistrict, "village": village},
#             ).mappings().first()
#             if not row:
#                 raise HTTPException(404, "Village not found")

#             plot_name = row["village_name"]
#             geometry = ee.Geometry(row["geometry"])
#             response_geometry = row["geometry"]

#         else:
#             raise HTTPException(400, "Invalid input combination")

#         # ==================================================
#         # 2) NDVI COLLECTIONS
#         # ==================================================
#         def landsat_ndvi(collection_id):
#             return (
#                 ee.ImageCollection(collection_id)
#                 .filterDate(start_date, end_date)
#                 .filterBounds(geometry)
#                 .map(lambda img: (
#                     img.select("SR.*")
#                     .multiply(ee.Number(img.get("REFLECTANCE_MULT_BAND_3")))
#                     .add(ee.Number(img.get("REFLECTANCE_ADD_BAND_2")))
#                     .normalizedDifference(["SR_B5", "SR_B4"])
#                     .rename("ndvi")
#                     .clip(geometry)
#                     .copyProperties(img, ["system:time_start"])
#                 ))
#             )

#         landsat8 = landsat_ndvi("LANDSAT/LC08/C02/T1_L2")
#         landsat9 = landsat_ndvi("LANDSAT/LC09/C02/T1_L2")

#         sentinel2 = (
#             ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
#             .filterDate(start_date, end_date)
#             .filterBounds(geometry)
#             .map(lambda img: (
#                 img.select("B.*")
#                 .multiply(0.0001)
#                 .normalizedDifference(["B8", "B4"])
#                 .rename("ndvi")
#                 .clip(geometry)
#                 .copyProperties(img, ["system:time_start"])
#             ))
#         )

#         ndvi_collection = (
#             landsat8.merge(landsat9).merge(sentinel2)
#             .sort("system:time_start")
#         )

#         # ==================================================
#         # 3) MONTH LIST STRICTLY FROM DATE RANGE
#         # ==================================================
#         start = ee.Date(start_date)
#         end = ee.Date(end_date)

#         month_count = end.difference(start, "month").add(1)

#         month_starts = ee.List.sequence(0, month_count.subtract(1)).map(
#             lambda m: start.advance(m, "month")
#         )

#         # ==================================================
#         # 4) MONTHLY NDVI (ONLY DATE RANGE)
#         # ==================================================
#         def monthly_composite(date):
#             date = ee.Date(date)
#             month_start = date
#             month_end = date.advance(1, "month")

#             col = ndvi_collection.filterDate(month_start, month_end)

#             img = ee.Image(
#                 ee.Algorithms.If(
#                     col.size().gt(0),
#                     col.median(),
#                     ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))
#                 )
#             ).clip(geometry)

#             return img.set({
#                 "system:time_start": month_start.millis(),
#                 "system:index": month_start.format("YYYY-MM")
#             })

#         monthly_ndvi = ee.ImageCollection(month_starts.map(monthly_composite))

#         # ==================================================
#         # 5) NDVI AREA (HECTARES)
#         # ==================================================
#         def area_calc(img):
#             veg_mask = img.select("ndvi").gte(ndvi_threshold)

#             area = (
#                 ee.Image.pixelArea()
#                 .divide(10000)
#                 .updateMask(veg_mask)
#                 .reduceRegion(
#                     reducer=ee.Reducer.sum(),
#                     geometry=geometry,
#                     scale=30,
#                     maxPixels=1e13
#                 )
#                 .get("area")
#             )

#             return ee.Feature(None, {
#                 "month": ee.Date(img.get("system:time_start")).format("YYYY-MM"),
#                 "ndvi_threshold": ndvi_threshold,
#                 "area_ha": area
#             })

#         ndvi_area_fc = ee.FeatureCollection(monthly_ndvi.map(area_calc))

#         # ==================================================
#         # 6) TILE (MEAN OF RANGE ONLY)
#         # ==================================================
#         sugarcane_mask = monthly_ndvi.mean().gte(ndvi_threshold).selfMask()
# # --- RGB BASE (Sentinel-2 True Color) ---
#         rgb_base = (
#             ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
#             .filterDate(start_date, end_date)
#             .filterBounds(geometry)
#             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
#             .median()
#             .clip(geometry)
#         )

#         # rgb_vis = rgb_base.visualize(
#         #     bands=["B4", "B3", "B2"],  # True color
#         #     min=0,
#         #     max=3000
#         # )
#         # --- Black background ---
#         black_bg = ee.Image.constant(0).clip(geometry).visualize(
#                 min=0,
#                 max=1,
#                 palette=["000000"]
#             )

#         # --- Sugarcane Overlay (GREEN ONLY) ---
#         sugarcane_vis = sugarcane_mask.visualize(
#             palette=["00FF00"]
#         )

#         # --- Blend RGB + Sugarcane ---
#         final_viz = black_bg.blend(sugarcane_vis)

#         tile_url = final_viz.getMapId()["tile_fetcher"].url_format

#         # ==================================================
#         # 7) RESPONSE
#         # ==================================================
#         return {
#             "type": "FeatureCollection",
#             "features": [{
#                 "type": "Feature",
#                 "geometry": response_geometry,
#                 "properties": {
#                     "plot_id": plot_name,
#                     "tile_url": tile_url,
#                     "data_source": "Landsat 8/9 + Sentinel-2 NDVI",
#                     "start_date": start_date,
#                     "end_date": end_date,
#                     "ndvi_threshold": ndvi_threshold,
#                     "last_updated": datetime.now().isoformat(),
#                 }
#             }],
#             "ndvi_area_summary": ndvi_area_fc.getInfo()
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(500, f"NDVI detection failed: {e}")


@app.post("/ndvi-sugarcane-detection")
async def ndvi_sugarcane_detection(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    month: int | None = Query(None, ge=1, le=12),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION (UNCHANGED)
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) MONTH RESOLUTION
        # ==================================================
        now = datetime.utcnow()
        resolved_month = month if month is not None else now.month
        resolved_year = now.year

        start_date = datetime(resolved_year, resolved_month, 1)
        end_date = datetime(
            resolved_year + (resolved_month == 12),
            1 if resolved_month == 12 else resolved_month + 1,
            1
        )

        # ==================================================
        # 3) NDVI RANGE BY MONTH (FIXED)
        # ==================================================
        ndvi_ranges = {
            10: (0.63, 0.65),  # October
            11: (0.63, 0.67),  # November
            12: (0.63, 0.69),  # December
            1: (0.63, 0.71)
        }

        # January–September fallback
        ndvi_min, ndvi_max = ndvi_ranges.get(resolved_month, (0.60, 0.63))

        # ==================================================
        # 4) NDVI COLLECTIONS
        # ==================================================
        def landsat_ndvi(collection_id):
            return (
                ee.ImageCollection(collection_id)
                .filterDate(start_date.isoformat(), end_date.isoformat())
                .filterBounds(geometry)
                .map(lambda img: (
                    img.select("SR.*")
                    .multiply(ee.Number(img.get("REFLECTANCE_MULT_BAND_3")))
                    .add(ee.Number(img.get("REFLECTANCE_ADD_BAND_2")))
                    .normalizedDifference(["SR_B5", "SR_B4"])
                    .rename("ndvi")
                    .clip(geometry)
                ))
            )

        landsat8 = landsat_ndvi("LANDSAT/LC08/C02/T1_L2")
        landsat9 = landsat_ndvi("LANDSAT/LC09/C02/T1_L2")

        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filterBounds(geometry)
            .map(lambda img: (
                img.select("B.*")
                .multiply(0.0001)
                .normalizedDifference(["B8", "B4"])
                .rename("ndvi")
                .clip(geometry)
            ))
        )

        ndvi_collection = landsat8.merge(landsat9).merge(sentinel2)

        # ==================================================
        # 5) SAFE MONTHLY NDVI
        # ==================================================
        monthly_ndvi = ee.Image(
            ee.Algorithms.If(
                ndvi_collection.size().gt(0),
                ndvi_collection.median(),
                ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))
            )
        )

        has_imagery = ndvi_collection.size().gt(0)

        # ==================================================
        # 6) NDVI MASK & AREA
        # ==================================================
        ndvi_mask = monthly_ndvi.gte(ndvi_min).And(monthly_ndvi.lte(ndvi_max))

        area_ha = (
            ee.Image.pixelArea()
            .divide(10000)
            .updateMask(ndvi_mask)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=30,
                maxPixels=1e13
            )
            .get("area")
        )

        # ==================================================
        # 7) VISUALIZATION
        # ==================================================
        final_viz = (
            ee.Image.constant(0)
            .clip(geometry)
            .visualize(palette=["000000"])
            .blend(ndvi_mask.selfMask().visualize(palette=["00FF00"]))
        )

        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 8) EXPORT
        # ==================================================
        export_name = f"{plot_name}_{resolved_year}_{resolved_month}"

        ee.batch.Export.image.toDrive(
            image=final_viz,
            description=export_name,
            folder="sugarcane_ndvi_exports",
            fileNamePrefix=export_name,
            region=geometry,
            scale=30,
            maxPixels=1e13,
            fileFormat="GEO_TIFF"
        ).start()

        # ==================================================
        # 9) RESPONSE
        # ==================================================
        return {
            "plot_id": plot_name,
            "year": resolved_year,
            "month": calendar.month_name[resolved_month],
            "ndvi_range": [ndvi_min, ndvi_max],
            "has_imagery": has_imagery.getInfo(),
            "area_ha": area_ha.getInfo(),
            "tile_url": tile_url,
            "export": "STARTED",
            "last_updated": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"NDVI detection failed: {e}")

@app.post("/land-surface-temperature")
async def land_surface_temperature(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    start_date: str = Query(...),
    end_date: str = Query(...),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION (UNCHANGED)
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) LANDSAT LST COLLECTION (L8 + L9)
        # ==================================================
        def landsat_lst(collection_id):
            return (
                ee.ImageCollection(collection_id)
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .filter(ee.Filter.lt("CLOUD_COVER", 10))
                .select("ST_B10")
                .map(lambda img: (
                    img.multiply(0.00341802)
                       .add(149.0)
                       .subtract(273.15)
                       .rename("LST_C")
                       .clip(geometry)
                       .copyProperties(img, ["system:time_start"])
                ))
            )

        lst_collection = (
            landsat_lst("LANDSAT/LC08/C02/T1_L2")
            .merge(landsat_lst("LANDSAT/LC09/C02/T1_L2"))
        )

        # ==================================================
        # 3) LAST 12 MONTHS COMPOSITES
        # ==================================================
        end = ee.Date(end_date)
        start = end.advance(-11, "month")

        months = ee.List.sequence(0, 11).map(
            lambda m: start.advance(m, "month")
        )

        def monthly_lst(date):
            date = ee.Date(date)
            col = lst_collection.filterDate(date, date.advance(1, "month"))

            img = ee.Image(
                ee.Algorithms.If(
                    col.size().gt(0),
                    col.mean(),
                    ee.Image.constant(0).rename("LST_C").updateMask(ee.Image(0))
                )
            )

            mean_val = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e13
            ).get("LST_C")

            return ee.Feature(None, {
                "month": date.format("YYYY-MM"),
                "mean_lst_c": mean_val
            })

        lst_timeseries_fc = ee.FeatureCollection(months.map(monthly_lst))

        # ==================================================
        # 4) TILE VISUALIZATION (MEAN OF 12 MONTHS)
        # ==================================================
        lst_mean = lst_collection.mean().clip(geometry)

        lst_vis = {
            "min": 20,
            "max": 45,
            "palette": ["0000FF", "00FFFF", "00FF00", "FFFF00", "FFA500", "FF0000"]
        }

        lst_viz = lst_mean.visualize(**lst_vis)

        boundary = (
            ee.Image()
            .byte()
            .paint(
                featureCollection=ee.FeatureCollection([ee.Feature(geometry)]),
                color=1,
                width=2
            )
            .visualize(palette=["FFFFFF"])
        )

        final_viz = lst_viz.blend(boundary)
        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 5) RESPONSE
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "data_source": "Landsat 8/9 C2 L2 – LST",
                    "start_date": start_date,
                    "end_date": end_date,
                    "last_updated": datetime.now().isoformat(),
                }
            }],
            "lst_monthly_timeseries": lst_timeseries_fc.getInfo()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"LST processing failed: {e}")

@app.post("/methane-concentration")
async def methane_concentration(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    start_date: str = Query(...),
    end_date: str = Query(...),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) SENTINEL-5P CH4 COLLECTION (FIXED)
        # ==================================================
        ch4_collection = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
            .select("CH4_column_volume_mixing_ratio_dry_air")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .map(lambda img: (
                img.rename("CH4")
                   .clip(geometry)
                   .copyProperties(img, ["system:time_start"])
            ))
        )

        # ==================================================
        # 3) LAST 12 MONTHS TIME SERIES
        # ==================================================
        end = ee.Date(end_date)
        start = end.advance(-11, "month")

        months = ee.List.sequence(0, 11).map(
            lambda m: start.advance(m, "month")
        )

        def monthly_ch4(date):
            date = ee.Date(date)
            col = ch4_collection.filterDate(date, date.advance(1, "month"))

            img = ee.Image(
                ee.Algorithms.If(
                    col.size().gt(0),
                    col.mean(),
                    ee.Image.constant(0).rename("CH4").updateMask(ee.Image(0))
                )
            )

            mean_val = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10000,
                maxPixels=1e13
            ).get("CH4")

            return ee.Feature(None, {
                "month": date.format("YYYY-MM"),
                "mean_ch4_ppb": mean_val
            })

        ch4_timeseries_fc = ee.FeatureCollection(months.map(monthly_ch4))

        # ==================================================
        # 4) TILE VISUALIZATION
        # ==================================================
        ch4_mean = ch4_collection.mean().clip(geometry)

        ch4_vis = {
            "min": 1750,
            "max": 1900,
            "palette": [
                "000000", "0000FF", "800080",
                "00FFFF", "00FF00", "FFFF00", "FF0000"
            ]
        }

        ch4_viz = ch4_mean.visualize(**ch4_vis)

        boundary = (
            ee.Image()
            .byte()
            .paint(
                featureCollection=ee.FeatureCollection([ee.Feature(geometry)]),
                color=1,
                width=2
            )
            .visualize(palette=["FFFFFF"])
        )

        final_viz = ch4_viz.blend(boundary)
        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 5) RESPONSE
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "data_source": "Sentinel-5P OFFL L3 CH₄",
                    "unit": "ppb",
                    "start_date": start_date,
                    "end_date": end_date,
                    "last_updated": datetime.now().isoformat(),
                }
            }],
            "ch4_monthly_timeseries": ch4_timeseries_fc.getInfo()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Methane processing failed: {e}")

# -------------------------------
# Caches
# -------------------------------
current_cache = TTLCache(maxsize=4000, ttl=3600)

et_cache = TTLCache(maxsize=2000, ttl=3600)

# -------------------------------
# API CONFIG
# -------------------------------
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

API_KEY = "69a7059d01e243e4b7293109261201"
WEATHERAPI_URL = "https://api.weatherapi.com/v1/current.json"

# ============================================================
# CURRENT WEATHER (WEATHERAPI.COM)
# ============================================================
@app.get("/current-weather")
async def get_curr_weather(
    request: Request,
    lat: float = Query(None, description="Latitude"),
    lon: float = Query(None, description="Longitude"),
    city: str = Query(None, description="City name (optional if lat/lon given)")
):
    # Build query
    if lat is not None and lon is not None:
        q = f"{lat},{lon}"
    elif city:
        q = city
    else:
        raise HTTPException(400, "Provide either city or lat/lon")

    # Current local hour (12:46 → 12)
    now = datetime.now()
    current_hour = now.hour

    cache_key = f"{q}_{now.date()}_{current_hour}"
    if cache_key in current_cache:
        return current_cache[cache_key]

    params = {
        "key": API_KEY,
        "q": q,
        "aqi": "yes"
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(WEATHERAPI_URL, params=params)
        r.raise_for_status()
        data = r.json()

    current = data["current"]

    response = {
        "location": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "date": now.date().isoformat(),
        "hour": current_hour,

        # Current hour snapshot
        "temperature_c": current["temp_c"],
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"],
        "precip_mm": current["precip_mm"],
        "condition": current["condition"]["text"]
    }

    current_cache[cache_key] = response
    return response


# ============================================================
# ET LOGIC (OPEN-METEO)
# ============================================================
async def fetch_current_hour_et(lat: float, lon: float):
    """Fetch ET for current hour only (12:46 → 12)"""
    now = datetime.utcnow()
    hour = now.hour
    today = now.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "et0_fao_evapotranspiration",
        "start_date": today,
        "end_date": today,
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        times = data["hourly"]["time"]
        values = data["hourly"]["et0_fao_evapotranspiration"]

        for t, v in zip(times, values):
            if int(t.split("T")[1].split(":")[0]) == hour:
                return round(v, 2) if v is not None else None

        return None

    except Exception as e:
        print(f"Error fetching current hour ET: {e}")
        return None


# ============================================================
# ET ENDPOINT
# ============================================================
@app.get("/compute-et")
async def compute_et(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    today = date.today().strftime("%Y-%m-%d")
    hour = datetime.now().hour

    cache_key = f"et_{lat}_{lon}_{today}_{hour}"
    if cache_key in et_cache:
        return et_cache[cache_key]

    et_value = await fetch_current_hour_et(lat, lon)

    if et_value is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch current hour ET from Open-Meteo"
        )

    response = {
        "date": today,
        "hour": hour,
        "latitude": lat,
        "longitude": lon,
        "ET_current_hour_mm": et_value
    }

    et_cache[cache_key] = response
    return JSONResponse(response)


@app.post("/ndvi-sugarcane-detection123")
async def ndvi_sugarcane_detection(
    district: str = Query(...),
    subdistrict: str | None = Query(None),
    village: str | None = Query(None),
    start_date: str = Query(...),
    end_date: str = Query(...),
    ndvi_threshold: float = Query(0.72),
    db: Session = Depends(get_db),
):
    try:
        # ==================================================
        # 1) GEOMETRY SELECTION
        # ==================================================
        geometry = None
        response_geometry = None
        plot_name = None

        if district and not subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT district_name,
                           ST_AsGeoJSON(geom)::json AS geometry
                    FROM districts
                    WHERE LOWER(district_name) = LOWER(:district)
                    LIMIT 1
                """),
                {"district": district},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "District not found")

            plot_name = row["district_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and not village:
            row = db.execute(
                text("""
                    SELECT v.sub_dist,
                           ST_AsGeoJSON(ST_Union(b.geom))::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                    GROUP BY v.sub_dist
                """),
                {"district": district, "subdistrict": subdistrict},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Subdistrict not found")

            plot_name = row["sub_dist"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        elif district and subdistrict and village:
            row = db.execute(
                text("""
                    SELECT v.village_name,
                           ST_AsGeoJSON(b.geom)::json AS geometry
                    FROM village v
                    JOIN village_boundaries b ON b.village_id = v.id
                    WHERE LOWER(v.district) = LOWER(:district)
                      AND LOWER(v.sub_dist) = LOWER(:subdistrict)
                      AND LOWER(v.village_name) = LOWER(:village)
                    LIMIT 1
                """),
                {"district": district, "subdistrict": subdistrict, "village": village},
            ).mappings().first()
            if not row:
                raise HTTPException(404, "Village not found")

            plot_name = row["village_name"]
            geometry = ee.Geometry(row["geometry"])
            response_geometry = row["geometry"]

        else:
            raise HTTPException(400, "Invalid input combination")

        # ==================================================
        # 2) NDVI COLLECTIONS
        # ==================================================
        def landsat_ndvi(collection_id):
            return (
                ee.ImageCollection(collection_id)
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .map(lambda img: (
                    img.select("SR.*")
                    .multiply(ee.Number(img.get("REFLECTANCE_MULT_BAND_3")))
                    .add(ee.Number(img.get("REFLECTANCE_ADD_BAND_2")))
                    .normalizedDifference(["SR_B5", "SR_B4"])
                    .rename("ndvi")
                    .clip(geometry)
                    .copyProperties(img, ["system:time_start"])
                ))
            )

        landsat8 = landsat_ndvi("LANDSAT/LC08/C02/T1_L2")
        landsat9 = landsat_ndvi("LANDSAT/LC09/C02/T1_L2")

        sentinel2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .map(lambda img: (
                img.select("B.*")
                .multiply(0.0001)
                .normalizedDifference(["B8", "B4"])
                .rename("ndvi")
                .clip(geometry)
                .copyProperties(img, ["system:time_start"])
            ))
        )

        ndvi_collection = landsat8.merge(landsat9).merge(sentinel2)

        # ==================================================
        # 3) MONTHLY NDVI
        # ==================================================
        start = ee.Date(start_date)
        end = ee.Date(end_date)

        month_count = end.difference(start, "month").add(1)
        month_starts = ee.List.sequence(0, month_count.subtract(1)).map(
            lambda m: start.advance(m, "month")
        )

        def monthly_composite(date):
            date = ee.Date(date)
            col = ndvi_collection.filterDate(date, date.advance(1, "month"))

            img = ee.Image(
                ee.Algorithms.If(
                    col.size().gt(0),
                    col.median(),
                    ee.Image.constant(0).rename("ndvi").updateMask(ee.Image(0))
                )
            ).clip(geometry)

            return img.set("system:time_start", date.millis())

        monthly_ndvi = ee.ImageCollection(month_starts.map(monthly_composite))

        # ==================================================
        # 4) NDVI AREA (UNCHANGED)
        # ==================================================
        def area_calc(img):
            mask = img.gte(ndvi_threshold)
            area = (
                ee.Image.pixelArea()
                .divide(10000)
                .updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometry,
                    scale=30,
                    maxPixels=1e13
                )
                .get("area")
            )

            return ee.Feature(None, {
                "month": ee.Date(img.get("system:time_start")).format("YYYY-MM"),
                "area_ha": area
            })

        ndvi_area_fc = ee.FeatureCollection(monthly_ndvi.map(area_calc))

        # ==================================================
        # 5) FINAL TILE IMAGE (TREE CANOPY IN YELLOW)
        # ==================================================
        mean_ndvi = monthly_ndvi.mean()

        # Tree canopy (height > 2 m)
        canopy = (
            ee.ImageCollection(
                "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
            )
            .mosaic()
            .clip(geometry)
        )

        canopy_mask = canopy.gt(3).selfMask()

        sugarcane_mask = mean_ndvi.gte(ndvi_threshold).selfMask()

        moderate_ndvi_mask = (
            mean_ndvi.gte(0.30)
            .And(mean_ndvi.lte(0.34))
            .selfMask()
        )

        black_bg = ee.Image.constant(0).clip(geometry).visualize(
            palette=["000000"]
        )

        moderate_vis = moderate_ndvi_mask.visualize(
            palette=["0000FF"]
        )

        sugarcane_vis = sugarcane_mask.visualize(
            palette=["00FF00"]
        )

        canopy_vis = canopy_mask.visualize(
            palette=["FFFF00"]
        )

        boundary = (
            ee.Image()
            .byte()
            .paint(
                ee.FeatureCollection([ee.Feature(geometry)]),
                1,
                2
            )
            .visualize(palette=["FFFFFF"])
        )

        final_viz = (
            black_bg
            .blend(moderate_vis)
            .blend(sugarcane_vis)
            .blend(canopy_vis)
            .blend(boundary)
        )

        tile_url = final_viz.getMapId()["tile_fetcher"].url_format

        # ==================================================
        # 6) EXPORT
        # ==================================================
        export_name = f"{plot_name}_{start_date}_{end_date}"

        ee.batch.Export.image.toDrive(
            image=final_viz,
            description=export_name,
            folder="sugarcane_ndvi_exports",
            fileNamePrefix=export_name,
            region=geometry,
            scale=30,
            maxPixels=1e13,
            fileFormat="GEO_TIFF"
        ).start()

        # ==================================================
        # 7) RESPONSE
        # ==================================================
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": response_geometry,
                "properties": {
                    "plot_id": plot_name,
                    "tile_url": tile_url,
                    "ndvi_threshold": ndvi_threshold,
                    "layers": {
                        "tree_canopy": "height > 2 m (yellow)",
                        "sugarcane": "NDVI >= threshold (green)",
                        "moderate_ndvi": "0.30–0.34 (blue)"
                    },
                    "last_updated": datetime.now().isoformat(),
                }
            }],
            "ndvi_area_summary": ndvi_area_fc.getInfo()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"NDVI detection failed: {e}")

@app.get("/PalusVillages")
def list_villages():
    villages = sorted(gdf["village_name"].unique().tolist())
    return {
        "count": len(villages),
        "villages": villages
    }
 
# -----------------------------
# GET FARMS BY VILLAGE
# -----------------------------
@app.get("/farms/{village_name}")
def get_farms_by_village(village_name: str):
    village_data = gdf[gdf["village_name"] == village_name]
 
    if village_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for village: {village_name}"
        )
 
    return village_data.__geo_interface__


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    uvicorn.run("up44:app", host="192.168.42.56", port=8010, reload=True)

