# Includes Brix/Recovery analysis, vegetation indices, biomass, stress events, and irrigation detection
# Comprehensive Agriculture Analysis API - Merged Version
 
from fastapi import FastAPI, HTTPException, Query
import pandas as pd 
from cachetools import TTLCache 
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import ee
from dateutil.relativedelta import relativedelta
import statistics, httpx
from datetime import datetime, timedelta
import uvicorn
from pydantic import BaseModel
from shapely.geometry import shape, Point, Polygon
from geopy.distance import geodesic  
import matplotlib.colors as mcolors
import numpy as np
from shared_services import PlotSyncService
from Admin import calculate_area_hectares
 
 
# -------------------------------
# Earth Engine Initialization
# -------------------------------
try:
    ee.Authenticate()
    ee.Initialize(project="arctic-hybrid-462806-c9")
except Exception as e:
    raise RuntimeError(f"Earth Engine failed to initialize: {e}")
 
# Initialize plot sync service
plot_sync_service = PlotSyncService()

# -------------------------------AG
# Load ROI/Plot Dictionary from Django API
# -------------------------------
 
plot_dict = plot_sync_service.get_plots_dict()
print(f"ðŸš€ events.py startup: Loaded {len(plot_dict)} plots from Django")

def find_plot_by_name(plot_name: str) -> Optional[Dict]:
    """Find plot by name with flexible matching"""
    # Direct lookup
    if plot_name in plot_dict:
        print(f"Found plot '{plot_name}' with direct lookup")
        return plot_dict[plot_name]
    
    # Try different variations for leading zeros
    variations = []
    
    # If plot_name contains underscore, try different combinations
    if '_' in plot_name:
        parts = plot_name.split('_')
        if len(parts) == 2:
            gat_part, plot_part = parts
            
            # Try original
            variations.append(plot_name)
            
            # Try with leading zeros (only for exact matches)
            try:
                if gat_part.startswith('0'):
                    variations.append(f"{int(gat_part)}_{plot_part}")
                if plot_part.startswith('0'):
                    variations.append(f"{gat_part}_{int(plot_part)}")
                if gat_part.startswith('0') and plot_part.startswith('0'):
                    variations.append(f"{int(gat_part)}_{int(plot_part)}")
            except ValueError:
                pass
    
    # Try variations
    for variation in variations:
        if variation in plot_dict:
            print(f"Found plot '{plot_name}' using variation '{variation}'")
            return plot_dict[variation]
    
    print(f"Plot '{plot_name}' not found. Available plots: {list(plot_dict.keys())}")
    return None

def get_plot_feature_collection():
    features = []
    for plot_name, plot in plot_dict.items():
        feat = ee.Feature(plot["geometry"], {"plot_name": plot_name})
        features.append(feat)
    return ee.FeatureCollection(features)

plot_fc = get_plot_feature_collection()

def _round_safe(val, digits=4):
    try:
        return round(float(val), digits)
    except Exception:
        return None
    
soil_layers = {
    'organic_carbon_stock': ee.Image("projects/soilgrids-isric/ocs_mean"),
    'phh2o': ee.Image("projects/soilgrids-isric/phh2o_mean")
}

def calculate_all_stats_soil(scale=250):
    """Optimized: Single batched call for all plots"""
    parameter_bands = {
        'organic_carbon_stock': 'ocs_0-30cm_mean',
        'phh2o': 'phh2o_0-5cm_mean'
    }
    
    # ðŸ”¹ Build batch computation for ALL plots
    batch_computations = {}
    
    for plot_name, plot in plot_dict.items():
        geom = plot["geometry"]
        
        # Get both soil stats in one dictionary
        ocs_band = parameter_bands['organic_carbon_stock']
        ph_band = parameter_bands['phh2o']
        
        ocs_mean = soil_layers['organic_carbon_stock'].select(ocs_band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13
        ).get(ocs_band)
        
        ph_mean = soil_layers['phh2o'].select(ph_band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13
        ).get(ph_band)
        
        area_m2 = geom.area()
        
        # Store all computations for this plot
        batch_computations[plot_name] = ee.Dictionary({
            'ocs': ocs_mean,
            'ph': ph_mean,
            'area': area_m2
        })
    
    # ðŸ”¹ Single getInfo() call for ALL plots
    try:
        all_results = ee.Dictionary(batch_computations).getInfo()
    except Exception as e:
        print(f"Batch soil calculation failed: {e}")
        all_results = {}
    
    # ðŸ”¹ Process results
    stats_dict = {}
    for plot_name in plot_dict.keys():
        result = all_results.get(plot_name, {})
        ocs_val = result.get('ocs')
        ph_val = result.get('ph')
        area_m2 = result.get('area', 0)
        
        stats_dict[plot_name] = {
            "organic_carbon_stock": _round_safe(ocs_val / 2.47) if ocs_val is not None else None,
            "phh2o": _round_safe(ph_val / 10.0) if ph_val is not None else None,
            "area_acres": _round_safe((area_m2 / 10000.0) * 2.47)
        }
    
    return stats_dict

def detect_sugarcane_harvest_for_plot(

    geometry: ee.Geometry,

    plantation_date: str,

    end_date: str,

    window_days: int = 60,

    hpi_thr: float = 0.3,

    nhpi_thr: float = 0.6,

    ndvi_drop_thr: float = 0.15,

) -> Dict[str, Any]:
 
    import math
 
    def clean_value(v):

        if isinstance(v, float) and not math.isfinite(v):

            return None

        return v
 
    def clean_record(rec):

        return {k: clean_value(v) for k, v in rec.items()}
 
    # -------- Sentinel-2 collection (SR Harmonized) --------

    def mask_s2(img):

        scl = img.select("SCL")

        mask = (

            scl.neq(3)

            .And(scl.neq(8))

            .And(scl.neq(9))

            .And(scl.neq(10))

            .And(scl.neq(11))

        )

        return img.updateMask(mask)
 
    def add_indices(img):

        nir = img.select("B8").multiply(0.0001)

        red = img.select("B4").multiply(0.0001)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

        return img.addBands(nir.rename("NIR")).addBands(ndvi)
 
    s2 = (

        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

        .filterBounds(geometry)

        .filterDate(plantation_date, end_date)

        .map(mask_s2)

        .map(add_indices)

    )
 
    s2_count = s2.size().getInfo()

    if s2_count == 0:

        raise HTTPException(

            status_code=404,

            detail="No Sentinel-2 images available for this plot in the given date range.",

        )
 
    # -------- Reduce time series --------

    def to_feature(img):

        stats = img.reduceRegion(

            reducer=ee.Reducer.mean(),

            geometry=geometry,

            scale=10,

            maxPixels=1e9,

        )

        return ee.Feature(

            None,

            {

                "date": img.date().format("YYYY-MM-dd"),

                "NIR": stats.get("NIR"),

                "NDVI": stats.get("NDVI"),

            },

        )
 
    ts = s2.map(to_feature).getInfo()
 
    df = pd.DataFrame([

        {

            "date": f["properties"]["date"],

            "NDVI": f["properties"]["NDVI"],

            "NIR": f["properties"]["NIR"],

        }

        for f in ts["features"]

    ])
 
    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna().sort_values("date")
 
    # -------- HPI & NHPI --------

    df["HPI"] = df["NIR"] / df["NDVI"].replace(0, np.nan)
 
    df["NHPI"] = np.nan

    for i in range(len(df)):

        t = df.index[i]

        t_end = df.loc[t, "date"]

        t_start = t_end - pd.Timedelta(days=window_days)

        win = df[(df["date"] >= t_start) & (df["date"] <= t_end)]
 
        hmin = win["HPI"].min()

        hmax = win["HPI"].max()
 
        if pd.isna(hmin) or pd.isna(hmax) or hmax == hmin:

            df.loc[t, "NHPI"] = 0

        else:

            df.loc[t, "NHPI"] = (df.loc[t, "HPI"] - hmin) / (hmax - hmin)
 
    # -------- NDVI drop --------

    df["NDVI_prev"] = df["NDVI"].shift(1)

    df["NDVI_drop"] = df["NDVI_prev"] - df["NDVI"]
 
    # -------- Candidate harvest detection --------

    df["is_harvest"] = (df["HPI"] > hpi_thr) & (df["NDVI_drop"] > ndvi_drop_thr)
 
    candidate_dates = df.loc[df["is_harvest"], "date"].tolist()
 
    def validate_harvest_recursive(df_local, candidate_dates_local):

        if not candidate_dates_local:

            return None
 
        last_date_in_data = df_local["date"].max()
 
        for candidate in candidate_dates_local:

            if candidate == last_date_in_data:

                return candidate
 
            ndvi0 = df_local.loc[df_local["date"] == candidate, "NDVI"].values[0]

            next3 = df_local[df_local["date"] > candidate].head(3)
 
            if len(next3) == 0:

                return candidate
 
            if (next3["NDVI"] > ndvi0).any():

                continue
 
            return candidate
 
        return None
 
    harvest_date = validate_harvest_recursive(df, candidate_dates)
 
    # -------- Make JSON-safe --------

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
 
    timeseries_records = [

        clean_record({

            "date": row["date"],

            "NDVI": row["NDVI"],

            "NIR": row["NIR"],

            "HPI": row["HPI"],

            "NHPI": row["NHPI"],

            "NDVI_drop": row["NDVI_drop"],

            "is_harvest": bool(row["is_harvest"]),

        })

        for _, row in df.iterrows()

    ]
 
    return {

        "harvest_date": harvest_date.strftime("%Y-%m-%d") if harvest_date else None,

        "has_harvest": harvest_date is not None,

        "hpi_threshold": hpi_thr,

        "nhpi_threshold": nhpi_thr,

        "ndvi_drop_threshold": ndvi_drop_thr,

        "timeseries": timeseries_records,

        "image_count": int(s2_count),

    }
 
 

# -------------------------------
# Pydantic Models
# -------------------------------
 
# Brix/Recovery/Sugar Yield Models
class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any
 
class GeoJSONProperties(BaseModel):
    plot_name: str
    date: str
    brix_statistics: Dict[str, float]
    recovery_statistics: Dict[str, float]
    sugar_yield_statistics: Dict[str, float]  # Added sugar yield statistics
    area_hectares: float
    area_acres: float
    eligibility_status:str
 
class GeoJSONFeature(BaseModel):
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties
 
class GeoJSONResponse(BaseModel):
    features: List[GeoJSONFeature]

# Sugar Yield Models
class SugarYieldStats(BaseModel):
    mean: float
    min: float
    max: float
    std: float
    median: float
    expected_yield_percent: float  # Sugar Yield as % of Cane

class SugarYieldAnalysis(BaseModel):
    plot_name: str
    date: str
    brix_mean: float
    recovery_mean: float
    sugar_yield_percent: float
    area_hectares: float
    yield_grade: str  # Excellent, Good, Fair, Poor based on yield percentage

# Harvest Analysis Models
class GrowthStage(BaseModel):
    stage_name: str
    days_from_planting: int
    description: str
    management_actions: List[str]

class HarvestAnalysis(BaseModel):
    plot_name: str
    plantation_date: str
    current_date: str
    days_since_planting: int
    days_to_harvest: int
    estimated_harvest_date: str
    current_growth_stage: str
    next_growth_stage: str
    days_to_next_stage: int
    harvest_readiness_percentage: float
    recommended_actions: List[str]

class HarvestPlanning(BaseModel):
    plantation_date: str
    variety_type: str  # Early, Mid, Late maturing
    estimated_harvest_date: str
    total_growth_days: int
    growth_stages: List[GrowthStage]
    harvest_recommendations: List[str]

# Vegetation Index Models
class IndexTimeSeries(BaseModel):
    date: str
    MSAVI: Optional[float] = None
    NDRE: Optional[float] = None
    NDVI: Optional[float] = None
    NDMI: Optional[float] = None
    NDWI: Optional[float] = None
 
class RviTimeSeries(BaseModel):
    date: str
    RVI: Optional[float] = None
 
class BiomassStats(BaseModel):
    mean: float
    min: float
    max: float
    total: float
 
class StressEvent(BaseModel):
    from_date: str
    to_date: str
    stress: float
    index_type: str
 
class StressAnalysis(BaseModel):
    total_events: int
    threshold_used: float
    index_type: str
    events: List[StressEvent]
 
class IrrigationEvent(BaseModel):
    date: str
    delta_ndmi: float
    delta_ndwi: float
 
class IrrigationAnalysis(BaseModel):
    total_events: int
    threshold_ndmi: float
    threshold_ndwi: float
    min_days_between_events: int
    events: List[IrrigationEvent]
 
class PlotInfo(BaseModel):
    name: str
    geometry_type: str
    area_hectares: float
    available_layers: List[str]
 
# -------------------------------
# Brix, Recovery, and Sugar Yield Calculation Functions
# -------------------------------
 
def calculate_statistics(image, geometry, scale=10) -> Dict[str, Dict[str, float]]:
    """Pixel-based statistics for Brix, Recovery, and Sugar Yield"""
    try:
        # Clip and sample pixels
        clipped = image.clip(geometry)
        pixels = clipped.sample(
            region=geometry,
            scale=scale,
            geometries=False,   # no need for geometry here
            tileScale=4,
            numPixels=1e13
        ).getInfo()

        # Extract pixel values for each band
        brix_vals = []
        recovery_vals = []
        sugar_vals = []

        for f in pixels["features"]:
            props = f["properties"]
            if "Brix" in props:
                brix_vals.append(props["Brix"])
            if "Recovery" in props:
                recovery_vals.append(props["Recovery"])
            if "SugarYield" in props:
                sugar_vals.append(props["SugarYield"])

        def compute_stats(values):
            if not values:
                return {}
            return {
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "median": round(statistics.median(values), 2),
                "stdDev": round(statistics.pstdev(values), 2)
            }

        return {
            "brix": compute_stats(brix_vals),
            "recovery": compute_stats(recovery_vals),
            "sugar_yield": compute_stats(sugar_vals)
        }

    except Exception as e:
        print(f"Error in pixel-based statistics calculation: {e}")
        return {"brix": {}, "recovery": {}, "sugar_yield": {}}

 
def get_brix_recovery_sugar_yield_images(start_date: str, end_date: str):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 75))
        .select(["B3","B4", "B6", "B8", "B11"])
        .median()
    )
    green=s2.select("B3")
    red = s2.select("B4")
    band6 = s2.select("B6")
    band8 = s2.select("B8")
    band11 = s2.select("B11")
    
    gndvi=(band8.subtract(green)).divide(band8.add(green)).rename("GNDVI")

    brix = (
        red.multiply(0.0122621)
        .subtract(band6.multiply(0.00706689))
        .add(gndvi.multiply(21.5251))
        .add(34.784)
        .rename("Brix")
    ).divide(2.0)
   
    # Recovery calculation: Brix/2 = Recovery
    recovery = brix.multiply(0.44).rename('Recovery')
    
    # Sugar Yield calculation: Brix * Recovery
    sugar_yield = brix.multiply(recovery).rename('SugarYield')
   
    return brix.addBands(recovery).addBands(sugar_yield)

def get_sugar_yield_grade(yield_percent: float) -> str:
    """Determine sugar yield grade based on percentage"""
    if yield_percent >= 12.0:
        return "Excellent"
    elif yield_percent >= 10.0:
        return "Good"
    elif yield_percent >= 8.0:
        return "Fair"
    else:
        return "Poor"

def calculate_irrigation_events(plantation_date_str, frequency_days=3, total_days=21):
    """
    Calculate irrigation events from the plantation date.

    Parameters:
    - plantation_date_str: Plantation date in 'YYYY-MM-DD' format
    - frequency_days: Irrigation interval (default 3 days)
    - total_days: Total duration for which to plan irrigation (default 21 days)

    Returns:
    - List of irrigation event dates
    """
    plantation_date = datetime.strptime(plantation_date_str, "%Y-%m-%d")
    irrigation_events = []

    current_date = plantation_date
    while (current_date - plantation_date).days <= total_days:
        irrigation_events.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=frequency_days)

    return irrigation_events

def count_irrigation_events(plantation_date_str, interval_days=21):
    """
    Count irrigation events from plantation date to today, with a given interval.

    Parameters:
    - plantation_date_str: Plantation date in 'YYYY-MM-DD' format
    - interval_days: Interval between irrigation events (default 21 days)

    Returns:
    - Dictionary with count and list of event dates
    """
    plantation_date = datetime.strptime(plantation_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    
    irrigation_events = []
    current_event_date = plantation_date
    
    while current_event_date <= current_date:
        irrigation_events.append(current_event_date.strftime("%Y-%m-%d"))
        current_event_date += timedelta(days=interval_days)
    
    return {
        "total_events": len(irrigation_events),
        "events": irrigation_events,
        "plantation_date": plantation_date_str,
        "current_date": current_date.strftime("%Y-%m-%d"),
        "interval_days": interval_days
    }

# -------------------------------
# Harvest Calculation Functions
# -------------------------------

def get_growth_stages(variety_type: str = "Mid") -> List[Dict[str, Any]]:
    """
    Get sugarcane growth stages based on variety type.
    
    Parameters:
    - variety_type: "Early", "Mid", or "Late" maturing variety
    
    Returns:
    - List of growth stage dictionaries
    """
    # Growth stages for different variety types (days from planting)
    growth_stages_data = {
        "Mid": [
            {"stage_name": "Germination", "days": 0, "description": "Seed germination and initial growth", 
             "actions": ["Maintain soil moisture", "Monitor for pests"]},
            {"stage_name": "Tillering", "days": 31, "description": "Development of side shoots", 
             "actions": ["Apply nitrogen fertilizer", "Control weeds"]},
            {"stage_name": "Grand Growth", "days": 91, "description": "Rapid stem elongation", 
             "actions": ["Ensure adequate irrigation", "Monitor for diseases"]},
            {"stage_name": "Maturity", "days": 211, "description": "Sugar accumulation phase", 
             "actions": ["Reduce irrigation", "Prepare for harvest"]}
        ],
    }
    
    return growth_stages_data.get(variety_type, growth_stages_data["Mid"])

def calculate_harvest_timing(plantation_date_str: str, variety_type: str = "adsali") -> Dict[str, Any]:
    """
    Calculate harvest timing and growth stage information.
    
    Parameters:
    - plantation_date_str: Plantation date in 'YYYY-MM-DD' format
    - variety_type: "Early", "Mid", or "Late" maturing variety
    
    Returns:
    - Dictionary with harvest timing information
    """
    plantation_date = datetime.strptime(plantation_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    days_since_planting = (current_date - plantation_date).days
    
    growth_stages = get_growth_stages(variety_type)
    
    growth_days={"suru":365,"adsali":548,"ratoon":365,"pre-seasonal":425}
    total_growth_days=growth_days[variety_type]
    

    # Find current growth stage
    current_stage = growth_stages[0]
    next_stage = growth_stages[1] if len(growth_stages) > 1 else growth_stages[0]
    
    for i, stage in enumerate(growth_stages):
        if days_since_planting >= stage["days"]:
            current_stage = stage
            if i + 1 < len(growth_stages):
                next_stage = growth_stages[i + 1]
    
    days_to_harvest =  total_growth_days - days_since_planting
    estimated_harvest_date = plantation_date + timedelta(days=total_growth_days)
    days_to_next_stage = max(0, next_stage["days"] - days_since_planting)
    
    # Calculate harvest readiness percentage
    if days_since_planting >= total_growth_days:
        harvest_readiness = 100.0
    else:
        harvest_readiness = (days_since_planting / total_growth_days) * 100
    
    # Determine recommended actions
    recommended_actions = current_stage["actions"].copy()
    if days_to_harvest <= 30:
        recommended_actions.append("Final harvest preparation")
    elif days_to_harvest <= 60:
        recommended_actions.append("Begin harvest planning")
    
    return {
        "plantation_date": plantation_date_str,
        "current_date": current_date.strftime("%Y-%m-%d"),
        "days_since_planting": days_since_planting,
        "days_to_harvest": days_to_harvest,
        "estimated_harvest_date": estimated_harvest_date.strftime("%Y-%m-%d"),
        "current_growth_stage": current_stage["stage_name"],
        "next_growth_stage": next_stage["stage_name"],
        "days_to_next_stage": days_to_next_stage,
        "harvest_readiness_percentage": round(harvest_readiness, 1),
        "recommended_actions": recommended_actions,
        "variety_type": variety_type,
        "total_growth_days": total_growth_days
    }

def get_optimal_harvest_window(plantation_date_str: str, variety_type: str = "Mid") -> Dict[str, Any]:
    """
    Calculate optimal harvest window based on plantation date and variety.
    
    Parameters:
    - plantation_date_str: Plantation date in 'YYYY-MM-DD' format
    - variety_type: "Early", "Mid", or "Late" maturing variety
    
    Returns:
    - Dictionary with optimal harvest window information
    """
    plantation_date = datetime.strptime(plantation_date_str, "%Y-%m-%d")
    growth_stages = get_growth_stages(variety_type)
    total_growth_days = growth_stages[-1]["days"]
    
    # Optimal harvest window is Â±15 days around the target harvest date
    optimal_start = plantation_date + timedelta(days=total_growth_days - 15)
    optimal_end = plantation_date + timedelta(days=total_growth_days + 15)
    
    return {
        "plantation_date": plantation_date_str,
        "target_harvest_date": (plantation_date + timedelta(days=total_growth_days)).strftime("%Y-%m-%d"),
        "optimal_start_date": optimal_start.strftime("%Y-%m-%d"),
        "optimal_end_date": optimal_end.strftime("%Y-%m-%d"),
        "total_growth_days": total_growth_days,
        "variety_type": variety_type
    }

# -------------------------------
# Vegetation Index Functions
# -------------------------------
 
def get_indices_time_series(roi) -> List[Dict[str, Any]]:
    """Get vegetation indices time series for ROI"""
    today = datetime.now()
    start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
   
    collection = (ee.ImageCollection('COPERNICUS/S2')
                  .filterDate(start_date, end_date)
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
   
    def calculate_indices(image):
        B4 = image.select('B4')
        B5 = image.select('B5')
        B8 = image.select('B8')
        B11 = image.select('B11')
        B12 = image.select('B12')
       
        msavi = image.expression('2 * B8 + 1 - sqrt((2 * B8 + 1) ** 2 - 8 * (B8 - B4))',
                                {'B8': B8, 'B4': B4}).rename('MSAVI')
        ndre = image.expression('(B8 - B5) / (B8 + B5)', {'B8': B8, 'B5': B5}).rename('NDRE')
        ndvi = image.expression('(B8 - B4) / (B8 + B4)', {'B8': B8, 'B4': B4}).rename('NDVI')
        ndmi = image.expression('(B8 - B11) / (B8 + B11)', {'B8': B8, 'B11': B11}).rename('NDMI')
        ndwi = image.expression('(B8 - B12) / (B8 + B12)', {'B8': B8, 'B12': B12}).rename('NDWI')
       
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        indices = msavi.addBands(ndre).addBands(ndvi).addBands(ndmi).addBands(ndwi)
        indices_values = indices.reduceRegion(ee.Reducer.mean(), roi, 10, maxPixels=1e8)
       
        return ee.Feature(None, indices_values.set('date', date))
   
    collection_with_indices = collection.map(calculate_indices)
    features = collection_with_indices.getInfo()['features']
   
    return [{
        "date": f['properties'].get('date'),
        "MSAVI": f['properties'].get('MSAVI'),
        "NDRE": f['properties'].get('NDRE'),
        "NDVI": f['properties'].get('NDVI'),
        "NDMI": f['properties'].get('NDMI'),
        "NDWI": f['properties'].get('NDWI')
    } for f in features]
 
def get_rvi_time_series(roi) -> List[Dict[str, Any]]:
    """Get RVI time series using Sentinel-1"""
    today = datetime.now()
    start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
   
    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterDate(start_date, end_date)
                  .filterBounds(roi)
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                  .select(['VV', 'VH']))
   
    def calculate_rvi(image):
        vv = image.select('VV')
        vh = image.select('VH')
        rvi = image.expression('(4 * VH) / (VV + VH)', {'VV': vv, 'VH': vh}).rename('RVI')
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        rvi_value = rvi.reduceRegion(ee.Reducer.mean(), roi, 10, maxPixels=1e13)
        return ee.Feature(None, rvi_value.set('date', date))
   
    collection_with_rvi = collection.map(calculate_rvi)
    features = collection_with_rvi.getInfo()['features']
   
    return [{"date": f['properties'].get('date'), "RVI": f['properties'].get('RVI')} for f in features]
 
def get_biomass_summary(roi) -> BiomassStats:
    """Calculate biomass statistics using Sentinel-1 RVI"""
    # Use current date and go back 30 days for recent data
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
   
    sentinel1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterBounds(roi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .select(['VV', 'VH']))
   
    def calculate_rvi(image):
        vh = image.select('VH')
        vv = image.select('VV')
        diff = vh.subtract(vv).abs().max(ee.Image.constant(0.01))
        rvi = vh.add(vv).divide(diff).rename('RVI')
        return image.addBands(rvi)
   
    rvi_collection = sentinel1.map(calculate_rvi)
    median_image = rvi_collection.median().clip(roi)
    biomass = median_image.select('RVI').multiply(50).abs().rename('Biomass')
   
    stats = biomass.reduceRegion(
        reducer=ee.Reducer.minMax().combine('mean', '', True),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
   
    pixel_area = ee.Image.pixelArea().divide(10000)
    total_biomass = biomass.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).get('Biomass')
   
    return BiomassStats(
        mean=abs(stats.get('Biomass_mean').getInfo()),
        min=abs(stats.get('Biomass_min').getInfo()),
        max=abs(stats.get('Biomass_max').getInfo()),
        total=abs(total_biomass.getInfo())
    )
 

def get_biomass_summary_batch(plot_dict) -> Dict[str, BiomassStats]:
    """Calculate biomass for all plots in a single batch operation"""
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # ðŸ”¹ Single Sentinel-1 collection for all plots (reused)
    sentinel1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
                 .select(['VV', 'VH']))
    
    def calculate_rvi(image):
        vh = image.select('VH')
        vv = image.select('VV')
        diff = vh.subtract(vv).abs().max(ee.Image.constant(0.01))
        rvi = vh.add(vv).divide(diff).rename('RVI')
        return image.addBands(rvi)
    
    rvi_collection = sentinel1.map(calculate_rvi)
    
    # ðŸ”¹ Build computations for all plots
    all_computations = {}
    
    for plot_name, plot in plot_dict.items():
        roi = plot["geometry"]
        
        # Filter to this ROI and calculate biomass
        plot_sentinel = rvi_collection.filterBounds(roi)
        median_image = plot_sentinel.median().clip(roi)
        biomass = median_image.select('RVI').multiply(50).abs().rename('Biomass')
        
        # Stats
        stats = biomass.reduceRegion(
            reducer=ee.Reducer.minMax().combine('mean', '', True),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        )
        
        # Total
        pixel_area = ee.Image.pixelArea().divide(10000)
        total_biomass = biomass.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).get('Biomass')
        
        # Store computation
        all_computations[plot_name] = ee.Dictionary({
            'mean': stats.get('Biomass_mean'),
            'min': stats.get('Biomass_min'),
            'max': stats.get('Biomass_max'),
            'total': total_biomass
        })
    
    # ðŸ”¹ Single getInfo() call for ALL plots
    try:
        batch_results = ee.Dictionary(all_computations).getInfo()
    except Exception as e:
        print(f"Batch biomass calculation failed: {e}")
        batch_results = {}
    
    # ðŸ”¹ Convert to BiomassStats objects
    biomass_stats = {}
    for plot_name in plot_dict.keys():
        result = batch_results.get(plot_name, {})
        biomass_stats[plot_name] = BiomassStats(
            mean=abs(result.get('mean', 0)) if result.get('mean') is not None else 0,
            min=abs(result.get('min', 0)) if result.get('min') is not None else 0,
            max=abs(result.get('max', 0)) if result.get('max') is not None else 0,
            total=abs(result.get('total', 0)) if result.get('total') is not None else 0
        )
    
    return biomass_stats

# -------------------------------
# Analysis Functions
# -------------------------------
 
def detect_stress_events(data: List[Dict[str, Any]], index_type: str, threshold: float = 0.15) -> StressAnalysis:
    """Detect stress events based on vegetation index drops"""
    valid_data = [d for d in data if d.get(index_type) is not None]
    valid_data.sort(key=lambda x: x["date"])
   
    stress_events = []
    for i in range(1, len(valid_data)):
        prev_value = valid_data[i-1][index_type]
        curr_value = valid_data[i][index_type]
        stress_value = prev_value - curr_value
       
        if stress_value >= threshold:
            stress_events.append(StressEvent(
                from_date=valid_data[i-1]["date"],
                to_date=valid_data[i]["date"],
                stress=round(stress_value, 4),
                index_type=index_type
            ))
   
    return StressAnalysis(
        total_events=len(stress_events),
        threshold_used=threshold,
        index_type=index_type,
        events=stress_events
    )
 
def detect_irrigation_events(
    data: List[Dict[str, Any]],
    threshold_ndmi: float = 0.05,
    threshold_ndwi: float = 0.05,
    min_days_between_events: int = 10
) -> IrrigationAnalysis:
    """Detect irrigation events based on NDMI and NDWI increases"""
    valid_data = [d for d in data if d.get("NDMI") is not None and d.get("NDWI") is not None]
    valid_data.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
   
    irrigation_events = []
    last_event_date = None
   
    for i in range(1, len(valid_data)):
        current = valid_data[i]
        previous = valid_data[i - 1]
       
        delta_ndmi = current["NDMI"] - previous["NDMI"]
        delta_ndwi = current["NDWI"] - previous["NDWI"]
        current_date = datetime.strptime(current["date"], "%Y-%m-%d")
       
        if delta_ndmi > threshold_ndmi and delta_ndwi > threshold_ndwi:
            if not last_event_date or (current_date - last_event_date).days >= min_days_between_events:
                irrigation_events.append(IrrigationEvent(
                    date=current["date"],
                    delta_ndmi=round(delta_ndmi, 4),
                    delta_ndwi=round(delta_ndwi, 4)
                ))
                last_event_date = current_date
   
    return IrrigationAnalysis(
        total_events=len(irrigation_events),
        threshold_ndmi=threshold_ndmi,
        threshold_ndwi=threshold_ndwi,
        min_days_between_events=min_days_between_events,
        events=irrigation_events
    )
 
# -------------------------------
# FastAPI Setup
# -------------------------------
 
app = FastAPI(
    title="Comprehensive Agriculture Analysis API",
    description="Complete API for Brix/Recovery/Sugar Yield analysis, vegetation indices, biomass, stress events, and irrigation detection using Earth Engine",
    version="4.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# -------------------------------
# API Endpoints
# -------------------------------
 
@app.get("/")
async def root():
    """API root endpoint with comprehensive information"""
    return {
        "message": "Comprehensive Agriculture Analysis API",
        "version": "4.0.0",
        "data_source": "Django /plots/ API",
        "status": "dynamic",
        "total_plots": len(plot_dict),
        "capabilities": {
            "brix_recovery_sugar_yield": "Brix, Recovery, and Sugar Yield analysis from Sentinel-2",
            "vegetation_indices": "NDVI, NDRE, MSAVI, NDMI, NDWI time series",
            "biomass_analysis": "Biomass estimation using Sentinel-1 RVI",
            "stress_detection": "Automated crop stress event detection",
            "irrigation_detection": "Irrigation event detection using water indices",
            "irrigation_planning": "Irrigation event planning and counting",
            "harvest_analysis": "Days to harvest calculation and growth stage tracking"
        },
        "endpoints": {
            "plots": "/plots - List all plots",
            "plot_info": "/plots/{plot_name}/info - Get plot information",
            "brix_analysis": "/analyze - Analyze Brix, Recovery, and Sugar Yield",
            "sugar_yield": "/plots/{plot_name}/sugar-yield - Get Sugar Yield analysis",
            "vegetation_indices": "/plots/{plot_name}/indices - Get vegetation indices",
            "biomass": "/plots/{plot_name}/biomass - Get biomass statistics",
            "stress_analysis": "/plots/{plot_name}/stress - Analyze stress events",
            "irrigation_analysis": "/plots/{plot_name}/irrigation - Detect irrigation events",
            "irrigation_planning": "/irrigation/plan - Plan irrigation events",
            "irrigation_count": "/irrigation/count - Count irrigation events",
            "harvest_timing": "/harvest/timing - Calculate days to harvest",
            "harvest_window": "/harvest/optimal-window - Get optimal harvest window",
            "growth_stages": "/harvest/growth-stages - Get growth stage information",
            "harvest_planning": "/harvest/planning - Get comprehensive harvest planning",
            "plot_harvest": "/plots/{plot_name}/harvest-analysis - Get plot-specific harvest analysis"
        }
    }
 
@app.get("/plots", response_model=List[str])
async def get_plots():
    """Get list of all available plots"""
    return list(plot_dict.keys())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Comprehensive Agriculture Analysis API",
        "data_source": "Django /plots/ API",
        "plot_count": len(plot_dict),
        "last_sync": plot_sync_service.last_sync.isoformat() if plot_sync_service.last_sync else None
    }

@app.get("/plots/debug")
async def get_plots_debug():
    """Get detailed plot information for debugging"""
    debug_info = []
    for plot_name, plot_data in plot_dict.items():
        debug_info.append({
            "plot_name": plot_name,
            "gat_number": plot_data.get('properties', {}).get('gat_number'),
            "plot_number": plot_data.get('properties', {}).get('plot_number'),
            "plantation_date": plot_data.get('properties',{}).get('plantation_date'),
            "plantation_type": plot_data.get('properties',{}).get('plantation_type'),
            "django_id": plot_data.get('properties', {}).get('django_id'),
            "has_geometry": plot_data.get('geometry') is not None,
            "geom_type": plot_data.get('geom_type')
        })
    return {
        "total_plots": len(plot_dict),
        "plots": debug_info
    }
 
# -------------------------------
# Brix/Recovery/Sugar Yield Endpoints
# -------------------------------



@app.post("/analyze", response_model=GeoJSONResponse)
async def analyze_brix_recovery_sugar_yield(
    plot_name: str, 
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD"),
    end_date: str = Query(datetime.now().strftime('%Y-%m-%d'), description="Date for Stats")
):
    """Analyze Brix, Recovery, and Sugar Yield for a specific plot"""
    # Parse dates
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    plantation_dt = datetime.strptime(plantation_date, "%Y-%m-%d")

    # Calculate crop age in months
    crop_age_months = relativedelta(end_dt, plantation_dt).months + (relativedelta(end_dt, plantation_dt).years * 12)

    # Auto-calc start_date = end_date - 12 days
    start_dt = end_dt - timedelta(days=18)
    start_date = start_dt.strftime("%Y-%m-%d")

    plot = find_plot_by_name(plot_name)
    if not plot:
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
    aoi = plot["geometry"]

    # Check eligibility
    if crop_age_months < 6:
        # Not eligible: return zero stats
        brix_stats = {"mean": 0, "min": 0, "max": 0, "stdDev": 0}
        recovery_stats = {"mean": 0, "min": 0, "max": 0, "stdDev": 0}
        sugar_yield_stats = {"mean": 0, "min": 0, "max": 0, "stdDev": 0}
        eligibility_status = "Not eligible"
        area = aoi.area().divide(10000).getInfo()
    else:
        # Eligible: calculate normally
        combined_image = get_brix_recovery_sugar_yield_images(start_date, end_date)
        stats = calculate_statistics(combined_image, aoi)
        area = aoi.area().divide(10000).getInfo()

        brix_stats = stats["brix"]
        recovery_stats = stats["recovery"]

        sugar_yield_stats = {
            key: round(value * area, 2) if isinstance(value, (int, float)) else value
            for key, value in stats["sugar_yield"].items()
        }
        eligibility_status = "Eligible"

    feature = GeoJSONFeature(
        geometry=GeoJSONGeometry(
            type=plot["geom_type"],
            coordinates=plot["original_coords"]
        ),
        properties=GeoJSONProperties(
            plot_name=plot_name,
            plantation_date=plantation_date,
            crop_age_months=crop_age_months,
            eligibility_status=eligibility_status,
            area_hectares=round(area, 2),
            area_acres=round(area * 2.471, 2),
            date=end_date,
            brix_statistics=brix_stats,
            recovery_statistics=recovery_stats,
            sugar_yield_statistics=sugar_yield_stats
        )
    )

    return GeoJSONResponse(features=[feature])



class SoilStats(BaseModel):
    organic_carbon_stock: Optional[float]
    phh2o: Optional[float]
    area_acres: float  

class BrixSugarStats(BaseModel):
    brix: Dict[str, float]
    recovery: Dict[str, float]
    sugar_yield: Dict[str, float]


class Geometry(BaseModel):
    type: str
    coordinates: Any

class PlotStatsResponse(BaseModel):
    geometry: Geometry
    soil: SoilStats
    brix_sugar: BrixSugarStats
    biomass:BiomassStats
    Sugarcane_Status:str
    area_acres:float
    distance_km: Optional[float] = None
    days_to_harvest: Optional[int] = None
    current_growth_stage: Optional[str] = None
    plantation_date: Optional[str] = None
    plantation_type: Optional[str] = None
    eligibility_status:Optional[str]=None

# ------------------------------
# API Endpoints
# ------------------------------


# Cache: max 100 entries, TTL = 14400 seconds (4 hours)
cache = TTLCache(maxsize=100, ttl=14400)

# Helper function to fetch weather data
async def fetch_weather_data(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "windspeed_10m_max,relative_humidity_2m_max",
        "timezone": "UTC"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch weather data")
        return response.json()


@app.get("/monthwise-weather-summary")
async def get_monthwise_weather_summary(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date (take atleaast 3-4 days before today) in YYYY-MM-DD format")
):
    cache_key = f"month_{lat}_{lon}_{start_date}_{end_date}"
    if cache_key in cache:
        return {"source": "cache", "data": cache[cache_key]}

    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("Start date must be before end date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await fetch_weather_data(lat, lon, start, end)

    try:
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.to_period("M")

        monthly = df.groupby("month").agg({
            "temperature_2m_max": "mean",
            "temperature_2m_min": "mean",
            "precipitation_sum": "mean",
            "windspeed_10m_max": "mean",
            "relative_humidity_2m_max": "mean"
        }).reset_index()

        monthly = monthly.round(2)
        monthly["month"] = monthly["month"].astype(str)

        result = monthly.to_dict(orient="records")
        cache[cache_key] = result
        return {"source": "api", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get("/weekly-weather-summary")
async def get_weekly_weather_summary(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date (take atleaast 3-4 days before today) in YYYY-MM-DD format")
):
    cache_key = f"week_{lat}_{lon}_{start_date}_{end_date}"
    if cache_key in cache:
        return {"source": "cache", "data": cache[cache_key]}

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("Start date must be before end date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await fetch_weather_data(lat, lon, start, end)

    try:
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df["week"] = df["time"].dt.to_period("W")

        weekly = df.groupby("week").agg({
            "temperature_2m_max": "mean",
            "temperature_2m_min": "mean",
            "precipitation_sum": "mean",
            "windspeed_10m_max": "mean",
            "relative_humidity_2m_max": "mean"
        }).reset_index()

        weekly = weekly.round(2)
        weekly["week"] = weekly["week"].astype(str)

        result = weekly.to_dict(orient="records")
        cache[cache_key] = result
        return {"source": "api", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get("/daily-weather-summary")
async def get_daily_weather_summary(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date (take atleaast 3-4 days before today) in YYYY-MM-DD format")
):
    cache_key = f"day_{lat}_{lon}_{start_date}_{end_date}"
    if cache_key in cache:
        return {"source": "cache", "data": cache[cache_key]}

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            raise ValueError("Start date must be before end date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await fetch_weather_data(lat, lon, start, end)

    try:
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.round(2)
        df["time"] = df["time"].astype(str)

        result = df.to_dict(orient="records")
        cache[cache_key] = result
        return {"source": "api", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

import statistics
def calculate_all_stats_soil(scale=250):
    """Optimized: Single batched call for all plots"""
    parameter_bands = {
        'organic_carbon_stock': 'ocs_0-30cm_mean',
        'phh2o': 'phh2o_0-5cm_mean'
    }
    
    # ðŸ”¹ Build batch computation for ALL plots
    batch_computations = {}
    
    for plot_name, plot in plot_dict.items():
        geom = plot["geometry"]
        
        # Get both soil stats in one dictionary
        ocs_band = parameter_bands['organic_carbon_stock']
        ph_band = parameter_bands['phh2o']
        
        ocs_mean = soil_layers['organic_carbon_stock'].select(ocs_band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13
        ).get(ocs_band)
        
        ph_mean = soil_layers['phh2o'].select(ph_band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e13
        ).get(ph_band)
        
        area_m2 = geom.area()
        
        # Store all computations for this plot
        batch_computations[plot_name] = ee.Dictionary({
            'ocs': ocs_mean,
            'ph': ph_mean,
            'area': area_m2
        })
    
    # ðŸ”¹ Single getInfo() call for ALL plots
    try:
        all_results = ee.Dictionary(batch_computations).getInfo()
    except Exception as e:
        print(f"Batch soil calculation failed: {e}")
        all_results = {}
    
    # ðŸ”¹ Process results
    stats_dict = {}
    for plot_name in plot_dict.keys():
        result = all_results.get(plot_name, {})
        ocs_val = result.get('ocs')
        ph_val = result.get('ph')
        area_m2 = result.get('area', 0)
        
        stats_dict[plot_name] = {
            "organic_carbon_stock": _round_safe(ocs_val / 2.47) if ocs_val is not None else None,
            "phh2o": _round_safe(ph_val / 10.0) if ph_val is not None else None,
            "area_acres": _round_safe((area_m2 / 10000.0) * 2.47)
        }
    
    return stats_dict


def calculate_brix_sugar_stats(start_date, end_date,area_acres_map):
    """Compute Brix, Recovery, SugarYield stats per plot with all summary metrics"""
   
    image = get_brix_recovery_sugar_yield_images(start_date, end_date)
   
    plots_fc = ee.FeatureCollection([
        ee.Feature(plot["geometry"], {'plot_name': name})
        for name, plot in plot_dict.items()
    ])
   
    # Combine reducers to get mean, min, max, median, stdDev all at once
    reducer = ee.Reducer.mean() \
                .combine(ee.Reducer.min(), "", True) \
                .combine(ee.Reducer.max(), "", True) \
                .combine(ee.Reducer.median(), "", True) \
                .combine(ee.Reducer.stdDev(), "", True)
   
    reduced = image.reduceRegions(
        collection=plots_fc,
        reducer=reducer,
        scale=10
    ).getInfo()
   
    def safe_round(props, key):
        val = props.get(key)
        return round(val, 2) if val is not None else 0
   
    stats = {}
    for feature in reduced.get('features', []):
        props = feature['properties']
        plot_name = props['plot_name']
        geometry = ee.Geometry(feature['geometry'])
        area_acres = area_acres_map.get(plot_name, 0) or 1e-6
 
        stats[plot_name] = {
            "brix": {
                "mean": safe_round(props, "Brix_mean"),
                "min": safe_round(props, "Brix_min"),
                "max": safe_round(props, "Brix_max"),
                "median": safe_round(props, "Brix_median"),
                "stdDev": safe_round(props, "Brix_stdDev")
            },
            "recovery": {
                "mean": safe_round(props, "Recovery_mean"),
                "min": safe_round(props, "Recovery_min"),
                "max": safe_round(props, "Recovery_max"),
                "median": safe_round(props, "Recovery_median"),
                "stdDev": safe_round(props, "Recovery_stdDev")
            },
            "sugar_yield": {
                "mean": safe_round(props, "SugarYield_mean")/area_acres,
                "min": safe_round(props, "SugarYield_min")/area_acres,
                "max": safe_round(props, "SugarYield_max")/area_acres,
                "median": safe_round(props, "SugarYield_median")/area_acres,
                "stdDev": safe_round(props, "SugarYield_stdDev")/area_acres,
               
            }
        }
 
    return stats
 
 
def calculate_brix_sugar_stats1(start_date, end_date):
    """Compute Brix, Recovery, SugarYield stats per plot with all summary metrics"""
   
    image = get_brix_recovery_sugar_yield_images(start_date, end_date)
   
    plots_fc = ee.FeatureCollection([
        ee.Feature(plot["geometry"], {'plot_name': name})
        for name, plot in plot_dict.items()
    ])
   
    # Combine reducers to get mean, min, max, median, stdDev all at once
    reducer = ee.Reducer.mean() \
                .combine(ee.Reducer.min(), "", True) \
                .combine(ee.Reducer.max(), "", True) \
                .combine(ee.Reducer.median(), "", True) \
                .combine(ee.Reducer.stdDev(), "", True)
   
    reduced = image.reduceRegions(
        collection=plots_fc,
        reducer=reducer,
        scale=10
    ).getInfo()
   
    def safe_round(props, key):
        val = props.get(key)
        return round(val, 2) if val is not None else 0
   
    stats = {}
    for feature in reduced.get('features', []):
        props = feature['properties']
        plot_name = props['plot_name']
        geometry = ee.Geometry(feature['geometry'])
       
       
 
        stats[plot_name] = {
            "brix": {
                "mean": safe_round(props, "Brix_mean"),
                "min": safe_round(props, "Brix_min"),
                "max": safe_round(props, "Brix_max"),
                "median": safe_round(props, "Brix_median"),
                "stdDev": safe_round(props, "Brix_stdDev")
            },
            "recovery": {
                "mean": safe_round(props, "Recovery_mean"),
                "min": safe_round(props, "Recovery_min"),
                "max": safe_round(props, "Recovery_max"),
                "median": safe_round(props, "Recovery_median"),
                "stdDev": safe_round(props, "Recovery_stdDev")
            },
            "sugar_yield": {
                "mean": safe_round(props, "SugarYield_mean"),
                "min": safe_round(props, "SugarYield_min"),
                "max": safe_round(props, "SugarYield_max"),
                "median": safe_round(props, "SugarYield_median"),
                "stdDev": safe_round(props, "SugarYield_stdDev"),
               
            }
        }
 
    return stats
    
    
@app.get("/plots/agroStats", response_model=Dict[str, PlotStatsResponse])
def get_all_stats(
    end_date: str = Query(datetime.now().strftime('%Y-%m-%d')),
    factory_lat: Optional[float] = Query(None, description="Factory latitude for distance calculation"),
    factory_lon: Optional[float] = Query(None, description="Factory longitude for distance calculation")
):
    # Auto start_date = end_date - 12 days
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=18)
    start_date = start_dt.strftime("%Y-%m-%d")
 
    # ?? Batch area calculation ONCE
    area_computations = {}
    for plot_name, plot in plot_dict.items():
        roi = plot["geometry"]
        area_computations[plot_name] = roi.area().divide(10000)  # hectares
 
    try:
        batch_areas = ee.Dictionary(area_computations).getInfo()
    except Exception as e:
        print(f"Batch area calculation failed: {e}")
        batch_areas = {}
 
    # ?? Convert to acres once
    area_acres_map = {
        k: round(v * 2.471, 2) for k, v in batch_areas.items()
    }
 
 
    soil_stats = calculate_all_stats_soil()
    brix_stats = calculate_brix_sugar_stats(start_date, end_date,area_acres_map)
   
    # ðŸ”¹ Batch biomass calculations - ONE API CALL FOR ALL PLOTS
    biomass_results = get_biomass_summary_batch(plot_dict)
   
    # ðŸ”¹ Setup factory point for distance calculations (local, no API calls)
    factory_point = None
    factory_coords = None
    if factory_lat is not None and factory_lon is not None:
        factory_point = Point(factory_lon, factory_lat)
        factory_coords = (factory_lat, factory_lon)
   
    # ðŸ”¹ Build response
    combined_stats = {}
    for plot_name, plot in plot_dict.items():
        geom_type = plot["geom_type"]
        coords = plot["original_coords"]
        properties = plot.get("properties", {})
 
        soil = SoilStats(**soil_stats.get(plot_name, {}))
        brix_sugar = BrixSugarStats(**brix_stats.get(plot_name, {}))
        biomass = biomass_results.get(plot_name, BiomassStats(mean=0, min=0, max=0, total=0))
       
        # Get batched area
        area_acres = area_acres_map.get(plot_name, 0)
       
        # ðŸ”¹ Calculate distance (local computation, very fast)
        distance_km = None
        if factory_point and factory_coords:
            try:
                if geom_type == "Polygon":
                    plot_geom = Polygon(coords[0])
                    boundary = plot_geom.exterior
                    closest_point = boundary.interpolate(boundary.project(factory_point))
                else:
                    plot_geom = shape({"type": geom_type, "coordinates": coords})
                    closest_point = plot_geom.centroid
               
                closest_coords = (closest_point.y, closest_point.x)
                distance_km = round(geodesic(factory_coords, closest_coords).km, 2)
            except Exception as e:
                print(f"Distance calc failed for {plot_name}: {e}")
                distance_km = None
       
        # ðŸ”¹ Calculate harvest info from plot properties (local calculation, very fast)
        days_to_harvest = None
        current_growth_stage = None
        plantation_date = properties.get('plantation_date')
        variety_type_raw = properties.get('plantation_type')
 
        variety_type = variety_type_raw.lower().strip() if variety_type_raw else None
       
        crop_age_months = 0
        eligibility_status = "Not eligible"
       
        if plantation_date:
            try:
                plantation_dt = datetime.strptime(plantation_date, "%Y-%m-%d")
                crop_age_months = relativedelta(end_dt, plantation_dt).months + (relativedelta(end_dt, plantation_dt).years * 12)
               
                if crop_age_months >= 6:
                    eligibility_status = "Eligible"
            except Exception as e:
                print(f"Error calculating crop age for {plot_name}: {e}")
 
        if eligibility_status == "Eligible":
            brix_sugar = BrixSugarStats(**brix_stats.get(plot_name, {}))
        else:
            # Not eligible: return zero stats
            brix_sugar = BrixSugarStats(
                brix={"mean": 0, "min": 0, "max": 0, "stdDev": 0},
                recovery={"mean": 0, "min": 0, "max": 0, "stdDev": 0},
                sugar_yield={"mean": 0, "min": 0, "max": 0, "stdDev": 0}
            )
                 
        if plantation_date:
            try:
                # Ensure variety_type is valid
                if variety_type not in ["adsali", "suru", "ratoon", "pre-seasonal"]:
                    variety_type = "adsali"
               
                harvest_info = calculate_harvest_timing(plantation_date, variety_type)
                days_to_harvest = harvest_info["days_to_harvest"]
                current_growth_stage = harvest_info["current_growth_stage"]
            except Exception as e:
                print(f"Harvest calc failed for {plot_name}: {e}")
 
        combined_stats[plot_name] = PlotStatsResponse(
            geometry=Geometry(type=geom_type, coordinates=coords),
            soil=soil,
            brix_sugar=brix_sugar,
            biomass=biomass,
            Sugarcane_Status="Growing",
            area_acres=area_acres,
            distance_km=distance_km,
            plantation_date=plantation_date,
            plantation_type=variety_type,
            days_to_harvest=days_to_harvest,
            current_growth_stage=current_growth_stage,
            eligibility_status=eligibility_status
        )
 
    return combined_stats
 

@app.get("/plots/{plot_name}/indices", response_model=List[IndexTimeSeries])
async def get_vegetation_indices(plot_name: str):
    """Get vegetation indices time series for a specific plot"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    return get_indices_time_series(plot_data['geometry'])
 
@app.get("/plots/{plot_name}/rvi", response_model=List[RviTimeSeries])
async def get_rvi_time_series_endpoint(plot_name: str):
    """Get RVI time series for a specific plot"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    return get_rvi_time_series(plot_data['geometry'])
 
@app.get("/plots/{plot_name}/biomass", response_model=BiomassStats)
async def get_biomass_stats(plot_name: str):
    """Get biomass statistics for a specific plot"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    return get_biomass_summary(plot_data['geometry'])
 
# -------------------------------
# Analysis Endpoints
# -------------------------------
 
@app.get("/plots/{plot_name}/stress", response_model=StressAnalysis)
async def analyze_stress_events(
    plot_name: str,
    index_type: str = Query(default="NDRE", description="Vegetation index type (NDRE, NDVI, MSAVI, NDMI, NDWI)"),
    threshold: float = Query(default=0.15, description="Stress detection threshold", ge=0.01, le=1.0)
):
    """Analyze stress events for a specific plot based on vegetation index drops"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    valid_indices = ["NDRE", "NDVI", "MSAVI", "NDMI", "NDWI"]
    if index_type not in valid_indices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index_type. Must be one of: {', '.join(valid_indices)}"
        )
   
    indices_data = get_indices_time_series(plot_data['geometry'])
    return detect_stress_events(indices_data, index_type, threshold)
 
@app.get("/plots/{plot_name}/stress/all", response_model=Dict[str, StressAnalysis])
async def analyze_all_stress_events(
    plot_name: str,
    threshold: float = Query(default=0.15, description="Stress detection threshold", ge=0.01, le=1.0)
):
    """Analyze stress events for all vegetation indices for a specific plot"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    indices_data = get_indices_time_series(plot_data['geometry'])
    valid_indices = ["NDRE", "NDVI", "MSAVI", "NDMI", "NDWI"]
   
    results = {}
    for index_type in valid_indices:
        results[index_type] = detect_stress_events(indices_data, index_type, threshold)
   
    return results
 
@app.get("/plots/{plot_name}/irrigation", response_model=IrrigationAnalysis)
async def analyze_irrigation_events(
    plot_name: str,
    threshold_ndmi: float = Query(default=0.05, description="NDMI increase threshold", ge=0.01, le=1.0),
    threshold_ndwi: float = Query(default=0.05, description="NDWI increase threshold", ge=0.01, le=1.0),
    min_days_between_events: int = Query(default=10, description="Minimum days between irrigation events", ge=1, le=365)
):
    """Detect irrigation events for a specific plot based on NDMI and NDWI increases"""
    plot_data = find_plot_by_name(plot_name)
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
   
    indices_data = get_indices_time_series(plot_data['geometry'])
    return detect_irrigation_events(
        indices_data,
        threshold_ndmi,
        threshold_ndwi,
        min_days_between_events
    )

# -------------------------------
# Irrigation Planning Endpoints
# -------------------------------

@app.get("/irrigation/plan")
async def plan_irrigation_events(
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    frequency_days: int = Query(default=3, description="Irrigation interval in days", ge=1, le=30),
    total_days: int = Query(default=21, description="Total duration for irrigation planning", ge=1, le=365)
):
    """Plan irrigation events from plantation date"""
    try:
        irrigation_events = calculate_irrigation_events(plantation_date, frequency_days, total_days)
        return {
            "plantation_date": plantation_date,
            "frequency_days": frequency_days,
            "total_days": total_days,
            "total_events": len(irrigation_events),
            "irrigation_events": irrigation_events
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

@app.get("/irrigation/count")
async def count_irrigation_events_endpoint(
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    interval_days: int = Query(default=21, description="Interval between irrigation events", ge=1, le=365)
):
    """Count irrigation events from plantation date to current date"""
    try:
        result = count_irrigation_events(plantation_date, interval_days)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

# -------------------------------
# Harvest Analysis Endpoints
# -------------------------------

@app.get("/harvest/timing", response_model=HarvestAnalysis)
async def get_harvest_timing(
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    variety_type: str = Query(default="Mid", description="Sugarcane variety type: Early, Mid, or Late maturing")
):
    """Calculate days to harvest and current growth stage information"""
    try:
        if variety_type not in ["Early", "Mid", "Late"]:
            raise HTTPException(status_code=400, detail="Variety type must be Early, Mid, or Late")
        
        result = calculate_harvest_timing(plantation_date, variety_type)
        
        return HarvestAnalysis(
            plot_name="General Analysis",
            plantation_date=result["plantation_date"],
            current_date=result["current_date"],
            days_since_planting=result["days_since_planting"],
            days_to_harvest=result["days_to_harvest"],
            estimated_harvest_date=result["estimated_harvest_date"],
            current_growth_stage=result["current_growth_stage"],
            next_growth_stage=result["next_growth_stage"],
            days_to_next_stage=result["days_to_next_stage"],
            harvest_readiness_percentage=result["harvest_readiness_percentage"],
            recommended_actions=result["recommended_actions"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

@app.get("/harvest/optimal-window")
async def get_optimal_harvest_window_endpoint(
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    variety_type: str = Query(default="Mid", description="Sugarcane variety type: Early, Mid, or Late maturing")
):
    """Get optimal harvest window for sugarcane"""
    try:
        if variety_type not in ["Early", "Mid", "Late"]:
            raise HTTPException(status_code=400, detail="Variety type must be Early, Mid, or Late")
        
        return get_optimal_harvest_window(plantation_date, variety_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

@app.get("/harvest/growth-stages")
async def get_growth_stages_endpoint(
    variety_type: str = Query(default="Mid", description="Sugarcane variety type: Early, Mid, or Late maturing")
):
    """Get detailed growth stages for sugarcane variety"""
    if variety_type not in ["Early", "Mid", "Late"]:
        raise HTTPException(status_code=400, detail="Variety type must be Early, Mid, or Late")
    
    growth_stages = get_growth_stages(variety_type)
    
    return {
        "variety_type": variety_type,
        "growth_stages": [
            GrowthStage(
                stage_name=stage["stage_name"],
                days_from_planting=stage["days"],
                description=stage["description"],
                management_actions=stage["actions"]
            ) for stage in growth_stages
        ]
    }

@app.get("/harvest/planning", response_model=HarvestPlanning)
async def get_harvest_planning(
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    variety_type: str = Query(default="Mid", description="Sugarcane variety type: Early, Mid, or Late maturing")
):
    """Get comprehensive harvest planning information"""
    try:
        if variety_type not in ["Early", "Mid", "Late"]:
            raise HTTPException(status_code=400, detail="Variety type must be Early, Mid, or Late")
        
        timing_info = calculate_harvest_timing(plantation_date, variety_type)
        optimal_window = get_optimal_harvest_window(plantation_date, variety_type)
        growth_stages = get_growth_stages(variety_type)
        
        # Generate harvest recommendations
        harvest_recommendations = []
        if timing_info["days_to_harvest"] <= 30:
            harvest_recommendations.extend([
                "Schedule harvest equipment and labor",
                "Monitor weather conditions for optimal harvest timing",
                "Check sugar content and quality parameters",
                "Prepare storage and transportation logistics"
            ])
        elif timing_info["days_to_harvest"] <= 60:
            harvest_recommendations.extend([
                "Begin harvest planning and resource allocation",
                "Monitor crop maturity indicators",
                "Prepare harvest equipment maintenance"
            ])
        else:
            harvest_recommendations.extend([
                "Continue regular crop monitoring",
                "Maintain optimal irrigation and nutrition",
                "Monitor for pests and diseases"
            ])
        
        return HarvestPlanning(
            plantation_date=plantation_date,
            variety_type=variety_type,
            estimated_harvest_date=timing_info["estimated_harvest_date"],
            total_growth_days=timing_info["total_growth_days"],
            growth_stages=[
                GrowthStage(
                    stage_name=stage["stage_name"],
                    days_from_planting=stage["days"],
                    description=stage["description"],
                    management_actions=stage["actions"]
                ) for stage in growth_stages
            ],
            harvest_recommendations=harvest_recommendations
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

@app.get("/plots/{plot_name}/harvest-analysis", response_model=HarvestAnalysis)
async def get_plot_harvest_analysis(
    plot_name: str,
    plantation_date: str = Query(..., description="Plantation date in YYYY-MM-DD format"),
    variety_type: str = Query(default="adsali", description="Sugarcane variety type")
):
    """Get harvest analysis for a specific plot"""
    plot_data = find_plot_by_name(plot_name) 
    if not plot_data:
        # Provide helpful error message with available plots
        available_plots = list(plot_dict.keys())[:10]  # Show first 10 plots
        raise HTTPException(
            status_code=404, 
            detail=f"Plot '{plot_name}' not found. Available plots: {available_plots}"
        )
    
    try:
        if variety_type not in ["adsali", "suru", "ratoon","pre-seasonal"]:
            raise HTTPException(status_code=400, detail="Variety type mismatch")
        
        result = calculate_harvest_timing(plantation_date, variety_type)
        
        return HarvestAnalysis(
            plot_name=plot_name,
            plantation_date=result["plantation_date"],
            current_date=result["current_date"],
            days_since_planting=result["days_since_planting"],
            days_to_harvest=result["days_to_harvest"],
            estimated_harvest_date=result["estimated_harvest_date"],
            current_growth_stage=result["current_growth_stage"],
            next_growth_stage=result["next_growth_stage"],
            days_to_next_stage=result["days_to_next_stage"],
            harvest_readiness_percentage=result["harvest_readiness_percentage"],
            recommended_actions=result["recommended_actions"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
@app.post("/sugarcane-harvest", response_model=Dict[str, Any])
async def sugarcane_harvest_endpoint(
    plot_name: str,
    plantation_date: str | None = Query(None),
    end_date: str = Query(default_factory=lambda: date.today().strftime("%Y-%m-%d")),
):
 
    if plot_name not in plot_dict:
        raise HTTPException(status_code=404, detail="Plot not found")
 
    plot_data = plot_dict[plot_name]
    props = plot_data["properties"]
 
    auto_plantation_date = props.get("plantation_date")
    plantation_type = props.get("plantation_type")
 
    if plantation_date is None:
        if not auto_plantation_date:
            raise HTTPException(status_code=400, detail="No plantation_date available")
        plantation_date = auto_plantation_date
 
    plantation_dt = datetime.strptime(plantation_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
 
    if end_dt < plantation_dt:
        raise HTTPException(status_code=400, detail="end_date < plantation_date")
 
    geometry = plot_data["geometry"]
 
    result = detect_sugarcane_harvest_for_plot(
        geometry=geometry,
        plantation_date=plantation_date,
        end_date=end_date,
    )
 
    # ------------------------------
    # Plantation Age Logic
    # ------------------------------
    month_diff = (end_dt.year - plantation_dt.year) * 12 + (end_dt.month - plantation_dt.month)
 
    plantation_rules = {
        "Adsali":      (17, 18),
        "Ratoon":      (10, 11),
        "Preseasonal": (14, 14),
        "Suru":        (12, 12),
    }
 
    low, high = plantation_rules.get(plantation_type, (None, None))
 
    if low is not None:
        if month_diff < low:
            status = "growing"
        elif low <= month_diff <= high:
            status = "harvested" if result["has_harvest"] else "harvest_ready"
        else:
            status = "harvested" if result["has_harvest"] else "growing"
    else:
        status = "harvested" if result["has_harvest"] else "growing"
 
    feature = {
        "type": "Feature",
        "geometry": {
            "type": plot_data["geom_type"],
            "coordinates": plot_data["original_coords"],
        },
        "properties": {
            "plot_name": plot_name,
            "plantation_date": plantation_date,
            "plantation_type": plantation_type,
            "end_date": end_date,
            "harvest_date": result["harvest_date"],
            "has_harvest": result["has_harvest"],
            "harvest_status": status,
            "image_count": result["image_count"],
            "last_updated": datetime.now().isoformat(),
        },
    }
 
    return {
        "type": "FeatureCollection",
        "features": [feature],
        "harvest_summary": {
            "plantation_date": plantation_date,
            "plantation_type": plantation_type,
            "end_date": end_date,
            "harvest_date": result["harvest_date"],
            "has_harvest": result["has_harvest"],
            "harvest_status": status,
            "hpi_threshold": result["hpi_threshold"],
            "nhpi_threshold": result["nhpi_threshold"],
            "ndvi_drop_threshold": result["ndvi_drop_threshold"],
        },
        "timeseries": result["timeseries"],
    }
 
 
 
# -------------------------------
# Sync Endpoints for Django Integration
# -------------------------------

@app.post("/sync/plot")
async def sync_plot(plot_data: Dict[str, Any]):
    """Sync a single plot from Django"""
    try:
        plot_name = plot_data.get("name", f"plot_{plot_data.get('id', 'unknown')}")
        
        # Extract geometry data
        geometry_data = plot_data.get("geometry", {})
        geom_type = geometry_data.get("type", "Polygon")
        coordinates = geometry_data.get("coordinates", [])
        
        # Create Earth Engine geometry
        if geom_type == "Polygon" and coordinates:
            # Handle polygon coordinates - ensure proper format
            if isinstance(coordinates[0], (list, tuple)):
                if isinstance(coordinates[0][0], (list, tuple)):
                    # Already in proper format
                    geom = ee.Geometry.Polygon(coordinates)
                else:
                    # Single polygon ring
                    geom = ee.Geometry.Polygon([coordinates])
            else:
                raise ValueError("Invalid polygon coordinates format")
        elif geom_type == "Point" and coordinates:
            # Handle point coordinates
            if len(coordinates) >= 2:
                geom = ee.Geometry.Point(coordinates[:2])  # Use only lat, lng
            else:
                raise ValueError("Invalid point coordinates format")
        else:
            raise ValueError("Invalid geometry data")
        
        # Update plot_dict
        plot_dict[plot_name] = {
            "geometry": geom,
            "geom_type": geom_type,
            "original_coords": coordinates,
            "properties": plot_data.get("properties", {}),
            "django_id": plot_data.get("id")
        }
        
        return {"status": "success", "message": f"Plot {plot_name} synced successfully", "plot_name": plot_name}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sync plot: {str(e)}")

@app.post("/sync/plots")
async def sync_plots(plots_data: Dict[str, List[Dict[str, Any]]]):
    """Sync multiple plots from Django"""
    try:
        plots = plots_data.get("plots", [])
        synced_count = 0
        
        for plot_data in plots:
            try:
                plot_name = plot_data.get("name", f"plot_{plot_data.get('id', 'unknown')}")
                
                # Extract geometry data
                geometry_data = plot_data.get("geometry", {})
                geom_type = geometry_data.get("type", "Polygon")
                coordinates = geometry_data.get("coordinates", [])
                
                # Create Earth Engine geometry
                if geom_type == "Polygon" and coordinates:
                    if isinstance(coordinates[0], (list, tuple)):
                        if isinstance(coordinates[0][0], (list, tuple)):
                            # Multi-polygon or polygon with holes
                            geom = ee.Geometry.Polygon(coordinates)
                        else:
                            # Simple polygon
                            geom = ee.Geometry.Polygon([coordinates])
                    else:
                        continue  # Skip invalid geometries
                elif geom_type == "Point" and coordinates:
                    if len(coordinates) >= 2:
                        geom = ee.Geometry.Point(coordinates[:2])
                    else:
                        continue  # Skip invalid geometries
                else:
                    continue  # Skip invalid geometries
                
                # Update plot_dict
                plot_dict[plot_name] = {
                    "geometry": geom,
                    "geom_type": geom_type,
                    "original_coords": coordinates,
                    "properties": plot_data.get("properties", {}),
                    "django_id": plot_data.get("id")
                }
                synced_count += 1
            except Exception as e:
                # Log error but continue with other plots
                print(f"Error syncing plot {plot_data.get('id', 'unknown')}: {str(e)}")
                continue
        
        return {"status": "success", "message": f"Synced {synced_count} plots successfully", "synced_count": synced_count}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sync plots: {str(e)}")

@app.delete("/sync/plot/{plot_id}")
async def delete_plot(plot_id: int):
    """Delete a plot from events.py"""
    try:
        # Find plot by Django ID
        plot_name_to_delete = None
        for plot_name, plot_info in plot_dict.items():
            if plot_info.get("django_id") == plot_id:
                plot_name_to_delete = plot_name
                break
        
        if plot_name_to_delete:
            del plot_dict[plot_name_to_delete]
            return {"status": "success", "message": f"Plot {plot_name_to_delete} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Plot with Django ID {plot_id} not found")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to delete plot: {str(e)}")

@app.get("/sync/status")
async def get_sync_status():
    """Get sync status and plot count"""
    return {
        "total_plots": len(plot_dict),
        "plots_with_django_ids": len([p for p in plot_dict.values() if p.get("django_id")]),
        "plots_from_geojson": len([p for p in plot_dict.values() if not p.get("django_id")]),
        "status": "active"
    }

@app.post("/refresh-from-django")
async def refresh_from_django():
    """Manually refresh all plots from Django - useful after Django restart"""
    try:
        global plot_dict
        print("ðŸ”„ Manual refresh from Django requested for events.py...")
        plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)
        return {"status": "success", "message": f"Refreshed {len(plot_dict)} plots."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh from Django: {str(e)}")

 
# -------------------------------
# Run Server
# -------------------------------
 
if __name__ == "__main__":
    uvicorn.run("eventsBack:app", host="0.0.0.0", port=9000, reload=True)
 
