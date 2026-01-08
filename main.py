from fastapi import FastAPI, UploadFile, Depends,HTTPException,Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon
from crud import get_sub_districts_by_district
from db import get_db, engine
from models import Base, Village, VillageBoundary,District
from kml import parse_kml,parse_district_kml

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Village Boundary API")


# ------------------------------------------------------------------
# UPLOAD KML
# ------------------------------------------------------------------
@app.post("/upload-kml")
async def upload_kml(file: UploadFile, db: Session = Depends(get_db)):
    content = await file.read()
    villages = parse_kml(content)

    for v in villages:
        village = Village(
            village_name=v["village_name"],
            district=v["district"],
            sub_dist=v["sub_dist"],
            state=v["state"]
        )
        db.add(village)
        db.flush()

        polygon = Polygon(v["coordinates"])
        boundary = VillageBoundary(
            village_id=village.id,
            geom=from_shape(polygon, srid=4326)
        )
        db.add(boundary)

    db.commit()
    return {"inserted": len(villages)}

@app.post("/upload-district-kml")
async def upload_district_kml(
    file: UploadFile,
    db: Session = Depends(get_db)
):
    content = await file.read()
    districts = parse_district_kml(content)

    inserted = 0

    for d in districts:
        exists = db.query(District).filter(
            District.district_name == d["district"]
        ).first()

        if exists:
            continue

        polygon = Polygon(d["coordinates"])

        rec = District(
            district_name=d["district"],
            state=d["state"],
            geom=from_shape(polygon, srid=4326)
        )

        db.add(rec)
        inserted += 1

    db.commit()

    return {
        "total_found": len(districts),
        "inserted": inserted
    }
@app.get("/Alldistricts")
def get_all_districts(db: Session = Depends(get_db)):
    """
    Returns all districts with boundaries (GeoJSON-ready)
    """

    sql = text("""
        SELECT
            d.id,
            d.district_name,
            d.state,
            GeometryType(d.geom) AS geom_type,
            ST_AsGeoJSON(d.geom)::json -> 'coordinates' AS coordinates
        FROM districts d
        ORDER BY d.district_name
    """)

    rows = db.execute(sql).mappings().all()

    districts = []
    for r in rows:
        districts.append({
            "district_name": r["district_name"],
            "state": r["state"],
            "geom_type": r["geom_type"].replace("ST_", ""),
            "coordinates": r["coordinates"]
        })

    return {
        "total": len(districts),
        "districts": districts
    }
@app.get("/sub-districts")
def get_sub_districts(
    district: str = Query(..., description="District name, e.g. Beed"),
    db: Session = Depends(get_db)
):
    result = get_sub_districts_by_district(db, district)

    return {
        "district": district,
        "sub_districts": [r[0] for r in result],
        "count": len(result)
    } 
@app.get("/villages-by-subdistrict")
def villages_by_subdistrict(
    subdistrict: str = Query(..., description="Sub-district / Taluka name"),
    db: Session = Depends(get_db)
):
    sql = text("""
        SELECT
            v.village_name,
            v.district,
            v.sub_dist,
            v.state,
            GeometryType(b.geom) AS geom_type,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coordinates
        FROM village v
        JOIN village_boundaries b
            ON b.village_id = v.id
        WHERE LOWER(v.sub_dist) = LOWER(:subdistrict)
        ORDER BY v.village_name
    """)

    rows = db.execute(sql, {"subdistrict": subdistrict}).mappings().all()

    villages = []
    for r in rows:
        villages.append({
            "village_name": r["village_name"],
            "district": r["district"],
            "subdistrict": r["sub_dist"],
            "state": r["state"],
            "geometry": {
                "type": r["geom_type"].replace("ST_", ""),
                "coordinates": r["coordinates"]
            }
        })

    return {
        "subdistrict": subdistrict,
        "total_villages": len(villages),
        "villages": villages
    }
@app.get("/districts")
def list_districts(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT DISTINCT district
        FROM village
        ORDER BY district
    """))
    return {"districts": [r[0] for r in result.fetchall()]}

@app.get("/subdistricts")
def list_subdistricts(db: Session = Depends(get_db)):
    sql = text("""
        SELECT DISTINCT sub_dist
        FROM village
        WHERE sub_dist IS NOT NULL
        ORDER BY sub_dist
    """)

    rows = db.execute(sql).all()

    return {
        "subdistricts": [r[0] for r in rows]
    }


@app.get("/village-by-subdist")
def get_village_by_subdist(
    sub_dist: str,
    village_name: str,
    db: Session = Depends(get_db)
):
    """
    Returns single village using sub-district and village name
    """

    sql = text("""
        SELECT
            v.village_name,
            v.district,
            v.sub_dist,
            GeometryType(b.geom) AS geom_type,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coordinates
        FROM village v
        JOIN village_boundaries b
            ON b.village_id = v.id
        WHERE LOWER(v.sub_dist) = LOWER(:sub_dist)
          AND LOWER(v.village_name) = LOWER(:village_name)
        LIMIT 1
    """)

    row = db.execute(
        sql,
        {
            "sub_dist": sub_dist,
            "village_name": village_name
        }
    ).mappings().first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Village not found for given sub-district"
        )

    return {
        "village_name": row["village_name"],
        "district": row["district"],
        "sub_dist": row["sub_dist"],
        "geom_type": row["geom_type"].replace("ST_", ""),
        "coordinates": row["coordinates"]
    }
# ------------------------------------------------------------------
# DISTRICTS
# ------------------------------------------------------------------


@app.get("/districts/geo")
def district_with_villages(district: str, db: Session = Depends(get_db)):
    sql = text("""
        SELECT
            v.village_name,
            v.sub_dist,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coords
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.district) = LOWER(:district)
    """)
    rows = db.execute(sql, {"district": district}).mappings().all()

    return {
        "district": district,
        "villages": rows
    }


# ------------------------------------------------------------------
# SUB-DISTRICTS (TALUKA)
# ------------------------------------------------------------------
@app.get("/subdistricts/by-district")
def subdistricts_by_district(district: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT DISTINCT sub_dist
        FROM village
        WHERE LOWER(district) = LOWER(:district)
        ORDER BY sub_dist
    """), {"district": district})

    return {
        "district": district,
        "subdistricts": [r[0] for r in result.fetchall()]
    }


@app.get("/subdistricts/by-district/geo")
def subdistricts_with_coords(district: str, db: Session = Depends(get_db)):
    sql = text("""
        SELECT
            v.sub_dist,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coords
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.district) = LOWER(:district)
    """)

    rows = db.execute(sql, {"district": district}).mappings().all()

    grouped = {}
    for r in rows:
        grouped.setdefault(r["sub_dist"], []).append(r["coords"])

    return {
        "district": district,
        "subdistricts": grouped
    }


# ------------------------------------------------------------------
# VILLAGES
# ------------------------------------------------------------------
@app.get("/villages")
def all_villages(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT DISTINCT village_name
        FROM village
        ORDER BY village_name
    """))
    return {"villages": [r[0] for r in result.fetchall()]}


@app.get("/villages/by-district")
def villages_by_district(district: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT DISTINCT village_name
        FROM village
        WHERE LOWER(district) = LOWER(:district)
        ORDER BY village_name
    """), {"district": district})

    return {
        "district": district,
        "villages": [r[0] for r in result.fetchall()]
    }


@app.get("/villages/by-district/geo")
def villages_by_district_geo(district: str, db: Session = Depends(get_db)):
    sql = text("""
        SELECT
            v.village_name,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coords
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.district) = LOWER(:district)
    """)
    rows = db.execute(sql, {"district": district}).mappings().all()

    return {
        "district": district,
        "villages": rows
    }


@app.get("/villages/by-subdist")
def villages_by_subdist(sub_dist: str, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT DISTINCT village_name
        FROM village
        WHERE LOWER(sub_dist) = LOWER(:sub_dist)
        ORDER BY village_name
    """), {"sub_dist": sub_dist})

    return {
        "sub_dist": sub_dist,
        "villages": [r[0] for r in result.fetchall()]
    }


@app.get("/villages/by-subdist/geo")
def villages_by_subdist_geo(sub_dist: str, db: Session = Depends(get_db)):
    sql = text("""
        SELECT
            v.village_name,
            ST_AsGeoJSON(b.geom)::json -> 'coordinates' AS coords
        FROM village v
        JOIN village_boundaries b ON b.village_id = v.id
        WHERE LOWER(v.sub_dist) = LOWER(:sub_dist)
    """)
    rows = db.execute(sql, {"sub_dist": sub_dist}).mappings().all()

    return {
        "sub_dist": sub_dist,
        "villages": rows
    }


# ------------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
if __name__ == "__main__": 
    import uvicorn 
    uvicorn.run("main:app", host="0.0.0.0", port=9099,reload=True)