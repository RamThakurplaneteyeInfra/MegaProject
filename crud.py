from sqlalchemy import text
from sqlalchemy.orm import Session

def list_villages(db: Session):
    sql = text("""
        SELECT
            v.id,
            v.village_name,
            v.district,
            v.sub_dist,
            v.state,
            COALESCE(
                json_agg(
                    json_build_object(
                        'id', b.id,
                        'geom', ST_AsGeoJSON(b.geom)::json
                    )
                ) FILTER (WHERE b.id IS NOT NULL),
                '[]'::json
            ) AS boundaries
        FROM village v
        LEFT JOIN village_boundaries b ON b.village_id = v.id
        GROUP BY v.id
        ORDER BY v.id;
    """)

    return db.execute(sql).mappings().all()
from models import Village   # ✅ THIS LINE FIXES IT

def get_sub_districts_by_district(db: Session, district_name: str):
    return (
        db.query(Village.sub_district)
        .filter(Village.district.ilike(district_name))
        .distinct()
        .order_by(Village.sub_district)
        .all()
    )