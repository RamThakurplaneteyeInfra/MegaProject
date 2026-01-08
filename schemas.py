from pydantic import BaseModel
from typing import List

class BoundaryOut(BaseModel):
    coordinates: list

class VillageOut(BaseModel):
    id: int
    village_name: str
    district: str
    sub_dist: str
    state: str
    boundaries: List[BoundaryOut]
