from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, field_validator
import json

ZONES_PATH = Path("zones.json")

class Zone(BaseModel):
    id: str
    name: str
    points: List[Tuple[float, float]]  # normalized [x,y], 0..1

    @field_validator("points")
    @classmethod
    def _norm(cls, pts: List[Tuple[float, float]]):
        if len(pts) < 3:
            raise ValueError("polygon must have >= 3 points")
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("points must be normalized 0..1")
        return pts

def load_zones() -> List[Zone]:
    if not ZONES_PATH.exists():
        return []
    text = ZONES_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []  # empty file -> no zones
    try:
        raw = json.loads(text)
        if not isinstance(raw, list):
            return []
        return [Zone(**z) for z in raw]
    except json.JSONDecodeError:
        # Log in real code; for now, treat as no zones
        return []

def save_zones(zones: List[Zone]) -> None:
    ZONES_PATH.write_text(
        json.dumps([z.model_dump() for z in zones], indent=2),
        encoding="utf-8"
    )
