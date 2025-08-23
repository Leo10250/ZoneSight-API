from typing import List, Tuple
from pydantic import BaseModel, field_validator

class Zone(BaseModel):
    id: str
    name: str
    points: List[Tuple[float, float]]  # normalized [x,y] in [0,1]

    @field_validator("points")
    @classmethod
    def _norm(cls, pts: List[Tuple[float, float]]):
        if len(pts) < 3:
            raise ValueError("polygon must have >= 3 points")
        for x, y in pts:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("points must be normalized 0..1")
        return pts
