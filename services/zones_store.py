from pathlib import Path
from typing import List
import json
from models.zone import Zone

ZONES_PATH = Path("zones.json")

def load_zones() -> List[Zone]:
    if not ZONES_PATH.exists():
        return []
    text = ZONES_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        zones_list = json.loads(text)
        if not isinstance(zones_list, list):
            return []
        return [Zone(**z) for z in zones_list]
    except json.JSONDecodeError:
        return []

def save_zones(zones: List[Zone]) -> None:
    ZONES_PATH.write_text(
        json.dumps([z.model_dump() for z in zones], indent=2),
        encoding="utf-8"
    )
