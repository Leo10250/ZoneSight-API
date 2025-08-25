from fastapi import APIRouter
from typing import List
from core import state
from models.zone import Zone
from services.zones_store import load_zones, save_zones

router = APIRouter(prefix="/zones", tags=["zones"])

@router.get("", response_model=list[Zone])
def get_zones():
    with state.zones_lock:
        return list(state.zones)

@router.put("", response_model=list[Zone])
def put_zones(new_zones: list[Zone]):
    with state.zones_lock:
        state.zones = new_zones
        save_zones(state.zones)
        return state.zones
