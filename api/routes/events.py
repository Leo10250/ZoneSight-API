from fastapi import APIRouter
from core import state
from core.config import Settings

router = APIRouter(tags=["events"])

@router.get("/events")
def get_events(limit: int = 50):
    settings = Settings()
    return list(state.events)[:max(1, min(limit, settings.EVENTS_MAX))]
