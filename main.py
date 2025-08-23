from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from collections import deque

from core import state
from core.config import Settings
from services.zones_store import load_zones
from worker.infer import run_inference
from api.routes import video, ws, zones, events, health

def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="ZoneSight API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(video.router)
    app.include_router(ws.router)
    app.include_router(zones.router)
    app.include_router(events.router)
    app.include_router(health.router)

    @app.on_event("startup")
    def on_startup():
        # reset events capacity from settings
        state.events = deque(maxlen=settings.EVENTS_MAX)
        # load zones
        with state.zones_lock:
            state.zones = load_zones()
        # start worker
        th = threading.Thread(target=run_inference, args=(settings,), daemon=True)
        th.start()

    return app

app = create_app()
