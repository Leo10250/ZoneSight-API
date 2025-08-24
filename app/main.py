from contextlib import asynccontextmanager
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- startup ---
        state.events = deque(maxlen=settings.EVENTS_MAX)
        with state.zones_lock:
            state.zones = load_zones()
        worker = threading.Thread(target=run_inference, args=(settings,), daemon=True)
        worker.start()
        try:
            yield
        finally:
            # --- shutdown ---
            state.stop_flag = True
            worker.join(timeout=2)

    app = FastAPI(title="ZoneSight API", lifespan=lifespan)

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

    return app


app = create_app()
