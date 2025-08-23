import json, asyncio
from fastapi import APIRouter, WebSocket
from core import state
from core.config import Settings

router = APIRouter()

@router.websocket("/ws")
async def ws_metrics(ws: WebSocket):
    settings = Settings()
    origin = ws.headers.get("origin", "")
    if origin not in settings.allowed_origins:
        await ws.close(code=1008); return

    await ws.accept()
    try:
        while True:
            with state.frame_lock:
                msg = json.dumps(state.latest_metrics)
            await ws.send_text(msg)
            await asyncio.sleep(0.2)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
