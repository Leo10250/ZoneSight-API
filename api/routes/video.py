import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from core import state
from core.config import Settings

router = APIRouter()

@router.get("/video")
def video():
    settings = Settings()
    boundary = "frame"

    def gen():
        while True:
            with state.frame_lock:
                jpg = state.latest_jpeg
            if jpg is not None:
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                       jpg + b"\r\n")
            time.sleep(1.0 / settings.STREAM_FPS)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
