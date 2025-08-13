import os, time, threading, json
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
from typing import Dict, Any, Optional

import cv2
import torch
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

from ultralytics import YOLO
from config import Settings

# ---- Config ----
settings = Settings()
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = bool(int(os.getenv("HALF", "1"))) and torch.cuda.is_available()

# ---- Globals updated by worker ----
latest_jpeg: Optional[bytes] = None
latest_metrics: Dict[str, Any] = {
    "fps": 0.0,
    "counts": {},
    "ts": 0.0,
    "img_shape": [0, 0],
}
stop_flag = False
lock = threading.Lock()

# ---- Model ----
model = YOLO("yolov8n.pt")


def open_capture():
    if settings.source_is_index:
        # For Windows webcam
        return cv2.VideoCapture(int(settings.SOURCE), cv2.CAP_DSHOW)
    # For RTSP/HTTP sources
    return cv2.VideoCapture(settings.SOURCE)


def infer_worker():
    global latest_jpeg, latest_metrics
    # open camera/video capture device
    cap = open_capture()
    if not cap.isOpened():
        # retry opening camera if camera failed to open (unless force stopped)
        while not cap.isOpened() and not stop_flag:
            time.sleep(1.0)
            cap.open(
                int(settings.SOURCE) if settings.source_is_index else settings.SOURCE
            )

    t_last = time.perf_counter()
    ema_fps = None

    while not stop_flag:
        ok, frame = cap.read()
        if not ok:
            # try to recover the stream
            cap.release()
            time.sleep(0.5)
            cap = open_capture()
            continue

        # Ultralytics built-in tracking (single frame)
        results = model.track(
            frame,
            imgsz=settings.IMG_SIZE,
            conf=settings.CONF,
            iou=0.5,
            device=DEVICE,
            half=HALF,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )[0]

        # counts per class label
        names = model.names
        boxes = results.boxes
        
        ids: list[int] = []
        cls: list[int] = []
        conf: list[float] = []
        xyxy: list[list[float]] = []
        
        if boxes is not None:
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
            if boxes.cls is not None:
                cls = boxes.cls.cpu().tolist()
            if boxes.conf is not None:
                conf  = boxes.conf.cpu().tolist()
            if boxes.xyxy is not None:
                xyxy  = boxes.xyxy.cpu().tolist()
        # {0: "person", 1: "bicycle", 2: "car", ...} + [0.0, 2.0, 0.0, ...] -> ["person", "car", "person", ...]
        labels = [names[int(i)] for i in cls]
        # ["person", "car", "person", ...] -> {"person": 2, "car": 1, ...}
        counts = Counter(labels)
        
        # Build per-detection payload for the UI
        detection_list = [
            {
                "tracking_id": int(i),
                "cls": names[int(c)],
                "conf": float(p),
                "xyxy": [float(v) for v in b]
            }
            for i, c, p, b in zip(ids, cls, conf, xyxy)
        ]

        # draw + encode JPEG
        annotated = results.plot()  # BGR
        ok_jpg, buf = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        )
        if not ok_jpg:
            continue
        jpg = buf.tobytes()

        # measuring and smoothing the frame rate
        now = time.perf_counter()
        inst_fps = 1.0 / max(now - t_last, 1e-6)
        t_last = now
        ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

        # thread safety
        with lock:
            latest_jpeg = jpg
            h, w = annotated.shape[:2]
            latest_metrics = {
                "ts": time.time(),
                "fps": round(ema_fps or inst_fps, 2),
                "img_shape": [h, w],
                "counts": dict(counts),
                "detections": detection_list,
            }

    cap.release()


app = FastAPI()

# Add CORS
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    th = threading.Thread(target=infer_worker, daemon=True)
    th.start()


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "source": settings.SOURCE,
        "img_size": settings.IMG_SIZE,
        "conf": settings.CONF,
        "streaming_fps": settings.STREAM_FPS,
        "jpeg_quality": settings.JPEG_QUALITY,
        "half": HALF,
    }


@app.get("/video")
def video():
    boundary = "frame"

    def gen():
        # MJPEG stream
        while True:
            with lock:
                jpg = latest_jpeg
            if jpg is not None:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpg)).encode()
                    + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
            time.sleep(1.0 / settings.STREAM_FPS)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

@app.websocket("/ws")
async def ws_metrics(ws: WebSocket):
    # rejects unknown origins
    origin = ws.headers.get("origin", "")
    if origin not in ALLOW_ORIGINS:
        await ws.close(code=1008)
        return
    
    await ws.accept()
    try:
        while True:
            with lock:
                msg = json.dumps(latest_metrics)
            await ws.send_text(msg)
            # throttles so that metrics are updated every 200ms
            await asyncio.sleep(
                0.2
            )  
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
