import os, time, threading, json
from collections import Counter
from typing import Dict, Any, Optional

import cv2
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

from ultralytics import YOLO

# ---- Config ----
SOURCE = int(os.getenv("CAM_INDEX", "1"))      # 0 = default webcam, or RTSP URL string
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF = float(os.getenv("CONF", "0.25"))
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = bool(int(os.getenv("HALF", "1"))) and torch.cuda.is_available()
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
STREAM_FPS = float(os.getenv("STREAM_FPS", "20"))  # target MJPEG send rate

# ---- Globals updated by worker ----
latest_jpeg: Optional[bytes] = None
latest_metrics: Dict[str, Any] = {"fps": 0.0, "counts": {}, "ts": 0.0, "img_shape": [0, 0]}
stop_flag = False
lock = threading.Lock()

# ---- Model ----
model = YOLO("yolov8n.pt")

def infer_worker():
    global latest_jpeg, latest_metrics
    # open camera/video capture device
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # retry opening camera if camera failed to open (unless force stopped)
        while not cap.isOpened() and not stop_flag:
            time.sleep(1.0)
            cap.open(SOURCE)

    t_last = time.perf_counter()
    ema_fps = None

    while not stop_flag:
        ok, frame = cap.read()
        if not ok:
            # try to recover the stream
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
            continue

        # Ultralytics inference (single frame)
        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            half=HALF,
            verbose=False
        )[0]

        # counts per class label
        names = model.names
        cls = results.boxes.cls.tolist() if results.boxes is not None else []
        # {0: "person", 1: "bicycle", 2: "car", ...} + [0.0, 2.0, 0.0, ...] -> ["person", "car", "person", ...]
        labels = [names[int(i)] for i in cls]
        # ["person", "car", "person", ...] -> {"person": 2, "car": 1, ...}
        counts = Counter(labels)

        # draw + encode JPEG
        annotated = results.plot()  # BGR
        ok_jpg, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
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
            }

    cap.release()

app = FastAPI()

@app.on_event("startup")
def _startup():
    th = threading.Thread(target=infer_worker, daemon=True)
    th.start()

@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}

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
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )
            time.sleep(1.0 / STREAM_FPS)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

@app.websocket("/ws")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with lock:
                msg = json.dumps(latest_metrics)
            await ws.send_text(msg)
            await ws.receive_text()  # optional: if you want pings from client; else use sleep
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
