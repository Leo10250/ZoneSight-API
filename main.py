import os, time, threading, json
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter, defaultdict, deque
from typing import Dict, Any, Optional, List, Deque

import cv2
import torch
import asyncio
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from dataclasses import dataclass

from ultralytics import YOLO
from config import Settings
from zones import Zone, load_zones, save_zones
from geometry import point_in_poly

# ---- Config ----
settings = Settings()
# Lock for zones (reads/writes)
zones_lock = threading.RLock()
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = bool(int(os.getenv("HALF", "1"))) and torch.cuda.is_available()
TRACK_MISS_TTL = float(
    os.getenv("TRACK_MISS_TTL", "1.5")
)  # seconds unseen before exiting
EVENTS_MAX = int(os.getenv("EVENTS_MAX", "200"))  # size of in-memory event feed
OCCUPANCY_CLASSES = {s.strip() for s in os.getenv("OCCUPANCY_CLASSES", "person").split(",")}

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

@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    zone: str | None = None
    entered_at: float | None = None  # when we entered current zone
    cls: str | None = None          # class label (e.g., "person")


# Live in-memory state (owned by the worker thread)
tracks: dict[int, TrackState] = {}  # NEW
events: Deque[dict[str, Any]] = deque(maxlen=EVENTS_MAX)  # NEW


def open_capture():
    if settings.source_is_index:
        # For Windows webcam
        return cv2.VideoCapture(int(settings.SOURCE), cv2.CAP_DSHOW)
    # For RTSP/HTTP sources
    return cv2.VideoCapture(settings.SOURCE)


def _norm_center(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) * 0.5) / max(w, 1)
    cy = ((y1 + y2) * 0.5) / max(h, 1)
    return cx, cy


def _push_event(
    ev_type: str,
    now: float,
    *,
    track_id: int,
    zone_to: str | None,
    zone_from: str | None = None,
    dwell_s: float | None = None,
):
    events.appendleft(
        {
            "ts": time.time(),  # wall-clock for UI readability
            "type": ev_type,  # "entry" | "exit" | "transfer"
            "track_id": int(track_id),
            "zone": zone_to,  # for exit, this is the zone we left
            "from": zone_from,
            "dwell_s": round(dwell_s, 2) if dwell_s is not None else None,
        }
    )


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
            tracker="bytetrack.yaml",
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
                conf = boxes.conf.cpu().tolist()
            if boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().tolist()
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
                "xyxy": [float(v) for v in b],
            }
            for i, c, p, b in zip(ids, cls, conf, xyxy)
        ]

        h, w = frame.shape[:2]

        # --- snapshot zones for this frame (thread-safe) ---
        with zones_lock:
            current_zones = tuple(zones)
            # (optional) names if you want to label later
            # zone_names = {z.id: z.name for z in current_zones}

        # Assign zone to each detection via centroid-in-polygon (normalized)
        for d in detection_list:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = _norm_center(x1, y1, x2, y2, w, h)
            zhit = None
            for z in current_zones:
                if point_in_poly(cx, cy, z.points):
                    zhit = z.id
                    break
            d["zone"] = zhit

        # ---- Phase 3: update track states, produce events, compute dwell ----
        now = time.perf_counter()
        seen_ids = set()

        for d in detection_list:
            tid = d.get("tracking_id")
            if tid is None:
                continue
            # restrict events/dwell to humans only
            if d.get("cls") not in OCCUPANCY_CLASSES:   # e.g., "person"
                continue
            
            seen_ids.add(tid)

            st = tracks.get(tid)
            if st is None:
                st = TrackState(
                    id=tid, first_seen=now, last_seen=now, zone=None, entered_at=None
                )
                tracks[tid] = st
            else:
                st.last_seen = now
                
            # (create/update st as you already do)
            st.cls = d.get("cls") or st.cls      # NEW: remember/update class label

            new_zone = d.get("zone")
            prev_zone = st.zone

            if new_zone != prev_zone:
                # ENTRY
                if prev_zone is None and new_zone is not None:
                    st.zone = new_zone
                    st.entered_at = now
                    _push_event(
                        "entry", now, track_id=tid, zone_to=new_zone, zone_from=None
                    )

                # EXIT
                elif prev_zone is not None and new_zone is None:
                    dwell = now - (st.entered_at or now)
                    _push_event(
                        "exit",
                        now,
                        track_id=tid,
                        zone_to=prev_zone,
                        zone_from=prev_zone,
                        dwell_s=dwell,
                    )
                    st.zone = None
                    st.entered_at = None

                # TRANSFER
                else:
                    dwell = now - (st.entered_at or now)
                    _push_event(
                        "transfer",
                        now,
                        track_id=tid,
                        zone_to=new_zone,
                        zone_from=prev_zone,
                        dwell_s=dwell,
                    )
                    st.zone = new_zone
                    st.entered_at = now

            # Attach live dwell preview to the detection for UI
            if st.zone is not None and st.entered_at is not None:
                d["dwell_s"] = round(now - st.entered_at, 2)

        # Finalize tracks not seen this frame (miss TTL)
        for tid, st in list(tracks.items()):
            if tid in seen_ids:
                continue
            if now - st.last_seen > TRACK_MISS_TTL:
                if st.zone is not None:
                    dwell = now - (st.entered_at or st.last_seen)
                    _push_event(
                        "exit",
                        now,
                        track_id=tid,
                        zone_to=st.zone,
                        zone_from=st.zone,
                        dwell_s=dwell,
                    )
                del tracks[tid]

        # Stable occupancy by track state, filtered to the target class (e.g., "person")
        occ_counter = Counter(
            st.zone
            for st in tracks.values()
            if st.zone is not None and st.cls in OCCUPANCY_CLASSES
        )
        occupancy = {k: int(v) for k, v in occ_counter.items()}

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
            ah, aw = annotated.shape[:2]
            latest_metrics = {
                "ts": time.time(),
                "fps": round(ema_fps or inst_fps, 2),
                "img_shape": [ah, aw],
                "counts": dict(counts),
                "detections": detection_list,  # now includes optional dwell_s
                "occupancy": occupancy,  # computed from tracks
                "recent_events": list(events)[:10],  # NEW: small rolling feed
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

zones: List[Zone] = []


@app.on_event("startup")
def _startup():
    global zones
    with zones_lock:  # NEW
        zones = load_zones()
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
            await asyncio.sleep(0.2)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/zones", response_model=list[Zone])
def get_zones():
    with zones_lock:  # NEW
        return list(zones)  # snapshot


@app.put("/zones", response_model=list[Zone])
def put_zones(new_zones: list[Zone]):
    global zones
    with zones_lock:  # NEW
        zones = new_zones
        save_zones(zones)
    return zones

@app.get("/events")
def get_events(limit: int = 50):
    return list(events)[:max(1, min(limit, EVENTS_MAX))]