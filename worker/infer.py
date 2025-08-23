import time
from typing import cast
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from core import state
from core.config import Settings
from services.detector import Detector
from services.tracker import TracksManager

def _open_capture(settings: Settings):
    if settings.source_is_index:
        return cv2.VideoCapture(int(settings.SOURCE), cv2.CAP_DSHOW)
    return cv2.VideoCapture(settings.SOURCE)

def run_inference(settings: Settings):
    device = 0 if torch.cuda.is_available() else "cpu"
    half   = bool(settings.HALF) and torch.cuda.is_available()
    det    = Detector(settings.YOLO_WEIGHTS, device, half, settings.IMG_SIZE, settings.CONF)
    mgr    = TracksManager(settings)

    cap = _open_capture(settings)
    if not cap.isOpened():
        while not cap.isOpened() and not state.stop_flag:
            time.sleep(1.0)
            cap.open(int(settings.SOURCE) if settings.source_is_index else settings.SOURCE)

    t_last = time.perf_counter()
    ema_fps = None

    while not state.stop_flag:
        ok, frame = cap.read()
        if not ok:
            cap.release(); time.sleep(0.5); cap = _open_capture(settings); continue
            
        frame_np: NDArray[np.uint8] = cast(NDArray[np.uint8], frame)
        dets, counts, annotated = det.track(frame_np)
        h, w = frame_np.shape[:2]

        # thread-safe zone snapshot
        with state.zones_lock:
            zones_snapshot = tuple(state.zones)

        occupancy = mgr.update(dets, zones_snapshot, (w, h))

        ok_jpg, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
        if not ok_jpg:
            continue
        jpg = buf.tobytes()

        now = time.perf_counter()
        inst_fps = 1.0 / max(now - t_last, 1e-6)
        t_last = now
        ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

        with state.frame_lock:
            state.latest_jpeg = jpg
            ah, aw = annotated.shape[:2]
            state.latest_metrics = {
                "ts": time.time(),
                "fps": round(ema_fps or inst_fps, 2),
                "img_shape": [ah, aw],
                "counts": counts,
                "detections": dets,             # includes zone + dwell_s
                "occupancy": occupancy,
                "recent_events": list(state.events)[:10],
            }

    cap.release()
