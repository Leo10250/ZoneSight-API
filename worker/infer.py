import time
from typing import cast
import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from core import state
from core.config import Settings
from services.detector import Detector
from services.tracker import Tracker


def _open_capture(settings: Settings):
    if settings.source_is_index:
        return cv2.VideoCapture(int(settings.SOURCE), cv2.CAP_DSHOW)
    return cv2.VideoCapture(settings.SOURCE)


def run_inference(settings: Settings):
    compute_device = 0 if torch.cuda.is_available() else "cpu"
    use_half_precision = bool(settings.USE_HALF_PRECISION) and torch.cuda.is_available()
    yolo_detector = Detector(
        settings.YOLO_WEIGHTS,
        compute_device,
        use_half_precision,
        settings.IMG_SIZE,
        settings.CONF,
    )
    occupancy_tracker = Tracker(settings)

    video_capture_source = _open_capture(settings)
    if not video_capture_source.isOpened():
        while not video_capture_source.isOpened() and not state.stop_flag:
            time.sleep(1.0)
            video_capture_source.open(
                int(settings.SOURCE) if settings.source_is_index else settings.SOURCE
            )

    previous_frame_time = time.perf_counter()
    smoothed_fps = None  # ema_fps

    while not state.stop_flag:
        frame_read_success, frame = video_capture_source.read()
        if not frame_read_success:
            video_capture_source.release()
            time.sleep(0.5)
            video_capture_source = _open_capture(settings)
            continue

        frame_np: NDArray[np.uint8] = cast(NDArray[np.uint8], frame)
        detections, object_counts, annotated_frame = yolo_detector.track(frame_np)
        height, width = frame_np.shape[:2]

        # thread-safe zone snapshot
        with state.zones_lock:
            zones_snapshot = tuple(state.zones)

        occupancy: dict[str, int] = occupancy_tracker.process_detections(
            detections, zones_snapshot, (width, height)
        )

        jpeg_encode_successful, frame_jpeg = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY],
        )
        if not jpeg_encode_successful:
            continue
        frame_jpeg_bytes = frame_jpeg.tobytes()

        current_frame_time = time.perf_counter()
        instantaneous_fps = 1.0 / max(current_frame_time - previous_frame_time, 1e-6)
        previous_frame_time = current_frame_time
        smoothed_fps = (
            instantaneous_fps
            if smoothed_fps is None
            else (0.9 * smoothed_fps + 0.1 * instantaneous_fps)
        )

        with state.frame_lock:
            state.latest_jpeg = frame_jpeg_bytes
            annotated_height, annotated_width = annotated_frame.shape[:2]
            state.latest_metrics = {
                "timestamp": time.time(),
                "fps": round(smoothed_fps or instantaneous_fps, 2),
                "img_shape": [annotated_height, annotated_width],
                "object_counts": object_counts,
                "detections": detections,
                "occupancy": occupancy,
                "recent_events": list(state.events)[:10],
            }

    video_capture_source.release()
