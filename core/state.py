from __future__ import annotations
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List

from models.track_state import TrackState
from models.zone import Zone

# Frame & metrics shared across endpoints
frame_lock = threading.Lock()
latest_jpeg: bytes | None = None
latest_metrics: dict[str, Any] = {
    "fps": 0.0,
    "counts": {},
    "ts": 0.0,
    "img_shape": [0, 0],
}

# Zones are edited via API; read by worker
zones_lock = threading.RLock()
zones: List[Zone] = []

# Live tracking state (owned by worker)
tracks: dict[int, TrackState] = {}
events: Deque[dict[str, Any]] = deque(maxlen=200)

# Worker control
stop_flag = False
