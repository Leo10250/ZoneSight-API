from __future__ import annotations
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from models.zone import Zone

# Frame & metrics shared across endpoints
latest_jpeg: Optional[bytes] = None
latest_metrics: Dict[str, Any] = {"fps": 0.0, "counts": {}, "ts": 0.0, "img_shape": [0, 0]}
frame_lock = threading.Lock()

# Zones are edited via API; read by worker
zones_lock = threading.RLock()
zones: List[Zone] = []  # forward ref to models.zone.Zone

# Live tracking state (owned by worker)
@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    zone: str | None = None
    entered_at: float | None = None
    cls: str | None = None

tracks: dict[int, TrackState] = {}
events: Deque[dict[str, Any]] = deque(maxlen=200)

# Worker control
stop_flag = False
