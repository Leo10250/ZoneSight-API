from __future__ import annotations
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple
from core import state
from core.config import Settings
from models.track_state import TrackState
from models.zone import Zone
from services.geometry import point_in_poly

def _norm_center(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) * 0.5) / max(w, 1)
    cy = ((y1 + y2) * 0.5) / max(h, 1)
    return cx, cy

class TracksManager:
    """Maintains TrackState, emits entry/exit/transfer events, and computes occupancy."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def update(self, detections: List[Dict[str, Any]], zones_snapshot: Iterable[Zone], frame_wh: Tuple[int,int]) -> Dict[str,int]:
        w, h = frame_wh
        zones_list = list(zones_snapshot)

        # Assign zone to each detection (centroid ∈ polygon)
        for d in detections:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = _norm_center(x1, y1, x2, y2, w, h)
            zhit = None
            for z in zones_list:
                if point_in_poly(cx, cy, z.points):
                    zhit = z.id
                    break
            d["zone"] = zhit

        now = time.perf_counter()
        seen: set[int] = set()
        occ_classes = self.settings.occupancy_classes_set

        for d in detections:
            tid = d.get("tracking_id")
            if tid is None:
                continue
            if d.get("cls") not in occ_classes:
                continue  # only manage human-like tracks

            seen.add(tid)
            st = state.tracks.get(tid)
            if st is None:
                st = TrackState(id=tid, first_seen=now, last_seen=now)
                state.tracks[tid] = st
            else:
                st.last_seen = now

            st.cls = d.get("cls") or st.cls
            new_zone = d.get("zone")
            prev_zone = st.zone

            if new_zone != prev_zone:
                if prev_zone is None and new_zone is not None:
                    st.zone = new_zone
                    st.entered_at = now
                    state.events.appendleft({"ts": time.time(), "type": "entry", "track_id": tid, "zone": new_zone, "from": None, "dwell_s": None})
                elif prev_zone is not None and new_zone is None:
                    dwell = now - (st.entered_at or now)
                    state.events.appendleft({"ts": time.time(), "type": "exit", "track_id": tid, "zone": prev_zone, "from": prev_zone, "dwell_s": round(dwell, 2)})
                    st.zone = None
                    st.entered_at = None
                else:
                    dwell = now - (st.entered_at or now)
                    state.events.appendleft({"ts": time.time(), "type": "transfer", "track_id": tid, "zone": new_zone, "from": prev_zone, "dwell_s": round(dwell, 2)})
                    st.zone = new_zone
                    st.entered_at = now

            if st.zone is not None and st.entered_at is not None:
                d["dwell_s"] = round(now - st.entered_at, 2)

        # Expire unseen tracks
        for tid, st in list(state.tracks.items()):
            if tid in seen:
                continue
            if now - st.last_seen > self.settings.TRACK_MISS_TTL:
                if st.zone is not None:
                    dwell = now - (st.entered_at or st.last_seen)
                    state.events.appendleft({"ts": time.time(), "type": "exit", "track_id": tid, "zone": st.zone, "from": st.zone, "dwell_s": round(dwell, 2)})
                del state.tracks[tid]

        # Occupancy (humans only, by track state)
        occ_counter = Counter(
            st.zone for st in state.tracks.values() if st.zone is not None and (st.cls in occ_classes)
        )
        return {k: int(v) for k, v in occ_counter.items()}
