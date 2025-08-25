from __future__ import annotations
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple
from core import state
from core.config import Settings
from models.track_state import TrackState
from models.zone import Zone
from services.geometry import point_inside_polygon


def _get_normalized_bbox_center(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) * 0.5) / max(w, 1)
    cy = ((y1 + y2) * 0.5) / max(h, 1)
    return cx, cy


class Tracker:
    """Maintains TrackState, emits entry/exit/transfer events, and computes occupancy."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        zones_snapshot: Iterable[Zone],
        frame_dimensions: Tuple[int, int],
    ) -> Dict[str, int]:
        frame_width, frame_height = frame_dimensions
        zones_list = list(zones_snapshot)

        # Assign zone to each detection (centroid ∈ polygon)
        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            cx, cy = _get_normalized_bbox_center(
                x1, y1, x2, y2, frame_width, frame_height
            )
            matched_zone_id = None
            # TODO: Handle edge/partial overlap and multi-zone hits
            for zone in zones_list:
                # checks if the center of the object is within the zone
                if point_inside_polygon(cx, cy, zone.points):
                    matched_zone_id = zone.id
                    break
            detection["zone"] = matched_zone_id

        now = time.perf_counter()
        seen: set[int] = set()
        occupancy_classes = self.settings.occupancy_classes_set

        for detection in detections:
            tracking_id = detection.get("tracking_id")
            if tracking_id is None:
                continue
            if detection.get("cls") not in occupancy_classes:
                continue  # only manage human-like tracks

            seen.add(tracking_id)
            track_state = state.tracks.get(tracking_id)
            if track_state is None:
                track_state = TrackState(id=tracking_id, first_seen=now, last_seen=now)
                state.tracks[tracking_id] = track_state
            else:
                track_state.last_seen = now

            track_state.cls = detection.get("cls") or track_state.cls
            new_zone = detection.get("zone")
            prev_zone = track_state.zone

            if new_zone != prev_zone:
                if prev_zone is None and new_zone is not None:
                    track_state.zone = new_zone
                    track_state.entered_at = now
                    state.events.appendleft(
                        {
                            "timestamp": time.time(),
                            "type": "entry",
                            "track_id": tracking_id,
                            "zone": new_zone,
                            "from": None,
                            "dwell_s": None,
                        }
                    )
                elif prev_zone is not None and new_zone is None:
                    dwell = now - (track_state.entered_at or now)
                    state.events.appendleft(
                        {
                            "timestamp": time.time(),
                            "type": "exit",
                            "track_id": tracking_id,
                            "zone": prev_zone,
                            "from": prev_zone,
                            "dwell_s": round(dwell, 2),
                        }
                    )
                    track_state.zone = None
                    track_state.entered_at = None
                else:
                    dwell = now - (track_state.entered_at or now)
                    state.events.appendleft(
                        {
                            "timestamp": time.time(),
                            "type": "transfer",
                            "track_id": tracking_id,
                            "zone": new_zone,
                            "from": prev_zone,
                            "dwell_s": round(dwell, 2),
                        }
                    )
                    track_state.zone = new_zone
                    track_state.entered_at = now

            if track_state.zone is not None and track_state.entered_at is not None:
                detection["dwell_s"] = round(now - track_state.entered_at, 2)

        # Expire unseen tracks
        for tracking_id, track_state in list(state.tracks.items()):
            if tracking_id in seen:
                continue
            if now - track_state.last_seen > self.settings.TRACK_EXPIRATION_TIME:
                if track_state.zone is not None:
                    dwell = now - (track_state.entered_at or track_state.last_seen)
                    state.events.appendleft(
                        {
                            "timestamp": time.time(),
                            "type": "exit",
                            "track_id": tracking_id,
                            "zone": track_state.zone,
                            "from": track_state.zone,
                            "dwell_s": round(dwell, 2),
                        }
                    )
                del state.tracks[tracking_id]

        # Occupancy (humans only, by track state)
        occupancy_counter = Counter(
            st.zone
            for st in state.tracks.values()
            if st.zone is not None and (st.cls in occupancy_classes)
        )
        return {k: int(v) for k, v in occupancy_counter.items()}
