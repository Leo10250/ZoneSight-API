from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from ultralytics import YOLO


def _tolist(x: Any) -> list:
    """Convert torch.Tensor/np.ndarray/None to a (nested) Python list."""
    if x is None:
        return []
    # torch.Tensor -> detach -> cpu -> numpy
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    # ndarray or tensor -> list
    return x.tolist()


class Detector:
    """YOLOv8 + ByteTrack wrapper that returns (detections, counts, annotated_frame)."""
    def __init__(self, weights: str, device: str | int, half: bool, imgsz: int, conf: float):
        self.model = YOLO(weights)
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names

    def track(self, frame: NDArray[np.uint8]) -> Tuple[List[Dict[str, Any]], Dict[str, int], NDArray[np.uint8]]:
        res = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.5,
            device=self.device,
            half=self.half,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )[0]

        boxes = res.boxes

        # Safely convert to Python lists (works for torch/np/None)
        ids_list  = _tolist(getattr(boxes, "id", None))
        cls_list  = _tolist(getattr(boxes, "cls", None))
        conf_list = _tolist(getattr(boxes, "conf", None))
        xyxy_list = _tolist(getattr(boxes, "xyxy", None))

        # Normalize types
        ids:  List[int] = [int(i) for i in ids_list]
        cls:  List[int] = [int(c) for c in cls_list]
        conf: List[float] = [float(p) for p in conf_list]
        xyxy: List[List[float]] = [list(map(float, row)) for row in xyxy_list]

        labels  = [self.names[int(i)] for i in cls]
        counts  = Counter(labels)

        dets = [
            {
                "tracking_id": i,
                "cls": self.names[int(c)],
                "conf": p,
                "xyxy": b,
            }
            for i, c, p, b in zip(ids, cls, conf, xyxy)
        ]

        annotated: NDArray[np.uint8] = res.plot()  # BGR ndarray
        return dets, dict(counts), annotated
