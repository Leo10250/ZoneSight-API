import torch
from fastapi import APIRouter
from core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    s = Settings()
    device = 0 if torch.cuda.is_available() else "cpu"
    half = bool(s.USE_HALF_PRECISION) and torch.cuda.is_available()
    return {
        "ok": True,
        "device": str(device),
        "source": s.SOURCE,
        "img_size": s.IMG_SIZE,
        "conf": s.CONF,
        "streaming_fps": s.STREAM_FPS,
        "jpeg_quality": s.JPEG_QUALITY,
        "half": half,
    }
