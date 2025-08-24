from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Source can be "0" (webcam index) or RTSP/HTTP URL
    SOURCE: str = "1"
    IMG_SIZE: int = 640
    CONF: float = 0.25
    STREAM_FPS: float = 20
    JPEG_QUALITY: int = 80
    HALF: bool = True
    CAMERA_ID: str = "cam-1"
    YOLO_WEIGHTS: str = "yolov8n.pt"

    # Security / features
    ALLOW_ORIGINS: str = "http://localhost:5173"
    TRACK_MISS_TTL: float = 1.5
    EVENTS_MAX: int = 200
    OCCUPANCY_CLASSES: str = "person"  # CSV → set()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def source_is_index(self) -> bool:
        return self.SOURCE.isdigit()

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.ALLOW_ORIGINS.split(",") if o.strip()]

    @property
    def occupancy_classes_set(self) -> set[str]:
        return {s.strip() for s in self.OCCUPANCY_CLASSES.split(",") if s.strip()}
