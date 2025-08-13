from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # keep SOURCE as string; we’ll parse to int if it’s all digits
    SOURCE: str = "1"
    IMG_SIZE: int = 640
    CONF: float = 0.25
    STREAM_FPS: float = 20.0
    JPEG_QUALITY: int = 80
    HALF: bool = True
    CAMERA_ID: str = "cam-1"
    
    # tell pydantic-settings to read .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    @property
    def source_is_index(self) -> bool:
        return self.SOURCE.isdigit()