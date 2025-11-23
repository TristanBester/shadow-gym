from pydantic_settings import BaseSettings


class CameraConfig(BaseSettings):
    """Camera configuration."""

    distance: float = 0.046
    azimuth: float = -89.5
    elevation: float = -89.0
    lookat: list[float] = [0.978, 0.950, 0.737]
    width: int = 640
    height: int = 480
