from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration."""

    scene_path: str = "../assets/hand/reach.xml"
    mj_steps_per_action: int = 4
    distance_weight: float = 10.0
    smoothness_weight: float = 0.5
    velocity_weight: float = 0.1
    energy_weight: float = 0.1
    labelling_function_tol: float = 0.1
