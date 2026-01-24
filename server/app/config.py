"""
Application Configuration using Pydantic Settings
"""
import os
from typing import List
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # Server
    app_name: str = "FastFit AI Server"
    debug: bool = False

    # CORS
    cors_origins: List[str] = ["*"]

    # Model Paths
    fastfit_model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "FastFit"
    )

    # Inference Settings
    device: str = "cuda"  # cuda or cpu
    mixed_precision: str = "bf16"  # bf16, fp16, or no
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    default_seed: int = 42
    
    # Reference Image Settings
    default_ref_height: int = 512  # 512, 768, or 1024

    # Image Settings
    person_width: int = 768
    person_height: int = 1024
    
    # Performance
    enable_tf32: bool = True  # A100/RTX 30xx 이상에서 성능 향상

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )



# 싱글톤 설정 인스턴스
settings = Settings()

