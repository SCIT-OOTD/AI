"""
Pydantic Schemas for Fitting API
"""
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class GarmentCategory(str, Enum):
    """의류 카테고리"""
    TOP = "top"
    BOTTOM = "bottom"
    OUTER = "outer"
    DRESS = "dress"
    SHOES = "shoes"
    BAG = "bag"


class GarmentItem(BaseModel):
    """단일 의류 아이템"""
    category: GarmentCategory = Field(
        ...,
        description="의류 카테고리 (top, bottom, outer, dress, shoes, bag)"
    )
    image: str = Field(
        ...,
        description="의류 이미지 (Base64 인코딩)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "top",
                "image": "base64_encoded_image_string..."
            }
        }
    )


class SingleFittingRequest(BaseModel):
    """단일 아이템 피팅 요청"""
    person_image: str = Field(
        ...,
        description="사용자 전신 이미지 (Base64 인코딩)"
    )
    garment_image: str = Field(
        ...,
        description="착용할 의류 이미지 (Base64 인코딩)"
    )
    category: GarmentCategory = Field(
        ...,
        description="의류 카테고리"
    )
    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Diffusion 스텝 수 (1-100)"
    )
    guidance_scale: float = Field(
        default=2.5,
        ge=1.0,
        le=10.0,
        description="Guidance scale (1.0-10.0)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="랜덤 시드 (재현성 확보용)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_image": "base64_encoded_person_image...",
                "garment_image": "base64_encoded_garment_image...",
                "category": "top",
                "num_inference_steps": 50,
                "guidance_scale": 2.5,
                "seed": 42
            }
        }
    )


class MultiFittingRequest(BaseModel):
    """다중 아이템 피팅 요청"""
    person_image: str = Field(
        ...,
        description="사용자 전신 이미지 (Base64 인코딩)"
    )
    garments: List[GarmentItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="착용할 의류 목록 (최대 5개)"
    )
    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Diffusion 스텝 수 (1-100)"
    )
    guidance_scale: float = Field(
        default=2.5,
        ge=1.0,
        le=10.0,
        description="Guidance scale (1.0-10.0)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="랜덤 시드 (재현성 확보용)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_image": "base64_encoded_person_image...",
                "garments": [
                    {"category": "top", "image": "base64_encoded_top_image..."},
                    {"category": "bottom", "image": "base64_encoded_bottom_image..."}
                ],
                "num_inference_steps": 50,
                "guidance_scale": 2.5,
                "seed": 42
            }
        }
    )


class FittingResponse(BaseModel):
    """피팅 결과 응답"""
    status: int = Field(
        ...,
        description="HTTP 상태 코드"
    )
    message: str = Field(
        ...,
        description="응답 메시지"
    )
    data: Optional[dict] = Field(
        default=None,
        description="결과 데이터"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "피팅 성공",
                "data": {
                    "result_image": "base64_encoded_result_image...",
                    "inference_time_ms": 3200
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """에러 응답"""
    status: int = Field(..., description="HTTP 상태 코드")
    message: str = Field(..., description="에러 메시지")
    data: None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 400,
                "message": "잘못된 이미지 형식입니다.",
                "data": None
            }
        }
    )
