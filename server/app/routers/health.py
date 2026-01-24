"""
Health Check Router
"""
from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict
from datetime import datetime


router = APIRouter()


class HealthResponse(BaseModel):
    """헬스체크 응답 스키마"""
    status: str
    message: str
    timestamp: str
    version: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "message": "서버가 정상 작동 중입니다.",
                "timestamp": "2026-01-24T21:00:00+09:00",
                "version": "1.0.0"
            }
        }
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="서버 상태 확인",
    description="서버의 현재 상태와 버전 정보를 반환합니다.",
    responses={
        200: {
            "description": "서버 정상 작동",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "message": "서버가 정상 작동 중입니다.",
                        "timestamp": "2026-01-24T21:00:00+09:00",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """
    서버 헬스체크 API
    
    - 서버 상태 확인
    - 타임스탬프 및 버전 정보 반환
    """
    return HealthResponse(
        status="healthy",
        message="서버가 정상 작동 중입니다.",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )
