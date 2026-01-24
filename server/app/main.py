"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, fitting

# FastAPI 앱 생성 (Swagger 메타데이터 설정)
app = FastAPI(
    title="FastFit AI Inference Server",
    description="""
## FastFit 기반 가상 피팅 AI 추론 서버

이 API는 사용자 이미지와 의류 이미지를 받아 가상 착용(Virtual Try-On) 결과를 반환합니다.

### 주요 기능
- **단일 피팅**: 한 개의 의류 아이템만 착용
- **다중 피팅**: 여러 의류 아이템 동시 착용 (상의 + 하의 + 신발 등)

### 지원 카테고리
- `top`: 상의
- `bottom`: 하의  
- `dress`: 원피스/드레스
- `shoes`: 신발
- `bag`: 가방
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Fukubukuro Team",
        "email": "contact@fukubukuro.com",
    },
    license_info={
        "name": "Non-Commercial License",
        "url": "https://github.com/Zheng-Chong/FastFit/blob/main/LICENSE.md",
    },
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(fitting.router, prefix="/api/v1/fitting", tags=["Fitting"])


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩 (선택적)"""
    # TODO: 필요시 모델 사전 로딩
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    pass
