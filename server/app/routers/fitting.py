"""
Fitting API Router
"""
import base64
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import Response

from app.schemas.fitting import (
    SingleFittingRequest,
    MultiFittingRequest,
    FittingResponse,
    ErrorResponse,
    GarmentCategory,
    GarmentItem,
)
from app.services.fastfit_service import FastFitService


router = APIRouter()

# FastFit 서비스 인스턴스 (지연 로딩)
_service: FastFitService | None = None


def get_service() -> FastFitService:
    """FastFit 서비스 싱글톤 반환"""
    global _service
    if _service is None:
        _service = FastFitService()
    return _service


async def file_to_base64(file: UploadFile) -> str:
    """UploadFile을 Base64 문자열로 변환"""
    content = await file.read()
    return base64.b64encode(content).decode("utf-8")


@router.post(
    "/single",
    response_model=FittingResponse,
    status_code=status.HTTP_200_OK,
    summary="단일 아이템 피팅 (JSON)",
    description="""
단일 의류 아이템을 사용자 이미지에 가상 착용합니다.

### 요청 파라미터
- **person_image**: 사용자 전신 이미지 (Base64 인코딩)
- **garment_image**: 착용할 의류 이미지 (Base64 인코딩)
- **category**: 의류 카테고리 (top, bottom, dress, shoes, bag)

### 주의사항
- 이미지는 반드시 Base64 인코딩되어야 합니다.
- 권장 이미지 크기: 768x1024 (사람), 512x512 (의류)
    """,
    responses={
        200: {
            "description": "피팅 성공",
            "model": FittingResponse
        },
        400: {
            "description": "잘못된 요청 (이미지 형식 오류 등)",
            "model": ErrorResponse
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse
        }
    }
)
async def single_fitting(request: SingleFittingRequest) -> FittingResponse:
    """
    단일 아이템 가상 피팅 API
    
    하나의 의류 아이템을 사용자 이미지에 착용시킵니다.
    """
    try:
        start_time = time.time()
        service = get_service()
        
        result_image = await service.run_single_fitting(
            person_image=request.person_image,
            garment_image=request.garment_image,
            category=request.category,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        return FittingResponse(
            status=200,
            message="피팅 성공",
            data={
                "result_image": result_image,
                "inference_time_ms": inference_time_ms
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"피팅 처리 중 오류 발생: {str(e)}"
        )


@router.post(
    "/multi",
    response_model=FittingResponse,
    status_code=status.HTTP_200_OK,
    summary="다중 아이템 피팅 (JSON)",
    description="""
여러 의류 아이템을 동시에 사용자 이미지에 가상 착용합니다.

### 요청 파라미터
- **person_image**: 사용자 전신 이미지 (Base64 인코딩)
- **garments**: 착용할 의류 목록 (최대 5개)

### 지원되는 조합
- 상의(top) + 하의(bottom)
- 상의(top) + 하의(bottom) + 신발(shoes)
- 원피스(dress) + 신발(shoes) + 가방(bag)

### 주의사항
- 원피스(dress)는 상의(top)/하의(bottom)와 함께 사용할 수 없습니다.
    """,
    responses={
        200: {
            "description": "피팅 성공",
            "model": FittingResponse
        },
        400: {
            "description": "잘못된 요청",
            "model": ErrorResponse
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse
        }
    }
)
async def multi_fitting(request: MultiFittingRequest) -> FittingResponse:
    """
    다중 아이템 가상 피팅 API
    
    여러 의류 아이템을 동시에 사용자 이미지에 착용시킵니다.
    FastFit의 Multi-Reference 기능을 활용합니다.
    """
    try:
        # 카테고리 조합 검증
        categories = [g.category for g in request.garments]
        if "dress" in categories and ("top" in categories or "bottom" in categories):
            raise ValueError("원피스(dress)는 상의(top)/하의(bottom)와 함께 사용할 수 없습니다.")
        
        start_time = time.time()
        service = get_service()
        
        result_image = await service.run_multi_fitting(
            person_image=request.person_image,
            garments=request.garments,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        return FittingResponse(
            status=200,
            message="피팅 성공",
            data={
                "result_image": result_image,
                "inference_time_ms": inference_time_ms,
                "garment_count": len(request.garments)
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"피팅 처리 중 오류 발생: {str(e)}"
        )


# ============================================================
# 파일 업로드 엔드포인트 (Spring Boot 연동용)
# ============================================================

@router.post(
    "/upload/multi",
    summary="다중 아이템 피팅 (파일 업로드)",
    description="""
**Spring Boot 백엔드 연동용** - Multipart/form-data로 이미지 파일 업로드

### 요청 파라미터 (form-data)
- **person_image**: 사용자 전신 이미지 파일 (필수)
- **upper_image**: 상의 이미지 파일 (선택)
- **lower_image**: 하의 이미지 파일 (선택)
- **outer_image**: 외투 이미지 파일 (선택)
- **dress_image**: 드레스 이미지 파일 (선택)
- **shoe_image**: 신발 이미지 파일 (선택)
- **bag_image**: 가방 이미지 파일 (선택)

### 결과
- 피팅된 이미지를 **PNG 바이너리**로 반환합니다.
    """,
    responses={
        200: {
            "description": "피팅 성공",
            "content": {"image/png": {}}
        },
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 내부 오류"}
    }
)
async def multi_fitting_upload(
    person_image: UploadFile = File(..., description="사용자 전신 이미지"),
    upper_image: Optional[UploadFile] = File(None, description="상의 이미지"),
    lower_image: Optional[UploadFile] = File(None, description="하의 이미지"),
    outer_image: Optional[UploadFile] = File(None, description="외투 이미지"),
    dress_image: Optional[UploadFile] = File(None, description="드레스 이미지"),
    shoe_image: Optional[UploadFile] = File(None, description="신발 이미지"),
    bag_image: Optional[UploadFile] = File(None, description="가방 이미지"),
) -> Response:
    """
    파일 업로드 기반 다중 아이템 피팅 API
    
    Spring Boot 등 외부 서버에서 Multipart/form-data로 이미지를 전송할 때 사용합니다.
    결과는 PNG 이미지 바이너리로 반환됩니다.
    """
    try:
        # 의류 검증 1: dress와 upper/lower 동시 사용 불가
        if dress_image and (upper_image or lower_image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="드레스는 상의/하의와 함께 사용할 수 없습니다."
            )
            
        # 의류 검증 2: outer와 dress 동시 사용 불가
        if outer_image and dress_image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="외투는 드레스와 함께 사용할 수 없습니다."
            )
        
        # 최소 하나의 의류 필요
        if not any([upper_image, lower_image, outer_image, dress_image, shoe_image, bag_image]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="최소 하나의 의류 이미지가 필요합니다."
            )
        
        start_time = time.time()
        service = get_service()
        
        # 사람 이미지 Base64 변환
        person_base64 = await file_to_base64(person_image)
        
        # 의류 아이템 목록 구성
        garments = []
        
        if upper_image:
            garments.append(GarmentItem(
                category=GarmentCategory.TOP,
                image=await file_to_base64(upper_image)
            ))
        
        if lower_image:
            garments.append(GarmentItem(
                category=GarmentCategory.BOTTOM,
                image=await file_to_base64(lower_image)
            ))
        
        if outer_image:
            garments.append(GarmentItem(
                category=GarmentCategory.OUTER,
                image=await file_to_base64(outer_image)
            ))
        
        if dress_image:
            garments.append(GarmentItem(
                category=GarmentCategory.DRESS,
                image=await file_to_base64(dress_image)
            ))
        
        if shoe_image:
            garments.append(GarmentItem(
                category=GarmentCategory.SHOES,
                image=await file_to_base64(shoe_image)
            ))
        
        if bag_image:
            garments.append(GarmentItem(
                category=GarmentCategory.BAG,
                image=await file_to_base64(bag_image)
            ))
        
        # 피팅 실행
        result_base64 = await service.run_multi_fitting(
            person_image=person_base64,
            garments=garments,
            num_inference_steps=50,
            guidance_scale=2.5,
            seed=42,
        )
        
        inference_time_ms = int((time.time() - start_time) * 1000)
        print(f"[Fitting] 피팅 완료 - {len(garments)}개 아이템, {inference_time_ms}ms")
        
        # Base64 -> 바이너리 변환하여 이미지 반환
        result_bytes = base64.b64decode(result_base64)
        
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={
                "X-Inference-Time-Ms": str(inference_time_ms),
                "X-Garment-Count": str(len(garments))
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"[Fitting] 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"피팅 처리 중 오류 발생: {str(e)}"
        )

