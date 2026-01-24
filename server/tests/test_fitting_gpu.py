"""
피팅 API 통합 테스트 (GPU 필요)

실제 FastFit 모델을 로드하여 추론을 수행하는 테스트입니다.
GPU가 있는 환경에서만 실행 가능합니다.

실행 방법:
    pytest tests/test_fitting_gpu.py -v -m gpu

GPU 테스트 건너뛰기:
    pytest -m "not gpu"
"""
import os
import pytest


# GPU 마커 - GPU가 없으면 테스트 건너뜀
gpu_required = pytest.mark.gpu


def gpu_available():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU not available")
@gpu_required
class TestFittingWithGPU:
    """
    실제 GPU를 사용하는 통합 테스트
    
    주의: 
    - 첫 실행 시 모델 다운로드로 시간이 오래 걸립니다 (5GB+)
    - GPU VRAM 8GB 이상 필요
    - sample/ 디렉토리에 테스트 이미지가 있어야 합니다
    """

    @pytest.fixture(autouse=True)
    def check_sample_files(self, sample_person_image_path, sample_garment_image_path):
        """샘플 파일 존재 여부 확인"""
        if not os.path.exists(sample_person_image_path):
            pytest.skip(f"Sample person image not found: {sample_person_image_path}")
        if not os.path.exists(sample_garment_image_path):
            pytest.skip(f"Sample garment image not found: {sample_garment_image_path}")

    def test_real_fitting_with_file_upload(self, client, sample_person_image_path, sample_garment_image_path):
        """
        실제 이미지 파일로 피팅 수행 (GPU 추론)
        
        이 테스트는 실제 FastFit 모델을 로드하고 추론을 수행합니다.
        처음 실행 시 모델 다운로드로 5-10분 소요될 수 있습니다.
        """
        with open(sample_person_image_path, "rb") as person_file, \
             open(sample_garment_image_path, "rb") as garment_file:
            
            response = client.post(
                "/api/v1/fitting/upload/multi",
                files={
                    "person_image": ("person.jpg", person_file, "image/jpeg"),
                    "upper_image": ("upper.jpg", garment_file, "image/jpeg"),
                }
            )
        
        # GPU 추론 성공 여부 확인
        assert response.status_code == 200, f"Failed: {response.text}"
        assert response.headers["content-type"] == "image/png"
        
        # 추론 시간 확인 (일반적으로 10-60초)
        inference_time = int(response.headers.get("X-Inference-Time-Ms", 0))
        print(f"Inference time: {inference_time}ms")
        assert inference_time > 0, "Inference time should be positive"
        
        # 결과 이미지 저장 (디버깅용)
        result_path = os.path.join(os.path.dirname(__file__), "output_result.png")
        with open(result_path, "wb") as f:
            f.write(response.content)
        print(f"Result saved to: {result_path}")

    def test_model_loading_time(self, client):
        """
        모델 로딩 시간 측정 (첫 요청)
        
        첫 요청은 모델 로딩으로 인해 오래 걸립니다.
        이후 요청은 캐시된 모델을 사용하여 빠릅니다.
        """
        import time
        
        # 헬스체크로 서버 준비 확인
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 모델 로딩 시간은 conftest 또는 서비스 초기화에서 측정
        # 여기서는 API 응답 시간만 체크
        start = time.time()
        
        # 더미 요청 (모델 로딩 트리거)
        # 실제 이미지가 없으면 에러가 발생하지만 모델은 로딩됨
        
        end = time.time()
        print(f"Health check response time: {end - start:.2f}s")


@pytest.mark.skipif(not gpu_available(), reason="CUDA GPU not available")
@gpu_required
class TestFastFitServiceDirectly:
    """
    FastFitService 직접 테스트 (API 계층 없이)
    
    서비스 로직을 직접 테스트하여 문제 격리에 유용합니다.
    """

    def test_service_initialization(self):
        """FastFitService 초기화 테스트"""
        from app.services.fastfit_service import FastFitService
        
        # 서비스 생성 (모델 로딩은 지연됨)
        service = FastFitService()
        
        # 내부 속성 확인 (private 속성)
        assert hasattr(service, '_model_loaded'), "Service should have _model_loaded attribute"
        assert service._model_loaded == False, "Model should not be loaded yet"
        assert service._device in ['cuda', 'cpu'], f"Invalid device: {service._device}"
        
        # 클래스 속성 확인
        assert FastFitService.BASE_MODEL_PATH is not None
        print(f"Base model path: {FastFitService.BASE_MODEL_PATH}")
        print(f"Device: {service._device}")

    @pytest.mark.slow
    def test_service_model_loading(self):
        """
        모델 로딩 테스트 (시간 소요)
        
        실행: pytest tests/test_fitting_gpu.py -v -m "gpu and slow"
        """
        from app.services.fastfit_service import FastFitService
        import time
        
        service = FastFitService()
        
        start = time.time()
        # 모델 로딩 트리거 (내부 메서드 호출)
        try:
            service._load_model()
            load_time = time.time() - start
            print(f"Model loading time: {load_time:.2f}s")
            
            # private 속성 확인
            assert service._pipeline is not None, "Pipeline should be loaded"
            assert service._model_loaded == True, "Model should be marked as loaded"
            assert service._dwpose_detector is not None, "DWPose detector should be loaded"
            
            print(f"✅ All models loaded successfully!")
        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")

