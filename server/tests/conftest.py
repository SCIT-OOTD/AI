"""
Pytest Fixtures - 공유 테스트 설정
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """
    테스트용 FastAPI 클라이언트
    
    실제 서버를 시작하지 않고 API 호출을 시뮬레이션합니다.
    """
    return TestClient(app)


@pytest.fixture
def dummy_image_bytes():
    """
    테스트용 1x1 PNG 이미지 (바이너리)
    
    실제 이미지 파일 없이 업로드 테스트가 가능합니다.
    """
    import base64
    # 1x1 투명 PNG 이미지
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    return base64.b64decode(png_base64)


@pytest.fixture
def dummy_image_base64():
    """테스트용 1x1 PNG 이미지 (Base64 문자열)"""
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def sample_person_image_path():
    """샘플 사람 이미지 경로 (실제 파일이 있어야 함)"""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "sample", "person1.jpg")
    return os.path.abspath(path)


@pytest.fixture
def sample_garment_image_path():
    """샘플 의류 이미지 경로 (실제 파일이 있어야 함)"""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "sample", "clothes", "upper1.jpg")
    return os.path.abspath(path)
