"""
피팅 API 테스트 (Mock 기반)

GPU 없이 CI/CD 환경에서 실행 가능한 단위 테스트입니다.
FastFitService를 Mock으로 대체하여 API 계층만 테스트합니다.
"""
from unittest.mock import patch, AsyncMock


class TestFittingUploadValidation:
    """파일 업로드 API 입력 검증 테스트"""

    def test_upload_multi_without_garments(self, client, dummy_image_bytes):
        """
        의류 이미지 없이 요청하면 400 에러 반환
        """
        response = client.post(
            "/api/v1/fitting/upload/multi",
            files={
                "person_image": ("person.png", dummy_image_bytes, "image/png"),
            }
        )
        
        assert response.status_code == 400
        assert "최소 하나의 의류" in response.json()["detail"]

    def test_upload_multi_dress_with_upper(self, client, dummy_image_bytes):
        """
        드레스와 상의를 동시에 요청하면 400 에러 반환
        (비즈니스 로직: dress는 top/bottom과 함께 사용 불가)
        """
        response = client.post(
            "/api/v1/fitting/upload/multi",
            files={
                "person_image": ("person.png", dummy_image_bytes, "image/png"),
                "dress_image": ("dress.png", dummy_image_bytes, "image/png"),
                "upper_image": ("top.png", dummy_image_bytes, "image/png"),
            }
        )
        
        assert response.status_code == 400
        assert "드레스" in response.json()["detail"]


class TestFittingUploadWithMock:
    """Mock을 사용한 피팅 API 테스트 (GPU 불필요)"""

    @patch("app.routers.fitting.get_service")
    def test_upload_multi_success_with_mock(self, mock_get_service, client, dummy_image_bytes):
        """
        정상적인 피팅 요청 시 이미지 바이너리 반환 (Mock)
        """
        import base64
        
        # Mock 설정: 서비스가 Base64 이미지를 반환하도록 설정
        mock_service = AsyncMock()
        mock_service.run_multi_fitting.return_value = base64.b64encode(dummy_image_bytes).decode()
        mock_get_service.return_value = mock_service
        
        response = client.post(
            "/api/v1/fitting/upload/multi",
            files={
                "person_image": ("person.png", dummy_image_bytes, "image/png"),
                "upper_image": ("top.png", dummy_image_bytes, "image/png"),
                "lower_image": ("bottom.png", dummy_image_bytes, "image/png"),
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "X-Inference-Time-Ms" in response.headers
        assert response.headers["X-Garment-Count"] == "2"

    @patch("app.routers.fitting.get_service")
    def test_upload_multi_with_all_garments(self, mock_get_service, client, dummy_image_bytes):
        """
        모든 의류 카테고리(상의+하의+신발+가방) 동시 피팅 (Mock)
        """
        import base64
        
        mock_service = AsyncMock()
        mock_service.run_multi_fitting.return_value = base64.b64encode(dummy_image_bytes).decode()
        mock_get_service.return_value = mock_service
        
        response = client.post(
            "/api/v1/fitting/upload/multi",
            files={
                "person_image": ("person.png", dummy_image_bytes, "image/png"),
                "upper_image": ("top.png", dummy_image_bytes, "image/png"),
                "lower_image": ("bottom.png", dummy_image_bytes, "image/png"),
                "shoe_image": ("shoes.png", dummy_image_bytes, "image/png"),
                "bag_image": ("bag.png", dummy_image_bytes, "image/png"),
            }
        )
        
        assert response.status_code == 200
        assert response.headers["X-Garment-Count"] == "4"


class TestFittingJsonValidation:
    """JSON API 입력 검증 테스트"""

    def test_multi_json_missing_person_image(self, client):
        """
        person_image 없이 요청하면 422 에러 반환
        """
        response = client.post(
            "/api/v1/fitting/multi",
            json={
                "garments": [
                    {"category": "top", "image": "base64_string"}
                ]
            }
        )
        
        assert response.status_code == 422  # Validation Error

    def test_multi_json_invalid_category(self, client, dummy_image_base64):
        """
        유효하지 않은 카테고리로 요청하면 422 에러 반환
        """
        response = client.post(
            "/api/v1/fitting/multi",
            json={
                "person_image": dummy_image_base64,
                "garments": [
                    {"category": "invalid_category", "image": dummy_image_base64}
                ]
            }
        )
        
        assert response.status_code == 422
