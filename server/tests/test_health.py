"""
헬스체크 API 테스트
"""


def test_health_check_success(client):
    """
    /health 엔드포인트가 정상 응답을 반환하는지 확인
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_health_check_response_format(client):
    """
    헬스체크 응답의 형식이 올바른지 확인
    """
    response = client.get("/health")
    data = response.json()
    
    # 필수 필드 확인
    required_fields = ["status", "timestamp", "version"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
