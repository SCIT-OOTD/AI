# FastFit AI Inference Server
# CUDA 기반 GPU 이미지 사용

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# easy-dwpose 설치 (의존성 충돌 방지)
RUN pip3 install --no-cache-dir easy-dwpose --no-dependencies

# 애플리케이션 코드 복사
COPY app/ ./app/

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda
ENV FASTFIT_MODEL_PATH=/models/FastFit

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
