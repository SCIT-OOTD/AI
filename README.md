# FastFit AI Inference Server

FastFit 기반의 가상 피팅(Virtual Try-On) AI 추론 서버입니다.

> **Note**: 이 서버는 로컬 GPU에서 직접 추론을 수행합니다.

## 📁 프로젝트 구조

```
AI/
├── server/                 # FastAPI 추론 서버
│   ├── app/
│   │   ├── main.py         # FastAPI 진입점 (Swagger 설정)
│   │   ├── config.py       # 환경설정
│   │   ├── routers/
│   │   │   ├── health.py   # 헬스체크 API
│   │   │   └── fitting.py  # 피팅 API
│   │   ├── services/
│   │   │   └── fastfit_service.py  # FastFit 추론 서비스
│   │   ├── schemas/
│   │   │   └── fitting.py  # Pydantic 스키마
│   │   └── utils/
│   │       └── image.py    # 이미지 유틸리티
│   ├── requirements.txt
│   └── Dockerfile
├── models/
│   └── FastFit/            # FastFit 모델 (자동 다운로드)
└── sample/                 # 샘플 이미지
```

## 🚀 빠른 시작

### 1. 사전 요구사항

- **Python 3.10+**
- **CUDA 지원 GPU** (VRAM 8GB 이상 권장)
- **PyTorch with CUDA**

### 2. 가상환경 생성 및 활성화 (Anaconda)

```bash
conda create -n fastfit python=3.10
conda activate fastfit
```

### 3. PyTorch with CUDA 설치

```bash
# CUDA 13.0 기준 (본인 CUDA 버전에 맞게 선택: cu121, cu124 등)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 4. 의존성 설치

```bash
cd AI/server
pip install -r requirements.txt
pip install easy-dwpose --no-dependencies
```

### 5. 환경 변수 설정

```bash
cp .env.example .env
# 필요시 .env 파일 수정 (DEVICE, CORS 등)
```

### 6. 서버 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. API 문서 접근

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/api/v1/fitting/single` | 단일 아이템 피팅 |
| `POST` | `/api/v1/fitting/multi` | 다중 아이템 피팅 (최대 5개) |

## ⚙️ 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `FASTFIT_MODEL_PATH` | `../models/FastFit` | FastFit 모델 경로 |
| `DEVICE` | `cuda` | 추론 디바이스 (cuda/cpu) |
| `MIXED_PRECISION` | `bf16` | 혼합 정밀도 (bf16/fp16/no) |
| `NUM_INFERENCE_STEPS` | `50` | Diffusion 스텝 수 |
| `GUIDANCE_SCALE` | `2.5` | Guidance scale |
| `ENABLE_TF32` | `true` | TF32 가속 활성화 (RTX 30xx+) |

## 🖥️ 시스템 요구사항

| 항목 | 최소 요구사항 | 권장 사양 |
|------|--------------|----------|
| GPU VRAM | 8GB | 12GB+ |
| RAM | 16GB | 32GB |
| 디스크 공간 | 10GB | 20GB |
| GPU | RTX 3060 | RTX 4070+ |

## 🐳 Docker 실행 (GPU)

> 로컬 환경이 아닌 **Vast.ai, RunPod 등 클라우드 GPU 서버**에 배포 시 사용합니다.

```bash
cd AI/server
docker build -t fastfit-server .
docker run --gpus all -p 8000:8000 -v $(pwd)/../models:/models fastfit-server
```
