"""
FastFit Model Service - Local Inference Mode

FastFit 모델을 로드하고 로컬에서 실제 추론을 수행하는 서비스 클래스입니다.
"""
import base64
import io
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from app.config import settings
from app.schemas.fitting import GarmentCategory, GarmentItem


# FastFit 모듈 경로 추가
FASTFIT_PATH = settings.fastfit_model_path
if FASTFIT_PATH not in sys.path:
    sys.path.insert(0, FASTFIT_PATH)


def center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """이미지를 목표 종횡비로 중앙 크롭"""
    width, height = img.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
        left = (width - new_width) // 2
        top = 0
    else:
        new_width = width
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
    
    return img.crop((left, top, left + new_width, top + new_height))


class FastFitService:
    """FastFit 모델 추론 서비스 (로컬 추론)"""

    # 모델 경로 설정
    BASE_MODEL_PATH = os.path.join(FASTFIT_PATH, "Models", "FastFit-MR-1024")
    UTIL_MODEL_PATH = os.path.join(FASTFIT_PATH, "Models", "Human-Toolkit")
    
    # 이미지 크기 설정
    PERSON_SIZE = (768, 1024)  # (width, height)

    def __init__(self):
        """서비스 초기화 (모델 지연 로딩)"""
        self._pipeline = None
        self._dwpose_detector = None
        self._densepose_detector = None
        self._schp_lip_detector = None
        self._schp_atr_detector = None
        self._model_loaded = False
        self._device = settings.device if torch.cuda.is_available() else "cpu"
        
        print(f"[FastFitService] Initialized (device: {self._device})")

    def _download_models_if_needed(self):
        """필요시 HuggingFace에서 모델 다운로드"""
        # FastFit 모델 다운로드
        if not os.path.exists(self.BASE_MODEL_PATH):
            print(f"[FastFitService] Downloading FastFit-MR-1024 model...")
            os.makedirs(self.BASE_MODEL_PATH, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/FastFit-MR-1024",
                local_dir=self.BASE_MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print(f"[FastFitService] FastFit-MR-1024 downloaded to {self.BASE_MODEL_PATH}")
        
        # Human Toolkit 모델 다운로드
        if not os.path.exists(self.UTIL_MODEL_PATH):
            print(f"[FastFitService] Downloading Human-Toolkit models...")
            os.makedirs(self.UTIL_MODEL_PATH, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/Human-Toolkit",
                local_dir=self.UTIL_MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print(f"[FastFitService] Human-Toolkit downloaded to {self.UTIL_MODEL_PATH}")

    def _load_model(self):
        """FastFit 모델 및 유틸리티 모델 로딩"""
        if self._model_loaded:
            return

        print(f"[FastFitService] Loading models...")
        
        # 모델 다운로드 확인
        self._download_models_if_needed()

        try:
            # FastFit 모듈 임포트
            from module.pipeline_fastfit import FastFitPipeline
            from parse_utils import DWposeDetector, DensePose, SCHP, multi_ref_cloth_agnostic_mask
            
            # 전역에서 사용할 수 있도록 저장
            self._multi_ref_cloth_agnostic_mask = multi_ref_cloth_agnostic_mask
            
            # DWPose 검출기 (CPU에서 실행 권장)
            self._dwpose_detector = DWposeDetector(
                pretrained_model_name_or_path=os.path.join(self.UTIL_MODEL_PATH, "DWPose"),
                device='cpu'
            )
            
            # DensePose 검출기
            self._densepose_detector = DensePose(
                model_path=os.path.join(self.UTIL_MODEL_PATH, "DensePose"),
                device=self._device
            )
            
            # SCHP 세그멘테이션 모델
            self._schp_lip_detector = SCHP(
                ckpt_path=os.path.join(self.UTIL_MODEL_PATH, "SCHP", "schp-lip.pth"),
                device=self._device
            )
            self._schp_atr_detector = SCHP(
                ckpt_path=os.path.join(self.UTIL_MODEL_PATH, "SCHP", "schp-atr.pth"),
                device=self._device
            )
            
            # FastFit 파이프라인
            self._pipeline = FastFitPipeline(
                base_model_path=self.BASE_MODEL_PATH,
                device=self._device,
                mixed_precision="bf16" if self._device == "cuda" else "no",
                allow_tf32=True
            )
            
            self._model_loaded = True
            print(f"[FastFitService] All models loaded successfully (device: {self._device})")
            
        except ImportError as e:
            raise RuntimeError(f"FastFit 모듈 로딩 실패: {e}. FastFit 경로를 확인하세요: {FASTFIT_PATH}")
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {e}")

    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Base64 문자열을 PIL Image로 디코딩"""
        try:
            # Data URL 형식 처리 (예: data:image/png;base64,...)
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"이미지 디코딩 실패: {str(e)}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """PIL Image를 Base64 문자열로 인코딩"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _preprocess_person_image(self, person_img: Image.Image) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """
        사람 이미지 전처리 (포즈 및 마스크 생성용)
        
        Returns:
            (pose_img, densepose_arr, lip_arr, atr_arr)
        """
        # 3:4 비율로 크롭 및 리사이즈
        person_img = person_img.convert("RGB")
        person_img = center_crop_to_aspect_ratio(person_img, 3/4)
        person_img = person_img.resize(self.PERSON_SIZE, Image.LANCZOS)
        
        # 포즈 추정
        pose_img = self._dwpose_detector(person_img)
        if not isinstance(pose_img, Image.Image):
            raise RuntimeError("포즈 추정 실패")
        
        # DensePose 및 SCHP 세그멘테이션
        densepose_arr = np.array(self._densepose_detector(person_img))
        lip_arr = np.array(self._schp_lip_detector(person_img))
        atr_arr = np.array(self._schp_atr_detector(person_img))
        
        return pose_img, densepose_arr, lip_arr, atr_arr

    def _generate_mask(
        self, 
        densepose_arr: np.ndarray, 
        lip_arr: np.ndarray, 
        atr_arr: np.ndarray,
        square_cloth_mask: bool = False
    ) -> Image.Image:
        """마스크 생성"""
        return self._multi_ref_cloth_agnostic_mask(
            densepose_arr, lip_arr, atr_arr,
            square_cloth_mask=square_cloth_mask,
            horizon_expand=True
        )

    def _prepare_reference_images(
        self,
        garments: List[Tuple[Optional[Image.Image], str]],
        ref_height: int = 512
    ) -> Tuple[List[Image.Image], List[str], List[int]]:
        """
        참조 이미지 준비 (FastFit 형식)
        
        Args:
            garments: [(image, category_label), ...] 형식의 리스트
            ref_height: 참조 이미지 높이 (512/768/1024)
            
        Returns:
            (ref_images, ref_labels, ref_attention_masks)
        """
        clothing_ref_size = (int(ref_height * 3 / 4), ref_height)
        accessory_ref_size = (384, 512)
        
        ref_images, ref_labels, ref_attention_masks = [], [], []
        
        # FastFit에서 사용하는 카테고리 순서
        category_order = ["upper", "lower", "overall", "shoe", "bag"]
        
        # 카테고리 매핑
        category_map = {
            "top": "upper",
            "bottom": "lower", 
            "dress": "overall",
            "shoes": "shoe",
            "bag": "bag"
        }
        
        # 입력된 의류들을 딕셔너리로 변환
        garment_dict = {}
        for img, cat in garments:
            if img is not None:
                mapped_cat = category_map.get(cat, cat)
                garment_dict[mapped_cat] = img
        
        # 순서대로 참조 이미지 구성
        for label in category_order:
            target_size = accessory_ref_size if label in ["shoe", "bag"] else clothing_ref_size
            
            if label in garment_dict:
                img = garment_dict[label].convert("RGB").resize(target_size, Image.LANCZOS)
                ref_images.append(img)
                ref_labels.append(label)
                ref_attention_masks.append(1)
            else:
                # 빈 이미지 생성 (FastFit 요구사항)
                ref_images.append(Image.new("RGB", target_size, color=(0, 0, 0)))
                ref_labels.append(label)
                ref_attention_masks.append(0)
        
        return ref_images, ref_labels, ref_attention_masks

    async def run_single_fitting(
        self,
        person_image: str,
        garment_image: str,
        category: GarmentCategory,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> str:
        """
        단일 아이템 피팅 수행
        
        Args:
            person_image: 사용자 전신 이미지 (Base64)
            garment_image: 의류 이미지 (Base64)
            category: 의류 카테고리
            num_inference_steps: Diffusion 스텝 수
            guidance_scale: Guidance scale
            seed: 랜덤 시드
            
        Returns:
            결과 이미지 (Base64)
        """
        # 단일 아이템을 다중 아이템 형식으로 변환
        garment_item = GarmentItem(category=category, image=garment_image)
        return await self.run_multi_fitting(
            person_image=person_image,
            garments=[garment_item],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    async def run_multi_fitting(
        self,
        person_image: str,
        garments: List[GarmentItem],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> str:
        """
        다중 아이템 피팅 수행
        
        Args:
            person_image: 사용자 전신 이미지 (Base64)
            garments: 의류 아이템 목록
            num_inference_steps: Diffusion 스텝 수
            guidance_scale: Guidance scale
            seed: 랜덤 시드
            
        Returns:
            결과 이미지 (Base64)
        """
        # 모델 로딩 (최초 1회)
        self._load_model()
        
        # 시드 설정
        if seed is None:
            seed = settings.default_seed
        
        # 이미지 디코딩
        person_img = self._decode_base64_image(person_image)
        
        # 의류 이미지 디코딩 및 준비
        garment_tuples = []
        for g in garments:
            img = self._decode_base64_image(g.image)
            garment_tuples.append((img, g.category.value))
        
        # 사람 이미지 전처리 (3:4 비율 크롭 및 리사이즈)
        processed_person = person_img.convert("RGB")
        processed_person = center_crop_to_aspect_ratio(processed_person, 3/4)
        processed_person = processed_person.resize(self.PERSON_SIZE, Image.LANCZOS)
        
        # 포즈 및 세그멘테이션
        pose_img, densepose_arr, lip_arr, atr_arr = self._preprocess_person_image(person_img)
        
        # 마스크 생성
        mask_img = self._generate_mask(densepose_arr, lip_arr, atr_arr, square_cloth_mask=False)
        
        # 참조 이미지 준비
        ref_images, ref_labels, ref_attention_masks = self._prepare_reference_images(
            garment_tuples, 
            ref_height=512  # 기본값
        )
        
        # Generator 설정
        generator = torch.Generator(device=self._device).manual_seed(seed)
        
        # 추론 실행
        with torch.no_grad():
            result = self._pipeline(
                person=processed_person,
                mask=mask_img,
                ref_images=ref_images,
                ref_labels=ref_labels,
                ref_attention_masks=ref_attention_masks,
                pose=pose_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                return_pil=True
            )
        
        # 결과 처리
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], Image.Image):
            return self._encode_image_to_base64(result[0])
        
        raise RuntimeError("추론 실패: 유효한 이미지가 반환되지 않았습니다.")
