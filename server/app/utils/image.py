"""
Image Utility Functions
"""
import base64
import io
from typing import Tuple

from PIL import Image


class ImageUtils:
    """이미지 처리 유틸리티"""

    @staticmethod
    def decode_base64(base64_string: str) -> Image.Image:
        """
        Base64 문자열을 PIL Image로 디코딩
        
        Args:
            base64_string: Base64 인코딩된 이미지 문자열
            
        Returns:
            PIL Image 객체
        """
        # Data URL 형식 처리
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")

    @staticmethod
    def encode_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """
        PIL Image를 Base64 문자열로 인코딩
        
        Args:
            image: PIL Image 객체
            format: 이미지 포맷 (PNG, JPEG 등)
            
        Returns:
            Base64 인코딩된 문자열
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def resize_to_fit(
        image: Image.Image,
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        이미지를 목표 크기에 맞게 리사이즈
        
        Args:
            image: 원본 이미지
            target_size: (width, height) 튜플
            maintain_aspect: 종횡비 유지 여부
            
        Returns:
            리사이즈된 이미지
        """
        if maintain_aspect:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)

    @staticmethod
    def validate_image_size(
        image: Image.Image,
        min_size: Tuple[int, int] = (256, 256),
        max_size: Tuple[int, int] = (4096, 4096)
    ) -> bool:
        """
        이미지 크기 유효성 검증
        
        Args:
            image: 검증할 이미지
            min_size: 최소 크기 (width, height)
            max_size: 최대 크기 (width, height)
            
        Returns:
            유효 여부
        """
        width, height = image.size
        min_w, min_h = min_size
        max_w, max_h = max_size
        
        return (min_w <= width <= max_w) and (min_h <= height <= max_h)
