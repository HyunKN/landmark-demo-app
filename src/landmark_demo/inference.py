"""이미지 전처리 + 텍스트 토크나이즈 + 임베딩 산출."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


class ImageRecognizer:
    def __init__(self, model, image_size: int, image_mean: list[float], image_std: list[float], device: str) -> None:
        self.model = model
        self.image_size = image_size
        self.mean = np.array(image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(image_std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """짧은 변 image_size 리사이즈 + center-crop + 정규화."""
        img = image.convert("RGB")
        w, h = img.size
        target = self.image_size
        scale = (int(target * 1.15)) / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        # center crop
        left = (new_w - target) // 2
        top = (new_h - target) // 2
        img = img.crop((left, top, left + target, top + target))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr).to(self.device)

    @torch.no_grad()
    def encode(self, image: Image.Image) -> tuple[np.ndarray, int]:
        """입력 이미지를 (512,) L2-normalized embedding과 처리시간 ms로 반환."""
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        _, embedding = self.model(tensor)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return embedding.cpu().numpy()[0].astype(np.float32), elapsed_ms

    @torch.no_grad()
    def encode_clip_image(self, image: Image.Image) -> np.ndarray:
        """학습된 head 임베딩이 아닌 CLIP image tower 원본 임베딩.

        Text tower와 같은 공간이라 자연어 검색 fusion에 사용한다.
        """
        tensor = self.preprocess(image)
        features = self.model.clip.encode_image(tensor).float()
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0].astype(np.float32)


class TextEncoder:
    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        embedding = self.model.encode_text(tokens)
        return embedding.cpu().numpy()[0].astype(np.float32)

    @torch.no_grad()
    def encode_many(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        tokens = self.tokenizer(text_list).to(self.device)
        embeddings = self.model.encode_text(tokens)
        return embeddings.cpu().numpy().astype(np.float32)


def validate_image_file(filename: str, size_bytes: int, max_mb: int = 10) -> tuple[bool, str]:
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        return False, f"지원하지 않는 이미지 형식입니다 ({ext or '알수없음'})"
    if size_bytes > max_mb * 1024 * 1024:
        return False, f"{max_mb}MB 이하 이미지만 지원합니다"
    return True, ""
