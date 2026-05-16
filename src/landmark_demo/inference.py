"""이미지 전처리 + 텍스트 토크나이즈 + 임베딩 산출."""
from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


@dataclass
class ImageQualityReport:
    ok: bool
    reason_codes: list[str]
    min_side: int
    brightness: float
    contrast: float
    sharpness: float


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


class OnnxImageRecognizer:
    """ONNX Runtime image embedding recognizer.

    This mirrors `ImageRecognizer.preprocess` using `preprocessing.json` from
    `scripts/export_mobile_onnx.py`. It is image-only; text search can still use
    keyword/text-index fallback or a separately loaded PyTorch text tower.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        onnx_path = artifact_dir / "landmark_encoder.onnx"
        preprocessing_path = artifact_dir / "preprocessing.json"
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing ONNX artifact: {onnx_path}")
        if not preprocessing_path.exists():
            raise FileNotFoundError(f"missing preprocessing metadata: {preprocessing_path}")

        import onnxruntime as ort

        self.preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        w, h = img.size
        target = int(self.preprocessing["image_size"])
        scale = float(self.preprocessing.get("resize_short_side_scale", 1.15)) * target / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        left = (new_w - target) // 2
        top = (new_h - target) // 2
        img = img.crop((left, top, left + target, top + target))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        mean = np.asarray(self.preprocessing["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(self.preprocessing["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        return (arr - mean) / std

    def encode(self, image: Image.Image) -> tuple[np.ndarray, int]:
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        embedding = self.session.run([self.output_name], {self.input_name: tensor})[0][0].astype(np.float32)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return embedding, elapsed_ms


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


def assess_image_quality(image: Image.Image) -> ImageQualityReport:
    """가벼운 품질 휴리스틱. 모바일 시연용 안내에만 사용하고 학습 평가는 대체하지 않는다."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    min_side = min(w, h)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    gy, gx = np.gradient(gray)
    sharpness = float(np.mean(gx * gx + gy * gy))

    reasons: list[str] = []
    if min_side < 128:
        reasons.append("too_small")
    if brightness < 25.0:
        reasons.append("too_dark")
    if brightness > 245.0 and contrast < 10.0:
        reasons.append("too_bright")
    if contrast < 5.0:
        reasons.append("low_contrast")
    if sharpness < 1.5 and min_side >= 128:
        reasons.append("blur_detected")

    return ImageQualityReport(
        ok=len(reasons) == 0,
        reason_codes=reasons,
        min_side=min_side,
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2),
    )
