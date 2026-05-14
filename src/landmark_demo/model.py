"""MobileCLIP2 image encoder model wrapper for the demo runtime.

이 모듈은 학습 레포의 train.py에 정의된 MobileClipLandmarkModel과 같은 구조를
재현해 best.pt를 로드한다. 학습 코드와 의존성을 분리하기 위해 같은 모듈 구조를
이 데모 안에 다시 정의한다.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None or self.margin <= 0:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class MobileClipLandmarkModel(nn.Module):
    """학습 레포의 동명 클래스와 같은 forward 구조."""

    def __init__(self, cfg: dict, num_classes: int, embedding_dim: int, loss_name: str) -> None:
        super().__init__()
        import open_clip

        pretrained = os.environ.get("MOBILECLIP_CHECKPOINT") or cfg["model"].get("pretrained", "dfndr2b")
        self.clip, _, _ = open_clip.create_model_and_transforms(cfg["model"]["model_name"], pretrained=pretrained)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.embedding = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        if "arcface" in loss_name:
            self.head = ArcMarginHead(embedding_dim, num_classes)
        else:
            self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.clip.encode_image(x)
        embeddings = F.normalize(self.embedding(features.float()))
        if isinstance(self.head, ArcMarginHead):
            logits = self.head(embeddings, labels)
        else:
            logits = self.head(embeddings)
        return logits, embeddings

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """MobileCLIP2의 text tower를 그대로 사용. 정규화된 임베딩을 반환."""
        with torch.no_grad():
            features = self.clip.encode_text(text_tokens).float()
        return F.normalize(features, dim=-1)


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple[MobileClipLandmarkModel, list[str], dict]:
    """best.pt를 로드해 평가 모드 모델, 클래스 목록, config를 반환."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    classes: list[str] = ckpt["classes"]
    embedding_dim = int(cfg["training"].get("embedding_dim", 512))
    loss_name = str(cfg["training"].get("loss", "ce"))

    model = MobileClipLandmarkModel(cfg, num_classes=len(classes), embedding_dim=embedding_dim, loss_name=loss_name)

    # LazyLinear 초기화
    image_size = int(cfg["training"]["image_size"])
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size)
        dummy_labels = torch.zeros(2, dtype=torch.long)
        model(dummy, dummy_labels)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, classes, cfg
