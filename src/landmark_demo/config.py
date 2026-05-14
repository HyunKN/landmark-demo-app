"""config.toml 로더."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore


@dataclass
class AppConfig:
    assets_dir: str
    log_path: str
    checkpoint: str
    device: str
    inference_backend: str
    warmup_on_start: bool
    reject_threshold: float
    image_only: dict           # FusionWeights kwargs
    text_only: dict
    max_image_mb: int
    slow_inference_ms: int
    title: str


def load_config(path: str) -> AppConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    runtime = raw.get("runtime", {})
    fusion = raw.get("fusion", {})
    ui = raw.get("ui", {})
    return AppConfig(
        assets_dir=paths.get("assets_dir", "./assets"),
        log_path=paths.get("log_path", "./logs/demo.jsonl"),
        checkpoint=paths.get("checkpoint", "./best.pt"),
        device=runtime.get("device", "auto"),
        inference_backend=runtime.get("inference_backend", "pytorch"),
        warmup_on_start=runtime.get("warmup_on_start", True),
        reject_threshold=float(fusion.get("reject_threshold", 0.25)),
        image_only=fusion.get("image_only", {"w_image": 1.0, "w_text": 0.0, "w_keyword": 0.0}),
        text_only=fusion.get("text_only", {"w_image": 0.0, "w_text": 0.6, "w_keyword": 0.4}),
        max_image_mb=int(ui.get("max_image_mb", 10)),
        slow_inference_ms=int(ui.get("slow_inference_ms", 5000)),
        title=ui.get("title", "Jongno Landmark Demo"),
    )
