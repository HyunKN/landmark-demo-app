"""Export Sprint 1 image encoder handoff package.

Outputs:
  mobile_artifacts/landmark_encoder.onnx
  mobile_artifacts/preprocessing.json
  mobile_artifacts/labels_master.json
  mobile_artifacts/prototype_index.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmark_demo.model import load_checkpoint


class ImageEmbeddingModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, embedding = self.model(image)
        return embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./best.pt")
    parser.add_argument("--assets-dir", default="./assets")
    parser.add_argument("--output-dir", default="./mobile_artifacts")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, classes, train_cfg = load_checkpoint(str(checkpoint), device="cpu")
    image_size = int(train_cfg["training"]["image_size"])
    image_mean = [float(x) for x in train_cfg["training"]["image_mean"]]
    image_std = [float(x) for x in train_cfg["training"]["image_std"]]

    wrapper = ImageEmbeddingModule(model).eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    onnx_path = output_dir / "landmark_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["embedding"],
            opset_version=args.opset,
            dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
            do_constant_folding=True,
            external_data=True,
        )

    preprocessing = {
        "version": "sprint1-mobile-preprocessing-v1",
        "input_name": "image",
        "output_name": "embedding",
        "image_size": image_size,
        "resize_short_side_scale": 1.15,
        "crop": "center",
        "color_space": "RGB",
        "layout": "NCHW",
        "dtype": "float32",
        "value_range": "0_1",
        "mean": image_mean,
        "std": image_std,
    }
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "version": "sprint1-labels-v1",
        "classes": [{"index": idx, "landmark_id": landmark_id} for idx, landmark_id in enumerate(classes)],
    }
    (output_dir / "labels_master.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    proto_src = assets_dir / "prototype_index.json"
    if proto_src.exists():
        shutil.copy2(proto_src, output_dir / "prototype_index.json")

    manifest = {
        "version": "sprint1-mobile-artifacts-v1",
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx_path.resolve()),
        "preprocessing": str((output_dir / "preprocessing.json").resolve()),
        "labels": str((output_dir / "labels_master.json").resolve()),
        "prototype_index": str((output_dir / "prototype_index.json").resolve()),
        "opset": args.opset,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
