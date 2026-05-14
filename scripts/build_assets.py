"""학습 산출물(best.pt) → 데모 자산 (prototype_index, landmark_text_index, hero_images) 빌드.

사용:
    python scripts/build_assets.py \
        --checkpoint ./best.pt \
        --data-root ../Dataset \
        --landmark-info ./assets/landmark_info.json \
        --output-dir ./assets

산출:
    ./assets/prototype_index.json        클래스별 image embedding 평균
    ./assets/landmark_text_index.json    description+keywords text embedding
    ./assets/hero_images/<id>.jpg        대표 이미지 1장 자동 선정
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 데모 패키지를 import 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmark_demo.model import load_checkpoint
from landmark_demo.inference import ImageRecognizer, TextEncoder


def pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def collect_class_images(data_root: Path, landmark_id: str) -> list[dict]:
    """Dataset/<landmark_id>/labels.json에서 confirmed 레코드만 모아 반환."""
    labels_path = data_root / landmark_id / "labels.json"
    if not labels_path.exists():
        return []
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    out = []
    for r in data:
        if r.get("label_status") != "confirmed":
            continue
        img_path = data_root / landmark_id / "images" / Path(r["file_name"]).name
        if img_path.exists():
            r["_abs_path"] = str(img_path)
            out.append(r)
    return out


def select_hero(records: list[dict]) -> Path | None:
    """quality_status==ok 우선, view_type==exterior 가산점, 첫 항목 반환."""
    if not records:
        return None
    scored = []
    for r in records:
        score = 0
        if r.get("quality_status") == "ok":
            score += 10
        if r.get("view_type") == "exterior":
            score += 5
        if r.get("view_type") == "night":
            score += 1
        scored.append((score, r["_abs_path"]))
    scored.sort(key=lambda x: -x[0])
    return Path(scored[0][1]) if scored else None


def build_prototypes(model, recognizer: ImageRecognizer, data_root: Path, classes: list[str]) -> dict:
    items = []
    for landmark_id in classes:
        records = collect_class_images(data_root, landmark_id)
        print(f"[proto] {landmark_id}: {len(records)} confirmed images")
        if not records:
            print(f"  WARN: no images, using zero prototype")
            items.append({
                "landmark_id": landmark_id,
                "prototype": [0.0] * 512,
                "n_samples_used": 0,
                "view_breakdown": {},
            })
            continue
        embeddings = []
        view_counts: dict[str, int] = {}
        for r in records:
            try:
                img = Image.open(r["_abs_path"]).convert("RGB")
                emb, _ = recognizer.encode(img)
                embeddings.append(emb)
                vt = r.get("view_type") or "unknown"
                view_counts[vt] = view_counts.get(vt, 0) + 1
            except Exception as exc:
                print(f"  skip {r['_abs_path']}: {exc}")
        if not embeddings:
            continue
        proto = np.mean(np.stack(embeddings), axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        items.append({
            "landmark_id": landmark_id,
            "prototype": proto.astype(float).tolist(),
            "n_samples_used": len(embeddings),
            "view_breakdown": view_counts,
        })
    return items


def build_text_index(text_encoder: TextEncoder, landmark_info: dict) -> list[dict]:
    items = []
    for entry in landmark_info["items"]:
        landmark_id = entry["landmark_id"]
        # description + name + keywords concatenation
        keywords = entry.get("aliases", []) + entry.get("tags", [])
        parts = [
            f"a photo of {entry['name_en']}",
            entry.get("description_ko", ""),
            f"a Korean landmark called {entry['name_ko']}",
            ", ".join(keywords),
        ]
        composite = ". ".join(p for p in parts if p)
        embedding = text_encoder.encode(composite)
        items.append({
            "landmark_id": landmark_id,
            "description_ko": entry.get("description_ko", ""),
            "description_en": "",
            "keywords": keywords,
            "embedding": embedding.astype(float).tolist(),
        })
        print(f"[text] {landmark_id}: encoded {len(composite)} chars")
    return items


def copy_hero_images(data_root: Path, classes: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for landmark_id in classes:
        records = collect_class_images(data_root, landmark_id)
        hero = select_hero(records)
        if hero is None:
            print(f"[hero] {landmark_id}: no candidate")
            continue
        dst = out_dir / f"{landmark_id}.jpg"
        try:
            img = Image.open(hero).convert("RGB")
            img.save(dst, "JPEG", quality=88)
            print(f"[hero] {landmark_id}: {hero.name} -> {dst.name}")
        except Exception as exc:
            print(f"[hero] {landmark_id}: failed ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--landmark-info", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-prototypes", action="store_true")
    parser.add_argument("--skip-text-index", action="store_true")
    parser.add_argument("--skip-hero-images", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[env] device={device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint: {args.checkpoint}")
    model, classes, cfg = load_checkpoint(args.checkpoint, device=device)
    image_size = int(cfg["training"]["image_size"])
    image_mean = list(cfg["training"]["image_mean"])
    image_std = list(cfg["training"]["image_std"])
    recognizer = ImageRecognizer(model, image_size, image_mean, image_std, device=device)

    landmark_info = json.loads(Path(args.landmark_info).read_text(encoding="utf-8"))
    info_classes = [it["landmark_id"] for it in landmark_info["items"]]
    if set(info_classes) != set(classes):
        print(f"  WARN: landmark_info={info_classes}\n  ckpt={classes}")

    # Hero images
    if not args.skip_hero_images:
        print("\n=== copying hero images ===")
        copy_hero_images(Path(args.data_root), classes, output_dir / "hero_images")

    # Prototypes
    if not args.skip_prototypes:
        print("\n=== building prototype index ===")
        proto_items = build_prototypes(model, recognizer, Path(args.data_root), classes)
        proto_doc = {
            "version": "sprint1-v1",
            "encoder": {
                "model_name": cfg["model"]["model_name"],
                "image_size": image_size,
                "embedding_dim": int(cfg["training"].get("embedding_dim", 512)),
                "image_mean": image_mean,
                "image_std": image_std,
            },
            "items": proto_items,
        }
        out_path = output_dir / "prototype_index.json"
        out_path.write_text(json.dumps(proto_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {out_path}")

    # Text index
    if not args.skip_text_index:
        print("\n=== building text index ===")
        import open_clip
        tokenizer = open_clip.get_tokenizer(cfg["model"]["model_name"])
        text_encoder = TextEncoder(model, tokenizer, device=device)
        text_items = build_text_index(text_encoder, landmark_info)
        text_doc = {
            "version": "sprint1-v1",
            "encoder": {
                "model_name": cfg["model"]["model_name"] + "-text",
                "embedding_dim": len(text_items[0]["embedding"]) if text_items else 512,
            },
            "items": text_items,
        }
        out_path = output_dir / "landmark_text_index.json"
        out_path.write_text(json.dumps(text_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {out_path}")

    print("\n[done]")


if __name__ == "__main__":
    main()
