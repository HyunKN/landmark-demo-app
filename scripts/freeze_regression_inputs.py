"""Freeze a reproducible image set for the demo regression suite.

Reads ../Dataset/<class>/labels.json, picks the highest-quality candidate per
class that is NOT already used as a hero image (so prototypes remain a fair
holdout target), copies it into assets/regression_inputs/positive/<class>.jpg,
and synthesizes a couple of out-of-scope inputs (uniform noise + a fake screen
capture image) into assets/regression_inputs/out_of_scope/.

Also rewrites tests/fixtures/demo_regression_v1.json's manual_image_cases so
input_hint values point at these frozen files.

Usage:
    python scripts/freeze_regression_inputs.py \
        --data-root ../Dataset \
        --assets-dir ./assets \
        --fixture ./tests/fixtures/demo_regression_v1.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "Dataset"))
    p.add_argument("--assets-dir", default=str(REPO_ROOT / "assets"))
    p.add_argument("--fixture",
                   default=str(REPO_ROOT / "tests" / "fixtures" / "demo_regression_v1.json"))
    p.add_argument("--landmark-info",
                   default=str(REPO_ROOT / "assets" / "landmark_info.json"))
    return p.parse_args()


def _hero_basename(assets_dir: Path, landmark_id: str) -> str | None:
    """If a hero image was copied for this class, return its source basename
    so we can pick a different image for regression and avoid information leak."""
    hero = assets_dir / "hero_images" / f"{landmark_id}.jpg"
    if not hero.exists():
        return None
    # We don't track the original path; use file size as a weak fingerprint.
    return f"size:{hero.stat().st_size}"


def _score_record(rec: dict) -> int:
    score = 0
    if rec.get("quality_status") == "ok":
        score += 10
    vt = rec.get("view_type")
    if vt == "exterior":
        score += 5
    elif vt == "night":
        score += 1
    if rec.get("label_status") == "confirmed":
        score += 2
    return score


def _pick_holdout(data_root: Path, assets_dir: Path, landmark_id: str) -> Path | None:
    labels_path = data_root / landmark_id / "labels.json"
    if not labels_path.exists():
        return None
    records = json.loads(labels_path.read_text(encoding="utf-8"))
    if not records:
        return None

    hero_fingerprint = _hero_basename(assets_dir, landmark_id)
    images_dir = data_root / landmark_id / "images"

    scored: list[tuple[int, Path]] = []
    for rec in records:
        if rec.get("label_status") != "confirmed":
            continue
        fname = Path(rec["file_name"]).name
        candidate = images_dir / fname
        if not candidate.exists():
            continue
        # Skip the hero by size fingerprint when possible.
        if hero_fingerprint and f"size:{candidate.stat().st_size}" == hero_fingerprint:
            continue
        scored.append((_score_record(rec), candidate))

    if not scored:
        # Fallback: pick the second-best confirmed record so we leave the hero alone.
        for rec in records:
            if rec.get("label_status") != "confirmed":
                continue
            fname = Path(rec["file_name"]).name
            candidate = images_dir / fname
            if candidate.exists():
                scored.append((_score_record(rec), candidate))
        if not scored:
            return None

    scored.sort(key=lambda kv: (-kv[0], kv[1].name))
    # Prefer the second highest if available (so hero != regression even when
    # fingerprints don't match).
    if len(scored) >= 2:
        return scored[1][1]
    return scored[0][1]


def _make_screen_capture_like(out_path: Path, size: int = 512) -> None:
    """Synthesize a UI screenshot-like image: gradient background + fake window
    chrome + text. Stable seed makes this reproducible across runs."""
    rng = np.random.default_rng(20260516)
    grad = np.linspace(0.92, 1.0, size, dtype=np.float32)[:, None]
    bg = np.repeat(grad, size, axis=1)
    rgb = np.stack([bg * 0.97, bg * 0.98, bg], axis=-1)
    img = Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(img)
    # Title bar
    draw.rectangle([0, 0, size, 36], fill=(230, 230, 235))
    draw.ellipse([10, 10, 24, 24], fill=(255, 95, 86))
    draw.ellipse([30, 10, 44, 24], fill=(255, 189, 46))
    draw.ellipse([50, 10, 64, 24], fill=(39, 201, 63))
    # Window content frame
    draw.rectangle([16, 56, size - 16, size - 16], outline=(180, 180, 195), width=2)
    # Faux text rows
    for i in range(8):
        y = 80 + i * 40
        w = int(size * (0.75 - 0.05 * (i % 3)))
        draw.rectangle([32, y, 32 + w, y + 12], fill=(60, 60, 70))
    # Some pixel jitter so it's not a constant image
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0.0, 2.5, arr.shape)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    img.save(out_path, "PNG")


def _make_uniform_noise(out_path: Path, size: int = 512) -> None:
    rng = np.random.default_rng(20260517)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(out_path, "JPEG", quality=85)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    assets_dir = Path(args.assets_dir)
    fixture_path = Path(args.fixture)
    landmark_info = json.loads(Path(args.landmark_info).read_text(encoding="utf-8"))
    classes = [it["landmark_id"] for it in landmark_info["items"]]

    positive_dir = assets_dir / "regression_inputs" / "positive"
    oos_dir = assets_dir / "regression_inputs" / "out_of_scope"
    positive_dir.mkdir(parents=True, exist_ok=True)
    oos_dir.mkdir(parents=True, exist_ok=True)

    # ---- positive set ----
    frozen_positive: list[dict] = []
    missing: list[str] = []
    for landmark_id in classes:
        src = _pick_holdout(data_root, assets_dir, landmark_id)
        if src is None:
            missing.append(landmark_id)
            print(f"[skip] {landmark_id}: no confirmed image available")
            continue
        dst = positive_dir / f"{landmark_id}.jpg"
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=92)
        except Exception as exc:
            print(f"[fail] {landmark_id}: copy failed ({exc})")
            missing.append(landmark_id)
            continue
        frozen_positive.append({
            "landmark_id": landmark_id,
            "source": str(src.relative_to(data_root.parent)),
            "frozen_at": str(dst.relative_to(REPO_ROOT)),
        })
        print(f"[freeze] {landmark_id}: {src.name} -> {dst.name}")

    # ---- out_of_scope set ----
    screen_path = oos_dir / "screen_capture_synthetic.png"
    noise_path = oos_dir / "uniform_noise.jpg"
    _make_screen_capture_like(screen_path)
    _make_uniform_noise(noise_path)
    print(f"[freeze] out_of_scope: {screen_path.name}, {noise_path.name}")

    # ---- patch fixture ----
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    manual_cases: list[dict] = []
    for entry in frozen_positive:
        landmark_id = entry["landmark_id"]
        manual_cases.append({
            "id": f"positive-{landmark_id}",
            "input_hint": entry["frozen_at"].replace("\\", "/"),
            "expected_top1": landmark_id,
            "expected_decision": ["matched", "ambiguous"],
        })
    manual_cases.append({
        "id": "out-of-scope-screen-capture-synthetic",
        "input_hint": str(screen_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "expected_decision": "out_of_scope",
    })
    manual_cases.append({
        "id": "out-of-scope-uniform-noise",
        "input_hint": str(noise_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "expected_decision": "out_of_scope",
    })

    fixture["manual_image_cases"] = manual_cases
    fixture["frozen_at"] = "2026-05-16"
    fixture["frozen_positive_sources"] = frozen_positive
    fixture["frozen_missing"] = missing
    fixture["notes"] = (
        "Manual image cases reference frozen inputs under "
        "assets/regression_inputs/. Regenerate via "
        "scripts/freeze_regression_inputs.py."
    )

    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[write] {fixture_path}  (manual_image_cases={len(manual_cases)})")

    if missing:
        print(f"[warn] missing classes: {missing}")
        sys.exit(0)


if __name__ == "__main__":
    main()
