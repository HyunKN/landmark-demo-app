"""Static QDQ INT8 quantization for the Sprint 1 image encoder.

Why static QDQ instead of dynamic INT8?
  - QAIRT (Qualcomm AI Hub's compiler) accepts standard QDQ ONNX directly,
    while ORT's dynamic INT8 (DynamicQuantizeLinear / QGemm) is rejected.
  - Static activation calibration uses real images, so embedding distribution
    is preserved much more faithfully than weight-only schemes.
  - Calibration set is fully in our control; we can scale it from 15 to 500+
    images without involving any cloud queue.

Pipeline:
  1. Reuse FP32 ONNX preprocessing from quantize_mobile_onnx.py (load with
     external data, strip stale value_info, in-memory shape inference).
  2. Sample N images per class from Dataset/<class>/labels.json (confirmed
     only) using the same preprocessing as the demo app.
  3. Run onnxruntime.quantization.quantize_static with QuantFormat.QDQ,
     per-channel weights INT8, per-tensor activations INT8 (configurable).
  4. Emit a parallel bundle directory.

Usage:
    python scripts/quantize_static_qdq.py \
        --src ./mobile_artifacts \
        --dst ./mobile_artifacts_int8_qdq \
        --data-root ../Dataset \
        --landmark-info ./assets/landmark_info.json \
        --per-class 20

Outputs in --dst:
    landmark_encoder.onnx (+ .data when external)
    preprocessing.json, labels_master.json, prototype_index.json (copied)
    manifest.json (records quant settings + calibration manifest)
    calibration_manifest.json (which images were used)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# Reuse the shape-inference / pre-process helpers from the dynamic script.
from quantize_mobile_onnx import (
    _self_contained_copy,
    _pre_process,
    _resolve_scratch_dir,
    _bundle_size_mb,
)


# ---------- args ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="./mobile_artifacts")
    p.add_argument("--dst", default="./mobile_artifacts_int8_qdq")
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "Dataset"))
    p.add_argument("--landmark-info", default=str(REPO_ROOT / "assets" / "landmark_info.json"))
    p.add_argument("--per-class", type=int, default=20,
                   help="Calibration images per class (drawn from confirmed set).")
    p.add_argument("--seed", type=int, default=20260516)
    p.add_argument("--per-channel", action="store_true", default=True)
    p.add_argument("--activation-type", choices=("int8", "uint8", "int16"),
                   default="uint8",
                   help="Activation dtype. uint8 (asymmetric) is the safest "
                        "default for ViT post-LayerNorm/GELU distributions; "
                        "int8 (symmetric) tends to mode-collapse on this "
                        "architecture; int16 is most accurate but slower.")
    p.add_argument("--calibrate-method", choices=("minmax", "entropy", "percentile"),
                   default="entropy",
                   help="Activation calibration method. entropy/percentile are "
                        "outlier-robust; minmax is fastest but fragile on ViT.")
    p.add_argument("--activation-symmetric", action="store_true", default=False,
                   help="Force symmetric activation quantization. Off by default "
                        "(asymmetric is safer for ViT).")
    p.add_argument("--exclude-layernorm", action="store_true", default=False,
                   help="Exclude LayerNorm fused MatMuls from quantization "
                        "(rarely needed; flip if HTP rejects an LN-related op).")
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--min-temp-free-gb", type=float, default=8.0)
    return p.parse_args()


# ---------- calibration ----------

def _select_calibration_paths(data_root: Path, classes: list[str],
                              per_class: int, hero_dir: Path,
                              regression_dir: Path, seed: int) -> list[dict]:
    """Pick `per_class` confirmed images per class, excluding hero/regression
    files so the regression run remains an honest holdout.
    """
    rng = random.Random(seed)

    hero_keys = {(p.stat().st_size if p.exists() else 0) for p in hero_dir.glob("*.jpg")} \
        if hero_dir.exists() else set()
    regression_basenames: set[str] = set()
    if regression_dir.exists():
        for p in regression_dir.rglob("*"):
            if p.is_file():
                regression_basenames.add(p.name)

    out: list[dict] = []
    for cls in classes:
        labels_path = data_root / cls / "labels.json"
        if not labels_path.exists():
            print(f"[calib] {cls}: labels.json missing, skipping")
            continue
        records = json.loads(labels_path.read_text(encoding="utf-8"))
        candidates: list[Path] = []
        for r in records:
            if r.get("label_status") != "confirmed":
                continue
            fname = Path(r["file_name"]).name
            p = data_root / cls / "images" / fname
            if not p.exists():
                continue
            if fname in regression_basenames:
                continue
            try:
                if p.stat().st_size in hero_keys:
                    continue
            except OSError:
                pass
            candidates.append(p)

        if not candidates:
            print(f"[calib] {cls}: no candidates")
            continue
        rng.shuffle(candidates)
        chosen = candidates[: per_class]
        for p in chosen:
            out.append({"landmark_id": cls, "path": str(p)})
        print(f"[calib] {cls}: picked {len(chosen)}/{len(candidates)}")
    return out


def _preprocess_image(image_path: str, preprocessing: dict) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target = int(preprocessing["image_size"])
    scale = float(preprocessing.get("resize_short_side_scale", 1.15)) * target / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    img = img.crop((left, top, left + target, top + target))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    mean = np.asarray(preprocessing["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(preprocessing["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    return (arr - mean) / std


# ---------- ORT calibration data reader ----------

def _make_calibration_reader(calib_paths: list[str], preprocessing: dict,
                             input_name: str):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(calib_paths)

        def get_next(self):
            try:
                p = next(self._iter)
            except StopIteration:
                return None
            tensor = _preprocess_image(p, preprocessing).astype(np.float32)
            return {input_name: tensor}

        def rewind(self):
            self._iter = iter(calib_paths)

    return _Reader()


# ---------- quantize ----------

def _quantize_static(pre_fp32: Path, out_path: Path, calib_paths: list[str],
                     preprocessing: dict, args: argparse.Namespace,
                     input_name: str) -> dict:
    from onnxruntime.quantization import (
        QuantType, QuantFormat, quantize_static, CalibrationMethod,
    )
    from onnxruntime.quantization import quant_utils

    # Apply the same opset-preserving loader patch as the dynamic script.
    original_loader = quant_utils.load_model_with_shape_infer

    def _opset_preserving_loader(model_path):
        import onnx as _onnx
        source = _onnx.load(str(model_path), load_external_data=False)
        model = original_loader(model_path)
        if not list(model.opset_import):
            del model.opset_import[:]
            model.opset_import.extend(source.opset_import)
        if not any(op.domain in ("", "ai.onnx") for op in model.opset_import):
            for op in source.opset_import:
                if op.domain in ("", "ai.onnx"):
                    model.opset_import.append(op)
                    break
        if not model.ir_version:
            model.ir_version = source.ir_version
        return model

    quant_utils.load_model_with_shape_infer = _opset_preserving_loader

    activation_type = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8,
        "int16": QuantType.QInt16,
    }[args.activation_type]

    calibrate = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[args.calibrate_method]

    nodes_to_exclude: list[str] = []
    if args.exclude_layernorm:
        # ORT exports LayerNorm as Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul chains;
        # Best practical lever is to exclude the surrounding Mul nodes if they
        # produce range issues. Left as a no-op slot for now; users can pass
        # --exclude-layernorm and we will look up names if compile fallback
        # tells us which ones to skip.
        pass

    try:
        reader = _make_calibration_reader(calib_paths, preprocessing, input_name)
        print(f"[quant] static QDQ  weights=INT8 per_channel={args.per_channel}  "
              f"activations={args.activation_type}  "
              f"calibrate={args.calibrate_method}  "
              f"act_symmetric={args.activation_symmetric}  "
              f"calib_size={len(calib_paths)}")
        t0 = time.perf_counter()
        quantize_static(
            model_input=str(pre_fp32),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            calibrate_method=calibrate,
            weight_type=QuantType.QInt8,
            activation_type=activation_type,
            per_channel=args.per_channel,
            reduce_range=False,
            op_types_to_quantize=None,
            nodes_to_exclude=nodes_to_exclude or None,
            use_external_data_format=True,
            extra_options={
                "MatMulConstBOnly": True,
                "ActivationSymmetric": bool(args.activation_symmetric),
                "WeightSymmetric": True,
            },
        )
        elapsed = time.perf_counter() - t0
        print(f"[quant] done in {elapsed:.1f}s -> {out_path}")
        return {
            "elapsed_s": round(elapsed, 1),
            "calibrate_method": args.calibrate_method,
            "quant_format": "QDQ",
            "weight_type": "QInt8",
            "activation_type": args.activation_type,
            "activation_symmetric": bool(args.activation_symmetric),
            "per_channel": bool(args.per_channel),
            "exclude_layernorm": bool(args.exclude_layernorm),
        }
    finally:
        quant_utils.load_model_with_shape_infer = original_loader


# ---------- metadata copy ----------

def _copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    for name in ("preprocessing.json", "labels_master.json", "prototype_index.json"):
        src_file = src_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / name)
            print(f"[copy] {name}")
        else:
            print(f"[skip] {name} (not in source bundle)")


# ---------- main ----------

def main() -> None:
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    data_root = Path(args.data_root)
    src_onnx = src_dir / "landmark_encoder.onnx"
    if not src_onnx.exists():
        sys.exit(f"missing source ONNX: {src_onnx}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    landmark_info = json.loads(Path(args.landmark_info).read_text(encoding="utf-8"))
    classes = [it["landmark_id"] for it in landmark_info["items"]]
    print(f"[info] {len(classes)} classes: {classes}")

    preprocessing = json.loads((src_dir / "preprocessing.json").read_text(encoding="utf-8"))
    input_name = preprocessing.get("input_name", "image")

    src_bytes = src_onnx.stat().st_size + sum(
        p.stat().st_size for p in src_dir.glob("landmark_encoder.onnx*")
        if p.suffix != ".onnx"
    )
    scratch_root = _resolve_scratch_dir(args.temp_dir, src_bytes,
                                        args.min_temp_free_gb)

    import os as _os
    _os.environ["TMPDIR"] = str(scratch_root)
    _os.environ["TEMP"] = str(scratch_root)
    _os.environ["TMP"] = str(scratch_root)
    tempfile.tempdir = str(scratch_root)

    # ---- calibration set
    print(f"\n[calib] selecting {args.per_class} images per class (seed={args.seed})")
    calib_records = _select_calibration_paths(
        data_root, classes, args.per_class,
        REPO_ROOT / "assets" / "hero_images",
        REPO_ROOT / "assets" / "regression_inputs",
        args.seed,
    )
    if not calib_records:
        sys.exit("no calibration images selected")
    calib_paths = [r["path"] for r in calib_records]
    print(f"[calib] total {len(calib_paths)} images")

    # ---- quantize
    with tempfile.TemporaryDirectory(prefix="lm_static_", dir=str(scratch_root)) as tmp:
        work_dir = Path(tmp)
        inline_fp32 = _self_contained_copy(src_onnx, work_dir)
        pre_fp32 = _pre_process(inline_fp32, work_dir)
        out_path = dst_dir / "landmark_encoder.onnx"
        quant_meta = _quantize_static(pre_fp32, out_path, calib_paths,
                                       preprocessing, args, input_name)

    _copy_metadata(src_dir, dst_dir)
    sizes = _bundle_size_mb(dst_dir)

    src_main = src_onnx.stat().st_size / (1024 * 1024)
    src_data = sum(p.stat().st_size for p in src_dir.glob("landmark_encoder.onnx*")
                   if p.suffix != ".onnx") / (1024 * 1024)
    src_total = src_main + src_data

    # ---- write calibration manifest
    calib_doc = {
        "version": "sprint1-calib-v1",
        "seed": args.seed,
        "per_class": args.per_class,
        "n_total": len(calib_records),
        "items": calib_records,
    }
    (dst_dir / "calibration_manifest.json").write_text(
        json.dumps(calib_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "version": "sprint1-mobile-int8-qdq-v1",
        "source_bundle": str(src_dir.resolve()),
        "source_total_mb": round(src_total, 2),
        "quant_settings": {
            "method": "static",
            **quant_meta,
        },
        "calibration": {
            "n": len(calib_records),
            "per_class": args.per_class,
            "manifest_file": str((dst_dir / "calibration_manifest.json").resolve()),
        },
        "destination_onnx": str(out_path.resolve()),
        **sizes,
        "compression_ratio": round(src_total / max(sizes["onnx_total_mb"], 1e-6), 3),
        "notes": [
            "QDQ format is QAIRT-compatible; upload this bundle to AI Hub via "
            "scripts/aihub_run_jobs.py --quant-source prequantized.",
        ],
    }
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
