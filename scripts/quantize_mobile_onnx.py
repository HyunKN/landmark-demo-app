"""Quantize Sprint 1 image encoder ONNX to weight-only INT8.

Strategy:
  - Load FP32 ONNX (with external data) into a single self-contained file.
  - Run shape inference (quant_pre_process) for ViT-friendly graph cleanup.
  - Apply weight-only dynamic quantization with per-channel scales.
  - Emit INT8 artifact bundle parallel to mobile_artifacts/.

Usage:
    python scripts/quantize_mobile_onnx.py \
        --src ./mobile_artifacts \
        --dst ./mobile_artifacts_int8

Outputs in --dst:
    landmark_encoder.onnx              (INT8, may use .data when needed)
    preprocessing.json                 (copied as-is; same input layout)
    labels_master.json                 (copied as-is)
    prototype_index.json               (copied as-is by default; rerun with
                                        --reencode-prototypes for INT8 prototypes)
    manifest.json                      (records source + quant settings)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="./mobile_artifacts",
                   help="Source bundle dir produced by export_mobile_onnx.py")
    p.add_argument("--dst", default="./mobile_artifacts_int8",
                   help="Destination bundle dir for INT8 artifacts")
    p.add_argument("--per-channel", action="store_true", default=True,
                   help="Per-channel weight quantization (better for ViT MatMul)")
    p.add_argument("--reduce-range", action="store_true", default=False,
                   help="Use 7-bit weights for older CPUs without VNNI")
    p.add_argument("--op-types", nargs="+",
                   default=["MatMul", "Gemm"],
                   help="Op types to quantize. ViT is MatMul-heavy.")
    p.add_argument("--temp-dir", default=None,
                   help="Override scratch dir. Quantization writes ~3-4x the "
                        "FP32 model size in temporaries; default %%TEMP%% on "
                        "Windows often points at C: which can be tight.")
    p.add_argument("--min-temp-free-gb", type=float, default=8.0,
                   help="Minimum free space (GB) required on the chosen scratch "
                        "drive. Quantization aborts early if not satisfied.")
    return p.parse_args()


def _resolve_scratch_dir(requested: str | None, model_bytes: int,
                         min_free_gb: float) -> Path:
    """Pick a scratch dir that has enough free space for ORT's temporaries.

    Strategy:
      1. If --temp-dir is given and has room, use it.
      2. Otherwise probe %TEMP%, the destination drive, the source drive, and
         common Windows roots; pick the first with > model_size * 4 + min_free.
      3. Fall back to %TEMP% with a warning.
    """
    import shutil
    headroom = max(model_bytes * 4, int(min_free_gb * 1024 ** 3))

    def fits(p: Path) -> bool:
        try:
            return shutil.disk_usage(p).free >= headroom
        except Exception:
            return False

    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested))
    import tempfile as _tf
    candidates.append(Path(_tf.gettempdir()))

    # Probe drive roots that exist
    import string
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:/")
        if root.exists():
            candidates.append(root)

    seen: set[str] = set()
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        key = str(cand.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        if fits(cand):
            free_gb = shutil.disk_usage(cand).free / (1024 ** 3)
            print(f"[scratch] using {cand} ({free_gb:.1f} GB free, "
                  f"need >= {headroom/(1024**3):.1f} GB)")
            return cand

    fallback = Path(_tf.gettempdir())
    print(f"[scratch] WARNING: no drive met the headroom requirement; "
          f"falling back to {fallback}. Quantization may fail with ENOSPC.")
    return fallback


def _self_contained_copy(src_onnx: Path, work_dir: Path) -> Path:
    """Load model with external data and write a single self-contained copy.

    quant_pre_process / quantize_dynamic behave more reliably when the input
    model carries its weights inline rather than referencing sibling .data files.
    """
    print(f"[load] {src_onnx} (with external data)")
    model = onnx.load(str(src_onnx), load_external_data=True)
    inline_path = work_dir / "fp32_inline.onnx"
    print(f"[write] inline -> {inline_path}")
    onnx.save(model, str(inline_path), save_as_external_data=False)
    return inline_path


def _pre_process(inline_fp32: Path, work_dir: Path) -> Path:
    """Prepare model for ORT's strict internal shape inference.

    ViT-style graphs often carry stale value_info entries that conflict with
    re-inferred shapes (e.g. (4096) vs (768) for attention reshapes). ORT's
    quantize_dynamic re-runs ONNX shape inference in strict mode and fails on
    those. We clear value_info, then re-infer with strict_mode=False so the
    saved model has consistent shapes downstream.
    """
    import onnx
    from onnx import shape_inference

    pre_path = work_dir / "fp32_preprocessed.onnx"
    print(f"[pre]  load + strip value_info")
    model = onnx.load(str(inline_fp32))
    del model.graph.value_info[:]

    try:
        # In-memory inference preserves opset_import / IR version which the
        # path-based variant can drop on some onnx versions.
        inferred = shape_inference.infer_shapes(
            model, check_type=False, strict_mode=False, data_prop=True,
        )
        print(f"[pre]  shape inference (non-strict, in-memory) ok")
        onnx.save(inferred, str(pre_path))
    except Exception as exc:
        print(f"[pre]  non-strict shape inference failed ({exc}); using stripped model")
        onnx.save(model, str(pre_path))
    return pre_path


def _quantize(pre_fp32: Path, dst_dir: Path, args: argparse.Namespace) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization import quant_utils

    # ORT 1.24's load_model_with_shape_infer rewrites the model through
    # onnx.shape_inference.infer_shapes_path which strips opset_import on
    # some pytorch-exported ViT graphs. We patch it to preserve the opset
    # entries from the source model.
    original_loader = quant_utils.load_model_with_shape_infer

    def _opset_preserving_loader(model_path):
        import onnx as _onnx
        source = _onnx.load(str(model_path), load_external_data=False)
        model = original_loader(model_path)
        if not list(model.opset_import):
            del model.opset_import[:]
            model.opset_import.extend(source.opset_import)
        if not any(op.domain == "ai.onnx" or op.domain == ""
                   for op in model.opset_import):
            for op in source.opset_import:
                if op.domain in ("", "ai.onnx"):
                    model.opset_import.append(op)
                    break
        if not model.ir_version:
            model.ir_version = source.ir_version
        return model

    quant_utils.load_model_with_shape_infer = _opset_preserving_loader

    out_path = dst_dir / "landmark_encoder.onnx"
    print(f"[quant] dynamic INT8 weight-only "
          f"per_channel={args.per_channel} reduce_range={args.reduce_range} "
          f"op_types={args.op_types}")
    t0 = time.perf_counter()
    try:
        quantize_dynamic(
            model_input=str(pre_fp32),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            op_types_to_quantize=args.op_types,
            use_external_data_format=True,
        )
    finally:
        quant_utils.load_model_with_shape_infer = original_loader
    elapsed = time.perf_counter() - t0
    print(f"[quant] done in {elapsed:.1f}s -> {out_path}")
    return out_path


def _copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    for name in ("preprocessing.json", "labels_master.json", "prototype_index.json"):
        src_file = src_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / name)
            print(f"[copy] {name}")
        else:
            print(f"[skip] {name} (not in source bundle)")


def _bundle_size_mb(dst_dir: Path) -> dict[str, float]:
    main = dst_dir / "landmark_encoder.onnx"
    main_mb = main.stat().st_size / (1024 * 1024) if main.exists() else 0.0
    data_mb = sum(p.stat().st_size for p in dst_dir.glob("landmark_encoder.onnx*")
                  if p.suffix != ".onnx") / (1024 * 1024)
    return {
        "onnx_file_mb": round(main_mb, 2),
        "onnx_external_data_mb": round(data_mb, 2),
        "onnx_total_mb": round(main_mb + data_mb, 2),
    }


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    src_onnx = src_dir / "landmark_encoder.onnx"
    if not src_onnx.exists():
        sys.exit(f"missing source ONNX: {src_onnx}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_bytes = src_onnx.stat().st_size + sum(
        p.stat().st_size for p in src_dir.glob("landmark_encoder.onnx*")
        if p.suffix != ".onnx"
    )
    scratch_root = _resolve_scratch_dir(args.temp_dir, src_bytes,
                                        args.min_temp_free_gb)
    # Force ORT's quantize_dynamic (which uses tempfile.TemporaryDirectory
    # internally) onto the scratch drive too.
    import os as _os
    _os.environ["TMPDIR"] = str(scratch_root)
    _os.environ["TEMP"] = str(scratch_root)
    _os.environ["TMP"] = str(scratch_root)
    tempfile.tempdir = str(scratch_root)

    with tempfile.TemporaryDirectory(prefix="lm_quant_", dir=str(scratch_root)) as tmp:
        work_dir = Path(tmp)
        inline_fp32 = _self_contained_copy(src_onnx, work_dir)
        pre_fp32 = _pre_process(inline_fp32, work_dir)
        out_path = _quantize(pre_fp32, dst_dir, args)

    _copy_metadata(src_dir, dst_dir)
    sizes = _bundle_size_mb(dst_dir)

    src_main = src_onnx.stat().st_size / (1024 * 1024)
    src_data = sum(p.stat().st_size for p in src_dir.glob("landmark_encoder.onnx*")
                   if p.suffix != ".onnx") / (1024 * 1024)
    src_total = src_main + src_data

    manifest = {
        "version": "sprint1-mobile-int8-v1",
        "source_bundle": str(src_dir.resolve()),
        "source_total_mb": round(src_total, 2),
        "quant_settings": {
            "method": "dynamic",
            "weight_type": "QInt8",
            "per_channel": bool(args.per_channel),
            "reduce_range": bool(args.reduce_range),
            "op_types_to_quantize": list(args.op_types),
        },
        "destination_onnx": str(out_path.resolve()),
        **sizes,
        "compression_ratio": round(src_total / max(sizes["onnx_total_mb"], 1e-6), 3),
        "notes": [
            "prototype_index.json is copied from the FP32 bundle; rebuild with "
            "build_assets.py against the INT8 model only if cosine drift is "
            "material in the regression report.",
        ],
    }
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
