"""Benchmark PyTorch vs ONNX Runtime image encoder handoff package."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmark_demo.data import load_asset_bundle
from landmark_demo.inference import ImageRecognizer
from landmark_demo.model import load_checkpoint
from landmark_demo.search import ConfidencePolicy, FusionWeights, search_by_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./best.pt")
    parser.add_argument("--assets-dir", default="./assets")
    parser.add_argument("--artifact-dir", default="./mobile_artifacts")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="./mobile_artifacts/benchmark_report.json")
    return parser.parse_args()


def preprocess_numpy(image: Image.Image, preprocessing: dict) -> np.ndarray:
    img = image.convert("RGB")
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


def top3_for_embedding(embedding: np.ndarray, bundle, policy: ConfidencePolicy) -> list[str]:
    weights = FusionWeights(w_image=1.0, w_text=0.0, w_keyword=0.0)
    outcome = search_by_image(embedding.astype(np.float32), bundle, weights, reject_threshold=policy.reject_threshold, policy=policy)
    return [item.landmark_id for item in outcome.top3]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def current_process_memory_mb() -> dict[str, float]:
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("Kernel32.dll")
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {
                "working_set_mb": round(counters.WorkingSetSize / (1024 * 1024), 2),
                "peak_working_set_mb": round(counters.PeakWorkingSetSize / (1024 * 1024), 2),
            }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_kb = usage.ru_maxrss
        if sys.platform == "darwin":
            peak_mb = peak_kb / (1024 * 1024)
        else:
            peak_mb = peak_kb / 1024
        return {"peak_working_set_mb": round(peak_mb, 2)}
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    onnx_path = artifact_dir / "landmark_encoder.onnx"
    preprocessing_path = artifact_dir / "preprocessing.json"
    if not onnx_path.exists():
        raise SystemExit(f"missing ONNX artifact: {onnx_path}")
    if not preprocessing_path.exists():
        raise SystemExit(f"missing preprocessing metadata: {preprocessing_path}")

    import onnxruntime as ort

    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    asset_result = load_asset_bundle(Path(args.assets_dir))
    if not asset_result.success or asset_result.bundle is None:
        raise SystemExit(f"asset load failed: {asset_result.errors}")
    bundle = asset_result.bundle
    policy = ConfidencePolicy()

    model, _classes, train_cfg = load_checkpoint(args.checkpoint, device="cpu")
    recognizer = ImageRecognizer(
        model,
        int(train_cfg["training"]["image_size"]),
        list(train_cfg["training"]["image_mean"]),
        list(train_cfg["training"]["image_std"]),
        device="cpu",
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    memory_after_session = current_process_memory_mb()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_paths = [Path(p) for p in args.image]
    if not image_paths:
        image_paths = sorted((Path(args.assets_dir) / "hero_images").glob("*.jpg"))[:4]

    records = []
    cold_times = []
    warm_times = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        pt_embedding, pt_ms = recognizer.encode(image)
        tensor = preprocess_numpy(image, preprocessing)

        t0 = time.perf_counter()
        ort_embedding = session.run([output_name], {input_name: tensor})[0][0].astype(np.float32)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        cold_times.append(cold_ms)

        per_image_warm = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            ort_embedding = session.run([output_name], {input_name: tensor})[0][0].astype(np.float32)
            per_image_warm.append((time.perf_counter() - t0) * 1000.0)
        warm_times.extend(per_image_warm)

        cosine = float(np.dot(pt_embedding, ort_embedding) / (np.linalg.norm(pt_embedding) * np.linalg.norm(ort_embedding) + 1e-12))
        records.append({
            "image": str(image_path),
            "pytorch_ms": pt_ms,
            "onnx_cold_ms": round(cold_ms, 2),
            "onnx_warm_median_ms": round(statistics.median(per_image_warm), 2),
            "onnx_warm_p90_ms": round(percentile(per_image_warm, 0.9), 2),
            "embedding_cosine": cosine,
            "pytorch_top3": top3_for_embedding(pt_embedding, bundle, policy),
            "onnx_top3": top3_for_embedding(ort_embedding, bundle, policy),
        })

    report = {
        "artifact_dir": str(artifact_dir.resolve()),
        "onnx_file_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "onnx_external_data_mb": round(sum(p.stat().st_size for p in artifact_dir.glob("landmark_encoder.onnx.data*")) / (1024 * 1024), 2),
        "onnx_total_mb": round((onnx_path.stat().st_size + sum(p.stat().st_size for p in artifact_dir.glob("landmark_encoder.onnx.data*"))) / (1024 * 1024), 2),
        "providers": session.get_providers(),
        "process_memory_after_session": memory_after_session,
        "process_memory_after_benchmark": current_process_memory_mb(),
        "pid": os.getpid(),
        "runs_per_image": args.runs,
        "cold_median_ms": round(statistics.median(cold_times), 2) if cold_times else 0.0,
        "warm_median_ms": round(statistics.median(warm_times), 2) if warm_times else 0.0,
        "warm_p90_ms": round(percentile(warm_times, 0.9), 2) if warm_times else 0.0,
        "top3_exact_match_rate": sum(1 for r in records if r["pytorch_top3"] == r["onnx_top3"]) / max(len(records), 1),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
