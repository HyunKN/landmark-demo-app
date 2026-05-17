"""FP32 vs INT8 ONNX regression for the Sprint 1 image encoder.

Compares two ONNX bundles (typically mobile_artifacts/ and
mobile_artifacts_int8/) on a shared image set and emits a single report:

  - per-image: cosine(emb_fp, emb_q), top-3 lists, decisions, scores, latencies
  - aggregate: top-1/top-3/decision agreement, mean |Δtop1|, mean |Δmargin|,
               mean embedding cosine, latency stats, bundle sizes
  - regression: pass rate against tests/fixtures/demo_regression_v1.json
                manual_image_cases (expected_top1 / expected_decision)

Usage:
    python scripts/quantization_regression.py \
        --fp32-dir ./mobile_artifacts \
        --int8-dir ./mobile_artifacts_int8 \
        --assets-dir ./assets \
        --warmup 1 --runs 5

Inputs default to assets/hero_images/*.jpg plus any manual_image_cases that
resolve to a file under assets/ or the repo root.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from landmark_demo.data import load_asset_bundle
from landmark_demo.search import (
    ConfidencePolicy,
    FusionWeights,
    search_by_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-dir", default="./mobile_artifacts")
    p.add_argument("--int8-dir", default="./mobile_artifacts_int8")
    p.add_argument("--assets-dir", default="./assets")
    p.add_argument("--regression-fixture",
                   default="./tests/fixtures/demo_regression_v1.json")
    p.add_argument("--image", action="append", default=[],
                   help="Extra image paths to evaluate (repeatable)")
    p.add_argument("--runs", type=int, default=5,
                   help="Warm latency runs per image per session")
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed warmup runs per session")
    p.add_argument("--output",
                   default="./mobile_artifacts_int8/quantization_regression_report.json")
    return p.parse_args()


# ---------- preprocessing ----------

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


# ---------- session helpers ----------

def _load_session(onnx_path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _run(session, tensor: np.ndarray) -> np.ndarray:
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    return session.run([out], {inp: tensor})[0][0].astype(np.float32)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2(a), _l2(b)))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def _bundle_size_mb(dir_path: Path) -> dict[str, float]:
    main = dir_path / "landmark_encoder.onnx"
    main_mb = main.stat().st_size / (1024 * 1024) if main.exists() else 0.0
    data_mb = sum(p.stat().st_size for p in dir_path.glob("landmark_encoder.onnx*")
                  if p.suffix != ".onnx") / (1024 * 1024)
    return {
        "onnx_file_mb": round(main_mb, 2),
        "onnx_external_data_mb": round(data_mb, 2),
        "onnx_total_mb": round(main_mb + data_mb, 2),
    }


# ---------- search adapter ----------

def _decision_for_embedding(embedding: np.ndarray, bundle, policy: ConfidencePolicy):
    weights = FusionWeights(w_image=1.0, w_text=0.0, w_keyword=0.0)
    outcome = search_by_image(
        embedding.astype(np.float32), bundle, weights,
        reject_threshold=policy.reject_threshold, policy=policy,
    )
    top3_ids = [t.landmark_id for t in outcome.top3]
    top1_score = float(outcome.top1_score)
    margin = float(outcome.margin)
    return {
        "top3": top3_ids,
        "top1": top3_ids[0] if top3_ids else None,
        "top1_score": top1_score,
        "margin": margin,
        "decision": outcome.decision,
        "reason_codes": list(outcome.reason_codes),
    }


# ---------- input collection ----------

def _resolve_manual_inputs(fixture_path: Path, assets_dir: Path) -> list[Path]:
    if not fixture_path.exists():
        return []
    spec = json.loads(fixture_path.read_text(encoding="utf-8"))
    out: list[Path] = []
    for case in spec.get("manual_image_cases", []):
        hint = case.get("input_hint")
        if not hint:
            continue
        candidates = [
            REPO_ROOT / hint,
            assets_dir / hint,
            assets_dir / "hero_images" / hint,
            REPO_ROOT.parent / hint,
            Path(hint),
        ]
        for c in candidates:
            try:
                if c.exists():
                    out.append(c)
                    break
            except OSError:
                continue
    return out


def _collect_inputs(args, assets_dir: Path) -> list[Path]:
    paths: list[Path] = [Path(p) for p in args.image]
    paths += _resolve_manual_inputs(Path(args.regression_fixture), assets_dir)
    if not paths:
        # Prefer the frozen regression set when present so that the report's
        # fixture pass rate is not empty.
        frozen = sorted((assets_dir / "regression_inputs").rglob("*"))
        frozen = [p for p in frozen if p.is_file() and p.suffix.lower()
                  in {".jpg", ".jpeg", ".png", ".webp"}]
        if frozen:
            paths = frozen
        else:
            paths = sorted((assets_dir / "hero_images").glob("*.jpg"))
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


# ---------- regression scoring ----------

def _evaluate_fixture(fixture_path: Path, per_image_records: list[dict]) -> dict:
    if not fixture_path.exists():
        return {"available": False}
    spec = json.loads(fixture_path.read_text(encoding="utf-8"))
    by_hint = {}
    for r in per_image_records:
        path = Path(r["image"])
        # Index by basename and by relative-to-repo path so fixtures can use
        # either form.
        by_hint[path.name] = r
        try:
            rel = str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
            by_hint[rel] = r
        except ValueError:
            pass
    results = {"fp32": [], "int8": []}
    for case in spec.get("manual_image_cases", []):
        hint = case.get("input_hint")
        rec = by_hint.get(hint) or by_hint.get(Path(hint).name) if hint else None
        if rec is None:
            continue
        for branch in ("fp32", "int8"):
            r = rec[branch]
            ok_top1 = (case.get("expected_top1") in (None, r["top1"]))
            expected_decision = case.get("expected_decision")
            if expected_decision is None:
                ok_decision = True
            elif isinstance(expected_decision, list):
                ok_decision = r["decision"] in expected_decision
            else:
                ok_decision = r["decision"] == expected_decision
            results[branch].append({
                "id": case.get("id"),
                "input_hint": hint,
                "expected_top1": case.get("expected_top1"),
                "expected_decision": expected_decision,
                "actual_top1": r["top1"],
                "actual_decision": r["decision"],
                "top1_pass": bool(ok_top1),
                "decision_pass": bool(ok_decision),
            })
    summary = {}
    for branch in ("fp32", "int8"):
        cases = results[branch]
        evaluated_top1 = [c for c in cases if c["expected_top1"] is not None]
        summary[branch] = {
            "n_cases": len(cases),
            "top1_pass_rate": (sum(1 for c in evaluated_top1 if c["top1_pass"])
                               / len(evaluated_top1)) if evaluated_top1 else None,
            "decision_pass_rate": (sum(1 for c in cases if c["decision_pass"])
                                   / len(cases)) if cases else None,
        }
    return {"available": True, "summary": summary, "cases": results}


# ---------- main ----------

def main() -> None:
    args = parse_args()
    fp32_dir = Path(args.fp32_dir)
    int8_dir = Path(args.int8_dir)
    assets_dir = Path(args.assets_dir)
    fp32_onnx = fp32_dir / "landmark_encoder.onnx"
    int8_onnx = int8_dir / "landmark_encoder.onnx"
    preprocessing = json.loads((fp32_dir / "preprocessing.json").read_text(encoding="utf-8"))

    if not fp32_onnx.exists():
        sys.exit(f"missing FP32 ONNX: {fp32_onnx}")
    if not int8_onnx.exists():
        sys.exit(f"missing INT8 ONNX: {int8_onnx} (run scripts/quantize_mobile_onnx.py first)")

    asset_result = load_asset_bundle(assets_dir)
    if not asset_result.success or asset_result.bundle is None:
        sys.exit(f"asset load failed: {asset_result.errors}")
    bundle = asset_result.bundle
    policy = ConfidencePolicy()

    fp_session = _load_session(fp32_onnx)
    q_session = _load_session(int8_onnx)

    inputs = _collect_inputs(args, assets_dir)
    if not inputs:
        sys.exit("no input images resolved (assets/hero_images/*.jpg empty?)")
    print(f"[run] {len(inputs)} inputs, runs={args.runs}, warmup={args.warmup}")

    records: list[dict] = []
    fp_warm: list[float] = []
    q_warm: list[float] = []
    fp_cold: list[float] = []
    q_cold: list[float] = []

    for image_path in inputs:
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess_numpy(image, preprocessing)

        # Cold (first hit each session per image)
        t0 = time.perf_counter()
        fp_emb = _run(fp_session, tensor)
        fp_cold_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        q_emb = _run(q_session, tensor)
        q_cold_ms = (time.perf_counter() - t0) * 1000.0

        # Warmup
        for _ in range(args.warmup):
            _run(fp_session, tensor)
            _run(q_session, tensor)

        per_fp: list[float] = []
        per_q: list[float] = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            fp_emb = _run(fp_session, tensor)
            per_fp.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            q_emb = _run(q_session, tensor)
            per_q.append((time.perf_counter() - t0) * 1000.0)

        fp_cold.append(fp_cold_ms)
        q_cold.append(q_cold_ms)
        fp_warm.extend(per_fp)
        q_warm.extend(per_q)

        fp_eval = _decision_for_embedding(fp_emb, bundle, policy)
        q_eval = _decision_for_embedding(q_emb, bundle, policy)

        records.append({
            "image": str(image_path),
            "embedding_cosine": _cosine(fp_emb, q_emb),
            "embedding_l2_diff": float(np.linalg.norm(_l2(fp_emb) - _l2(q_emb))),
            "fp32": {
                **fp_eval,
                "warm_median_ms": round(statistics.median(per_fp), 2),
                "warm_p90_ms": round(_percentile(per_fp, 0.9), 2),
                "cold_ms": round(fp_cold_ms, 2),
            },
            "int8": {
                **q_eval,
                "warm_median_ms": round(statistics.median(per_q), 2),
                "warm_p90_ms": round(_percentile(per_q, 0.9), 2),
                "cold_ms": round(q_cold_ms, 2),
            },
            "delta": {
                "top1_score": round(q_eval["top1_score"] - fp_eval["top1_score"], 5),
                "margin": round(q_eval["margin"] - fp_eval["margin"], 5),
            },
            "agreement": {
                "top1": fp_eval["top1"] == q_eval["top1"],
                "top3_exact": fp_eval["top3"] == q_eval["top3"],
                "top3_set": set(fp_eval["top3"]) == set(q_eval["top3"]),
                "decision": fp_eval["decision"] == q_eval["decision"],
            },
        })

    # ---------- aggregate ----------
    n = len(records)
    agg = {
        "n_images": n,
        "top1_agreement_rate": sum(r["agreement"]["top1"] for r in records) / n,
        "top3_exact_match_rate": sum(r["agreement"]["top3_exact"] for r in records) / n,
        "top3_set_match_rate": sum(r["agreement"]["top3_set"] for r in records) / n,
        "decision_agreement_rate": sum(r["agreement"]["decision"] for r in records) / n,
        "embedding_cosine_mean": round(
            statistics.mean(r["embedding_cosine"] for r in records), 5),
        "embedding_cosine_min": round(
            min(r["embedding_cosine"] for r in records), 5),
        "delta_top1_score_abs_mean": round(
            statistics.mean(abs(r["delta"]["top1_score"]) for r in records), 5),
        "delta_margin_abs_mean": round(
            statistics.mean(abs(r["delta"]["margin"]) for r in records), 5),
    }
    latency = {
        "fp32_warm_median_ms": round(statistics.median(fp_warm), 2),
        "int8_warm_median_ms": round(statistics.median(q_warm), 2),
        "fp32_warm_p90_ms": round(_percentile(fp_warm, 0.9), 2),
        "int8_warm_p90_ms": round(_percentile(q_warm, 0.9), 2),
        "fp32_cold_median_ms": round(statistics.median(fp_cold), 2),
        "int8_cold_median_ms": round(statistics.median(q_cold), 2),
        "speedup_warm_x": round(
            statistics.median(fp_warm) / max(statistics.median(q_warm), 1e-6), 3),
    }
    sizes = {
        "fp32": _bundle_size_mb(fp32_dir),
        "int8": _bundle_size_mb(int8_dir),
    }
    sizes["compression_ratio"] = round(
        max(sizes["fp32"]["onnx_total_mb"], 1e-6)
        / max(sizes["int8"]["onnx_total_mb"], 1e-6), 3)

    fixture_eval = _evaluate_fixture(Path(args.regression_fixture), records)

    report = {
        "version": "sprint1-quant-regression-v1",
        "fp32_bundle": str(fp32_dir.resolve()),
        "int8_bundle": str(int8_dir.resolve()),
        "policy": {
            "reject_threshold": policy.reject_threshold,
            "weak_reject_threshold": policy.weak_reject_threshold,
            "weak_margin": policy.weak_margin,
            "match_threshold": policy.match_threshold,
            "match_floor": policy.match_floor,
            "match_margin": policy.match_margin,
            "isolated_match_threshold": policy.isolated_match_threshold,
            "isolated_match_margin": policy.isolated_match_margin,
            "isolated_match_top2_max": policy.isolated_match_top2_max,
        },
        "aggregate": agg,
        "latency": latency,
        "sizes_mb": sizes,
        "fixture_regression": fixture_eval,
        "records": records,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    print(f"[done] full report -> {output}")


if __name__ == "__main__":
    main()
