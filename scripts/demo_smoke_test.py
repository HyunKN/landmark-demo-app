"""Demo app smoke checks for image/text search and detail-page readiness."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from landmark_demo.config import load_config
from landmark_demo.data import load_asset_bundle
from landmark_demo.inference import ImageRecognizer, OnnxImageRecognizer, TextEncoder, assess_image_quality
from landmark_demo.model import load_checkpoint
from landmark_demo.search import ConfidencePolicy, FusionWeights, apply_decision_policy, search_by_image, search_by_text


EXTRA_TEXT_CASES = [
    {"id": "text-mmca-ko-basic", "query": "미술관", "expected_top1": "mmca_seoul"},
    {"id": "text-mmca-en-art-gallery", "query": "art gallery", "expected_top1": "mmca_seoul"},
    {"id": "text-naksan-ko-stone-wall", "query": "돌담 있는 공원", "expected_top1": "naksan_park"},
    {"id": "text-gwanghwamun-en-palace-gate", "query": "palace gate", "expected_top3": ["gwanghwamun", "gyeongbokgung_geunjeongmun"]},
    {"id": "text-palace-ko-broad", "query": "왕이 있던 궁궐", "expected_top3": ["gyeongbokgung_geunjeongmun", "changgyeonggung", "deoksugung"]},
    {"id": "text-cheonggyecheon-en-stream", "query": "stream in Seoul", "expected_top1": "cheonggyecheon"},
]


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_image(hint: str) -> Path | None:
    candidates = [
        REPO_ROOT / hint,
        REPO_ROOT / "assets" / hint,
        Path(hint),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _detail_ready(bundle, landmark_id: str) -> dict:
    info = bundle.info_by_id.get(landmark_id)
    if info is None:
        return {"ok": False, "reason": "missing_info"}
    return {
        "ok": bool(info.name_ko and info.description_ko),
        "name_ko": info.name_ko,
        "has_description": bool(info.description_ko),
        "has_hero": bool(info.hero_image_path and Path(info.hero_image_path).exists()),
        "coordinates_valid": bool(info.coordinates_valid),
    }


def _decision_pass(actual: str, expected) -> bool:
    if expected is None:
        return True
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _top_pass(top_ids: list[str], case: dict) -> bool:
    if "expected_top1" in case:
        return bool(top_ids and top_ids[0] == case["expected_top1"])
    if "expected_primary" in case:
        return case["expected_primary"] in top_ids[:3]
    if "expected_top3" in case:
        return any(item in top_ids[:3] for item in case["expected_top3"])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.toml")
    parser.add_argument("--fixture", default="./tests/fixtures/demo_regression_v1.json")
    parser.add_argument("--output", default="./logs/demo_smoke_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    asset_result = load_asset_bundle(Path(cfg.assets_dir))
    if not asset_result.success or asset_result.bundle is None:
        raise SystemExit(f"asset load failed: {asset_result.errors}")
    bundle = asset_result.bundle
    policy = ConfidencePolicy(**cfg.policy)

    if cfg.inference_backend == "onnx":
        recognizer = OnnxImageRecognizer(cfg.mobile_artifact_dir)
        text_encoder = None
        backend = f"onnx:{cfg.mobile_artifact_dir}"
    else:
        model, _classes, train_cfg = load_checkpoint(cfg.checkpoint, device=cfg.device if cfg.device != "auto" else "cpu")
        recognizer = ImageRecognizer(
            model,
            int(train_cfg["training"]["image_size"]),
            list(train_cfg["training"]["image_mean"]),
            list(train_cfg["training"]["image_std"]),
            device="cpu",
        )
        try:
            import open_clip
            text_encoder = TextEncoder(model, open_clip.get_tokenizer(train_cfg["model"]["model_name"]), device="cpu")
        except Exception:
            text_encoder = None
        backend = "pytorch"

    fixture = _load_fixture(Path(args.fixture))
    image_cases = fixture.get("manual_image_cases", []) + fixture.get("hard_image_cases", [])
    text_cases = fixture.get("text_cases", []) + EXTRA_TEXT_CASES
    image_weights = FusionWeights(**cfg.image_only)
    text_weights = FusionWeights(**cfg.text_only)
    image_weights.validate()
    text_weights.validate()

    report = {
        "version": "sprint1-demo-smoke-v1",
        "config": str(Path(args.config).resolve()),
        "backend": backend,
        "started_at_unix": int(time.time()),
        "image_cases": [],
        "text_cases": [],
        "summary": {},
    }

    for case in image_cases:
        path = _resolve_image(case.get("input_hint", ""))
        if path is None:
            report["image_cases"].append({"id": case.get("id"), "pass": False, "reason": "missing_input"})
            continue
        image = Image.open(path).convert("RGB")
        embedding, elapsed_ms = recognizer.encode(image)
        outcome = search_by_image(embedding, bundle, image_weights, cfg.reject_threshold, policy=policy)
        quality = assess_image_quality(image)
        if not quality.ok:
            outcome = apply_decision_policy(outcome, "image", policy, low_quality=True, quality_reason_codes=quality.reason_codes)
        top_ids = [item.landmark_id for item in outcome.top3]
        forbidden = set(case.get("forbidden_matched", []))
        top_ok = _top_pass(top_ids, case)
        decision_ok = _decision_pass(outcome.decision, case.get("expected_decision"))
        forbidden_ok = not (outcome.decision == "matched" and top_ids and top_ids[0] in forbidden)
        detail = [_detail_ready(bundle, item.landmark_id) for item in outcome.top3]
        report["image_cases"].append({
            "id": case.get("id"),
            "input": str(path),
            "top3": top_ids,
            "decision": outcome.decision,
            "reason_codes": outcome.reason_codes,
            "top1_score": outcome.top1_score,
            "top2_score": outcome.top2_score,
            "margin": outcome.margin,
            "elapsed_ms": elapsed_ms,
            "detail_ready_top3": detail,
            "pass": bool(top_ok and decision_ok and forbidden_ok and all(d["ok"] for d in detail)),
        })

    for case in text_cases:
        query = case["query"]
        embedding = text_encoder.encode(query) if text_encoder is not None else None
        outcome = search_by_text(embedding, query, bundle, text_weights, cfg.reject_threshold, policy=policy)
        top_ids = [item.landmark_id for item in outcome.top3]
        top_ok = _top_pass(top_ids, case)
        decision_ok = _decision_pass(outcome.decision, case.get("expected_decision"))
        detail = [_detail_ready(bundle, item.landmark_id) for item in outcome.top3]
        report["text_cases"].append({
            "id": case.get("id"),
            "query": query,
            "top3": top_ids,
            "decision": outcome.decision,
            "reason_codes": outcome.reason_codes,
            "top1_score": outcome.top1_score,
            "top2_score": outcome.top2_score,
            "margin": outcome.margin,
            "detail_ready_top3": detail,
            "pass": bool(top_ok and decision_ok and all(d["ok"] for d in detail)),
        })

    image_pass = sum(1 for item in report["image_cases"] if item["pass"])
    text_pass = sum(1 for item in report["text_cases"] if item["pass"])
    report["summary"] = {
        "image_pass": image_pass,
        "image_total": len(report["image_cases"]),
        "text_pass": text_pass,
        "text_total": len(report["text_cases"]),
        "all_pass": image_pass == len(report["image_cases"]) and text_pass == len(report["text_cases"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if not report["summary"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
