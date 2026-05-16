"""Compile / profile / infer the INT8 landmark encoder on Qualcomm AI Hub.

Pipeline per device (S23 / S24 / S25 proxy):
  1. submit_compile_job (ONNX -> QNN context binary, NPU/HTP)
  2. submit_profile_job  (latency + compute-unit breakdown)
  3. submit_inference_job (frozen fixture images)

Locally we also encode the same images through the FP32 PC ONNX as ground
truth, so the report can answer:
  - latency on each device (ms)
  - HTP vs CPU/GPU compute-unit split per layer
  - top-1 / top-3 match between device-side INT8 NPU run and PC FP32

Inputs:
  --int8-dir  : mobile_artifacts_int8/ (uploaded as the model)
  --fp32-dir  : mobile_artifacts/      (PC ground-truth ONNX)
  --assets-dir, --regression-fixture: same as quantization_regression.py

Output:
  mobile_artifacts_int8/aihub_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

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


DEVICE_TARGETS = [
    {
        "label": "Snapdragon 8 Gen 2 (Galaxy S23)",
        "name": "Samsung Galaxy S23 (Family)",
        "device_class": "phone",
        "is_proxy": False,
    },
    {
        "label": "Snapdragon 8 Gen 3 (Galaxy S24)",
        "name": "Samsung Galaxy S24 (Family)",
        "device_class": "phone",
        "is_proxy": False,
    },
    {
        "label": "Snapdragon 8 Elite (S25 proxy)",
        "name": "Snapdragon 8 Elite QRD",
        "device_class": "phone",
        "is_proxy": True,
    },
]

COMPILE_OPTIONS = "--target_runtime qnn_context_binary --compute_unit npu"


def _pick_scratch_dir(source_dir: Path) -> str:
    """Choose a temp parent with enough free space for zipping the bundle.

    The default %TEMP% on this machine sits on a small drive, so we probe
    drive roots and prefer the one with the most free space (excluding
    network drives). Falls back to gettempdir().
    """
    import shutil
    import string
    import tempfile as _tf

    needed = 0
    for p in source_dir.glob("landmark_encoder.onnx*"):
        try:
            needed += p.stat().st_size
        except OSError:
            pass
    needed = int(needed * 1.6) + 512 * 1024 * 1024  # zip + headroom

    candidates: list[Path] = []
    candidates.append(Path(_tf.gettempdir()))
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:/")
        if root.exists():
            candidates.append(root)

    best = None
    best_free = -1
    for c in candidates:
        try:
            free = shutil.disk_usage(c).free
        except Exception:
            continue
        if free >= needed and free > best_free:
            best = c
            best_free = free
    if best is None:
        return _tf.gettempdir()
    print(f"[scratch] zipping under {best} ({best_free/(1024**3):.1f} GB free)")
    return str(best)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--int8-dir", default="./mobile_artifacts_int8")
    p.add_argument("--fp32-dir", default="./mobile_artifacts")
    p.add_argument("--assets-dir", default="./assets")
    p.add_argument("--regression-fixture",
                   default="./tests/fixtures/demo_regression_v1.json")
    p.add_argument("--output", default="./mobile_artifacts_int8/aihub_report.json")
    p.add_argument("--max-inputs", type=int, default=15,
                   help="Cap input count to keep the inference job small.")
    p.add_argument("--skip-inference", action="store_true",
                   help="Run only compile + profile (no inference job).")
    p.add_argument("--quant-source", choices=("fp32-aihub", "ort-int8"),
                   default="fp32-aihub",
                   help="fp32-aihub: upload FP32 ONNX and quantize on AI Hub "
                        "(QAIRT-friendly path). ort-int8: upload our pre-built "
                        "ORT dynamic-INT8 model directly (often rejected by "
                        "QAIRT).")
    p.add_argument("--calibration-per-class", type=int, default=20,
                   help="Calibration images per class drawn from "
                        "Dataset/<class>/labels.json (confirmed). 0 disables "
                        "the rich calibration set and falls back to fixture "
                        "inputs (the 1st-attempt behavior, mode-collapse risk).")
    p.add_argument("--calibration-seed", type=int, default=20260516)
    p.add_argument("--data-root",
                   default=str(REPO_ROOT.parent / "Dataset"))
    p.add_argument("--landmark-info",
                   default=str(REPO_ROOT / "assets" / "landmark_info.json"))
    return p.parse_args()


# ---------- preprocessing (mirrors quantization_regression.py) ----------

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


# ---------- search adapter ----------

def _decision_for_embedding(embedding: np.ndarray, bundle, policy: ConfidencePolicy):
    weights = FusionWeights(w_image=1.0, w_text=0.0, w_keyword=0.0)
    outcome = search_by_image(
        embedding.astype(np.float32), bundle, weights,
        reject_threshold=policy.reject_threshold, policy=policy,
    )
    top3_ids = [t.landmark_id for t in outcome.top3]
    return {
        "top3": top3_ids,
        "top1": top3_ids[0] if top3_ids else None,
        "top1_score": float(outcome.top1_score),
        "margin": float(outcome.margin),
        "decision": outcome.decision,
        "reason_codes": list(outcome.reason_codes),
    }


# ---------- input collection ----------

def _build_calibration_set(data_root: Path, landmark_info_path: Path,
                            assets_dir: Path, preprocessing: dict,
                            per_class: int, seed: int):
    """Sample per_class confirmed images per class from Dataset/, excluding
    hero and regression inputs so the inference fixture remains a fair
    holdout.

    Returns (records: list[dict], tensors: list[np.ndarray]).
    """
    import random
    rng = random.Random(seed)

    landmark_info = json.loads(landmark_info_path.read_text(encoding="utf-8"))
    classes = [it["landmark_id"] for it in landmark_info["items"]]

    hero_dir = assets_dir / "hero_images"
    hero_keys = {p.stat().st_size for p in hero_dir.glob("*.jpg")} \
        if hero_dir.exists() else set()
    regression_dir = assets_dir / "regression_inputs"
    regression_basenames: set[str] = set()
    if regression_dir.exists():
        for p in regression_dir.rglob("*"):
            if p.is_file():
                regression_basenames.add(p.name)

    records: list[dict] = []
    tensors: list[np.ndarray] = []
    for cls in classes:
        labels_path = data_root / cls / "labels.json"
        if not labels_path.exists():
            print(f"[calib] {cls}: labels.json missing, skipping")
            continue
        rows = json.loads(labels_path.read_text(encoding="utf-8"))
        candidates: list[Path] = []
        for r in rows:
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
            try:
                img = Image.open(p).convert("RGB")
                arr = preprocess_numpy(img, preprocessing).astype(np.float32)
            except Exception as exc:
                print(f"[calib] skip {p.name}: {exc}")
                continue
            tensors.append(arr)
            records.append({"landmark_id": cls, "path": str(p)})
        print(f"[calib] {cls}: picked {len(chosen)}/{len(candidates)}")
    return records, tensors


def _resolve_fixture_inputs(fixture_path: Path, assets_dir: Path) -> list[dict]:
    spec = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = []
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
        path = None
        for c in candidates:
            try:
                if c.exists():
                    path = c
                    break
            except OSError:
                continue
        if path is None:
            continue
        cases.append({
            "id": case.get("id"),
            "input_hint": hint,
            "expected_top1": case.get("expected_top1"),
            "expected_decision": case.get("expected_decision"),
            "path": path,
        })
    return cases


# ---------- profile parsing ----------

def _summarize_profile(profile: dict) -> dict:
    """Extract the few fields we want from AI Hub's profile JSON.

    AI Hub profile schema is stable but verbose. We pull execution_summary
    fields if present, otherwise scan layer_details for compute-unit counts.
    """
    summary = {
        "estimated_inference_time_us": None,
        "first_load_time_us": None,
        "peak_memory_usage_bytes": None,
        "compute_unit_breakdown": {},
        "layer_count": None,
    }
    exec_summary = profile.get("execution_summary") or {}
    summary["estimated_inference_time_us"] = exec_summary.get("estimated_inference_time")
    summary["first_load_time_us"] = exec_summary.get("first_load_time")
    summary["peak_memory_usage_bytes"] = exec_summary.get("inference_memory_peak_range")
    if isinstance(summary["peak_memory_usage_bytes"], (list, tuple)) and \
            len(summary["peak_memory_usage_bytes"]) == 2:
        # store the upper bound for readability
        summary["peak_memory_usage_bytes"] = int(summary["peak_memory_usage_bytes"][1])

    layer_details = profile.get("execution_detail") or profile.get("layer_details") or []
    breakdown: dict[str, int] = {}
    for layer in layer_details:
        unit = (layer.get("compute_unit") or layer.get("backend") or "UNKNOWN").upper()
        breakdown[unit] = breakdown.get(unit, 0) + 1
    summary["compute_unit_breakdown"] = breakdown
    summary["layer_count"] = sum(breakdown.values()) or None

    # Fallback: use top-level totals if present
    if not summary["estimated_inference_time_us"]:
        for k in ("inference_time", "execution_time"):
            if k in profile:
                summary["estimated_inference_time_us"] = profile[k]
                break

    return summary


# ---------- main ----------

def main() -> None:
    import qai_hub as hub
    import onnxruntime as ort

    args = parse_args()
    int8_dir = Path(args.int8_dir).resolve()
    fp32_dir = Path(args.fp32_dir).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    fixture_path = Path(args.regression_fixture).resolve()
    output_path = Path(args.output).resolve()

    int8_onnx = int8_dir / "landmark_encoder.onnx"
    fp32_onnx = fp32_dir / "landmark_encoder.onnx"
    if not int8_onnx.exists():
        sys.exit(f"missing INT8 ONNX: {int8_onnx}")
    if not fp32_onnx.exists():
        sys.exit(f"missing FP32 ONNX: {fp32_onnx}")

    preprocessing = json.loads((int8_dir / "preprocessing.json").read_text(encoding="utf-8"))
    asset_result = load_asset_bundle(assets_dir)
    if not asset_result.success or asset_result.bundle is None:
        sys.exit(f"asset load failed: {asset_result.errors}")
    bundle = asset_result.bundle
    policy = ConfidencePolicy()

    # ---- 1. resolve fixture images and build PC FP32 ground truth
    cases = _resolve_fixture_inputs(fixture_path, assets_dir)[: args.max_inputs]
    if not cases:
        sys.exit("no fixture inputs resolved")
    print(f"[fixture] {len(cases)} inputs resolved")

    fp_session = ort.InferenceSession(str(fp32_onnx),
                                       providers=["CPUExecutionProvider"])
    fp_in = fp_session.get_inputs()[0].name
    fp_out = fp_session.get_outputs()[0].name

    inputs_arr: list[np.ndarray] = []
    for c in cases:
        image = Image.open(c["path"]).convert("RGB")
        tensor = preprocess_numpy(image, preprocessing).astype(np.float32)
        c["tensor_shape"] = list(tensor.shape)
        emb = fp_session.run([fp_out], {fp_in: tensor})[0][0].astype(np.float32)
        gt = _decision_for_embedding(emb, bundle, policy)
        c["fp32"] = gt
        inputs_arr.append(tensor)
        print(f"[fp32] {c['id']}: top1={gt['top1']} decision={gt['decision']}")

    # ---- 2. upload model
    # AI Hub requires "ONNX model directory" format when external weight files
    # are present, so we ship the entire bundle dir as a zip and let the Hub
    # unpack it on its side.
    import shutil
    import tempfile

    if args.quant_source == "ort-int8":
        source_dir = int8_dir
        bundle_label = "ort_int8"
    else:
        source_dir = fp32_dir
        bundle_label = "fp32"

    scratch_root = _pick_scratch_dir(source_dir)
    # qai-hub also uses tempfile.mkdtemp() / TemporaryDirectory() internally
    # to validate uploaded zips, so route ALL temp activity to the scratch
    # drive: env vars + tempfile module variable.
    import os as _os
    _os.environ["TMPDIR"] = scratch_root
    _os.environ["TEMP"] = scratch_root
    _os.environ["TMP"] = scratch_root
    tempfile.tempdir = scratch_root

    upload_root = Path(tempfile.mkdtemp(prefix="aihub_upload_", dir=scratch_root))
    bundle_root = upload_root / "landmark_encoder"
    bundle_root.mkdir()
    for src in sorted(source_dir.glob("landmark_encoder.onnx*")):
        shutil.copy2(src, bundle_root / src.name)
    zip_base = upload_root / "landmark_encoder"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=upload_root,
                                        base_dir="landmark_encoder"))
    main_mb = (source_dir / "landmark_encoder.onnx").stat().st_size / 1024 / 1024
    zip_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[upload] zipped ONNX dir ({bundle_label}, {main_mb:.1f} MB main + "
          f"external data) -> {zip_path.name} ({zip_mb:.1f} MB)")
    fp_uploaded = hub.upload_model(str(zip_path))
    print(f"[upload] model_id={fp_uploaded.model_id}")

    # ---- 3. upload datasets
    # Two distinct datasets:
    #   - calibration: ~per_class * n_classes images sampled from Dataset/
    #     used by quantize_job to derive activation scales
    #   - inference:   the 15 fixture inputs (positive holdouts + OOS) used
    #     by inference_job to score the device-side model against PC FP32
    inference_payload = {"image": [arr for arr in inputs_arr]}
    inference_dataset = hub.upload_dataset(
        inference_payload, name="demo_regression_v1_inputs")
    print(f"[upload] inference_dataset_id={inference_dataset.dataset_id}  "
          f"n={len(inputs_arr)}")

    calibration_dataset = inference_dataset
    calibration_meta = {"source": "fixture", "n": len(inputs_arr), "per_class": None}
    if args.calibration_per_class and args.calibration_per_class > 0:
        calib_records, calib_arrs = _build_calibration_set(
            data_root=Path(args.data_root),
            landmark_info_path=Path(args.landmark_info),
            assets_dir=assets_dir,
            preprocessing=preprocessing,
            per_class=args.calibration_per_class,
            seed=args.calibration_seed,
        )
        if calib_arrs:
            calibration_dataset = hub.upload_dataset(
                {"image": calib_arrs},
                name=f"calibration_per_class_{args.calibration_per_class}")
            print(f"[upload] calibration_dataset_id={calibration_dataset.dataset_id}"
                  f"  n={len(calib_arrs)}  per_class={args.calibration_per_class}")
            calibration_meta = {
                "source": "dataset_holdout",
                "n": len(calib_arrs),
                "per_class": args.calibration_per_class,
                "items": calib_records,
            }
        else:
            print("[calib] no images sampled; falling back to fixture set")

    dataset = inference_dataset  # back-compat alias used downstream

    # ---- 2b. quantize on AI Hub if requested
    if args.quant_source == "fp32-aihub":
        # AI Hub's quantize_job rejects dynamic shapes, so first run an ONNX
        # compile job that bakes in static (1, 3, 224, 224) and feed THAT model
        # into quantization.
        print("[static-shape] compiling FP32 ONNX into static-shape ONNX")
        static_compile = hub.submit_compile_job(
            model=fp_uploaded,
            device=hub.Device(name=DEVICE_TARGETS[1]["name"]),  # any device works
            input_specs={"image": ((1, 3, 224, 224), "float32")},
            options="--target_runtime onnx",
            name="landmark_encoder_static_shape",
        )
        print(f"[static-shape] job_id={static_compile.job_id}")
        static_compile.wait()
        sstatus = static_compile.get_status()
        sok = bool(getattr(sstatus, "success", False) or sstatus.code == "SUCCESS")
        if not sok:
            print(f"[static-shape] FAILED: {sstatus}")
            sys.exit(1)
        static_model = static_compile.get_target_model()
        print(f"[static-shape] OK -> target_model_id={static_model.model_id}")

        print(f"[quantize] submitting AI Hub quantize_job "
              f"(weights=INT8, activations=INT16)  "
              f"calibration n={calibration_meta['n']}")
        quant_job = hub.submit_quantize_job(
            model=static_model,
            calibration_data=calibration_dataset,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"landmark_encoder_quantize_w8a16_calib{calibration_meta['n']}",
        )
        print(f"[quantize] job_id={quant_job.job_id}")
        quant_job.wait()
        qstatus = quant_job.get_status()
        qok = bool(getattr(qstatus, "success", False) or qstatus.code == "SUCCESS")
        if not qok:
            print(f"[quantize] FAILED: {qstatus}")
            sys.exit(1)
        model_handle = quant_job.get_target_model()
        print(f"[quantize] OK -> target_model_id={model_handle.model_id}")
    else:
        model_handle = fp_uploaded

    devices = []
    for spec in DEVICE_TARGETS:
        try:
            d = hub.Device(name=spec["name"])
            devices.append({"spec": spec, "device": d})
        except Exception as exc:
            print(f"[device] skip {spec['name']}: {exc}")

    # ---- 4. submit compile jobs in parallel
    compile_jobs = {}
    for d in devices:
        spec = d["spec"]
        cj = hub.submit_compile_job(
            model=model_handle,
            device=d["device"],
            input_specs={"image": ((1, 3, 224, 224), "float32")},
            options=COMPILE_OPTIONS,
            name=f"landmark_encoder_int8_{spec['name'].replace(' ', '_')}",
        )
        compile_jobs[spec["name"]] = cj
        print(f"[compile] {spec['name']:38s} job_id={cj.job_id}")

    # ---- 5. wait for compile, then fan out profile + inference
    target_models: dict[str, object] = {}
    compile_summary: dict[str, dict] = {}
    for name, cj in compile_jobs.items():
        print(f"[wait]    compile {name}")
        cj.wait()
        status = cj.get_status()
        ok = bool(getattr(status, "success", False) or status.code == "SUCCESS")
        compile_summary[name] = {
            "job_id": cj.job_id,
            "status": str(status),
            "success": ok,
        }
        if ok:
            target_models[name] = cj.get_target_model()
            print(f"[compile] {name} OK -> target_model_id={target_models[name].model_id}")
        else:
            print(f"[compile] {name} FAILED: {status}")

    profile_jobs = {}
    inference_jobs = {}
    for d in devices:
        name = d["spec"]["name"]
        tm = target_models.get(name)
        if tm is None:
            continue
        pj = hub.submit_profile_job(
            model=tm,
            device=d["device"],
            name=f"landmark_encoder_int8_profile_{name.replace(' ', '_')}",
        )
        profile_jobs[name] = pj
        print(f"[profile] {name:38s} job_id={pj.job_id}")
        if not args.skip_inference:
            ij = hub.submit_inference_job(
                model=tm,
                device=d["device"],
                inputs=dataset,
                name=f"landmark_encoder_int8_infer_{name.replace(' ', '_')}",
            )
            inference_jobs[name] = ij
            print(f"[infer]   {name:38s} job_id={ij.job_id}")

    # ---- 6. collect profile results
    profile_summary: dict[str, dict] = {}
    for name, pj in profile_jobs.items():
        print(f"[wait]    profile {name}")
        pj.wait()
        status = pj.get_status()
        ok = bool(getattr(status, "success", False) or status.code == "SUCCESS")
        record = {
            "job_id": pj.job_id,
            "status": str(status),
            "success": ok,
        }
        if ok:
            try:
                profile = pj.download_profile()
                record["raw"] = profile
                record["summary"] = _summarize_profile(profile)
            except Exception as exc:
                record["download_error"] = str(exc)
        profile_summary[name] = record
        if ok and "summary" in record:
            s = record["summary"]
            ms = (s.get("estimated_inference_time_us") or 0) / 1000.0
            print(f"[profile] {name:38s} latency~{ms:.1f} ms  "
                  f"compute={s.get('compute_unit_breakdown')}")

    # ---- 7. collect inference results
    inference_summary: dict[str, dict] = {}
    for name, ij in inference_jobs.items():
        print(f"[wait]    inference {name}")
        ij.wait()
        status = ij.get_status()
        ok = bool(getattr(status, "success", False) or status.code == "SUCCESS")
        record = {
            "job_id": ij.job_id,
            "status": str(status),
            "success": ok,
            "cases": [],
        }
        if ok:
            try:
                outputs = ij.download_output_data()
            except Exception as exc:
                record["download_error"] = str(exc)
                inference_summary[name] = record
                continue
            # outputs is a dict {output_name: list_of_ndarray}
            output_name = next(iter(outputs))
            embeddings = outputs[output_name]
            top1_match = 0
            top3_match = 0
            decision_match = 0
            for case, emb in zip(cases, embeddings):
                emb_arr = np.asarray(emb).reshape(-1).astype(np.float32)
                # If output is L2-normalized already, reshape is enough; else
                # normalize to be robust against mixed-precision artifacts.
                norm = float(np.linalg.norm(emb_arr))
                if norm > 0:
                    emb_arr = emb_arr / norm
                eval_dev = _decision_for_embedding(emb_arr, bundle, policy)
                t1 = case["fp32"]["top1"] == eval_dev["top1"]
                t3 = case["fp32"]["top3"] == eval_dev["top3"]
                dec = case["fp32"]["decision"] == eval_dev["decision"]
                top1_match += int(t1)
                top3_match += int(t3)
                decision_match += int(dec)
                record["cases"].append({
                    "id": case["id"],
                    "expected_top1": case["expected_top1"],
                    "fp32_top1": case["fp32"]["top1"],
                    "fp32_top3": case["fp32"]["top3"],
                    "fp32_decision": case["fp32"]["decision"],
                    "device_top1": eval_dev["top1"],
                    "device_top3": eval_dev["top3"],
                    "device_decision": eval_dev["decision"],
                    "top1_match_vs_fp32": t1,
                    "top3_match_vs_fp32": t3,
                    "decision_match_vs_fp32": dec,
                })
            n = len(record["cases"])
            record["summary"] = {
                "n": n,
                "top1_match_rate_vs_fp32": top1_match / n if n else None,
                "top3_match_rate_vs_fp32": top3_match / n if n else None,
                "decision_match_rate_vs_fp32": decision_match / n if n else None,
            }
            print(f"[infer]   {name:38s} top1_match={top1_match}/{n} "
                  f"top3_match={top3_match}/{n} decision={decision_match}/{n}")
        inference_summary[name] = record

    report = {
        "version": "sprint1-aihub-int8-v2",
        "policy": {
            "reject_threshold": policy.reject_threshold,
            "weak_reject_threshold": policy.weak_reject_threshold,
            "weak_margin": policy.weak_margin,
            "match_threshold": policy.match_threshold,
            "match_floor": policy.match_floor,
            "match_margin": policy.match_margin,
        },
        "compile_options": COMPILE_OPTIONS,
        "model": {
            "int8_onnx": str(int8_onnx),
            "uploaded_model_id": model_handle.model_id,
            "quant_source": args.quant_source,
        },
        "calibration": {
            **calibration_meta,
            "dataset_id": calibration_dataset.dataset_id,
        },
        "inference_dataset_id": inference_dataset.dataset_id,
        "devices": [
            {
                "label": d["spec"]["label"],
                "name": d["spec"]["name"],
                "is_proxy": d["spec"]["is_proxy"],
                "compile": compile_summary.get(d["spec"]["name"]),
                "profile": profile_summary.get(d["spec"]["name"]),
                "inference": inference_summary.get(d["spec"]["name"]),
            }
            for d in devices
        ],
        "fixture_inputs": [
            {
                "id": c["id"],
                "input_hint": c["input_hint"],
                "expected_top1": c["expected_top1"],
                "expected_decision": c["expected_decision"],
                "fp32": c["fp32"],
            }
            for c in cases
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n[done] report -> {output_path}")

    # Console digest
    print("\n=== DIGEST ===")
    for d in devices:
        name = d["spec"]["name"]
        prof = profile_summary.get(name, {}).get("summary") or {}
        inf = inference_summary.get(name, {}).get("summary") or {}
        ms = (prof.get("estimated_inference_time_us") or 0) / 1000.0
        breakdown = prof.get("compute_unit_breakdown") or {}
        print(f"  {d['spec']['label']:38s} latency~{ms:6.2f} ms  "
              f"top1_vs_fp32={inf.get('top1_match_rate_vs_fp32')}  "
              f"compute={breakdown}")


if __name__ == "__main__":
    main()
