"""Streamlit entry point. 단일 페이지에서 이미지/텍스트 검색 + 정보 페이지를 라우팅."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import streamlit as st

# 패키지 자체를 임포트 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landmark_demo.config import load_config
from landmark_demo.data import LANDMARK_CATALOG, load_asset_bundle
from landmark_demo.inference import ImageRecognizer, OnnxImageRecognizer, OnnxTextEncoder, TextEncoder, assess_image_quality, validate_image_file
from landmark_demo.logging_util import DebugLogger
from landmark_demo.model import load_checkpoint
from landmark_demo.search import (
    ConfidencePolicy,
    FusionWeights,
    apply_decision_policy,
    search_by_image,
    search_by_text,
)


@st.cache_resource(show_spinner="모델과 자산을 로드하고 있습니다 ...")
def boot(config_path: str):
    cfg = load_config(config_path)
    asset_dir = Path(cfg.assets_dir).resolve()
    asset_result = load_asset_bundle(asset_dir)

    if not asset_result.success or asset_result.bundle is None:
        return {"ok": False, "errors": asset_result.errors, "config": cfg, "asset_dir": asset_dir}

    # 모델 로드
    import torch
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    classes = list(LANDMARK_CATALOG)
    train_cfg = None
    if cfg.inference_backend == "onnx":
        recognizer = OnnxImageRecognizer(cfg.mobile_artifact_dir)
        device = "onnxruntime-cpu"
    else:
        model, classes, train_cfg = load_checkpoint(cfg.checkpoint, device=device)
        image_size = int(train_cfg["training"]["image_size"])
        image_mean = list(train_cfg["training"]["image_mean"])
        image_std = list(train_cfg["training"]["image_std"])
        recognizer = ImageRecognizer(model, image_size, image_mean, image_std, device=device)

    # Text encoder. PyTorch mode reuses the loaded checkpoint's text tower.
    # ONNX/INT8 modes load text_encoder.onnx so natural-language search can run
    # as semantic retrieval instead of keyword/alias-only fallback.
    text_encoder = None
    if cfg.inference_backend == "onnx":
        try:
            text_encoder = OnnxTextEncoder(cfg.mobile_artifact_dir)
        except FileNotFoundError as exc:
            print(f"[boot] ONNX text encoder unavailable: {exc}")
        except Exception as exc:
            print(f"[boot] ONNX text encoder failed to load: {exc}")
    elif model is not None and train_cfg is not None:
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer(train_cfg["model"]["model_name"])
            text_encoder = TextEncoder(model, tokenizer, device=device)
        except Exception as exc:
            print(f"[boot] text encoder unavailable: {exc}")

    logger = DebugLogger(Path(cfg.log_path))

    return {
        "ok": True,
        "config": cfg,
        "asset_dir": asset_dir,
        "bundle": asset_result.bundle,
        "model": model,
        "recognizer": recognizer,
        "text_encoder": text_encoder,
        "classes": classes,
        "device": device,
        "logger": logger,
        "warnings": asset_result.errors,
        "inference_backend": cfg.inference_backend,
    }


def _outcome_log_extra(outcome, policy: ConfidencePolicy, model_version: str, quality_report=None) -> dict:
    extra = {
        "policy_version": "sprint1-reliability-v1",
        "decision": outcome.decision,
        "reason_codes": outcome.reason_codes,
        "top1_score": outcome.top1_score,
        "top2_score": outcome.top2_score,
        "margin": outcome.margin,
        "thresholds": {
            "reject_threshold": policy.reject_threshold,
            "weak_reject_threshold": policy.weak_reject_threshold,
            "weak_margin": policy.weak_margin,
            "match_threshold": policy.match_threshold,
            "match_floor": policy.match_floor,
            "match_margin": policy.match_margin,
            "isolated_match_threshold": policy.isolated_match_threshold,
            "isolated_match_margin": policy.isolated_match_margin,
            "isolated_match_top2_max": policy.isolated_match_top2_max,
            "text_no_keyword_reject_threshold": policy.text_no_keyword_reject_threshold,
        },
        "model_version": model_version,
    }
    if quality_report is not None:
        extra["quality"] = {
            "ok": quality_report.ok,
            "reason_codes": quality_report.reason_codes,
            "min_side": quality_report.min_side,
            "brightness": quality_report.brightness,
            "contrast": quality_report.contrast,
            "sharpness": quality_report.sharpness,
        }
    return extra


def render_top3(outcome, bundle, key_prefix: str) -> None:
    show_key = f"{key_prefix}_show_low_confidence"
    if outcome.decision == "out_of_scope" and not st.session_state.get(show_key, False):
        st.warning("지원 범위의 랜드마크로 확인되지 않았습니다.")
        st.caption("원하면 내부 점수 기준의 가까운 후보를 확인할 수 있지만, 확정 결과로 보지는 않습니다.")
        if st.button("후보 참고용으로 보기", key=f"{key_prefix}_toggle_out"):
            st.session_state[f"{key_prefix}_show_below"] = True
            st.session_state[show_key] = True
            st.rerun()
        return

    if outcome.decision == "low_quality" and not st.session_state.get(show_key, False):
        st.warning("사진 품질이 낮아 판별하기 어렵습니다. 더 밝고 선명한 사진으로 다시 시도해 주세요.")
        if st.button("후보 참고용으로 보기", key=f"{key_prefix}_toggle_quality"):
            st.session_state[show_key] = True
            st.rerun()
        return

    if outcome.decision == "ambiguous":
        st.info("한 곳으로 확정하기 어렵습니다. 가까운 후보를 함께 보여드립니다.")
    elif outcome.decision == "low_quality":
        st.warning("품질 이슈가 있어 참고용 후보로 표시합니다.")

    st.caption("표시된 %는 정답 확률이 아니라 모델 유사도 점수를 사용자용으로 변환한 값입니다.")
    cols = st.columns(min(3, max(1, len(outcome.top3))))
    for col, item in zip(cols, outcome.top3):
        info = bundle.info_by_id.get(item.landmark_id)
        name = info.name_ko if info else item.landmark_id
        with col:
            st.markdown(f"### #{item.rank}")
            st.markdown(f"**{name}**")
            st.progress(item.percentage / 100.0, text=f"{item.percentage}%")
            st.caption(f"`{item.landmark_id}`")
            if st.button("자세히 보기", key=f"{key_prefix}_detail_{item.landmark_id}_{item.rank}"):
                st.session_state["selected_landmark_id"] = item.landmark_id
                st.session_state["page"] = "landmark"
                st.rerun()


def render_landmark_page(landmark_id: str, bundle, asset_dir: Path) -> None:
    info = bundle.info_by_id.get(landmark_id)
    if info is None:
        st.error(f"메타데이터를 찾을 수 없습니다: `{landmark_id}`")
        if st.button("← 검색으로 돌아가기"):
            st.session_state["page"] = "search"
            st.rerun()
        return

    if st.button("← 검색으로 돌아가기", key="back_to_search"):
        st.session_state["page"] = "search"
        st.rerun()

    st.title(info.name_ko)
    st.caption(f"{info.name_en} · `{info.landmark_id}`")

    left, right = st.columns([3, 2])
    with left:
        hero_path = info.hero_image_path
        if hero_path and Path(hero_path).exists():
            st.image(hero_path, use_container_width=True)
        else:
            st.info("대표 이미지 없음")
        st.subheader("설명")
        st.write(info.description_ko or "(설명 없음)")

    with right:
        st.subheader("정보")
        if info.aliases:
            st.markdown("**별칭**")
            st.write(", ".join(info.aliases))
        if info.tags:
            st.markdown("**태그**")
            st.write(", ".join(info.tags))
        st.markdown("**위치**")
        if info.coordinates_valid:
            st.write(f"{info.latitude}, {info.longitude}")
            st.markdown(f"[Google Maps에서 보기]({info.map_url})")
            st.map({"latitude": [info.latitude], "longitude": [info.longitude]}, zoom=14)
        else:
            st.write("위치 정보 없음")


def main() -> None:
    st.set_page_config(page_title="Landmark Assistant", layout="wide")

    config_path = os.environ.get("LANDMARK_DEMO_CONFIG", "./config.toml")
    state = boot(config_path)

    if not state.get("ok"):
        st.title("자산 로드 실패")
        st.error("필수 자산을 로드하지 못했습니다. `python scripts/build_assets.py`를 먼저 실행하세요.")
        for err in state.get("errors", []):
            st.write(f"- {err}")
        st.code(f"assets_dir = {state.get('asset_dir')}")
        return

    cfg = state["config"]
    bundle = state["bundle"]
    asset_dir = state["asset_dir"]
    policy = ConfidencePolicy(**cfg.policy)
    model_version = Path(cfg.checkpoint).name

    if state.get("warnings"):
        with st.sidebar.expander("자산 경고", expanded=False):
            for w in state["warnings"]:
                st.write(f"- {w}")

    # ---- Sidebar ----
    st.sidebar.title("Landmark Assistant")
    st.sidebar.caption(f"device: `{state['device']}`")
    st.sidebar.caption(f"backend: {cfg.inference_backend}")
    dev_mode = st.sidebar.toggle("개발자 모드", value=False)
    st.sidebar.divider()
    if st.sidebar.button("모든 검색 초기화"):
        for k in list(st.session_state.keys()):
            if k.startswith("query_") or k.startswith("img_"):
                del st.session_state[k]
        st.session_state["last_outcome"] = None
        st.rerun()

    if "page" not in st.session_state:
        st.session_state["page"] = "search"

    if st.session_state["page"] == "landmark":
        render_landmark_page(st.session_state.get("selected_landmark_id", ""), bundle, asset_dir)
        return

    # ---- Search page ----
    st.title("Landmark Assistant")
    st.caption("MobileCLIP2-S4 기반. 이미지·자연어로 지원 범위의 랜드마크를 검색합니다.")

    tab_image, tab_text = st.tabs(["📷 이미지", "💬 자연어"])

    last_outcome = st.session_state.get("last_outcome")

    # ---- Image search ----
    with tab_image:
        uploaded = st.file_uploader("이미지 업로드 (JPEG/PNG/WEBP, 10MB 이하)", type=["jpg", "jpeg", "png", "webp"], key="img_upload")
        if uploaded is not None:
            ok, msg = validate_image_file(uploaded.name, uploaded.size, max_mb=cfg.max_image_mb)
            if not ok:
                st.error(msg)
            else:
                from PIL import Image
                pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                st.image(pil_img, caption=uploaded.name, use_container_width=False, width=320)
                quality_report = assess_image_quality(pil_img)
                if not quality_report.ok:
                    st.caption(f"품질 경고: {', '.join(quality_report.reason_codes)}")
                with st.spinner("추론 중..."):
                    embedding, elapsed_ms = state["recognizer"].encode(pil_img)
                weights = FusionWeights(**cfg.image_only)
                weights.validate()
                outcome = search_by_image(embedding, bundle, weights, cfg.reject_threshold, policy=policy)
                if not quality_report.ok:
                    outcome = apply_decision_policy(
                        outcome,
                        "image",
                        policy,
                        low_quality=True,
                        quality_reason_codes=quality_report.reason_codes,
                    )
                st.caption(f"처리 시간: {elapsed_ms} ms")
                if elapsed_ms > cfg.slow_inference_ms:
                    st.warning("추론이 지연되고 있습니다.")
                state["logger"].log(
                    kind="image", input_id=uploaded.name, elapsed_ms=elapsed_ms,
                    below_threshold=outcome.below_threshold,
                    top3=[{"landmark_id": t.landmark_id, "fusion_score": t.fusion_score, "rank": t.rank} for t in outcome.top3],
                    scores=outcome.all_scores,
                    extra=_outcome_log_extra(outcome, policy, model_version, quality_report),
                )
                last_outcome = outcome
                st.session_state["last_outcome"] = outcome
                render_top3(outcome, bundle, key_prefix="img")

    # ---- Text search ----
    with tab_text:
        query = st.text_input("자연어 검색 (한국어/영어, 최대 200자)", key="query_text")
        run_text = st.button("검색", key="query_text_run")
        if run_text:
            stripped = query.strip()
            if not stripped:
                st.warning("검색어를 입력하세요")
            else:
                truncated = False
                if len(stripped) > 200:
                    stripped = stripped[:200]
                    truncated = True
                    st.info("검색어가 200자로 잘렸습니다")
                text_embedding = None
                if state.get("text_encoder") is not None:
                    try:
                        with st.spinner("텍스트 인코딩 중..."):
                            text_embedding = state["text_encoder"].encode(stripped)
                    except Exception as exc:
                        st.warning(f"텍스트 인코더 실패: {exc}")
                weights = FusionWeights(**cfg.text_only)
                weights.validate()
                outcome = search_by_text(text_embedding, stripped, bundle, weights, cfg.reject_threshold, policy=policy)
                state["logger"].log(
                    kind="text", input_id=stripped[:80], elapsed_ms=0,
                    below_threshold=outcome.below_threshold,
                    top3=[{"landmark_id": t.landmark_id, "fusion_score": t.fusion_score, "rank": t.rank} for t in outcome.top3],
                    scores=outcome.all_scores,
                    extra=_outcome_log_extra(outcome, policy, model_version),
                )
                last_outcome = outcome
                st.session_state["last_outcome"] = outcome
                render_top3(outcome, bundle, key_prefix="text")

    # ---- Dev panel ----
    if dev_mode and last_outcome is not None:
        st.divider()
        st.subheader("개발자 모드 — 13 클래스 점수")
        rows = []
        for lid in bundle.landmark_ids:
            sc = last_outcome.all_scores.get(lid, {})
            info = bundle.info_by_id.get(lid)
            rows.append({
                "landmark_id": lid,
                "name_ko": info.name_ko if info else "",
                "image": round(sc.get("image", 0.0), 4),
                "text": round(sc.get("text", 0.0), 4),
                "keyword": round(sc.get("keyword", 0.0), 4),
                "fusion": round(sc.get("fusion", 0.0), 4),
            })
        rows.sort(key=lambda r: -r["fusion"])
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
