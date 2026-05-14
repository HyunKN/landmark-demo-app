"""Streamlit entry point. 단일 페이지에서 이미지/텍스트/이름 검색 + 정보 페이지를 라우팅."""
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
from landmark_demo.inference import ImageRecognizer, TextEncoder, validate_image_file
from landmark_demo.logging_util import DebugLogger
from landmark_demo.model import load_checkpoint
from landmark_demo.search import (
    FusionWeights,
    name_search,
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

    model, classes, train_cfg = load_checkpoint(cfg.checkpoint, device=device)
    image_size = int(train_cfg["training"]["image_size"])
    image_mean = list(train_cfg["training"]["image_mean"])
    image_std = list(train_cfg["training"]["image_std"])

    recognizer = ImageRecognizer(model, image_size, image_mean, image_std, device=device)

    # Text encoder (선택적)
    text_encoder = None
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
    }


def render_top3(outcome, bundle, key_prefix: str) -> None:
    if outcome.below_threshold and not st.session_state.get(f"{key_prefix}_show_below", False):
        st.warning("범위 외 입력으로 판단됩니다.")
        if st.button("후보 보기", key=f"{key_prefix}_toggle_below"):
            st.session_state[f"{key_prefix}_show_below"] = True
            st.rerun()
        return

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
    st.set_page_config(page_title="Jongno Landmark Demo", layout="wide")

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

    if state.get("warnings"):
        with st.sidebar.expander("자산 경고", expanded=False):
            for w in state["warnings"]:
                st.write(f"- {w}")

    # ---- Sidebar ----
    st.sidebar.title("Landmark Demo")
    st.sidebar.caption(f"device: `{state['device']}`")
    st.sidebar.caption(f"backend: pytorch")
    dev_mode = st.sidebar.toggle("개발자 모드", value=False)
    st.sidebar.divider()
    if st.sidebar.button("모든 검색 초기화"):
        for k in list(st.session_state.keys()):
            if k.startswith("query_") or k.startswith("img_") or k.startswith("name_"):
                del st.session_state[k]
        st.session_state["last_outcome"] = None
        st.rerun()

    if "page" not in st.session_state:
        st.session_state["page"] = "search"

    if st.session_state["page"] == "landmark":
        render_landmark_page(st.session_state.get("selected_landmark_id", ""), bundle, asset_dir)
        return

    # ---- Search page ----
    st.title("종로 랜드마크 검색 데모")
    st.caption("MobileCLIP2-S4 기반. 이미지·자연어·이름으로 13개 종로 랜드마크를 검색합니다.")

    tab_image, tab_text, tab_name = st.tabs(["📷 이미지", "💬 자연어", "🔤 이름"])

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
                with st.spinner("추론 중..."):
                    embedding, elapsed_ms = state["recognizer"].encode(pil_img)
                weights = FusionWeights(**cfg.image_only)
                weights.validate()
                outcome = search_by_image(embedding, bundle, weights, cfg.reject_threshold)
                st.caption(f"처리 시간: {elapsed_ms} ms")
                if elapsed_ms > cfg.slow_inference_ms:
                    st.warning("추론이 지연되고 있습니다.")
                state["logger"].log(
                    kind="image", input_id=uploaded.name, elapsed_ms=elapsed_ms,
                    below_threshold=outcome.below_threshold,
                    top3=[{"landmark_id": t.landmark_id, "fusion_score": t.fusion_score, "rank": t.rank} for t in outcome.top3],
                    scores=outcome.all_scores,
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
                outcome = search_by_text(text_embedding, stripped, bundle, weights, cfg.reject_threshold)
                state["logger"].log(
                    kind="text", input_id=stripped[:80], elapsed_ms=0,
                    below_threshold=outcome.below_threshold,
                    top3=[{"landmark_id": t.landmark_id, "fusion_score": t.fusion_score, "rank": t.rank} for t in outcome.top3],
                    scores=outcome.all_scores,
                )
                last_outcome = outcome
                st.session_state["last_outcome"] = outcome
                render_top3(outcome, bundle, key_prefix="text")

    # ---- Name search ----
    with tab_name:
        name_query = st.text_input("장소 이름 (한/영, 부분 일치)", key="name_query")
        if name_query.strip():
            result = name_search(name_query, bundle.name_entries, limit=10)
            if not result.matches:
                st.info("검색 결과 없음")
            else:
                st.caption(f"{len(result.matches)}개 후보")
                for entry in result.matches:
                    info = bundle.info_by_id.get(entry.landmark_id)
                    label = f"**{info.name_ko}** · {entry.display} ({entry.kind})" if info else entry.display
                    if st.button(label, key=f"name_pick_{entry.landmark_id}_{entry.kind}"):
                        st.session_state["selected_landmark_id"] = entry.landmark_id
                        st.session_state["page"] = "landmark"
                        st.rerun()

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
