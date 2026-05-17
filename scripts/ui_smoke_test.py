"""Streamlit UI smoke test using streamlit.testing.v1.

This verifies the Sprint 1 demo flow without opening a real browser:
  - only Image/Text tabs are exposed
  - text search renders Top-3 detail buttons
  - detail page can render metadata for a selected landmark
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", default="art gallery")
    parser.add_argument("--detail-landmark-id", default="mmca_seoul")
    parser.add_argument("--output", default="./logs/ui_smoke_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LANDMARK_DEMO_CONFIG"] = str(Path(args.config).resolve())
    app_path = Path(__file__).resolve().parents[1] / "src" / "landmark_demo" / "app.py"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    tabs = [tab.label for tab in at.tabs]
    text_inputs = [item.label for item in at.text_input]
    initial_buttons = [button.label for button in at.button]
    initial_ok = tabs == ["📷 이미지", "💬 자연어"] and all("이름" not in label for label in tabs + text_inputs)

    at.text_input[0].set_value(args.query)
    at.button[0].click().run(timeout=60)
    result_buttons = [button.label for button in at.button]
    top_markdowns = [item.value for item in at.markdown if item.value.startswith("**")]
    search_ok = result_buttons.count("자세히 보기") >= 3 and bool(top_markdowns)

    at.session_state["page"] = "landmark"
    at.session_state["selected_landmark_id"] = args.detail_landmark_id
    at.run(timeout=30)
    titles = [title.value for title in at.title]
    subheaders = [subheader.value for subheader in at.subheader]
    captions = [caption.value for caption in at.caption]
    detail_ok = bool(titles and args.detail_landmark_id in " ".join(captions) and {"설명", "정보"}.issubset(set(subheaders)))

    report = {
        "version": "sprint1-ui-smoke-v1",
        "config": str(Path(args.config).resolve()),
        "tabs": tabs,
        "text_inputs": text_inputs,
        "initial_buttons": initial_buttons,
        "query": args.query,
        "result_buttons": result_buttons,
        "top_result_labels": top_markdowns[:3],
        "detail_landmark_id": args.detail_landmark_id,
        "detail_titles": titles,
        "detail_subheaders": subheaders,
        "detail_captions": captions[:3],
        "checks": {
            "name_tab_removed": initial_ok,
            "text_search_renders_detail_buttons": search_ok,
            "detail_page_renders": detail_ok,
        },
    }
    report["checks"]["all_pass"] = all(report["checks"].values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    if not report["checks"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
