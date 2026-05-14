# landmark-demo-app

종로구 13개 랜드마크 인식 데모 앱. 학습된 MobileCLIP2-S4 체크포인트(`best.pt`)를 그대로 로드해 Streamlit UI에서 이미지/자연어/이름 3가지 검색을 시연한다.

## 빠른 시작

### 1. 의존성 설치

```bash
cd landmark-demo-app
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -e .
```

또는 onnx까지:
```bash
pip install -e .[onnx]
```

### 2. 자산 빌드 (학습 산출물 → 데모 자산)

`best.pt`는 이미 폴더에 있어야 한다. Dataset 경로를 인자로 줘서 prototype/text/hero 자산을 생성:

```bash
python scripts/build_assets.py ^
    --checkpoint ./best.pt ^
    --data-root ../Dataset ^
    --landmark-info ./assets/landmark_info.json ^
    --output-dir ./assets ^
    --device auto
```

산출물:
- `assets/prototype_index.json` — 13개 클래스 이미지 임베딩 평균
- `assets/landmark_text_index.json` — description+keyword 텍스트 임베딩
- `assets/hero_images/<id>.jpg` — 13개 대표 이미지 자동 선정

### 3. 데모 실행

```bash
python run.py
```

브라우저에서 자동 열리는 `http://localhost:8501` 접속.

## 폴더 구조

```
landmark-demo-app/
├── README.md
├── pyproject.toml
├── config.toml                          # 가중치, threshold, 경로
├── best.pt                              # 학습 체크포인트 (gitignore)
├── run.py                               # streamlit launcher
├── assets/                              # 빌드 산출물 (gitignore)
│   ├── landmark_info.json               # 13개 메타데이터 (커밋 대상)
│   ├── prototype_index.json
│   ├── landmark_text_index.json
│   └── hero_images/
├── logs/                                # JSONL 디버그 로그 (gitignore)
├── scripts/
│   └── build_assets.py
├── src/landmark_demo/
│   ├── app.py                           # Streamlit entry
│   ├── config.py
│   ├── model.py                         # MobileCLIP2 wrapper
│   ├── inference.py                     # 이미지/텍스트 인코더
│   ├── data.py                          # Asset bundle 로더
│   ├── search.py                        # Fusion ranker, name search
│   └── logging_util.py                  # Debug log
└── .kiro/specs/landmark-demo-app/       # spec (requirements/design/tasks)
```

## 동작

| 탭 | 입력 | 동작 |
|---|---|---|
| 📷 이미지 | JPEG/PNG/WEBP, 10MB 이하 | classifier head 임베딩 ↔ prototype cosine, Top-3 |
| 💬 자연어 | 한/영, 200자 이하 | text encoder + keyword 매칭 fusion (가중치 0.6/0.4) |
| 🔤 이름 | 1자 이상 | NFC+lower-case, 부분 일치 자동완성 |

각 결과 카드 클릭 시 상세 페이지로 이동.

## Fallback

자산이 없으면 시작 화면에 안내 + 검색 비활성화. `python scripts/build_assets.py`로 자산 생성 후 재실행.

## 관련 문서

- `../landmark-assistant/docs/decisions/ADR-0001-model-architecture.html`
- `../landmark-assistant/docs/decisions/ADR-0003-text-encoder-natural-language-search.html`
- `../landmark-assistant/docs/decisions/ADR-0004-image-recognizer-model-selection.html`
- `.kiro/specs/landmark-demo-app/requirements.md`
- `.kiro/specs/landmark-demo-app/design.md`
- `.kiro/specs/landmark-demo-app/tasks.md`
