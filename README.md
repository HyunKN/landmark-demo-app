# Landmark Assistant

Landmark Assistant는 지원 범위의 랜드마크를 이미지·자연어로 검색하는 MobileCLIP2-S4 기반 시연 앱이다. 학습된 체크포인트(`best.pt`) 또는 export된 ONNX/INT8 artifact로 Streamlit UI에서 동작을 시연한다.

> 실행 전 주의: `best.pt`는 약 1.7GB라 GitHub git 저장소에 포함하지 않는다. 저장소를 받은 뒤 학습 산출물 `best.pt`를 프로젝트 루트(`landmark-demo-app/best.pt`)에 놓으면 바로 실행할 수 있다.

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
    --device auto ^
    --prototype-source head
```

산출물:
- `assets/prototype_index.json` — 13개 클래스 prototype. Sprint1 갱신본은 ArcFace head weight를 class center로 사용한다.
- `assets/landmark_text_index.json` — bilingual text catalog 기반 텍스트 임베딩
- `assets/hero_images/<id>.jpg` — 13개 대표 이미지 자동 선정

이미 저장소에 포함된 `assets/prototype_index.json`, `assets/landmark_text_index.json`, `assets/hero_images/`를 그대로 쓸 수도 있다. 단, 새 `best.pt`로 바꾸면 위 명령으로 자산을 다시 빌드한다.

### 3. 데모 실행

| 명령 | 가중치 | 필요 의존성 |
|---|---|---|
| `python run.py` | `best.pt` (PyTorch FP32, 기본) | `pip install -e .` |
| `python run.py --int8` | `mobile_artifacts_int8/landmark_encoder.onnx` (ONNX INT8) | `pip install -e .[onnx]` |

```bash
# PyTorch 기본 (권장)
python run.py

# INT8 ONNX (onnxruntime 필요: pip install -e .[onnx])
python run.py --int8
```

브라우저에서 `http://localhost:8501` 접속. 종료는 터미널에서 **Ctrl+C**.

### 4. ONNX로 실행/비교

일반 데모는 `best.pt`를 PyTorch로 직접 로드한다. export 결과와 성능 차이를 보고 싶으면 ONNX artifact를 만든 뒤 ONNX config로 실행한다.

```powershell
pip install -e .[onnx]

python scripts\export_mobile_onnx.py ^
  --checkpoint .\best.pt ^
  --assets-dir .\assets ^
  --output-dir .\mobile_artifacts

python scripts\benchmark_mobile_artifact.py ^
  --checkpoint .\best.pt ^
  --assets-dir .\assets ^
  --artifact-dir .\mobile_artifacts ^
  --runs 5

python run.py --onnx
```

ONNX config에서는 이미지 인식 경로가 ONNX Runtime CPU로 실행된다. 자연어 검색은 PyTorch checkpoint를 다시 로드하지 않고 keyword/text-index 중심으로 동작한다. 이는 Windows 노트북에서 큰 checkpoint를 중복 로드하며 발생하는 paging file 오류를 피하기 위한 선택이다.

검증된 dynamic INT8 artifact를 받은 경우에는 `mobile_artifacts_int8/` 폴더를 프로젝트 루트에 두고 다음 명령으로 실행한다.

```powershell
python run.py --int8
```

팀원 배포용으로는 `mobile_artifacts_int8/` 전체 폴더를 전달한다. `.onnx` 파일만 전달하면 external data와 prototype metadata가 빠져 실행되지 않는다.

## 폴더 구조

```
landmark-demo-app/
├── README.md
├── pyproject.toml
├── config.toml                          # 가중치, threshold, 경로
├── config.onnx.toml                     # ONNX Runtime 실행 설정
├── config.int8.toml                     # dynamic INT8 ONNX Runtime 실행 설정
├── best.pt                              # 학습 체크포인트 (gitignore)
├── run.py                               # streamlit launcher
├── assets/                              # 빌드 산출물 (gitignore)
│   ├── landmark_info.json               # 13개 메타데이터 (커밋 대상)
│   ├── landmark_text_catalog_v2.json     # 한/영 alias, query, contrast catalog
│   ├── prototype_index.json
│   ├── landmark_text_index.json
│   └── hero_images/
├── logs/                                # JSONL 디버그 로그 (gitignore)
├── scripts/
│   ├── build_assets.py
│   ├── augment_dataset_captions_v2.py
│   └── export_mobile_onnx.py
├── src/landmark_demo/
│   ├── app.py                           # Streamlit entry
│   ├── config.py
│   ├── model.py                         # MobileCLIP2 wrapper
│   ├── inference.py                     # 이미지/텍스트 인코더
│   ├── data.py                          # Asset bundle 로더
│   ├── search.py                        # Fusion ranker and confidence policy
│   └── logging_util.py                  # Debug log
└── .kiro/specs/landmark-demo-app/       # spec (requirements/design/tasks)
```

## 동작

| 탭 | 입력 | 동작 |
|---|---|---|
| 📷 이미지 | JPEG/PNG/WEBP, 10MB 이하 | classifier head 임베딩 ↔ prototype cosine, Top-3 + 신뢰도 상태 |
| 💬 자연어 | 한/영, 200자 이하 | text encoder + keyword 매칭 fusion (가중치 0.6/0.4) + 신뢰도 상태 |

각 결과 카드 클릭 시 상세 페이지로 이동.

신뢰도 상태는 `matched`, `ambiguous`, `out_of_scope`, `low_quality` 네 가지다. 표시되는 %는 정답 확률이 아니라
유사도 점수를 사용자용으로 변환한 값이다. JSONL 로그에는 `decision`, `reason_codes`, `top1_score`, `top2_score`,
`margin`, `thresholds`, `model_version`을 함께 남긴다.

`40/0/0`처럼 1위만 고립되어 높은 경우는 `isolated_match`로 `matched` 처리할 수 있다. 반대로 `40/30/30`처럼 후보가 붙어 있으면 `ambiguous`로 두고 Top-3를 함께 보여 사용자가 상세 페이지에서 고르게 한다.

## Sprint 1 모바일 artifact

이미지 인식 경로는 ONNX Runtime Mobile로 넘길 수 있도록 image encoder handoff package를 만든다.

```bash
python scripts/export_mobile_onnx.py \
    --checkpoint ./best.pt \
    --assets-dir ./assets \
    --output-dir ./mobile_artifacts

python scripts/benchmark_mobile_artifact.py \
    --checkpoint ./best.pt \
    --assets-dir ./assets \
    --artifact-dir ./mobile_artifacts \
    --runs 3
```

산출물:
- `mobile_artifacts/landmark_encoder.onnx`
- `mobile_artifacts/landmark_encoder.onnx.data`
- `mobile_artifacts/preprocessing.json`
- `mobile_artifacts/labels_master.json`
- `mobile_artifacts/prototype_index.json`
- `mobile_artifacts/benchmark_report.json`

PC FP32 ONNX Runtime baseline은 external data 포함 약 1.23GB다.
검증된 dynamic INT8 weight-only artifact는 `mobile_artifacts_int8/`에 두며, external data 포함 약 983MB다. 2026-05-17 새 S4 모델 기준 PC 회귀에서 FP32 대비 top-1 일치율 100%, decision 일치율 100%, embedding cosine mean 0.99942, INT8 warm median 648.95ms로 기록했다. Static QDQ 계열은 이전 실험에서 mode collapse가 확인되어 Sprint 1 팀원 배포용으로 쓰지 않는다.

로컬 C 드라이브 여유 공간이 부족하면 ONNX/INT8 생성은 D 드라이브 같은 scratch 경로에서 수행한 뒤, 완성된 `mobile_artifacts/` 또는 `mobile_artifacts_int8/` 폴더를 프로젝트 루트에 복사해 실행한다.

## Sprint 1 종합 보고서 (동결)

INT8 양자화, NPU 측정(Snapdragon 8 Gen 2/Gen 3/Elite), PC 회귀, 그리고 다음 스프린트 결정점은 다음 문서에서 다룬다.

- [`docs/sprint1_overview.md`](docs/sprint1_overview.md) — 전체 작업 정리
- [`docs/sprint1_int8_npu_report.md`](docs/sprint1_int8_npu_report.md) — 양자화·NPU 깊이 있는 보고서
- [`docs/runbook.md`](docs/runbook.md) — 재현 명령어
- [`docs/dataset_v2_retraining_plan.md`](docs/dataset_v2_retraining_plan.md) — dataset v2 caption 보강과 재학습 절차
- [`.kiro/specs/landmark-demo-app/sprint1_status.md`](.kiro/specs/landmark-demo-app/sprint1_status.md) — spec 상태 메모

### 핵심 측정값

| 지표 | 값 |
|---|---|
| NPU latency (warm) | S23 518 ms / S24 390 ms / S25(proxy) 374 ms |
| NPU layer 비율 | 1014 / 1014 (100% 탑재, fallback 0) |
| AI Hub FP32 동치성 | cos(PC FP32, AI Hub FP32 device) = 0.9995 |
| AI Hub w8a16 정확도 | calibration 15장 / 260장 모두 cos ≈ 0.08 (mode collapse) |
| 로컬 PC INT8 baseline | cos 0.999, top-1 100% |

## Fallback

자산이 없으면 시작 화면에 안내 + 검색 비활성화. `python scripts/build_assets.py`로 자산 생성 후 재실행.

## 관련 문서

- `../landmark-assistant/docs/decisions/ADR-0001-model-architecture.html`
- `../landmark-assistant/docs/decisions/ADR-0003-text-encoder-natural-language-search.html`
- `../landmark-assistant/docs/decisions/ADR-0004-image-recognizer-model-selection.html`
- `.kiro/specs/landmark-demo-app/requirements.md`
- `.kiro/specs/landmark-demo-app/design.md`
- `.kiro/specs/landmark-demo-app/tasks.md`
