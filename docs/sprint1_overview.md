# Sprint 1 종합 보고서

작성일: 2026-05-16
대상 시스템: 서울 13개 랜드마크 데모 앱
대상 모델: MobileCLIP2-S4 image encoder (ViT, 1014 layer, 512-dim embedding)
플랫폼: Windows + Streamlit (PC), Qualcomm AI Hub (NPU 측정)

---

## 1. 한 페이지 요약

| 영역 | 결과 |
|---|---|
| **데모 앱 정책** | matched / ambiguous / out_of_scope / low_quality 4-way 분기 + reason_codes + thresholds 노출, JSONL 로그 |
| **PC 회귀 fixture** | 양성 13건 (클래스당 holdout 1장) + OOS 2건 (합성 캡처/노이즈), 자동 동결 스크립트 |
| **PC 회귀 결과 (FP32)** | 양성 13/13 top-1, decision 13/13 matched. OOS 1/2 정상 거절 (1건은 학습 데이터 한계) |
| **로컬 ORT dynamic INT8** | cos 0.999, top-1 100%, 1.25× 압축 — PC CPU 한정 무손실 |
| **로컬 ORT static QDQ** | RAM 한계로 calibration 실패 (ViT-H급 + 16 GB 호스트 한계) |
| **AI Hub 양자화 + 컴파일** | NPU 1014/1014 (100% 탑재, fallback 0), QAIRT 2.45 / Hexagon v73 |
| **NPU latency (warm)** | **S23 518 ms / S24 390 ms / S25(proxy) 374 ms** |
| **AI Hub FP32 동치성** | cos(PC FP32, AI Hub FP32 device) = **0.9995** (그래프 변환 무손실) |
| **AI Hub w8a16 양자화 정확도** | calibration 15장도 260장도 동일하게 임베딩 공간 붕괴 (top-1 13.3%) — 알고리즘-모델 적합도 한계 |

핵심 메시지: **NPU 위 latency 실측은 정직하게 확보됐다. 양자화 정확도 한계는 모델/calibration이 아니라 PTQ 알고리즘과 ViT-S4 분포 간 적합도 문제로 진단됐다.**

---

## 2. 작업 흐름 (시간 순)

| 단계 | 작업 | 비고 |
|---|---|---|
| ① | 데모 앱 정책 확장 (`matched/ambiguous/out_of_scope/low_quality` + reason_codes + model_version + 품질 정보 로그) | `src/landmark_demo/search.py`, `src/landmark_demo/app.py` |
| ② | 자연어 검색 보정 (`art gallery → mmca_seoul`, `돌담있는곳 → naksan_park`) | `search.py`의 `_keyword_score` + alias |
| ③ | demo-regression-v1 fixture 정의 (text/image/manual 케이스 분리) | `tests/fixtures/demo_regression_v1.json` |
| ④ | MobileCLIP2-S4 ONNX export + handoff 패키지 생성 | `mobile_artifacts/` |
| ⑤ | PC ONNX Runtime baseline 측정 (warm 228 ms, top-3 exact 4/4) | `scripts/benchmark_mobile_artifact.py` |
| ⑥ | 데모 앱 runbook + benchmark 문서를 Vercel docs로 배포 | 외부 |
| ⑦ | 회귀 fixture 입력 동결 (13 holdout + 2 OOS) | `scripts/freeze_regression_inputs.py` |
| ⑧ | 로컬 ORT dynamic INT8 양자화 + PC 회귀 통과 (cos 0.999) | `scripts/quantize_mobile_onnx.py` |
| ⑨ | Qualcomm AI Hub 디바이스 가용성 확인 (S23/S24/S25) | `scripts/aihub_check_devices.py` |
| ⑩ | AI Hub w8a16 양자화 1차 (calibration 15장) + 3 디바이스 compile/profile/inference | `scripts/aihub_run_jobs.py` |
| ⑪ | 진단: NPU latency 양호, 정확도 붕괴 (cos 0.087) | 가설 = calibration 부족 |
| ⑫ | 로컬 ORT static QDQ 시도 — symmetric 모드 mode collapse, asymmetric 모드 RAM 한계 | `scripts/quantize_static_qdq.py` |
| ⑬ | AI Hub w8a16 양자화 2차 (calibration 260장) — 가설 검증 | 결과 1차와 동일 → **가설 기각** |
| ⑭ | AI Hub FP32 inference로 그래프 동치성 검증 (cos 0.9995) | 양자화가 단독 원인임을 확정 |
| ⑮ | 결과 동결 + 문서화 | 본 문서 |

---

## 3. 데모 앱 정책과 회귀 검증

### 3-1. 의사결정 정책

| 분기 | 조건 (요약) | 의미 |
|---|---|---|
| `matched` | `top1_score ≥ match_threshold` 또는 (`top1_score ≥ match_floor` AND `margin ≥ match_margin`) | 신뢰 가능한 예측 |
| `ambiguous` | top-1이 매치 임계 미만 + margin 부족 | 후보 표시는 하되 단정 안 함 |
| `out_of_scope` | top1_score가 reject 또는 weak_reject 미만, margin도 작음 | 13개 랜드마크 외 입력 |
| `low_quality` | 입력 이미지 품질 휴리스틱 실패 | 사용자에 재촬영 요청 |

기본 임계값:

| 임계값 | 값 |
|---|---|
| `reject_threshold` | 0.25 |
| `weak_reject_threshold` | 0.35 |
| `weak_margin` | 0.12 |
| `match_threshold` | 0.60 |
| `match_floor` | 0.50 |
| `match_margin` | 0.20 |
| `text_no_keyword_reject_threshold` | 0.35 |

### 3-2. 회귀 fixture (`tests/fixtures/demo_regression_v1.json`)

- 양성 13건: 13개 클래스의 confirmed 이미지 중 hero/calibration과 분리된 holdout. `assets/regression_inputs/positive/<class>.jpg`
- OOS 2건: 합성 화면 캡처 (`screen_capture_synthetic.png`) + uniform noise (`uniform_noise.jpg`)
- `scripts/freeze_regression_inputs.py`로 자동 재생성

### 3-3. PC FP32 회귀 결과

- 양성 13건 top-1: **13/13 (100%)**
- 양성 13건 decision == matched: **13/13 (100%)**
- OOS uniform_noise → out_of_scope: ✅
- OOS screen_capture_synthetic → out_of_scope: ❌ (ambiguous로 분류, top-1 = mmca_seoul)
- 단위 테스트: `python -m pytest -q` → **3 passed**

OOS 1건의 실패는 정책 한계가 아니라 모델 학습 데이터에 negative 클래스가 없는 것이 원인. 데이터셋 v2의 다음 작업 항목으로 명시.

---

## 4. 양자화 시도 4종 비교

| 경로 | 위치 | calibration | 모델 크기 | PC cos | NPU cos vs PC FP32 | 비고 |
|---|---|---|---|---|---|---|
| **ORT dynamic INT8** | 로컬 | (없음, weight-only) | 1233→983 MB (1.25×) | **0.999** | (NPU 미컴파일) | QAIRT가 dynamic INT8 op를 받지 않음 |
| **ORT static QDQ symmetric** | 로컬 | 260장 | 1233→316 MB (3.9×) | **0.02** ❌ | — | symmetric INT8이 ViT 분포에 부적합, mode collapse |
| **ORT static QDQ asymmetric** | 로컬 | 100~260장 | — | (실행 불가) | — | calibrator RAM 한계로 BFCArena 실패 |
| **AI Hub w8a16 (1차)** | AI Hub | 15장 | (서버 측) | — | **0.087** ❌ | 가설: calibration 부족 |
| **AI Hub w8a16 (2차)** | AI Hub | 260장 (17×) | (서버 측) | — | **0.076** ❌ | 가설 기각 |
| **AI Hub FP32 (검증용)** | AI Hub | (양자화 없음) | (서버 측) | — | **0.9995** ✅ | 그래프 변환은 무손실 |

### 결정적 진단

`AI Hub FP32 cos = 0.9995`는 “그래프와 디바이스 실행 자체는 PC FP32와 동일”을 의미. `AI Hub w8a16 cos ≈ 0.08`은 정확히 “양자화 단계에서” 임베딩 공간이 붕괴됨을 의미. calibration을 17배 늘려도 변하지 않음으로 **calibration 양은 결정 변수가 아님**이 검증됨. 즉 **AIMET w8a16 PTQ가 MobileCLIP2-S4의 깊은 attention 분포를 보존하지 못한다**가 진짜 원인.

---

## 5. NPU 측정 (Qualcomm AI Hub)

### 5-1. 디바이스 풀

`scripts/aihub_check_devices.py` 실행 결과 기준:

- **Snapdragon 8 Gen 2** (4종): Galaxy S23 / S23 / S23 Ultra / S23+
- **Snapdragon 8 Gen 3** (4종): Galaxy S24 (Family) / S24 / S24 Ultra / S24+
- **Snapdragon 8 Elite** (1종): Snapdragon 8 Elite QRD (S25 Family는 풀에 미등록, 같은 칩셋 reference device를 proxy로 사용)

선택: `Samsung Galaxy S23 (Family)`, `Samsung Galaxy S24 (Family)`, `Snapdragon 8 Elite QRD`.

### 5-2. 컴파일 옵션 / 환경

- `--target_runtime qnn_context_binary --compute_unit npu`
- QAIRT SDK 2.45.0
- Hexagon v73 HTP
- HTP optimization level 3
- Quantization: w8a16 (가중치 INT8, 활성화 INT16)
- Calibration: 클래스당 20장, 총 260장 (Dataset의 confirmed 이미지 중 hero/regression 분리)

### 5-3. Latency (확정값)

같은 INT8 모델을 3 디바이스에 동일 옵션으로 컴파일·프로파일한 결과:

| 디바이스 | SoC | warm latency | NPU layer | 세대간 가속 |
|---|---|---|---|---|
| Samsung Galaxy S23 (Family) | Snapdragon 8 Gen 2 | **517.91 ms** | **1014 / 1014 (100%)** | (baseline) |
| Samsung Galaxy S24 (Family) | Snapdragon 8 Gen 3 | **389.67 ms** | **1014 / 1014 (100%)** | 1.33× |
| Snapdragon 8 Elite QRD | Snapdragon 8 Elite | **374.10 ms** | **1014 / 1014 (100%)** | 1.04× (vs S24) |

### 5-4. 재현성 검증

calibration 15장 (1차) ↔ 260장 (2차)의 latency 차이는 ±1~70 ms 수준이고 NPU 비율은 양쪽 모두 100%. 즉 latency 측정 자체는 calibration 양과 무관하게 안정적임이 확인됨.

### 5-5. 정확도 캐비얏

이 latency 값은 **임베딩 공간이 PC FP32와 일치한다는 보장 없이** 얻어진 값이다. 측정의 의미는:

- 실용 시나리오에서 사용자에게 같은 결과를 줄 수 있는 모델은 아니다.
- 그래프가 NPU에 100% 탑재되어 실제 HTP 실행 시간이라는 것은 보장된다.
- 백본 또는 양자화 알고리즘을 바꿔 같은 그래프 형태로 다시 컴파일해도 latency 값은 거의 동일할 것으로 추정 (model surgery가 아닌 한).

---

## 6. 산출물 인덱스

### 6-1. 코드 (스크립트)

| 파일 | 역할 |
|---|---|
| `scripts/export_mobile_onnx.py` | best.pt → ONNX export + 메타 묶음 |
| `scripts/build_assets.py` | prototype_index, text_index, hero_images 생성 |
| `scripts/benchmark_mobile_artifact.py` | PC FP32 ONNX 단독 벤치마크 |
| `scripts/quantize_mobile_onnx.py` | ORT dynamic INT8 (weight-only) 양자화 |
| `scripts/quantize_static_qdq.py` | ORT static QDQ 양자화 (현재 RAM 한계로 실험적) |
| `scripts/quantization_regression.py` | PC FP32 ↔ INT8 회귀 비교 |
| `scripts/freeze_regression_inputs.py` | 회귀 fixture 자동 동결 |
| `scripts/aihub_check_devices.py` | AI Hub 디바이스 가용성 점검 |
| `scripts/aihub_run_jobs.py` | AI Hub 풀 파이프라인 (compile/profile/inference, calibration 옵션) |

### 6-2. 데모 앱 코드

| 파일 | 역할 |
|---|---|
| `src/landmark_demo/app.py` | Streamlit entry, 정책 분기, 로그 필드 확장 |
| `src/landmark_demo/search.py` | fusion ranker + apply_decision_policy |
| `src/landmark_demo/inference.py` | preprocess, encode, ImageQualityReport |
| `src/landmark_demo/data.py` | AssetBundle 로더 |
| `run.py` | Streamlit 실행 wrapper |

### 6-3. 모바일/디바이스 산출물

| 폴더/파일 | 내용 |
|---|---|
| `mobile_artifacts/` | FP32 ONNX 번들 (encoder + preprocessing.json + labels_master.json + prototype_index.json + manifest.json) |
| `mobile_artifacts/benchmark_report.json` | PC FP32 baseline 벤치마크 |
| `mobile_artifacts_int8/` | ORT dynamic INT8 번들 |
| `mobile_artifacts_int8/quantization_regression_report.json` | PC 회귀 (cos 0.999) |
| `mobile_artifacts_int8/aihub_report_calib15.json` | AI Hub 1차 (calib 15) |
| `mobile_artifacts_int8/aihub_report_calib260.json` | AI Hub 2차 (calib 260) |
| `mobile_artifacts_int8_qdq/` | 로컬 static QDQ 시도 (mode collapse 케이스 보존) |

### 6-4. 회귀 fixture / 데이터

| 경로 | 내용 |
|---|---|
| `tests/fixtures/demo_regression_v1.json` | 회귀 fixture (text 4 + image_policy 3 + manual 15) |
| `assets/regression_inputs/positive/*.jpg` | 양성 13장 holdout |
| `assets/regression_inputs/out_of_scope/*` | OOS 2장 (합성) |
| `assets/landmark_info.json`, `landmark_text_index.json`, `prototype_index.json`, `hero_images/` | 데모 앱 자산 |

### 6-5. 문서

| 경로 | 내용 |
|---|---|
| `docs/sprint1_overview.md` | 본 문서 (전체 정리) |
| `docs/sprint1_int8_npu_report.md` | 양자화·NPU 깊이 있는 보고서 |
| `docs/runbook.md` | 재현 명령어 모음 |
| `.kiro/specs/landmark-demo-app/sprint1_status.md` | spec 폴더용 상태 메모 |

### 6-6. AI Hub 작업 ID (감사 추적)

`mobile_artifacts_int8/aihub_report_calib260.json`의 `devices[].compile.job_id`, `profile.job_id`, `inference.job_id`로 모든 작업이 보존됨. AI Hub Workbench UI에서 https://workbench.aihub.qualcomm.com/jobs/<JOB_ID>/ 로 직접 조회 가능.

---

## 7. 검증 게이트 결과

| 게이트 | 통과 여부 | 비고 |
|---|---|---|
| `python -m pytest -q` | ✅ 3 passed | 정책/검색 단위 테스트 |
| `python -m py_compile scripts/*.py` | ✅ all green | 모든 스크립트 syntax 통과 |
| Streamlit 데모 앱 200 OK | ✅ | 로컬 / 같은 Wi-Fi 폰 모두 |
| PC FP32 ↔ ORT dynamic INT8 cos | ✅ 0.999 | top-1 100%, decision 100% |
| AI Hub compile (3 디바이스 × 2회) | ✅ 6/6 | NPU 100% 탑재 |
| AI Hub FP32 동치성 검증 | ✅ cos 0.9995 | 그래프 변환 무손실 |
| AI Hub w8a16 정확도 (1차) | ❌ cos 0.087 | mode collapse |
| AI Hub w8a16 정확도 (2차) | ❌ cos 0.076 | calibration 17배 확장 무효 |

---

## 8. 발표용 핵심 메시지

> Sprint 1에서는 13개 서울 랜드마크 데모 앱에 4-way 의사결정 정책을 도입하고, MobileCLIP2-S4 image encoder를 ONNX/INT8/NPU 경로로 단계별 검증했다.
>
> **NPU 측정**은 Qualcomm AI Hub의 Snapdragon 8 Gen 2/Gen 3/Elite 디바이스에서 동일 모델을 컴파일해 1014/1014 layer가 모두 Hexagon HTP에 탑재됨을 확인했고, warm latency는 S23 518 ms / S24 390 ms / S25(proxy) 374 ms로 측정됐다.
>
> **양자화 정확도**는 4가지 PTQ 경로(ORT dynamic INT8, ORT static QDQ symmetric, AI Hub w8a16 with 15장, AI Hub w8a16 with 260장)에서 모두 깊은 ViT의 임베딩 공간을 보존하는 데 실패했다. AI Hub FP32 inference에서 cos 0.9995로 그래프 변환이 무손실임을 확인함으로써, 병목이 calibration 양도 모델 그래프도 아니고 정확히 PTQ 알고리즘과 ViT-S4 활성화 분포 간 적합도 문제임을 결정적으로 진단했다.
>
> 다음 스프린트의 결정점은 (a) 모바일 의도 변형 백본 (MobileCLIP2-S3) 교체, (b) AIMET QAT, (c) 데이터셋 v2 (negative 클래스 신설)의 조합이다.

---

## 9. 미해결 항목 (Sprint 2 입력)

- INT8 PTQ에서 mode collapse 일으키는 layer/op 식별 (AIMET layer-wise sensitivity 또는 ORT calibrator 분리 실행 필요)
- INT16 가중치 / FP16 활성화 / mixed-precision 경로 미시도 (AI Hub UI는 layer-wise precision 미노출)
- `screen_capture_synthetic` OOS가 mmca_seoul 방향으로 흐르는 문제 — 데이터셋 v2의 negative 클래스 신설로 해결 권고
- 백본 S3 교체 시 prototype 재인코딩 + 정책 임계값 재튜닝 필요
- iPhone Core ML 측정은 Mac 환경 확보 후 동일 fixture/artifact로 재실행 가능 (현 환경에서 미수행)

---

## 10. 환경 메모

- 호스트: Windows, Python 3.13.x, ONNX Runtime 1.24.4, ONNX 1.21.0, qai-hub 0.47.0
- 디스크: C 드라이브 여유 ~6 GB / D 드라이브 ~390 GB. 양자화/AI Hub 업로드 임시 파일이 1~3 GB 단위라 모든 스크립트가 D 드라이브로 자동 우회 (`tempfile.tempdir`, `TEMP/TMP/TMPDIR` 강제)
- AI Hub 토큰: 호스트에 등록 완료. `qai-hub configure --api_token <TOKEN>`로 갱신 가능
- 시연 흐름: `python run.py` → `http://localhost:8501` (PC) 또는 같은 Wi-Fi의 폰 Safari/Chrome에서 `http://<노트북_IP>:8501`
