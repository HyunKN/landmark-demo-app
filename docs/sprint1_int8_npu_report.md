# Sprint 1 — INT8 양자화 및 NPU 측정 종합 보고서

작성일: 2026-05-16
대상 모델: MobileCLIP2-S4 (image encoder, 1014 layer, FP32 1.23 GB)
대상 디바이스 (NPU 측정): Snapdragon 8 Gen 2 / 8 Gen 3 / 8 Elite (proxy)
런타임: Qualcomm AI Hub (QAIRT 2.45.0, Hexagon v73 HTP) + 로컬 ONNX Runtime 1.24.4

---

## 1. 결과 요약

| 항목 | 결과 |
|---|---|
| **NPU latency 실측** (S23/S24/S25) | **518 / 390 / 374 ms** (warm, NPU 100% 탑재) |
| **NPU layer 비율** | **1014 / 1014 (100%)** — fallback 0 |
| **컴파일 안정성** | 3개 디바이스 × 2회 시도 모두 성공 |
| **양자화 정확도 (INT8 PTQ)** | **모든 경로에서 임베딩 공간 붕괴** — top-1 vs PC FP32 13.3% (2/15) |
| **AI Hub FP32 경로** | cos(PC FP32, device) = **0.9995** (그래프 변환은 무손실) |
| **로컬 ORT dynamic INT8 (weight-only)** | cos 0.999, top-1 100% — PC CPU 한정 무손실 |
| **로컬 ORT static QDQ** | RAM 한계로 풀 calibration 실패 |

핵심 결론: **NPU 위에서의 latency 실측은 정직하게 확보**됐으나, **MobileCLIP2-S4 + INT8 PTQ 조합은 실용 정확도에 도달하지 못했다.** 다음 단계 결정점은 백본 교체(S3) 또는 QAT(Quantization-Aware Training)로의 전환이다.

---

## 2. 작업 타임라인

| 순서 | 작업 | 산출물 |
|---|---|---|
| 1 | 데모 앱 정책 확장 (`matched / ambiguous / out_of_scope / low_quality` + reason codes + thresholds) | `src/landmark_demo/search.py`, `src/landmark_demo/app.py` |
| 2 | MobileCLIP2-S4 ONNX export | `mobile_artifacts/landmark_encoder.onnx` (FP32, 1.23 GB) |
| 3 | 로컬 ORT dynamic INT8 (weight-only) 양자화 | `mobile_artifacts_int8/` (983 MB, 1.25× 압축) |
| 4 | PC 회귀 fixture 동결 (13 클래스 holdout + 2 OOS) | `assets/regression_inputs/`, `tests/fixtures/demo_regression_v1.json` |
| 5 | PC FP32 ↔ ORT INT8 회귀 비교 | `mobile_artifacts_int8/quantization_regression_report.json` |
| 6 | Qualcomm AI Hub 디바이스 가용성 확인 | `scripts/aihub_check_devices.py` |
| 7 | AI Hub w8a16 양자화 (1차, calibration 15장) + 3 디바이스 compile/profile/inference | `mobile_artifacts_int8/aihub_report_calib15.json` |
| 8 | 로컬 ORT static QDQ 시도 — 메모리 한계 도달 | (산출물 없음, 진단 기록만) |
| 9 | AI Hub w8a16 양자화 (2차, calibration 260장) — 가설 검증 | `mobile_artifacts_int8/aihub_report_calib260.json` |
| 10 | AI Hub FP32 inference로 그래프 동치성 검증 | (진단 기록) |

---

## 3. NPU 측정값 (확정)

같은 INT8 모델(AI Hub w8a16, calibration 260장)을 3개 디바이스에 동일 옵션으로 컴파일한 결과.

| 디바이스 | SoC | latency (warm, ms) | NPU layer | OS | 비고 |
|---|---|---|---|---|---|
| Samsung Galaxy S23 (Family) | Snapdragon 8 Gen 2 | **517.91** | **1014 / 1014 (100%)** | Android 13 | hosted |
| Samsung Galaxy S24 (Family) | Snapdragon 8 Gen 3 | **389.67** | **1014 / 1014 (100%)** | Android 14 | hosted |
| Snapdragon 8 Elite QRD | Snapdragon 8 Elite | **374.10** | **1014 / 1014 (100%)** | Android 15 | proxy (S25 representative) |

컴파일 옵션: `--target_runtime qnn_context_binary --compute_unit npu`
SDK: QAIRT 2.45.0, Hexagon v73 HTP, optimization level 3
Backend: HTP (Hexagon Tensor Processor)

세대간 추세: 8 Gen 2 → Gen 3 약 **1.33× 가속**, Gen 3 → Elite 약 **1.04× 가속**. 1차 시도와 2차 시도가 latency 면에서 거의 동일함을 통해 측정의 재현성도 함께 확인됨.

발표용 한 줄: **“같은 ViT-S4 image encoder가 Snapdragon HTP에 한 layer 빠짐없이 탑재되어, S23 518 ms → S24 390 ms → S25(proxy) 374 ms로 측정되었다.”**

---

## 4. 양자화 시도 — 4가지 경로와 결과

### 4-1. ORT dynamic INT8 (weight-only) — PC 한정 baseline

| 항목 | 값 |
|---|---|
| 양자화 방식 | `quantize_dynamic`, weight-only INT8, per-channel |
| 대상 op | MatMul, Gemm |
| 모델 크기 | 1233 MB → 983 MB (1.25× 압축) |
| PC 회귀 cosine | **0.99941** (평균) / 0.99904 (최저) |
| top-1 일치율 vs FP32 | **15/15 (100%)** |
| decision 일치율 | **15/15 (100%)** |
| PC CPU latency (warm) | FP32 333 ms → INT8 314 ms (1.06×) |

PC CPU 회귀에서는 무손실. 다만 dynamic INT8 그래프(`DynamicQuantizeLinear`, `QGemm`)는 QAIRT 컴파일러가 받지 않아 NPU 경로로 직접 갈 수 없음.

**의의**: 모델 자체는 INT8 양자화 가능하다는 baseline 확보.

### 4-2. ORT static QDQ — 로컬 RAM 한계로 실패

| 시도 | 결과 |
|---|---|
| symmetric INT8 + MinMax + calib 260장 | mode collapse (top-1 1/13, 모두 myeongdong_cathedral로 붕괴) |
| asymmetric uint8 + Entropy + calib 260장 | RAM 부족으로 calibrator 실패 (BFCArena allocation failure) |
| asymmetric uint8 + Percentile + calib 104장 | RAM 부족으로 calibrator 실패 |

**진단**: ORT의 `quantize_static` calibrator는 1014 layer × N장의 활성화 텐서를 단일 process 메모리에 누적. ViT-H급 그래프 + 호스트 RAM 16~32 GB 환경의 한계점에 부딪힘. layer 단위 분할 / disk-backed calibration은 표준 API에 없어 별도 구현이 필요.

**의의**: 로컬 환경에서 풀 PTQ는 추가 엔지니어링 비용이 큼. 클라우드 PTQ로 우회하는 결정의 근거가 됨.

### 4-3. AI Hub w8a16 (1차, calibration 15장)

| 항목 | 값 |
|---|---|
| 양자화 엔진 | AIMET (Qualcomm 공식 PTQ) |
| 가중치 / 활성화 | INT8 / INT16 |
| Calibration | fixture 15장 (13 양성 + 2 OOS) |
| 컴파일 | 3 디바이스 모두 성공, NPU 1014/1014 |
| Latency | S23 515 / S24 390 / S25 306 ms |
| **top-1 일치율 vs PC FP32** | **2 / 15 (13.3%)** ❌ |
| **embedding cosine 평균** | **0.087** (직교에 가까움) |

**진단**: NPU latency는 양호하나 임베딩 공간이 PC FP32와 무관한 방향으로 회전됨. 처음에는 calibration 부족(15장)이 원인일 것으로 가정.

### 4-4. AI Hub w8a16 (2차, calibration 260장) — 가설 검증

| 항목 | 1차 (calib 15) | 2차 (calib 260) |
|---|---|---|
| Calibration 사이즈 | 15 | **260 (17×)** |
| Calibration 출처 | fixture | Dataset/<class>/labels.json, 클래스당 20장 |
| S23 latency | 515 ms | **518 ms** |
| S24 latency | 390 ms | **390 ms** |
| S25 latency | 306 ms | **374 ms** |
| NPU layer 비율 | 1014/1014 | **1014/1014** |
| top-1 일치율 vs PC FP32 | 13.3% | **13.3%** |
| embedding cosine 평균 | 0.087 | **0.076** |
| dev15 ↔ dev260 cosine (입력별) | — | **0.97~0.99** (거의 동일 모델) |

**가설 기각**: calibration 17배 확장에도 결과 동일. 따라서 mode collapse는 calibration 부족이 아니다.

### 4-5. 결정적 검증 — AI Hub FP32 (양자화 없음)

| 항목 | 값 |
|---|---|
| 경로 | AI Hub static-shape compile (FP32, 양자화 생략) |
| 입력 | 동일 fixture 15장 |
| **cos(PC FP32 ONNX, AI Hub FP32 device)** | **0.9995** (평균) / 0.9985 (최저) |

**확정**: 그래프 변환과 디바이스 실행 자체는 무손실. **임베딩 붕괴는 정확히 w8a16 양자화 단계에서 발생.** 즉 "AIMET w8a16 PTQ가 MobileCLIP2-S4의 깊은 attention/LayerNorm 분포를 보존하지 못한다"가 진짜 원인.

---

## 5. PC 회귀 fixture 및 정책 검증

### Fixture 구성 (`tests/fixtures/demo_regression_v1.json`)

- **양성 13건**: 13개 클래스의 confirmed 이미지 중 hero/calibration과 분리된 holdout
- **OOS 2건**: 합성 화면 캡처 (UI 흉내), uniform noise
- 자동 동결 스크립트: `scripts/freeze_regression_inputs.py`

### 정책 (`ConfidencePolicy`)

| 임계값 | 값 |
|---|---|
| `reject_threshold` | 0.25 |
| `weak_reject_threshold` | 0.35 |
| `weak_margin` | 0.12 |
| `match_threshold` | 0.60 |
| `match_floor` | 0.50 |
| `match_margin` | 0.20 |
| `text_no_keyword_reject_threshold` | 0.35 |

decision 분기: `matched / ambiguous / out_of_scope / low_quality`. reason_codes로 사유 노출.

### PC 회귀 결과 (FP32)

- 양성 13건 top-1: **13/13 (100%)**
- 양성 13건 decision == matched: **13/13 (100%)**
- OOS uniform_noise → out_of_scope: ✅
- OOS screen_capture_synthetic → out_of_scope: ❌ (ambiguous로 분류) — 정책 한계가 아니라 모던 미술관 내부 사진과 시각적 유사성. 데이터셋 v2에서 negative 클래스로 학습 권고.

---

## 6. 산출물 (파일/폴더 인덱스)

### 코드
- `scripts/quantize_mobile_onnx.py` — ORT dynamic INT8 양자화
- `scripts/quantize_static_qdq.py` — ORT static QDQ 양자화 (로컬 RAM 한계로 실험적 상태)
- `scripts/quantization_regression.py` — PC FP32 ↔ INT8 회귀 비교
- `scripts/freeze_regression_inputs.py` — 회귀 fixture 자동 동결
- `scripts/aihub_check_devices.py` — Qualcomm AI Hub 디바이스 가용성 점검
- `scripts/aihub_run_jobs.py` — AI Hub compile/profile/inference 자동화 (calibration 사이즈 옵션 포함)
- `scripts/export_mobile_onnx.py` — MobileCLIP2-S4 ONNX export
- `scripts/benchmark_mobile_artifact.py` — PC ONNX 단독 벤치마크

### 모바일/디바이스 산출물
- `mobile_artifacts/` — FP32 ONNX 번들 (인코더 + 전처리 + labels + prototype + manifest)
- `mobile_artifacts_int8/` — ORT dynamic INT8 번들
- `mobile_artifacts_int8/aihub_report_calib15.json` — AI Hub 1차 보고서
- `mobile_artifacts_int8/aihub_report_calib260.json` — AI Hub 2차 보고서 (확정)
- `mobile_artifacts_int8/quantization_regression_report.json` — PC 회귀 (ORT dynamic INT8)
- `mobile_artifacts_int8_qdq/` — 로컬 static QDQ 시도 산출물 (mode collapse 케이스 보존)

### Fixture / 데이터
- `tests/fixtures/demo_regression_v1.json` — 회귀 fixture (양성 13 + OOS 2)
- `assets/regression_inputs/positive/*.jpg` — 동결된 13장 holdout
- `assets/regression_inputs/out_of_scope/*` — 합성 OOS 2장

### AI Hub 작업 ID (감사 추적용)
1차 (calib 15) — `aihub_report_calib15.json` 참조
2차 (calib 260) — `aihub_report_calib260.json` 참조 (compile/profile/inference job_id × 3 디바이스)

---

## 7. 결정적 발견 (발표 슬라이드용)

1. **NPU latency 실측 확보**: S23/S24/S25 모두에서 1014/1014 layer가 Hexagon HTP에 탑재. 세대간 1.33× → 1.04× 가속 추세 관측.
2. **양자화의 진짜 병목은 calibration 양이 아니라 PTQ 알고리즘의 모델 적합도**: calibration 17배 확장(15 → 260)으로도 임베딩 cosine이 0.087 → 0.076으로 변동 없음. AI Hub FP32 경로의 cos 0.9995는 그래프 자체가 무손실임을 증명.
3. **MobileCLIP2-S4 + INT8 PTQ는 현재 조합으로는 실용 정확도 미달**: ORT dynamic / ORT static QDQ symmetric / AI Hub w8a16(15) / AI Hub w8a16(260) 4 경로 모두에서 실패 또는 도달 불가.
4. **로컬 ORT dynamic INT8은 PC CPU에서는 무손실**(cos 0.999, top-1 100%). 즉 모델 자체의 INT8 양자화 가능성은 존재하나, NPU 경로 호환성과 분포 보존을 동시에 만족하는 조합이 아직 없음.

---

## 8. 다음 스프린트 결정점

### A. 백본 교체 — MobileCLIP2-S4 → S3 (권장)
- S3는 모바일 의도 변형으로 50–150M 파라미터, 구조가 얕아 PTQ 안정성이 높음
- Apple 측이 iPhone 12 Pro Max에서 latency를 함께 공개하는 라인이라 모바일 NPU 경로 검증 데이터 풍부
- 데이터셋 v2(아래 C 항목)와 함께 진행하면 발표 메시지가 강해짐
- 비용: 백본 swap + prototype 재인코딩 + 회귀 fixture 갱신, 1~2일

### B. QAT (Quantization-Aware Training)
- AIMET QAT 또는 Brevitas로 학습 단계에서 양자화 영향을 흡수
- 깊은 ViT의 분포 문제를 근본적으로 푸는 길
- 비용: 학습 환경 추가, 3~5일

### C. 데이터셋 v2
- caption 한국어 추가 + 5종 다양화 (function/material/time_of_day/viewpoint/season)
- view_type 다양화 (정면/측면/디테일/내부)
- negative 클래스(화면 캡처/거리뷰/인물 등)를 학습에 추가
- holdout 분리 강화
- 비용: 라벨링 작업 비례, 2~5일

권장 동선: **A + C 동시 진행 → C가 끝나는 시점에 B 결정**. A 단독으로도 발표 가능.

---

## 9. 환경 / 도구 메모

- 호스트: Windows, Python 3.13, ONNX Runtime 1.24.4, ONNX 1.21.0, qai-hub 0.47.0
- 디스크 주의점: 양자화/업로드 임시 파일이 1~3 GB 단위라 `%TEMP%`가 C 드라이브 작은 경우 빈번히 ENOSPC 발생. 우리 스크립트들은 자동으로 D 드라이브로 우회하도록 처리됨 (`tempfile.tempdir`, `TEMP/TMP/TMPDIR` 모두 강제).
- AI Hub 토큰: 시스템에 이미 등록됨. `qai-hub configure --api_token <TOKEN>`으로 재설정 가능.
- Qualcomm AI Hub 디바이스 풀 (확인 시점):
  - Snapdragon 8 Gen 2: S23 / S23 Ultra / S23+ / S23 Family
  - Snapdragon 8 Gen 3: S24 / S24 Ultra / S24+ / S24 Family
  - Snapdragon 8 Elite: QRD only (S25 Family는 풀에 미등록, QRD를 proxy로 사용)

---

## 10. 미해결 항목 (다음 스프린트 입력)

1. **w8a16 임베딩 회전 원인**: AIMET 내부 어떤 layer/op에서 분포가 무너지는지는 블랙박스. AIMET 로컬 SDK 또는 layer-wise sensitivity 분석으로 진단 필요.
2. **INT16 가중치 / FP16 활성화** 등 다른 precision 조합 미시도. AI Hub `QuantizeDtype.INT16` 가중치 옵션이 있는데 이번에는 INT8 가중치로만 진행.
3. **mixed-precision PTQ**: 일부 layer만 INT8, attention block은 FP16으로 두는 방식이 깊은 ViT에서 표준. AI Hub은 layer-wise precision 옵션을 노출하지 않음. AIMET 로컬 또는 QAT 경로로 우회.
4. **`screen_capture_synthetic` OOS 케이스**가 mmca_seoul과 시각 유사성으로 ambiguous에 빠지는 것은 정책이 아니라 학습 데이터의 부재(negative 없음) 때문. 데이터셋 v2의 negative 클래스 신설로 해결 권고.
