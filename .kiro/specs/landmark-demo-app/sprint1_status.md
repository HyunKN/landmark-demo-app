# Sprint 1 상태 메모

작성일: 2026-05-16
스프린트 상태: **동결 (frozen)**

이 문서는 spec 폴더의 운영 측 메모다. 상세 보고서는 다음 문서를 참조한다.

- 전체 정리: [`docs/sprint1_overview.md`](../../../docs/sprint1_overview.md)
- 양자화·NPU 깊이 보고: [`docs/sprint1_int8_npu_report.md`](../../../docs/sprint1_int8_npu_report.md)
- 재현 명령어: [`docs/runbook.md`](../../../docs/runbook.md)

---

## 1. 동결 상태

| 영역 | 상태 |
|---|---|
| 데모 앱 정책 (matched/ambiguous/out_of_scope/low_quality) | 완료 + 회귀 통과 |
| 회귀 fixture (`demo-regression-v1`) | 완료, `manual_image_cases` 15건 |
| MobileCLIP2-S4 ONNX export + handoff package | 완료 (`mobile_artifacts/`) |
| PC ORT dynamic INT8 양자화 | 완료, 무손실 baseline 확보 |
| 로컬 ORT static QDQ 양자화 | RAM 한계로 실험 상태 동결 |
| Qualcomm AI Hub 디바이스 측정 | 완료 — 3 디바이스, NPU 100% 탑재, latency 확정 |
| AI Hub w8a16 양자화 정확도 | 한계 도달 (mode collapse, 알고리즘 적합도) |
| 데모 앱 시연 (PC + 같은 Wi-Fi 폰) | 동작 확인 |

## 2. 발표용 한 줄

NPU 위 latency를 정직하게 확보했고, INT8 PTQ 정확도 한계의 진짜 원인을 결정적으로 진단함.

| 메시지 | 수치 |
|---|---|
| NPU latency (warm) | S23 518 ms / S24 390 ms / S25(proxy) 374 ms |
| NPU layer 비율 | 1014 / 1014 (100% 탑재, fallback 0) |
| 그래프 동치성 (AI Hub FP32) | cos 0.9995 |
| 양자화 정확도 (AI Hub w8a16) | cos 0.076~0.087 (calibration 양 무관) |
| 로컬 PC INT8 baseline | cos 0.999, top-1 100% |

## 3. Sprint 2로 넘기는 결정점

- (A) **백본 교체 — MobileCLIP2-S4 → S3** (권장 1순위)
- (B) **AIMET QAT** 또는 layer-wise mixed precision
- (C) **데이터셋 v2** (negative 클래스 신설, 한국어 caption, view/quality 다양화)

권장 동선: A + C 동시 진행 → C 끝나는 시점에 B 결정. 자세한 근거는 `docs/sprint1_int8_npu_report.md` §8 참조.

## 4. 미해결 항목 (Sprint 2 입력)

- INT8 PTQ에서 임베딩을 회전시키는 layer/op 식별
- INT16 가중치 / FP16 활성화 / mixed-precision 시도
- `screen_capture_synthetic` OOS의 mmca_seoul 방향 흐름 (학습 데이터 negative 부재)
- iPhone Core ML 측정 (Mac 환경 확보 시)

## 5. 비포함 (의도적, Sprint 1 범위 밖)

- Flutter UI / 모바일 네이티브 앱
- Multilingual-e5-small 통합
- 사용자 인증, 멀티유저
- 자동 confusion matrix UI
- 자동 모델 업데이트
