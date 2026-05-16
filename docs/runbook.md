# Runbook — Sprint 1 재현 명령

이 문서는 Sprint 1의 각 단계를 다시 돌릴 수 있도록 명령어만 모은 카드다. 모든 명령은 Windows cmd 기준이고, repo root는 `landmark-demo-app/`이다.

---

## 0. 사전 준비

### 의존성

```cmd
python -m pip install -e .
python -m pip install qai-hub onnxruntime>=1.24
```

### Qualcomm AI Hub 인증 (1회)

```cmd
qai-hub configure --api_token <YOUR_TOKEN>
```

### 디스크 주의

C 드라이브 여유가 5 GB 이하면 양자화/AI Hub 업로드가 ENOSPC로 죽는다. 모든 스크립트는 자동으로 여유 큰 드라이브(예: D)로 우회한다.

---

## 1. 데모 앱 실행

```cmd
python run.py
```

같은 Wi-Fi 폰에서 시연:

```cmd
netsh advfirewall firewall add rule name="Streamlit 8501" dir=in action=allow protocol=TCP localport=8501
```

폰 Safari/Chrome에서 `http://<노트북_IP>:8501`. 시연 끝나면:

```cmd
netsh advfirewall firewall delete rule name="Streamlit 8501"
```

---

## 2. 자산 빌드 (학습 산출물 변경 시)

```cmd
python scripts/build_assets.py ^
    --checkpoint .\best.pt ^
    --data-root ..\Dataset ^
    --landmark-info .\assets\landmark_info.json ^
    --output-dir .\assets
```

산출물: `assets/prototype_index.json`, `assets/landmark_text_index.json`, `assets/hero_images/*.jpg`

---

## 3. ONNX export

```cmd
python scripts/export_mobile_onnx.py ^
    --checkpoint .\best.pt ^
    --assets-dir .\assets ^
    --output-dir .\mobile_artifacts ^
    --opset 17
```

산출물: `mobile_artifacts/landmark_encoder.onnx` (+ `.data`), `preprocessing.json`, `labels_master.json`, `prototype_index.json`, `manifest.json`

---

## 4. 회귀 fixture 동결

```cmd
python scripts/freeze_regression_inputs.py
```

산출물: `assets/regression_inputs/positive/<class>.jpg` 13장, `assets/regression_inputs/out_of_scope/*` 2장. `tests/fixtures/demo_regression_v1.json`의 `manual_image_cases`를 갱신.

---

## 5. 로컬 ORT dynamic INT8 양자화

```cmd
python scripts/quantize_mobile_onnx.py ^
    --src .\mobile_artifacts ^
    --dst .\mobile_artifacts_int8
```

산출물: `mobile_artifacts_int8/landmark_encoder.onnx` (+ `.data`), `manifest.json`. PC CPU 무손실 baseline용.

---

## 6. PC 회귀 비교

```cmd
python scripts/quantization_regression.py ^
    --fp32-dir .\mobile_artifacts ^
    --int8-dir .\mobile_artifacts_int8 ^
    --assets-dir .\assets ^
    --runs 5 --warmup 1
```

산출물: `mobile_artifacts_int8/quantization_regression_report.json`. 한 번에 cosine, top-1/3, decision, latency 비교.

---

## 7. 단위 테스트

```cmd
python -m pytest -q
```

기대: `3 passed`. 정책 분기, fusion, name search 회귀.

---

## 8. AI Hub 디바이스 가용성

```cmd
python scripts/aihub_check_devices.py
```

S23/S24/S25 SoC별 디바이스 리스트와 권장 타겟을 출력.

---

## 9. AI Hub 풀 파이프라인

### A. 우리 Dataset에서 calibration 260장으로 양자화 + 3 디바이스 측정

```cmd
python scripts/aihub_run_jobs.py ^
    --quant-source fp32-aihub ^
    --calibration-per-class 20 ^
    --output .\mobile_artifacts_int8\aihub_report_calib260.json
```

흐름: FP32 ONNX zip 업로드 → static-shape compile → quantize_job (w8a16, calibration 260장) → 3 디바이스 compile → profile + inference. 큐 시간 포함 ~30~50분.

### B. fixture 15장만으로 양자화 (1차 시도 재현용)

```cmd
python scripts/aihub_run_jobs.py ^
    --quant-source fp32-aihub ^
    --calibration-per-class 0 ^
    --output .\mobile_artifacts_int8\aihub_report_calib15.json
```

`--calibration-per-class 0`은 fixture 입력을 calibration으로 재사용 (1차 시도와 동일 구성).

### C. 우리가 만든 ORT INT8을 AI Hub에 그대로 던지기 (실험용)

```cmd
python scripts/aihub_run_jobs.py ^
    --quant-source ort-int8 ^
    --output .\mobile_artifacts_int8\aihub_report_ortint8.json
```

QAIRT가 dynamic INT8 op를 받지 않아 컴파일 실패하는 케이스를 보존하려면 사용. 정상 경로는 A.

---

## 10. PC ONNX baseline 단독 벤치마크

```cmd
python scripts/benchmark_mobile_artifact.py
```

산출물: `mobile_artifacts/benchmark_report.json`. PC FP32 / ONNX warm/cold latency, 메모리 사용량, top-3 exact 일치율.

---

## 11. 정리 / 청소

작업 도중 D 드라이브에 쌓이는 임시 폴더 제거:

```cmd
python -c "from pathlib import Path; import shutil; t=Path('D:/'); pats=['aihub_upload_*','lm_quant_*','lm_static_*']; [(shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)) for pat in pats for p in t.glob(pat)]; print('clean')"
```

---

## 12. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `OSError: [Errno 28] No space left on device` | C 드라이브 부족. 위 11번으로 임시 폴더 정리 후 재시도. 우리 스크립트는 자동으로 D로 우회하지만, 외부 라이브러리 호출 직후 이미 ENOSPC가 났다면 디스크부터 비울 것. |
| `QAIRT converter failed with exit code 255` | ORT dynamic INT8 (`DynamicQuantizeLinear`/`QGemm`)을 던졌을 때 발생. AI Hub `quantize_job`을 거쳐 표준 QDQ로 만든 뒤 컴파일하는 정상 경로(A)로 우회. |
| `Model input 'image' has dynamic shapes` | AI Hub `quantize_job`이 dynamic axis ONNX를 거절. 스크립트가 자동으로 `--target_runtime onnx` static-shape compile을 먼저 돌리도록 처리됨. |
| `ValueError: Failed to find proper ai.onnx domain` | ORT 1.24의 path 기반 shape inference가 opset_import를 잃는 케이스. `quantize_mobile_onnx.py`/`quantize_static_qdq.py`가 monkeypatch로 보존. |
| `BFCArena allocation failure` (calibrator) | ViT-H급 + ORT static QDQ + 16 GB 호스트 RAM 한계. 클래스당 calibration 수를 더 줄이거나 AI Hub 경로로 우회. |
| AI Hub 인증 오류 | `qai-hub configure --api_token <TOKEN>` 재실행. |

---

## 13. 자주 묻는 것

**Q. NPU latency를 다시 측정하려면?**
A. 9-A를 다시 실행하면 됨. 같은 모델/입력이면 ±수 ms 차이. AI Hub 큐 상태에 따라 시간만 다름.

**Q. 다른 Snapdragon 디바이스로 바꾸려면?**
A. `scripts/aihub_run_jobs.py`의 `DEVICE_TARGETS` 리스트만 수정. 가용 디바이스 이름은 8번으로 확인.

**Q. fixture만 입력으로 calibration하면 안 되나?**
A. 가능하지만 mode collapse 영역. Sprint 1 검증 결과 calibration 사이즈가 결정 변수가 아니므로 실험적 의미만 있음.

**Q. AI Hub w8a16에서 mode collapse를 푸는 길은?**
A. Sprint 1 범위 밖. 백본 교체(S3) 또는 QAT, 또는 layer-wise mixed precision 시도가 다음 후보. 보고서 9장 참조.
