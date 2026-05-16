# Dataset v2 Retraining Plan

작성일: 2026-05-16

## 목적

이번 보강의 목적은 두 가지다.

1. 자연어 검색을 위해 영어/한국어 검색 표현, 시각 특징, 혼동 방지 문장을 명시한다.
2. 광화문-청와대처럼 넓은 궁궐권 사진에서 생기는 혼동을 hard case로 고정하고, 재학습/정책 검증 기준을 만든다.

## 적용된 변경

- `assets/landmark_text_catalog_v2.json` 추가
  - 13개 랜드마크별 영어/한국어 alias
  - 사용자식 query
  - visual feature
  - contrast caption
- `scripts/build_assets.py` 수정
  - 긴 composite prompt 하나 대신 여러 짧은 prompt를 각각 encoding한 뒤 평균 embedding 생성
  - text catalog가 있으면 `landmark_text_index.json`에 반영
- `scripts/augment_dataset_captions_v2.py` 추가
  - `Dataset/*/labels.json`에 `text_ko`, `function`, `class_visual_anchor`, `contrast_with` 보강
  - 실행 전 `Dataset/_label_backups/<timestamp>/.../labels.json` 백업 생성
- `tests/fixtures/demo_regression_v1.json`에 광화문 hard image case 추가
- `assets/regression_inputs/hard_cases/`에 광화문 wide/crop fixture 추가
- `assets/landmark_text_index.json` 재생성

## 데이터셋 변경 결과

실행 명령:

```powershell
python scripts\augment_dataset_captions_v2.py --data-root ..\Dataset --catalog assets\landmark_text_catalog_v2.json
```

결과:

- 총 13개 클래스
- 총 4,699개 record
- confirmed 4,457개
- 모든 record에 한국어 caption 존재
- confirmed record에는 `function`, `class_visual_anchor`, `contrast_with` 추가
- 백업 위치: `Dataset/_label_backups/20260516_204904/`

주의: 현재 서버 학습 코드 `landmark_candidate.train`은 image classification 학습이며 caption을 직접 사용하지 않는다. 따라서 caption 보강은 즉시 자연어 text index 품질에 반영되고, image classifier 재학습에는 label/status/image distribution만 직접 영향을 준다. caption을 학습에 쓰려면 Sprint 2에서 image-text contrastive 또는 text tower LoRA 학습 코드를 추가해야 한다.

## 광화문 hard case 진단

대상 이미지: `assets/regression_inputs/hard_cases/gwanghwamun_wide_with_bluehouse_background.jpg`

| crop | top-1 | top-2 | decision | 해석 |
|---|---:|---:|---|---|
| full | cheongwadae 0.480 | gwanghwamun 0.281 | ambiguous | 넓은 palace scene에서는 청와대 방향으로 끌리지만 확정은 안 함 |
| no top 40% | cheongwadae 0.339 | changgyeonggung 0.313 | out_of_scope | 후경 제거 후에도 낮은 신뢰도 혼동 발생 |
| left gate focus | gwanghwamun 0.664 | deoksugung 0.307 | matched | 주 피사체 crop에서는 광화문 회복 |

정책 결론:

- wide/mixed scene은 `cheongwadae matched`로 확정 표시하면 안 된다.
- object-focused crop에서는 `gwanghwamun`이 top-1이어야 한다.
- Sprint 2에서는 class mean prototype 1개보다 multi-prototype 또는 hard-negative 재학습이 필요하다.

## 서버 재학습 절차

서버의 `Dataset`을 이번 로컬 `Dataset`과 동기화한 뒤 실행한다. 이미지 파일은 거의 그대로고 `labels.json`이 바뀐 상태이므로, 서버에는 각 클래스의 `labels.json` 업데이트가 핵심이다.

```bash
cd /workspace/landmark-assistant-model
source .venv/bin/activate

export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
python -m landmark_candidate.split_data \
  --data-root "$DATA_ROOT" \
  --out splits/kfold_seed20260513.json \
  --seed 20260513 \
  --folds 5 \
  --test-ratio 0.15

EXPORT_ONNX=0 GPUS=1,2,3,4 NPROC=4 bash scripts/run_candidate_tmux.sh mobileclip2_s4 0
```

학습이 끝나면:

```bash
tail -n 120 logs/jongno-mobileclip2_s4-fold0.log
cat runs/<RUN_NAME>/metrics.json
```

## 학습 후 데모 자산 재생성

새 `best.pt`를 `landmark-demo-app/best.pt`로 옮긴 뒤:

```powershell
python scripts\build_assets.py ^
  --checkpoint .\best.pt ^
  --data-root ..\Dataset ^
  --landmark-info .\assets\landmark_info.json ^
  --output-dir .\assets ^
  --device cpu

python -m pytest -q
python run.py
```

## 다음 개선

- 광화문/청와대/근정문 계열 hard negative 이미지 추가 수집
- class mean prototype을 multi-prototype으로 확장
- OOS/negative class를 데이터셋 v2에 별도 split으로 추가
- caption을 실제 학습에 쓰는 image-text contrastive 또는 text tower LoRA 경로 설계
