# Known Issues — 트러블슈팅 기록

작성일: 2026-05-18
마지막 갱신: 2026-05-18

이 문서는 개발/시연 중 발견된 문제를 기록하고, 해결 여부와 원인을 추적한다.
새 이슈 발견 시 아래 표에 추가하고 상태를 갱신한다.

---

## 이슈 목록

| # | 상태 | 카테고리 | 제목 | 발견일 | 해결일 |
|---|---|---|---|---|---|
| 1 | ✅ 해결 | 텍스트 검색 | "경복궁" 검색 → mmca_seoul 1위 | 2026-05-18 | 2026-05-18 |
| 2 | ⚠️ 미해결 | 텍스트 검색 | "한복 입고 사진 찍는 곳" → mmca_seoul | 2026-05-18 | — |
| 3 | ✅ 해결 | ONNX 모드 | ONNX 모드 자연어 검색이 keyword만 사용 (text encoder 미연결) | 2026-05-16 | 2026-05-18 |
| 4 | ⚠️ 미해결 | 이미지 검색 | 광화문 wide shot → cheongwadae/changgyeonggung 혼동 | 2026-05-17 | — |
| 5 | ✅ 해결 | 양자화 | AI Hub w8a16 INT8 임베딩 공간 붕괴 (cos 0.08) | 2026-05-16 | 진단 완료, 수정은 Sprint 2 |
| 6 | ✅ 해결 | 인프라 | C 드라이브 ENOSPC로 양자화/AI Hub 업로드 실패 | 2026-05-16 | 2026-05-16 |
| 7 | ✅ 해결 | 인프라 | ONNX 세션 빌드 5분+ (INT8 모드 부팅 느림) | 2026-05-18 | 원인 진단 (ViT-S4 크기 + RAM 압박) |
| 8 | ✅ 해결 | UI | 텍스트 검색 후 "자세히 보기" 클릭 시 페이지 안 넘어감 | 2026-05-16 | 2026-05-18 |
| 9 | ⚠️ 미해결 | 텍스트 검색 | 근정문 검색 시 최대 40%, 0% 결과들 나옴 | 2026-05-18 | — |

---

## 상세 기록

### #1 ✅ "경복궁" 검색 → mmca_seoul 1위

**증상**: `경복궁` 입력 시 국립현대미술관이 93%로 1위, 광화문 91%, 근정문 89%.

**원인**:
- mmca의 text catalog에 "경복궁 옆 미술관", "museum near Gyeongbokgung", "경복궁 인근" 등 위치 설명이 포함돼 있었음
- "경복궁"이라는 query가 keyword 매칭 + text embedding 모두에서 mmca를 강하게 당김
- 13개 클래스에 "경복궁 전체" 클래스가 없어서 정답 자체가 모호

**수정**:
- `landmark_text_catalog_v2.json` mmca 항목에서 경복궁 언급 전부 제거
- `landmark_info.json` mmca description_ko에서 "경복궁 옆" 제거
- `landmark_text_index.json` 재생성

**교훈**: catalog 위치 설명에 다른 랜드마크 이름을 직접 언급하면 검색 오염 발생. 위치는 행정구역/방향으로만 표현할 것.

---

### #2 ⚠️ "한복 입고 사진 찍는 곳" → mmca_seoul

**증상**: 한복 관련 query인데 미술관으로 연결됨. mmca catalog에 "한복" 관련 내용은 전혀 없음.

**원인**:
- MobileCLIP2-S4가 영어 중심 CLIP 계열이라 한국어 "사진"을 "photo"(촬영)가 아닌 "picture/painting"(그림/작품)으로 해석
- "한복 입고 그림 보는 곳 = 미술관"으로 임베딩 공간에서 연결됨
- 모델 버그가 아니라 영어 중심 CLIP의 한국어 의미 연결 한계

**완화 방향**:
- 단기: 경복궁/덕수궁/창경궁 catalog에 "한복 체험", "한복 대여", "hanbok photo" 키워드 추가
- Sprint 2: 데이터셋 v2 한국어 caption 다양화

---

### #3 ✅ ONNX 모드 자연어 검색이 keyword만 사용

**증상**: `python run.py --onnx` 또는 `--int8`로 띄우면 자연어 검색이 semantic 임베딩 없이 keyword/alias 매칭만 동작.

**원인**:
- text encoder ONNX export가 Sprint 1 초기에 미실시
- ONNX 모드에서 PyTorch checkpoint 재로드를 회피하는 설계 결정으로 text_encoder = None

**수정**:
- `scripts/export_text_encoder_onnx.py` 신규 작성 — text tower ONNX export (473 MB FP32, 229 MB INT8)
- `OnnxTextEncoder` 클래스 추가 (`src/landmark_demo/inference.py`)
- `app.py` ONNX 모드에서 OnnxTextEncoder 사용하도록 변경
- PyTorch vs ONNX cosine 1.00000 (5/5) parity 검증

---

### #4 ⚠️ 광화문 wide shot → cheongwadae/changgyeonggung 혼동

**증상**: 광화문을 넓게 찍은 사진(배경에 청와대/산/궁궐이 보임)에서 top-1이 cheongwadae로 나옴.

**원인**:
- 같은 "한국 궁궐 외관" cluster 안에서 미세 분별 실패
- gwanghwamun 학습 데이터 52장으로 적어 prototype이 정면 외관 한 점에 뭉쳐 있음
- 배경에 다른 랜드마크가 보이면 점수가 분산됨

**완화 방향**:
- Sprint 2: 광화문 데이터 보강 (100장+), view_type별 prototype 분할, parent landmark 계층 구조

---

### #5 ✅ AI Hub w8a16 INT8 임베딩 공간 붕괴

**증상**: AI Hub에서 양자화한 INT8 모델의 디바이스 임베딩이 PC FP32와 cos 0.076~0.087 (거의 직교).

**원인**:
- calibration 부족이 아님 (15장 → 260장으로 17배 확장해도 동일)
- AI Hub FP32 inference cos 0.9995로 그래프 변환 자체는 무손실
- AIMET w8a16 PTQ가 MobileCLIP2-S4의 깊은 attention/LayerNorm 분포를 보존하지 못함

**진단 완료**: PTQ 알고리즘과 ViT-S4 분포 간 적합도 문제. Sprint 2에서 QAT 또는 백본 교체로 해결.

---

### #6 ✅ C 드라이브 ENOSPC

**증상**: 양자화/AI Hub 업로드 중 `OSError: [Errno 28] No space left on device`.

**원인**: C 드라이브 여유 0~4 GB. 양자화 임시 파일이 1~3 GB 단위.

**수정**: 모든 스크립트에 자동 scratch 드라이브 선택 로직 추가 (`tempfile.tempdir = D:/`). 환경변수 `TEMP/TMP/TMPDIR`도 강제.

---

### #7 ✅ ONNX 세션 빌드 5분+ (INT8 모드 부팅 느림)

**증상**: `python run.py --int8` 실행 시 Streamlit이 5분 이상 부팅 안 됨.

**원인**:
- ViT-S4 1014 layer + INT8 QDQ pair로 노드 수 ~2000개
- ORT InferenceSession 생성 시 그래프 분석 + 메모리 plan + op fusion에 분 단위 소요
- 호스트 RAM 15.8 GB 중 가용 4~6 GB → swap 발생으로 추가 지연

**결론**: PC CPU에서 INT8 ONNX는 가성비 떨어짐. 시연은 PyTorch 모드(`python run.py`) 권장. INT8은 NPU 측정용.

---

### #8 ✅ 텍스트 검색 후 "자세히 보기" 클릭 시 페이지 안 넘어감

**증상**: 자연어 검색 결과 카드의 "자세히 보기" 버튼 클릭 시 상세 페이지로 이동 안 됨. 이미지 검색에서는 정상 동작.

**원인**:
- Streamlit 버튼은 클릭된 그 한 번의 rerun에서만 True
- 텍스트 탭은 `if run_text:` 안에서만 `render_top3`이 호출되는 구조
- 검색 버튼 클릭 → outcome 생성 → "자세히 보기" 버튼 그려짐 → 클릭 → 다음 rerun에서 `run_text=False`라 outcome이 안 만들어지고 카드 자체가 사라짐 → 클릭 핸들러가 어디로도 안 감
- 이미지 탭은 `file_uploader`가 파일을 들고 있어 매 rerun마다 추론이 다시 돌아 카드가 다시 그려져서 우연히 동작했음

**수정** (2026-05-18):
- 검색 결과를 `st.session_state["last_image_outcome"]` / `last_text_outcome`에 저장
- `render_top3`을 `if run_text:` 블록 밖으로 빼서 매 rerun마다 session_state에서 읽어 호출
- "모든 검색 초기화" 버튼에서도 두 키를 같이 정리

---

### #9 ⚠️ 근정문 검색 시 최대 40%, 0% 결과들 나옴

**증상**: "근정문" 또는 "근정전"으로 검색 시 top-1이 40% 정도이고 나머지가 0%에 가까움.

**원인 추정**:
- 근정문 클래스의 text_index 임베딩이 "근정문"이라는 query와 cosine이 낮음
- keyword 매칭으로 40% 정도 올라가지만 semantic score가 약함
- 13개 클래스 중 "경복궁 내부 문"이라는 개념이 text embedding 공간에서 독립적 위치를 못 잡음

**완화 방향**: gyeongbokgung_geunjeongmun catalog에 "근정문", "근정전", "경복궁 안쪽 문" 등 한국어 query 보강.

---

## 이슈 추가 방법

새 이슈 발견 시:
1. 위 표에 행 추가 (번호, 상태, 카테고리, 제목, 발견일)
2. 상세 기록 섹션에 증상/원인/수정(또는 방향) 작성
3. 해결되면 상태를 ✅로 변경하고 해결일 기입
