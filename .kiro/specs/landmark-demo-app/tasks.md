# 구현 계획

이 작업 목록은 `requirements.md`와 `design.md`를 직접 구현 가능한 단위로 분해한 것이다. 각 task는 30분~3시간 안에 끝낼 수 있는 크기이며, 코드 변경에 직접 매핑된다. 시연일 2026-05-18 이전에 1차 데모를 시연 가능 상태로 만드는 것이 목표이며, 학습 fold0 완료 시점부터 시작한다.

- [ ] 1. 프로젝트 골격과 의존성 설정
  - 1.1. `pyproject.toml`에 Python 3.10+, streamlit, onnxruntime>=1.18, pillow, numpy, pyyaml, tomli 의존성 정의
  - 1.2. `src/landmark_demo/__init__.py`, `tests/__init__.py` 빈 패키지 파일 생성
  - 1.3. `.gitignore`에 `assets/`, `logs/`, `.venv/`, `__pycache__/`, `*.onnx` 추가
  - 1.4. `README.md`에 5분 quick start (venv 생성 → 의존성 설치 → 자산 빌드 → `streamlit run` 절차)
  - 1.5. `config.toml` 기본값 작성: fusion 가중치, reject_threshold=0.25, asset/log 경로
  - _요구사항: 6_

- [ ] 2. Asset_Bundle 로더와 검증
  - 2.1. `src/landmark_demo/data/asset_loader.py` 구현. `LANDMARK_DEMO_ASSETS` 환경변수와 `--assets` CLI 인자, 기본 `./assets` 우선순위
  - 2.2. 4개 필수 파일 부재/파싱 실패 시 오류 화면 표시 + 이후 검색 기능 비활성화 로직
  - 2.3. 13개 Landmark_Catalog와 prototype_index/text_index/landmark_info의 landmark_id 집합 정합성 검증
  - 2.4. `tests/test_asset_loader.py`에서 정상/누락/형식오류/landmark_id 불일치 4가지 케이스 단위 테스트
  - _요구사항: 6.1, 6.2, 6.3, 6.10, 6.11_

- [ ] 3. Prototype_Index, Text_Index, Landmark_Info 데이터 클래스
  - 3.1. `src/landmark_demo/data/prototype_index.py`에 dataclass 정의와 JSON 로더, prototype 행렬 (13, 512) numpy 배열 노출
  - 3.2. `src/landmark_demo/data/text_index.py`에 dataclass와 로더, embedding 행렬 (13, 512) 노출
  - 3.3. `src/landmark_demo/data/landmark_info.py`에 dataclass, `coordinates_valid` 검증, hero_image_path 절대경로 변환
  - 3.4. 각 모듈 단위 테스트 (정상 데이터, 잘못된 embedding 차원, 잘못된 좌표)
  - _요구사항: 4.1, 4.2, 4.3, 4.4, 6.4, 6.5, 6.6_

- [ ] 4. Image_Recognizer 구현
  - 4.1. `src/landmark_demo/inference/image_recognizer.py`에 ONNX session 1회 생성, intra/inter op threads 설정
  - 4.2. Pillow 기반 전처리: 짧은 변 224 리사이즈 + center-crop + image_mean/std 정규화 + (1,3,224,224) FP32 텐서
  - 4.3. embedding 산출 후 L2-normalize, 처리 시간 ms 반환
  - 4.4. JPEG/PNG/WEBP 외 형식 거부, 10MB 초과 거부 헬퍼
  - 4.5. `tests/test_image_preprocess.py`에서 다양한 해상도/형식 입력에 대해 출력 shape과 정규화 값 검증
  - _요구사항: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8_

- [ ] 5. PyTorch ↔ ONNX parity 검증 스크립트
  - 5.1. `scripts/verify_onnx_parity.py` 구현. `--checkpoint`, `--onnx`, `--data-root`, `--samples` 인자
  - 5.2. dummy 5장 + 실제 검증 이미지 5장에 대해 PyTorch best.pt와 ONNX 출력 cosine similarity 측정
  - 5.3. 임계 0.999 미만이면 종료 코드 1, 통과 시 0
  - 5.4. 검증 결과를 stdout에 표 형식으로 출력
  - _요구사항: 6.1 (export 검증)_

- [ ] 6. Text_Encoder 구현 (MobileCLIP2 text tower)
  - 6.1. `src/landmark_demo/inference/text_encoder.py`에 OpenCLIP BPE tokenizer + ONNX session 로드
  - 6.2. 입력 문자열 → 토큰화 → ONNX 추론 → L2-normalized embedding 반환
  - 6.3. 0자/200자 경계 검증과 200자 초과 시 잘라내기
  - 6.4. text_encoder.onnx 부재 시 graceful degrade (text_score=0, keyword_score만으로 검색)
  - _요구사항: 2.1, 2.2, 2.3, 2.4_

- [ ] 7. Fusion_Ranker 구현
  - 7.1. `src/landmark_demo/inference/fusion.py`에 `compute_image_score`, `compute_text_score`, `compute_keyword_score` 함수
  - 7.2. keyword_score는 NFC + lower-case 후 부분 일치 토큰 수 비례 산출
  - 7.3. fusion_score 가중 합산 + Top-3 정렬 + Reject_Threshold 비교
  - 7.4. `config.toml`에서 가중치 로드, 합 1.0 검증, 위반 시 fail-fast
  - 7.5. `tests/test_fusion.py`에서 가중치 합 검증, fusion 정확성, Top-3 동률 처리 단위 테스트
  - _요구사항: 2.5, 2.6, 2.7, 2.8, 2.9, 3.1, 3.4, 3.8_

- [ ] 8. Name_Search_Index 구현
  - 8.1. `src/landmark_demo/data/name_search.py`에 NameEntry dataclass와 build/search 함수
  - 8.2. landmark_info의 name_ko/en/aliases 모두 entry로 펼침, NFC + lower-case 키
  - 8.3. 부분 일치 검색, 최대 10개, (longest match length, alphabetical) 정렬
  - 8.4. 검색 실행 시점 후보 0/1/2+개 분기 반환 타입
  - 8.5. `tests/test_name_search.py`에서 정확 일치, 부분 일치, 한/영 alias, 빈 입력 케이스
  - _요구사항: 5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9_

- [ ] 9. Debug_Log 구현
  - 9.1. `src/landmark_demo/logging/debug_log.py`에 JSONL 기록기 구현
  - 9.2. `LANDMARK_DEMO_LOG_PATH` 환경변수, 미지정 시 `./logs/demo.jsonl` 사용
  - 9.3. 검색 1건당 timestamp/kind/input_id/elapsed_ms/below_threshold/top3/scores 기록
  - 9.4. out-of-scope 시 raw 점수 전체 기록
  - _요구사항: 3.5, 7.1, 7.2, 7.8_

- [ ] 10. Streamlit 검색 페이지 (이미지/텍스트/이름 통합)
  - 10.1. `src/landmark_demo/app.py` Streamlit entry point, 사이드바에 페이지 라우팅
  - 10.2. `src/landmark_demo/pages/search.py`에 3가지 입력 탭 (이미지 업로드, 자연어 검색, 이름 검색)
  - 10.3. 이미지 탭: 파일 업로드 위젯, 카메라 input, 처리 시간 ms 표시
  - 10.4. 자연어 탭: 텍스트 입력, 검색 버튼, 200자 잘라내기 안내
  - 10.5. 이름 탭: `st.selectbox`로 자동완성 후보 표시, 후보 수에 따른 분기 메시지
  - 10.6. 모든 탭 공통: 진행 표시기 (spinner), 5초 지연 알림
  - _요구사항: 1.1, 1.7, 1.8, 1.9, 2.1, 2.4, 5.3, 5.5, 7.6, 7.7_

- [ ] 11. 결과 카드 UI
  - 11.1. `src/landmark_demo/ui/components.py`에 `render_top3_card` 함수
  - 11.2. 각 카드에 한국어 이름, 백분율, 신뢰도 막대 (CSS bar) 표시
  - 11.3. 카드 클릭 시 정보 페이지로 라우팅 (`st.session_state`로 selected_landmark_id 전달)
  - 11.4. Reject_Threshold 미만 시 카드 대신 "범위 외 입력" 안내 + "후보 보기" 토글
  - 11.5. 토글 활성 시 일반 Top-3 카드 렌더
  - _요구사항: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7_

- [ ] 12. Streamlit 정보 페이지
  - 12.1. `src/landmark_demo/pages/landmark.py`에 정보 페이지 렌더 로직
  - 12.2. landmark_id로 landmark_info에서 메타데이터 조회
  - 12.3. 한/영 이름, description, alias 목록, tags 표시
  - 12.4. coordinates_valid면 좌표 텍스트 + 외부 지도 링크 (Naver/Google Maps URL)
  - 12.5. coordinates_valid가 false면 "위치 정보 없음"
  - 12.6. hero_image 로드, 실패 시 placeholder + Debug_Log 기록
  - 12.7. 메타데이터 누락 시 오류 화면 + 검색 페이지로 돌아가기 링크
  - _요구사항: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 13. 개발자 모드 패널과 보조 기능
  - 13.1. `src/landmark_demo/ui/components.py`에 `render_dev_panel` 함수
  - 13.2. 사이드바 토글로 개발자 모드 on/off
  - 13.3. on이면 가장 최근 검색의 13개 landmark별 image/text/keyword/fusion score 정렬 표
  - 13.4. "검색 초기화" 버튼: 입력 폼 텍스트, 결과 카드, 처리 시간, 이미지 미리보기 4개 요소 초기화
  - 13.5. 일부 초기화 실패 시 성공한 것은 유지하고 실패한 것만 Debug_Log 기록
  - _요구사항: 7.3, 7.4, 7.5_

- [ ] 14. landmark_info.json 시드 작성 도우미
  - 14.1. `scripts/seed_landmark_info.py` 구현. `--labels-master`, `--classes`, `--output` 인자
  - 14.2. labels_master.json에서 13개 클래스의 name_ko/en 자동 추출
  - 14.3. description/aliases/latitude/longitude/hero_image_path/tags 빈 필드 갖는 초안 JSON 출력
  - 14.4. README에 사람이 채우는 절차 문서화
  - _요구사항: 6.6_

- [ ] 15. Asset 빌드 파이프라인
  - 15.1. `scripts/build_assets.py` 구현. 학습 산출물 디렉토리에서 best.pt + classes.json + config.yaml 로드
  - 15.2. ONNX export (이미 export됐으면 skip), reparameterize 강제
  - 15.3. trainval split 전체에 대해 image embedding 산출
  - 15.4. 클래스별 평균 + L2-normalize → prototype_index.json
  - 15.5. landmark_info의 description+keywords를 text encoder로 인코딩 → landmark_text_index.json
  - 15.6. dataset_fingerprint를 split manifest에서 추출
  - 15.7. parity 검증 자동 호출 (Task 5)
  - _요구사항: 6.1, 6.4, 6.5_

- [ ] 16. Reject_Threshold calibration 노트북
  - 16.1. `scripts/calibrate_threshold.ipynb` Jupyter 노트북 (또는 Python 스크립트) 작성
  - 16.2. holdout_non_confirmed 242장과 test 669장에서 1순위 cosine similarity 산출
  - 16.3. 두 분포 히스토그램 + ROC 분석
  - 16.4. 권장 threshold 출력 + config.toml 갱신 안내
  - _요구사항: 3.4, 3.5_

- [ ] 17. Hero 이미지 선정 도우미
  - 17.1. `scripts/select_hero_images.py` 구현. trainval에서 quality_status=ok, view_type=exterior 이미지 후보 추출
  - 17.2. 클래스별 상위 5장을 thumbnails로 표시 (HTML 갤러리)
  - 17.3. 사람이 1장 선정 후 assets/hero_images/로 복사
  - _요구사항: 6.6, 4.5_

- [ ] 18. End-to-end 시연 리허설
  - 18.1. 시연 시나리오 5개 작성 (정상 이미지, 정상 텍스트, 이름 정확 일치, 이름 부분 일치, 범위 외 입력)
  - 18.2. 각 시나리오를 데모 앱에서 실행하고 예상 결과와 일치 여부 기록
  - 18.3. Reject_Threshold 동작 확인
  - 18.4. Debug_Log 결과 검토
  - 18.5. 시연 환경(노트북) 추론 시간 측정, 1초 SLA 통과 확인
  - _요구사항: 1.6, 3.4, 7.1, 시연 ladder_

## 작업 순서 권장

1. Task 1 (프로젝트 골격)
2. Task 14 (landmark_info 시드) — 사람이 description 작성 시작
3. Task 5 (parity 검증) → Task 15 (asset 빌드) — 학습 완료 후 즉시
4. Task 2-9 (서비스 레이어) — 병렬 가능
5. Task 10-13 (UI) — 서비스 레이어 완료 후
6. Task 16, 17 (calibration, hero 선정)
7. Task 18 (리허설)

## 명시적 비포함 (Sprint 2)

- 휴대폰 ONNX Runtime Mobile 빌드
- Flutter UI
- INT8 양자화
- 자동 confusion matrix UI
- multilingual-e5-small 통합
- 사용자 인증, 멀티유저
- 자동 모델 업데이트
