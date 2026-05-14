# Requirements Document

## Introduction

`landmark-demo-app`은 학습 중인 MobileCLIP2-S4 image encoder 체크포인트(ONNX export 산출물)를 노트북 또는 휴대폰에서 로컬 실행하여, 종로구 13개 한국 전통/문화 랜드마크의 인식·검색 동작을 시연하기 위한 임시 데모 애플리케이션이다. 본 데모는 ADR-0001 Demo Ladder의 두 번째 단(노트북 ONNX Runtime + 간단한 로컬 웹 UI) 또는 네 번째 단(Python local demo)에 해당하며, 정식 Flutter 앱이 아닌 빠른 검증용 surface다.

데모는 ADR-0003에서 채택한 dual-engine retrieval(image score + text score + metadata keyword score fusion)과 ADR-0004의 image recognizer 산출물(landmark_encoder.onnx + JSON prototype index)을 그대로 사용하며, 외부 네트워크 호출 없이 로컬 자산만으로 다음 5가지 기능을 제공한다.

1. 이미지 업로드/촬영 기반 Top-3 landmark 인식과 신뢰도 표시
2. 한국어/영어 자연어 질의 기반 landmark 검색
3. 식별된 landmark의 메타데이터 정보 페이지(이름, 설명, 위치, 대표 사진)
4. 장소 이름 직접 검색(자동완성, 대소문자 무시, 한/영 alias)
5. 시연 보조 기능(처리시간 표시, 디버그 로그, out-of-scope 안내, 입력 초기화)

시연 목표일은 2026-05-18이며, 본 데모는 정식 앱과 별개의 검증용 산출물로서 시연 후 폐기되거나 정식 Flutter 앱의 참조 구현으로만 활용된다.

## Glossary

- **Demo_App**: 본 명세의 대상이 되는 임시 데모 애플리케이션 전체. Python 기반 로컬 웹 UI(Streamlit 또는 Gradio) 또는 ONNX Runtime Mobile을 통한 휴대폰 build 형태로 구동된다.
- **Image_Recognizer**: ONNX Runtime으로 `landmark_encoder.onnx`(MobileCLIP2-S4 image encoder, ADR-0004)를 실행하여 입력 이미지로부터 512차원 L2-normalized embedding을 산출하는 모듈.
- **Text_Encoder**: 자연어 질의 문자열을 embedding 벡터로 변환하는 모듈. ADR-0003에 따라 MobileCLIP2 text encoder 또는 multilingual-e5-small 중 build-time에 선택된 단일 모델을 사용한다.
- **Prototype_Index**: 13개 landmark별 image prototype embedding과 landmark_id를 저장한 JSON 파일(`prototype_index.json`).
- **Text_Index**: 13개 landmark별 사전 계산된 text embedding과 keyword/alias를 저장한 JSON 파일(`landmark_text_index.json`).
- **Landmark_Info_Store**: 13개 landmark의 한국어 이름, 영어 이름, alias, 한국어 설명, 위도/경도, 대표 이미지 경로를 저장한 JSON 파일(`landmark_info.json`).
- **Name_Search_Index**: 한국어 이름, 영어 이름, alias 목록을 검색 키로 사용하는 in-memory 문자열 인덱스.
- **Fusion_Ranker**: image_score, text_score, keyword_score를 landmark_id 단위로 가중 합산하여 fusion_score를 산출하고 Top-3와 reject 판단을 반환하는 모듈(ADR-0003).
- **Debug_Log**: 사용자 화면에 노출하지 않고 별도 로그 파일 또는 개발자 모드 패널에만 기록되는 진단 로그(ADR-0001 "범위 외 입력은 화면에 표시하지 않고 debug log에만 남긴다" 정책).
- **Reject_Threshold**: 1순위 fusion_score가 이 값 미만이면 화면에 Top-3를 표시하지 않고 out-of-scope로 처리하는 임계값. 초기값 0.25, 설정 파일에서 변경 가능.
- **Asset_Bundle**: `landmark_encoder.onnx`, `prototype_index.json`, `landmark_text_index.json`, `landmark_info.json`, 그리고 선택적으로 `text_encoder.onnx`/tokenizer 파일을 포함하는 로컬 자산 묶음.
- **Landmark_Catalog**: 본 데모가 인식 대상으로 삼는 13개 landmark의 고정 집합. `bohyunsanshingak`, `changgyeonggung`, `cheonggyecheon`, `cheongwadae`, `deoksugung`, `gwanghwamun`, `gyeongbokgung_geunjeongmun`, `jogyesa`, `jongmyo_shrine`, `mmca_seoul`, `myeongdong_cathedral`, `naksan_park`, `statue_of_king_sejong`.

## Requirements

### Requirement 1: 이미지 입력 기반 랜드마크 인식

**User Story:** 데모 시연자로서, 시연 현장에서 촬영하거나 미리 준비한 사진을 업로드하면 어떤 랜드마크인지 Top-3와 신뢰도로 확인하고 싶다. 그래야 학습된 모델의 인식 능력을 한 번에 시각적으로 보여줄 수 있다.

#### Acceptance Criteria

1. WHEN 사용자가 JPEG, JPG, PNG, WEBP 중 하나의 형식으로 단일 이미지 파일을 업로드, THE Demo_App SHALL 해당 파일을 이미지 입력 후보로 수신한다.
2. IF 업로드된 파일의 크기가 10MB를 초과하면, THEN THE Demo_App SHALL "10MB 이하 이미지만 지원합니다" 메시지를 표시하고 인식 절차를 중단한다.
3. IF 업로드된 파일이 JPEG, JPG, PNG, WEBP 중 어느 형식도 아니면, THEN THE Demo_App SHALL "지원하지 않는 이미지 형식입니다" 메시지를 표시하고 인식 절차를 중단한다.
4. WHEN 이미지 입력이 수신, THE Image_Recognizer SHALL 짧은 변(short side)을 224 픽셀로 리사이즈한 뒤 224x224 center-crop을 적용한다.
5. WHEN center-crop이 완료, THE Image_Recognizer SHALL image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711]로 채널별 정규화를 적용한다.
6. WHEN 정규화가 완료, THE Image_Recognizer SHALL ONNX Runtime CPU execution provider를 사용해 512차원 L2-normalized image embedding을 단일 이미지당 1초 이내에 산출한다.
7. WHEN image embedding이 산출, THE Fusion_Ranker SHALL Prototype_Index의 각 landmark prototype과 cosine similarity를 계산하여 image_score[landmark_id]를 생성한다.
8. WHERE 사용자가 모바일 기기에서 카메라 촬영 모드를 사용하면, THE Demo_App SHALL 촬영된 이미지를 갤러리 업로드와 동일한 전처리·추론 경로로 처리한다.
9. WHEN 이미지 입력이 수신된 시점부터 결과 화면이 사용자에게 표시될 때까지의 처리 시간을 측정하여, THE Demo_App SHALL 결과 화면에 0 이상의 정수 밀리초로 표시한다.

### Requirement 2: 자연어 텍스트 검색

**User Story:** 데모 시연 청중으로서, "성곽 옆 돌담 공원"처럼 랜드마크 이름을 모르는 자연어 설명을 입력해서 어떤 장소인지 알아내고 싶다. 그래야 단순 분류기가 아닌 검색 어시스턴트라는 가치를 체감할 수 있다.

#### Acceptance Criteria

1. WHEN 사용자가 자연어 검색창에 한국어 또는 영어 문자열을 입력하고 검색을 실행, THE Demo_App SHALL 해당 문자열을 텍스트 검색 입력으로 수신한다.
2. IF 입력 문자열의 trim 결과 길이가 0자이면, THEN THE Demo_App SHALL "검색어를 입력하세요" 메시지를 표시하고 검색을 중단한다.
3. IF 입력 문자열의 길이가 200자를 초과하면, THEN THE Demo_App SHALL 입력을 앞에서부터 200자로 잘라내고 "검색어가 200자로 잘렸습니다" 안내 메시지를 함께 표시한다.
4. WHEN 텍스트 입력이 수신, THE Text_Encoder SHALL 입력 문자열을 단일 embedding 벡터로 인코딩한다.
5. WHEN text embedding이 산출, THE Fusion_Ranker SHALL Text_Index의 각 landmark text embedding과 cosine similarity를 계산하여 text_score[landmark_id]를 생성한다.
6. WHEN 텍스트 입력이 수신, THE Fusion_Ranker SHALL 입력 문자열을 NFC 정규화 후 lower-case로 변환하고 Landmark_Info_Store의 한국어 이름, 영어 이름, alias, keyword 필드와 부분 일치 검색하여 일치한 토큰 수에 비례하는 keyword_score[landmark_id]를 생성한다.
7. WHEN text_score와 keyword_score가 모두 산출, THE Fusion_Ranker SHALL `fusion_score = w_image * image_score + w_text * text_score + w_keyword * keyword_score` 공식으로 fusion_score를 계산하며, 이미지 입력이 없는 텍스트-단독 검색에서는 w_image를 0으로 둔다.
8. THE Fusion_Ranker SHALL w_image, w_text, w_keyword 가중치를 설정 파일에서 로드하며, 세 가중치의 합은 1.0이고 각 값은 0.0 이상 1.0 이하다.
9. IF 설정 파일의 w_image, w_text, w_keyword 중 하나라도 0.0 미만이거나 1.0을 초과하거나, 세 값의 합이 1.0과 1e-6 이상 차이나면, THEN THE Demo_App SHALL 시작 시점에 "fusion 가중치 설정 오류" 메시지를 표준 오류로 출력하고 Requirement 6의 자산 검증 실패와 동일한 절차로 처리한다.

### Requirement 3: Top-3 결과 표시 및 Out-of-Scope 처리

**User Story:** 데모 시연자로서, 모든 검색 결과를 동일한 형태의 Top-3 카드로 보여주고 싶고, 모델이 자신 없는 입력은 함부로 단정하지 않게 만들고 싶다. 그래야 시연 중 잘못된 단정 출력으로 신뢰를 잃지 않는다.

#### Acceptance Criteria

1. WHEN fusion_score가 산출, THE Demo_App SHALL fusion_score 내림차순으로 상위 3개 landmark_id와 점수를 선정한다.
2. THE Demo_App SHALL 선정된 각 landmark_id의 fusion_score를 0 이상 100 이하 정수 백분율로 변환하여 표시하며, 변환식은 `percentage = round(clip(fusion_score, 0, 1) * 100)`이다.
3. THE Demo_App SHALL 각 결과 카드에 landmark의 한국어 이름, 백분율, 그리고 신뢰도 막대(0~100% 시각화)를 표시한다.
4. IF 1순위 landmark의 fusion_score가 Reject_Threshold 미만이면, THEN THE Demo_App SHALL 사용자 화면 기본 상태에서 Top-3 카드 대신 "범위 외 입력으로 판단됩니다" 안내와 "후보 보기" 토글 버튼만 표시한다.
5. WHEN 1순위 fusion_score가 Reject_Threshold 미만, THE Demo_App SHALL 입력 종류, 입력 식별자, Top-3 후보 landmark_id, raw fusion_score, image_score, text_score, keyword_score를 Debug_Log에 기록한다.
6. WHERE 사용자가 결과 카드 중 하나를 클릭 또는 탭하면, THE Demo_App SHALL 해당 landmark_id의 정보 페이지로 이동한다.
7. WHERE 사용자가 "후보 보기" 토글 버튼을 활성화하면, THE Demo_App SHALL 1순위 fusion_score가 Reject_Threshold 미만인 경우에도 Top-3 카드를 표시하고 정보 페이지로의 이동을 허용한다.
8. THE Demo_App SHALL 한 번의 검색에서 동일한 landmark_id를 Top-3에 두 번 이상 표시하지 않는다.

### Requirement 4: 장소 정보 페이지

**User Story:** 데모 시연자로서, 인식되거나 검색된 장소의 설명, 위치, 대표 사진을 한 화면에서 보여주고 싶다. 그래야 인식 결과가 의미 있는 정보로 연결되는 흐름을 청중에게 전달할 수 있다.

#### Acceptance Criteria

1. WHEN 사용자가 landmark_id의 정보 페이지를 요청, THE Demo_App SHALL Landmark_Info_Store에서 해당 landmark_id에 대응하는 메타데이터 레코드를 조회한다.
2. THE Demo_App SHALL 정보 페이지에 한국어 이름, 영어 이름, 한국어 설명 본문, 위도, 경도, 대표 이미지 1장, alias 목록, landmark_id를 표시한다.
3. IF Landmark_Info_Store에 요청된 landmark_id가 존재하지 않으면, THEN THE Demo_App SHALL "메타데이터를 찾을 수 없습니다" 오류 메시지를 표시하고 결과 화면으로 돌아가는 링크를 제공한다.
4. THE Demo_App SHALL 정보 페이지 렌더 직전에 latitude와 longitude를 검증하여 단일 boolean 플래그 coordinates_valid를 산출하며, 두 값이 모두 부동소수점이고 latitude는 -90 이상 90 이하, longitude는 -180 이상 180 이하이면 coordinates_valid는 true다.
5. WHERE coordinates_valid가 true이면, THE Demo_App SHALL 좌표 텍스트와 함께 정적 지도 이미지 또는 외부 지도 링크 중 하나를 표시한다.
6. IF coordinates_valid가 false이면, THEN THE Demo_App SHALL 좌표 영역에 "위치 정보 없음"을 표시한다.
7. IF 해당 landmark 레코드의 대표 이미지 경로가 부재하거나 파일 로드에 실패하면, THEN THE Demo_App SHALL 대표 이미지 영역에 placeholder 이미지를 표시하고 Debug_Log에 이미지 로드 실패를 기록한다.

### Requirement 5: 장소 이름 직접 검색

**User Story:** 데모 시연자로서, 청중이 "경복궁", "Gyeongbokgung", "근정문" 같이 이름을 정확히 또는 부분적으로 입력하면 곧바로 정보 페이지로 이동하도록 만들고 싶다. 그래야 인식·검색 단계를 거치지 않고도 특정 장소 정보를 빠르게 보여줄 수 있다.

#### Acceptance Criteria

1. WHEN 사용자가 장소 이름 검색창에 1자 이상의 문자열을 입력, THE Demo_App SHALL 입력 문자열을 NFC 정규화 후 lower-case로 변환한다.
2. THE Name_Search_Index SHALL Landmark_Catalog의 13개 landmark 각각에 대해 한국어 이름, 영어 이름, alias 목록을 검색 키로 포함한다.
3. WHEN 정규화된 입력 문자열이 1자 이상이면, THE Demo_App SHALL Name_Search_Index에서 prefix 일치 또는 부분 문자열 일치하는 landmark 후보를 조회하여 최대 10개를 자동완성 후보로 표시한다.
4. THE Name_Search_Index SHALL 검색 시 입력 문자열과 검색 키의 대소문자 차이를 무시한다.
5. WHEN 사용자가 자동완성 후보 중 하나를 선택, THE Demo_App SHALL 해당 landmark_id의 정보 페이지로 이동한다.
6. WHEN 사용자가 검색을 실행, THE Demo_App SHALL 그 시점의 현재 자동완성 후보 목록 상태를 기준으로 후속 동작을 결정한다.
7. IF 사용자가 검색 실행 시점에 자동완성 후보 목록이 비어 있으면, THEN THE Demo_App SHALL "검색 결과 없음" 메시지를 표시하고 정보 페이지로 이동하지 않는다.
8. WHEN 검색 실행 시점에 자동완성 후보가 정확히 1개이면, THE Demo_App SHALL 해당 후보의 정보 페이지로 자동 이동한다.
9. IF 검색 실행 시점에 자동완성 후보가 2개 이상이고 사용자가 후보 중 하나를 선택하지 않은 상태이면, THEN THE Demo_App SHALL "자동완성 목록에서 항목을 선택하세요" 메시지를 표시하고 정보 페이지로 이동하지 않는다.

### Requirement 6: 모델 자산 및 런타임 환경

**User Story:** 모델 팀 담당자로서, 데모 앱이 합의된 자산 경로와 형식으로 ONNX 모델과 JSON 인덱스를 로드하고, 외부 네트워크 의존 없이 로컬에서만 동작하도록 보장하고 싶다. 그래야 시연 환경의 네트워크 상태와 무관하게 결과를 재현할 수 있다.

#### Acceptance Criteria

1. THE Demo_App SHALL 시작 시점에 Asset_Bundle 디렉터리에서 `landmark_encoder.onnx`, `prototype_index.json`, `landmark_text_index.json`, `landmark_info.json` 4개 파일을 로드한다.
2. THE Demo_App SHALL Asset_Bundle 디렉터리 경로를 환경 변수 `LANDMARK_DEMO_ASSETS` 또는 CLI 인자 `--assets`로 받으며, 둘 다 부재하면 기본 경로 `./assets`를 사용한다.
3. IF Asset_Bundle 4개 필수 파일 중 하나라도 부재하거나 파싱에 실패하면, THEN THE Demo_App SHALL UI를 자산 로드 실패 오류 화면으로 시작하여 부재/실패 파일명을 사용자에게 표시하고, 종료 절차에 진입하는 시점에 검색·정보 페이지·이름 검색 기능을 비활성화한다.
4. THE Prototype_Index SHALL 13개 Landmark_Catalog 항목 각각에 대해 landmark_id, prototype embedding(차원 512, L2-normalized 부동소수점 배열) 필드를 포함한다.
5. THE Text_Index SHALL 13개 Landmark_Catalog 항목 각각에 대해 landmark_id, encoder 식별자, text embedding 벡터, 한국어/영어 description 텍스트 필드를 포함한다.
6. THE Landmark_Info_Store SHALL 13개 Landmark_Catalog 항목 각각에 대해 landmark_id, name_ko, name_en, description_ko, latitude, longitude, hero_image_path, aliases(문자열 배열) 필드를 포함한다.
7. THE Demo_App SHALL 이미지 인식, 텍스트 검색, 정보 페이지, 이름 검색 4개 기능을 외부 네트워크 호출 없이 Asset_Bundle만으로 수행한다.
8. WHERE 시연 환경이 노트북 ONNX Runtime이면, THE Demo_App SHALL Python 3.10 이상에서 Streamlit 또는 Gradio 기반 로컬 웹 UI로 구동된다.
9. WHERE 시연 환경이 휴대폰이면, THE Demo_App SHALL ONNX Runtime Mobile을 통해 동일한 Asset_Bundle을 로드한다.
10. WHEN Asset_Bundle 로드가 완료, THE Demo_App SHALL Prototype_Index, Text_Index, Landmark_Info_Store, Name_Search_Index의 landmark_id 집합이 Landmark_Catalog와 정확히 일치하는지 검증한다.
11. IF Asset_Bundle 검증에서 landmark_id 집합 불일치가 발견되면, THEN THE Demo_App SHALL 불일치 항목 목록을 Debug_Log에 기록하고 "Asset 일관성 검증 실패" 오류 메시지를 표준 오류로 출력한 뒤 종료한다.

### Requirement 7: 디버그 로깅 및 시연 보조 기능

**User Story:** 데모 시연자로서, 시연 중 발생하는 입력·결과를 사후에 검토하고 싶고, 청중에게는 깔끔한 화면을 보여주되 필요할 때 raw 점수를 펼쳐 볼 수 있게 하고 싶다. 그래야 시연 후 회고와 모델 디버깅에 사용할 수 있다.

#### Acceptance Criteria

1. WHEN 이미지 검색, 텍스트 검색, 이름 검색 중 어느 한 검색이 실행, THE Demo_App SHALL 입력 종류(`image`, `text`, `name` 중 하나), 입력 식별자(파일명 또는 입력 문자열), Top-3 landmark_id와 fusion_score, 처리 시간을 ISO-8601 타임스탬프와 함께 Debug_Log 파일에 한 줄씩 추가한다.
2. THE Debug_Log 파일 경로는 환경 변수 `LANDMARK_DEMO_LOG_PATH`로 지정 가능하며, 미지정 시 `./logs/demo.jsonl`을 사용한다.
3. WHERE 사용자가 화면의 "개발자 모드" 토글을 활성화하면, THE Demo_App SHALL 결과 화면 하단 또는 별도 패널에 가장 최근 검색 1건의 image_score, text_score, keyword_score, fusion_score 전체 13개 landmark 값을 정렬된 표로 표시한다.
4. WHEN 사용자가 "검색 초기화" 또는 "다른 입력 시도" 버튼을 누르면, THE Demo_App SHALL 입력 폼 텍스트, 결과 카드, 처리 시간 표시, 직전 검색의 이미지 미리보기 4개 요소를 초기 상태로 되돌린다.
5. IF Requirement 7 AC 4의 4개 초기화 대상 중 일부 요소의 초기화에 실패하면, THEN THE Demo_App SHALL 성공한 요소의 초기화는 그대로 유지하고 실패한 요소만 Debug_Log에 기록한다.
6. WHILE Image_Recognizer 또는 Text_Encoder 추론이 진행 중이면, THE Demo_App SHALL 사용자 화면에 진행 표시기(spinner 또는 진행 텍스트)를 표시한다.
7. IF Image_Recognizer 추론이 5초 이내에 완료되지 않으면, THEN THE Demo_App SHALL 사용자에게 "추론이 지연되고 있습니다" 메시지를 표시하고 Debug_Log에 지연 이벤트를 기록한다.
8. THE Debug_Log SHALL out-of-scope 판단(Requirement 3, AC 5)에 해당하는 모든 검색 요청에 대해 화면에 표시되지 않은 Top-3 후보의 raw 점수를 함께 기록한다.
