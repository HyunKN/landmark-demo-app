# 자연어 검색 알려진 문제 및 분석

작성일: 2026-05-18
대상: Landmark Assistant 데모 앱 자연어 검색 탭 (PyTorch / ONNX / INT8 모드 공통)

---

## 1. "경복궁" 검색 → 국립현대미술관 1위

### 증상

| 입력 | 기대 결과 | 실제 결과 |
|---|---|---|
| `경복궁` | gyeongbokgung_geunjeongmun (경복궁 근정문) | **mmca_seoul (국립현대미술관) 93%** |

Top-3: mmca_seoul 93% → gwanghwamun 91% → gyeongbokgung_geunjeongmun 89%

### 원인 분석

**원인 1 — 클래스 ID 불일치**: 우리 13개 클래스에 "경복궁 전체"가 없다. `gwanghwamun`(광화문)과 `gyeongbokgung_geunjeongmun`(근정문)만 있어서 "경복궁"이라는 query에 대한 정답 클래스가 명확하지 않다.

**원인 2 — mmca catalog에 "경복궁" 위치 설명 포함**: 수정 전 `landmark_text_catalog_v2.json`의 mmca 항목에 다음이 포함돼 있었다:
- `user_queries_ko`: `"경복궁 옆 미술관"`
- `user_queries_en`: `"museum near Gyeongbokgung"`
- `visual_features_ko`: `"경복궁 인근의 현대미술관으로..."`
- `visual_features_en`: `"A modern museum complex near Gyeongbokgung..."`
- `landmark_info.json` description_ko: `"...2013년 경복궁 옆 옛 기무사 부지에 개관했습니다..."`

"경복궁"이라는 query가 들어오면 keyword 매칭과 text embedding 모두에서 mmca가 강하게 당겨졌다.

**원인 3 — 점수 분포 평탄화**: 세 후보가 93/91/89%로 거의 평평하다. "경복궁"이라는 단어가 mmca/gwanghwamun/geunjeongmun 세 클래스 모두의 catalog에 등장해서 keyword score가 셋 다 높게 나왔다.

### 수정 내용 (2026-05-18)

`assets/landmark_text_catalog_v2.json` mmca_seoul 항목에서 경복궁 언급 전부 제거:
- `user_queries_ko`: `"경복궁 옆 미술관"` 삭제
- `user_queries_en`: `"museum near Gyeongbokgung"` 삭제
- `visual_features_ko`: `"경복궁 인근의"` → 제거
- `visual_features_en`: `"near Gyeongbokgung"` → 제거

`assets/landmark_info.json` mmca_seoul description_ko:
- `"2013년 경복궁 옆 옛 기무사 부지에 개관했습니다"` → `"2013년 개관한 현대미술 전문 미술관입니다"`

`assets/landmark_text_index.json` 재생성 완료.

### 남은 한계

위치 설명 제거 후에도 "경복궁"이라는 query에 대한 정답 클래스가 없다는 근본 문제는 남아 있다. 사용자가 "경복궁"을 입력하면 gwanghwamun 또는 geunjeongmun이 1위로 올라오는 게 기대이지만, 두 클래스 모두 "경복궁의 일부"라는 점에서 ambiguous 처리가 자연스럽다.

**Sprint 2 해결 방향**: parent landmark `gyeongbokgung_complex` 계층 구조 도입. 사용자가 "경복궁"으로 검색하면 parent로 매칭하고, 상세 보기에서 광화문/근정문/근정전 sub-landmark를 선택하게 한다.

---

## 2. "한복 입고 사진 찍는 곳" → 국립현대미술관 1위

### 증상

| 입력 | 기대 결과 | 실제 결과 |
|---|---|---|
| `한복 입고 사진 찍는 곳` | 경복궁/덕수궁/창경궁 계열 | **mmca_seoul (국립현대미술관)** |

### 원인 분석

**MobileCLIP2-S4의 한국어 이해 한계 + 의미 연결 오류**:

MobileCLIP2-S4는 영어 중심 CLIP 계열 모델이다. 한국어 "사진"을 처리할 때:
- 영어 "photo"(촬영)로 연결되면 → 카메라, 관광지, 야외 촬영
- 영어 "picture/painting"(그림/작품)으로 연결되면 → 미술관, 갤러리, 전시

모델이 "사진"을 후자 방향으로 해석해 "한복 입고 그림/작품 보는 곳 = 미술관"으로 연결했다. mmca의 catalog에 "한복" 관련 내용은 전혀 없음에도 불구하고 임베딩 공간에서 가장 가까운 위치에 있었다.

이는 **모델 버그가 아니라 영어 중심 CLIP의 한국어 의미 연결 한계**다.

### 수정 내용

이 케이스는 catalog 수정으로 완전히 해결되지 않는다. 근본 원인이 모델의 한국어 이해 한계이기 때문이다.

**단기 완화**: 경복궁/덕수궁/창경궁의 catalog에 "한복 체험", "한복 대여", "한복 사진", "hanbok photo" 키워드 추가 → keyword score가 궁궐 쪽으로 당겨짐.

**Sprint 2 해결 방향**: 데이터셋 v2 한국어 caption 다양화. 각 클래스에 한국어 user_query를 풍부하게 넣으면 text_index 임베딩 자체가 한국어 query에 더 잘 반응하게 된다.

---

## 3. 공통 패턴 — 위치 설명이 검색 오염을 일으킨다

두 케이스 모두 **"A 근처의 B"라는 위치 설명이 A를 검색했을 때 B가 올라오는** 패턴이다.

| 위치 설명 | 검색 오염 |
|---|---|
| "경복궁 옆 미술관" (mmca) | "경복궁" 검색 → mmca 1위 |
| "경복궁 인근의 현대미술관" (mmca) | "경복궁" 검색 → mmca 1위 |

**원칙**: catalog의 위치 설명에서 다른 랜드마크 이름을 직접 언급하지 않는다. 위치는 행정구역(종로구, 중구 등) 또는 방향(도심, 북악산 아래 등)으로만 표현한다.

---

## 4. 검증 방법

수정 후 다음 query로 회귀 확인:

```
경복궁          → gwanghwamun 또는 gyeongbokgung_geunjeongmun 1위 (mmca 아님)
경복궁 옆 미술관 → mmca_seoul 1위 (이건 여전히 맞아야 함)
미술관          → mmca_seoul 1위
art gallery     → mmca_seoul 1위
한복 입고 사진 찍는 곳 → 궁궐 계열 (개선 목표, 완전 해결은 Sprint 2)
```

---

## 5. 관련 파일

- `assets/landmark_text_catalog_v2.json` — 수정됨 (2026-05-18)
- `assets/landmark_info.json` — 수정됨 (2026-05-18)
- `assets/landmark_text_index.json` — 재생성됨 (2026-05-18)
- `scripts/build_assets.py` — text_index 재생성 명령: `python scripts/build_assets.py --skip-prototypes --skip-hero-images`
