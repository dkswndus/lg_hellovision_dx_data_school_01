# 2단계: EDA 리포트 자동화 프롬프트

**설계지침서 산출물** — 탐색적 데이터 분석 리포트용 프롬프트 템플릿

---

## 컬럼 삭제 규칙

- **삭제 조건**: 결측 70% 이상 **또는** 고유값 1개 이하
- **적용 테이블**: vod_log, vod_content, user_profile

| 테이블 | 삭제 전 | 삭제 | 최종 | 행 수 |
|--------|---------|------|------|-------|
| vod_log | 9 | 0 | 9 | 60,679,778 |
| vod_content | 88 | 33 | 55 | 470,095 |
| user_profile | 39 | 0 | 39 | 45,369,463 |

---

## 데이터 타입 변환 규칙 (JSON 스키마 기반)

| 규칙 | 조건 | 변환 결과 |
|------|------|-----------|
| VARCHAR → BOOLEAN | 고유값이 Y/N·YES/NO·T/F·0/1만 | TRUE/FALSE/NULL |
| VARCHAR → BIGINT | 고유값이 모두 정수 형태 | 정수형 |
| VARCHAR → DOUBLE | 고유값이 모두 실수 형태 | 실수형 |
| VARCHAR → CATEGORY | 고유값 500개 이하 문자열 | 정수코드 (매핑 JSON 저장) |
| VARCHAR 유지 | ID, 해시, 고카디널리티 문자열 | VARCHAR |

### 적용 컬럼 예시

- **BOOLEAN**: PROD_OLD_YN, BUNDLE_YN, NFX_USE_YN, is_hot_fl
- **BIGINT**: TOTAL_USED_DAYS, p_mt, broad_ymd, strt_dt
- **DOUBLE**: INHOME_RATE, CH_HH_AVG_MONTH1
- **CATEGORY**: CT_CL, genre_of_ct_cl, SVC_USE_DAYS_GRP

---

## CSV → Parquet 변환 효과

- **용량 절감**: 26,015 MB → 8,716 MB (66.5%)
- **처리 속도**: DuckDB 기준 1,333배
- **메모리**: 컬럼 단위 로드 시 50~90% 절감

---

## 도메인 특화: 키즈 관련 컬럼

- **CT_CL**: "키즈" = 코드 14
- **user_profile**: KIDS_USE_PV_MONTH1
- **vod_content/vod_log**: category, description에 "키즈" 포함
