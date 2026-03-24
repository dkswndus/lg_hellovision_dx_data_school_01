# 3단계: Data Scientist 에이전트

**설계지침서 산출물** — 데이터 분포 분석·피처 엔지니어링 역할

---

## 역할

- 데이터 분포 분석
- 피처 엔지니어링
- 변수 선정 (PRC-AUC, Feature Importance)
- 가설검정 (H0/H1, p-value, 효과 크기)

---

## 데이터셋 구성 (churn_dataset)

| 항목 | 내용 |
|------|------|
| 기준 테이블 | user_profile |
| 집계 테이블 | vod_log |
| JOIN 키 | sha2_hash |
| 샘플 비율 | 5% (유지:해지 80:20) |
| 시드 | SEED=42 |

### vod_log 집계

- vod_view_cnt, vod_use_tms_sum, vod_asset_cnt

---

## 변수 선정 방식

- **기준**: PRC-AUC (불균형 데이터)
- **방법**: RF feature importance → top-k별 Logistic Regression → PRC-AUC 비교
- **결과**: 상위 10개 변수 사용

---

## 2차 변수 추가 (churn_dataset_v2)

| 변수 | 정의 |
|------|------|
| Flag_Kids | CT_CL=14(키즈) 시청 여부 |
| Flag_Movie | CT_CL=12(영화) 시청 여부 |
| Recency | 기준일 - 마지막 시청일 |
| Total_Watch_Time | VOD 시청 시간 합계 |
| Log_Watch_Time | ln(1 + Total_Watch_Time) |
| Is_Netflix | NFX_USE_YN |
| Risk_Segment_Kids_NFX | Flag_Kids=1 AND Is_Netflix=1 |

---

## CT_CL 비율 차이 (해지 - 유지)

- **해지 그룹 ↑**: TV드라마(+2.27%p), 키즈(+0.25%p)
- **유지 그룹 ↑**: 기타(-1.17%p), 영화(-0.82%p)

---

## 가설검정

- 상위 5개 변수: t-test(연속형), chi2(범주형)
- p-value 모두 유의 (α=0.05)
- 효과 크기: Cohen's d, Cramer's V 병행
