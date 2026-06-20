---
name: EDA Agent
description: 유지/해지 고객 특성 비교·시각화·가설검정 (체크포인트 2)
---
# EDA Agent

> formerly **data-scientist** 의 분포 분석 책무를 분리

---

## 역할

- 타깃 그룹별(유지·해지) 분포 차이 시각화
- CT_CL·시청시간·VOD 활성도 등 핵심 변수 비교
- 가설검정 (t-test · χ² · Cohen's d · Cramér's V)

---

## 입력

- `data/processed/{dataset}.parquet`
- `output/profile/{table}_profile.json`
- `output/leakage/leakage_report.json` (drop 권장 컬럼 반영)

---

## 산출물

- `output/eda/dist_{col}.png` — 그룹별 분포 비교
- `output/eda/ct_cl_share_diff.png` — CT_CL 비율 차 (해지 − 유지)
- `output/eda/hypothesis_tests.csv` — 변수·검정·통계량·p-value·효과크기
- `output/eda/eda_summary.md` — 3~5개 핵심 인사이트 (FE 시드)

---

## 표준 차트

| # | 차트 | 목적 |
|---|------|------|
| 1 | numeric mean±std 비교 (churn vs retained) | 평균 차이 직관 |
| 2 | CT_CL share diff (상위 5 카테고리) | 콘텐츠 선호 차이 |
| 3 | Total_Watch_Time 분포 (log scale) | 시청량 분포 |
| 4 | Recency 히스토그램 by group | 최근성 효과 |
| 5 | Risk_Segment 별 churn rate | 세그먼트 효과 검증 |

---

## 검정 방법

- **연속형**: t-test (정규성 위반 시 Mann-Whitney U), 효과크기 Cohen's d
- **범주형**: χ² test, 효과크기 Cramér's V
- **유의수준**: α = 0.05, Bonferroni 보정 (변수 ≥ 10 일 때)

---

## 다음

- (사람 확인 후) [[feature-engineering-agent]]

---

## 코드 위치

- `scripts/agents/eda.py` (신규)
- 기존 `scripts/visualization/plot_ct_cl_by_churn.py`, `plot_tv_vod_expansion.py` 재사용
