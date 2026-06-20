---
name: Feature Engineering Agent
description: 가입기간·이용 변화율·VOD 활성도 등 파생변수 생성·관리
---
# Feature Engineering Agent

> formerly **data-scientist** 의 FE 책무를 분리

---

## 역할

- 원시 컬럼에서 의미 있는 파생변수 생성
- `output/eda/eda_summary.md` 의 인사이트를 코드로 반영
- **변수 사전 (feature_spec.json) 작성** — 재현성·문서화

---

## 입력

- `data/interim/*_clean.parquet`
- `output/eda/eda_summary.md` (인사이트 시드)
- `output/leakage/leakage_report.json` (드롭 권장 반영)

---

## 산출물

- `data/processed/{dataset}.parquet` — 학습 입력
- `output/features/feature_spec.json`
  ```json
  {
    "Tenure_Months": {
      "formula": "(BASE_DT - JOIN_DT).days / 30",
      "dtype": "float32",
      "source_cols": ["JOIN_DT", "BASE_DT"],
      "rationale": "신규 고객일수록 해지 위험 높음 (EDA #2)",
      "version": "v3"
    }
  }
  ```
- `output/features/feature_importance_prior.csv` — RF 기반 사전 중요도

---

## 기본 파생변수 세트

| 변수 | 정의 | 출처 |
|------|------|------|
| `Tenure_Months` | 가입일 → 기준일 개월 수 | v3 신규 |
| `Watch_Trend_Slope` | 최근 3개월 시청시간 회귀계수 | v3 신규 |
| `VOD_Activity` | `log(1 + RVOD_CNT)` | v3 신규 |
| `Recency` | 기준일 − 마지막 시청일 | v2 |
| `Total_Watch_Time` | VOD 시청시간 합 | v2 |
| `Log_Watch_Time` | `ln(1 + Total_Watch_Time)` | v2 |
| `Flag_Kids` / `Flag_Movie` | CT_CL=14 / 12 시청 여부 | v2 |
| `Is_Netflix` | `NFX_USE_YN` | v2 |
| `Risk_Segment_Kids_NFX` | `Flag_Kids=1 AND Is_Netflix=1` | v2 |

---

## 규칙 (CLAUDE.md 준수)

- `SEED=42` 고정
- `BASE = Path(__file__).resolve().parent.parent.parent` 패턴
- 새 변수 추가 시 `feature_spec.json` 에 **rationale 필수**
- 버전 증가는 dataset 파일명에 반영 (`churn_dataset_v3.parquet`)

---

## 다음

- [[modeling-agent]]

---

## 코드 위치

- `scripts/agents/feature_engineering.py` (신규)
- 기존 `scripts/build/build_churn_dataset_v2.py` 흐름 통합·확장
