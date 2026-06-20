---
name: Data Profiler Agent
description: 데이터 구조·결측·타입·타깃 분포 자동 진단
---
# Data Profiler Agent

---

## 역할

모델 학습 전 데이터의 기본 진단 리포트를 자동 생성한다. 후속 에이전트(leakage-checker, EDA, FE)가 참조할 정량 지표를 표준화된 JSON 으로 출력.

---

## 입력

- `data/interim/*_clean.parquet`
- `data/interim/*_schema.json`, `*_categories.json` (참고)

---

## 산출물

- `output/profile/{table}_profile.json`
  - `row_count`, `col_count`
  - `dtypes_map`: `{col: dtype}`
  - `missing`: `{col: ratio}`
  - `basic_stats`:
    - numeric: `{mean, std, min, max, p25, p50, p75}`
    - categorical: `{top_k: [{value, count}]}`
  - `target_distribution`: `{label: count, ratio}` (churn 이면 `cancel_yn`, VOD 면 `rvod_cnt`)
  - `cardinality`: `{col: n_unique}`
- `output/profile/{table}_profile.md` — 사람 친화 요약

---

## 검사 항목 & 임계값

| 항목 | 임계값 | 조치 |
|------|--------|------|
| 결측률 | > 30% | 경고 |
| 결측률 | > 70% | drop 제안 |
| 카디널리티 | `n_unique == 1` | drop 제안 |
| 타입 미스매치 | `schema.json` 와 dtype 불일치 | 알림 |
| churn 타깃 비율 | 80:20 ±5%p | 경고 (샘플링 재확인) |

---

## 다음

- [[leakage-checker-agent]]

---

## 코드 위치

- `scripts/agents/profiler.py` (신규)
- 기존 `scripts/eda/eda_and_clean.py` 의 분포 분석 로직 추출·재사용
