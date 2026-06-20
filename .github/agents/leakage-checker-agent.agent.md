---
name: Leakage Checker Agent
description: 타깃 누수 의심 컬럼·시점 누수·중복 키 탐지 (체크포인트 1)
---
# Leakage Checker Agent

> ⚠️ **누락 시 가장 큰 비용** — 모든 모델 성능 수치의 신뢰도가 여기서 결정됨

---

## 역할

학습 데이터에서 다음 3가지 누수 위험을 자동 점검:

1. **타깃 누수** — 피처가 타깃을 거의 그대로 담고 있는 경우
2. **시점 누수** — 예측 기준 시점 *이후* 정보가 피처에 섞인 경우 (특히 VOD 월별 데이터)
3. **중복 키** — 동일 `sha2_hash` 가 여러 행으로 학습·검증에 양다리

---

## 입력

- `data/processed/{dataset}.parquet`
- `output/profile/{table}_profile.json`

---

## 산출물

- `output/leakage/leakage_report.json`
  - `high_corr_features`: `[{col, corr_with_target, severity}]`
  - `temporal_leakage`: `[{col, p_mt_basis, evidence}]`
  - `duplicate_keys`: `{sha2_hash: count}` (count > 1 만)
  - `feature_dominance`: `[{col, importance, single_top}]`
  - `drop_recommendations`: `[col, ...]`
- `output/leakage/leakage_report.md`

---

## 검사 로직

| 검사 | 방법 | 트리거 |
|------|------|--------|
| 타깃 상관 누수 | numeric: Pearson \| categorical: Cramér's V | `|corr| ≥ 0.95` → **CRITICAL** |
| 부분 누수 | RF top-1 importance 단독 | `> 0.5` → WARN |
| 시점 누수 (VOD) | feature 의 `p_mt` > 예측 기준 `p_mt` | 발견 시 **CRITICAL** |
| 키 중복 | `groupby(sha2_hash).count() > 1` | 발견 시 **CRITICAL** |
| 식별자 혼입 | `sha2_hash`·`CT_CL` 등 비-피처가 학습 X 에 포함 | → WARN |

---

## HITL 트리거

- **CRITICAL 1건 이상** → supervisor 가 즉시 HITL 호출 (자동 진행 차단)
- **WARN 3건 이상** → 체크포인트 1 에서 사람 확인 요청

---

## 다음

- (사람 확인 후) [[eda-agent]]
- CRITICAL 시 → human-in-the-loop

---

## 코드 위치

- `scripts/agents/leakage_checker.py` (신규)
