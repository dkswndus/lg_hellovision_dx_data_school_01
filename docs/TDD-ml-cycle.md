# 5단계: 실험 자동화 사이클 (Red-Green-Refactor)

**설계지침서 산출물** — TDD-ml-metric 기반 실험 사이클

---

## Red — 목표 미달·실패 케이스

| 구분 | 내용 |
|------|------|
| 해지 1차 | PRC-AUC 0.478로 비즈니스 활용 부족, CT_CL·VOD 세분화 부재 |
| VOD 구매 | FOD/RVOD/SVOD 미분리 시 R2 0.293 |

---

## Green — 목표 달성

| 사이클 | 조치 | 결과 |
|--------|------|------|
| 해지 2차 | 7개 변수 추가 | PRC-AUC **0.521** (XGBoost) |
| VOD 구매 | fod_cnt, svod_cnt, lagged_rvod_cnt 추가 | R2 **0.624** (Random Forest) |

---

## 모델별 성능

### 해지 예측

- 1차 최고: LightGBM PRC-AUC 0.478
- 2차 최고: XGBoost PRC-AUC 0.521

### VOD 구매 예측

- 최고: Random Forest R2 0.624, MAE 0.60, RMSE 7.74

---

## Refactor — 최적화 포인트

- 변수 중요도 기반 피처 축소: Recency, Log_Watch_Time, lagged_rvod_cnt
- 추론 최적화: 입력 스키마 고정

---

## 관련 산출물

- `tests/test_model_performance.py` — 목표 지표 검증 테스트
