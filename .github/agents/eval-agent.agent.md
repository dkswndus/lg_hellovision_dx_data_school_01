---
name: Eval Agent
description: 모델 출력 품질 자동 검증
---
# 에이전트 아키텍처: 평가 에이전트 (Eval Agent)

**설계지침서 산출물** — 출력 품질 자동 검증 역할

---

## 역할

모델 출력이 사전 정의된 기준(목표 지표)을 충족하는지 자동으로 검증합니다.

---

## 검증 기준

| 모델 | 지표 | 통과 기준 |
|------|------|-----------|
| 해지 예측 | PRC-AUC | ≥ 0.521 (2차 베이스라인) |
| VOD 구매 예측 | R2 | ≥ 0.624 (현재 최고) |
| VOD 구매 예측 | MAE | ≤ 0.60 |

---

## 검증 트리거

- 모델 학습 완료 후 자동 실행
- `tests/test_model_performance.py` 연동

---

## 결과 처리

| 결과 | 조치 |
|------|------|
| 통과 | `output/` 에 결과 저장, 다음 단계 진행 |
| 실패 (Red) | TDD-ml-cycle Red 단계 진입, 피처·하이퍼파라미터 재조정 |
| 미결정 | 휴먼인더루프 트리거 |

---

## 관련 문서

- [TDD-ml-cycle.md](../../docs/TDD-ml-cycle.md) — Red-Green-Refactor 사이클
- [EXPERIMENT.md](../../docs/EXPERIMENT.md) — 목표 평가 지표
