---
name: Modeling Agent
description: Baseline + RandomForest·LightGBM·XGBoost·CatBoost 병렬 학습·튜닝
---
# Modeling Agent

---

## 역할

5개 모델을 **병렬 학습**하고 leaderboard 를 [[eval-agent]] 에 전달한다.

---

## 입력

- `data/processed/{dataset}.parquet`
- `output/features/feature_spec.json`

---

## 모델 라인업

| 모델 | 역할 | 비고 |
|------|------|------|
| Baseline | LogisticRegression (churn) · LinearRegression (VOD) | 하한 기준 |
| Random Forest | 기존 analysis 스크립트 챔피언 (VOD R² 0.624) | 과적합 내구성 |
| LightGBM | 빠른 트리 부스팅 | 카테고리 native |
| XGBoost | 현재 챔피언 (v2 PRC-AUC 0.5214) | 안정성 |
| CatBoost | 범주형 자동 처리 | 적은 튜닝 비용 |

---

## 학습 설정

| 항목 | 값 |
|------|---|
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| 튜닝 | Optuna `n_trials=20`, `timeout=15min/model` |
| Early stopping | LGBM·XGB·CatBoost `rounds=50` |
| 병렬 | 모델별 별도 프로세스 (joblib n_jobs=5) |

---

## 산출물

- `output/models/{model}_{run_id}/model.pkl`
- `output/models/{model}_{run_id}/metrics.json`
  - PRC-AUC · ROC-AUC · F1 · Precision@K (K=10%, 20%) · Lift
- `output/models/{model}_{run_id}/cv_folds.csv` — fold 별 점수
- `output/models/{model}_{run_id}/hyperparams.json` — Optuna best trial
- `output/models/leaderboard_{task}.csv` — 태스크별 전체 모델 비교 (churn/vod_purchase 분리)

---

## 게이트 (eval-agent 가 검증)

- Churn: PRC-AUC ≥ 0.5
- VOD: R² ≥ 0.6, MAE ≤ 0.6

게이트 미통과 시 supervisor 가 HITL 트리거 → 피처·하이퍼파라미터 재조정.

---

## 다음

- [[eval-agent]]

---

## 코드 위치

- `scripts/agents/modeling.py` (신규)
- 기존 `scripts/analysis/run_churn_v2.py`, `vod_purchase_model.py` 통합
