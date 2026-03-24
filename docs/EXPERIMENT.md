# 4단계: 실험 컨텍스트 설계

**설계지침서 산출물** — 실험 패킷, 목표 평가 지표, 기술 제약

---

## 실험 패킷 — 목표 평가 지표

| 모델 | 목표 지표 | 베이스라인 |
|------|-----------|------------|
| 해지 예측 (1차) | PRC-AUC, Accuracy, Precision, Recall, F1 | — |
| 해지 예측 (2차) | PRC-AUC 개선 | 1차 최고 0.478 (LightGBM) |
| VOD 구매 예측 | R2, MAE, RMSE | FOD/RVOD/SVOD 추가 전 R2 0.293 |

---

## 데이터 스펙

| 항목 | 해지 예측 | VOD 구매 예측 |
|------|-----------|---------------|
| 데이터셋 | churn_dataset, churn_dataset_v2 | vod_purchase_dataset |
| 규모 | 2,268,472건 (80:20) | 사용자·월별(sha2_hash, p_mt) |
| 종속변수 | cancel_yn (0/1) | vod_purchase_cnt (RVOD 건수) |
| 특성 | 불균형 분류 | zero-inflated (구매 1건↑ 3.1%) |

---

## 기술 제약 조건

| 항목 | 설정 |
|------|------|
| 과적합 방지 | 해시 기반 샘플링(SEED=42), train/val/test 분리 |
| 클래스 균형 | 유지:해지 80:20 |
| 재현성 | 결정적 샘플링 공식 |

---

## Context Packet (Model Focus)

- **해지**: 약정 만료, 채널 시청, VOD 이용 패턴, VOC 해지 문의
- **VOD 구매**: 전월 RVOD, FOD/SVOD 시청, INHOME_RATE
