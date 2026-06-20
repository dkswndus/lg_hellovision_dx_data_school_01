---
name: Explanation Agent
description: SHAP·Feature Importance·고객군 프로파일링 (체크포인트 4)
---
# Explanation Agent

---

## 역할

우승 모델(leaderboard 1위)의 예측 근거를 비즈니스 언어로 변환한다.

---

## 입력

- `output/models/{winner}/model.pkl`
- `output/models/leaderboard.csv`
- `data/processed/{dataset}.parquet`

---

## 산출물

- `output/explain/shap_summary.png` — 전역 중요도 beeswarm
- `output/explain/shap_dependence/{feature}.png` — 상위 5개 변수 dependence plot
- `output/explain/feature_importance.csv` — 전역 중요도 표
- `output/explain/segment_profiles.csv` — 위험점수 분위(decile) 별 평균 피처
- `output/explain/explain_brief.md` — 마케팅 인사이트 3~5개

---

## 분석 절차

1. **Explainer 선택**
   - LGBM·XGB·CatBoost → `TreeExplainer`
   - Logistic Regression → `KernelExplainer` (샘플 1000)
2. **전역 중요도** — `mean(|shap_values|)` 정렬
3. **Dependence plot** — 상위 5개 변수 vs SHAP value
4. **세그먼트 프로파일링** — 예측 위험점수 quantile(0.0~0.1, ..., 0.9~1.0) 별 피처 평균
5. **비즈니스 해석 요약** — 인사이트 → `explain_brief.md`

---

## 출력 예시 (explain_brief.md)

```markdown
1. 키즈 콘텐츠 + 넷플릭스 동시 사용 (`Risk_Segment_Kids_NFX`) 고객은
   상위 10% 위험군의 38%를 차지 → 가족 요금제 타겟 마케팅 권장
2. Recency > 30일 고객의 해지 확률 평균 67% → 휴면 알림 캠페인
3. ...
```

---

## 다음

- (사람 확인 후) [[ml-ops]] · [[web-agent]]

---

## 코드 위치

- `scripts/agents/explanation.py` (신규)
- 의존: `shap>=0.43`, `matplotlib`
