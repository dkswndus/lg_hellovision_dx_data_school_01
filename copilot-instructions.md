# Copilot / AI 코딩 지침

**1단계 산출물** — 실험 로깅 및 공통 규칙

---

## 프로젝트 컨텍스트

- **도메인**: LG헬로비전 VOD·해지·매출 분석
- **데이터**: user_profile, vod_log, vod_content (Parquet)
- **모델**: 해지 예측(분류), VOD 구매 예측(회귀)

---

## 실험 로깅 규칙

1. **재현성**: `SEED=42` 기준 결정적 샘플링
2. **경로**: 데이터는 `data/`, 결과는 `output/` 사용
3. **스키마**: JSON 스키마(`*_schema.json`, `*_categories.json`) 기반 타입 변환

---

## 코드 스타일

- Python: scripts/ 하위 모듈화 (eda, build, analysis, visualization)
- 출력: `output/{태스크명}/` 하위에 CSV, PNG 저장
