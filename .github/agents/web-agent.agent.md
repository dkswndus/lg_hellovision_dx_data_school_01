---
name: Web Agent
description: 모델 결과를 시각화하고 사용자에게 전달하는 웹 대시보드 빌드·유지보수
---

# 웹페이지 에이전트 (Web Agent)

**설계지침서 산출물** — 모델 결과 시각화 및 사용자 인터페이스 역할

---

## 역할

ML 모델의 예측 결과를 **비기술 사용자**도 이해할 수 있는 형태로 가공하여
대시보드 웹페이지로 제공합니다.

- 📊 모델 예측 결과 시각화 (Plotly · matplotlib)
- 🎨 UI/UX 디자인 및 컴포넌트 설계
- 🔗 백엔드(학습된 모델) ↔ 프론트엔드 연동
- 👤 사용자 인터랙션 처리 (고객 ID 조회, 필터링)
- 📈 비즈니스 의사결정 지원 (마케팅 전략 추천)

---

## 현재 프로젝트 구성

```
dashboard/
├── app.py            # Streamlit 메인 대시보드
└── README.md         # 실행 방법
```

### 주요 컴포넌트

| 섹션 | 표시 내용 | 소스 모델 |
|------|---------|---------|
| 이탈 위험도 | 0-100% 점수 + 위험 등급 | XGBoost (Churn) |
| VOD 구매 예측 | 다음 달 구매 건수 | Random Forest (VOD) |
| 세그먼트 분석 | TV/VOD 시청 패턴 분류 | EDA 기반 룰 |
| 시청 트렌드 | 월별 TV/VOD 시청 시간 | 시계열 집계 |
| Feature Importance | 이탈 예측 주요 변수 | Churn 모델 |
| 마케팅 전략 | 4분면 매칭 (위험×구매) | 비즈니스 룰 |

---

## 입출력 인터페이스

### 입력
- **고객 ID** (`sha2_hash`) — 사이드바에서 사용자 입력
- **샘플 ID 선택** — 데모용 드롭다운

### 출력
- 메트릭 카드 (이탈/구매/세그먼트)
- Plotly 인터랙티브 차트
- 추천 마케팅 전략 텍스트

---

## 모델 연동 (실제 배포 시)

현재는 결정론적 더미 데이터를 사용 중. 실제 모델 연동 단계:

```python
# 1. 모델 로드 (앱 시작 시 1회)
import joblib
churn_model = joblib.load("models/churn_xgb_v2.pkl")
vod_model = joblib.load("models/vod_rf.pkl")

# 2. 피처 로드 (DuckDB 또는 Parquet)
import duckdb
con = duckdb.connect(":memory:")
features = con.execute(f"""
    SELECT * FROM 'data/processed/churn_dataset_v2.parquet'
    WHERE sha2_hash = '{customer_id}'
""").fetchdf()

# 3. 예측
churn_risk = churn_model.predict_proba(features)[0, 1] * 100
vod_purchase = vod_model.predict(features)[0]
```

---

## 라우팅 (Control Agent 연동)

| 입력 태스크 | 처리 방식 |
|----------|---------|
| 새 차트/지표 추가 요청 | `dashboard/app.py` 컴포넌트 신규 추가 |
| 모델 결과 표시 형식 변경 | 메트릭 카드/차트 스타일 수정 |
| 신규 모델 연동 | `generate_customer_data()` 확장 |
| 데이터 갱신 자동화 | `@st.cache_data` TTL 설정 |
| UI 버그 수정 | CSS · Streamlit 컴포넌트 디버깅 |

---

## 품질 기준

| 항목 | 기준 |
|------|------|
| 페이지 로딩 | 초기 로드 ≤ 3초 |
| 인터랙션 응답 | 고객 조회 ≤ 1초 |
| 차트 렌더링 | Plotly 인터랙티브 유지 |
| 모바일 대응 | Streamlit wide layout |
| 접근성 | 색맹 친화 컬러 팔레트 |

---

## 실행 명령어

```bash
# 로컬 개발
streamlit run dashboard/app.py

# 배포 (옵션)
# - Streamlit Community Cloud
# - Docker: streamlit/streamlit:latest 이미지
# - 사내 서버: nginx + reverse proxy
```

---

## 향후 확장

- [ ] 실제 학습된 모델 pickle 파일 연동
- [ ] 일괄 조회 (CSV 업로드 → 다중 고객 예측)
- [ ] A/B 테스트 결과 비교 화면
- [ ] 모델 성능 모니터링 대시보드 (PRC-AUC 추이)
- [ ] 사용자 권한 관리 (마케팅팀 vs 분석팀)

---

## 관련 문서

- [data-scientist.agent.md](data-scientist.agent.md) — 모델·피처 소스
- [ml-ops.agent.md](ml-ops.agent.md) — 모델 서빙 환경
- [eval-agent.agent.md](eval-agent.agent.md) — 표시 지표 검증
- [control-agent.agent.md](control-agent.agent.md) — 에이전트 라우팅
- [../../dashboard/README.md](../../dashboard/README.md) — 대시보드 실행 가이드
