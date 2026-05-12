# 고객 분석 대시보드

LG HelloVision ML 모델 기반 고객 분석 Streamlit 대시보드.

## 기능

- 📊 **이탈 위험도 예측** (XGBoost, PRC-AUC 0.5214)
- 💰 **VOD 구매 예측** (Random Forest, R² 0.624)
- 📺 **TV/VOD 시청 세그먼트 분석**
- 🎯 **세그먼트 기반 마케팅 전략 추천**

## 실행 방법

```bash
# 1. 필요한 패키지 설치
pip install streamlit plotly pandas numpy

# 2. 대시보드 실행 (프로젝트 루트에서)
streamlit run dashboard/app.py
```

브라우저에서 자동으로 열림 (기본: http://localhost:8501).

## 화면 구성

1. **헤더**: 모델 성능 요약
2. **사이드바**: 고객 ID 조회
3. **메인 메트릭**: 이탈 위험도 / VOD 구매 예측 / 세그먼트
4. **시청 트렌드**: 월별 TV/VOD 시청 시간 차트
5. **Feature Importance**: 이탈 예측 주요 변수
6. **마케팅 전략 추천**: 위험도+구매 경험 기반 4분면 매칭
7. **비즈니스 인사이트**: 핵심 발견 사항

## 데모 데이터

현재는 학습된 모델 파일이 없어 `customer_id` 기반 결정론적 더미 데이터를 사용합니다.
실제 모델 연동 시 `generate_customer_data()` 함수를 다음과 같이 교체:

```python
# 실제 모델 사용 시
import joblib
churn_model = joblib.load("models/churn_xgb_v2.pkl")
vod_model = joblib.load("models/vod_rf.pkl")

def generate_customer_data(customer_id):
    features = load_features(customer_id)  # 실제 데이터 로드
    churn_risk = churn_model.predict_proba(features)[0, 1] * 100
    vod_purchase = vod_model.predict(features)[0]
    ...
```
