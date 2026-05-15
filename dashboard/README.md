# 고객 분석 대시보드

LG HelloVision ML 모델 기반 고객 분석 Streamlit 대시보드 — **실제 데이터 연동 버전**.

## 기능

- 📊 **이탈 위험도 예측** (XGBoost v2, PRC-AUC 0.5214) — 실측 피처 기반 점수
- 💰 **VOD 구매 예측** (Random Forest, R² 0.624) — lagged_rvod_cnt 기반
- 📺 **TV·VOD 월별 트렌드** — 2023년 실측 집계 데이터
- 📊 **세그먼트별 해지율** — 저TV·고VOD 15.93% vs 전체 8.79%
- 🎯 **세그먼트 기반 마케팅 전략 추천**

## 실행 방법

```bash
# 1. 필요한 패키지 설치
pip install streamlit plotly pandas numpy

# 2. 대시보드 실행 (프로젝트 루트에서)
streamlit run dashboard/app.py
```

브라우저에서 자동으로 열림 (기본: http://localhost:8501).

## 데이터 소스

| 소스 | 설명 |
|------|------|
| `data/processed/churn_dataset_v2.parquet` | 고객 피처 (226만 행, 49 컬럼) |
| `data/processed/vod_purchase_dataset.parquet` | lagged_rvod_cnt 포함 VOD 구매 데이터 |
| `output/tv_vod_shift/monthly_tv_vod_trend.csv` | 2023년 월별 TV·VOD 트렌드 |
| `output/tv_vod_shift/segment_churn_ratio.csv` | 세그먼트별 실측 해지율 |

## 화면 구성

1. **전체 KPI**: 2023-12 기준 고객 수 · 해지율 · VOD 활성 비율 · Netflix 중복율
2. **월별 트렌드**: TV(시간/일) vs VOD(시간/월) 듀얼축 차트 (실측)
3. **고객 조회**: sha2_hash 입력 또는 위험도별 샘플 5종 선택
4. **핵심 지표 카드**: 이탈 위험도 / VOD 구매 예측 / 세그먼트
5. **실측 피처 상세**: 시청 행동 + 계약·서비스 정보 테이블
6. **Feature Importance**: XGBoost(해지) / RF(VOD) 탭 분리
7. **마케팅 전략 추천**: 위험도 × 구매 경험 4분면 매칭
8. **세그먼트 해지율**: 실측 바차트
9. **모델 성능 비교**: 전체 5종 실험 결과 (접기/펼치기)

## Feature Importance (실제 모델 훈련 결과)

**해지 예측 (XGBoost v2)**
| 변수 | 중요도 |
|------|--------|
| CH_LAST_DAYS_BF_GRP (미시청 기간) | 21.07% |
| SMS_SEND_CLS_NM (SMS 수신 설정) | 7.47% |
| TOTAL_INTERNET_SCRB (인터넷 가입 수) | 6.27% |
| VOC_TOTAL_MONTH1_YN (VOC 접수 여부) | 5.45% |
| AGMT_END_SEG (계약 만료 시점) | 4.30% |

**VOD 구매 예측 (Random Forest)**
| 변수 | 중요도 |
|------|--------|
| lagged_rvod_cnt (전월 구매 건수) | 45.69% |
| vod_asset_cnt (보유 VOD 자산 수) | 11.86% |
| vod_view_cnt (VOD 시청 횟수) | 10.78% |
