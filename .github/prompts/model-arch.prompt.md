# 2단계: 모델 아키텍처 초기화 프롬프트

**설계지침서 산출물** — 초기 모델 설계용 변수 템플릿

---

## 해지 예측 (분류)

### 1차 변수 10개

| 순위 | 변수명 | 설명 |
|------|--------|------|
| 1 | AGMT_END_YMD | 약정 만료일 |
| 2 | TOTAL_USED_DAYS | 총 이용 일수 |
| 3 | AGMT_END_SEG | 약정 만료 구간 |
| 4 | CH_HH_AVG_MONTH1 | 최근 1개월 채널 시청 가구 수 |
| 5 | CH_LAST_DAYS_BF_GRP | 마지막 채널 시청 일수 그룹 |
| 6 | SMS_SEND_CLS_NM | SMS 수신 동의 |
| 7 | VOC_STOP_CANCEL_MONTH1_YN | 최근 1개월 해지/중단 VOC 여부 |
| 8 | vod_view_cnt | VOD 시청 횟수 |
| 9 | vod_use_tms_sum | VOD 시청 시간 합계(초) |
| 10 | vod_asset_cnt | VOD 시청 콘텐츠 종류 수 |

### 2차 추가 변수 7개

- Flag_Kids, Flag_Movie, Recency, Total_Watch_Time, Log_Watch_Time, Is_Netflix, Risk_Segment_Kids_NFX

### 평가 지표

- PRC-AUC, Accuracy, Precision, Recall, F1

---

## VOD 구매 예측 (회귀)

### 핵심 변수

| 변수 | 설명 |
|------|------|
| fod_cnt | 해당 월 FOD(무료) 시청 건수 |
| svod_cnt | 해당 월 SVOD(구독) 시청 건수 |
| lagged_rvod_cnt | 전월 RVOD(유료) 구매 건수 |
| vod_asset_cnt | VOD 시청 콘텐츠 종류 수 |

### 평가 지표

- R2, MAE, RMSE

---

## 모델 후보

- Logistic Regression, Random Forest, Gradient Boosting, LightGBM, XGBoost
- 분류: PRC-AUC 기준, 회귀: R2 기준 선택
