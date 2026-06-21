# EDA Summary (churn)

- 샘플 50,000건 — 유지 40,106 / 해지 9,894
- 검정 변수 44개, 유의수준 α=0.05 (Bonferroni 보정 적용)

## 핵심 인사이트 (효과크기 상위)
1. `CH_LAST_DAYS_BF_GRP` — 해지 평균 3.09 vs 유지 평균 3.47 (Cohen's d=-0.305, Mann-Whitney U p=3.272e-135)
2. `TOTAL_USED_DAYS` — 해지 평균 2.46e+03 vs 유지 평균 2.77e+03 (Cohen's d=-0.255, Mann-Whitney U p=5.074e-106)
3. `CH_HH_AVG_MONTH1` — 해지 평균 3.58 vs 유지 평균 4.55 (Cohen's d=-0.239, Mann-Whitney U p=1.798e-139)
4. `AGMT_KIND_NM` — 해지 평균 2.13 vs 유지 평균 2.62 (Cohen's d=-0.219, Mann-Whitney U p=7.026e-81)
5. `SMS_SEND_CLS_NM` — 해지 평균 2.28 vs 유지 평균 2.18 (Cohen's d=0.184, Mann-Whitney U p=1.238e-75)