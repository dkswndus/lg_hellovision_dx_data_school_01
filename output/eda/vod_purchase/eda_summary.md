# EDA Summary (vod_purchase)

- 샘플 50,000건 — 비구매 48,442 / 구매 1,558
- 검정 변수 42개, 유의수준 α=0.05 (Bonferroni 보정 적용)

## 핵심 인사이트 (효과크기 상위)
1. `SVOD_SCRB_CNT_GRP` — 구매 평균 0.446 vs 비구매 평균 0.0183 (Cohen's d=2.032, Mann-Whitney U p=0)
2. `vod_asset_cnt` — 구매 평균 147 vs 비구매 평균 6.49 (Cohen's d=1.874, Mann-Whitney U p=0)
3. `vod_use_tms_sum` — 구매 평균 1.04e+06 vs 비구매 평균 4.19e+04 (Cohen's d=1.689, Mann-Whitney U p=0)
4. `lagged_rvod_cnt` — 구매 평균 16.4 vs 비구매 평균 0.197 (Cohen's d=1.576, Mann-Whitney U p=0)
5. `vod_view_cnt` — 구매 평균 679 vs 비구매 평균 30 (Cohen's d=1.454, Mann-Whitney U p=0)