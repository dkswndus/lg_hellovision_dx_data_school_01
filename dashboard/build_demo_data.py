"""
배포용 데모 데이터 생성 스크립트
실행: python dashboard/build_demo_data.py
대용량 parquet → 소형 JSON/CSV (dashboard/demo_data/)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "demo_data"
OUT.mkdir(exist_ok=True)

CHURN_COLS = [
    "sha2_hash", "p_mt", "cancel_yn",
    "CH_LAST_DAYS_BF_GRP", "VOC_TOTAL_MONTH1_YN", "VOC_STOP_CANCEL_MONTH1_YN",
    "AGMT_END_SEG", "Is_Netflix", "NFX_USE_YN", "Flag_Kids", "Flag_Movie",
    "Recency", "Total_Watch_Time", "vod_view_cnt", "vod_use_tms_sum",
    "vod_asset_cnt", "SMS_SEND_CLS_NM", "TOTAL_INTERNET_SCRB",
    "TOTAL_USED_DAYS", "CH_HH_AVG_MONTH1", "AGMT_END_YMD",
    "SVC_USE_DAYS_GRP", "AGE_GRP10", "BUNDLE_YN",
]

print("1. churn_dataset_v2 로딩...")
df = pd.read_parquet(BASE / "data/processed/churn_dataset_v2.parquet", columns=CHURN_COLS)
df202312 = df[df["p_mt"] == 202312].reset_index(drop=True)
print(f"   {len(df202312):,}명")

# ── KPI 집계 ──────────────────────────────────
print("2. KPI 집계...")
kpi = {
    "total_customers": int(len(df202312)),
    "churn_rate": round(float(df202312["cancel_yn"].mean() * 100), 2),
    "vod_active_rate": round(float((df202312["vod_view_cnt"] > 0).mean() * 100), 2),
    "netflix_rate": round(float(df202312["Is_Netflix"].mean() * 100), 2),
}
(OUT / "kpi.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=2))
print(f"   저장: {OUT / 'kpi.json'}")

# ── 샘플 고객 5명 ──────────────────────────────
print("3. 샘플 고객 선정...")
high = df202312[
    (df202312["CH_LAST_DAYS_BF_GRP"] >= 4) &
    df202312["VOC_STOP_CANCEL_MONTH1_YN"].astype(bool)
].sample(2, random_state=42)

mid = df202312[
    (df202312["CH_LAST_DAYS_BF_GRP"].between(2, 3)) &
    (df202312["Is_Netflix"] == 1)
].sample(1, random_state=1)

low = df202312[
    (df202312["CH_LAST_DAYS_BF_GRP"] <= 1) &
    (~df202312["VOC_TOTAL_MONTH1_YN"].astype(bool)) &
    (df202312["vod_view_cnt"] > 0)
].sample(2, random_state=7)

samples = pd.concat([high, mid, low]).reset_index(drop=True)

# lagged_rvod_cnt 조인
print("   vod_purchase_dataset 로딩...")
vod_df = pd.read_parquet(
    BASE / "data/processed/vod_purchase_dataset.parquet",
    columns=["sha2_hash", "p_mt", "lagged_rvod_cnt", "vod_purchase_cnt"],
)
vod202312 = vod_df[vod_df["p_mt"] == 202312]
samples = samples.merge(
    vod202312[["sha2_hash", "lagged_rvod_cnt", "vod_purchase_cnt"]],
    on="sha2_hash", how="left"
)
samples["lagged_rvod_cnt"] = samples["lagged_rvod_cnt"].fillna(0)
samples["vod_purchase_cnt"] = samples["vod_purchase_cnt"].fillna(0)

# bool 컬럼 → int (JSON 직렬화 호환)
for col in samples.select_dtypes(include="bool").columns:
    samples[col] = samples[col].astype(int)

samples.to_csv(OUT / "sample_customers.csv", index=False, encoding="utf-8-sig")
print(f"   저장: {OUT / 'sample_customers.csv'}  ({len(samples)}명)")

# ── 월별 트렌드 복사 ────────────────────────────
print("4. 월별 트렌드 복사...")
trend = pd.read_csv(BASE / "output/tv_vod_shift/monthly_tv_vod_trend.csv")
trend = trend[trend["VOD_avg"] > 0].reset_index(drop=True)
trend.to_csv(OUT / "monthly_trend.csv", index=False, encoding="utf-8-sig")
print(f"   저장: {OUT / 'monthly_trend.csv'}  ({len(trend)}개월)")

# ── 4분면 고객 데이터 (300명) ──────────────────
print("5. 4분면 고객 데이터 생성...")
quad_raw = df202312.sample(n=min(300, len(df202312)), random_state=99).copy()
quad_raw = quad_raw.merge(
    vod202312[["sha2_hash", "lagged_rvod_cnt"]],
    on="sha2_hash", how="left"
)
quad_raw["lagged_rvod_cnt"] = quad_raw["lagged_rvod_cnt"].fillna(0)
for col in quad_raw.select_dtypes(include="bool").columns:
    quad_raw[col] = quad_raw[col].astype(int)

# 해지 위험도 산출 (피처 기반 공식)
score = (quad_raw["CH_LAST_DAYS_BF_GRP"].fillna(0).clip(0, 5) / 5) * 40
score += quad_raw["VOC_STOP_CANCEL_MONTH1_YN"].fillna(0).astype(int) * 25
voc_total = quad_raw["VOC_TOTAL_MONTH1_YN"].fillna(0).astype(int)
voc_cancel = quad_raw["VOC_STOP_CANCEL_MONTH1_YN"].fillna(0).astype(int)
score += voc_total.where(voc_cancel == 0, 0) * 12
score += ((7 - quad_raw["AGMT_END_SEG"].fillna(7)).clip(0) / 6) * 15
score += quad_raw["Is_Netflix"].fillna(0).astype(int) * 8
recency = quad_raw["Recency"].fillna(0)
score += np.where(recency >= 999, 5, np.where(recency > 180, 12, np.where(recency > 90, 6, 0)))
quad_raw["churn_risk"] = score.clip(5, 95).round(1)

# VOD 구매 예측 산출
lagged = quad_raw["lagged_rvod_cnt"].fillna(0)
vod_view = quad_raw["vod_view_cnt"].fillna(0)
vod_asset = quad_raw["vod_asset_cnt"].fillna(0)
quad_raw["vod_purchase"] = (lagged * 0.75 + (vod_view / 500) * 0.15 + (vod_asset / 50) * 0.10).clip(0, 30).round(2)

# 4분면 레이블
def _quadrant(r):
    hi = r["churn_risk"] >= 50
    buy = r["vod_purchase"] >= 1.0
    if hi and not buy:  return "Q1_고위험_저구매"
    if hi and buy:      return "Q2_고위험_고구매"
    if not hi and not buy: return "Q3_저위험_저구매"
    return "Q4_저위험_고구매"

quad_raw["quadrant"] = quad_raw.apply(_quadrant, axis=1)
quad_out = quad_raw[["sha2_hash", "churn_risk", "vod_purchase", "quadrant", "cancel_yn"]]
quad_out.to_csv(OUT / "quadrant_customers.csv", index=False, encoding="utf-8-sig")
print(f"   저장: {OUT / 'quadrant_customers.csv'}  ({len(quad_out)}명)")
print("   4분면 분포:", quad_raw["quadrant"].value_counts().to_dict())

# ── 결과 확인 ──────────────────────────────────
print("\n=== 생성 완료 ===")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:30s}  {f.stat().st_size / 1024:.1f} KB")

print("\nKPI:", kpi)
print("\n샘플 고객:")
for _, r in samples.iterrows():
    print(f"  {r['sha2_hash'][:16]}...  CH_LAST={r['CH_LAST_DAYS_BF_GRP']}  "
          f"cancel={r['cancel_yn']}  Netflix={r['Is_Netflix']}  lagged={r['lagged_rvod_cnt']}")
