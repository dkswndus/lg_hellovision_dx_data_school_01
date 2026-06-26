"""포트폴리오용: 변수 상관관계 heatmap + VIF 표 + 파생변수 전/후 성능 비교 차트 생성 (임시 스크립트)"""
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE / "output" / "churn_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(BASE / "data" / "processed" / "churn_dataset_v2.parquet")

cols = [
    "AGMT_END_YMD", "TOTAL_USED_DAYS", "AGMT_END_SEG", "CH_HH_AVG_MONTH1",
    "CH_LAST_DAYS_BF_GRP", "SMS_SEND_CLS_NM", "VOC_STOP_CANCEL_MONTH1_YN",
    "vod_view_cnt", "vod_use_tms_sum", "vod_asset_cnt",
    "Flag_Kids", "Flag_Movie", "Recency", "Total_Watch_Time",
    "Log_Watch_Time", "Is_Netflix", "Risk_Segment_Kids_NFX",
]
sample = df[cols].sample(n=min(50_000, len(df)), random_state=42).copy()
for c in sample.columns:
    if not pd.api.types.is_numeric_dtype(sample[c]):
        sample[c] = sample[c].astype("category").cat.codes
sample = sample.fillna(sample.median(numeric_only=True))

# 1. 상관관계 heatmap
corr = sample.corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(cols, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=7, color="white" if abs(v) > 0.5 else "black")
plt.colorbar(im, ax=ax, label="Pearson 상관계수")
ax.set_title("churn_dataset_v2 — 주요 변수 17개 상관관계 행렬 (n=50,000 샘플)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"저장: {OUTPUT_DIR / 'correlation_heatmap.png'}")

# 2. VIF 계산 (statsmodels 없이 수동: VIF = 1 / (1 - R^2))
vif_rows = []
for target_col in cols:
    X = sample.drop(columns=[target_col]).values
    y = sample[target_col].values
    lr = LinearRegression().fit(X, y)
    r2 = lr.score(X, y)
    vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
    vif_rows.append({"변수": target_col, "VIF": vif})
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif_df.to_csv(OUTPUT_DIR / "vif_table.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
table_data = [["변수", "VIF", "판정"]]
for _, row in vif_df.iterrows():
    judge = "높음(>10)" if row["VIF"] > 10 else ("주의(>5)" if row["VIF"] > 5 else "양호")
    table_data.append([row["변수"], f"{row['VIF']:.2f}", judge])
tbl = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.5, 0.25, 0.25])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.5)
for j in range(3):
    tbl[0, j].set_facecolor("#3498db")
    tbl[0, j].set_text_props(color="white", weight="bold")
ax.set_title("VIF(분산팽창계수) 기반 다중공선성 점검 (VIF>10 위험)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vif_table.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"저장: {OUTPUT_DIR / 'vif_table.png'}")
print(vif_df.to_string(index=False))

# 3. 파생변수 전/후 성능 비교 차트
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
labels = ["1차\n(변수선택)", "2차\n(+파생변수 7개)"]
churn_vals = [0.478, 0.521]
vod_vals = [0.293, 0.624]
for ax, vals, title, ylab in zip(
    axes, [churn_vals, vod_vals],
    ["해지 예측 — PRC-AUC", "VOD 구매 예측 — R²"],
    ["PRC-AUC", "R²"],
):
    bars = ax.bar(labels, vals, color=["#95a5a6", "#3498db"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, max(vals) * 1.25)
plt.suptitle("파생변수 추가 전/후 모델 성능 비교", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "before_after_derived_features.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"저장: {OUTPUT_DIR / 'before_after_derived_features.png'}")
