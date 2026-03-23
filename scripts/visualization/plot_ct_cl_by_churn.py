"""
해지/유지별 CT_CL(콘텐츠 클래스) 분포 상위 10개 막대그래프 시각화
- 비율(%) 기준으로 그룹별 차이 비교
"""
import json
import os
from pathlib import Path

import duckdb
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
CHURN_DATASET = BASE / "data" / "processed" / "churn_dataset.parquet"
VOD_LOG = BASE / "data" / "interim" / "vod_log_clean.parquet"
CATEGORIES_PATH = BASE / "data" / "interim" / "vod_log_categories.json"
OUTPUT_PATH = BASE / "output" / "churn_analysis" / "ct_cl_by_churn.png"
OUTPUT_CSV = BASE / "output" / "churn_analysis" / "ct_cl_diff.csv"

# 한글 폰트
if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_ct_cl_reverse_map() -> dict:
    """CT_CL 코드 → 라벨 역매핑"""
    if not CATEGORIES_PATH.exists():
        return {i: str(i) for i in range(20)}
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        cat = json.load(f)
    ct_cl = cat.get("CT_CL", {})
    return {int(v): k for k, v in ct_cl.items()}


def main():
    rev_map = load_ct_cl_reverse_map()

    con = duckdb.connect(":memory:")

    # churn_dataset의 sha2_hash, cancel_yn + vod_log의 CT_CL 조인
    # cancel_yn: 0=유지, 1=해지
    df = con.execute(
        """
        SELECT c.cancel_yn, vl.CT_CL, COUNT(*) AS cnt
        FROM read_parquet(?) c
        INNER JOIN read_parquet(?) vl ON c.sha2_hash = vl.sha2_hash
        WHERE vl.CT_CL IS NOT NULL
        GROUP BY c.cancel_yn, vl.CT_CL
        ORDER BY c.cancel_yn, cnt DESC
        """,
        [str(CHURN_DATASET), str(VOD_LOG)],
    ).fetchdf()

    con.close()

    # 그룹별 총 시청 횟수로 비율 계산
    for cy in [0, 1]:
        total = df[df["cancel_yn"] == cy]["cnt"].sum()
        df.loc[df["cancel_yn"] == cy, "pct"] = df.loc[df["cancel_yn"] == cy, "cnt"] / total * 100

    # 전체 상위 10개 CT_CL (유지+해지 합산 기준) → 이 카테고리로 유지/해지 비율 비교
    all_ct_cl = df.groupby("CT_CL")["cnt"].sum().sort_values(ascending=False).head(10)
    top10_codes = all_ct_cl.index.tolist()

    # pivot: CT_CL x cancel_yn → pct
    pivot = df[df["CT_CL"].isin(top10_codes)].pivot(
        index="CT_CL", columns="cancel_yn", values="pct"
    ).reindex(top10_codes).fillna(0)

    pct_maintain = pivot[0].values
    pct_churn = pivot[1].values
    diff = pct_churn - pct_maintain  # 해지 - 유지 (양수: 해지가 더 높음)

    labels = [rev_map.get(c, str(c)) for c in top10_codes]
    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.2, 1])

    # 1) 비율 비교
    ax1 = axes[0]
    ax1.bar(x - w / 2, pct_maintain, w, label="유지", color="#3498db", alpha=0.8)
    ax1.bar(x + w / 2, pct_churn, w, label="해지", color="#e74c3c", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.set_ylabel("비율 (%)")
    ax1.set_title("해지/유지별 CT_CL 분포 (그룹 내 비율)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2) 차이 (해지 - 유지)
    ax2 = axes[1]
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in diff]
    ax2.barh(labels, diff, color=colors, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("차이 (해지 % - 유지 %p)")
    ax2.set_title("해지 vs 유지 비율 차이 (양수: 해지가 더 많이 시청)")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close()

    # CSV 저장
    diff_df = pivot.copy()
    diff_df.columns = ["유지(%)", "해지(%)"]
    diff_df.index = labels
    diff_df["차이(해지-유지 %p)"] = diff
    diff_df = diff_df.round(2)
    diff_df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"저장: {OUTPUT_PATH}")
    print(f"저장: {OUTPUT_CSV}")
    print("\n[CT_CL 차이 요약]")
    print(diff_df.to_string())


if __name__ == "__main__":
    main()
