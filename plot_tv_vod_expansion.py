"""
TV vs FOD/RVOD/SVOD 월별 트렌드 - "소비 방식 TV→VOD 확장" 시각화
"""
import os
from pathlib import Path

import duckdb
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent
CHURN_PATH = BASE / "data" / "processed" / "churn_dataset.parquet"
VOD_BY_TYPE_PATH = BASE / "data" / "processed" / "monthly_vod_by_type.parquet"
OUTPUT_DIR = BASE / "output" / "tv_vod_shift"

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")

    # 월별 TV 평균 (churn_dataset)
    tv_monthly = con.execute(
        """
        SELECT p_mt, AVG(CH_HH_AVG_MONTH1) AS TV_avg
        FROM read_parquet(?)
        WHERE p_mt BETWEEN 202301 AND 202312 AND CH_HH_AVG_MONTH1 IS NOT NULL
        GROUP BY p_mt ORDER BY p_mt
        """,
        [str(CHURN_PATH)],
    ).fetchdf()

    # 월별 FOD/RVOD/SVOD 평균 (monthly_vod_by_type)
    vod_monthly = con.execute(
        """
        SELECT p_mt,
               AVG(fod_tms) AS FOD_avg,
               AVG(rvod_tms) AS RVOD_avg,
               AVG(svod_tms) AS SVOD_avg,
               AVG(total_tms) AS VOD_total_avg
        FROM read_parquet(?)
        GROUP BY p_mt ORDER BY p_mt
        """,
        [str(VOD_BY_TYPE_PATH)],
    ).fetchdf()
    con.close()

    monthly = tv_monthly.merge(vod_monthly, on="p_mt", how="outer").sort_values("p_mt")
    monthly["p_mt_str"] = monthly["p_mt"].astype(str).str[:4] + "-" + monthly["p_mt"].astype(str).str[4:]
    monthly = monthly.fillna(0)

    monthly.to_csv(OUTPUT_DIR / "monthly_tv_vod_by_type.csv", index=False, encoding="utf-8-sig")

    # 시각화: TV vs FOD vs RVOD vs SVOD (라인)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(len(monthly))

    ax1.plot(x, monthly["TV_avg"], "o-", color="#2c3e50", linewidth=2.5, label="TV 시청 (CH_HH_AVG)", marker="s", markersize=8)
    ax1.set_ylabel("TV 시청 가구 수 평균", color="#2c3e50", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, monthly["FOD_avg"] / 60, "o-", color="#3498db", linewidth=2, label="FOD (분)")
    ax2.plot(x, monthly["RVOD_avg"] / 60, "s-", color="#e74c3c", linewidth=2, label="RVOD (분)")
    ax2.plot(x, monthly["SVOD_avg"] / 60, "^-", color="#2ecc71", linewidth=2, label="SVOD (분)")
    ax2.plot(x, monthly["VOD_total_avg"] / 60, "d-", color="#9b59b6", linewidth=1.5, label="VOD 합계 (분)", linestyle="--")
    ax2.set_ylabel("VOD 시청 시간 (분) 평균", color="#2c3e50", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["p_mt_str"], rotation=45, ha="right")
    ax1.set_title("월별 TV vs VOD(FOD/RVOD/SVOD) 이용 트렌드 - 소비 방식 TV->VOD 확장", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tv_vod_expansion_by_type.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 막대 그래프: 월별 비교 (TV, FOD, RVOD, SVOD)
    fig2, ax = plt.subplots(figsize=(12, 6))
    w = 0.2
    x = range(len(monthly))
    ax.bar([i - 1.5 * w for i in x], monthly["TV_avg"], w, label="TV", color="#3498db", alpha=0.8)
    ax.bar([i - 0.5 * w for i in x], monthly["FOD_avg"] / 60, w, label="FOD(분)", color="#e74c3c", alpha=0.8)
    ax.bar([i + 0.5 * w for i in x], monthly["RVOD_avg"] / 60, w, label="RVOD(분)", color="#f39c12", alpha=0.8)
    ax.bar([i + 1.5 * w for i in x], monthly["SVOD_avg"] / 60, w, label="SVOD(분)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["p_mt_str"], rotation=45, ha="right")
    ax.set_ylabel("평균 (TV=가구수, VOD=분)")
    ax.set_title("월별 TV vs FOD/RVOD/SVOD - 소비 방식 확장")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tv_vod_expansion_stacked.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"저장: {OUTPUT_DIR / 'tv_vod_expansion_by_type.png'}")
    print(f"저장: {OUTPUT_DIR / 'tv_vod_expansion_stacked.png'}")
    print(f"저장: {OUTPUT_DIR / 'monthly_tv_vod_by_type.csv'}")
    print("\n[월별 요약]")
    print(monthly[["p_mt_str", "TV_avg", "FOD_avg", "RVOD_avg", "SVOD_avg"]].to_string(index=False))


if __name__ == "__main__":
    main()
