"""
TV 시청 감소 ≠ 이탈, 소비 방식 TV→VOD 확장 증명
1. 저TV·고VOD 그룹 분석 (해지율 비교)
2. 월별 TV vs VOD 트렌드 시각화 (월별 집계 VOD 사용)
3. VOD 이용량 vs 해지율 관계
"""
import os
from pathlib import Path

import duckdb
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
CHURN_PATH = BASE / "data" / "processed" / "churn_dataset.parquet"
MONTHLY_VOD_PATH = BASE / "data" / "processed" / "monthly_vod_by_user.parquet"
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

    if MONTHLY_VOD_PATH.exists():
        print("월별 VOD 집계 사용 (monthly_vod_by_user.parquet)")
        df = con.execute(
            """
            SELECT c.sha2_hash, c.cancel_yn, c.p_mt,
                   c.CH_HH_AVG_MONTH1, c.vod_view_cnt AS vod_view_cnt_total,
                   COALESCE(m.vod_use_tms_sum, 0) AS vod_use_tms_sum,
                   COALESCE(m.vod_view_cnt, 0) AS vod_view_cnt
            FROM read_parquet(?) c
            LEFT JOIN read_parquet(?) m ON c.sha2_hash = m.sha2_hash AND c.p_mt = m.p_mt
            WHERE c.p_mt >= 202301 AND c.p_mt <= 202312
              AND c.CH_HH_AVG_MONTH1 IS NOT NULL
            """,
            [str(CHURN_PATH), str(MONTHLY_VOD_PATH)],
        ).fetchdf()
    else:
        print("churn_dataset만 사용 (월별 VOD 없음 - build_monthly_vod_dataset.py 실행 필요)")
        df = con.execute(
            """
            SELECT sha2_hash, cancel_yn, p_mt,
                   CH_HH_AVG_MONTH1, vod_view_cnt, vod_use_tms_sum, vod_asset_cnt
            FROM read_parquet(?)
            WHERE p_mt >= 202301 AND p_mt <= 202312
              AND CH_HH_AVG_MONTH1 IS NOT NULL
            """,
            [str(CHURN_PATH)],
        ).fetchdf()

    con.close()

    # 사용자별 집계 (월별 여러 행 → 1행)
    user_df = df.groupby("sha2_hash").agg(
        cancel_yn=("cancel_yn", "first"),
        CH_HH_AVG_mean=("CH_HH_AVG_MONTH1", "mean"),
        vod_use_tms_sum=("vod_use_tms_sum", "mean"),
    ).reset_index()

    # ---- 1. 저TV·고VOD 그룹 분석 ----
    tv_q25 = user_df["CH_HH_AVG_mean"].quantile(0.25)
    vod_q75 = user_df["vod_use_tms_sum"].quantile(0.75)
    low_tv_high_vod = (user_df["CH_HH_AVG_mean"] <= tv_q25) & (user_df["vod_use_tms_sum"] >= vod_q75)
    other = ~low_tv_high_vod

    churn_low_tv_high_vod = user_df.loc[low_tv_high_vod, "cancel_yn"].mean()
    churn_other = user_df.loc[other, "cancel_yn"].mean()
    n_low_tv_high_vod = low_tv_high_vod.sum()

    seg_result = pd.DataFrame([
        {"그룹": "저TV·고VOD (TV 하위25% + VOD 상위25%)", "해지율": churn_low_tv_high_vod, "인원": n_low_tv_high_vod},
        {"그룹": "그 외", "해지율": churn_other, "인원": len(user_df) - n_low_tv_high_vod},
    ])
    seg_result.to_csv(OUTPUT_DIR / "segment_churn_ratio.csv", index=False, encoding="utf-8-sig")

    # 4분위 그룹 해지율
    user_df["TV_quartile"] = pd.qcut(user_df["CH_HH_AVG_mean"].rank(method="first"), 4, labels=["TV하위", "TV중하", "TV중상", "TV상위"])
    user_df["VOD_quartile"] = pd.qcut(user_df["vod_use_tms_sum"].fillna(0).rank(method="first"), 4, labels=["VOD하위", "VOD중하", "VOD중상", "VOD상위"])
    q4_churn = user_df.groupby(["TV_quartile", "VOD_quartile"], observed=True)["cancel_yn"].agg(["mean", "count"]).reset_index()
    q4_churn.columns = ["TV분위", "VOD분위", "해지율", "인원"]
    q4_churn.to_csv(OUTPUT_DIR / "tv_vod_quartile_churn.csv", index=False, encoding="utf-8-sig")

    # ---- 2. 월별 TV vs VOD 트렌드 ----
    monthly = df.groupby("p_mt").agg(
        TV_avg=("CH_HH_AVG_MONTH1", "mean"),
        VOD_avg=("vod_use_tms_sum", "mean"),
        view_cnt=("sha2_hash", "nunique"),
    ).reset_index()
    monthly["p_mt_str"] = monthly["p_mt"].astype(str).str[:4] + "-" + monthly["p_mt"].astype(str).str[4:]
    monthly.to_csv(OUTPUT_DIR / "monthly_tv_vod_trend.csv", index=False, encoding="utf-8-sig")

    use_monthly = "월별" if MONTHLY_VOD_PATH.exists() else "사용자전체"
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(monthly))
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(x, monthly["TV_avg"], "o-", color="#3498db", linewidth=2, label="TV 시청 (CH_HH_AVG)")
    l2 = ax1_twin.plot(x, monthly["VOD_avg"], "s-", color="#e74c3c", linewidth=2, label="VOD 시청 시간")
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["p_mt_str"], rotation=45, ha="right")
    ax1.set_ylabel("TV 시청 가구 수 평균", color="#3498db")
    ax1_twin.set_ylabel("VOD 시청 시간 평균", color="#e74c3c")
    ax1.set_title(f"월별 TV vs VOD 이용 트렌드 (2023) - VOD {use_monthly} 집계")
    lns = l1 + l2
    ax1.legend(lns, [l.get_label() for l in lns], loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_tv_vod_trend.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---- 3. VOD 이용량 vs 해지율 ----
    vod_val = user_df["vod_use_tms_sum"].fillna(0)
    user_df["VOD_quintile"] = pd.qcut(vod_val.rank(method="first"), 5, labels=["Q1(최소)", "Q2", "Q3", "Q4", "Q5(최대)"])
    vod_churn = user_df.groupby("VOD_quintile", observed=True)["cancel_yn"].agg(["mean", "count"]).reset_index()
    vod_churn.columns = ["VOD 이용량 분위", "해지율", "인원"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(vod_churn))
    bars = ax2.bar(x, vod_churn["해지율"], color=["#e74c3c" if r > vod_churn["해지율"].mean() else "#3498db" for r in vod_churn["해지율"]])
    ax2.axhline(vod_churn["해지율"].mean(), color="gray", linestyle="--", label="전체 평균")
    ax2.set_xticks(x)
    ax2.set_xticklabels(vod_churn["VOD 이용량 분위"])
    ax2.set_ylabel("해지율")
    ax2.set_title("VOD 이용량 분위별 해지율 (VOD 많이 쓸수록 해지율 감소)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vod_vs_churn.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 요약 대시보드
    fig3, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1) 저TV·고VOD vs 그 외
    ax = axes[0]
    ax.bar(["저TV·고VOD", "그 외"], [churn_low_tv_high_vod, churn_other], color=["#3498db", "#e74c3c"], alpha=0.8)
    ax.axhline(user_df["cancel_yn"].mean(), color="gray", linestyle="--")
    ax.set_ylabel("해지율")
    ax.set_title("1. TV 감소 ≠ 이탈\n저TV·고VOD 그룹 해지율 비교")

    # 2) 월별 트렌드 요약 (첫월 vs 마지막월)
    ax = axes[1]
    m1, m2 = monthly.iloc[0], monthly.iloc[-1]
    ax.bar([0, 1], [m1["TV_avg"], m2["TV_avg"]], width=0.35, label="TV", color="#3498db")
    ax.bar([0.35, 1.35], [m1["VOD_avg"] / 1000, m2["VOD_avg"] / 1000], width=0.35, label="VOD(천분)", color="#e74c3c")
    ax.set_xticks([0.175, 1.175])
    ax.set_xticklabels([str(m1["p_mt"]), str(m2["p_mt"])])
    ax.set_title("2. TV→VOD 확장\n2023년 초 vs 말")

    # 3) VOD 분위별 해지율
    ax = axes[2]
    ax.bar(range(len(vod_churn)), vod_churn["해지율"], color="#3498db", alpha=0.8)
    ax.set_xticks(range(len(vod_churn)))
    ax.set_xticklabels(vod_churn["VOD 이용량 분위"], rotation=30, ha="right")
    ax.set_ylabel("해지율")
    ax.set_title("3. VOD 많이 쓸수록 해지율 감소")

    plt.suptitle("TV 시청 감소 ≠ 이탈, 소비 방식 TV→VOD 확장 증명", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_dashboard.png", dpi=120, bbox_inches="tight")
    plt.close()

    # TV x VOD 분위 해지율 히트맵
    pivot_churn = q4_churn.pivot(index="TV분위", columns="VOD분위", values="해지율")
    pivot_churn = pivot_churn.reindex(["TV하위", "TV중하", "TV중상", "TV상위"])[["VOD하위", "VOD중하", "VOD중상", "VOD상위"]]
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    im = ax4.imshow(pivot_churn.values, cmap="RdYlGn_r", vmin=0.06, vmax=0.17, aspect="auto")
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(pivot_churn.columns)
    ax4.set_yticklabels(pivot_churn.index)
    ax4.set_xlabel("VOD 이용 분위")
    ax4.set_ylabel("TV 시청 분위")
    ax4.set_title("TV x VOD 분위별 해지율 (우상단=저TV+고VOD)")
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f"{pivot_churn.values[i,j]:.1%}", ha="center", va="center", fontsize=11)
    plt.colorbar(im, ax=ax4, label="해지율")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tv_vod_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 결과 요약 CSV
    vq1 = vod_churn["해지율"].iloc[0]
    vq5 = vod_churn["해지율"].iloc[-1]
    summary = [
        {"분석": "저TV·고VOD 해지율", "값": f"{churn_low_tv_high_vod:.2%}", "해석": "TV 줄고 VOD 많이 쓰는 그룹"},
        {"분석": "그 외 해지율", "값": f"{churn_other:.2%}", "해석": "대조군"},
        {"분석": "VOD Q5(최대) 해지율", "값": f"{vq5:.2%}", "해석": "VOD 가장 많이 쓰는 그룹"},
        {"분석": "VOD Q1(최소) 해지율", "값": f"{vq1:.2%}", "해석": "VOD 거의 안 쓰는 그룹"},
    ]
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n[결과 요약]")
    print(f"  저TV·고VOD 그룹 해지율: {churn_low_tv_high_vod:.2%} (n={n_low_tv_high_vod:,})")
    print(f"  그 외 해지율: {churn_other:.2%}")
    print(f"  VOD Q1 해지율: {vod_churn.iloc[0]['해지율']:.2%}, VOD Q5 해지율: {vod_churn.iloc[-1]['해지율']:.2%}")
    print(f"\n저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
