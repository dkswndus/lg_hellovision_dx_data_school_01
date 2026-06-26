"""
발표용 누락 그래프 2개 생성
1. 키즈·영화 콘텐츠 이용 여부별 해지율 (Flag_Kids, Flag_Movie 설계 근거)
2. TV/VOD 소비 패턴 세그먼트별 해지율 (Risk_Segment 설계 근거)
"""
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
V2_PATH  = BASE / "data" / "processed" / "churn_dataset_v2.parquet"
V1_PATH  = BASE / "data" / "processed" / "churn_dataset.parquet"
OUT_DIR  = BASE / "output" / "churn_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE  = "#003876"
RED   = "#CC0000"
GRAY  = "#888888"
GREEN = "#1A6B2A"


# ──────────────────────────────────────────────
# 그래프 1: Flag_Kids · Flag_Movie 별 해지율
# ──────────────────────────────────────────────
def _load_content_data():
    df = pd.read_parquet(V2_PATH)
    df["vod_active"] = (df["vod_view_cnt"] > 0).astype(int)
    df["kids_vod"]   = (df["KIDS_USE_PV_MONTH1"] > 0).astype(int)
    return df


def plot_radar_content():
    df = pd.read_parquet(V2_PATH)
    df["vod_active"] = (df["vod_view_cnt"] > 0).astype(int)
    df["kids_vod"]   = (df["KIDS_USE_PV_MONTH1"] > 0).astype(int)

    RADAR_LABELS = ["키즈 콘텐츠", "영화 콘텐츠", "Netflix", "YouTube", "VOD 시청", "키즈 VOD"]
    RADAR_COLS   = ["Flag_Kids", "Flag_Movie", "Is_Netflix", "YTB_USE_YN", "vod_active", "kids_vod"]

    churners  = df[df["cancel_yn"] == 1]
    retainers = df[df["cancel_yn"] == 0]

    vals_churn = [churners[c].mean() * 100  for c in RADAR_COLS]
    vals_keep  = [retainers[c].mean() * 100 for c in RADAR_COLS]

    N = len(RADAR_LABELS)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vc = vals_churn + vals_churn[:1]
    vk = vals_keep  + vals_keep[:1]

    fig = plt.figure(figsize=(10, 2))
    fig.suptitle(
        "해지자 vs 유지자 콘텐츠 취향 비교 — 키즈·영화 콘텐츠 락인 효과",
        fontsize=9, fontweight="bold", y=1.04,
    )

    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vc, "o-", linewidth=1.8, color=RED,  label=f"해지자 (n={len(churners):,})")
    ax.fill(angles, vc, alpha=0.15, color=RED)
    ax.plot(angles, vk, "o-", linewidth=1.8, color=BLUE, label=f"유지자 (n={len(retainers):,})")
    ax.fill(angles, vk, alpha=0.15, color=BLUE)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=7, fontweight="bold")
    ax.set_yticklabels([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=7, framealpha=0.9)
    ax.grid(color="gray", alpha=0.3)

    for angle, val_c in zip(angles[:-1], vals_churn):
        ax.text(angle, val_c + 4, f"{val_c:.1f}%",
                ha="center", va="center", fontsize=6, color=RED, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "radar_content_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"저장: {out}")


def plot_bubble_content():
    df = pd.read_parquet(V2_PATH)

    SEGMENTS = [
        ("키즈 콘텐츠",      df["Flag_Kids"]  == 1,                                                                   GREEN,     ( 10, -14)),
        ("영화 콘텐츠",      df["Flag_Movie"] == 1,                                                                   "#2980B9", ( 10,   6)),
        ("Netflix",          df["Is_Netflix"] == 1,                                                                   "#E74C3C", (-75,   6)),
        ("YouTube",          df["YTB_USE_YN"] == 1,                                                                   "#F39C12", ( 10, -14)),
        ("미이용(무콘텐츠)", (df["Flag_Kids"]==0)&(df["Flag_Movie"]==0)&(df["Is_Netflix"]==0)&(df["YTB_USE_YN"]==0), GRAY,      (-85,   6)),
    ]

    records = []
    for name, mask, color, offset in SEGMENTS:
        sub = df[mask]
        records.append({
            "name": name, "n": len(sub),
            "churn_pct": sub["cancel_yn"].mean() * 100,
            "vod_avg":   sub["vod_view_cnt"].mean(),
            "color": color, "offset": offset,
        })
    bdf = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.suptitle(
        "콘텐츠 유형별 이용자 수(버블 크기) × 해지율",
        fontsize=9, fontweight="bold", y=1.04,
    )

    size_scale = (bdf["n"] / bdf["n"].max() * 1200).clip(lower=80)
    ax.scatter(bdf["vod_avg"], bdf["churn_pct"],
               s=size_scale, c=bdf["color"], alpha=0.75, edgecolors="white", linewidths=1.5)

    overall_churn = df["cancel_yn"].mean() * 100
    ax.axhline(overall_churn, color=GRAY, linestyle="--", linewidth=1.2,
               label=f"전체 평균 {overall_churn:.1f}%")

    for _, row in bdf.iterrows():
        ax.annotate(
            f"{row['name']}  {row['churn_pct']:.1f}%\n(n={row['n']:,})",
            xy=(row["vod_avg"], row["churn_pct"]),
            xytext=row["offset"], textcoords="offset points",
            fontsize=6.5, fontweight="bold", color="#1a1a1a",
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8),
        )

    ax.set_xlabel("VOD 평균 시청 횟수 (회/월)", fontsize=7)
    ax.set_ylabel("해지율 (%)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_xlim(-10, bdf["vod_avg"].max() * 1.6)
    ax.set_ylim(bdf["churn_pct"].min() - 1.5, bdf["churn_pct"].max() + 3)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "bubble_content_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


# ──────────────────────────────────────────────
# 그래프 2: TV·VOD 4분면 세그먼트별 해지율
# ──────────────────────────────────────────────
def plot_risk_segment():
    df = pd.read_parquet(V1_PATH)

    tv_col  = "CH_HH_AVG_MONTH1"
    vod_col = "vod_use_tms_sum"

    sub = df[[tv_col, vod_col, "cancel_yn"]].dropna()

    tv_q25  = sub[tv_col].quantile(0.25)
    vod_q75 = sub[vod_col].quantile(0.75)

    def assign_seg(row):
        tv_low  = row[tv_col]  <= tv_q25
        vod_high = row[vod_col] >= vod_q75
        if tv_low and vod_high:
            return "저TV·고VOD\n(핵심 위험군)"
        elif not tv_low and vod_high:
            return "고TV·고VOD\n(충성 고객)"
        elif tv_low and not vod_high:
            return "저TV·저VOD\n(이탈 잠재)"
        else:
            return "고TV·저VOD\n(TV 중심)"

    sub = sub.copy()
    sub["segment"] = sub.apply(assign_seg, axis=1)

    seg_order = ["저TV·고VOD\n(핵심 위험군)", "저TV·저VOD\n(이탈 잠재)",
                 "고TV·저VOD\n(TV 중심)", "고TV·고VOD\n(충성 고객)"]

    stats = (
        sub.groupby("segment")["cancel_yn"]
        .agg(churn_rate="mean", count="count")
        .reindex(seg_order)
        .reset_index()
    )
    stats["churn_pct"] = stats["churn_rate"] * 100
    overall = sub["cancel_yn"].mean() * 100

    seg_colors = [RED, "#E67E22", "#3498DB", GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(
        "TV·VOD 소비 패턴 4분면 세그먼트별 해지율 — Risk_Segment 설계 근거",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # 왼쪽: 막대 그래프
    ax = axes[0]
    bars = ax.bar(stats["segment"], stats["churn_pct"],
                  color=seg_colors, width=0.55, edgecolor="white", linewidth=1.5)

    ax.axhline(overall, color=GRAY, linestyle="--", linewidth=1.8, label=f"전체 평균 {overall:.2f}%")

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{row['churn_pct']:.2f}%\n(n={row['count']:,})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("해지율 (%)", fontsize=11)
    ax.set_title("세그먼트별 해지율", fontsize=12, fontweight="bold")
    ax.set_ylim(0, stats["churn_pct"].max() * 1.35)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # 오른쪽: 4분면 매트릭스
    ax2 = axes[1]
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.set_xticks([0.5, 1.5])
    ax2.set_xticklabels(["저 VOD", "고 VOD"], fontsize=11)
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(["저 TV", "고 TV"], fontsize=11)
    ax2.set_title("4분면 세그먼트 매트릭스", fontsize=12, fontweight="bold")

    quadrants = [
        # (x, y, label, color)
        (0, 0, "저TV·저VOD\n(이탈 잠재)", "#E67E22"),
        (1, 0, "저TV·고VOD\n(핵심 위험군)", RED),
        (0, 1, "고TV·저VOD\n(TV 중심)",    "#3498DB"),
        (1, 1, "고TV·고VOD\n(충성 고객)",  GREEN),
    ]

    for qx, qy, qlabel, qcolor in quadrants:
        rect = mpatches.FancyBboxPatch(
            (qx + 0.05, qy + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=qcolor, alpha=0.25, edgecolor=qcolor, linewidth=2,
        )
        ax2.add_patch(rect)

        seg_key = qlabel
        row_match = stats[stats["segment"] == seg_key]
        pct_text = f"\n해지율 {row_match['churn_pct'].values[0]:.1f}%" if len(row_match) else ""
        ax2.text(
            qx + 0.5, qy + 0.5,
            qlabel + pct_text,
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1a1a1a",
        )

    ax2.axhline(1, color="#555", linewidth=1.5, linestyle="--", alpha=0.6)
    ax2.axvline(1, color="#555", linewidth=1.5, linestyle="--", alpha=0.6)
    ax2.set_xlabel("VOD 이용량 (하위 75% / 상위 25%)", fontsize=10)
    ax2.set_ylabel("TV 시청량 (하위 25% / 상위 75%)", fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / "risk_segment_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    plot_radar_content()
    plot_bubble_content()
    plot_risk_segment()
    print("완료.")
