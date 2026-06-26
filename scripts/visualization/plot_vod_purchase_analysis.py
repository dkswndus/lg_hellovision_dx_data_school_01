"""
포트폴리오용: VOD 구매 예측 분석 시각화
  1. 상관관계 히트맵 (핵심 변수 15개)
  2. VIF 다중공선성 테이블
  3. 가설검정 (5개 가설, Mann-Whitney U + 효과크기)
"""
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "vod_purchase_dataset.parquet"
OUTPUT_DIR = BASE / "output" / "vod_purchase_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_N = 50_000
SEED = 42

CORR_COLS = [
    "vod_purchase_cnt",
    "lagged_rvod_cnt",
    "vod_view_cnt",
    "vod_use_tms_sum",
    "vod_asset_cnt",
    "fod_cnt",
    "svod_cnt",
    "CH_HH_AVG_MONTH1",
    "KIDS_USE_PV_MONTH1",
    "TOTAL_USED_DAYS",
    "AGMT_END_SEG",
    "SVOD_SCRB_CNT_GRP",
    "NFX_USE_YN",
    "YTB_USE_YN",
    "cancel_yn",
]

CORR_LABELS = [
    "VOD구매건수(y)",
    "전월RVOD구매",
    "VOD시청건수",
    "VOD시청시간",
    "VOD자산수",
    "FOD건수",
    "SVOD건수",
    "채널시청빈도",
    "키즈이용량",
    "가입기간(일)",
    "계약만료구간",
    "SVOD구독수",
    "넷플릭스이용",
    "유튜브이용",
    "해지여부",
]


def load_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=SEED).copy()
    for c in sample.columns:
        if not pd.api.types.is_numeric_dtype(sample[c]):
            sample[c] = sample[c].astype("category").cat.codes
    return sample.fillna(sample.median(numeric_only=True))


# ── 1. 상관관계 히트맵 ────────────────────────────────────────────────────────
def plot_correlation(sample: pd.DataFrame):
    cols_avail = [c for c in CORR_COLS if c in sample.columns]
    labels_avail = [CORR_LABELS[CORR_COLS.index(c)] for c in cols_avail]

    corr = sample[cols_avail].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols_avail)))
    ax.set_yticks(range(len(cols_avail)))
    ax.set_xticklabels(labels_avail, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels_avail, fontsize=9)

    for i in range(len(cols_avail)):
        for j in range(len(cols_avail)):
            v = corr.iloc[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, label="Pearson 상관계수", shrink=0.8)
    ax.set_title(
        "VOD 구매 예측 — 핵심 변수 상관관계 행렬 (n=50,000 샘플)",
        fontsize=13, pad=15,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "vod_purchase_correlation_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


# ── 2. VIF 테이블 ─────────────────────────────────────────────────────────────
def plot_vif(sample: pd.DataFrame):
    feature_cols = [c for c in CORR_COLS if c in sample.columns and c != "vod_purchase_cnt"]

    X = sample[feature_cols].copy()
    vif_rows = []
    for col in feature_cols:
        others = [c for c in feature_cols if c != col]
        lr = LinearRegression().fit(X[others].values, X[col].values)
        r2 = lr.score(X[others].values, X[col].values)
        vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else float("inf")
        label = CORR_LABELS[CORR_COLS.index(col)]
        vif_rows.append({"변수": label, "VIF": round(vif, 2)})

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(OUTPUT_DIR / "vod_purchase_vif_table.csv", index=False, encoding="utf-8-sig")

    def judge(v):
        if v > 10:
            return ("위험(>10)", "#e74c3c")
        elif v > 5:
            return ("주의(>5)", "#f39c12")
        else:
            return ("양호(≤5)", "#27ae60")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    rows = []
    cell_colors = []
    for _, row in vif_df.iterrows():
        text, color = judge(row["VIF"])
        rows.append([row["변수"], f"{row['VIF']:.2f}", text])
        cell_colors.append(["#f8f9fa", "#f8f9fa", color + "33"])

    tbl = ax.table(
        cellText=rows,
        colLabels=["변수", "VIF", "판정"],
        loc="center",
        cellLoc="center",
        colWidths=[0.52, 0.22, 0.26],
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.6)
    for j in range(3):
        tbl[0, j].set_facecolor("#3498db")
        tbl[0, j].set_text_props(color="white", weight="bold")

    ax.set_title(
        "VOD 구매 예측 — VIF(분산팽창계수) 다중공선성 점검",
        fontsize=13, pad=20,
    )

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="양호 (VIF ≤ 5)"),
        mpatches.Patch(color="#f39c12", label="주의 (VIF 5~10)"),
        mpatches.Patch(color="#e74c3c", label="위험 (VIF > 10)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / "vod_purchase_vif_table.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


# ── 3. 가설검정 ───────────────────────────────────────────────────────────────
HYPOTHESES = [
    {
        "id": "H1",
        "title": "전월 구매 경험 → 당월 구매 증가",
        "desc": "lagged_rvod_cnt > 0\n(전월 구매 경험자)",
        "group_col": "lagged_rvod_cnt",
        "group_fn": lambda df: df["lagged_rvod_cnt"] > 0,
        "label_pos": "경험 있음",
        "label_neg": "경험 없음",
        "expected": "양(+)",
    },
    {
        "id": "H2",
        "title": "SVOD 구독자 → RVOD 구매 증가",
        "desc": "SVOD_SCRB_CNT_GRP > 0\n(SVOD 구독자)",
        "group_col": "SVOD_SCRB_CNT_GRP",
        "group_fn": lambda df: df["SVOD_SCRB_CNT_GRP"] > 0,
        "label_pos": "SVOD 구독",
        "label_neg": "미구독",
        "expected": "양(+)",
    },
    {
        "id": "H3",
        "title": "넷플릭스 사용자 → RVOD 구매 감소",
        "desc": "NFX_USE_YN == 1\n(넷플릭스 이용자)",
        "group_col": "NFX_USE_YN",
        "group_fn": lambda df: df["NFX_USE_YN"] == 1,
        "label_pos": "NFX 이용",
        "label_neg": "NFX 미이용",
        "expected": "음(-)",
    },
    {
        "id": "H4",
        "title": "키즈 콘텐츠 이용자 → RVOD 구매 증가",
        "desc": "KIDS_USE_PV_MONTH1 > 0\n(키즈 콘텐츠 이용)",
        "group_col": "KIDS_USE_PV_MONTH1",
        "group_fn": lambda df: df["KIDS_USE_PV_MONTH1"] > 0,
        "label_pos": "키즈 이용",
        "label_neg": "미이용",
        "expected": "양(+)",
    },
    {
        "id": "H5",
        "title": "해지 고객 → RVOD 구매 감소",
        "desc": "cancel_yn == 1\n(해지 고객)",
        "group_col": "cancel_yn",
        "group_fn": lambda df: df["cancel_yn"] == 1,
        "label_pos": "해지",
        "label_neg": "유지",
        "expected": "음(-)",
    },
]


def rank_biserial(group1, group2):
    n1, n2 = len(group1), len(group2)
    stat, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return (2 * stat) / (n1 * n2) - 1


def plot_hypothesis(df: pd.DataFrame):
    y = "vod_purchase_cnt"
    n_h = len(HYPOTHESES)

    fig, axes = plt.subplots(2, n_h, figsize=(16, 9),
                              gridspec_kw={"height_ratios": [2.5, 1]})

    colors_pos = "#3498db"
    colors_neg = "#bdc3c7"

    results_summary = []

    for col_i, h in enumerate(HYPOTHESES):
        ax_bar = axes[0, col_i]
        ax_text = axes[1, col_i]

        if h["group_col"] not in df.columns:
            ax_bar.set_visible(False)
            ax_text.set_visible(False)
            continue

        mask = h["group_fn"](df)
        g_pos = df.loc[mask, y].clip(upper=df[y].quantile(0.99))
        g_neg = df.loc[~mask, y].clip(upper=df[y].quantile(0.99))

        mean_pos = g_pos.mean()
        mean_neg = g_neg.mean()

        stat, p_val = stats.mannwhitneyu(g_pos, g_neg, alternative="two-sided")
        rb = rank_biserial(g_pos.values, g_neg.values)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        direction_ok = (mean_pos > mean_neg) == (h["expected"] == "양(+)")

        bar_colors = [colors_pos, colors_neg]
        bars = ax_bar.bar(
            [h["label_pos"], h["label_neg"]],
            [mean_pos, mean_neg],
            color=bar_colors,
            width=0.5,
            edgecolor="white",
        )
        for bar, val in zip(bars, [mean_pos, mean_neg]):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, weight="bold",
            )

        title_color = "#27ae60" if direction_ok else "#e74c3c"
        ax_bar.set_title(
            f"{h['id']}: {h['title']}",
            fontsize=9.5, weight="bold", color=title_color, pad=6,
        )
        ax_bar.set_ylabel("평균 VOD 구매건수", fontsize=8)
        ax_bar.tick_params(axis="x", labelsize=9)
        ax_bar.tick_params(axis="y", labelsize=8)
        y_max = max(mean_pos, mean_neg) * 1.4
        ax_bar.set_ylim(0, max(y_max, 0.01))

        n_pos = len(g_pos)
        n_neg = len(g_neg)

        if p_val < 0.001:
            p_str = "p < 0.001"
        else:
            p_str = f"p = {p_val:.4f}"

        result_text = (
            f"p값: {p_str}  {sig}\n"
            f"효과크기 (rb): {rb:.3f}\n"
            f"n({h['label_pos']}): {n_pos:,}\n"
            f"n({h['label_neg']}): {n_neg:,}\n"
            f"예상 방향: {h['expected']}  {'[O]' if direction_ok else '[X]'}"
        )

        ax_text.axis("off")
        box_color = "#eafaf1" if direction_ok else "#fdedec"
        ax_text.text(
            0.5, 0.5, result_text,
            transform=ax_text.transAxes,
            ha="center", va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor="#ccc", alpha=0.9),
        )

        results_summary.append({
            "가설": h["id"],
            "설명": h["title"],
            "p값": round(p_val, 6),
            "유의성": sig,
            "효과크기(rb)": round(rb, 4),
            "예상방향": h["expected"],
            "결과": "채택" if direction_ok and sig != "ns" else "기각",
        })

    plt.suptitle(
        "VOD 구매 예측 — 핵심 가설 5개 검증 (Mann-Whitney U, 단측·양측 혼합)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "vod_purchase_hypothesis.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")

    sum_df = pd.DataFrame(results_summary)
    sum_df.to_csv(OUTPUT_DIR / "vod_purchase_hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT_DIR / 'vod_purchase_hypothesis_summary.csv'}")
    print(sum_df.to_string(index=False))


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    if not DATA_PATH.exists():
        print("vod_purchase_dataset 없음. build_vod_purchase_dataset.py 실행 필요.")
        return

    print("데이터 로드 중...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  전체: {len(df):,}행")

    sample = load_sample(df)
    print(f"  샘플: {len(sample):,}행")

    print("\n[1/3] 상관관계 히트맵 생성...")
    plot_correlation(sample)

    print("\n[2/3] VIF 테이블 생성...")
    plot_vif(sample)

    print("\n[3/3] 가설검정 시각화 생성...")
    plot_hypothesis(df)

    print("\n완료. 저장 위치:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
