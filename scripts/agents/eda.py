"""EDA Agent — 그룹별 분포 비교·가설검정·시각화 (체크포인트 2).

.github/agents/eda-agent.agent.md 스펙 구현체. churn 은 cancel_yn(유지/해지),
vod_purchase 는 구매여부(vod_purchase_cnt>0) 로 그룹을 나눠 동일한 검정 절차를 적용한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"
OUTPUT_EDA = BASE / "output" / "eda"

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

DATASETS = {
    "churn": PROCESSED / "churn_dataset_v2.parquet",
    "vod_purchase": PROCESSED / "vod_purchase_dataset.parquet",
}
EXCLUDE_BASE = {
    "churn": {"sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN"},
    "vod_purchase": {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"},
}
GROUP_LABELS = {"churn": {0: "유지", 1: "해지"}, "vod_purchase": {0: "비구매", 1: "구매"}}
SAMPLE_SIZE = 50_000
NORMALITY_SUBSAMPLE = 2000
RANDOM_STATE = 42
ALPHA = 0.05
TOP_N_CHARTS = 5


def _group_series(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "churn":
        return df["cancel_yn"].astype(int)
    return (df["vod_purchase_cnt"] > 0).astype(int)


def _cohens_d(g1: pd.Series, g0: pd.Series) -> float:
    n1, n0 = len(g1), len(g0)
    pooled_std = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n0 - 1) * g0.var(ddof=1)) / (n1 + n0 - 2))
    return float((g1.mean() - g0.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.sum()
    r, k = table.shape
    denom = min(r - 1, k - 1) or 1
    return float(np.sqrt((chi2 / n) / denom))


def _numeric_test(g0: pd.Series, g1: pd.Series) -> tuple[str, float, float]:
    s0 = g0.sample(min(NORMALITY_SUBSAMPLE, len(g0)), random_state=RANDOM_STATE) if len(g0) > 8 else g0
    s1 = g1.sample(min(NORMALITY_SUBSAMPLE, len(g1)), random_state=RANDOM_STATE) if len(g1) > 8 else g1
    try:
        normal = stats.normaltest(s0).pvalue > ALPHA and stats.normaltest(s1).pvalue > ALPHA
    except ValueError:
        normal = False
    if normal:
        stat, p = stats.ttest_ind(g0, g1, equal_var=False)
        return "t-test", float(stat), float(p)
    stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    return "Mann-Whitney U", float(stat), float(p)


def run_eda(task: Literal["churn", "vod_purchase"]) -> dict:
    """그룹별 분포 비교 + 가설검정을 수행하고 hypothesis_tests.csv·차트·요약을 산출한다."""
    data_path = DATASETS[task]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 없음 — build_*.py 로 데이터셋 생성 필요")

    out_dir = OUTPUT_EDA / task
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    group = _group_series(sample, task)
    labels = GROUP_LABELS[task]
    target_col = "cancel_yn" if task == "churn" else "vod_purchase_cnt"
    exclude = EXCLUDE_BASE[task] | {target_col}

    feature_cols = [c for c in sample.columns if c not in exclude]
    numeric_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(sample[c]) and sample[c].dtype != bool
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    n_tests = len(numeric_cols) + len(categorical_cols)
    alpha_corrected = ALPHA / n_tests if n_tests >= 10 else ALPHA

    rows = []
    for c in numeric_cols:
        g0 = sample.loc[group == 0, c].dropna()
        g1 = sample.loc[group == 1, c].dropna()
        if len(g0) < 5 or len(g1) < 5:
            continue
        method, stat, p = _numeric_test(g0, g1)
        d = _cohens_d(g1, g0)
        rows.append({
            "변수": c, "검정": method, "통계량": stat, "p_value": p, "효과크기": d,
            f"{labels[0]}_평균": float(g0.mean()), f"{labels[1]}_평균": float(g1.mean()),
            "유의(보정)": bool(p < alpha_corrected),
        })

    for c in categorical_cols:
        ct = pd.crosstab(sample[c], group)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        v = _cramers_v(ct.values)
        rows.append({
            "변수": c, "검정": "chi2", "통계량": float(chi2), "p_value": float(p), "효과크기": v,
            f"{labels[0]}_평균": None, f"{labels[1]}_평균": None,
            "유의(보정)": bool(p < alpha_corrected),
        })

    hyp_df = pd.DataFrame(rows)
    hyp_df["효과크기_abs"] = hyp_df["효과크기"].abs()
    hyp_df = hyp_df.sort_values("효과크기_abs", ascending=False).drop(columns="효과크기_abs")

    hyp_path = out_dir / "hypothesis_tests.csv"
    hyp_df.to_csv(hyp_path, index=False, encoding="utf-8-sig")

    chart_paths = _make_charts(sample, group, labels, hyp_df, task, out_dir)

    insights = []
    for _, r in hyp_df.head(5).iterrows():
        if r["검정"] == "chi2":
            insights.append(f"`{r['변수']}` — Cramér's V={r['효과크기']:.3f}, p={r['p_value']:.4g} (범주형 연관성)")
        else:
            insights.append(
                f"`{r['변수']}` — {labels[1]} 평균 {r.get(f'{labels[1]}_평균', float('nan')):.3g} vs "
                f"{labels[0]} 평균 {r.get(f'{labels[0]}_평균', float('nan')):.3g} "
                f"(Cohen's d={r['효과크기']:.3f}, {r['검정']} p={r['p_value']:.4g})"
            )

    summary_lines = [
        f"# EDA Summary ({task})",
        "",
        f"- 샘플 {len(sample):,}건 — {labels[0]} {(group == 0).sum():,} / {labels[1]} {(group == 1).sum():,}",
        f"- 검정 변수 {n_tests}개, 유의수준 α={ALPHA} ({'Bonferroni 보정 적용' if n_tests >= 10 else '보정 없음'})",
        "",
        "## 핵심 인사이트 (효과크기 상위)",
    ] + [f"{i}. {line}" for i, line in enumerate(insights, 1)]
    summary_path = out_dir / "eda_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[eda] task={task} sample={len(sample)} tests={n_tests} charts={len(chart_paths)}")
    return {
        "eda_summary_path": str(summary_path.relative_to(BASE)).replace("\\", "/"),
        "hypothesis_tests_path": str(hyp_path.relative_to(BASE)).replace("\\", "/"),
        "chart_paths": chart_paths,
        "n_tests": n_tests,
    }


def _make_charts(
    sample: pd.DataFrame, group: pd.Series, labels: dict, hyp_df: pd.DataFrame, task: str, out_dir: Path
) -> list[str]:
    paths = []

    # 1. 상위 N개 수치형 변수: 그룹별 분포(히스토그램) — dist_{col}.png
    numeric_top = hyp_df[hyp_df["검정"].isin(["t-test", "Mann-Whitney U"])].head(TOP_N_CHARTS)["변수"].tolist()
    for c in numeric_top:
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, label, color in [(0, labels[0], "#3498db"), (1, labels[1], "#e74c3c")]:
            vals = sample.loc[group == g, c].dropna()
            ax.hist(vals, bins=40, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(f"{c} 분포 — {labels[0]} vs {labels[1]}")
        ax.legend()
        plt.tight_layout()
        p = out_dir / f"dist_{c}.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(str(p.relative_to(BASE)).replace("\\", "/"))

    # 2. 상위 1개 범주형 변수: 그룹별 비율 차 (CT_CL share diff 의 일반화)
    cat_top = hyp_df[hyp_df["검정"] == "chi2"].head(1)["변수"].tolist()
    for c in cat_top:
        share0 = sample.loc[group == 0, c].value_counts(normalize=True)
        share1 = sample.loc[group == 1, c].value_counts(normalize=True)
        diff = (share1 - share0).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in diff.values]
        ax.barh(diff.index.astype(str), diff.values, color=colors, alpha=0.8)
        ax.set_title(f"{c} 비율 차 ({labels[1]} − {labels[0]})")
        ax.invert_yaxis()
        plt.tight_layout()
        p = out_dir / f"share_diff_{c}.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(str(p.relative_to(BASE)).replace("\\", "/"))

    # 3~5. churn 전용 표준 차트 (컬럼 존재시에만)
    if "Total_Watch_Time" in sample.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, label, color in [(0, labels[0], "#3498db"), (1, labels[1], "#e74c3c")]:
            vals = np.log1p(sample.loc[group == g, "Total_Watch_Time"].dropna())
            ax.hist(vals, bins=40, alpha=0.5, label=label, color=color, density=True)
        ax.set_title("Total_Watch_Time 분포 (log scale)")
        ax.legend()
        plt.tight_layout()
        p = out_dir / "total_watch_time_log_dist.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(str(p.relative_to(BASE)).replace("\\", "/"))

    if "Recency" in sample.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, label, color in [(0, labels[0], "#3498db"), (1, labels[1], "#e74c3c")]:
            vals = sample.loc[group == g, "Recency"].dropna()
            ax.hist(vals, bins=40, alpha=0.5, label=label, color=color, density=True)
        ax.set_title("Recency 분포 (최근 시청 후 경과일)")
        ax.legend()
        plt.tight_layout()
        p = out_dir / "recency_hist.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(str(p.relative_to(BASE)).replace("\\", "/"))

    if "Risk_Segment_Kids_NFX" in sample.columns and task == "churn":
        rate = sample.groupby("Risk_Segment_Kids_NFX")[("cancel_yn")].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(rate.index.astype(str), rate.values, color="#e67e22", alpha=0.85)
        ax.set_title("Risk_Segment_Kids_NFX 별 해지율")
        ax.set_ylabel("해지율")
        plt.tight_layout()
        p = out_dir / "risk_segment_churn_rate.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(str(p.relative_to(BASE)).replace("\\", "/"))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA agent — 그룹별 분포비교·가설검정·시각화")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    args = parser.parse_args()
    result = run_eda(args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
