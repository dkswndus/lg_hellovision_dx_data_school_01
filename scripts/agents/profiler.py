"""Data Profiler Agent — 결측·dtype·타깃 분포 자동 진단.

.github/agents/data-profiler-agent.agent.md 스펙 구현체. data/processed/{dataset}.parquet
(modeling.py 가 읽는 동일 파일) 을 진단해 output/profile/{table}_profile.json + .md 를 생성한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"
OUTPUT_PROFILE = BASE / "output" / "profile"

DATASETS = {
    "churn": PROCESSED / "churn_dataset_v2.parquet",
    "vod_purchase": PROCESSED / "vod_purchase_dataset.parquet",
}
TARGET_COL = {"churn": "cancel_yn", "vod_purchase": "vod_purchase_cnt"}
TABLE_NAME = {"churn": "churn_dataset_v2", "vod_purchase": "vod_purchase_dataset"}

MISSING_WARN = 0.30
MISSING_DROP = 0.70
CHURN_RATIO_TARGET = 0.20
CHURN_RATIO_TOL = 0.05


def _basic_stats(series: pd.Series) -> dict:
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        return {k: (float(desc[k]) if k in desc.index else None) for k in ["mean", "std", "min", "max"]} | {
            "p25": float(desc.get("25%", np.nan)),
            "p50": float(desc.get("50%", np.nan)),
            "p75": float(desc.get("75%", np.nan)),
        }
    top = series.value_counts(dropna=True).head(5)
    return {"top_k": [{"value": str(v), "count": int(c)} for v, c in top.items()]}


def profile_dataset(task: Literal["churn", "vod_purchase"]) -> dict:
    """data/processed/{dataset}.parquet 를 읽어 진단 리포트를 산출한다."""
    data_path = DATASETS[task]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 없음 — build_*.py 로 데이터셋 생성 필요")

    df = pd.read_parquet(data_path)
    table = TABLE_NAME[task]
    target = TARGET_COL[task]

    dtypes_map = {c: str(df[c].dtype) for c in df.columns}
    missing = {c: float(df[c].isna().mean()) for c in df.columns}
    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
    basic_stats = {c: _basic_stats(df[c]) for c in df.columns}

    if df[target].nunique() <= 10:
        vc = df[target].value_counts(normalize=True).sort_index()
        target_distribution = {str(k): float(v) for k, v in vc.items()}
    else:
        target_distribution = {k: float(v) for k, v in df[target].describe().items()}

    warnings: list[dict] = []
    for c, ratio in missing.items():
        if ratio > MISSING_DROP:
            warnings.append({"col": c, "issue": f"결측률 {ratio:.1%} (>70%)", "action": "drop 제안"})
        elif ratio > MISSING_WARN:
            warnings.append({"col": c, "issue": f"결측률 {ratio:.1%} (>30%)", "action": "경고"})
    for c, n in cardinality.items():
        if n <= 1:
            warnings.append({"col": c, "issue": "n_unique==1", "action": "drop 제안"})

    if task == "churn":
        pos_ratio = float(df[target].mean())
        if abs(pos_ratio - CHURN_RATIO_TARGET) > CHURN_RATIO_TOL:
            warnings.append({
                "col": target,
                "issue": f"타깃비율 {pos_ratio:.1%} (80:20 기준 ±5%p 이탈)",
                "action": "샘플링 재확인",
            })

    profile = {
        "table": table,
        "row_count": int(len(df)),
        "col_count": int(df.shape[1]),
        "dtypes_map": dtypes_map,
        "missing": missing,
        "basic_stats": basic_stats,
        "target_distribution": target_distribution,
        "cardinality": cardinality,
        "warnings": warnings,
    }

    OUTPUT_PROFILE.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_PROFILE / f"{table}_profile.json"
    json_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# {table} 프로파일",
        "",
        f"- 행 수: {profile['row_count']:,}",
        f"- 컬럼 수: {profile['col_count']}",
        f"- 타깃(`{target}`) 분포: {target_distribution}",
        "",
        "## 경고",
    ]
    md_lines += [f"- `{w['col']}`: {w['issue']} → {w['action']}" for w in warnings] or ["- 없음"]
    md_path = OUTPUT_PROFILE / f"{table}_profile.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[profiler] task={task} table={table} rows={profile['row_count']} "
        f"cols={profile['col_count']} warnings={len(warnings)}"
    )
    return {
        "profile_path": str(json_path.relative_to(BASE)).replace("\\", "/"),
        "profile_md_path": str(md_path.relative_to(BASE)).replace("\\", "/"),
        "row_count": profile["row_count"],
        "col_count": profile["col_count"],
        "warning_count": len(warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Profiler agent — 결측·dtype·타깃분포 진단")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    args = parser.parse_args()
    result = profile_dataset(args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
