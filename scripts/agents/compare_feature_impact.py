"""원본 컬럼 vs 파생변수 포함 모델 성능 비교 CLI.

modeling.py 로 만든 output/models/leaderboard_{task}.csv (파생변수 포함) 와
output/models/leaderboard_{task}_raw.csv (--drop-features 로 파생변수 제외) 를
모델별로 비교해 개선폭을 보여준다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
OUTPUT_MODELS = BASE / "output" / "models"

METRIC_COL = {"churn": "Val PRC-AUC", "vod_purchase": "Val R2"}


def compare(task: Literal["churn", "vod_purchase"]) -> pd.DataFrame:
    metric = METRIC_COL[task]
    engineered_path = OUTPUT_MODELS / f"leaderboard_{task}.csv"
    raw_path = OUTPUT_MODELS / f"leaderboard_{task}_raw.csv"
    for p in (engineered_path, raw_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 — modeling.py 를 먼저 실행 (variant=raw 포함)")

    engineered = pd.read_csv(engineered_path, encoding="utf-8-sig")[["모델", metric]].rename(
        columns={metric: "파생변수 포함"}
    )
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")[["모델", metric]].rename(columns={metric: "원본 컬럼만"})

    merged = engineered.merge(raw, on="모델", how="outer")
    merged["개선폭"] = merged["파생변수 포함"] - merged["원본 컬럼만"]
    merged["개선율"] = (merged["개선폭"] / merged["원본 컬럼만"].abs()) * 100
    return merged.sort_values("파생변수 포함", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 컬럼 vs 파생변수 모델 성능 비교")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    args = parser.parse_args()

    df = compare(args.task)
    metric = METRIC_COL[args.task]
    print(f"\n=== {args.task} ({metric} 기준) ===\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
