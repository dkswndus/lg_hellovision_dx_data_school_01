"""
채널 시청 예측용 데이터셋 생성
- 종속변수: CH_HH_AVG_MONTH1
- 독립변수: log_VOD_TIME, Adult, FOD, Trend, INHOME_RATE, AGE_GRP10
- Trend: 2023 갤럽 월별 인기 Top5 (현재 churn_dataset에는 월별 vod 없음 → placeholder)
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"
CHURN_PATH = PROCESSED / "churn_dataset.parquet"
OUTPUT_PATH = PROCESSED / "channel_model_dataset.parquet"
SAMPLE_N = 100_000
RANDOM_STATE = 42


def main():
    con = duckdb.connect(":memory:")

    df = con.execute(
        """
        SELECT sha2_hash, p_mt, CH_HH_AVG_MONTH1 AS y,
               INHOME_RATE, AGE_GRP10, vod_use_tms_sum
        FROM read_parquet(?)
        WHERE p_mt >= 202301 AND p_mt <= 202312
          AND CH_HH_AVG_MONTH1 IS NOT NULL
        """,
        [str(CHURN_PATH)],
    ).fetchdf()

    if len(df) == 0:
        raise ValueError("churn_dataset에 2023 p_mt 없음")

    df = df.sample(n=min(SAMPLE_N, len(df)), random_state=RANDOM_STATE)
    df["log_VOD_TIME"] = np.log1p(df["vod_use_tms_sum"].fillna(0))
    df["INHOME_RATE"] = pd.to_numeric(df["INHOME_RATE"], errors="coerce").fillna(0)
    df["AGE_GRP10"] = pd.to_numeric(df["AGE_GRP10"], errors="coerce").fillna(5)

    # Adult, FOD, Trend: vod_log/ vod_content 상세 조인 필요 → placeholder
    df["Adult"] = 0.0
    df["FOD"] = 0.0
    df["Trend"] = 0.0

    out = df[["sha2_hash", "p_mt", "y", "log_VOD_TIME", "Adult", "FOD", "Trend", "INHOME_RATE", "AGE_GRP10"]].copy()
    out = out.dropna(subset=["y"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"저장: {OUTPUT_PATH} ({len(out):,}행)")


if __name__ == "__main__":
    main()
