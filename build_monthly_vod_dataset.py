"""
vod_log 월별 집계 생성
- sha2_hash, p_mt(월) 기준 vod_use_tms_sum, vod_view_cnt
- churn_dataset 사용자만 대상 (처리 효율)
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).parent
INTERIM = BASE / "data" / "interim"
PROCESSED = BASE / "data" / "processed"
OUTPUT_PATH = PROCESSED / "monthly_vod_by_user.parquet"

VOD_LOG = INTERIM / "vod_log_clean.parquet"
CHURN_PATH = PROCESSED / "churn_dataset.parquet"


def main():
    con = duckdb.connect(":memory:")

    # churn_dataset의 sha2_hash (2023년 포함된 사용자)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE sample_users AS
        SELECT DISTINCT sha2_hash
        FROM read_parquet(?)
        WHERE p_mt >= 202301 AND p_mt <= 202312
        """,
        [str(CHURN_PATH)],
    )
    n_users = con.execute("SELECT COUNT(*) FROM sample_users").fetchone()[0]
    print(f"대상 사용자: {n_users:,}명")

    # vod_log 월별 집계 (strt_dt에서 월 추출)
    print("vod_log 월별 집계 중...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE monthly_vod AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COUNT(*) AS vod_view_cnt,
            COALESCE(SUM(vl.use_tms), 0)::BIGINT AS vod_use_tms_sum
        FROM read_parquet(?) vl
        INNER JOIN sample_users s ON vl.sha2_hash = s.sha2_hash
        WHERE vl.strt_dt IS NOT NULL
          AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) BETWEEN 202301 AND 202312
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG)],
    )

    n_rows = con.execute("SELECT COUNT(*) FROM monthly_vod").fetchone()[0]
    print(f"월별 VOD 집계: {n_rows:,}행")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    con.execute("COPY monthly_vod TO ? (FORMAT PARQUET)", [str(OUTPUT_PATH)])
    con.close()
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
