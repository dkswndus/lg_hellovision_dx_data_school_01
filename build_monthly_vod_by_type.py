"""
vod_log 월별 + FOD/RVOD/SVOD 타입별 집계
- TV(월별) vs FOD vs RVOD vs SVOD 트렌드로 "소비 방식 TV→VOD 확장" 시각화
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).parent
INTERIM = BASE / "data" / "interim"
PROCESSED = BASE / "data" / "processed"
OUTPUT_PATH = PROCESSED / "monthly_vod_by_type.parquet"

VOD_LOG = INTERIM / "vod_log_clean.parquet"
VOD_CONTENT = INTERIM / "vod_content_clean.parquet"
CHURN_PATH = PROCESSED / "churn_dataset.parquet"


def main():
    con = duckdb.connect(":memory:")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE sample_users AS
        SELECT DISTINCT sha2_hash FROM read_parquet(?)
        WHERE p_mt >= 202301 AND p_mt <= 202312
        """,
        [str(CHURN_PATH)],
    )
    n = con.execute("SELECT COUNT(*) FROM sample_users").fetchone()[0]
    print(f"대상 사용자: {n:,}명")

    # vod_log + vod_content 조인 (asset = full_asset_id), asset_prod로 FOD/RVOD/SVOD 구분
    # asset_prod: 0=FOD, 1=RVOD, 2=SVOD
    print("vod_log + vod_content 조인, FOD/RVOD/SVOD 월별 집계 중...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE monthly_by_type AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COALESCE(SUM(CASE WHEN vc.asset_prod = 0 THEN vl.use_tms ELSE 0 END), 0)::BIGINT AS fod_tms,
            COALESCE(SUM(CASE WHEN vc.asset_prod = 1 THEN vl.use_tms ELSE 0 END), 0)::BIGINT AS rvod_tms,
            COALESCE(SUM(CASE WHEN vc.asset_prod = 2 THEN vl.use_tms ELSE 0 END), 0)::BIGINT AS svod_tms,
            COALESCE(SUM(vl.use_tms), 0)::BIGINT AS total_tms
        FROM read_parquet(?) vl
        INNER JOIN sample_users s ON vl.sha2_hash = s.sha2_hash
        LEFT JOIN read_parquet(?) vc ON vl.asset = vc.full_asset_id
        WHERE vl.strt_dt IS NOT NULL
          AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) BETWEEN 202301 AND 202312
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG), str(VOD_CONTENT)],
    )

    n_rows = con.execute("SELECT COUNT(*) FROM monthly_by_type").fetchone()[0]
    print(f"월별 타입별 집계: {n_rows:,}행")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    con.execute("COPY monthly_by_type TO ? (FORMAT PARQUET)", [str(OUTPUT_PATH)])
    con.close()
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
