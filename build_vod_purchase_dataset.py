"""
VOD 구매 건수 예측용 데이터셋 생성 (2차 모델)
- 종속변수: vod_purchase_cnt (RVOD 시청 건수 = 유료 VOD 구매/렌탈 건수)
- RVOD = asset_prod=1 (vod_content)
- 추가 변수: FOD/RVOD/SVOD 타입별 건수 (fod_cnt, rvod_cnt, svod_cnt), 전월 RVOD(lagged_rvod_cnt)
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).parent
INTERIM = BASE / "data" / "interim"
PROCESSED = BASE / "data" / "processed"
CHURN_PATH = PROCESSED / "churn_dataset.parquet"
VOD_LOG = INTERIM / "vod_log_clean.parquet"
VOD_CONTENT = INTERIM / "vod_content_clean.parquet"
OUTPUT_PATH = PROCESSED / "vod_purchase_dataset.parquet"


def main():
    con = duckdb.connect(":memory:")

    # 1. churn_dataset에서 (sha2_hash, p_mt) + user_profile 컬럼 로드
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT * FROM read_parquet(?)
        WHERE p_mt >= 202301 AND p_mt <= 202312
        """,
        [str(CHURN_PATH)],
    )
    n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
    print(f"base (churn_dataset 2023): {n_base:,}행")

    # 2. vod_log + vod_content 조인 → FOD/RVOD/SVOD 건수 월별 집계 (asset_prod: 0=FOD, 1=RVOD, 2=SVOD)
    print("vod_log + vod_content 조인, FOD/RVOD/SVOD 건수 월별 집계 중...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE base_users AS
        SELECT DISTINCT sha2_hash FROM base
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE monthly_by_type AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COUNT(*) FILTER (WHERE vc.asset_prod = 0) AS fod_cnt,
            COUNT(*) FILTER (WHERE vc.asset_prod = 1) AS rvod_cnt,
            COUNT(*) FILTER (WHERE vc.asset_prod = 2) AS svod_cnt
        FROM read_parquet(?) vl
        INNER JOIN base_users u ON vl.sha2_hash = u.sha2_hash
        LEFT JOIN read_parquet(?) vc ON vl.asset = vc.full_asset_id
        WHERE vl.strt_dt IS NOT NULL
          AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) BETWEEN 202301 AND 202312
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG), str(VOD_CONTENT)],
    )
    n_monthly = con.execute("SELECT COUNT(*) FROM monthly_by_type").fetchone()[0]
    print(f"  FOD/RVOD/SVOD 월별 집계: {n_monthly:,}행")

    # 3. base + monthly_by_type 조인, vod_purchase_cnt = rvod_cnt
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE with_type AS
        SELECT
            b.*,
            COALESCE(m.fod_cnt, 0)::BIGINT AS fod_cnt,
            COALESCE(m.rvod_cnt, 0)::BIGINT AS rvod_cnt,
            COALESCE(m.svod_cnt, 0)::BIGINT AS svod_cnt
        FROM base b
        LEFT JOIN monthly_by_type m ON b.sha2_hash = m.sha2_hash AND b.p_mt = m.p_mt
        """
    )

    # 4. 전월 RVOD (lagged_rvod_cnt) - "돈 써본 사람이 또 쓴다" 검증용
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE with_lag AS
        SELECT
            w.*,
            w.rvod_cnt AS vod_purchase_cnt,
            COALESCE(lag.rvod_cnt, 0)::BIGINT AS lagged_rvod_cnt
        FROM with_type w
        LEFT JOIN monthly_by_type lag
          ON w.sha2_hash = lag.sha2_hash
          AND lag.p_mt = CASE
              WHEN w.p_mt % 100 = 1 THEN w.p_mt - 89
              ELSE w.p_mt - 1
          END
        """
    )

    # 5. 최종 (vod_view_cnt 등과 중복 가능한 컬럼 정리 - base에 이미 있음)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE final AS
        SELECT * FROM with_lag
        """
    )

    # 4. 저장
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con.execute("COPY final TO ? (FORMAT PARQUET)", [str(OUTPUT_PATH)])

    # 요약
    stats = con.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN vod_purchase_cnt > 0 THEN 1 ELSE 0 END) AS has_purchase,
            AVG(vod_purchase_cnt) AS avg_purchase,
            MAX(vod_purchase_cnt) AS max_purchase
        FROM final
        """
    ).fetchone()
    print(f"총 행: {stats[0]:,}, 구매 1건 이상: {stats[1]:,} ({100*stats[1]/stats[0]:.1f}%)")
    print(f"평균 구매 건수: {stats[2]:.2f}, 최대: {stats[3]:,}")
    print(f"저장: {OUTPUT_PATH}")
    con.close()
    print("완료.")


if __name__ == "__main__":
    main()
