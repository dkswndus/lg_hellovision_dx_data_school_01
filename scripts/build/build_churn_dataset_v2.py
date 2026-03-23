"""
해지 예측용 확장 데이터셋 생성 (churn_dataset_v2)
- 기존 churn_dataset 기반
- 추가 변수 7개:
  ① Flag_Kids: 키즈 콘텐츠 시청 여부
  ② Flag_Movie: 영화 콘텐츠 시청 여부
  ③ Recency: 최신성 = 마지막 시청 경과일
  ④ Total_Watch_Time: 총 시청 시간 (VOD)
  ⑤ Log_Watch_Time: 로그 변환된 총 시청 시간
  ⑥ Is_Netflix: 넷플릭스 이용 여부 플래그
  ⑦ Risk_Segment_Kids_NFX: 위험군 세그먼트 (넷플릭스 × 키즈)
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent.parent
INTERIM_DIR = BASE / "data" / "interim"
PROCESSED_DIR = BASE / "data" / "processed"
CHURN_PATH = PROCESSED_DIR / "churn_dataset.parquet"
VOD_LOG_PATH = INTERIM_DIR / "vod_log_clean.parquet"
OUTPUT_FILE = PROCESSED_DIR / "churn_dataset_v2.parquet"

# 기준일: 관찰 기간 마지막 (Recency 계산용)
REFERENCE_DATE = "2023-12-31"
NO_VOD_RECENCY = 999  # VOD 미시청 시 부여값


def main():
    con = duckdb.connect(":memory:")

    # 1. churn_dataset 기준
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE churn AS
        SELECT * FROM read_parquet(?)
        """,
        [str(CHURN_PATH)],
    )
    n_churn = con.execute("SELECT COUNT(*) FROM churn").fetchone()[0]
    print(f"churn_dataset: {n_churn:,}행")

    # 2. vod_log에서 Flag_Kids, Flag_Movie, Recency, Total_Watch_Time 집계
    # CT_CL: 14=키즈, 12=영화
    # strt_dt: YYYYMMDDHHMMSS → 날짜 파싱
    print("vod_log 집계 중 (Flag_Kids, Flag_Movie, Recency, Total_Watch_Time)...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE vod_agg AS
        SELECT
            vl.sha2_hash,
            MAX(CASE WHEN vl.CT_CL = 14 THEN 1 ELSE 0 END) AS Flag_Kids,
            MAX(CASE WHEN vl.CT_CL = 12 THEN 1 ELSE 0 END) AS Flag_Movie,
            COALESCE(SUM(vl.use_tms), 0)::BIGINT AS Total_Watch_Time,
            CAST(DATEDIFF('day',
                MAX(STRPTIME(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 8), '%Y%m%d')),
                CAST(? AS DATE)
            ) AS INT) AS recency_raw
        FROM read_parquet(?) vl
        INNER JOIN churn c ON vl.sha2_hash = c.sha2_hash
        WHERE vl.strt_dt IS NOT NULL
        GROUP BY vl.sha2_hash
        """,
        [REFERENCE_DATE, str(VOD_LOG_PATH)],
    )

    # 3. Recency: VOD 미시청자는 NO_VOD_RECENCY
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE vod_with_recency AS
        SELECT
            sha2_hash,
            Flag_Kids,
            Flag_Movie,
            Total_Watch_Time,
            CASE WHEN recency_raw < 0 THEN 0 ELSE recency_raw END AS Recency
        FROM vod_agg
        """
    )

    # 4. churn + vod_agg 조인, 파생변수
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE with_vod AS
        SELECT
            c.*,
            COALESCE(v.Flag_Kids, 0) AS Flag_Kids,
            COALESCE(v.Flag_Movie, 0) AS Flag_Movie,
            COALESCE(v.Recency, ?) AS Recency,
            COALESCE(v.Total_Watch_Time, c.vod_use_tms_sum) AS Total_Watch_Time
        FROM churn c
        LEFT JOIN vod_with_recency v ON c.sha2_hash = v.sha2_hash
        """,
        [NO_VOD_RECENCY],
    )

    # 5. Log_Watch_Time, Is_Netflix, Risk_Segment_Kids_NFX
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE final AS
        SELECT
            *,
            LN(1 + Total_Watch_Time) AS Log_Watch_Time,
            CASE WHEN NFX_USE_YN THEN 1 ELSE 0 END AS Is_Netflix,
            CASE WHEN COALESCE(Flag_Kids, 0) = 1 AND NFX_USE_YN THEN 1 ELSE 0 END AS Risk_Segment_Kids_NFX
        FROM with_vod
        """
    )

    # 6. 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY final TO ? (FORMAT PARQUET)",
        [str(OUTPUT_FILE)],
    )

    # 요약
    n_final = con.execute("SELECT COUNT(*) FROM final").fetchone()[0]
    flag_kids = con.execute("SELECT COUNT(*) FROM final WHERE Flag_Kids = 1").fetchone()[0]
    flag_movie = con.execute("SELECT COUNT(*) FROM final WHERE Flag_Movie = 1").fetchone()[0]
    risk_seg = con.execute("SELECT COUNT(*) FROM final WHERE Risk_Segment_Kids_NFX = 1").fetchone()[0]
    print(f"Flag_Kids=1: {flag_kids:,} ({100*flag_kids/n_final:.1f}%)")
    print(f"Flag_Movie=1: {flag_movie:,} ({100*flag_movie/n_final:.1f}%)")
    print(f"Risk_Segment_Kids_NFX=1: {risk_seg:,} ({100*risk_seg/n_final:.1f}%)")
    print(f"저장: {OUTPUT_FILE}")
    con.close()
    print("완료.")


if __name__ == "__main__":
    main()
