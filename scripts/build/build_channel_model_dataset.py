"""
채널 시청 예측용 데이터셋 생성
- 종속변수: CH_HH_AVG_MONTH1
- 독립변수: log_VOD_TIME, Adult, FOD, Trend, INHOME_RATE, AGE_GRP10

변수 정의:
  log_VOD_TIME : ln(1 + 전체 VOD 시청 시간)
  Adult        : 성인 콘텐츠 시청 비율 (성인 시청 건수 / 전체 시청 건수)
  FOD          : FOD(무료) 시청 시간 비율 (fod_tms / total_tms)
  Trend        : 갤럽 Top5 콘텐츠 시청 비율 (trend_cnt / total_cnt)
  INHOME_RATE  : 재택률
  AGE_GRP10    : 연령대
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
INTERIM = BASE / "data" / "interim"
PROCESSED = BASE / "data" / "processed"
CHURN_PATH = PROCESSED / "churn_dataset.parquet"
VOD_LOG = INTERIM / "vod_log_clean.parquet"
VOD_CONTENT = INTERIM / "vod_content_clean.parquet"
OUTPUT_PATH = PROCESSED / "channel_model_dataset.parquet"
SAMPLE_N = 100_000
RANDOM_STATE = 42

# 한국 갤럽 2023년 월별 인기 Top5 키워드 (공백 제거 후 LIKE 매칭)
GALLUP_KEYWORDS: dict[int, list[str]] = {
    202301: ["글로리", "미스터트롯2", "유퀴즈", "나혼자산다", "콩깍지"],
    202302: ["미스터트롯2", "글로리", "콩깍지", "불타는트롯맨", "일타스캔들"],
    202303: ["글로리", "미스터트롯2", "삼남매가용감", "불타는트롯맨", "콩깍지"],
    202304: ["모범택시2", "미스터트롯2", "글로리", "서진이네", "진짜가나타났다"],
    202305: ["차정숙", "미스터트롯2", "낭만닥터김사부3", "금이야옥이야", "비밀의여자"],
    202306: ["지구오락실2", "낭만닥터김사부3", "나쁜엄마", "차정숙", "나혼자산다"],
    202307: ["진짜가나타났다", "지구오락실2", "킹더랜드", "악귀", "나혼자산다"],
    202308: ["진짜가나타났다", "나혼자산다", "금이야옥이야", "유퀴즈", "연인"],
    202309: ["무빙", "나는SOLO", "유퀴즈", "나혼자산다", "런닝맨"],
    202310: ["연인드라마", "나혼자산다", "유퀴즈", "나는SOLO", "미운우리새끼"],
    202311: ["연인드라마", "나혼자산다", "자연인이다", "더라이브", "강남순"],
    202312: ["나혼자산다", "유퀴즈", "거란전쟁", "자연인이다", "미운우리새끼"],
}

# 성인 콘텐츠 컬럼 후보 (vod_content)
ADULT_COL_CANDIDATES = ["adult_yn", "adult_fl", "is_adult", "adult_rate_yn", "adult"]


def detect_adult_col(con: duckdb.DuckDBPyConnection) -> str | None:
    cols = [r[0] for r in con.execute("DESCRIBE vod_content_tbl").fetchall()]
    for candidate in ADULT_COL_CANDIDATES:
        if candidate in cols:
            print(f"  성인 컬럼 감지: {candidate}")
            return candidate
    print(f"  ⚠️  성인 컬럼 없음 — Adult=0 고정 (후보: {ADULT_COL_CANDIDATES})")
    print(f"     vod_content 전체 컬럼: {cols}")
    return None


def build_trend_asset_table(con: duckdb.DuckDBPyConnection) -> int:
    """월별 갤럽 Top5 (p_mt, full_asset_id) 매핑 테이블 생성"""
    rows = []
    for p_mt, keywords in GALLUP_KEYWORDS.items():
        conditions = " OR ".join(
            [f"REPLACE(asset_nm, ' ', '') LIKE '%{kw}%'" for kw in keywords]
        )
        result = con.execute(
            f"SELECT full_asset_id FROM vod_content_tbl WHERE {conditions}"
        ).fetchdf()
        for asset_id in result["full_asset_id"]:
            rows.append({"p_mt": p_mt, "full_asset_id": asset_id})

    if not rows:
        print("  ⚠️  갤럽 매칭 0건 — Trend=0 고정")
        con.execute(
            "CREATE OR REPLACE TEMP TABLE trend_assets AS "
            "SELECT 0::INT AS p_mt, ''::VARCHAR AS full_asset_id WHERE 1=0"
        )
        return 0

    trend_df = pd.DataFrame(rows).drop_duplicates()
    con.register("trend_df_ch", trend_df)
    con.execute("CREATE OR REPLACE TEMP TABLE trend_assets AS SELECT * FROM trend_df_ch")
    print(f"  갤럽 Top5 매칭: {len(trend_df)}건")
    return len(trend_df)


def main():
    con = duckdb.connect(":memory:")

    # 1. 기준 사용자 (churn_dataset 2023년)
    print("churn_dataset 로드 중...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT sha2_hash, p_mt,
               CH_HH_AVG_MONTH1 AS y,
               INHOME_RATE, AGE_GRP10, vod_use_tms_sum
        FROM read_parquet(?)
        WHERE p_mt >= 202301 AND p_mt <= 202312
          AND CH_HH_AVG_MONTH1 IS NOT NULL
        """,
        [str(CHURN_PATH)],
    )
    n_base = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
    if n_base == 0:
        raise ValueError("churn_dataset에 2023 p_mt 없음")
    print(f"  기준 행: {n_base:,}개")

    # 샘플링
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE base_sample AS
        SELECT * FROM base
        USING SAMPLE {SAMPLE_N} ROWS (reservoir, {RANDOM_STATE})
        """
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE base_users AS SELECT DISTINCT sha2_hash FROM base_sample"
    )

    # 2. vod_content 로드 및 adult 컬럼 감지
    print("\nvod_content 로드 중...")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE vod_content_tbl AS SELECT * FROM read_parquet(?)",
        [str(VOD_CONTENT)],
    )
    adult_col = detect_adult_col(con)

    # 3. 갤럽 Top5 매핑 테이블
    print("\n갤럽 Top5 매핑 중...")
    build_trend_asset_table(con)

    # 4. vod_log 집계: FOD 시청 시간 + Adult 시청 건수 + Trend 시청 건수 + 전체 시청
    print("\nvod_log 집계 중...")

    adult_expr = (
        f"COUNT(*) FILTER (WHERE vc.{adult_col} = TRUE OR vc.{adult_col} = 1)"
        if adult_col else "0"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE vod_agg AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)     AS p_mt,
            COALESCE(SUM(CASE WHEN vc.asset_prod = 0 THEN vl.use_tms ELSE 0 END), 0)
                                                                         AS fod_tms,
            COALESCE(SUM(vl.use_tms), 0)                                AS total_tms,
            COUNT(*)                                                      AS total_cnt,
            {adult_expr}                                                  AS adult_cnt
        FROM read_parquet(?) vl
        INNER JOIN base_users u ON vl.sha2_hash = u.sha2_hash
        LEFT JOIN vod_content_tbl vc ON vl.asset = vc.full_asset_id
        WHERE vl.strt_dt IS NOT NULL
          AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) BETWEEN 202301 AND 202312
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG)],
    )

    # 5. Trend 집계
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE trend_agg AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COUNT(*) AS trend_cnt
        FROM read_parquet(?) vl
        INNER JOIN base_users u ON vl.sha2_hash = u.sha2_hash
        INNER JOIN trend_assets ta
            ON vl.asset = ta.full_asset_id
            AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) = ta.p_mt
        WHERE vl.strt_dt IS NOT NULL
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG)],
    )

    # 6. base_sample + 집계 결과 병합 → 파생변수 계산
    print("파생변수 계산 중...")
    final = con.execute(
        """
        SELECT
            b.sha2_hash,
            b.p_mt,
            b.y,
            LN(1 + COALESCE(b.vod_use_tms_sum, 0))                              AS log_VOD_TIME,
            CASE WHEN COALESCE(a.total_cnt, 0) = 0 THEN 0.0
                 ELSE a.adult_cnt::DOUBLE / a.total_cnt END                       AS Adult,
            CASE WHEN COALESCE(a.total_tms, 0) = 0 THEN 0.0
                 ELSE a.fod_tms::DOUBLE / a.total_tms END                         AS FOD,
            CASE WHEN COALESCE(a.total_cnt, 0) = 0 THEN 0.0
                 ELSE COALESCE(t.trend_cnt, 0)::DOUBLE / a.total_cnt END          AS Trend,
            TRY_CAST(b.INHOME_RATE AS DOUBLE)                                     AS INHOME_RATE,
            TRY_CAST(b.AGE_GRP10  AS DOUBLE)                                      AS AGE_GRP10
        FROM base_sample b
        LEFT JOIN vod_agg   a ON b.sha2_hash = a.sha2_hash AND b.p_mt = a.p_mt
        LEFT JOIN trend_agg t ON b.sha2_hash = t.sha2_hash AND b.p_mt = t.p_mt
        WHERE b.y IS NOT NULL
        """
    ).fetchdf()

    # 결측 채우기
    final["INHOME_RATE"] = final["INHOME_RATE"].fillna(0.0)
    final["AGE_GRP10"] = final["AGE_GRP10"].fillna(5.0)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)

    # 변수 분포 요약
    print(f"\n저장: {OUTPUT_PATH} ({len(final):,}행)")
    for col in ["log_VOD_TIME", "Adult", "FOD", "Trend"]:
        nonzero = (final[col] > 0).mean()
        print(f"  {col}: mean={final[col].mean():.4f}, nonzero={nonzero:.1%}")
    con.close()
    print("완료.")


if __name__ == "__main__":
    main()
