"""
churn_dataset_v3 생성 — 갤럽 Top5 트렌드 파생변수 추가

한국 갤럽 2023년 월별 인기 베스트 컨텐츠 Top5 기반:
- Trend_Watch_Cnt   : 해당 월 갤럽 Top5 콘텐츠 시청 건수
- Trend_Watch_Ratio : 갤럽 Top5 시청 건수 / 전체 VOD 시청 건수
- Trend_Flag        : 갤럽 Top5 콘텐츠 1건 이상 시청 여부 (0/1)
"""
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
INTERIM = BASE / "data" / "interim"
PROCESSED = BASE / "data" / "processed"
VOD_LOG = INTERIM / "vod_log_clean.parquet"
VOD_CONTENT = INTERIM / "vod_content_clean.parquet"
BASE_DATASET = PROCESSED / "churn_dataset_v2.parquet"
OUTPUT_PATH = PROCESSED / "churn_dataset_v3.parquet"

# 한국 갤럽 2023년 월별 인기 Top5 (공백 제거 후 LIKE 매칭)
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

# plot_clean_data.py SKIP_COLS에서 확인: asset_nm = 콘텐츠명 컬럼
TITLE_COL_CANDIDATES = ["asset_nm", "prog_nm", "title", "content_nm", "vod_ttl", "ep_ttl"]


def detect_title_col(con: duckdb.DuckDBPyConnection) -> str:
    cols = [r[0] for r in con.execute("DESCRIBE vod_content_tbl").fetchall()]
    print(f"vod_content 컬럼 ({len(cols)}개): {cols}")
    for candidate in TITLE_COL_CANDIDATES:
        if candidate in cols:
            print(f"제목 컬럼 사용: {candidate}")
            return candidate
    raise ValueError(
        f"제목 컬럼을 찾지 못했습니다. 실제 컬럼: {cols}"
    )


def build_trend_asset_table(con: duckdb.DuckDBPyConnection, title_col: str) -> int:
    """월별 갤럽 Top5에 해당하는 (p_mt, full_asset_id) 매핑 테이블 생성"""
    rows = []
    for p_mt, keywords in GALLUP_KEYWORDS.items():
        # 공백 제거 후 LIKE 매칭
        conditions = " OR ".join(
            [f"REPLACE({title_col}, ' ', '') LIKE '%{kw}%'" for kw in keywords]
        )
        result = con.execute(
            f"SELECT full_asset_id FROM vod_content_tbl WHERE {conditions}"
        ).fetchdf()
        for asset_id in result["full_asset_id"]:
            rows.append({"p_mt": p_mt, "full_asset_id": asset_id})

    if not rows:
        print("⚠️  갤럽 Top5 매칭 결과 없음 — 제목 컬럼 또는 키워드를 확인하세요.")
        con.execute(
            "CREATE OR REPLACE TEMP TABLE trend_assets AS "
            "SELECT 0::INT AS p_mt, ''::VARCHAR AS full_asset_id WHERE 1=0"
        )
        return 0

    trend_df = pd.DataFrame(rows).drop_duplicates()
    con.register("trend_df", trend_df)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE trend_assets AS SELECT * FROM trend_df"
    )
    n = len(trend_df)
    print(f"갤럽 Top5 매칭 콘텐츠: {n}개 (월별 중복 포함)")
    return n


def main():
    if not BASE_DATASET.exists():
        raise FileNotFoundError(
            f"{BASE_DATASET} 없음. build_churn_dataset_v2.py 먼저 실행하세요."
        )

    con = duckdb.connect(":memory:")

    # vod_content 로드 및 제목 컬럼 감지
    print("vod_content 로드 중...")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE vod_content_tbl AS SELECT * FROM read_parquet(?)",
        [str(VOD_CONTENT)],
    )
    title_col = detect_title_col(con)

    # asset_nm 샘플 출력 (키워드 매칭 검증용)
    sample_titles = con.execute(
        f"SELECT DISTINCT asset_nm FROM vod_content_tbl "
        f"WHERE asset_nm IS NOT NULL ORDER BY RANDOM() LIMIT 20"
    ).fetchdf()
    print(f"\nasset_nm 샘플 20개:\n{sample_titles['asset_nm'].tolist()}\n")

    # 갤럽 Top5 매핑 테이블 생성
    print("갤럽 Top5 콘텐츠 ID 매핑 중...")
    n_trend = build_trend_asset_table(con, title_col)

    # 대상 사용자 (churn_dataset_v2 기준)
    print("churn_dataset_v2 로드 중...")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE base AS SELECT * FROM read_parquet(?)",
        [str(BASE_DATASET)],
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE base_users AS SELECT DISTINCT sha2_hash FROM base"
    )

    # 사용자별·월별 전체 VOD 시청 건수 집계
    print("vod_log 집계 중 (전체 시청 + 트렌드 시청)...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE monthly_total AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COUNT(*) AS total_vod_cnt
        FROM read_parquet(?) vl
        INNER JOIN base_users u ON vl.sha2_hash = u.sha2_hash
        WHERE vl.strt_dt IS NOT NULL
          AND CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) BETWEEN 202301 AND 202312
        GROUP BY vl.sha2_hash, CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT)
        """,
        [str(VOD_LOG)],
    )

    # 사용자별·월별 갤럽 Top5 콘텐츠 시청 건수 집계
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE monthly_trend AS
        SELECT
            vl.sha2_hash,
            CAST(SUBSTR(CAST(vl.strt_dt AS VARCHAR), 1, 6) AS INT) AS p_mt,
            COUNT(*) AS trend_watch_cnt
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

    # base에 트렌드 변수 합산
    print("파생변수 합산 중...")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE final AS
        SELECT
            b.*,
            COALESCE(mt.trend_watch_cnt, 0)::BIGINT        AS Trend_Watch_Cnt,
            CASE
                WHEN COALESCE(tot.total_vod_cnt, 0) = 0 THEN 0.0
                ELSE COALESCE(mt.trend_watch_cnt, 0)::DOUBLE / tot.total_vod_cnt
            END                                             AS Trend_Watch_Ratio,
            CASE WHEN COALESCE(mt.trend_watch_cnt, 0) > 0
                 THEN 1 ELSE 0 END::INTEGER                 AS Trend_Flag
        FROM base b
        LEFT JOIN monthly_trend mt  ON b.sha2_hash = mt.sha2_hash AND b.p_mt = mt.p_mt
        LEFT JOIN monthly_total tot ON b.sha2_hash = tot.sha2_hash AND b.p_mt = tot.p_mt
        """
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    con.execute("COPY final TO ? (FORMAT PARQUET)", [str(OUTPUT_PATH)])
    n_final = con.execute("SELECT COUNT(*) FROM final").fetchone()[0]

    # 트렌드 변수 분포 요약
    summary = con.execute(
        """
        SELECT
            AVG(Trend_Watch_Cnt)   AS avg_trend_cnt,
            AVG(Trend_Watch_Ratio) AS avg_trend_ratio,
            AVG(Trend_Flag)        AS trend_flag_rate,
            COUNT(*) FILTER (WHERE Trend_Watch_Cnt > 0) AS users_with_trend
        FROM final
        """
    ).fetchdf()
    print(f"\n저장: {OUTPUT_PATH} ({n_final:,}행)")
    print(f"  트렌드 시청 사용자 비율: {summary['trend_flag_rate'].iloc[0]:.1%}")
    print(f"  평균 트렌드 시청 건수:   {summary['avg_trend_cnt'].iloc[0]:.2f}")
    print(f"  평균 트렌드 시청 비율:   {summary['avg_trend_ratio'].iloc[0]:.4f}")
    if n_trend == 0:
        print("\n⚠️  트렌드 매칭 0건 — TITLE_COL_CANDIDATES 또는 GALLUP_KEYWORDS 키워드를 조정하세요.")
    print("완료.")
    con.close()


if __name__ == "__main__":
    main()
