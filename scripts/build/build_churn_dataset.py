"""
해지 예측용 최종 데이터셋 생성

- 기준: user_profile_clean.parquet
- vod_log 집계(시청 횟수, 시청 시간 합계 등) → sha2_hash 기준 JOIN
- 샘플링: 전체의 5%, 유지:해지 = 80:20
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent.parent
INTERIM_DIR = BASE / "data" / "interim"
OUTPUT_DIR = BASE / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USER_PROFILE = INTERIM_DIR / "user_profile_clean.parquet"
VOD_LOG = INTERIM_DIR / "vod_log_clean.parquet"
OUTPUT_FILE = OUTPUT_DIR / "churn_dataset.parquet"

SAMPLE_PCT = 0.05  # 전체의 5%
MAINTAIN_RATIO = 0.80  # 유지 80%
CHURN_RATIO = 0.20  # 해지 20%
SEED = 42  # 재현성을 위한 시드
# cancel_yn: 유지=0, 해지=1


def main():
    con = duckdb.connect(":memory:")

    # 1. user_profile 로드 후 유지/해지 분리
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE up AS
        SELECT * FROM read_parquet(?)
        """,
        [str(USER_PROFILE)],
    )

    # 유지/해지 건수 확인
    counts = con.execute(
        """
        SELECT cancel_yn, COUNT(*) AS cnt
        FROM up
        GROUP BY cancel_yn
        ORDER BY cancel_yn
        """
    ).fetchall()

    total = sum(c[1] for c in counts)
    maintain_total = next((c[1] for c in counts if c[0] == 0), 0)
    churn_total = next((c[1] for c in counts if c[0] == 1), 0)

    print(f"user_profile: 전체 {total:,} (유지 {maintain_total:,}, 해지 {churn_total:,})")

    # 3. 5% 샘플, 유지:해지 = 80:20 (해시 기반 - ORDER BY random 대비 빠름)
    target_total = int(total * SAMPLE_PCT)
    target_maintain = int(target_total * MAINTAIN_RATIO)
    target_churn = int(target_total * CHURN_RATIO)

    maintain_pct = target_maintain / maintain_total if maintain_total else 0
    churn_pct = target_churn / churn_total if churn_total else 0
    # 해시 기반: MOD(ABS(hash(...)), 10000) < pct*10000 이면 샘플에 포함
    maintain_thresh = int(maintain_pct * 10000)
    churn_thresh = int(churn_pct * 10000)

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE sampled AS
        SELECT * FROM up
        WHERE (cancel_yn = 0 AND MOD(ABS(hash(sha2_hash || ?)), 10000) < ?)
           OR (cancel_yn = 1 AND MOD(ABS(hash(sha2_hash || ?)), 10000) < ?)
        """,
        [str(SEED), maintain_thresh, str(SEED), churn_thresh],
    )

    sampled_counts = con.execute(
        """
        SELECT cancel_yn, COUNT(*) AS cnt
        FROM sampled
        GROUP BY cancel_yn
        ORDER BY cancel_yn
        """
    ).fetchall()
    print(
        f"샘플: 전체 {sum(c[1] for c in sampled_counts):,} "
        f"(유지 {next((c[1] for c in sampled_counts if c[0] == 0), 0):,}, "
        f"해지 {next((c[1] for c in sampled_counts if c[0] == 1), 0):,})"
    )

    # 4. vod_log 집계 (샘플된 sha2_hash만 대상으로 - 효율화)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE vod_agg AS
        SELECT
            vl.sha2_hash,
            COUNT(*) AS vod_view_cnt,
            COALESCE(SUM(vl.use_tms), 0)::BIGINT AS vod_use_tms_sum,
            COUNT(DISTINCT vl.asset) AS vod_asset_cnt
        FROM read_parquet(?) vl
        INNER JOIN sampled s ON vl.sha2_hash = s.sha2_hash
        GROUP BY vl.sha2_hash
        """,
        [str(VOD_LOG)],
    )
    print("vod_log 집계 완료 (샘플 대상만)")

    # 6. vod_agg LEFT JOIN (vod_log 없는 고객은 0으로)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE final AS
        SELECT
            s.*,
            COALESCE(v.vod_view_cnt, 0)::BIGINT AS vod_view_cnt,
            COALESCE(v.vod_use_tms_sum, 0)::BIGINT AS vod_use_tms_sum,
            COALESCE(v.vod_asset_cnt, 0)::BIGINT AS vod_asset_cnt
        FROM sampled s
        LEFT JOIN vod_agg v ON s.sha2_hash = v.sha2_hash
        """
    )

    # 7. 저장
    con.execute(
        "COPY final TO ? (FORMAT PARQUET)",
        [str(OUTPUT_FILE)],
    )
    print(f"저장: {OUTPUT_FILE}")

    # 요약
    final_cnt = con.execute("SELECT COUNT(*) FROM final").fetchone()[0]
    print(f"최종 행 수: {final_cnt:,}")

    con.close()
    print("완료.")


if __name__ == "__main__":
    main()
