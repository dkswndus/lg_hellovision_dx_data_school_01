"""
raw 데이터를 DuckDB 테이블로 적재하는 스크립트

- TPS_cancel_*_p_mt.csv -> user_profile
- vod_mart_data.csv -> vod_content
- *_VOD.csv (10개 파일) -> vod_log
"""
import duckdb
import argparse
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "lghellovision.duckdb"
INTERIM_DIR = BASE_DIR / "data" / "interim"

# raw CSV glob 패턴 (경로 호환: as_posix로 / 사용)
TPS_GLOB = (RAW_DIR / "TPS_cancel_*_p_mt.csv").as_posix()
VOD_MART_CSV = (RAW_DIR / "vod_mart_data.csv").as_posix()
VOD_LOG_GLOB = (RAW_DIR / "*_VOD.csv").as_posix()


def _drop_if_exists(con: duckdb.DuckDBPyConnection, table: str) -> None:
    con.execute(f'DROP TABLE IF EXISTS "{table}"')


def create_user_profile(con: duckdb.DuckDBPyConnection, *, drop: bool) -> int:
    if drop:
        _drop_if_exists(con, "user_profile")
    else:
        # If not dropping and table exists, keep it as-is.
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='user_profile'"
        ).fetchone()[0]
        if exists:
            return con.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0]

    con.execute(
        f"""
        CREATE TABLE user_profile AS
        SELECT *
        FROM read_csv_auto(
            '{TPS_GLOB}',
            header=true,
            auto_detect=true,
            all_varchar=true
        )
        """
    )
    return con.execute("SELECT COUNT(*) FROM user_profile").fetchone()[0]


def create_vod_content(con: duckdb.DuckDBPyConnection, *, drop: bool) -> int:
    if drop:
        _drop_if_exists(con, "vod_content")
    else:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='vod_content'"
        ).fetchone()[0]
        if exists:
            return con.execute("SELECT COUNT(*) FROM vod_content").fetchone()[0]

    con.execute(
        f"""
        CREATE TABLE vod_content AS
        SELECT *
        FROM read_csv_auto(
            '{VOD_MART_CSV}',
            header=true,
            auto_detect=true,
            all_varchar=true
        )
        """
    )
    return con.execute("SELECT COUNT(*) FROM vod_content").fetchone()[0]


def create_vod_log(con: duckdb.DuckDBPyConnection, *, drop: bool) -> int:
    if drop:
        _drop_if_exists(con, "vod_log")
    else:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='vod_log'"
        ).fetchone()[0]
        if exists:
            return con.execute("SELECT COUNT(*) FROM vod_log").fetchone()[0]

    con.execute(
        f"""
        CREATE TABLE vod_log AS
        SELECT *
        FROM read_csv_auto(
            '{VOD_LOG_GLOB}',
            header=true,
            auto_detect=true,
            all_varchar=true,
            strict_mode=false,
            ignore_errors=true,
            null_padding=true
        )
        """
    )
    return con.execute("SELECT COUNT(*) FROM vod_log").fetchone()[0]


def export_table_to_parquet(
    con: duckdb.DuckDBPyConnection, *, table: str, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (SELECT * FROM "{table}")
        TO '{out_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["user_profile", "vod_content", "vod_log", "all"],
        default="all",
        help="생성할 테이블 선택",
    )
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="기존 테이블이 있어도 DROP 하지 않음(기본은 DROP 후 재생성)",
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="data/interim에 Parquet로 내보내기(user_profile/vod_content/vod_log)",
    )
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))
    drop = not args.no_drop

    try:
        if args.only in ("user_profile", "all"):
            cnt = create_user_profile(con, drop=drop)
            print(f"user_profile 테이블 적재 완료: {cnt:,}건")

        if args.only in ("vod_content", "all"):
            cnt = create_vod_content(con, drop=drop)
            print(f"vod_content 테이블 적재 완료: {cnt:,}건")

        if args.only in ("vod_log", "all"):
            cnt = create_vod_log(con, drop=drop)
            print(f"vod_log 테이블 적재 완료: {cnt:,}건")
    finally:
        con.close()

    print(f"\nDB 저장 위치: {DB_PATH}")

    if args.export_parquet:
        con = duckdb.connect(str(DB_PATH))
        try:
            export_table_to_parquet(con, table="user_profile", out_path=INTERIM_DIR / "user_profile.parquet")
            export_table_to_parquet(con, table="vod_content", out_path=INTERIM_DIR / "vod_content.parquet")
            export_table_to_parquet(con, table="vod_log", out_path=INTERIM_DIR / "vod_log.parquet")
        finally:
            con.close()
        print(f"Parquet 저장 완료: {INTERIM_DIR}")


if __name__ == "__main__":
    main()
