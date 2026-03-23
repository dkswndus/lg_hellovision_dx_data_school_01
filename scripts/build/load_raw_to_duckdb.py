"""
raw CSV → DuckDB 적재

- *_VOD.csv       → vod_log
- vod_mart*.csv   → vod_content
- *cancel*.csv    → user_profile
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE / "data" / "raw"
DB_PATH = BASE / "data" / "lghellovision.duckdb"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    # 1. VOD로 끝나는 파일들 → vod_log
    vod_files = sorted(RAW_DIR.glob("*VOD.csv"))
    if vod_files:
        pattern = str(RAW_DIR / "*VOD.csv").replace("\\", "/")
        con.execute(
            """
            CREATE OR REPLACE TABLE vod_log AS
            SELECT * FROM read_csv_auto(?, union_by_name=True, ignore_errors=True)
            """,
            [pattern],
        )
        print(f"vod_log: {len(vod_files)}개 VOD 파일 적재 완료")
    else:
        print("vod_log: VOD 파일 없음")

    # 2. vod_mart → vod_content
    vod_mart_files = list(RAW_DIR.glob("*vod_mart*.csv"))
    if vod_mart_files:
        path = str(vod_mart_files[0])
        con.execute(
            "CREATE OR REPLACE TABLE vod_content AS SELECT * FROM read_csv_auto(?, ignore_errors=True)",
            [path],
        )
        print(f"vod_content: {path} 적재 완료")
    else:
        print("vod_content: vod_mart 파일 없음")

    # 3. cancel 포함 파일들 → user_profile
    cancel_files = sorted(RAW_DIR.glob("*cancel*.csv"))
    if cancel_files:
        pattern = str(RAW_DIR / "*cancel*.csv").replace("\\", "/")
        con.execute(
            """
            CREATE OR REPLACE TABLE user_profile AS
            SELECT * FROM read_csv_auto(?, union_by_name=True, ignore_errors=True)
            """,
            [pattern],
        )
        print(f"user_profile: {len(cancel_files)}개 cancel 파일 적재 완료")
    else:
        print("user_profile: cancel 파일 없음")

    con.close()
    print(f"DuckDB 저장: {DB_PATH}")


if __name__ == "__main__":
    main()
