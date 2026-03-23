"""
DuckDB 테이블별 스키마·고유값·결측률·샘플(20행)을 JSON으로 data 폴더에 저장
"""
import json
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE / "data" / "lghellovision.duckdb"
OUT_DIR = BASE / "data"
TABLES = ["vod_log", "vod_content", "user_profile"]
UNIQUE_DISPLAY_LIMIT = 20  # 고유값 20개씩만 표시 (초과 시 샘플)
SAMPLE_ROWS = 20


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)

    for table in TABLES:
        info = con.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [r[1] for r in info]
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        schema_list = []
        for col in col_names:
            # 결측 수, 고유값 개수 (한 번에)
            r = con.execute(f"""
                SELECT
                    COUNT(*) - COUNT("{col}") AS null_cnt,
                    COUNT(DISTINCT "{col}") AS unique_cnt
                FROM {table}
            """).fetchone()
            null_cnt, unique_cnt = r[0], r[1]
            missing_rate = round(null_cnt / row_count * 100, 2) if row_count else 0

            unique_vals = con.execute(
                f'SELECT DISTINCT "{col}" FROM {table} WHERE "{col}" IS NOT NULL ORDER BY 1 LIMIT {UNIQUE_DISPLAY_LIMIT + 1}'
            ).fetchall()
            vals = [str(v[0]) for v in unique_vals]
            if len(vals) > UNIQUE_DISPLAY_LIMIT:
                vals = vals[:UNIQUE_DISPLAY_LIMIT]
                vals.append(f"... (총 {unique_cnt}개 중 {UNIQUE_DISPLAY_LIMIT}개만 표시)")

            schema_list.append({
                "name": col,
                "type": next(r[2] for r in info if r[1] == col),
                "unique_count": unique_cnt,
                "unique_values": vals,
                "missing_rate_pct": missing_rate,
            })

        sample = con.execute(f"SELECT * FROM {table} LIMIT {SAMPLE_ROWS}").fetchall()
        sample_rows = [
            dict(zip(col_names, [str(v) for v in row]))
            for row in sample
        ]

        result = {
            "table": table,
            "row_count": row_count,
            "column_count": len(col_names),
            "schema": schema_list,
            "sample": sample_rows,
        }

        out_path = OUT_DIR / f"{table}_schema.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"저장: {out_path}")

    con.close()
    print("완료.")


if __name__ == "__main__":
    main()
