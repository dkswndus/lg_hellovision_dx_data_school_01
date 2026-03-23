"""
JSON 스키마 기반 EDA/전처리
1. 결측률 70%↑, 고유값 1개 이하 컬럼 삭제
2. JSON 보고 적절한 타입 변환 (category, int, double, boolean)
3. clean.parquet 저장
"""
import json
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE / "data" / "lghellovision.duckdb"
DATA_DIR = BASE / "data"
INTERIM_DIR = BASE / "data" / "interim"
TABLES = ["vod_log", "vod_content", "user_profile"]

MISSING_THRESH = 70  # 70% 이상 결측 → 삭제
UNIQUE_THRESH = 1  # 고유값 1개 이하 → 삭제
CATEGORY_MAX_UNIQUE = 500  # 고유값 이하면 category(정수코드)로 변환


def _filter_display_only(vals: list) -> list:
    """'... (총 N개 중 20개만 표시)' 같은 문자열 제거"""
    return [v for v in vals if not (isinstance(v, str) and v.startswith("..."))]


def _is_boolean_vals(vals: list) -> bool:
    vset = {str(x).strip().upper() for x in vals}
    return vset.issubset({"Y", "N", "YES", "NO", "T", "F", "1", "0", ""})


def _is_int_vals(vals: list) -> bool:
    for v in vals:
        s = str(v).strip()
        if not s or s in (".", "-"):
            continue
        try:
            int(float(s)) if "." in s else int(s)
        except ValueError:
            return False
    return len([v for v in vals if str(v).strip()]) > 0


def _is_float_vals(vals: list) -> bool:
    for v in vals:
        s = str(v).strip()
        if not s:
            continue
        try:
            float(s)
        except ValueError:
            return False
    return len([v for v in vals if str(v).strip()]) > 0


def infer_target_type(col_schema: dict) -> str:
    """
    JSON 스키마로부터 변환 타입 추론.
    반환: BOOLEAN | BIGINT | DOUBLE | CATEGORY | VARCHAR
    """
    dtype = col_schema.get("type", "VARCHAR")
    unique_count = col_schema.get("unique_count", 0)
    vals = _filter_display_only(col_schema.get("unique_values", []))

    if dtype in ("BIGINT", "INTEGER", "DOUBLE", "BOOLEAN"):
        return dtype

    if dtype != "VARCHAR" or not vals:
        return "VARCHAR"

    # Y/N → BOOLEAN
    if _is_boolean_vals(vals):
        return "BOOLEAN"

    # 숫자형 (정수 우선, 소수점 없을 때만)
    non_empty = [v for v in vals if str(v).strip()]
    if non_empty and not any("." in str(v) for v in non_empty) and _is_int_vals(vals):
        return "BIGINT"
    if _is_float_vals(vals):
        return "DOUBLE"

    # 저카디널리티 문자열 → CATEGORY
    if unique_count <= CATEGORY_MAX_UNIQUE and unique_count > 1:
        return "CATEGORY"

    return "VARCHAR"


def build_select_with_conversions(
    con, table: str, cols_to_keep: list[dict], all_schema: list[dict]
) -> tuple[str, dict]:
    """
    변환 적용 SELECT 절 생성.
    반환: (select_clause, category_mappings)
    """
    selects = []
    category_mappings = {}

    for col_info in cols_to_keep:
        col = col_info["name"]
        dtype = col_info.get("type", "VARCHAR")
        target = infer_target_type(col_info)

        if target == "BOOLEAN":
            selects.append(
                f"CASE WHEN TRIM(CAST(\"{col}\" AS VARCHAR)) IN ('Y','YES','T','1') THEN TRUE "
                f"WHEN TRIM(CAST(\"{col}\" AS VARCHAR)) IN ('N','NO','F','0') THEN FALSE ELSE NULL END AS \"{col}\""
            )
        elif target == "BIGINT":
            if dtype == "VARCHAR":
                selects.append(f'TRY_CAST(TRIM("{col}") AS BIGINT) AS "{col}"')
            else:
                selects.append(f'"{col}"')
        elif target == "DOUBLE":
            if dtype == "VARCHAR":
                selects.append(f'TRY_CAST(TRIM("{col}") AS DOUBLE) AS "{col}"')
            else:
                selects.append(f'"{col}"')
        elif target == "CATEGORY":
            distinct_rows = con.execute(
                f'SELECT DISTINCT "{col}" FROM {table} WHERE "{col}" IS NOT NULL ORDER BY 1'
            ).fetchall()
            mapping = {str(r[0]): i for i, r in enumerate(distinct_rows)}
            category_mappings[col] = mapping
            cases = " ".join(
                f'WHEN TRIM("{col}") = \'{str(k).replace(chr(39), chr(39)+chr(39))}\' THEN {v}'
                for k, v in mapping.items()
            )
            selects.append(f"CASE {cases} ELSE NULL END AS \"{col}\"")
        else:
            selects.append(f'"{col}"')

    return ", ".join(selects), category_mappings


def process_table(con, table: str) -> None:
    schema_path = DATA_DIR / f"{table}_schema.json"
    if not schema_path.exists():
        print(f"  {table}: 스키마 JSON 없음. export_schema_json.py 먼저 실행.")
        return

    with open(schema_path, encoding="utf-8") as f:
        data = json.load(f)

    schema_list = data.get("schema", [])
    cols_to_keep = []
    dropped = []

    for col_info in schema_list:
        name = col_info["name"]
        missing = col_info.get("missing_rate_pct", 0)
        unique = col_info.get("unique_count", 0)

        if missing >= MISSING_THRESH:
            dropped.append(f"{name}(결측{missing}%)")
            continue
        if unique <= UNIQUE_THRESH:
            dropped.append(f"{name}(고유{unique}개)")
            continue
        cols_to_keep.append(col_info)

    if dropped:
        print(f"  삭제: {len(dropped)}개 {dropped[:5]}{'...' if len(dropped) > 5 else ''}")

    if not cols_to_keep:
        print(f"  {table}: 유지 컬럼 없음")
        return

    select_clause, cat_mappings = build_select_with_conversions(
        con, table, cols_to_keep, schema_list
    )

    out_path = INTERIM_DIR / f"{table}_clean.parquet"
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT {select_clause}
            FROM {table}
        ) TO ? (FORMAT PARQUET)
        """,
        [str(out_path)],
    )
    print(f"  저장: {out_path}")

    if cat_mappings:
        map_path = INTERIM_DIR / f"{table}_categories.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(cat_mappings, f, ensure_ascii=False, indent=2)
        print(f"  카테고리 매핑: {map_path}")


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    for table in TABLES:
        print(f"\n[{table}]")
        process_table(con, table)

    con.close()
    print("\n완료.")


if __name__ == "__main__":
    main()
