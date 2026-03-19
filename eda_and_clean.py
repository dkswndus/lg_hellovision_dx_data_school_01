"""
EDA 및 전처리: 결측 70% 이상 삭제, 타입 변환(메모리 절약), 이상값 리포트
"""
import duckdb
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "lghellovision.duckdb"
INTERIM_DIR = BASE_DIR / "data" / "interim"
REPORT_PATH = BASE_DIR / "data" / "interim" / "eda_report.json"

TABLES = ["user_profile", "vod_content", "vod_log"]


def analyze_table(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    """결측률, 유니크 수, 샘플값 분석"""
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    result = {"table": table, "total_rows": total, "columns": {}}

    for col in cols:
        # 결측률 (NULL + 빈문자열)
        null_cnt = con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL'
        ).fetchone()[0]
        empty_cnt = con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE TRIM("{col}") = \'\''
        ).fetchone()[0]
        missing = null_cnt + empty_cnt
        missing_pct = (missing / total * 100) if total else 0

        # 유니크 수
        try:
            uniq = con.execute(
                f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"'
            ).fetchone()[0]
        except Exception:
            uniq = -1

        # 샘플값 (상위 5개)
        try:
            samples = con.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL AND TRIM("{col}") != \'\' LIMIT 5'
            ).fetchall()
            samples = [str(s[0])[:50] for s in samples]
        except Exception:
            samples = []

        result["columns"][col] = {
            "missing_pct": round(missing_pct, 2),
            "unique_count": uniq,
            "sample_values": samples,
        }
    return result


def _is_numeric_int(samples: list) -> bool:
    """샘플이 정수 형태인지"""
    for s in samples:
        s = str(s).strip().replace('"', "")
        if not s:
            continue
        if "." in s:
            return False
        if s.replace("-", "").isdigit():
            continue
        return False
    return bool(samples)


def _is_numeric_float(samples: list) -> bool:
    """샘플이 실수 형태인지"""
    for s in samples:
        s = str(s).strip().replace('"', "")
        if not s:
            continue
        try:
            float(s)
        except ValueError:
            return False
    return bool(samples)


def _cast_expr(col: str, info: dict) -> tuple[str, str]:
    """메모리 절약을 위한 타입 변환 표현식. (sql_expr, type_name) 반환."""
    uniq = info.get("unique_count", 999)
    samples = [str(s).strip().replace('"', "") for s in info.get("sample_values", []) if s]
    samples_upper = [s.upper() for s in samples]

    # 1. Y/N, 0/1 등 -> BOOLEAN
    if uniq <= 3 and samples:
        yn_vals = {"Y", "N", "YES", "NO", "1", "0", ""}
        if all(s in yn_vals or (s.isdigit() and int(s) in (0, 1)) for s in samples_upper):
            sql = f'''CASE WHEN TRIM("{col}") IN ('Y','1','YES') THEN true
                WHEN TRIM("{col}") IN ('N','0','NO','') OR "{col}" IS NULL THEN false
                ELSE NULL END AS "{col}"'''
            return sql, "BOOLEAN"

    # 2. 정수 형태 -> INTEGER (범위에 따라 TINYINT/SMALLINT는 min/max 필요해 생략)
    if _is_numeric_int(samples):
        return f'TRY_CAST("{col}" AS INTEGER) AS "{col}"', "INTEGER"

    # 3. 실수 형태 -> FLOAT (32bit, DOUBLE보다 절약)
    if _is_numeric_float(samples):
        return f'TRY_CAST("{col}" AS FLOAT) AS "{col}"', "FLOAT"

    # 4. 그 외 (문자열, 카테고리) -> VARCHAR 유지 (Parquet dictionary encoding으로 압축)
    return f'"{col}"', "VARCHAR"


def run_eda_and_clean():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    full_report = {"tables": {}, "issues": []}

    for table in TABLES:
        print(f"\n=== {table} 분석 중 ===")
        analysis = analyze_table(con, table)
        full_report["tables"][table] = analysis

        # 70% 이상 결측 컬럼
        drop_cols = [
            c for c, info in analysis["columns"].items()
            if info["missing_pct"] >= 70.0
        ]
        if drop_cols:
            full_report["issues"].append({
                "table": table,
                "type": "dropped_high_missing",
                "columns": drop_cols,
                "reason": "결측 70% 이상",
            })
            print(f"  삭제할 컬럼 (결측≥70%): {drop_cols}")

        # 값이 1개뿐인 컬럼
        single_val = [
            c for c, info in analysis["columns"].items()
            if info["unique_count"] == 1
        ]
        if single_val:
            full_report["issues"].append({
                "table": table,
                "type": "single_value",
                "columns": single_val,
                "reason": "유니크 값 1개",
            })
            print(f"  값 1개뿐인 컬럼: {single_val}")

        # 이상값 후보 (빈문자열 비율 높음, 또는 의심스러운 패턴)
        for col, info in analysis["columns"].items():
            if info["missing_pct"] >= 50 and info["missing_pct"] < 70:
                full_report["issues"].append({
                    "table": table,
                    "type": "high_missing",
                    "column": col,
                    "missing_pct": info["missing_pct"],
                    "reason": "결측 50~70%",
                })

    # 리포트 저장
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\nEDA 리포트 저장: {REPORT_PATH}")

    # 전처리: 70% 이상 결측 삭제 + 타입 변환 + parquet 저장
    type_conversions = {}

    for table in TABLES:
        analysis = full_report["tables"][table]
        cols = analysis["columns"]
        keep_cols = [(c, info) for c, info in cols.items() if info["missing_pct"] < 70.0]
        if not keep_cols:
            print(f"  {table}: 유지할 컬럼 없음, 스킵")
            continue

        sel_parts = []
        type_conversions[table] = {"BOOLEAN": [], "INTEGER": [], "FLOAT": [], "VARCHAR": []}
        for c, info in keep_cols:
            sql, type_name = _cast_expr(c, info)
            sel_parts.append(sql)
            if type_name != "VARCHAR":
                type_conversions[table][type_name].append(c)

        sel = ", ".join(sel_parts)
        out_path = INTERIM_DIR / f"{table}_clean.parquet"
        con.execute(f"""
            COPY (
                SELECT {sel}
                FROM "{table}"
            ) TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        print(f"  {table} -> {out_path} ({len(keep_cols)} 컬럼)")
        for tname, cols_list in type_conversions[table].items():
            if cols_list and tname != "VARCHAR":
                print(f"    {tname}: {cols_list}")

    full_report["type_conversions"] = type_conversions
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    con.close()
    return full_report


def update_result_report(report: dict) -> None:
    """결과보고서.md 갱신"""
    report_path = BASE_DIR / "결과보고서.md"
    tc = report.get("type_conversions", {})

    lines = [
        "# EDA 전처리 결과 보고서",
        "",
        "## 결측치 70% 이상 컬럼 삭제 결과",
        "",
        "| 테이블 | 삭제 전 컬럼 수 | 삭제 후 컬럼 수 | 삭제된 컬럼 수 |",
        "|--------|----------------|----------------|----------------|",
    ]

    totals = {"before": 0, "after": 0, "dropped": 0}
    for table in TABLES:
        cols = report["tables"][table]["columns"]
        before = len(cols)
        dropped = next(
            (len(i["columns"]) for i in report.get("issues", [])
             if i.get("type") == "dropped_high_missing" and i.get("table") == table),
            0,
        )
        after = before - dropped
        totals["before"] += before
        totals["after"] += after
        totals["dropped"] += dropped
        lines.append(f"| {table} | {before} | {after} | {dropped} |")
    lines.append(f"| **합계** | **{totals['before']}** | **{totals['after']}** | **{totals['dropped']}** |")
    lines.append("")

    dropped_issue = next(
        (i for i in report.get("issues", []) if i.get("type") == "dropped_high_missing"),
        None,
    )
    if dropped_issue:
        lines.append("### vod_content 삭제된 컬럼 (24개)")
        lines.append("결측 70% 이상으로 삭제된 컬럼:")
        lines.append("- " + ", ".join(dropped_issue["columns"]))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 데이터 타입 변환 (메모리 절약)")
    lines.append("")
    lines.append("원본: 모든 컬럼 VARCHAR (CSV all_varchar 적재)")
    lines.append("")
    lines.append("### 변환 규칙")
    lines.append("- **VARCHAR → BOOLEAN**: Y/N, 0/1, YES/NO 형태 (유니크≤3)")
    lines.append("- **VARCHAR → INTEGER**: 정수 형태 샘플")
    lines.append("- **VARCHAR → FLOAT**: 실수 형태 샘플 (32bit, DOUBLE 대비 절약)")
    lines.append("- **VARCHAR 유지**: 문자열/카테고리 (Parquet dictionary encoding으로 압축)")
    lines.append("")

    for table in TABLES:
        if table not in tc:
            continue
        t = tc[table]
        lines.append(f"### {table}")
        for type_name in ["BOOLEAN", "INTEGER", "FLOAT"]:
            cols_list = t.get(type_name, [])
            if cols_list:
                lines.append(f"- **{type_name}**: {', '.join(cols_list)}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"결과보고서 갱신: {report_path}")


def print_user_report(report: dict):
    """사용자용 요약 리포트 출력"""
    print("\n" + "=" * 60)
    print("[이상/주의 컬럼 요약] (확인 필요)")
    print("=" * 60)
    for issue in report["issues"]:
        t = issue["table"]
        if issue["type"] == "single_value":
            print(f"\n[{t}] 값이 1개뿐인 컬럼 (분석에 무의미): {issue['columns']}")
        elif issue["type"] == "dropped_high_missing":
            print(f"\n[{t}] 삭제된 컬럼 (결측≥70%): {issue['columns']}")
        elif issue["type"] == "high_missing":
            print(f"\n[{t}] 결측 50~70% 컬럼 (주의): {issue['column']} ({issue['missing_pct']}%)")
    print("\n상세 리포트:", REPORT_PATH)


if __name__ == "__main__":
    try:
        report = run_eda_and_clean()
        update_result_report(report)
        print_user_report(report)
    except Exception as e:
        if "Cannot open file" in str(e) or "already open" in str(e).lower():
            print("\n⚠️ DuckDB 파일이 다른 프로그램에서 사용 중입니다.")
            print("   SQLTools/DBeaver에서 연결을 끊은 뒤 다시 실행하세요.")
        raise
