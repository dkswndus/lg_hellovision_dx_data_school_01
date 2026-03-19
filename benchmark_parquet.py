"""
Parquet 변환 전후: 파일 용량, 처리속도, 메모리 비교
"""
import time
from pathlib import Path

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw"
INTERIM = BASE / "data" / "interim"
DB_PATH = BASE / "data" / "lghellovision.duckdb"

# 테이블별 원본 CSV
SOURCES = {
    "user_profile": list(RAW.glob("TPS_cancel_*_p_mt.csv")),
    "vod_content": [RAW / "vod_mart_data.csv"],
    "vod_log": list(RAW.glob("*_VOD.csv")),
}

PARQUET = {
    "user_profile": INTERIM / "user_profile_clean.parquet",
    "vod_content": INTERIM / "vod_content_clean.parquet",
    "vod_log": INTERIM / "vod_log_clean.parquet",
}


def get_size_mb(paths) -> float:
    if isinstance(paths, Path):
        return paths.stat().st_size / (1024 * 1024) if paths.exists() else 0
    return sum(p.stat().st_size for p in paths if p.exists()) / (1024 * 1024)


def benchmark_read(con, query: str, n=3) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        con.execute(query).fetchall()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    import duckdb

    results = []

    for table in ["user_profile", "vod_content", "vod_log"]:
        csv_size = get_size_mb(SOURCES[table])
        pq_path = PARQUET[table]
        pq_size = get_size_mb(pq_path) if pq_path.exists() else 0

        size_reduction = (1 - pq_size / csv_size) * 100 if csv_size > 0 else 0

        con = duckdb.connect(":memory:")

        # CSV 읽기 속도 (원본)
        csv_paths = SOURCES[table]
        if len(csv_paths) > 1:
            csv_arg = "[" + ",".join(f"'{p.as_posix()}'" for p in csv_paths) + "]"
        else:
            csv_arg = f"'{csv_paths[0].as_posix()}'"
        csv_opts = "header=true, all_varchar=true"
        if table == "vod_log":
            csv_opts += ", strict_mode=false, ignore_errors=true"
        t_csv = benchmark_read(con, f"SELECT COUNT(*) FROM read_csv_auto({csv_arg}, {csv_opts})")
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto({csv_arg}, {csv_opts})").fetchone()[0]

        # Parquet 읽기 속도
        t_pq = benchmark_read(con, f"SELECT COUNT(*) FROM read_parquet('{pq_path.as_posix()}')")

        con.close()

        speed_improvement = (1 - t_pq / t_csv) * 100 if t_csv > 0 else None

        results.append({
            "table": table,
            "csv_mb": round(csv_size, 2),
            "parquet_mb": round(pq_size, 2),
            "size_reduction_pct": round(size_reduction, 1),
            "rows": row_count,
            "t_csv_sec": round(t_csv, 3),
            "t_parquet_sec": round(t_pq, 3),
            "speed_improvement_pct": round(speed_improvement, 1) if speed_improvement is not None else None,
        })

    return results


if __name__ == "__main__":
    r = main()
    for x in r:
        print(x)
