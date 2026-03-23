"""
clean parquet 데이터 시각화
- output/{table}/수치형/, output/{table}/범주형/ 폴더 구조
- 각 유형별 모든 시각화를 하나의 이미지에 그리드로 배치
"""
import json
import os
from pathlib import Path

import duckdb
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Windows 맑은 고딕)
if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
INTERIM = BASE / "data" / "interim"
OUTPUT = BASE / "output"
TABLES = ["vod_log", "vod_content", "user_profile"]

SKIP_COLS = {"sha2_hash", "asset", "asset_nm", "asset_id", "full_asset_id", "epsd_id", "cts_id"}
BAR_MAX = 20
COLS_PER_ROW = 4  # 수치형 그리드 열 수


def load_categories_reverse(table: str) -> dict:
    path = INTERIM / f"{table}_categories.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        cat = json.load(f)
    result = {}
    for col, mapping in cat.items():
        result[col] = {}
        for k, v in mapping.items():
            try:
                result[col][int(v)] = k
            except (ValueError, TypeError):
                result[col][v] = k
    return result


def get_categorical_data(con, table: str, col: str, rev_map: dict) -> tuple[list, list] | None:
    path = str((INTERIM / f"{table}_clean.parquet").resolve()).replace("\\", "/")
    df = con.execute(
        f'SELECT "{col}", COUNT(*) AS cnt FROM read_parquet(\'{path}\') GROUP BY 1 ORDER BY 2 DESC LIMIT {BAR_MAX}'
    ).fetchdf()
    if df.empty:
        return None
    vals = df.iloc[:, 0].astype(str)
    counts = df.iloc[:, 1].values
    if col in rev_map:
        def _label(v):
            try:
                return rev_map[col].get(int(float(v)), str(v)[:15])
            except (ValueError, TypeError):
                return rev_map[col].get(v, str(v)[:15])
        labels = [_label(v) for v in vals]
    else:
        labels = [str(v)[:15] + (".." if len(str(v)) > 15 else "") for v in vals]
    return labels, counts


def get_numeric_data(con, table: str, col: str) -> np.ndarray | None:
    path = str((INTERIM / f"{table}_clean.parquet").resolve()).replace("\\", "/")
    df = con.execute(
        f'SELECT "{col}" FROM read_parquet(\'{path}\') WHERE "{col}" IS NOT NULL'
    ).fetchdf()
    if df.empty:
        return None
    arr = df.iloc[:, 0].dropna().astype(float)
    return arr.values if len(arr) > 0 else None


def plot_all_categorical(con, table: str, cols: list[str], rev_map: dict, out_dir: Path):
    """범주형 컬럼들을 하나의 figure에 그리드로 (2x2, 3x3 등 정사각형에 가깝게)"""
    if not cols:
        return
    n = len(cols)
    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if n == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(cols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        data = get_categorical_data(con, table, col, rev_map)
        if data is None:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            ax.set_title(col, fontsize=9)
            continue
        labels, counts = data
        total = counts.sum()
        pcts = counts / total * 100 if total else counts
        x = range(len(labels))
        bars = ax.bar(x, counts, color="#3498db", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        max_cnt = max(counts) if len(counts) > 0 else 1
        ax.set_ylim(0, max_cnt * 1.25)
        for bar, pct in zip(bars, pcts):
            h = bar.get_height()
            ax.annotate(f"{pct:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=6)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis("off")

    plt.suptitle(f"{table} 범주형 분포", fontsize=12, y=1.02)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "범주형_전체.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  {table} 범주형: {len(cols)}개 컬럼 → {out_path}")


def plot_all_numeric(con, table: str, cols: list[str], out_dir: Path):
    """수치형 컬럼들을 하나의 figure에 그리드로 (박스플롯 + 이상치)"""
    if not cols:
        return
    n = len(cols)
    nrows = (n + COLS_PER_ROW - 1) // COLS_PER_ROW
    ncols = min(n, COLS_PER_ROW)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if n == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(cols):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        arr = get_numeric_data(con, table, col)
        if arr is None or len(arr) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
            ax.set_title(col, fontsize=9)
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = np.sum((arr < lower) | (arr > upper))
        pct_out = n_out / len(arr) * 100

        bp = ax.boxplot([arr], vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("#2ecc71")
        ax.set_xticklabels([col[:20]], fontsize=8)
        ax.set_ylabel("값", fontsize=8)
        ax.set_title(f"{col}\n이상치 {n_out:,}개 ({pct_out:.2f}%)", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis("off")

    plt.suptitle(f"{table} 수치형 분포 (박스플롯·이상치)", fontsize=12, y=1.02)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "수치형_전체.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  {table} 수치형: {len(cols)}개 컬럼 → {out_path}")


def get_col_types(con, table: str) -> list[tuple[str, str]]:
    path = str((INTERIM / f"{table}_clean.parquet").resolve()).replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0").fetchdf()
    return [(c, str(df[c].dtype)) for c in df.columns]


def main():
    con = duckdb.connect()

    for table in TABLES:
        pq = INTERIM / f"{table}_clean.parquet"
        if not pq.exists():
            print(f"[{table}] parquet 없음, 스킵")
            continue

        print(f"\n[{table}]")
        rev_map = load_categories_reverse(table)
        cols = get_col_types(con, table)

        cat_cols = []
        num_cols = []
        for col, dtype in cols:
            if col in SKIP_COLS:
                continue
            if col in rev_map or ("int" not in dtype and "float" not in dtype):
                cat_cols.append(col)
            else:
                num_cols.append(col)

        out_base = OUTPUT / table
        plot_all_categorical(con, table, cat_cols, rev_map, out_base / "범주형")
        plot_all_numeric(con, table, num_cols, out_base / "수치형")

    con.close()
    print(f"\n저장: {OUTPUT}")


if __name__ == "__main__":
    main()
