"""
cancel_yn(유지/해지) 비율 막대그래프 생성
"""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "lghellovision.duckdb"
OUTPUT = BASE / "output"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """
        SELECT cancel_yn, COUNT(*) AS cnt
        FROM user_profile
        WHERE cancel_yn IS NOT NULL AND LENGTH(TRIM(cancel_yn)) > 0
        GROUP BY cancel_yn
        ORDER BY cnt DESC
        """
    ).fetchall()
    con.close()

    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    total = sum(counts) if counts else 1
    pcts = [c / total * 100 for c in counts]

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        counts,
        color=["#2ecc71", "#e74c3c"][: len(labels)],
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_ylabel("건수")
    ax.set_title("cancel_yn 유지/해지 비율")
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)

    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    out_path = OUTPUT / "cancel_yn_ratio.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
