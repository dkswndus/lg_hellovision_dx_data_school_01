"""Eval Agent — leaderboard.csv 기반 게이트 검증 및 챔피언 모델 선정.

.github/agents/eval-agent.agent.md 스펙 구현체. tests/test_model_performance.py 와
동일한 임계값(PRC-AUC≥0.5 / R2≥0.6, MAE≤0.6)을 사용한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
OUTPUT_MODELS = BASE / "output" / "models"
OUTPUT_EVAL = BASE / "output" / "eval"

THRESH_PRC_AUC = 0.5
THRESH_R2 = 0.6
THRESH_MAE = 0.6

GATE_COL = {"churn": "Val PRC-AUC", "vod_purchase": "Val R2"}


def _check_gate(task: str, row: pd.Series) -> tuple[bool, str]:
    if task == "churn":
        prc = float(row["Val PRC-AUC"])
        passed = prc >= THRESH_PRC_AUC
        return passed, f"PRC-AUC {prc:.4f} {'>=' if passed else '<'} {THRESH_PRC_AUC}"
    r2 = float(row["Val R2"])
    mae = float(row["MAE"])
    passed = r2 >= THRESH_R2 and mae <= THRESH_MAE
    return passed, (
        f"R2 {r2:.4f} ({'>=' if r2 >= THRESH_R2 else '<'} {THRESH_R2}), "
        f"MAE {mae:.4f} ({'<=' if mae <= THRESH_MAE else '>'} {THRESH_MAE})"
    )


def run_evaluation(
    task: Literal["churn", "vod_purchase"] = "churn",
    leaderboard_path: str | Path | None = None,
) -> dict:
    """leaderboard.csv 를 읽어 게이트를 검증하고 챔피언 모델을 선정한다."""
    path = Path(leaderboard_path) if leaderboard_path else OUTPUT_MODELS / f"leaderboard_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — modeling 단계 먼저 실행 필요")

    leaderboard = pd.read_csv(path, encoding="utf-8-sig")
    sort_col = GATE_COL[task]
    leaderboard = leaderboard.sort_values(sort_col, ascending=False).reset_index(drop=True)

    per_model = []
    for _, row in leaderboard.iterrows():
        passed, reason = _check_gate(task, row)
        per_model.append({"모델": row["모델"], "gate_pass": passed, "reason": reason})

    passing_names = [m["모델"] for m in per_model if m["gate_pass"]]
    if passing_names:
        champion_name = passing_names[0]  # leaderboard 가 이미 정렬됨 → 통과 모델 중 1위
        gate_pass = True
    else:
        champion_name = leaderboard.iloc[0]["모델"]  # 미통과 시 참고용으로 1위 표기
        gate_pass = False

    champion_row = leaderboard[leaderboard["모델"] == champion_name].iloc[0]
    result = {
        "task": task,
        "gate_pass": gate_pass,
        "winner_model": str(champion_name),
        "winner_metric": float(champion_row[sort_col]),
        "metric_name": sort_col,
        "per_model": per_model,
    }

    OUTPUT_EVAL.mkdir(parents=True, exist_ok=True)
    gate_path = OUTPUT_EVAL / "gate_result.json"
    gate_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["gate_result_path"] = str(gate_path.relative_to(BASE)).replace("\\", "/")

    status = "PASS" if gate_pass else "FAIL"
    print(f"[evaluation] task={task} gate={status} winner={champion_name} {sort_col}={result['winner_metric']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval agent — leaderboard 게이트 검증")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    parser.add_argument("--leaderboard-path", default=None)
    args = parser.parse_args()

    result = run_evaluation(task=args.task, leaderboard_path=args.leaderboard_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
