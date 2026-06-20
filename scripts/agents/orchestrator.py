"""LangGraph orchestrator — 11-agent ML 파이프라인의 흐름·체크포인트·HITL.

각 노드는 현재 stub. 실제 ML 작업은 scripts/agents/{profiler,leakage_checker,
eda,feature_engineering,modeling,explanation}.py 가 채워질 때 import 해서 호출.

Studio 진입점: langgraph.json 의 graphs."churn_pipeline" -> "graph"
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

BASE = Path(__file__).resolve().parent.parent.parent


# ────────────────────────────────────────────────────────────────────
# State 정의
# ────────────────────────────────────────────────────────────────────


def _merge_list(existing: list, new: list) -> list:
    return existing + new


class ChurnPipelineState(TypedDict, total=False):
    """파이프라인 전역 상태. 노드들이 누적·갱신."""

    run_id: str
    task: Literal["churn", "vod_purchase"]

    # 산출물 경로
    profile_path: str
    leakage_report_path: str
    eda_summary_path: str
    feature_spec_path: str
    leaderboard_path: str
    explain_brief_path: str

    # 의사결정·게이트
    leakage_severity: Literal["none", "warn", "critical"]
    gate_pass: bool
    winner_model: str
    winner_metric: float

    # 체크포인트 응답
    checkpoint_decisions: Annotated[list[dict], _merge_list]

    # HITL 트리거
    hitl_triggered: bool
    hitl_reason: str


# ────────────────────────────────────────────────────────────────────
# 노드 함수 (stub)
# ────────────────────────────────────────────────────────────────────


def _stub(name: str, state: ChurnPipelineState, **outputs) -> ChurnPipelineState:
    """모든 노드 공통 stub — run_id 기록 + 산출물 경로 placeholder."""
    run_id = state.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[{name}] run_id={run_id} state_keys={list(state.keys())}")
    return {"run_id": run_id, **outputs}


def supervisor(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub("supervisor", state)


def data_profiler(state: ChurnPipelineState) -> ChurnPipelineState:
    table = "churn_dataset_v2" if state.get("task", "churn") == "churn" else "vod_purchase_dataset"
    return _stub(
        "data_profiler",
        state,
        profile_path=f"output/profile/{table}_profile.json",
    )


def leakage_checker(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub(
        "leakage_checker",
        state,
        leakage_report_path="output/leakage/leakage_report.json",
        leakage_severity="none",  # stub: 실제 구현시 corr·시점·중복키 검사 결과
    )


def eda(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub(
        "eda",
        state,
        eda_summary_path="output/eda/eda_summary.md",
    )


def feature_engineering(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub(
        "feature_engineering",
        state,
        feature_spec_path="output/features/feature_spec.json",
    )


def modeling(state: ChurnPipelineState) -> ChurnPipelineState:
    # 실제 구현시 LGBM/XGB/CatBoost 병렬 학습
    return _stub(
        "modeling",
        state,
        leaderboard_path="output/models/leaderboard.csv",
    )


def evaluation(state: ChurnPipelineState) -> ChurnPipelineState:
    # 실제 구현시 leaderboard 읽어서 PRC-AUC≥0.5 / R²≥0.6 / MAE≤0.6 게이트 검증
    return _stub(
        "evaluation",
        state,
        gate_pass=True,
        winner_model="xgboost",
        winner_metric=0.5214,
    )


def explanation(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub(
        "explanation",
        state,
        explain_brief_path="output/explain/explain_brief.md",
    )


def ml_ops(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub("ml_ops", state)


def web(state: ChurnPipelineState) -> ChurnPipelineState:
    return _stub("web", state)


def hitl(state: ChurnPipelineState) -> ChurnPipelineState:
    print(f"[HITL] triggered: {state.get('hitl_reason', 'unknown')}")
    return _stub("hitl", state, hitl_triggered=True)


# ────────────────────────────────────────────────────────────────────
# 조건부 라우팅
# ────────────────────────────────────────────────────────────────────


def route_after_leakage(state: ChurnPipelineState) -> Literal["hitl", "eda"]:
    if state.get("leakage_severity") == "critical":
        return "hitl"
    return "eda"


def route_after_evaluation(state: ChurnPipelineState) -> Literal["hitl", "explanation"]:
    if not state.get("gate_pass", False):
        return "hitl"
    return "explanation"


# ────────────────────────────────────────────────────────────────────
# Graph 빌드
# ────────────────────────────────────────────────────────────────────


def build_graph():
    builder = StateGraph(ChurnPipelineState)

    # 노드 등록
    builder.add_node("supervisor", supervisor)
    builder.add_node("data_profiler", data_profiler)
    builder.add_node("leakage_checker", leakage_checker)
    builder.add_node("eda", eda)
    builder.add_node("feature_engineering", feature_engineering)
    builder.add_node("modeling", modeling)
    builder.add_node("evaluation", evaluation)
    builder.add_node("explanation", explanation)
    builder.add_node("ml_ops", ml_ops)
    builder.add_node("web", web)
    builder.add_node("hitl", hitl)

    # 엣지: 선형 흐름
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "data_profiler")
    builder.add_edge("data_profiler", "leakage_checker")

    # 조건부: leakage 결과
    builder.add_conditional_edges(
        "leakage_checker",
        route_after_leakage,
        {"hitl": "hitl", "eda": "eda"},
    )

    builder.add_edge("eda", "feature_engineering")
    builder.add_edge("feature_engineering", "modeling")
    builder.add_edge("modeling", "evaluation")

    # 조건부: eval 게이트
    builder.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {"hitl": "hitl", "explanation": "explanation"},
    )

    builder.add_edge("explanation", "ml_ops")
    builder.add_edge("ml_ops", "web")
    builder.add_edge("web", END)
    builder.add_edge("hitl", END)

    # 체크포인트 4곳: leakage·eda·evaluation·explanation 노드 *후* 사람 확인
    # checkpointer 는 LangGraph platform/dev 가 자동 주입 (POSTGRES_URI 환경변수 참고)
    return builder.compile(
        interrupt_after=["leakage_checker", "eda", "evaluation", "explanation"],
    )


# Studio entry point — langgraph.json 에서 참조
graph = build_graph()


if __name__ == "__main__":
    # CLI 실행 (Studio 외에 직접 돌릴 때)
    config = {"configurable": {"thread_id": datetime.now().strftime("%Y%m%d-%H%M%S")}}
    initial = ChurnPipelineState(task="churn")
    for step in graph.stream(initial, config=config):
        print(step)
