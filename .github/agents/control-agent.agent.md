---
name: Control Agent (DEPRECATED)
description: → supervisor-agent.agent.md 로 이전됨. 본 파일은 백워드 호환용 스텁.
---
# Control Agent (DEPRECATED)

> ⚠️ **이 에이전트는 [Supervisor Agent](supervisor-agent.agent.md) 로 격상·리네임되었습니다.**
> 백워드 호환을 위해 파일은 유지하지만 신규 작업은 supervisor-agent 를 참조하세요.

---

## 이전 사유

11-에이전트 ML 파이프라인(profiler → leakage → eda → FE → modeling → eval → explain → ops/web) 도입에 맞춰 **semi-auto 오케스트레이션 + 체크포인트 + HITL 트리거** 책무가 추가되어 역할 범위가 넓어졌습니다.

| 구분 | Control Agent (구) | Supervisor Agent (신) |
|------|-------------------|----------------------|
| 자율성 | 수동 라우팅 | Semi-auto 파이프라인 실행 |
| 체크포인트 | 없음 | 4곳 (leakage·eda·leaderboard·explain) |
| HITL 트리거 | 임계값 미달만 | + leakage CRITICAL, 키 중복, 시점 누수 |
| 워크플로 | 문서 텍스트 | `.specify/workflows/churn-pipeline/workflow.yml` |

---

## 이동 안내

- 라우팅 규칙·HITL 트리거: → [supervisor-agent.agent.md](supervisor-agent.agent.md)
- 파이프라인 정의: → [.specify/workflows/churn-pipeline/workflow.yml](../../.specify/workflows/churn-pipeline/workflow.yml)
- 변경 이력: `git log .github/agents/control-agent.agent.md`
