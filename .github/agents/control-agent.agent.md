---
name: Control Agent
description: 에이전트 간 흐름·라우팅·오케스트레이션
---
# 에이전트 아키텍처: 제어 에이전트 (Control Agent)

**설계지침서 산출물** — 흐름·라우팅·오케스트레이션 역할

---

## 역할

에이전트 간 실행 순서를 조율하고 태스크를 적절한 에이전트로 라우팅합니다.

---

## 라우팅 규칙

| 입력 태스크 | 담당 에이전트 |
|------------|--------------|
| 데이터 분포 분석, 피처 엔지니어링 | data-scientist |
| 모델 배포, 서빙 환경 최적화 | ml-ops |
| 출력 품질 검증 | eval-agent |
| 대시보드/시각화/사용자 인터페이스 | web-agent |
| 지표 미달 또는 불확실 구간 | human-in-the-loop |

---

## 오케스트레이션 흐름

```mermaid
flowchart TD
    Start([사용자 요청]) --> Control{{Control Agent<br/>흐름·라우팅·오케스트레이션}}

    Control --> DS[Data Scientist Agent<br/>피처 엔지니어링 · 변수 선정]
    DS --> Eval{Eval Agent<br/>지표 검증<br/>PRC-AUC ≥ 0.521<br/>R² ≥ 0.624 · MAE ≤ 0.60}

    Eval -->|✅ 통과| MLOps[ML Ops Agent<br/>Parquet · DuckDB · 서빙]
    Eval -->|❌ 미통과| Red[🔴 Red 사이클 재진입]

    MLOps --> Web[Web Agent<br/>Streamlit 대시보드]
    Web --> User([👤 사용자<br/>마케팅팀])

    Red --> Check{임계값 미달<br/>또는 누수 의심?}
    Check -->|예| HITL[Human-in-the-Loop<br/>사람 개입·검토]
    Check -->|아니오| DS

    HITL -.재진입.-> DS

    classDef control fill:#003876,stroke:#001f4d,color:#fff,stroke-width:2px
    classDef agent fill:#0066CC,stroke:#003876,color:#fff,stroke-width:2px
    classDef eval fill:#FFA500,stroke:#cc8400,color:#fff,stroke-width:2px
    classDef ops fill:#28A745,stroke:#1e7e34,color:#fff,stroke-width:2px
    classDef web fill:#6F42C1,stroke:#553098,color:#fff,stroke-width:2px
    classDef hitl fill:#DC3545,stroke:#a71d2a,color:#fff,stroke-width:2px
    classDef red fill:#FFE5E5,stroke:#CC0000,color:#CC0000,stroke-width:2px
    classDef terminal fill:#F8F9FA,stroke:#666,color:#333,stroke-width:1px

    class Control control
    class DS agent
    class Eval eval
    class MLOps ops
    class Web web
    class HITL hitl
    class Red red
    class Check red
    class Start,User terminal
```

### 흐름 설명

| 단계 | 에이전트 | 역할 |
|------|---------|------|
| 1️⃣ | **Control Agent** | 작업을 받아 적절한 에이전트로 라우팅 |
| 2️⃣ | **Data Scientist** | 피처 엔지니어링·모델 실험 |
| 3️⃣ | **Eval Agent** | 지표 임계값 검증 |
| 4-A | **ML Ops** (통과) | 서빙 환경 구성 |
| 4-B | **Red 사이클** (미통과) | 재학습 또는 HITL |
| 5️⃣ | **Web Agent** | 결과를 대시보드로 제공 |
| ⚠️ | **Human-in-the-Loop** | 임계값 미달 시 사람 개입 |

---

## 현재 상태

- 수동 오케스트레이션 (`scripts/` 순차 실행)
- 6단계 통합 검증 후 자동화 파이프라인 전환 예정

---

## 관련 문서

- [harness.md](../instructions/harness.md) — 하네스 실행 환경
- [eval-agent.agent.md](eval-agent.agent.md) — 검증 기준
- [HUMAN_IN_THE_LOOP.md](../../docs/HUMAN_IN_THE_LOOP.md) — 개입 설계
