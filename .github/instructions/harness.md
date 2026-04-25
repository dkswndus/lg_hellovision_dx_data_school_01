---
applyTo: "**"
---
# 기반 인프라: 하네스 (Harness)

**설계지침서 산출물** — 에이전트 실행·관리 환경

---

## 역할

하네스(Harness)는 AI 에이전트의 실행·오케스트레이션·컨텍스트 주입을 담당하는 관리 환경입니다.

---

## 구성 요소

| 구성요소 | 위치 | 설명 |
|---------|------|------|
| 에이전트 러너 | `.github/agents/` | 역할별 에이전트 파일 기반 실행 |
| 컨텍스트 주입 | `.github/instructions/` | 지침을 모델 컨텍스트에 로드 |
| 프롬프트 관리 | `.github/prompts/` | 태스크별 프롬프트 템플릿 관리 |
| 결과 수집 | `output/` | 태스크명 하위 디렉터리에 저장 |

---

## 실행 흐름

1. **환경 초기화** — `ml-env.md` 기준 패키지·경로 설정
2. **컨텍스트 주입** — `instructions/` 지침 + `prompts/` 템플릿 로드
3. **에이전트 실행** — 역할(data-scientist / ml-ops / eval / control) 분기
4. **결과 수집** — `output/` 하위 저장, 평가 에이전트 검증
5. **휴먼인더루프** — 임계값 미달 시 사람 개입 요청

---

## 현재 프로젝트 적용

- **러너**: VS Code + GitHub Copilot (`.github/` 구조 기반)
- **컨텍스트**: `copilot-instructions.md` + `instructions/ml-env.md`
- **에이전트**: `data-scientist.agent.md`, `ml-ops.agent.md`, `eval-agent.agent.md`, `control-agent.agent.md`
