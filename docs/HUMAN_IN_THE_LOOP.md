# 에이전트 아키텍처: 휴먼인더루프 (Human-in-the-Loop)

**설계지침서 산출물** — 불확실 구간 사람 개입 설계

---

## 역할

평가 에이전트 또는 제어 에이전트가 자동 판단을 내리기 어려운 구간에 사람의 검토·결정을 요청합니다.

---

## 개입 트리거 조건

| 조건 | 기준 |
|------|------|
| 지표 임계값 미달 | PRC-AUC < 0.5 또는 R2 < 0.4 |
| 클래스 불균형 심화 | 샘플 비율 80:20 이탈 |
| 피처 충돌 | 변수 중요도 순위 급변 (±3 이상) |
| 배포 결정 | 신규 모델 프로덕션 전환 전 |

---

## 개입 방식

1. **리포트 생성** — `output/{태스크}/review_request.md` 자동 생성
2. **담당자 알림** — 리뷰 요청 내용·판단 근거 포함
3. **결정 기록** — 사람의 결정을 `output/{태스크}/decision_log.md` 저장
4. **재진입** — 결정 반영 후 해당 단계부터 사이클 재시작

---

## TDD 사이클과 연계

```
eval-agent 검증 실패
      ↓
임계값 미달 여부 판단
      ↓
  미달 → human-in-the-loop 개입
  재조정 가능 → Red 사이클 자동 재진입
```

---

## 관련 문서

- [control-agent.agent.md](../.github/agents/control-agent.agent.md) — 라우팅 규칙
- [eval-agent.agent.md](../.github/agents/eval-agent.agent.md) — 검증 기준
- [TDD-ml-cycle.md](TDD-ml-cycle.md) — Red 단계 재진입 조건
