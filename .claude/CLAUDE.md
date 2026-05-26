# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**매출 = 고객 수 × 고객당 매출** 구조 하에 LG헬로비전 VOD·해지·매출을 분석하는 ML 프로젝트.

- **해지 예측**: 분류 (PRC-AUC 기준, 불균형 데이터 80:20)
- **VOD 구매 예측**: 회귀 (R2 기준, zero-inflated)
- **핵심 인사이트**: TV 시청 감소 ≠ 이탈 — 소비 방식이 TV → VOD로 확장

## 개발 명령어

```bash
# 의존성 설치
pip install -r requirements.txt

# 1. 원본 CSV → DuckDB 적재
python scripts/build/load_raw_to_duckdb.py

# 2. EDA·전처리 (Parquet 생성)
python scripts/eda/eda_and_clean.py

# 3. 학습용 데이터셋 구축 (순서 중요)
python scripts/build/build_churn_dataset.py        # churn_dataset (1차)
python scripts/build/build_churn_dataset_v2.py     # churn_dataset_v2 (2차, +7변수)
python scripts/build/build_vod_purchase_dataset.py # VOD 구매 예측용

# 4. 분석·모델 실행
python scripts/analysis/churn_analysis.py          # 해지 1차 (변수 선정 + 가설검정)
python scripts/analysis/run_churn_v2.py            # 해지 2차 (PRC-AUC 개선)
python scripts/analysis/vod_purchase_model.py      # VOD 구매 예측

# 5. 시각화
python scripts/visualization/plot_ct_cl_by_churn.py
python scripts/visualization/plot_tv_vod_expansion.py

# 테스트 (모델 성능 지표 검증)
uv run pytest tests/test_model_performance.py
```

모든 스크립트는 **프로젝트 루트에서 실행**합니다. `BASE = Path(__file__).resolve().parent.parent.parent` 패턴으로 경로가 고정됩니다.

## 아키텍처

### 데이터 흐름

```
data/raw/*.csv
    → (load_raw_to_duckdb.py) → data/*.duckdb
    → (eda_and_clean.py)      → data/interim/*_clean.parquet
    → (build_*.py)            → data/processed/*.parquet  ← 모델 학습 입력
    → (analysis/*.py)         → output/{태스크}/          ← CSV·PNG 결과
```

### 핵심 데이터셋

| 파일 | 조인 키 | 특이사항 |
|------|---------|---------|
| `user_profile_clean.parquet` | `sha2_hash` | 기준 테이블 |
| `vod_log_clean.parquet` | `sha2_hash` | 60M 행, 집계 후 사용 |
| `churn_dataset.parquet` | — | 5% 샘플, 80:20 비율, SEED=42 |
| `churn_dataset_v2.parquet` | — | 1차 + Flag_Kids·Recency 등 7변수 추가 |
| `vod_purchase_dataset.parquet` | `sha2_hash, p_mt` | 사용자·월별, RVOD 건수 예측 |

### 모델 성능 기준 (평가 에이전트 통과 기준)

| 모델 | 지표 | 통과 기준 | 현재 달성 |
|------|------|-----------|-----------|
| 해지 예측 | PRC-AUC | ≥ 0.5 | 0.5214 (XGBoost v2) |
| VOD 구매 예측 | R2 | ≥ 0.6 | 0.624 (Random Forest) |
| VOD 구매 예측 | MAE | ≤ 0.6 | 0.596 |

> 통과 기준(임계값)은 `tests/test_model_performance.py` 와 `.github/agents/eval-agent.agent.md` 에 동일하게 정의되어 있다. 테스트는 하드코딩이 아니라 실제 데이터 재학습 또는 파이프라인 산출 CSV로 검증한다.

## 코드 규칙

- **샘플링**: `SEED=42` 고정 (재현성)
- **경로**: 스크립트 내 `BASE = Path(__file__).resolve().parent.parent.parent` 패턴 사용 — 하드코딩 금지
- **출력**: `output/{태스크명}/` 하위에 CSV·PNG 저장
- **한글 폰트**: matplotlib에서 `NanumGothic` 또는 시스템 한글 폰트 설정 필요
- **DuckDB**: `:memory:` 연결로 대용량 Parquet 집계, 파일 기반 DB는 `data/lghellovision.duckdb`
- **데이터 타입 변환**: `data/interim/*_schema.json`, `*_categories.json` 기반

## 에이전트 아키텍처 (.github/)

| 파일 | 역할 |
|------|------|
| `agents/data-scientist.agent.md` | 피처 엔지니어링·변수 선정 |
| `agents/ml-ops.agent.md` | 서빙 환경 최적화 |
| `agents/eval-agent.agent.md` | 모델 출력 품질 자동 검증 |
| `agents/control-agent.agent.md` | 흐름·라우팅·오케스트레이션 |
| `instructions/harness.md` | 에이전트 실행·관리 환경 |
| `prompts/eda-report.prompt.md` | EDA 리포트 자동화 프롬프트 |
| `prompts/model-arch.prompt.md` | 모델 아키텍처 초기화 프롬프트 |

TDD 사이클(`docs/TDD-ml-cycle.md`): Red(지표 미달) → Green(목표 달성) → Refactor(최적화). 지표 미달 시 `docs/HUMAN_IN_THE_LOOP.md` 기준으로 사람 개입 트리거.
