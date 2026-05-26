"""
Model Performance Validation Tests (TDD Red/Green Cycle)

하드코딩된 예시값 대신 **실제 결과**로 검증한다. 검증 소스 우선순위:
  1) data/processed/*.parquet 존재  → 파이프라인과 동일하게 재학습 후 지표 계산
  2) output/.../model_comparison.csv → 파이프라인이 실제로 산출한 지표 로드
  3) 둘 다 없음                      → skip (거짓 통과 방지)

통과 기준(임계값)은 CLAUDE.md / .github/agents/eval-agent.agent.md 와 동일:
  - 해지 예측(분류)   PRC-AUC ≥ 0.5
  - VOD 구매(회귀)    R²      ≥ 0.6
  - VOD 구매(회귀)    MAE     ≤ 0.6
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
OUTPUT = BASE / "output"
SCRIPTS = BASE / "scripts" / "analysis"

# 통과 기준 (eval-agent.agent.md 와 동일)
THRESH_PRC_AUC = 0.5
THRESH_R2 = 0.6
THRESH_MAE = 0.6

# 파이프라인 재학습 시 동일 조건 (run_churn_v2.py / vod_purchase_model.py)
RANDOM_STATE = 42
MODEL_SAMPLE = 100_000


def _load_module(path: Path):
    """스크립트 파일을 모듈로 로드해 전처리 헬퍼(get_col_types/prepare_xy)를 재사용."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────── 해지 예측 (XGBoost) ────────────────────────────
def _churn_from_training() -> float | None:
    data = PROCESSED / "churn_dataset_v2.parquet"
    if not data.exists():
        return None
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    mod = _load_module(SCRIPTS / "run_churn_v2.py")
    df = pd.read_parquet(data)
    numeric, categorical = mod.get_col_types(df)
    sample = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, _ = mod.prepare_xy(sample, numeric, categorical)
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE, eval_metric="logloss"
    )
    model.fit(X_tr, y_tr)
    return float(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))


def _churn_from_output() -> float | None:
    csv = OUTPUT / "churn_analysis" / "churn_v2_model_comparison.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, encoding="utf-8-sig")
    row = df[df["모델"] == "XGBoost"]
    return None if row.empty else float(row["Val PRC-AUC"].iloc[0])


@functools.lru_cache(maxsize=1)
def _churn_prc_auc() -> tuple[float, str] | None:
    prc = _churn_from_training()
    if prc is not None:
        return prc, "재학습(parquet)"
    prc = _churn_from_output()
    if prc is not None:
        return prc, "파이프라인 산출 CSV"
    return None


# ──────────────────────────── VOD 구매 (Random Forest) ───────────────────────
def _vod_from_training() -> tuple[float, float] | None:
    data = PROCESSED / "vod_purchase_dataset.parquet"
    if not data.exists():
        return None
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    mod = _load_module(SCRIPTS / "vod_purchase_model.py")
    df = pd.read_parquet(data)
    numeric, categorical = mod.get_col_types(df)
    sample = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, _ = mod.prepare_xy(sample, numeric, categorical)
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    pred = np.maximum(model.predict(X_va), 0)  # 음수 예측 방지 (파이프라인과 동일)
    return float(r2_score(y_va, pred)), float(mean_absolute_error(y_va, pred))


def _vod_from_output() -> tuple[float, float] | None:
    csv = OUTPUT / "vod_purchase_model" / "model_comparison.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, encoding="utf-8-sig")
    row = df[df["모델"] == "Random Forest"]
    return None if row.empty else (float(row["R2"].iloc[0]), float(row["MAE"].iloc[0]))


@functools.lru_cache(maxsize=1)
def _vod_metrics() -> tuple[float, float, str] | None:
    m = _vod_from_training()
    if m is not None:
        return m[0], m[1], "재학습(parquet)"
    m = _vod_from_output()
    if m is not None:
        return m[0], m[1], "파이프라인 산출 CSV"
    return None


# ──────────────────────────────── 테스트 ─────────────────────────────────────
class TestChurnModel:
    """해지 예측 모델 성능: PRC-AUC ≥ 0.5"""

    def test_churn_prc_auc_threshold(self):
        res = _churn_prc_auc()
        if res is None:
            pytest.skip(
                "churn_dataset_v2.parquet / churn_v2_model_comparison.csv 모두 없음 "
                "— 데이터셋 빌드 또는 run_churn_v2.py 실행 필요"
            )
        prc, source = res
        assert prc >= THRESH_PRC_AUC, (
            f"[{source}] Churn PRC-AUC {prc:.4f} < {THRESH_PRC_AUC} 기준 미달"
        )


class TestVODPurchaseModel:
    """VOD 구매 예측 모델 성능: R² ≥ 0.6, MAE ≤ 0.6"""

    def test_vod_r2_threshold(self):
        res = _vod_metrics()
        if res is None:
            pytest.skip(
                "vod_purchase_dataset.parquet / model_comparison.csv 모두 없음 "
                "— 데이터셋 빌드 또는 vod_purchase_model.py 실행 필요"
            )
        r2, _mae, source = res
        assert r2 >= THRESH_R2, f"[{source}] VOD R² {r2:.4f} < {THRESH_R2} 기준 미달"

    def test_vod_mae_threshold(self):
        res = _vod_metrics()
        if res is None:
            pytest.skip(
                "vod_purchase_dataset.parquet / model_comparison.csv 모두 없음 "
                "— 데이터셋 빌드 또는 vod_purchase_model.py 실행 필요"
            )
        _r2, mae, source = res
        assert mae <= THRESH_MAE, f"[{source}] VOD MAE {mae:.4f} > {THRESH_MAE} 기준 초과"
