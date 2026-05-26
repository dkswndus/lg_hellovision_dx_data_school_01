"""
모델 학습 및 저장 파이프라인
- Churn XGBoost v2  → models/churn_model.joblib
- VOD Random Forest → models/vod_model.joblib

실행: python scripts/serving/train_and_save.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE / "models"
RANDOM_STATE = 42

CHURN_DATA = BASE / "data" / "processed" / "churn_dataset_v2.parquet"
CHURN_EXCLUDE = {"sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN"}

VOD_DATA = BASE / "data" / "processed" / "vod_purchase_dataset.parquet"
VOD_EXCLUDE = {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"}


def _col_types(df: pd.DataFrame, exclude: set) -> tuple[list, list]:
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            if df[col].nunique() > 20:
                numeric.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _prepare(df: pd.DataFrame, numeric: list, categorical: list, target: str):
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df[target]
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


def train_churn() -> float:
    print("=== Churn 모델 학습 (XGBoost v2) ===")
    if not CHURN_DATA.exists():
        raise FileNotFoundError(f"데이터 없음: {CHURN_DATA}\nbuild_churn_dataset_v2.py 먼저 실행하세요.")

    df = pd.read_parquet(CHURN_DATA)
    sample = df.sample(n=min(100_000, len(df)), random_state=RANDOM_STATE)
    numeric, categorical = _col_types(sample, CHURN_EXCLUDE)
    X, y, cols = _prepare(sample, numeric, categorical, "cancel_yn")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE, eval_metric="logloss"
        )
    else:
        print("  XGBoost 없음 → RandomForest 대체")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)

    model.fit(X_train, y_train)
    prc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  Val PRC-AUC: {prc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "churn_scaler.joblib")
    (MODEL_DIR / "churn_features.json").write_text(
        json.dumps({"features": cols, "numeric": numeric, "categorical": categorical}),
        encoding="utf-8",
    )
    print(f"  저장: {MODEL_DIR / 'churn_model.joblib'}")
    return prc


def train_vod() -> tuple[float, float]:
    print("=== VOD 구매 예측 모델 학습 (Random Forest) ===")
    if not VOD_DATA.exists():
        raise FileNotFoundError(f"데이터 없음: {VOD_DATA}\nbuild_vod_purchase_dataset.py 먼저 실행하세요.")

    df = pd.read_parquet(VOD_DATA)
    sample = df.sample(n=min(100_000, len(df)), random_state=RANDOM_STATE)
    numeric, categorical = _col_types(sample, VOD_EXCLUDE)
    X, y, cols = _prepare(sample, numeric, categorical, "vod_purchase_cnt")
    y = y.astype(float)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    pred = np.maximum(model.predict(X_val), 0)
    r2 = r2_score(y_val, pred)
    mae = mean_absolute_error(y_val, pred)
    print(f"  Val R2: {r2:.4f}, MAE: {mae:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "vod_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "vod_scaler.joblib")
    (MODEL_DIR / "vod_features.json").write_text(
        json.dumps({"features": cols, "numeric": numeric, "categorical": categorical}),
        encoding="utf-8",
    )
    print(f"  저장: {MODEL_DIR / 'vod_model.joblib'}")
    return r2, mae


if __name__ == "__main__":
    prc = train_churn()
    r2, mae = train_vod()
    print(f"\n=== 완료 ===")
    print(f"  Churn  PRC-AUC : {prc:.4f}  (목표 ≥ 0.50)")
    print(f"  VOD    R2      : {r2:.4f}  (목표 ≥ 0.60)")
    print(f"  VOD    MAE     : {mae:.4f}  (목표 ≤ 0.60)")
