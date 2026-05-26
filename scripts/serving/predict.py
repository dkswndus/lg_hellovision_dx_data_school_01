"""
서빙 추론 모듈
- 저장된 모델(models/)을 lazy-load하여 단건 예측
- 대시보드·API 공통으로 사용

사용 예:
    from scripts.serving.predict import models_available, predict_churn, predict_vod
    if models_available():
        risk = predict_churn(row_dict)
        cnt  = predict_vod(row_dict)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE / "models"

_cache: dict = {}


def models_available() -> bool:
    return (MODEL_DIR / "churn_model.joblib").exists() and (MODEL_DIR / "vod_model.joblib").exists()


def _load(key: str, filename: str):
    if key not in _cache:
        import joblib
        _cache[key] = joblib.load(MODEL_DIR / filename)
    return _cache[key]


def _load_meta(filename: str) -> dict:
    if filename not in _cache:
        _cache[filename] = json.loads((MODEL_DIR / filename).read_text(encoding="utf-8"))
    return _cache[filename]


def _build_df(features: dict, meta: dict) -> pd.DataFrame:
    cols = meta["features"]
    categorical = set(meta["categorical"])
    row = {}
    for c in cols:
        val = features.get(c, 0)
        if val is None:
            val = 0
        row[c] = int(val) if c in categorical else float(val)
    return pd.DataFrame([row], columns=cols)


def predict_churn(features: dict) -> float:
    """반환값: 이탈 확률 0–100 (%)"""
    model = _load("churn_model", "churn_model.joblib")
    scaler = _load("churn_scaler", "churn_scaler.joblib")
    meta = _load_meta("churn_features.json")
    df = _build_df(features, meta)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    proba = model.predict_proba(df_scaled)[0][1]
    return float(np.clip(proba * 100, 5, 95))


def predict_vod(features: dict) -> float:
    """반환값: 예측 VOD 구매 건수 (≥ 0)"""
    model = _load("vod_model", "vod_model.joblib")
    scaler = _load("vod_scaler", "vod_scaler.joblib")
    meta = _load_meta("vod_features.json")
    df = _build_df(features, meta)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    pred = model.predict(df_scaled)[0]
    return float(max(pred, 0))
