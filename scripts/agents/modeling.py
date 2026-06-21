"""Modeling Agent — Baseline + RandomForest·LightGBM·XGBoost·CatBoost 학습·Optuna 튜닝·leaderboard 산출.

.github/agents/modeling-agent.agent.md 스펙(LGBM/XGB/CatBoost) + 기존 analysis 스크립트의
RandomForest(run_churn_v2.py, vod_purchase_model.py 에서 실제 챔피언이었던 모델)를 포함한
5개 모델을 모델별 별도 프로세스(joblib)로 병렬 학습하고, 모델별 산출물과
output/models/leaderboard.csv 를 생성한다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"
OUTPUT_MODELS = BASE / "output" / "models"

RANDOM_STATE = 42
MODEL_SAMPLE = 100_000  # 최종 학습/검증 샘플 (run_churn_v2.py / vod_purchase_model.py 와 동일 기준)
TUNE_SAMPLE = 30_000  # Optuna 튜닝용 서브샘플 (속도) — 최종 fit 은 전체 학습셋 사용
CV_FOLDS = 5

DATASETS = {
    "churn": PROCESSED / "churn_dataset_v2.parquet",
    "vod_purchase": PROCESSED / "vod_purchase_dataset.parquet",
}
TARGET_COL = {"churn": "cancel_yn", "vod_purchase": "vod_purchase_cnt"}
EXCLUDE_COLS = {
    # VOC_STOP_CANCEL_MONTH1_YN/AGMT_END_YMD/AGMT_END_SEG: leakage_checker.py 가 critical 로 탐지
    # (해지상담·계약종료 신호는 라벨링 시점 이후/근접 시점에 갱신될 위험 — 타깃 누수 의심)
    "churn": {
        "sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN",
        "VOC_STOP_CANCEL_MONTH1_YN", "AGMT_END_YMD", "AGMT_END_SEG",
    },
    "vod_purchase": {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"},
}

MODEL_NAMES = ["baseline", "random_forest", "lightgbm", "xgboost", "catboost"]

SEARCH_SPACE = {
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 300),
        "max_depth": t.suggest_int("max_depth", 8, 15),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 4),
        "max_features": t.suggest_float("max_features", 0.7, 1.0),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 350),
        "max_depth": t.suggest_int("max_depth", 3, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": t.suggest_int("num_leaves", 30, 100),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 350),
        "max_depth": t.suggest_int("max_depth", 3, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 0.95),
    },
    "catboost": lambda t: {
        "iterations": t.suggest_int("iterations", 100, 350),
        "depth": t.suggest_int("depth", 3, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
    },
}


# ──────────────────────────── 전처리 (기존 analysis 스크립트와 동일 규칙) ────────────────────────────


def get_col_types(df: pd.DataFrame, task: str) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    exclude = EXCLUDE_COLS[task]
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            (numeric if df[col].nunique() > 20 else categorical).append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def prepare_xy(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], task: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET_COL[task]]
    y = y.astype(float) if task == "vod_purchase" else y
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


# ──────────────────────────── 모델 팩토리 ────────────────────────────


def _make_model(model_name: str, task: str, params: dict):
    if task == "churn":
        if model_name == "random_forest":
            return RandomForestClassifier(random_state=RANDOM_STATE, **params)
        if model_name == "lightgbm":
            return lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params)
        if model_name == "xgboost":
            return xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **params)
        if model_name == "catboost":
            return cb.CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, **params)
        return LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced")
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, **params)
    if model_name == "lightgbm":
        return lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **params)
    if model_name == "xgboost":
        return xgb.XGBRegressor(random_state=RANDOM_STATE, **params)
    if model_name == "catboost":
        return cb.CatBoostRegressor(random_state=RANDOM_STATE, verbose=False, **params)
    return LinearRegression()


def _cv_split(cv, X: pd.DataFrame, y: pd.Series, task: str):
    return cv.split(X, y) if task == "churn" else cv.split(X)


def _cv_score(model_name: str, task: str, params: dict, X: pd.DataFrame, y: pd.Series, cv) -> float:
    scores = []
    for tr_idx, va_idx in _cv_split(cv, X, y, task):
        model = _make_model(model_name, task, params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        if task == "churn":
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[va_idx], proba))
        else:
            pred = np.maximum(model.predict(X.iloc[va_idx]), 0)
            scores.append(r2_score(y.iloc[va_idx], pred))
    return float(np.mean(scores))


def _cv_fold_scores(model_name: str, task: str, params: dict, X: pd.DataFrame, y: pd.Series, cv) -> list[dict]:
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(_cv_split(cv, X, y, task), start=1):
        model = _make_model(model_name, task, params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        if task == "churn":
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            rows.append({"fold": fold, "prc_auc": average_precision_score(y.iloc[va_idx], proba)})
        else:
            pred = np.maximum(model.predict(X.iloc[va_idx]), 0)
            rows.append({
                "fold": fold,
                "r2": r2_score(y.iloc[va_idx], pred),
                "mae": mean_absolute_error(y.iloc[va_idx], pred),
            })
    return rows


def _tune(model_name: str, task: str, X: pd.DataFrame, y: pd.Series, cv, n_trials: int, timeout: int | None) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACE[model_name](trial)
        return _cv_score(model_name, task, params, X, y, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def _precision_at_k(y_true: pd.Series, scores: np.ndarray, k: float) -> float:
    n = max(1, int(len(scores) * k))
    order = np.argsort(scores)[::-1][:n]
    return float(np.asarray(y_true)[order].mean())


def _evaluate(model, task: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    if task == "churn":
        proba_tr = model.predict_proba(X_train)[:, 1]
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        base_rate = float(y_val.mean())
        prec10 = _precision_at_k(y_val, proba, 0.10)
        return {
            "Train PRC-AUC": float(average_precision_score(y_train, proba_tr)),
            "Val PRC-AUC": float(average_precision_score(y_val, proba)),
            "ROC-AUC": float(roc_auc_score(y_val, proba)),
            "F1": float(f1_score(y_val, pred, zero_division=0)),
            "Precision@10%": prec10,
            "Precision@20%": _precision_at_k(y_val, proba, 0.20),
            "Lift@10%": prec10 / base_rate if base_rate > 0 else float("nan"),
        }
    pred_tr = np.maximum(model.predict(X_train), 0)
    pred = np.maximum(model.predict(X_val), 0)
    return {
        "Train R2": float(r2_score(y_train, pred_tr)),
        "Val R2": float(r2_score(y_val, pred)),
        "MAE": float(mean_absolute_error(y_val, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
    }


def _train_one(
    model_name: str,
    task: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv,
    n_trials: int,
    tune_timeout: int | None,
    run_dir: Path,
) -> dict:
    if model_name == "baseline":
        best_params: dict = {}
    else:
        best_params = _tune(model_name, task, X_tune, y_tune, cv, n_trials, tune_timeout)

    model = _make_model(model_name, task, best_params)
    model.fit(X_train, y_train)
    metrics = _evaluate(model, task, X_train, y_train, X_val, y_val)

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "hyperparams.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    if model_name != "baseline":
        cv_rows = _cv_fold_scores(model_name, task, best_params, X_tune, y_tune, cv)
        pd.DataFrame(cv_rows).to_csv(model_dir / "cv_folds.csv", index=False)

    print(f"  [{model_name}] {metrics}")
    return {"모델": model_name, **metrics}


# ──────────────────────────── 엔트리포인트 ────────────────────────────


def run_modeling(
    task: Literal["churn", "vod_purchase"] = "churn",
    run_id: str | None = None,
    n_trials: int = 20,
    tune_timeout: int | None = 900,
    sample_size: int = MODEL_SAMPLE,
    n_jobs: int = 5,
    models: list[str] | None = None,
    drop_features: list[str] | None = None,
    variant: str | None = None,
) -> dict:
    """5개 모델(Baseline·RandomForest·LightGBM·XGBoost·CatBoost)을 병렬 학습하고 leaderboard.csv 를 산출한다.

    models 를 지정하면 해당 모델만 (재)학습하고, 기존 leaderboard.csv 의 나머지 모델 행은 보존한다.
    drop_features 를 지정하면 해당 컬럼들을 피처에서 제외 (파생변수 효과 ablation 용).
    variant 를 지정하면 leaderboard_{task}_{variant}.csv 로 별도 저장 (원본 비교용 등).
    """
    model_names = models or MODEL_NAMES
    data_path = DATASETS[task]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 없음 — build_*.py 로 데이터셋 생성 필요")

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.read_parquet(data_path)
    numeric, categorical = get_col_types(df, task)
    if drop_features:
        numeric = [c for c in numeric if c not in drop_features]
        categorical = [c for c in categorical if c not in drop_features]
    sample = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)
    X, y, _cols = prepare_xy(sample, numeric, categorical, task)

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    split_kwargs = {"stratify": y} if task == "churn" else {}
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, **split_kwargs
    )

    tune_idx = X_train.sample(n=min(TUNE_SAMPLE, len(X_train)), random_state=RANDOM_STATE).index
    X_tune, y_tune = X_train.loc[tune_idx], y_train.loc[tune_idx]

    cv = (
        StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if task == "churn"
        else KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    )

    run_dir = OUTPUT_MODELS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[modeling] task={task} run_id={run_id} models={model_names} train={len(X_train)} val={len(X_val)} tune={len(X_tune)}")
    jobs = (
        delayed(_train_one)(
            name, task, X_train, y_train, X_tune, y_tune, X_val, y_val, cv, n_trials, tune_timeout, run_dir
        )
        for name in model_names
    )
    results = Parallel(n_jobs=min(n_jobs, len(model_names)), backend="loky")(jobs)

    sort_col = "Val PRC-AUC" if task == "churn" else "Val R2"
    suffix = f"_{variant}" if variant else ""
    leaderboard_path = OUTPUT_MODELS / f"leaderboard_{task}{suffix}.csv"
    new_rows = pd.DataFrame(results)
    if leaderboard_path.exists():
        existing = pd.read_csv(leaderboard_path, encoding="utf-8-sig")
        existing = existing[~existing["모델"].isin(model_names)]
        leaderboard = pd.concat([existing, new_rows], ignore_index=True)
    else:
        leaderboard = new_rows
    leaderboard = leaderboard.sort_values(sort_col, ascending=False).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    print(f"저장: {leaderboard_path}")

    return {
        "run_id": run_id,
        "leaderboard_path": str(leaderboard_path.relative_to(BASE)).replace("\\", "/"),
        "winner_model": str(leaderboard.iloc[0]["모델"]),
        "winner_metric": float(leaderboard.iloc[0][sort_col]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Modeling agent — Baseline/RandomForest/LGBM/XGB/CatBoost 학습")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--tune-timeout", type=int, default=900)
    parser.add_argument("--sample-size", type=int, default=MODEL_SAMPLE)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument(
        "--models", default=None, help="재학습할 모델만 콤마로 지정 (예: random_forest). 미지정시 전체 학습"
    )
    parser.add_argument(
        "--drop-features", default=None, help="피처에서 제외할 컬럼 콤마로 지정 (파생변수 ablation 용)"
    )
    parser.add_argument(
        "--variant", default=None, help="leaderboard_{task}_{variant}.csv 로 별도 저장 (예: raw)"
    )
    args = parser.parse_args()

    result = run_modeling(
        task=args.task,
        run_id=args.run_id,
        n_trials=args.n_trials,
        tune_timeout=args.tune_timeout,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
        models=args.models.split(",") if args.models else None,
        drop_features=args.drop_features.split(",") if args.drop_features else None,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
