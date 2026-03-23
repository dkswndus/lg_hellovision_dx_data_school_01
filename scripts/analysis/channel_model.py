"""
채널 시청 예측 회귀 모델
- 종속변수: CH_HH_AVG_MONTH1
- 독립변수: log_VOD_TIME, Adult, FOD, Trend, INHOME_RATE, AGE_GRP10
"""
import json
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "channel_model_dataset.parquet"
CHURN_PATH = BASE / "data" / "processed" / "churn_dataset.parquet"
OUTPUT_DIR = BASE / "output" / "channel_model"
RANDOM_STATE = 42
MODEL_SAMPLE = 100_000

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


def build_dataset():
    """channel_model_dataset 또는 churn_dataset에서 변수 로드"""
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        feature_cols = ["log_VOD_TIME", "Adult", "FOD", "Trend", "INHOME_RATE", "AGE_GRP10"]
        X = df[feature_cols]
        y = df["y"]
        return X, y, feature_cols

    if not CHURN_PATH.exists():
        raise FileNotFoundError("churn_dataset.parquet 없음. build_churn_dataset.py 실행 필요")

    df = pd.read_parquet(CHURN_PATH)
    df = df[df["p_mt"].between(202301, 202312)].dropna(subset=["CH_HH_AVG_MONTH1"])
    df["y"] = df["CH_HH_AVG_MONTH1"].astype(float)
    df["log_VOD_TIME"] = np.log1p(df["vod_use_tms_sum"].fillna(0))
    df["INHOME_RATE"] = pd.to_numeric(df["INHOME_RATE"], errors="coerce").fillna(0)
    for c in ["Adult", "FOD", "Trend"]:
        df[c] = 0.0
    feature_cols = ["log_VOD_TIME", "Adult", "FOD", "Trend", "INHOME_RATE", "AGE_GRP10"]
    X = df[feature_cols].copy()
    X["AGE_GRP10"] = pd.to_numeric(X["AGE_GRP10"], errors="coerce").fillna(5)
    y = df["y"]
    return X, y, feature_cols


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("데이터 로드 및 변수 생성...")
    X, y, feature_cols = build_dataset()

    # 샘플
    n = min(MODEL_SAMPLE, len(X))
    idx = X.sample(n=n, random_state=RANDOM_STATE).index
    X, y = X.loc[idx], y.loc[idx]
    X = X.dropna()

    y = y.loc[X.index]
    print(f"학습 데이터: {len(X):,}행")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    }
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results.append({
            "모델": name,
            "R2": r2_score(y_val, y_pred),
            "MAE": mean_absolute_error(y_val, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
        })
        print(f"  {name}: R2={results[-1]['R2']:.4f}, MAE={results[-1]['MAE']:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_DIR / 'model_results.csv'}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(results))
    ax.bar(x_pos - 0.2, [r["R2"] for r in results], 0.4, label="R2", color="#3498db")
    ax.bar(x_pos + 0.2, [r["MAE"] for r in results], 0.4, label="MAE", color="#e74c3c")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["모델"] for r in results], rotation=30, ha="right")
    ax.set_ylabel("점수")
    ax.set_title("채널 시청(CH_HH_AVG_MONTH1) 예측 모델 비교")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
