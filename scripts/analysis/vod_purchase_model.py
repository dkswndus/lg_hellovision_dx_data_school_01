"""
VOD 구매 건수 예측 모델 (2차 모델)
- 종속변수: vod_purchase_cnt (RVOD 시청 건수 = 유료 VOD 구매/렌탈 건수)
- 회귀 모델: R2, MAE, RMSE 기준 비교
"""
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
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "vod_purchase_dataset.parquet"
OUTPUT_DIR = BASE / "output" / "vod_purchase_model"
EXCLUDE = {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"}
MODEL_SAMPLE = 100_000
RANDOM_STATE = 42


def get_col_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in df.columns:
        if col in EXCLUDE:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            n_unique = df[col].nunique()
            if n_unique > 20 or col in ("vod_view_cnt", "vod_use_tms_sum", "vod_asset_cnt", "fod_cnt", "svod_cnt", "lagged_rvod_cnt"):
                numeric.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def prepare_xy(df: pd.DataFrame, numeric: list, categorical: list) -> tuple[pd.DataFrame, pd.Series, list]:
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df["vod_purchase_cnt"].astype(float)
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


def main():
    if not DATA_PATH.exists():
        print("vod_purchase_dataset 없음. build_vod_purchase_dataset.py 실행 필요.")
        return

    print("vod_purchase_dataset 로드 중...")
    df = pd.read_parquet(DATA_PATH)
    numeric, categorical = get_col_types(df)
    print(f"수치형 {len(numeric)}개, 범주형 {len(categorical)}개")

    sample = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, cols = prepare_xy(sample, numeric, categorical)

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
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = np.maximum(y_pred, 0)  # 음수 예측 방지
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results.append({"모델": name, "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"  {name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_DIR / 'model_comparison.csv'}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x_pos = np.arange(len(results))
    axes[0].bar(x_pos, [r["R2"] for r in results], color="#3498db", alpha=0.8)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([r["모델"] for r in results], rotation=30, ha="right")
    axes[0].set_ylabel("R2")
    axes[0].set_title("R2 (클수록 좋음)")

    axes[1].bar(x_pos, [r["MAE"] for r in results], color="#e74c3c", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([r["모델"] for r in results], rotation=30, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE (작을수록 좋음)")

    axes[2].bar(x_pos, [r["RMSE"] for r in results], color="#2ecc71", alpha=0.8)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([r["모델"] for r in results], rotation=30, ha="right")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("RMSE (작을수록 좋음)")

    plt.suptitle("VOD 구매 건수(vod_purchase_cnt) 예측 모델 비교", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'model_comparison.png'}")

    # Feature importance (최고 R2 모델)
    best_name = max(results, key=lambda r: r["R2"])["모델"]
    best_model = models[best_name]
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\n[Feature Importance] {best_name} 상위 10개:")
        for i, (col, v) in enumerate(imp.head(10).items(), 1):
            print(f"  {i:2}. {col}: {v:.4f}")

    print("완료.")


if __name__ == "__main__":
    main()
