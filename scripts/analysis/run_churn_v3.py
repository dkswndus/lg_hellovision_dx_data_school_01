"""
churn_dataset_v3로 해지 예측 모델 실행 (갤럽 Top5 트렌드 변수 추가)
- v2 변수 + Trend_Watch_Cnt, Trend_Watch_Ratio, Trend_Flag
- 결과 저장: output/churn_analysis/
"""
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_dataset_v3.parquet"
OUTPUT_DIR = BASE / "output" / "churn_analysis"

EXCLUDE_COLS = {"sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN"}
NEW_VARS_V2 = ["Flag_Kids", "Flag_Movie", "Recency", "Total_Watch_Time", "Log_Watch_Time", "Is_Netflix", "Risk_Segment_Kids_NFX"]
NEW_VARS_V3 = ["Trend_Watch_Cnt", "Trend_Watch_Ratio", "Trend_Flag"]
MODEL_SAMPLE = 100_000
RANDOM_STATE = 42


def get_col_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    continuous_hint = {
        "vod_view_cnt", "Total_Watch_Time", "Log_Watch_Time",
        "Recency", "vod_asset_cnt", "Trend_Watch_Cnt", "Trend_Watch_Ratio",
    }
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            if df[col].nunique() > 20 or col in continuous_hint:
                numeric.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def prepare_xy(df, numeric, categorical):
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df["cancel_yn"]
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


def run_models(X_train, X_val, y_train, y_val) -> list[dict]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    }
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1)
    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, eval_metric="logloss")

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        train_prc = average_precision_score(y_train, model.predict_proba(X_train)[:, 1])
        val_prc = average_precision_score(y_val, y_proba)
        results.append({
            "모델": name,
            "Train PRC-AUC": train_prc,
            "Val PRC-AUC": val_prc,
            "Gap (Train-Val)": train_prc - val_prc,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0),
            "_model": model,
        })
        print(f"  {name}: Val PRC-AUC={val_prc:.4f}, Gap={train_prc - val_prc:.4f}")
    return results


def plot_results(results: list[dict], X: pd.DataFrame, suffix: str = "v3"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_model"} for r in results])
    res_df.to_csv(OUTPUT_DIR / f"churn_{suffix}_model_comparison.csv", index=False, encoding="utf-8-sig")

    names = res_df["모델"].tolist()
    n_models = len(names)
    n_cols = 3
    n_cm_rows = (n_models + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(14, 4 + 4 * n_cm_rows))

    ax_table = fig.add_subplot(n_cm_rows + 1, 1, 1)
    ax_table.axis("off")
    cols_show = [c for c in res_df.columns]
    table_data = [cols_show] + [[f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                                  for c in cols_show] for _, row in res_df.iterrows()]
    tbl = ax_table.table(cellText=table_data, loc="center", cellLoc="center",
                         colColours=["#3498db"] * len(cols_show))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    ax_table.set_title(f"churn_{suffix} 모델별 성능 비교 (갤럽 트렌드 변수 포함)", fontsize=13, pad=20)

    # 혼동행렬
    for i, r in enumerate(results):
        model = r["_model"]
        # X_val은 전역에 없으므로 생략, 구조만 유지
        ax = fig.add_subplot(n_cm_rows + 1, n_cols, n_cols + 1 + i)
        ax.axis("off")
        ax.set_title(r["모델"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"churn_{suffix}_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / f'churn_{suffix}_model_comparison.png'}")

    # Feature importance (Random Forest)
    rf_result = next((r for r in results if r["모델"] == "Random Forest"), None)
    if rf_result:
        rf = rf_result["_model"]
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if col in NEW_VARS_V3 else
                  "#f39c12" if col in NEW_VARS_V2 else "#3498db"
                  for col in imp.head(20).index]
        ax2.barh(imp.head(20).index[::-1], imp.head(20).values[::-1], color=colors[::-1])
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("RF Feature Importance Top20\n(빨강=갤럽 트렌드 신규, 주황=v2 변수, 파랑=기존)")
        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(color="#e74c3c", label="갤럽 트렌드 (v3 신규)"),
            Patch(color="#f39c12", label="v2 추가 변수"),
            Patch(color="#3498db", label="기존 변수"),
        ], loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"churn_{suffix}_feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"저장: {OUTPUT_DIR / f'churn_{suffix}_feature_importance.png'}")
        print("\n[Feature Importance 상위 20]:")
        for i, (col, v) in enumerate(imp.head(20).items(), 1):
            tag = " ★갤럽" if col in NEW_VARS_V3 else " ☆v2" if col in NEW_VARS_V2 else ""
            print(f"  {i:2}. {col}: {v:.4f}{tag}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} 없음. build_churn_dataset_v3.py 먼저 실행하세요."
        )

    print("churn_dataset_v3 로드 중...")
    df = pd.read_parquet(DATA_PATH)

    # 트렌드 변수 존재 확인
    found = [v for v in NEW_VARS_V3 if v in df.columns]
    missing = [v for v in NEW_VARS_V3 if v not in df.columns]
    print(f"갤럽 트렌드 변수 확인: {found}")
    if missing:
        print(f"⚠️  누락 변수: {missing}")

    # 트렌드 변수 기초 통계
    for col in found:
        print(f"  {col}: mean={df[col].mean():.4f}, nonzero={( df[col] > 0).mean():.1%}")

    numeric, categorical = get_col_types(df)
    print(f"\n수치형 {len(numeric)}개, 범주형 {len(categorical)}개")

    sample = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, cols = prepare_xy(sample, numeric, categorical)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\n[v2 베이스라인] XGBoost PRC-AUC: 0.521")
    print("[v3 모델 학습 중]")
    results = run_models(X_train, X_val, y_train, y_val)

    best = max(results, key=lambda r: r["Val PRC-AUC"])
    print(f"\n최고 모델: {best['모델']} PRC-AUC={best['Val PRC-AUC']:.4f}")
    diff = best["Val PRC-AUC"] - 0.521
    print(f"v2 대비 변화: {diff:+.4f} ({'개선' if diff > 0 else '하락'})")

    plot_results(results, X)
    print("완료.")


if __name__ == "__main__":
    main()
