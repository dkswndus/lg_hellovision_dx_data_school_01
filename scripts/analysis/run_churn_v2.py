"""
churn_dataset_v2로 해지 예측 모델 실행 (7개 추가 변수 사용)
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

# 한글 폰트
if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_dataset_v2.parquet"
OUTPUT_DIR = BASE / "output" / "churn_analysis"

EXCLUDE = {"sha2_hash", "cancel_yn", "p_mt"}
# vod_use_tms_sum, NFX_USE_YN 제외 (Total_Watch_Time, Is_Netflix로 대체)
EXCLUDE_COLS = EXCLUDE | {"vod_use_tms_sum", "NFX_USE_YN"}
MODEL_SAMPLE = 100_000
RANDOM_STATE = 42

NEW_VARS = ["Flag_Kids", "Flag_Movie", "Recency", "Total_Watch_Time", "Log_Watch_Time", "Is_Netflix", "Risk_Segment_Kids_NFX"]


def get_col_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            n_unique = df[col].nunique()
            if n_unique > 20 or col in ("vod_view_cnt", "Total_Watch_Time", "Log_Watch_Time", "Recency", "vod_asset_cnt"):
                numeric.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def prepare_xy(df: pd.DataFrame, numeric: list, categorical: list) -> tuple[pd.DataFrame, pd.Series, list]:
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df["cancel_yn"]
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


def main():
    if not DATA_PATH.exists():
        print("churn_dataset_v2 없음. build_churn_dataset_v2.py 실행 필요.")
        return

    print("churn_dataset_v2 로드 중...")
    df = pd.read_parquet(DATA_PATH)
    numeric, categorical = get_col_types(df)
    print(f"수치형 {len(numeric)}개, 범주형 {len(categorical)}개")
    print(f"추가 변수 7개 포함 여부: {all(v in numeric + categorical for v in NEW_VARS)}")

    sample = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, cols = prepare_xy(sample, numeric, categorical)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

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
    cms = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_proba_train = model.predict_proba(X_train)[:, 1]

        train_prc = average_precision_score(y_train, y_proba_train)
        val_prc = average_precision_score(y_val, y_proba)
        cms[name] = confusion_matrix(y_val, y_pred)

        results.append({
            "모델": name,
            "Train PRC-AUC": train_prc,
            "Val PRC-AUC": val_prc,
            "Gap (Train-Val)": train_prc - val_prc,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0),
        })
        print(f"  {name}: Val PRC-AUC = {val_prc:.4f}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "churn_v2_model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_DIR / 'churn_v2_model_comparison.csv'}")

    # 혼동행렬 시각화
    n_models = len(cms)
    n_cols = 3
    n_cm_rows = (n_models + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(14, 4 + 4 * n_cm_rows))

    ax_table = fig.add_subplot(n_cm_rows + 1, 1, 1)
    ax_table.axis("off")
    table_data = [["모델"] + list(res_df.columns[1:])]
    for _, row in res_df.iterrows():
        table_data.append([row["모델"]] + [f"{row[k]:.4f}" for k in res_df.columns[1:]])
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center", colColours=["#3498db"] * len(table_data[0]))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)
    ax_table.set_title("churn_v2 (7개 변수 추가) 모델별 성능 비교", fontsize=14, pad=20)

    for i, (name, cm) in enumerate(cms.items()):
        ax = fig.add_subplot(n_cm_rows + 1, n_cols, n_cols + 1 + i)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["유지(0)", "해지(1)"])
        ax.set_yticklabels(["유지(0)", "해지(1)"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        ax.set_title(f"{name}\n혼동행렬")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14)
    plt.suptitle("해지 예측 모델 v2 (Flag_Kids, Flag_Movie, Recency, Total/Log_Watch_Time, Is_Netflix, Risk_Segment_Kids_NFX)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_v2_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'churn_v2_model_comparison.png'}")

    # 과적합 검증
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["모델"] for r in results]
    train_prc = [r["Train PRC-AUC"] for r in results]
    val_prc = [r["Val PRC-AUC"] for r in results]
    gaps = [r["Gap (Train-Val)"] for r in results]
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, train_prc, w, label="Train", color="#3498db", alpha=0.8)
    axes[0].bar(x + w/2, val_prc, w, label="Val", color="#e74c3c", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("PRC-AUC")
    axes[0].set_title("Train vs Val PRC-AUC")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    colors = ["#e74c3c" if g > 0.1 else "#2ecc71" for g in gaps]
    axes[1].barh(names, gaps, color=colors, alpha=0.8)
    axes[1].set_xlabel("Gap")
    axes[1].set_title("과적합: Gap > 0.1 의심")
    axes[1].axvline(0.1, color="red", linestyle="--", alpha=0.7)
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_v2_overfitting_check.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'churn_v2_overfitting_check.png'}")

    # Feature importance (Random Forest)
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[Feature Importance 상위 15] (Random Forest):")
    for i, (col, v) in enumerate(imp.head(15).items(), 1):
        mark = " ★" if col in NEW_VARS else ""
        print(f"  {i:2}. {col}: {v:.4f}{mark}")

    # 7개 변수만 사용한 모델 (참고)
    new_only = [c for c in NEW_VARS if c in X.columns]
    if new_only:
        X_new = X_scaled[new_only]
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_new, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1)
        lgb_model.fit(X_tr, y_tr)
        prc_new = average_precision_score(y_va, lgb_model.predict_proba(X_va)[:, 1])
        print(f"\n[7개 변수만 사용] LightGBM Val PRC-AUC: {prc_new:.4f}")

    print("완료.")


if __name__ == "__main__":
    main()
