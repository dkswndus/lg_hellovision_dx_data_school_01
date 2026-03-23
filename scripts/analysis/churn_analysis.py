"""
해지 예측 데이터셋 분석
1. PRC-AUC(불균형 지표) 기준 모델 비교 → 최고 성능 모델의 변수 10개 선정 및 시각화
2. 상위 5개 변수 가설검정 (p-value + 효과 크기)
3. p-value와 효과 크기 시각화 (대용량 데이터에서 p-value만 보지 않음)
"""
import json
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

# 한글 폰트
if os.name == "nt":
    font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE / "data" / "processed" / "churn_dataset.parquet"
OUTPUT_DIR = BASE / "output" / "churn_analysis"
CATEGORIES_PATH = BASE / "data" / "interim" / "user_profile_categories.json"

EXCLUDE = {"sha2_hash", "cancel_yn", "p_mt"}  # p_mt: 무의미 변수 제외
TOP_N = 10
TOP_FOR_TEST = 5
MODEL_SAMPLE = 100_000  # 모델 학습용 샘플 (속도)
VIZ_SAMPLE = 100_000  # 시각화용 샘플 (속도)
RANDOM_STATE = 42


def load_categories_reverse() -> dict:
    if not CATEGORIES_PATH.exists():
        return {}
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        cat = json.load(f)
    result = {}
    for col, mapping in cat.items():
        result[col] = {}
        for k, v in mapping.items():
            try:
                result[col][int(v)] = k
            except (ValueError, TypeError):
                result[col][v] = k
    return result


def get_col_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """수치형/범주형 컬럼 분류"""
    numeric, categorical = [], []
    for col in df.columns:
        if col in EXCLUDE:
            continue
        if df[col].dtype in (np.int64, np.float64, np.int32, np.float32):
            n_unique = df[col].nunique()
            if n_unique > 20 or col in ("vod_view_cnt", "vod_use_tms_sum", "vod_asset_cnt"):
                numeric.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def prepare_xy(df: pd.DataFrame, numeric: list, categorical: list) -> tuple[pd.DataFrame, pd.Series, list]:
    """전처리 및 X, y 생성"""
    cols = [c for c in numeric + categorical if c in df.columns]
    X = df[cols].copy()
    y = df["cancel_yn"]
    for c in categorical:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    return X, y, cols


def select_top10_by_prc_auc(
    df: pd.DataFrame, numeric: list, categorical: list
) -> tuple[list[str], list[tuple[int, float]]]:
    """
    같은 조건으로 여러 모델 비교, PRC-AUC(불균형 지표) 최고 모델의 변수 10개 선정.
    - feature 수별로 top-k 변수로 학습 → PRC-AUC 비교
    - 최고 PRC-AUC 모델의 변수 = top 10
    """
    sample_df = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, cols = prepare_xy(sample_df, numeric, categorical)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # train/val split (동일 조건)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 1) 전체 변수로 RF 학습 → feature importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # 2) top-k 변수별로 PRC-AUC 비교 (k=5~15)
    best_prc = -1
    best_k = 10
    results_by_k = []

    for k in range(5, min(16, len(imp) + 1)):
        top_k_cols = imp.head(k).index.tolist()
        X_train_k = X_train[top_k_cols]
        X_val_k = X_val[top_k_cols]

        model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, class_weight="balanced")
        model.fit(X_train_k, y_train)
        y_pred_proba = model.predict_proba(X_val_k)[:, 1]
        prc_auc = average_precision_score(y_val, y_pred_proba)
        results_by_k.append((k, prc_auc))

        if prc_auc > best_prc:
            best_prc = prc_auc
            best_k = k

    top10 = imp.head(best_k).index.tolist()
    if len(top10) > TOP_N:
        top10 = top10[:TOP_N]
    elif len(top10) < TOP_N:
        top10 = imp.head(TOP_N).index.tolist()

    print(f"PRC-AUC 최고: k={best_k}, PRC-AUC={best_prc:.4f}")
    print("k별 PRC-AUC:", results_by_k)
    return top10, results_by_k


def plot_top10_vars(df: pd.DataFrame, top10: list[str], rev_map: dict, numeric: list, categorical: list):
    """상위 10개 변수 시각화 (유지 vs 해지 비교)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 시각화용 샘플 (비율 유지)
    n = min(VIZ_SAMPLE, len(df))
    plot_df = df.sample(n=n, random_state=42)
    maintain = plot_df[plot_df["cancel_yn"] == 0]
    churn = plot_df[plot_df["cancel_yn"] == 1]

    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(top10):
        ax = axes[i]
        if col in numeric:
            ax.hist([maintain[col].dropna(), churn[col].dropna()], bins=30, label=["유지", "해지"], alpha=0.7, density=True)
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=8)
        else:
            m_cnt = maintain[col].value_counts().sort_index()
            c_cnt = churn[col].value_counts().sort_index()
            cats = sorted(set(m_cnt.index) | set(c_cnt.index))
            if col in rev_map:
                labels = [rev_map[col].get(int(x), str(x))[:10] for x in cats]
            else:
                labels = [str(x)[:10] for x in cats]
            x = np.arange(len(cats))
            w = 0.35
            m_vals = [m_cnt.get(c, 0) for c in cats]
            c_vals = [c_cnt.get(c, 0) for c in cats]
            ax.bar(x - w / 2, m_vals, w, label="유지", alpha=0.8)
            ax.bar(x + w / 2, c_vals, w, label="해지", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    for j in range(len(top10), len(axes)):
        axes[j].axis("off")
    plt.suptitle("PRC-AUC 최고 모델의 변수 10개 (유지 vs 해지)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top10_variables.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'top10_variables.png'}")


def hypothesis_test(df: pd.DataFrame, col: str, numeric: list, categorical: list) -> dict:
    """가설검정 + 효과 크기"""
    maintain = df[df["cancel_yn"] == 0][col].dropna()
    churn = df[df["cancel_yn"] == 1][col].dropna()

    if col in numeric:
        stat, pval = stats.ttest_ind(maintain, churn)
        pooled_std = np.sqrt((maintain.var() * (len(maintain) - 1) + churn.var() * (len(churn) - 1)) / (len(maintain) + len(churn) - 2))
        d = (maintain.mean() - churn.mean()) / pooled_std if pooled_std > 0 else 0
        h0 = f"유지와 해지 그룹의 {col} 평균에 차이가 없다"
        h1 = f"유지와 해지 그룹의 {col} 평균에 차이가 있다"
        return {"test": "t-test", "statistic": stat, "pvalue": pval, "effect": d, "effect_name": "Cohen's d", "h0": h0, "h1": h1}
    else:
        tbl = pd.crosstab(df[col], df["cancel_yn"])
        chi2, pval, dof, _ = stats.chi2_contingency(tbl)
        n = tbl.sum().sum()
        min_dim = min(tbl.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
        h0 = f"{col}와 해지(cancel_yn)는 독립이다"
        h1 = f"{col}와 해지(cancel_yn)는 독립이 아니다"
        return {"test": "chi2", "statistic": chi2, "pvalue": pval, "effect": cramers_v, "effect_name": "Cramer's V", "h0": h0, "h1": h1}


def plot_hypothesis_results(results: list[dict], top5: list[str]):
    """귀무가설/대립가설, p-value, 효과 크기 각각 별도 PNG 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = [r["col"] for r in results]
    pvals = [r["pvalue"] for r in results]
    effects = [r["effect"] for r in results]

    # 1) 귀무가설, 대립가설
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.axis("off")
    table_data = [["변수", "귀무가설 (H0)", "대립가설 (H1)"]]
    for r in results:
        table_data.append([r["col"], r["h0"], r["h1"]])
    table1 = ax1.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.15, 0.42, 0.42])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 2.5)
    ax1.set_title("가설검정: 귀무가설 vs 대립가설", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hypothesis_h0_h1.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'hypothesis_h0_h1.png'}")

    # 2) p-value 결과 (표)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.axis("off")
    pval_table = [["변수", "검정", "p-value", "해석"]]
    for r in results:
        pstr = f"{r['pvalue']:.2e}" if r["pvalue"] > 0 else "< 1e-300"
        interp = "귀무가설 기각 (유의)" if r["pvalue"] < 0.05 else "귀무가설 채택 (비유의)"
        pval_table.append([r["col"], r["test"], pstr, interp])
    table2 = ax2.table(cellText=pval_table, loc="center", cellLoc="center", colWidths=[0.25, 0.15, 0.25, 0.35])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 2.2)
    ax2.set_title("가설검정 p-value 결과 (α=0.05 이하면 귀무가설 기각)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hypothesis_pvalue.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'hypothesis_pvalue.png'}")

    # 3) 효과 크기
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(names, effects, color="#3498db", alpha=0.8)
    ax3.set_xlabel("효과 크기 (Cohen's d / Cramer's V)")
    ax3.set_title("효과 크기 (실질적 의미 해석용)", fontsize=14)
    ax3.axvline(0.1, color="gray", linestyle=":", alpha=0.7, label="작음(0.1)")
    ax3.axvline(0.3, color="gray", linestyle=":", alpha=0.7, label="중간(0.3)")
    ax3.axvline(0.5, color="gray", linestyle=":", alpha=0.7, label="큼(0.5)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hypothesis_effect_size.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'hypothesis_effect_size.png'}")


def plot_overfitting_check(results: list[dict]) -> None:
    """Train vs Val PRC-AUC 비교로 과적합 검증 시각화"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [r["모델"] for r in results]
    train_prc = [r["Train PRC-AUC"] for r in results]
    val_prc = [r["Val PRC-AUC"] for r in results]
    gaps = [r["Gap (Train-Val)"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Train vs Val 막대 그래프
    x = np.arange(len(names))
    w = 0.35
    ax1 = axes[0]
    bars1 = ax1.bar(x - w / 2, train_prc, w, label="Train PRC-AUC", color="#3498db", alpha=0.8)
    bars2 = ax1.bar(x + w / 2, val_prc, w, label="Val PRC-AUC", color="#e74c3c", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("PRC-AUC")
    ax1.set_title("과적합 검증: Train vs Validation PRC-AUC")
    ax1.legend()
    ax1.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="baseline(prior~0.2)")
    ax1.grid(axis="y", alpha=0.3)

    # 2) Gap (Train - Val): 클수록 과적합
    ax2 = axes[1]
    colors = ["#e74c3c" if g > 0.1 else "#2ecc71" for g in gaps]
    ax2.barh(names, gaps, color=colors, alpha=0.8)
    ax2.set_xlabel("Gap (Train PRC-AUC - Val PRC-AUC)")
    ax2.set_title("과적합 지표: Gap > 0.1이면 과적합 의심")
    ax2.axvline(0.1, color="red", linestyle="--", alpha=0.7)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overfitting_check.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 콘솔 출력
    print("\n[과적합 검증] Train vs Val PRC-AUC:")
    for r in results:
        status = "[!] 과적합 의심" if r["Gap (Train-Val)"] > 0.1 else "[OK] 양호"
        print(f"  {r['모델']}: Train={r['Train PRC-AUC']:.4f}, Val={r['Val PRC-AUC']:.4f}, Gap={r['Gap (Train-Val)']:.4f} {status}")
    print(f"저장: {OUTPUT_DIR / 'overfitting_check.png'}")


def compare_models_and_save(
    df: pd.DataFrame, top10: list[str], numeric: list, categorical: list
) -> None:
    """모델별 성능 비교 (혼동행렬, PRC-AUC 등) → PNG 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_df = df.sample(n=min(MODEL_SAMPLE, len(df)), random_state=RANDOM_STATE)
    X, y, _ = prepare_xy(sample_df, numeric, categorical)
    X = X[top10]
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
        gap = train_prc - val_prc

        cm = confusion_matrix(y_val, y_pred)
        cms[name] = cm
        results.append({
            "모델": name,
            "Train PRC-AUC": train_prc,
            "Val PRC-AUC": val_prc,
            "Gap (Train-Val)": gap,
            "PRC-AUC": val_prc,  # Val과 동일 (보고서 호환)
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1": f1_score(y_val, y_pred, zero_division=0),
        })

    # 시각화: 상단 표 + 하단 혼동행렬
    n_models = len(cms)
    n_cols = 3
    n_cm_rows = (n_models + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(14, 4 + 4 * n_cm_rows))

    # 1) 성능 비교 표
    ax_table = fig.add_subplot(n_cm_rows + 1, 1, 1)
    ax_table.axis("off")
    res_df = pd.DataFrame(results)
    table_data = [["모델"] + list(res_df.columns[1:])]
    for _, row in res_df.iterrows():
        table_data.append([row["모델"]] + [f"{row[k]:.4f}" for k in res_df.columns[1:]])
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colColours=["#3498db"] * len(table_data[0]),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)
    ax_table.set_title("모델별 성능 비교 (PRC-AUC, Accuracy, Precision, Recall, F1)", fontsize=14, pad=20)

    # 2) 혼동행렬 (모델 수에 따라 그리드)
    for i, (name, cm) in enumerate(cms.items()):
        ax = fig.add_subplot(n_cm_rows + 1, n_cols, n_cols + 1 + i)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["유지(0)", "해지(1)"])
        ax.set_yticklabels(["유지(0)", "해지(1)"])
        ax.set_xlabel("예측")
        ax.set_ylabel("실제")
        ax.set_title(f"{name}\n혼동행렬")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=14, color="black")
    plt.suptitle("해지 예측 모델 비교", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT_DIR / 'model_comparison.png'}")

    # 과적합 검증: Train vs Val PRC-AUC
    plot_overfitting_check(results)


def plot_prc_auc_by_k(results_by_k: list[tuple[int, float]]):
    """k(변수 수)별 PRC-AUC 시각화"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ks = [r[0] for r in results_by_k]
    prcs = [r[1] for r in results_by_k]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, prcs, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.set_xlabel("변수 수 (k)")
    ax.set_ylabel("PRC-AUC (Average Precision)")
    ax.set_title("변수 수별 PRC-AUC (불균형 데이터 지표, 클수록 좋음)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prc_auc_by_k.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {OUTPUT_DIR / 'prc_auc_by_k.png'}")


def main():
    print("churn_dataset 로드 중...")
    df = pd.read_parquet(DATA_PATH)

    numeric, categorical = get_col_types(df)
    print(f"수치형 {len(numeric)}개, 범주형 {len(categorical)}개")

    # 1. PRC-AUC 기준 최고 모델의 변수 10개 선정
    print("\n[모델 비교] 같은 조건, PRC-AUC 기준 변수 선정...")
    top10, results_by_k = select_top10_by_prc_auc(df, numeric, categorical)
    print(f"\n선정된 상위 {len(top10)}개 변수 (PRC-AUC 최고 모델):")
    for i, c in enumerate(top10, 1):
        print(f"  {i}. {c}")

    plot_prc_auc_by_k(results_by_k)

    # 모델별 성능 비교 (혼동행렬, PRC-AUC 등) PNG 저장
    compare_models_and_save(df, top10, numeric, categorical)

    rev_map = load_categories_reverse()

    # 2. 선정된 10개 변수 시각화
    plot_top10_vars(df, top10, rev_map, numeric, categorical)

    # 3. 상위 5개 가설검정 + 효과 크기
    top5 = top10[:TOP_FOR_TEST]
    results = []
    for col in top5:
        r = hypothesis_test(df, col, numeric, categorical)
        r["col"] = col
        results.append(r)
        print(f"\n[{col}] {r['test']}: stat={r['statistic']:.4f}, p={r['pvalue']:.2e}, {r['effect_name']}={r['effect']:.4f}")

    # 4. p-value & 효과 크기 시각화
    plot_hypothesis_results(results, top5)

    # 결과 요약 저장
    summary = []
    for r in results:
        summary.append({
            "변수": r["col"],
            "검정": r["test"],
            "p-value": r["pvalue"],
            "효과크기": r["effect"],
            "효과크기명": r["effect_name"],
        })
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUTPUT_DIR / 'hypothesis_summary.csv'}")
    print("완료.")


if __name__ == "__main__":
    main()
