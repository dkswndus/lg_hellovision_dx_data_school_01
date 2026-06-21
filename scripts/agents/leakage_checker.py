"""Leakage Checker Agent — 타깃 누수·시점 누수·중복 키 탐지 (체크포인트 1).

.github/agents/leakage-checker-agent.agent.md 스펙 구현체.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"
OUTPUT_LEAKAGE = BASE / "output" / "leakage"

DATASETS = {
    "churn": PROCESSED / "churn_dataset_v2.parquet",
    "vod_purchase": PROCESSED / "vod_purchase_dataset.parquet",
}
TARGET_COL = {"churn": "cancel_yn", "vod_purchase": "vod_purchase_cnt"}
ID_COLS = {"sha2_hash"}
EXCLUDE_BASE = {
    "churn": {"sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN"},
    "vod_purchase": {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"},
}

CORR_CRITICAL = 0.95
DOMINANCE_WARN = 0.5
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# 컬럼명에 종료/해지/중단 신호가 있으면 라벨링 시점 *이후* 갱신값일 위험이 있어 점검 대상으로 표시
# "_" 로 분리한 토큰 단위 완전일치만 검사 (예: SMS_SEND_CLS_NM 의 "SEND" 는 "END" 부분일치라 오탐 방지)
TEMPORAL_RISK_TOKENS = {"END", "CANCEL", "STOP"}


def _has_temporal_risk_token(col: str) -> bool:
    return bool(TEMPORAL_RISK_TOKENS & {tok.upper() for tok in col.split("_")})


def _encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c] if pd.api.types.is_numeric_dtype(df[c]) else df[c].astype("category").cat.codes
    return out.fillna(out.median(numeric_only=True))


def check_leakage(task: Literal["churn", "vod_purchase"]) -> dict:
    """data/processed/{dataset}.parquet 의 타깃 누수·시점 누수·중복 키를 점검한다."""
    data_path = DATASETS[task]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 없음 — build_*.py 로 데이터셋 생성 필요")

    df = pd.read_parquet(data_path)
    target = TARGET_COL[task]
    exclude = EXCLUDE_BASE[task]
    feature_cols = [c for c in df.columns if c not in exclude]

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    X = _encode(sample, feature_cols)
    y = sample[target]

    # 1. 타깃 상관 누수 (Pearson, 범주형은 코드 인코딩 후 동일 계산)
    high_corr_features = []
    for c in X.columns:
        corr = X[c].corr(y)
        if pd.notna(corr) and abs(corr) >= CORR_CRITICAL:
            high_corr_features.append({"col": c, "corr_with_target": float(corr), "severity": "critical"})

    # 2. 부분 누수 — RF 단독 top-1 importance
    rf_cls = RandomForestClassifier if task == "churn" else RandomForestRegressor
    rf = rf_cls(n_estimators=50, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_dominance = []
    if not imp.empty and imp.iloc[0] > DOMINANCE_WARN:
        feature_dominance.append({"col": str(imp.index[0]), "importance": float(imp.iloc[0]), "single_top": True})

    # 3. 시점 누수 의심 — 컬럼명 패턴 (END/CANCEL/STOP)
    temporal_leakage = [
        {
            "col": c,
            "p_mt_basis": "N/A",
            "evidence": "컬럼명에 종료/해지/중단 신호 포함 — 라벨링 시점 이후 갱신값일 위험, 검토 필요",
        }
        for c in feature_cols
        if _has_temporal_risk_token(c)
    ]

    # 4. 중복 키 — 월별 패널 데이터이므로 p_mt 존재 시 (sha2_hash, p_mt) 복합키 기준
    # 전체 목록은 수만 건일 수 있어 개수 + 샘플만 리포트에 담는다
    key_cols = ["sha2_hash", "p_mt"] if "p_mt" in df.columns else ["sha2_hash"]
    dup_counts = df.groupby(key_cols, observed=True).size()
    dup_only = dup_counts[dup_counts > 1]
    duplicate_keys = {str(k): int(v) for k, v in dup_only.head(20).items()}
    duplicate_key_count = int(len(dup_only))

    # 5. 식별자 혼입 — ID 컬럼이 피처에 섞여 있는지
    id_leak_warnings = [c for c in ID_COLS if c in feature_cols]

    # 6. p_mt 형식 점검 (YYYYMM 의 MM 이 1~12 범위 밖이면 데이터 품질 이슈)
    data_quality_issues = []
    if "p_mt" in df.columns:
        mm = df["p_mt"].astype(str).str[-2:].astype(int)
        invalid = int((~mm.between(1, 12)).sum())
        if invalid > 0:
            data_quality_issues.append({"col": "p_mt", "issue": f"YYYYMM 형식 위반(MM 1~12 밖) {invalid}건"})

    warn_count = len(feature_dominance) + len(id_leak_warnings) + len(data_quality_issues)
    if high_corr_features or temporal_leakage or duplicate_key_count > 0:
        leakage_severity = "critical"
    elif warn_count > 0:
        leakage_severity = "warn"
    else:
        leakage_severity = "none"

    drop_recommendations = sorted(
        {f["col"] for f in high_corr_features} | {t["col"] for t in temporal_leakage}
    )

    report = {
        "task": task,
        "leakage_severity": leakage_severity,
        "high_corr_features": high_corr_features,
        "temporal_leakage": temporal_leakage,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_keys_sample": duplicate_keys,
        "feature_dominance": feature_dominance,
        "id_leak_warnings": id_leak_warnings,
        "data_quality_issues": data_quality_issues,
        "drop_recommendations": drop_recommendations,
    }

    OUTPUT_LEAKAGE.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_LEAKAGE / "leakage_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# Leakage Report ({task})",
        "",
        f"- 심각도: **{leakage_severity}**",
        f"- 타깃 상관 누수: {len(high_corr_features)}건",
        f"- 시점 누수 의심: {len(temporal_leakage)}건",
        f"- 중복 키: {duplicate_key_count}건",
        f"- 피처 단독 지배(>0.5): {len(feature_dominance)}건",
        f"- 식별자 혼입: {len(id_leak_warnings)}건",
        f"- 데이터 품질 이슈: {len(data_quality_issues)}건",
    ]
    if drop_recommendations:
        md_lines += ["", "## Drop 권장"] + [f"- `{c}`" for c in drop_recommendations]
    md_path = OUTPUT_LEAKAGE / "leakage_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[leakage_checker] task={task} severity={leakage_severity} "
        f"high_corr={len(high_corr_features)} temporal={len(temporal_leakage)} dup={duplicate_key_count}"
    )
    return {
        "leakage_report_path": str(json_path.relative_to(BASE)).replace("\\", "/"),
        "leakage_severity": leakage_severity,
        "drop_recommendations": drop_recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage Checker agent — 타깃·시점 누수, 중복 키 점검")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    args = parser.parse_args()
    result = check_leakage(args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
