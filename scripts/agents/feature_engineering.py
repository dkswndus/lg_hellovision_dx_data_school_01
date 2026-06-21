"""Feature Engineering Agent — 파생변수 변수사전(feature_spec.json) 생성 (체크포인트 2 이후).

.github/agents/feature-engineering-agent.agent.md 스펙 구현체. 파생변수 자체는
build_churn_dataset_v2.py / build_vod_purchase_dataset.py 가 이미 data/processed/*.parquet
에 만들어 두었으므로, 이 에이전트는 (1) 그 변수들의 사전(formula·rationale·근거)을 문서화하고
(2) RF 기반 사전 중요도를 계산해 다음 단계(Modeling)에 넘긴다.
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
OUTPUT_FEATURES = BASE / "output" / "features"

DATASETS = {
    "churn": PROCESSED / "churn_dataset_v2.parquet",
    "vod_purchase": PROCESSED / "vod_purchase_dataset.parquet",
}
TARGET_COL = {"churn": "cancel_yn", "vod_purchase": "vod_purchase_cnt"}
EXCLUDE_BASE = {
    "churn": {"sha2_hash", "cancel_yn", "p_mt", "vod_use_tms_sum", "NFX_USE_YN"},
    "vod_purchase": {"sha2_hash", "p_mt", "vod_purchase_cnt", "cancel_yn", "rvod_cnt"},
}
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# 변수사전: build_churn_dataset_v2.py / build_vod_purchase_dataset.py 가 이미 생성한 파생변수 문서화
DERIVED_FEATURES = {
    "churn": {
        "Flag_Kids": {
            "formula": "vod_log.CT_CL == 14 인 시청 기록 존재 여부",
            "source_cols": ["CT_CL"],
            "rationale": "키즈 콘텐츠 시청 가구는 해지 패턴이 다름 (TDD v2)",
            "version": "v2",
        },
        "Flag_Movie": {
            "formula": "vod_log.CT_CL == 12 인 시청 기록 존재 여부",
            "source_cols": ["CT_CL"],
            "rationale": "영화 콘텐츠 시청 여부 — 콘텐츠 선호 세분화 (TDD v2)",
            "version": "v2",
        },
        "Recency": {
            "formula": "기준일(p_mt) - 마지막 VOD 시청일",
            "source_cols": ["p_mt", "vod_log.STRT_DT"],
            "rationale": "최근성 — 시청 안한지 오래될수록 해지 위험 (TDD v2)",
            "version": "v2",
        },
        "Total_Watch_Time": {
            "formula": "sum(vod_log.use_tms) — vod_use_tms_sum 의 정제판",
            "source_cols": ["vod_use_tms_sum"],
            "rationale": "총 시청시간 — TV 시청 감소가 VOD 로 이동했는지 확인 (TDD v2)",
            "version": "v2",
        },
        "Log_Watch_Time": {
            "formula": "ln(1 + Total_Watch_Time)",
            "source_cols": ["Total_Watch_Time"],
            "rationale": "시청시간 우측 꼬리분포 완화 (TDD v2). 트리모델엔 Total_Watch_Time 과 정보량 동일",
            "version": "v2",
        },
        "Is_Netflix": {
            "formula": "NFX_USE_YN 의 정제판 (0/1)",
            "source_cols": ["NFX_USE_YN"],
            "rationale": "OTT 동시이용 — TV→VOD/OTT 확장 가설 검증 (TDD v2)",
            "version": "v2",
        },
        "Risk_Segment_Kids_NFX": {
            "formula": "Flag_Kids == 1 AND Is_Netflix == 1",
            "source_cols": ["Flag_Kids", "Is_Netflix"],
            "rationale": "키즈+넷플릭스 동시 이용 고위험 세그먼트 (EDA 인사이트)",
            "version": "v2",
        },
    },
    "vod_purchase": {
        "fod_cnt": {
            "formula": "월별 FOD(무료 VOD) 시청 건수",
            "source_cols": ["vod_log.PROD_TYPE", "p_mt"],
            "rationale": "RVOD/FOD/SVOD 미분리 시 R2 0.293 → 분리 후 0.624 (TDD-ml-cycle Green)",
            "version": "v2",
        },
        "svod_cnt": {
            "formula": "월별 SVOD(구독형) 시청 건수",
            "source_cols": ["vod_log.PROD_TYPE", "p_mt"],
            "rationale": "구독형 VOD 별도 집계 — RVOD 구매와 대체관계 포착 (TDD v2)",
            "version": "v2",
        },
        "lagged_rvod_cnt": {
            "formula": "shift(rvod_cnt, 1) — 직전월 RVOD 건수",
            "source_cols": ["rvod_cnt", "p_mt", "sha2_hash"],
            "rationale": "구매 습관의 자기상관 포착 (lag feature), 시점누수 없음 (직전월만 사용)",
            "version": "v2",
        },
    },
}


def run_feature_engineering(task: Literal["churn", "vod_purchase"]) -> dict:
    """기존 파생변수의 변수사전을 작성하고 RF 기반 사전 중요도를 계산한다."""
    data_path = DATASETS[task]
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} 없음 — build_*.py 로 데이터셋 생성 필요")

    df = pd.read_parquet(data_path)
    spec = DERIVED_FEATURES[task]

    missing_cols = [c for c in spec if c not in df.columns]
    feature_spec = {}
    for name, meta in spec.items():
        entry = dict(meta)
        entry["dtype"] = str(df[name].dtype) if name in df.columns else None
        entry["present"] = name in df.columns
        feature_spec[name] = entry

    OUTPUT_FEATURES.mkdir(parents=True, exist_ok=True)
    spec_path = OUTPUT_FEATURES / f"feature_spec_{task}.json"
    spec_path.write_text(json.dumps(feature_spec, indent=2, ensure_ascii=False), encoding="utf-8")

    # RF 기반 사전 중요도
    target = TARGET_COL[task]
    exclude = EXCLUDE_BASE[task]
    feature_cols = [c for c in df.columns if c not in exclude]
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    X = pd.DataFrame(index=sample.index)
    for c in feature_cols:
        X[c] = sample[c] if pd.api.types.is_numeric_dtype(sample[c]) else sample[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    y = sample[target]

    rf_cls = RandomForestClassifier if task == "churn" else RandomForestRegressor
    rf = rf_cls(n_estimators=50, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importance = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance["is_derived"] = importance["feature"].isin(spec.keys())
    importance_path = OUTPUT_FEATURES / f"feature_importance_prior_{task}.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    derived_importance_sum = float(importance.loc[importance["is_derived"], "importance"].sum())

    print(
        f"[feature_engineering] task={task} 변수사전={len(feature_spec)}개 "
        f"(누락 {len(missing_cols)}) 파생변수 importance 합={derived_importance_sum:.4f}"
    )
    return {
        "feature_spec_path": str(spec_path.relative_to(BASE)).replace("\\", "/"),
        "feature_importance_path": str(importance_path.relative_to(BASE)).replace("\\", "/"),
        "dataset_path": str(data_path.relative_to(BASE)).replace("\\", "/"),
        "missing_derived_cols": missing_cols,
        "derived_importance_sum": derived_importance_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Engineering agent — 파생변수 변수사전 생성")
    parser.add_argument("--task", choices=["churn", "vod_purchase"], default="churn")
    args = parser.parse_args()
    result = run_feature_engineering(args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
