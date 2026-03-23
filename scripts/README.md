# 스크립트 구조

프로젝트 루트에서 실행합니다.

## 폴더 구성

| 폴더 | 역할 | 스크립트 |
|------|------|----------|
| **build/** | 데이터 구축 | `build_churn_dataset.py`, `build_churn_dataset_v2.py`, `build_vod_purchase_dataset.py`, `build_monthly_vod_dataset.py`, `build_monthly_vod_by_type.py`, `build_channel_model_dataset.py`, `load_raw_to_duckdb.py` |
| **visualization/** | 시각화 | `plot_ct_cl_by_churn.py`, `plot_tv_vod_expansion.py`, `plot_clean_data.py`, `plot_cancel_yn.py` |
| **analysis/** | 분석·모델 | `churn_analysis.py`, `run_churn_v2.py`, `vod_purchase_model.py`, `channel_model.py`, `analyze_tv_vod_shift.py` |
| **eda/** | EDA·유틸 | `eda_and_clean.py`, `export_schema_json.py` |

## 실행 예시

```bash
# 데이터 구축
python scripts/build/build_churn_dataset.py
python scripts/build/build_churn_dataset_v2.py
python scripts/build/build_vod_purchase_dataset.py

# 분석·모델
python scripts/analysis/churn_analysis.py
python scripts/analysis/run_churn_v2.py
python scripts/analysis/vod_purchase_model.py

# 시각화
python scripts/visualization/plot_ct_cl_by_churn.py
python scripts/visualization/plot_tv_vod_expansion.py
```
