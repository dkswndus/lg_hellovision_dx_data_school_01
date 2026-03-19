# 케이블 데이터 분석

**매출 = 고객 수 × 고객당 매출**

---

## 폴더 구조

```
lghellovision_01/
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/              # 원본 CSV (TPS, VOD 로그, vod_mart 등)
│   ├── interim/          # 전처리 결과 (parquet, eda_report.json)
│   └── lghellovision.duckdb
├── load_raw_to_duckdb.py # raw → DuckDB 적재
├── eda_and_clean.py      # EDA/전처리 (결측 삭제, 타입 변환)
├── requirements.txt
├── .gitignore
└── README.md
```

⚠️ **`.gitignore`**: `data/` 폴더 내 대용량 파일, `결과보고서.md`는 Git에서 제외됩니다.
