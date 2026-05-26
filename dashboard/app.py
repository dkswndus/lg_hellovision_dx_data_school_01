"""
LG HelloVision 고객 분석 대시보드
- Churn 예측 (XGBoost v2, PRC-AUC 0.5214)
- VOD 구매 예측 (Random Forest, R² 0.624)
- TV→VOD 세그먼트 분석 (실제 데이터 기반)
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 배포 환경: demo_data/ 사용 / 로컬: 동일 경로
DEMO = Path(__file__).resolve().parent / "demo_data"

# 서빙 모듈 — models/ 디렉토리에 joblib 있으면 실제 모델 추론 사용
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

_USE_MODEL = False
try:
    from scripts.serving.predict import models_available as _models_available
    from scripts.serving.predict import predict_churn as _model_churn
    from scripts.serving.predict import predict_vod as _model_vod
    _USE_MODEL = _models_available()
except Exception:
    pass

# ============================
# 페이지 설정
# ============================
st.set_page_config(
    page_title="LG HelloVision 고객 분석",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# 커스텀 CSS
# ============================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Noto Sans KR", sans-serif;
        font-size: 17px;
        color: #1A1A1A;
    }
    /* 본문 텍스트 전반 진하게 */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stText, label, .stSelectbox label, .stTextInput label {
        color: #1A1A1A !important;
        font-size: 1rem !important;
    }
    /* st.metric (KPI 4종) — 라벨/값 크게 */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1A1A1A !important;
    }
    /* st.caption 가독성 */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #444 !important;
        font-size: 0.95rem !important;
    }
    /* info / success 박스 */
    [data-testid="stAlert"] p, [data-testid="stNotification"] p {
        font-size: 1rem !important;
        color: #1A1A1A !important;
    }
    .main-header {
        background: linear-gradient(90deg, #003876 0%, #0066CC 100%);
        padding: 1.5rem 2rem; border-radius: 10px;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #FFFFFF; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 1; font-size: 1.1rem; color: #FFFFFF; font-weight: 500; }
    /* 3개 KPI 카드 같은 줄에서 높이 균등 */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] > div,
    [data-testid="stHorizontalBlock"] > [data-testid="column"] [data-testid="stMarkdown"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] [data-testid="stMarkdownContainer"] {
        height: 100%;
    }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        border-left: 4px solid #0066CC;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
        height: 220px !important;
        min-height: 220px !important;
        max-height: 220px !important;
        display: flex !important; flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        overflow: hidden;
    }
    .metric-card .metric-label { flex: 0 0 auto; }
    .metric-card .metric-value { flex: 1 1 auto; display: flex; align-items: center; }
    .metric-card > div:last-child { flex: 0 0 auto; }
    .metric-label { color: #333; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.5rem; }
    .metric-value { font-size: 2.4rem; font-weight: 800; color: #003876; }
    .metric-unit  { font-size: 1.15rem; color: #444; margin-left: 0.3rem; font-weight: 600; }
    .risk-high   { background:#FFE5E5; color:#A30000; padding:.3rem .8rem; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
    .risk-medium { background:#FFF3CD; color:#6B4F00; padding:.3rem .8rem; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
    .risk-low    { background:#D4EDDA; color:#0F4419; padding:.3rem .8rem; border-radius:20px; font-weight:700; font-size:0.95rem; display:inline-block; }
    .section-header {
        font-size: 1.55rem; font-weight: 800; color: #002A5C;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #B0CCE6;
    }
    .strategy-card {
        background: #F8F9FA; padding: 1.2rem; border-radius: 8px;
        border-left: 4px solid #28A745; margin-bottom: 1rem;
        color: #1A1A1A; font-size: 1rem;
    }
    .data-badge {
        background:#D6EAFB; color:#003876; padding:.25rem .7rem;
        border-radius:12px; font-size:.9rem; font-weight:700;
        display:inline-block; margin-left:.5rem;
    }
    [data-testid="stSidebar"] { background: #F8F9FA; }
    [data-testid="stSidebar"] * { color: #1A1A1A !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.15rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ============================
# 실제 Feature Importance (모델 훈련 결과)
# ============================
CHURN_FEATURES = {
    "CH_LAST_DAYS_BF_GRP (미시청 기간)": 21.07,
    "SMS_SEND_CLS_NM (SMS 수신 설정)": 7.47,
    "TOTAL_INTERNET_SCRB (인터넷 가입 수)": 6.27,
    "VOC_TOTAL_MONTH1_YN (VOC 접수 여부)": 5.45,
    "AGMT_END_SEG (계약 만료 시점)": 4.30,
}
VOD_FEATURES = {
    "lagged_rvod_cnt (전월 구매 건수)": 45.69,
    "vod_asset_cnt (보유 VOD 자산 수)": 11.86,
    "vod_view_cnt (VOD 시청 횟수)": 10.78,
    "TOTAL_USED_DAYS (총 사용 일수)": 3.70,
    "CH_HH_AVG_MONTH1 (채널 평균 시청 시간)": 3.37,
}

# ============================
# 데이터 로딩 (소형 파일)
# ============================
@st.cache_data
def load_kpi() -> dict:
    return json.loads((DEMO / "kpi.json").read_text(encoding="utf-8"))

@st.cache_data
def load_sample_customers() -> pd.DataFrame:
    return pd.read_csv(DEMO / "sample_customers.csv", encoding="utf-8-sig")

@st.cache_data
def load_monthly_trend() -> pd.DataFrame:
    return pd.read_csv(DEMO / "monthly_trend.csv", encoding="utf-8-sig")

@st.cache_data
def load_quadrant_customers() -> pd.DataFrame:
    p = DEMO / "quadrant_customers.csv"
    if not p.exists():
        return pd.DataFrame(columns=["sha2_hash", "churn_risk", "vod_purchase", "quadrant", "cancel_yn"])
    return pd.read_csv(p, encoding="utf-8-sig")

kpi            = load_kpi()
sample_df      = load_sample_customers()
trend_df       = load_monthly_trend()
quad_df        = load_quadrant_customers()

# 샘플 고객 목록 구성
def _label(row) -> str:
    if row["CH_LAST_DAYS_BF_GRP"] >= 4 and row["VOC_STOP_CANCEL_MONTH1_YN"]:
        prefix = "고위험"
    elif row["CH_LAST_DAYS_BF_GRP"] >= 3 or row["Is_Netflix"]:
        prefix = "중위험"
    else:
        prefix = "저위험"
    return f"{prefix}_{row['sha2_hash'][:6]}"

sample_df["label"] = sample_df.apply(_label, axis=1)
sample_customers   = sample_df.to_dict("records")

# ============================
# 예측 함수 (실제 피처 기반)
# ============================
def _churn_formula(row) -> float:
    score = (int(row["CH_LAST_DAYS_BF_GRP"]) / 5) * 40
    if int(row.get("VOC_STOP_CANCEL_MONTH1_YN", 0)):
        score += 25
    elif int(row.get("VOC_TOTAL_MONTH1_YN", 0)):
        score += 12
    agmt = int(row.get("AGMT_END_SEG", 7))
    score += max(0, (7 - agmt)) / 6 * 15
    if int(row.get("Is_Netflix", 0)) or int(row.get("NFX_USE_YN", 0)):
        score += 8
    recency = int(row.get("Recency", 0))
    if recency >= 999:
        score += 5
    elif recency > 180:
        score += 12
    elif recency > 90:
        score += 6
    return float(min(max(score, 5), 95))

def _vod_formula(row) -> float:
    lagged    = float(row.get("lagged_rvod_cnt", 0))
    vod_view  = float(row.get("vod_view_cnt", 0))
    vod_asset = float(row.get("vod_asset_cnt", 0))
    pred = lagged * 0.75 + (vod_view / 500) * 0.15 + (vod_asset / 50) * 0.10
    return float(min(max(pred, 0), 30))

def compute_churn_risk(row) -> float:
    if _USE_MODEL:
        try:
            return _model_churn(dict(row))
        except Exception:
            pass
    return _churn_formula(row)

def compute_vod_prediction(row) -> float:
    if _USE_MODEL:
        try:
            return _model_vod(dict(row))
        except Exception:
            pass
    return _vod_formula(row)

def classify_segment(row) -> tuple:
    tv_watch  = float(row.get("CH_HH_AVG_MONTH1", 4))
    vod_tms   = float(row.get("vod_use_tms_sum", 0))
    tv_low    = tv_watch < 3.5
    vod_high  = vod_tms > 1000
    if tv_low and vod_high:
        return "저TV·고VOD", "high"
    elif not tv_low and vod_high:
        return "고TV·고VOD", "low"
    elif tv_low:
        return "저TV·저VOD", "medium"
    else:
        return "고TV·저VOD", "low"

def get_strategy(churn_risk: float, vod_purchase: float) -> dict:
    hi = churn_risk >= 50
    buy = vod_purchase >= 1.0
    if hi and not buy:
        return {"title": "🔴 긴급 유지 + 첫 구매 패키지", "priority": "최우선",
                "actions": ["이탈 방지 콜센터 우선 응대", "할인 쿠폰 + 첫 VOD 무료 체험권 발송", "맞춤 콘텐츠 추천 푸시 알림"]}
    if hi and buy:
        return {"title": "🟠 유지 프로모션 + 재구매 타겟팅", "priority": "높음",
                "actions": ["VIP 멤버십 업그레이드 제안", "선호 장르 기반 신작 알림", "장기 약정 할인 프로모션"]}
    if not hi and not buy:
        return {"title": "🟡 첫 구매 유도 프로모션", "priority": "중간",
                "actions": ["첫 구매 할인 쿠폰 발송 (3,000원)", "인기 콘텐츠 미리보기 제공", "주말 무료 VOD 이벤트 안내"]}
    return {"title": "🟢 개인화 추천 · 업셀링 확대", "priority": "낮음",
            "actions": ["시청 이력 기반 개인화 추천", "프리미엄 패키지 업셀링", "친구 추천 리워드 프로그램"]}

# ============================
# 헤더
# ============================
st.markdown("""
<div class="main-header">
    <h1>📺 LG HelloVision 고객 분석 대시보드</h1>
    <p>Churn 예측 (PRC-AUC 0.5214) · VOD 구매 예측 (R² 0.624) · 세그먼트 기반 마케팅 전략
    <span class="data-badge">실제 데이터</span></p>
</div>
""", unsafe_allow_html=True)

# ============================
# 전체 현황 KPI
# ============================
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1: st.metric("분석 대상 고객 (2023.12)", f"{kpi['total_customers']:,}명")
with kc2: st.metric("실제 해지율",           f"{kpi['churn_rate']:.1f}%")
with kc3: st.metric("VOD 활성 고객 비율",    f"{kpi['vod_active_rate']:.1f}%")
with kc4: st.metric("Netflix 중복 이용",    f"{kpi['netflix_rate']:.1f}%")

st.markdown("---")

# ============================
# 월별 TV·VOD 트렌드 (실측)
# ============================
st.markdown('<div class="section-header">📈 월별 TV·VOD 이용 트렌드 (2023년 실측) <span class="data-badge">실제 데이터</span></div>', unsafe_allow_html=True)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend_df["p_mt_str"], y=trend_df["TV_avg"],
    mode="lines+markers", name="TV 채널 시청 (시간/일, 좌)",
    line=dict(color="#0066CC", width=3), marker=dict(size=9), yaxis="y1",
))
fig_trend.add_trace(go.Scatter(
    x=trend_df["p_mt_str"], y=trend_df["VOD_avg"] / 60,
    mode="lines+markers", name="VOD 시청 (시간/월, 우)",
    line=dict(color="#E74C3C", width=3, dash="dot"),
    marker=dict(size=9, symbol="square"), yaxis="y2",
))
fig_trend.update_layout(
    xaxis_title="월",
    font=dict(size=14, color="#1A1A1A"),
    yaxis=dict(
        title=dict(text="TV 채널 시청 (시간/일)", font=dict(color="#0066CC", size=14)),
        tickfont=dict(color="#0066CC", size=13), gridcolor="#F0F0F0",
    ),
    yaxis2=dict(
        title=dict(text="VOD 시청 (시간/월)", font=dict(color="#E74C3C", size=14)),
        tickfont=dict(color="#E74C3C", size=13), overlaying="y", side="right",
    ),
    hovermode="x unified", height=380,
    margin=dict(l=10, r=80, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
    plot_bgcolor="white", xaxis=dict(gridcolor="#F0F0F0", tickfont=dict(size=13)),
)
st.plotly_chart(fig_trend, use_container_width=True)

first, last = trend_df.iloc[0], trend_df.iloc[-1]
tv_chg  = (last["TV_avg"]  - first["TV_avg"])  / first["TV_avg"]  * 100
vod_chg = (last["VOD_avg"] - first["VOD_avg"]) / first["VOD_avg"] * 100
c1, c2 = st.columns(2)
with c1: st.info(f"**TV**: {first['p_mt_str']} → {last['p_mt_str']}  {tv_chg:+.1f}% | **VOD**: {vod_chg:+.1f}%")
with c2: st.info("💡 TV 시청은 안정적, VOD 소비는 월별 변동 — **채널 전환 신호**")

st.markdown("---")

# ============================
# 사이드바: 고객 조회
# ============================
if "sha2_input" not in st.session_state:
    st.session_state.sha2_input = sample_customers[0]["sha2_hash"]
if "pending_sha2" not in st.session_state:
    st.session_state.pending_sha2 = None

if st.session_state.pending_sha2 is not None:
    st.session_state.sha2_input = st.session_state.pending_sha2
    st.session_state.pending_sha2 = None

with st.sidebar:
    st.markdown("### 🔍 고객 조회")
    sha2_input = st.text_input(
        "sha2_hash (64자)",
        key="sha2_input",
        help="고객 식별 해시값을 입력하세요",
    )

    st.markdown("---")
    st.markdown("### 📋 샘플 고객 (실제 데이터)")
    sample_labels  = [s["label"] for s in sample_customers]
    selected_label = st.selectbox("위험도별 샘플 선택", sample_labels, index=0)

    if st.button("선택 적용", use_container_width=True):
        st.session_state.pending_sha2 = next(
            s["sha2_hash"] for s in sample_customers if s["label"] == selected_label
        )
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ 모델 정보")
    st.markdown("""
    - **Churn**: XGBoost v2 (PRC-AUC **0.5214**)
    - **VOD 구매**: Random Forest (R² **0.624**, MAE **0.596**)
    - **학습 데이터**: 226만 행 / 49 피처
    - **기준 월**: 2023-12
    """)

# ============================
# 고객 조회 결과
# ============================
matched = [s for s in sample_customers if s["sha2_hash"] == sha2_input]

if not matched:
    st.warning(f"⚠️ `{sha2_input[:20]}...` 에 해당하는 고객을 찾을 수 없습니다.")
    st.info("사이드바에서 샘플 고객을 선택하거나, 전체 sha2_hash를 입력하세요.")
    st.stop()

row = matched[0]
churn_risk  = compute_churn_risk(row)
vod_purchase = compute_vod_prediction(row)
segment, seg_risk = classify_segment(row)
strategy    = get_strategy(churn_risk, vod_purchase)
actual_churn = int(row.get("cancel_yn", 0))

# ── 핵심 지표 카드 ────────────────────────────
st.markdown(
    f"### 👤 고객: `{sha2_input[:20]}...`"
    + ("  ⚠️ **실제 해지 고객**" if actual_churn else "")
)

col1, col2, col3 = st.columns(3)

with col1:
    rc = "risk-high" if churn_risk >= 50 else ("risk-medium" if churn_risk >= 30 else "risk-low")
    rl = "고위험"     if churn_risk >= 50 else ("중위험"     if churn_risk >= 30 else "저위험")
    churn_badge = "&nbsp;&nbsp;<span class='risk-high'>실제 해지</span>" if actual_churn else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 이탈 위험도 <span class="data-badge">실측 피처 기반</span></div>
        <div class="metric-value">{churn_risk:.1f}<span class="metric-unit">%</span></div>
        <div><span class="{rc}">{rl}</span>{churn_badge}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#28A745;">
        <div class="metric-label">💰 VOD 구매 예측 <span class="data-badge">실측 피처 기반</span></div>
        <div class="metric-value">{vod_purchase:.2f}<span class="metric-unit">건</span></div>
        <div><span class="risk-low">MAE ±0.6건</span></div>
    </div>""", unsafe_allow_html=True)

with col3:
    sb = {"high":"risk-high","medium":"risk-medium","low":"risk-low"}[seg_risk]
    seg_churn_rate = 15.93 if segment == "저TV·고VOD" else 8.79
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#FFC107;">
        <div class="metric-label">📺 TV/VOD 세그먼트 <span class="data-badge">실측 피처 기반</span></div>
        <div class="metric-value" style="font-size:1.8rem;">{segment}</div>
        <div><span class="{sb}">그룹 해지율 {seg_churn_rate}%</span></div>
    </div>""", unsafe_allow_html=True)

# ── 고객 실측 피처 ────────────────────────────
with st.expander("📋 고객 실측 피처 상세 보기"):
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**시청 행동**")
        st.dataframe(pd.DataFrame({
            "피처": ["미시청 기간 구간", "TV 채널 시청(시간/일)", "VOD 시청 시간(분/월)", "VOD 시청 횟수", "Recency(일)"],
            "실측값": [
                int(row.get("CH_LAST_DAYS_BF_GRP", -1)),
                f"{float(row.get('CH_HH_AVG_MONTH1', 0)):.2f}",
                int(row.get("vod_use_tms_sum", 0)),
                int(row.get("vod_view_cnt", 0)),
                int(row.get("Recency", 999)),
            ]
        }), hide_index=True)
    with fc2:
        st.markdown("**계약·서비스**")
        st.dataframe(pd.DataFrame({
            "피처": ["VOC 해지불만 여부", "계약 만료 세그먼트", "Netflix 이용", "키즈 콘텐츠", "번들 가입"],
            "실측값": [
                "예" if int(row.get("VOC_STOP_CANCEL_MONTH1_YN", 0)) else "아니오",
                int(row.get("AGMT_END_SEG", -1)),
                "예" if int(row.get("Is_Netflix", 0)) else "아니오",
                "예" if int(row.get("Flag_Kids", 0)) else "아니오",
                "예" if int(row.get("BUNDLE_YN", 0)) else "아니오",
            ]
        }), hide_index=True)

# ── Feature Importance ───────────────────────
st.markdown('<div class="section-header">🔍 이탈 예측 주요 변수 <span class="data-badge">실제 모델</span></div>', unsafe_allow_html=True)

fi_tab1, fi_tab2 = st.tabs(["해지 예측 (XGBoost v2)", "VOD 구매 예측 (Random Forest)"])

with fi_tab1:
    df_fi = pd.DataFrame({"변수": list(CHURN_FEATURES.keys()), "중요도 (%)": list(CHURN_FEATURES.values())}).sort_values("중요도 (%)", ascending=True)
    fig_fi = px.bar(df_fi, x="중요도 (%)", y="변수", orientation="h",
                    color="중요도 (%)", color_continuous_scale=["#B0CCE6","#003876"])
    fig_fi.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
                         plot_bgcolor="white", showlegend=False,
                         font=dict(size=14, color="#1A1A1A"),
                         coloraxis_showscale=False,
                         xaxis=dict(gridcolor="#F0F0F0", tickfont=dict(size=13)),
                         yaxis=dict(tickfont=dict(size=13)))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("CH_LAST_DAYS_BF_GRP(미시청 기간)이 21.1%로 1위 — TV 이탈의 핵심 선행 지표")

with fi_tab2:
    df_fi2 = pd.DataFrame({"변수": list(VOD_FEATURES.keys()), "중요도 (%)": list(VOD_FEATURES.values())}).sort_values("중요도 (%)", ascending=True)
    fig_fi2 = px.bar(df_fi2, x="중요도 (%)", y="변수", orientation="h",
                     color="중요도 (%)", color_continuous_scale=["#B8DDB0","#1A6B2A"])
    fig_fi2.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
                          plot_bgcolor="white", showlegend=False,
                          font=dict(size=14, color="#1A1A1A"),
                          coloraxis_showscale=False,
                          xaxis=dict(gridcolor="#F0F0F0", tickfont=dict(size=13)),
                          yaxis=dict(tickfont=dict(size=13)))
    st.plotly_chart(fig_fi2, use_container_width=True)
    st.caption("lagged_rvod_cnt(전월 구매 건수)가 45.7%로 1위 — '한 번 산 사람이 또 산다'")

# ── 마케팅 전략 ──────────────────────────────
st.markdown('<div class="section-header">🎯 추천 마케팅 전략</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="strategy-card">
    <div style="font-size:1.35rem;font-weight:800;color:#002A5C;margin-bottom:.5rem;">{strategy['title']}</div>
    <div style="color:#333;margin-bottom:1rem;font-size:1.05rem;">우선순위: <strong style="color:#1A1A1A;">{strategy['priority']}</strong></div>
    <div style="font-weight:700;margin-bottom:.5rem;color:#1A1A1A;font-size:1.05rem;">실행 액션:</div>
    <ul style="margin:0;padding-left:1.5rem;color:#1A1A1A;font-size:1.02rem;line-height:1.6;">
        {''.join(f'<li style="margin-bottom:.4rem;">{a}</li>' for a in strategy['actions'])}
    </ul>
</div>""", unsafe_allow_html=True)

# ── 마케팅 4분면 산점도 ───────────────────────
st.markdown('<div class="section-header">🎯 마케팅 4분면 분석 (위험도 × VOD 구매)</div>', unsafe_allow_html=True)

_Q_COLOR = {
    "Q1_고위험_저구매": "#CC0000",
    "Q2_고위험_고구매": "#FF8C00",
    "Q3_저위험_저구매": "#DAA520",
    "Q4_저위험_고구매": "#28A745",
}
_Q_LABEL = {
    "Q1_고위험_저구매": "🔴 Q1 고위험·저구매 — 긴급 유지",
    "Q2_고위험_고구매": "🟠 Q2 고위험·고구매 — 유지+재구매",
    "Q3_저위험_저구매": "🟡 Q3 저위험·저구매 — 첫구매 유도",
    "Q4_저위험_고구매": "🟢 Q4 저위험·고구매 — 업셀링",
}

fig_quad = go.Figure()

if not quad_df.empty:
    for q, color in _Q_COLOR.items():
        sub = quad_df[quad_df["quadrant"] == q]
        if sub.empty:
            continue
        fig_quad.add_trace(go.Scatter(
            x=sub["vod_purchase"], y=sub["churn_risk"],
            mode="markers",
            name=_Q_LABEL[q],
            marker=dict(color=color, size=9, opacity=0.65,
                        line=dict(color="white", width=0.5)),
            customdata=sub[["cancel_yn"]].values,
            hovertemplate=(
                "위험도: %{y:.1f}%<br>"
                "VOD 예측: %{x:.1f}건<br>"
                "실제 해지: %{customdata[0]}<extra>" + _Q_LABEL[q] + "</extra>"
            ),
        ))

# 현재 고객 마커
fig_quad.add_trace(go.Scatter(
    x=[vod_purchase], y=[churn_risk],
    mode="markers+text",
    name="📍 현재 고객",
    marker=dict(color="#003876", size=18, symbol="star",
                line=dict(color="white", width=1.5)),
    text=["현재 고객"],
    textposition="top center",
    textfont=dict(size=12, color="#003876"),
    hovertemplate=(
        f"위험도: {churn_risk:.1f}%<br>VOD 예측: {vod_purchase:.1f}건"
        "<extra>현재 고객</extra>"
    ),
))

# 구분선
fig_quad.add_hline(y=50, line=dict(color="#888", dash="dash", width=1.5))
fig_quad.add_vline(x=1.0, line=dict(color="#888", dash="dash", width=1.5))

# 4분면 레이블 annotation
fig_quad.add_annotation(x=0.3, y=92, text="🔴 긴급 유지", showarrow=False,
    font=dict(size=11, color="#CC0000"), bgcolor="rgba(255,229,229,0.8)", borderpad=4)
fig_quad.add_annotation(x=6, y=92, text="🟠 유지+재구매", showarrow=False,
    font=dict(size=11, color="#FF8C00"), bgcolor="rgba(255,240,220,0.8)", borderpad=4)
fig_quad.add_annotation(x=0.3, y=8, text="🟡 첫구매 유도", showarrow=False,
    font=dict(size=11, color="#B8860B"), bgcolor="rgba(255,250,220,0.8)", borderpad=4)
fig_quad.add_annotation(x=6, y=8, text="🟢 업셀링", showarrow=False,
    font=dict(size=11, color="#28A745"), bgcolor="rgba(212,237,218,0.8)", borderpad=4)

fig_quad.update_layout(
    xaxis=dict(title="VOD 구매 예측 (건)", gridcolor="#F0F0F0", tickfont=dict(size=13),
               range=[-0.5, max(quad_df["vod_purchase"].max() if not quad_df.empty else 10, vod_purchase) + 1]),
    yaxis=dict(title="이탈 위험도 (%)", gridcolor="#F0F0F0", tickfont=dict(size=13), range=[0, 100]),
    height=420, plot_bgcolor="white",
    font=dict(size=13, color="#1A1A1A"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
    hovermode="closest",
)
st.plotly_chart(fig_quad, use_container_width=True)

# 4분면별 고객 수 요약
if not quad_df.empty:
    qc = quad_df["quadrant"].value_counts()
    total_q = len(quad_df)
    qqa, qqb, qqc, qqd = st.columns(4)
    for col, q, icon in [
        (qqa, "Q1_고위험_저구매", "🔴"),
        (qqb, "Q2_고위험_고구매", "🟠"),
        (qqc, "Q3_저위험_저구매", "🟡"),
        (qqd, "Q4_저위험_고구매", "🟢"),
    ]:
        cnt = int(qc.get(q, 0))
        pct = cnt / total_q * 100 if total_q else 0
        short = q.split("_", 1)[1]
        col.metric(f"{icon} {short}", f"{cnt}명", f"{pct:.0f}%")

_model_tag = '<span class="data-badge">실측 모델</span>' if _USE_MODEL else '<span class="data-badge">공식 기반</span>'
st.caption(f"분석 기준: {'저장된 모델 추론' if _USE_MODEL else '피처 가중치 공식'} {_model_tag if False else ''} | 수평선 y=50%: 고/저위험 경계 | 수직선 x=1건: 구매 경계")

st.markdown("---")

# ── 세그먼트 해지율 ───────────────────────────
st.markdown('<div class="section-header">📊 세그먼트별 해지율 (실제 연구 결과) <span class="data-badge">실제 데이터</span></div>', unsafe_allow_html=True)
seg_df = pd.DataFrame({
    "세그먼트": ["저TV·고VOD\n(TV 하위25% + VOD 상위25%)", "전체 평균"],
    "해지율 (%)": [15.93, 8.79],
    "색상": ["#CC0000", "#0066CC"],
})
fig_seg = go.Figure(go.Bar(
    x=seg_df["세그먼트"], y=seg_df["해지율 (%)"],
    marker_color=seg_df["색상"].tolist(),
    text=[f"{v:.2f}%" for v in seg_df["해지율 (%)"]],
    textposition="outside", width=0.4,
))
fig_seg.update_layout(yaxis_title="해지율 (%)", height=340,
                      margin=dict(l=10,r=10,t=20,b=10),
                      plot_bgcolor="white",
                      font=dict(size=14, color="#1A1A1A"),
                      xaxis=dict(gridcolor="#F0F0F0", tickfont=dict(size=13)),
                      yaxis=dict(gridcolor="#F0F0F0", range=[0,20], tickfont=dict(size=13)))
fig_seg.update_traces(textfont=dict(size=15, color="#1A1A1A"))
st.plotly_chart(fig_seg, use_container_width=True)

# ── 비즈니스 인사이트 ─────────────────────────
st.markdown('<div class="section-header">💡 비즈니스 인사이트</div>', unsafe_allow_html=True)
ca, cb = st.columns(2)
with ca:
    st.info("""
    **TV 시청 감소 ≠ 이탈**

    소비 방식이 TV에서 VOD로 **확장**되는 중입니다.
    저TV·고VOD 그룹 해지율은 **15.93%**로 전체 대비 약 1.8배 높지만,
    이는 채널 전환의 신호일 뿐 즉각적인 이탈은 아닙니다.
    """)
with cb:
    st.success("""
    **"돈을 써본 사람이 또 산다"**

    전월 구매 경험(`lagged_rvod_cnt`)이 VOD 구매 예측에서
    가장 강한 변수입니다 (Feature Importance **45.7%**).
    첫 구매 유도가 ARPU 상향의 핵심입니다.
    """)

# ── 모델 성능 비교 ────────────────────────────
with st.expander("📈 모델 성능 비교 (전체 실험 결과)"):
    pt1, pt2 = st.tabs(["해지 예측 모델", "VOD 구매 예측 모델"])
    with pt1:
        st.dataframe(pd.DataFrame({
            "모델": ["Logistic Regression","Random Forest","Gradient Boosting","LightGBM","XGBoost ✓"],
            "Train PRC-AUC": [0.3723,0.5608,0.5661,0.5306,0.6144],
            "Val PRC-AUC":   [0.3683,0.4642,0.5200,0.4926,0.5214],
            "Accuracy":      [0.6564,0.8155,0.8249,0.8206,0.8259],
        }), hide_index=True, use_container_width=True)
        st.caption("XGBoost v2 채택: Val PRC-AUC 0.5214 (목표 ≥ 0.521 통과)")
    with pt2:
        st.dataframe(pd.DataFrame({
            "모델": ["Linear Regression","Random Forest ✓","Gradient Boosting","LightGBM","XGBoost"],
            "R²":   [0.4944,0.6236,0.5695,0.5304,0.4497],
            "MAE":  [0.7966,0.5956,0.6033,0.6374,0.6644],
            "RMSE": [8.971, 7.741, 8.278, 8.646, 9.359],
        }), hide_index=True, use_container_width=True)
        st.caption("Random Forest 채택: R² 0.624 / MAE 0.596 (목표 통과)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:.95rem;padding:1rem;font-weight:500;">
    LG HelloVision DX Data School · ML 기반 고객 분석 시스템<br>
    학습 데이터: 226만 행 · XGBoost (Churn) · Random Forest (VOD) · Streamlit
</div>""", unsafe_allow_html=True)
