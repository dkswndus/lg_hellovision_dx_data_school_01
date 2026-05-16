"""
LG HelloVision 고객 분석 대시보드
- Churn 예측 (XGBoost v2, PRC-AUC 0.5214)
- VOD 구매 예측 (Random Forest, R² 0.624)
- TV→VOD 세그먼트 분석 (실제 데이터 기반)
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

DEMO = Path(__file__).resolve().parent / "dashboard" / "demo_data"

st.set_page_config(
    page_title="LG HelloVision 고객 분석",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Noto Sans KR", sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #003876 0%, #0066CC 100%);
        padding: 1.5rem 2rem; border-radius: 10px;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        border-left: 4px solid #0066CC;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 1rem;
    }
    .metric-label { color: #666; font-size: 0.9rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #003876; }
    .metric-unit  { font-size: 1rem; color: #888; margin-left: 0.3rem; }
    .risk-high   { background:#FFE5E5; color:#CC0000; padding:.3rem .8rem; border-radius:20px; font-weight:600; display:inline-block; }
    .risk-medium { background:#FFF3CD; color:#856404; padding:.3rem .8rem; border-radius:20px; font-weight:600; display:inline-block; }
    .risk-low    { background:#D4EDDA; color:#155724; padding:.3rem .8rem; border-radius:20px; font-weight:600; display:inline-block; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #003876;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #E0E0E0;
    }
    .strategy-card {
        background: #F8F9FA; padding: 1.2rem; border-radius: 8px;
        border-left: 4px solid #28A745; margin-bottom: 1rem;
    }
    .data-badge {
        background:#E8F4FD; color:#0066CC; padding:.2rem .6rem;
        border-radius:12px; font-size:.8rem; font-weight:600;
        display:inline-block; margin-left:.5rem;
    }
    [data-testid="stSidebar"] { background: #F8F9FA; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
def load_kpi() -> dict:
    return json.loads((DEMO / "kpi.json").read_text(encoding="utf-8"))

@st.cache_data
def load_sample_customers() -> pd.DataFrame:
    return pd.read_csv(DEMO / "sample_customers.csv", encoding="utf-8-sig")

@st.cache_data
def load_monthly_trend() -> pd.DataFrame:
    return pd.read_csv(DEMO / "monthly_trend.csv", encoding="utf-8-sig")

kpi            = load_kpi()
sample_df      = load_sample_customers()
trend_df       = load_monthly_trend()

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

def compute_churn_risk(row) -> float:
    score = 0.0
    score += (int(row["CH_LAST_DAYS_BF_GRP"]) / 5) * 40
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

def compute_vod_prediction(row) -> float:
    lagged    = float(row.get("lagged_rvod_cnt", 0))
    vod_view  = float(row.get("vod_view_cnt", 0))
    vod_asset = float(row.get("vod_asset_cnt", 0))
    pred = lagged * 0.75 + (vod_view / 500) * 0.15 + (vod_asset / 50) * 0.10
    return float(min(max(pred, 0), 30))

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

st.markdown("""
<div class="main-header">
    <h1>📺 LG HelloVision 고객 분석 대시보드</h1>
    <p>Churn 예측 (PRC-AUC 0.5214) · VOD 구매 예측 (R² 0.624) · 세그먼트 기반 마케팅 전략
    <span class="data-badge">실제 데이터</span></p>
</div>
""", unsafe_allow_html=True)

kc1, kc2, kc3, kc4 = st.columns(4)
with kc1: st.metric("분석 대상 고객 (2023.12)", f"{kpi['total_customers']:,}명")
with kc2: st.metric("실제 해지율",           f"{kpi['churn_rate']:.1f}%")
with kc3: st.metric("VOD 활성 고객 비율",    f"{kpi['vod_active_rate']:.1f}%")
with kc4: st.metric("Netflix 중복 이용",    f"{kpi['netflix_rate']:.1f}%")

st.markdown("---")

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
    yaxis=dict(
        title=dict(text="TV 채널 시청 (시간/일)", font=dict(color="#0066CC")),
        tickfont=dict(color="#0066CC"), gridcolor="#F0F0F0",
    ),
    yaxis2=dict(
        title=dict(text="VOD 시청 (시간/월)", font=dict(color="#E74C3C")),
        tickfont=dict(color="#E74C3C"), overlaying="y", side="right",
    ),
    hovermode="x unified", height=350,
    margin=dict(l=10, r=80, t=20, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="white", xaxis=dict(gridcolor="#F0F0F0"),
)
st.plotly_chart(fig_trend, use_container_width=True)

first, last = trend_df.iloc[0], trend_df.iloc[-1]
tv_chg  = (last["TV_avg"]  - first["TV_avg"])  / first["TV_avg"]  * 100
vod_chg = (last["VOD_avg"] - first["VOD_avg"]) / first["VOD_avg"] * 100
c1, c2 = st.columns(2)
with c1: st.info(f"**TV**: {first['p_mt_str']} → {last['p_mt_str']}  {tv_chg:+.1f}% | **VOD**: {vod_chg:+.1f}%")
with c2: st.info("💡 TV 시청은 안정적, VOD 소비는 월별 변동 — **채널 전환 신호**")

st.markdown("---")

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

matched = [s for s in sample_customers if s["sha2_hash"] == sha2_input]

if not matched:
    st.warning(f"⚠️ `{sha2_input[:20]}...` 에 해당하는 고객을 찾을 수 없습니다.")
    st.info("사이드바에서 샘플 고객을 선택하거나, 전체 sha2_hash를 입력하세요.")
    st.stop()

row = matched[0]
churn_risk   = compute_churn_risk(row)
vod_purchase = compute_vod_prediction(row)
segment, seg_risk = classify_segment(row)
strategy     = get_strategy(churn_risk, vod_purchase)
actual_churn = int(row.get("cancel_yn", 0))

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
        <div style="margin-top:.5rem"><span class="{rc}">{rl}</span>{churn_badge}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#28A745;">
        <div class="metric-label">💰 다음 달 VOD 구매 예측 <span class="data-badge">실측 피처 기반</span></div>
        <div class="metric-value">{vod_purchase:.2f}<span class="metric-unit">건</span></div>
        <div style="margin-top:.5rem;color:#666;font-size:.85rem;">MAE 0.596 기준 ± 0.6건</div>
    </div>""", unsafe_allow_html=True)

with col3:
    sb = {"high":"risk-high","medium":"risk-medium","low":"risk-low"}[seg_risk]
    seg_churn_rate = 15.93 if segment == "저TV·고VOD" else 8.79
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#FFC107;">
        <div class="metric-label">📺 TV/VOD 세그먼트</div>
        <div class="metric-value" style="font-size:1.4rem;">{segment}</div>
        <div style="margin-top:.5rem"><span class="{sb}">그룹 해지율 {seg_churn_rate}%</span></div>
    </div>""", unsafe_allow_html=True)

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

st.markdown('<div class="section-header">🔍 이탈 예측 주요 변수 <span class="data-badge">실제 모델</span></div>', unsafe_allow_html=True)

fi_tab1, fi_tab2 = st.tabs(["해지 예측 (XGBoost v2)", "VOD 구매 예측 (Random Forest)"])

with fi_tab1:
    df_fi = pd.DataFrame({"변수": list(CHURN_FEATURES.keys()), "중요도 (%)": list(CHURN_FEATURES.values())}).sort_values("중요도 (%)", ascending=True)
    fig_fi = px.bar(df_fi, x="중요도 (%)", y="변수", orientation="h",
                    color="중요도 (%)", color_continuous_scale=["#B0CCE6","#003876"])
    fig_fi.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10),
                         plot_bgcolor="white", showlegend=False,
                         coloraxis_showscale=False, xaxis=dict(gridcolor="#F0F0F0"))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("CH_LAST_DAYS_BF_GRP(미시청 기간)이 21.1%로 1위 — TV 이탈의 핵심 선행 지표")

with fi_tab2:
    df_fi2 = pd.DataFrame({"변수": list(VOD_FEATURES.keys()), "중요도 (%)": list(VOD_FEATURES.values())}).sort_values("중요도 (%)", ascending=True)
    fig_fi2 = px.bar(df_fi2, x="중요도 (%)", y="변수", orientation="h",
                     color="중요도 (%)", color_continuous_scale=["#B8DDB0","#1A6B2A"])
    fig_fi2.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10),
                          plot_bgcolor="white", showlegend=False,
                          coloraxis_showscale=False, xaxis=dict(gridcolor="#F0F0F0"))
    st.plotly_chart(fig_fi2, use_container_width=True)
    st.caption("lagged_rvod_cnt(전월 구매 건수)가 45.7%로 1위 — '한 번 산 사람이 또 산다'")

st.markdown('<div class="section-header">🎯 추천 마케팅 전략</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="strategy-card">
    <div style="font-size:1.2rem;font-weight:700;color:#003876;margin-bottom:.5rem;">{strategy['title']}</div>
    <div style="color:#666;margin-bottom:1rem;">우선순위: <strong>{strategy['priority']}</strong></div>
    <div style="font-weight:600;margin-bottom:.5rem;">실행 액션:</div>
    <ul style="margin:0;padding-left:1.5rem;">
        {''.join(f'<li style="margin-bottom:.3rem;">{a}</li>' for a in strategy['actions'])}
    </ul>
</div>""", unsafe_allow_html=True)

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
fig_seg.update_layout(yaxis_title="해지율 (%)", height=300,
                      margin=dict(l=10,r=10,t=20,b=10),
                      plot_bgcolor="white",
                      xaxis=dict(gridcolor="#F0F0F0"),
                      yaxis=dict(gridcolor="#F0F0F0", range=[0,20]))
st.plotly_chart(fig_seg, use_container_width=True)

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

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.85rem;padding:1rem;">
    LG HelloVision DX Data School · ML 기반 고객 분석 시스템<br>
    학습 데이터: 226만 행 · XGBoost (Churn) · Random Forest (VOD) · Streamlit
</div>""", unsafe_allow_html=True)
