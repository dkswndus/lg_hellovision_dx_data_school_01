"""
LG HelloVision 고객 분석 대시보드
- Churn 예측 (XGBoost, PRC-AUC 0.5214)
- VOD 구매 예측 (Random Forest, R² 0.624)
- TV→VOD 세그먼트 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
# 커스텀 CSS (깔끔한 스타일)
# ============================
st.markdown("""
<style>
    /* 전체 폰트 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Noto Sans KR", sans-serif;
    }
    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(90deg, #003876 0%, #0066CC 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    /* 메트릭 카드 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0066CC;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #003876;
    }
    .metric-unit {
        font-size: 1rem;
        color: #888;
        margin-left: 0.3rem;
    }
    /* 위험도 배지 */
    .risk-high {
        background: #FFE5E5;
        color: #CC0000;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-medium {
        background: #FFF3CD;
        color: #856404;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-low {
        background: #D4EDDA;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    /* 섹션 헤더 */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #003876;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E0E0E0;
    }
    /* 전략 카드 */
    .strategy-card {
        background: #F8F9FA;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #28A745;
        margin-bottom: 1rem;
    }
    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)


# ============================
# 데모 데이터 생성 함수
# ============================
@st.cache_data
def generate_customer_data(customer_id: str) -> dict:
    """
    customer_id 기반 결정론적 더미 데이터.
    실제 환경에서는 학습된 모델 + 사용자 피처를 불러와 예측.
    """
    seed = sum(ord(c) for c in customer_id) % 10000
    rng = np.random.RandomState(seed)

    # 이탈 위험도 (0-100)
    churn_risk = float(rng.beta(2, 5) * 100)

    # VOD 구매 예측 (월 건수)
    vod_purchase = float(rng.gamma(2, 1.5))

    # TV/VOD 시청 패턴
    tv_watch = float(rng.uniform(20, 180))
    vod_watch = float(rng.uniform(10, 200))

    # 세그먼트 분류
    tv_low = tv_watch < 60
    vod_high = vod_watch > 100

    if tv_low and vod_high:
        segment = "저TV·고VOD"
        segment_risk = "high"
    elif not tv_low and vod_high:
        segment = "고TV·고VOD"
        segment_risk = "low"
    elif tv_low and not vod_high:
        segment = "저TV·저VOD"
        segment_risk = "medium"
    else:
        segment = "고TV·저VOD"
        segment_risk = "low"

    # 월별 시청 트렌드 (12개월)
    months = pd.date_range("2023-01", periods=12, freq="MS").strftime("%Y-%m").tolist()
    tv_trend = [max(0, tv_watch + rng.normal(0, 15)) for _ in range(12)]
    vod_trend = [max(0, vod_watch + rng.normal(0, 20) + i * 2) for i in range(12)]

    # 주요 변수 기여도 (Feature Importance 가상)
    features = {
        "Recency (미시청일)": float(rng.uniform(5, 25)),
        "Total Watch Time": float(rng.uniform(10, 20)),
        "Flag_Kids": float(rng.uniform(8, 18)),
        "AGMT_END_YMD": float(rng.uniform(7, 15)),
        "Is_Netflix": float(rng.uniform(5, 12)),
    }

    return {
        "churn_risk": churn_risk,
        "vod_purchase": vod_purchase,
        "tv_watch": tv_watch,
        "vod_watch": vod_watch,
        "segment": segment,
        "segment_risk": segment_risk,
        "months": months,
        "tv_trend": tv_trend,
        "vod_trend": vod_trend,
        "features": features,
    }


def get_strategy(churn_risk: float, vod_purchase: float) -> dict:
    """이탈 위험 + 구매 예측 → 마케팅 전략 매핑"""
    is_high_risk = churn_risk >= 50
    has_purchase = vod_purchase >= 1.0

    if is_high_risk and not has_purchase:
        return {
            "title": "🔴 긴급 유지 + 첫 구매 패키지",
            "priority": "최우선",
            "actions": [
                "이탈 방지 콜센터 우선 응대",
                "할인 쿠폰 + 첫 VOD 구매 무료 체험권 발송",
                "맞춤 콘텐츠 추천 푸시 알림",
            ],
        }
    if is_high_risk and has_purchase:
        return {
            "title": "🟠 유지 프로모션 + 재구매 타겟팅",
            "priority": "높음",
            "actions": [
                "VIP 멤버십 업그레이드 제안",
                "선호 장르 기반 신작 알림",
                "장기 약정 할인 프로모션",
            ],
        }
    if not is_high_risk and not has_purchase:
        return {
            "title": "🟡 첫 구매 유도 프로모션",
            "priority": "중간",
            "actions": [
                "첫 구매 할인 쿠폰 발송 (3,000원)",
                "인기 콘텐츠 미리보기 제공",
                "주말 무료 VOD 이벤트 안내",
            ],
        }
    return {
        "title": "🟢 개인화 추천 · 업셀링 확대",
        "priority": "낮음",
        "actions": [
            "시청 이력 기반 개인화 추천",
            "프리미엄 패키지 업셀링",
            "친구 추천 리워드 프로그램",
        ],
    }


# ============================
# 헤더
# ============================
st.markdown("""
<div class="main-header">
    <h1>📺 LG HelloVision 고객 분석 대시보드</h1>
    <p>Churn 예측 (PRC-AUC 0.5214) · VOD 구매 예측 (R² 0.624) · 세그먼트 기반 마케팅 전략</p>
</div>
""", unsafe_allow_html=True)

# ============================
# 사이드바
# ============================
with st.sidebar:
    st.markdown("### 🔍 고객 조회")
    customer_id = st.text_input(
        "고객 ID (sha2_hash)",
        value="USER_00042",
        help="고객 식별자를 입력하세요",
    )

    st.markdown("---")
    st.markdown("### 📋 빠른 조회 (샘플)")
    sample_options = ["USER_00042", "USER_07321", "USER_15890", "USER_23106", "USER_31247"]
    selected = st.selectbox("샘플 선택", sample_options, index=0)
    if st.button("선택 적용", use_container_width=True):
        customer_id = selected
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ 모델 정보")
    st.markdown("""
    - **Churn**: XGBoost (PRC-AUC 0.5214)
    - **VOD 구매**: Random Forest (R² 0.624)
    - **데이터**: 44.9M rows / 60 cols
    """)

# ============================
# 메인 컨텐츠
# ============================
data = generate_customer_data(customer_id)

# 1. 핵심 지표 (3개 카드)
st.markdown(f"### 👤 고객 ID: `{customer_id}`")

col1, col2, col3 = st.columns(3)

with col1:
    risk_class = (
        "risk-high" if data["churn_risk"] >= 50
        else "risk-medium" if data["churn_risk"] >= 30
        else "risk-low"
    )
    risk_label = (
        "고위험" if data["churn_risk"] >= 50
        else "중위험" if data["churn_risk"] >= 30
        else "저위험"
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 이탈 위험도</div>
        <div class="metric-value">{data['churn_risk']:.1f}<span class="metric-unit">%</span></div>
        <div style="margin-top: 0.5rem;">
            <span class="{risk_class}">{risk_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #28A745;">
        <div class="metric-label">💰 다음 달 VOD 구매 예측</div>
        <div class="metric-value">{data['vod_purchase']:.2f}<span class="metric-unit">건</span></div>
        <div style="margin-top: 0.5rem; color: #666; font-size: 0.85rem;">
            MAE 0.596 기준 ± 0.6건
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    seg_class = {
        "high": "risk-high",
        "medium": "risk-medium",
        "low": "risk-low",
    }[data["segment_risk"]]
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: #FFC107;">
        <div class="metric-label">📺 TV/VOD 시청 세그먼트</div>
        <div class="metric-value" style="font-size: 1.4rem;">{data['segment']}</div>
        <div style="margin-top: 0.5rem;">
            <span class="{seg_class}">
                해지율 {15.93 if data['segment'] == '저TV·고VOD' else 8.79}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. 시청 트렌드 차트
st.markdown('<div class="section-header">📈 월별 시청 트렌드</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["months"], y=data["tv_trend"],
        mode="lines+markers",
        name="TV 시청 (시간)",
        line=dict(color="#0066CC", width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=data["months"], y=data["vod_trend"],
        mode="lines+markers",
        name="VOD 시청 (시간)",
        line=dict(color="#E74C3C", width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        xaxis_title="월",
        yaxis_title="시청 시간 (시간)",
        hovermode="x unified",
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#F0F0F0"),
        yaxis=dict(gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("**현재 평균 시청 시간**")
    st.metric("📺 TV", f"{data['tv_watch']:.1f} 시간/월")
    st.metric("🎬 VOD", f"{data['vod_watch']:.1f} 시간/월")

    tv_vod_ratio = data["vod_watch"] / max(data["tv_watch"], 1)
    if tv_vod_ratio > 1.5:
        st.info("💡 VOD 중심 소비 패턴")
    elif tv_vod_ratio < 0.6:
        st.info("💡 TV 중심 소비 패턴")
    else:
        st.info("💡 균형적 소비 패턴")

# 3. 이탈 위험 주요 변수
st.markdown('<div class="section-header">🔍 이탈 예측 주요 변수 (Feature Importance)</div>', unsafe_allow_html=True)

features_df = pd.DataFrame({
    "변수": list(data["features"].keys()),
    "중요도 (%)": list(data["features"].values()),
}).sort_values("중요도 (%)", ascending=True)

fig_features = px.bar(
    features_df,
    x="중요도 (%)",
    y="변수",
    orientation="h",
    color="중요도 (%)",
    color_continuous_scale=["#B0CCE6", "#003876"],
)
fig_features.update_layout(
    height=300,
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="white",
    showlegend=False,
    coloraxis_showscale=False,
    xaxis=dict(gridcolor="#F0F0F0"),
)
st.plotly_chart(fig_features, use_container_width=True)

# 4. 마케팅 전략 추천
st.markdown('<div class="section-header">🎯 추천 마케팅 전략</div>', unsafe_allow_html=True)

strategy = get_strategy(data["churn_risk"], data["vod_purchase"])

st.markdown(f"""
<div class="strategy-card">
    <div style="font-size: 1.2rem; font-weight: 700; color: #003876; margin-bottom: 0.5rem;">
        {strategy['title']}
    </div>
    <div style="color: #666; margin-bottom: 1rem;">
        우선순위: <strong>{strategy['priority']}</strong>
    </div>
    <div style="font-weight: 600; margin-bottom: 0.5rem;">실행 액션:</div>
    <ul style="margin: 0; padding-left: 1.5rem;">
        {''.join(f'<li style="margin-bottom: 0.3rem;">{action}</li>' for action in strategy['actions'])}
    </ul>
</div>
""", unsafe_allow_html=True)

# 5. 비즈니스 인사이트
st.markdown('<div class="section-header">💡 비즈니스 인사이트</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.info("""
    **TV 시청 감소 ≠ 이탈**

    소비 방식이 TV에서 VOD로 **확장**되는 중입니다.
    저TV·고VOD 그룹의 해지율은 15.93%로 일반 대비 2배 높지만,
    이는 채널 전환의 신호일 뿐 즉각적인 이탈은 아닙니다.
    """)

with col_b:
    st.success("""
    **"돈을 써본 사람이 또 쓴다"**

    전월 구매 경험(`lagged_rvod_cnt`)이 VOD 구매 예측에서
    가장 강한 변수입니다 (Feature Importance 60%↑).
    첫 구매 유도가 ARPU 상향의 핵심입니다.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
    LG HelloVision DX Data School · ML 기반 고객 분석 시스템<br>
    Powered by XGBoost · Random Forest · Streamlit
</div>
""", unsafe_allow_html=True)
