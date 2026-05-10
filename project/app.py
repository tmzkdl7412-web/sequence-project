import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from preprocessing import load_data, handle_missing, handle_outliers, infer_frequency
from analysis import stl_decompose, stationarity_tests, compute_acf_pacf, estimate_orders
from modeling import train_models, forecast_future, run_backtest
from evaluation import compute_metrics, residual_analysis
from visualization import (
    plot_series, plot_stl, plot_acf_pacf,
    plot_forecast_comparison, plot_future_forecast,
    plot_residuals, plot_backtest
)

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="시계열 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 전역 CSS (shadcn/ui dashboard style) ──────────────────────────────────────
st.markdown("""
<style>
/* ===== 기본 배경 & 폰트 ===== */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    color: #111827 !important;
    font-family: 'Inter', 'Pretendard', -apple-system, sans-serif;
}

/* ===== 헤더 영역 ===== */
[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #E5E7EB;
}

/* ===== 메인 컨텐츠 패딩 ===== */
[data-testid="stMain"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ===== 사이드바 숨김 (top-panel 방식 사용) ===== */
[data-testid="stSidebar"] {
    display: none !important;
}

/* ===== 흰색 카드 컴포넌트 ===== */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    color: #111827;
    margin-bottom: 1rem;
}

/* ===== KPI 카드 ===== */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.kpi-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    margin-top: 0.5rem;
}
.kpi-badge.best  { background: #D1FAE5; color: #065F46; }
.kpi-badge.info  { background: #DBEAFE; color: #1E40AF; }
.kpi-badge.warn  { background: #FEF3C7; color: #92400E; }

/* ===== 설정 패널 ===== */
.settings-panel {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
.settings-panel h3 {
    font-size: 0.85rem;
    font-weight: 600;
    color: #374151;
    margin: 0 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #F3F4F6;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.settings-divider {
    height: 1px;
    background: #F3F4F6;
    margin: 0.75rem 0;
}

/* ===== 탭 스타일 ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #F3F4F6 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.1rem !important;
    border: none !important;
    transition: all 0.15s ease;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #2563EB !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ===== 설명 박스 ===== */
.info-box {
    background: #EFF6FF;
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #1E3A5F;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}
.success-box {
    background: #F0FDF4;
    border-left: 3px solid #22C55E;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #14532D;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}
.warning-box {
    background: #FFFBEB;
    border-left: 3px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #78350F;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}
.neutral-box {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #374151;
    font-size: 0.875rem;
    line-height: 1.6;
    margin: 0.75rem 0;
}

/* ===== 그래프 래퍼 카드 ===== */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.chart-card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.75rem;
}

/* ===== AR/MA 추정 카드 ===== */
.order-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.order-card-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.order-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2563EB;
}
.order-desc {
    font-size: 0.78rem;
    color: #9CA3AF;
    margin-top: 0.25rem;
}

/* ===== Streamlit 기본 위젯 정리 ===== */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-color: #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    color: #111827 !important;
}
[data-testid="stSlider"] {
    color: #111827 !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #374151 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] input {
    border-color: #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    color: #111827 !important;
}

/* ===== 버튼 ===== */
[data-testid="stButton"] > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: background 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: #1D4ED8 !important;
}

/* ===== 파일 업로더 ===== */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #D1D5DB !important;
    border-radius: 10px !important;
    color: #374151 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563EB !important;
}

/* ===== metric 기본 색 보정 ===== */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
}
[data-testid="stMetricValue"] {
    color: #111827 !important;
}

/* ===== 구분선 ===== */
hr {
    border-color: #E5E7EB !important;
    margin: 1rem 0 !important;
}

/* ===== expander ===== */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}
[data-testid="stExpander"] summary {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* ===== dataframe / table ===== */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ===== 페이지 타이틀 ===== */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
}

/* ===== 섹션 제목 ===== */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin: 1.25rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ===== 상태 배지 ===== */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
}
.badge-blue   { background: #DBEAFE; color: #1E40AF; }
.badge-green  { background: #D1FAE5; color: #065F46; }
.badge-red    { background: #FEE2E2; color: #991B1B; }
.badge-yellow { background: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def card(content_fn, title=None):
    """흰색 카드로 감싸는 래퍼 (함수형)"""
    if title:
        st.markdown(f'<div class="chart-card"><div class="chart-card-title">{title}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)


def info_box(text, style="info"):
    st.markdown(f'<div class="{style}-box">{text}</div>', unsafe_allow_html=True)


def kpi_cards(metrics_dict):
    """metrics_dict: {label: (value, badge_text, badge_class)}"""
    cols = st.columns(len(metrics_dict))
    for col, (label, (value, badge_text, badge_cls)) in zip(cols, metrics_dict.items()):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div><span class="kpi-badge {badge_cls}">{badge_text}</span></div>
            </div>
            """, unsafe_allow_html=True)


def order_cards(p_val, q_val, p_desc="", q_desc=""):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="order-card">
            <div class="order-card-title">AR 차수 후보 p</div>
            <div class="order-value">{p_val}</div>
            <div class="order-desc">{p_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="order-card">
            <div class="order-card-title">MA 차수 후보 q</div>
            <div class="order-value">{q_val}</div>
            <div class="order-desc">{q_desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ── 상단 헤더 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">📈 시계열 분석 대시보드</div>
<div class="page-subtitle">CSV 업로드 또는 샘플 데이터로 전처리 · 구조 분석 · 예측까지 자동 실행</div>
""", unsafe_allow_html=True)

# ── 설정 패널 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="settings-panel"><h3>⚙️ 분석 설정</h3>', unsafe_allow_html=True)

# 1단: 데이터 소스
row1_c1, row1_c2 = st.columns([2, 2])
with row1_c1:
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], label_visibility="visible")
with row1_c2:
    sample_option = st.selectbox(
        "샘플 데이터 선택",
        ["없음 (CSV 업로드 사용)", "AirPassengers (월별 항공 승객)"],
    )

st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)

# 2단: 모델 파라미터
row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
with row2_c1:
    forecast_horizon = st.number_input("Forecast Horizon (n-step)", min_value=1, max_value=120, value=12, step=1)
with row2_c2:
    season_length = st.number_input("Season Length", min_value=1, max_value=52, value=12, step=1)
with row2_c3:
    diff_d = st.selectbox("차분 d (Differencing)", [0, 1, 2], index=1)
with row2_c4:
    train_ratio = st.slider("Train Ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.05)

st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)

# 변환 선택 (별도 줄)
transform_col, _ = st.columns([2, 4])
with transform_col:
    transform_method = st.selectbox(
        "변환 방법 (Scaler / Transform)",
        ["없음", "StandardScaler", "MinMaxScaler", "Log", "Box-Cox"],
    )

st.markdown('</div>', unsafe_allow_html=True)  # settings-panel 닫기

# ── 실행 버튼 ─────────────────────────────────────────────────────────────────
run_btn = st.button("🚀 분석 실행", use_container_width=False)

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(uploaded_file, sample_option):
    if uploaded_file is not None:
        return load_data(uploaded_file)
    elif sample_option != "없음 (CSV 업로드 사용)":
        return load_data(None, sample=True)
    return None


ts_data = None
if run_btn or ("ts_data" in st.session_state):
    with st.spinner("데이터 로딩 중..."):
        ts_data = get_data(uploaded_file, sample_option)
        if ts_data is not None:
            st.session_state["ts_data"] = ts_data
        elif "ts_data" in st.session_state:
            ts_data = st.session_state["ts_data"]

if ts_data is None:
    st.markdown("""
    <div class="info-box">
        📂 위 설정 패널에서 CSV 파일을 업로드하거나 샘플 데이터를 선택한 뒤 <b>분석 실행</b>을 클릭하세요.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── 전처리 ────────────────────────────────────────────────────────────────────
with st.spinner("전처리 중..."):
    ts_data = handle_missing(ts_data)
    ts_data, n_outliers = handle_outliers(ts_data)
    freq = infer_frequency(ts_data)

# 전처리 요약 KPI
kpi_cards({
    "데이터 포인트": (f"{len(ts_data):,}", "전처리 완료", "best"),
    "추론된 빈도": (freq or "Unknown", "자동 감지", "info"),
    "이상치 처리": (f"{n_outliers}건", "IQR 클리핑", "warn"),
    "결측치 처리": ("선형 보간", "완료", "best"),
})

# ── 메인 탭 ───────────────────────────────────────────────────────────────────
tab_raw, tab_decomp, tab_stationary, tab_acf, tab_transform, tab_model, tab_eval, tab_residual, tab_backtest = st.tabs([
    "📊 원시 데이터",
    "🔀 STL 분해",
    "📉 정상성 검정",
    "🔗 ACF / PACF",
    "🔄 변환",
    "🤖 모델링 & 예측",
    "📐 성능 평가",
    "🔍 잔차 분석",
    "🔁 백테스팅",
])

# ══════════════════════════════════════════════════════════════════════════════
# 탭 1 : 원시 데이터
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.markdown('<div class="section-title">원시 시계열 데이터</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chart-card"><div class="chart-card-title">시계열 플롯</div>', unsafe_allow_html=True)
        fig_raw = plot_series(ts_data, title="원시 시계열")
        st.plotly_chart(fig_raw, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 원시 데이터 테이블 보기"):
        st.dataframe(ts_data, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 2 : STL 분해
# ══════════════════════════════════════════════════════════════════════════════
with tab_decomp:
    st.markdown('<div class="section-title">STL 분해 (Trend · Seasonal · Residual)</div>', unsafe_allow_html=True)

    with st.spinner("STL 분해 중..."):
        stl_result = stl_decompose(ts_data, period=season_length)

    st.markdown('<div class="chart-card"><div class="chart-card-title">STL 분해 결과</div>', unsafe_allow_html=True)
    fig_stl = plot_stl(stl_result)
    st.plotly_chart(fig_stl, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    info_box("STL 분해는 시계열을 추세(Trend), 계절성(Seasonal), 잔차(Residual) 성분으로 분리합니다. 잔차가 백색잡음에 가까울수록 분해가 잘 된 것입니다.", "info")

# ══════════════════════════════════════════════════════════════════════════════
# 탭 3 : 정상성 검정
# ══════════════════════════════════════════════════════════════════════════════
with tab_stationary:
    st.markdown('<div class="section-title">정상성 검정 (ADF · KPSS)</div>', unsafe_allow_html=True)

    with st.spinner("정상성 검정 중..."):
        stat_results = stationarity_tests(ts_data, diff_d=diff_d)

    for label, result in stat_results.items():
        box_style = "success" if result.get("is_stationary") else "warning"
        adf_stat  = result.get("adf_stat",  "N/A")
        adf_p     = result.get("adf_p",     "N/A")
        kpss_stat = result.get("kpss_stat", "N/A")
        kpss_p    = result.get("kpss_p",    "N/A")
        interpretation = result.get("interpretation", "")

        st.markdown(f"""
        <div class="{box_style}-box">
            <strong>{label}</strong><br>
            ADF 통계량: {adf_stat} | ADF p-value: {adf_p}<br>
            KPSS 통계량: {kpss_stat} | KPSS p-value: {kpss_p}<br>
            {interpretation}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 4 : ACF / PACF
# ══════════════════════════════════════════════════════════════════════════════
with tab_acf:
    st.markdown('<div class="section-title">자기상관 분석 (ACF · PACF)</div>', unsafe_allow_html=True)

    with st.spinner("ACF/PACF 계산 중..."):
        acf_vals, pacf_vals, lags = compute_acf_pacf(ts_data, diff_d=diff_d)
        p_est, q_est, p_desc, q_desc = estimate_orders(acf_vals, pacf_vals)

    # AR(p) / MA(q) 후보 추정 – 흰색 카드
    st.markdown('<div class="section-title">🎯 AR(p) · MA(q) 차수 후보 추정</div>', unsafe_allow_html=True)
    order_cards(p_est, q_est, p_desc, q_desc)

    info_box(
        "PACF가 lag <i>p</i> 이후 절단(cut-off)되면 AR(<i>p</i>) 모델을, "
        "ACF가 lag <i>q</i> 이후 절단되면 MA(<i>q</i>) 모델을 제안합니다.",
        "info"
    )

    st.markdown('<div class="chart-card"><div class="chart-card-title">ACF / PACF 플롯</div>', unsafe_allow_html=True)
    fig_acf = plot_acf_pacf(acf_vals, pacf_vals, lags)
    st.plotly_chart(fig_acf, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 5 : 변환
# ══════════════════════════════════════════════════════════════════════════════
with tab_transform:
    st.markdown('<div class="section-title">변환 파이프라인</div>', unsafe_allow_html=True)

    if transform_method == "없음":
        info_box("변환 방법이 선택되지 않았습니다. 설정 패널에서 변환을 선택하세요.", "neutral")
    else:
        info_box(f"<b>{transform_method}</b> 변환이 적용됩니다. 모델링 탭에서 변환된 데이터로 학습이 진행됩니다.", "success")

# ══════════════════════════════════════════════════════════════════════════════
# 탭 6 : 모델링 & 예측
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-title">모델 학습 및 예측 비교</div>', unsafe_allow_html=True)

    with st.spinner("모델 학습 중 (ARIMA · AutoARIMA · ExponentialSmoothing · NaiveMA)..."):
        models, train_ts, test_ts = train_models(
            ts_data,
            train_ratio=train_ratio,
            season_length=season_length,
            p=p_est, d=diff_d, q=q_est,
            transform=transform_method,
        )
        future_pred = forecast_future(models, ts_data, horizon=forecast_horizon, transform=transform_method)

    st.markdown('<div class="chart-card"><div class="chart-card-title">Train / Test 예측 비교</div>', unsafe_allow_html=True)
    fig_comp = plot_forecast_comparison(train_ts, test_ts, models)
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-card-title">미래 예측 (최적 모델)</div>', unsafe_allow_html=True)
    fig_future = plot_future_forecast(ts_data, future_pred, horizon=forecast_horizon)
    st.plotly_chart(fig_future, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 7 : 성능 평가
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">모델 성능 평가</div>', unsafe_allow_html=True)

    metrics_df = compute_metrics(models, test_ts)
    best_model = metrics_df["RMSE"].idxmin()

    # KPI 카드로 최적 모델 표시
    best_rmse = metrics_df.loc[best_model, "RMSE"]
    best_mae  = metrics_df.loc[best_model, "MAE"]
    best_mape = metrics_df.loc[best_model, "MAPE"]

    kpi_cards({
        "최적 모델": (best_model, "RMSE 기준", "best"),
        "RMSE": (f"{best_rmse:.4f}", "최솟값", "best"),
        "MAE":  (f"{best_mae:.4f}",  "최솟값", "info"),
        "MAPE": (f"{best_mape:.2f}%", "최솟값", "info"),
    })

    st.markdown('<div class="chart-card"><div class="chart-card-title">전체 모델 성능 비교표</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#D1FAE5"),
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 8 : 잔차 분석
# ══════════════════════════════════════════════════════════════════════════════
with tab_residual:
    st.markdown('<div class="section-title">잔차 분석 (최적 모델 기준)</div>', unsafe_allow_html=True)

    resid_result = residual_analysis(models[best_model], test_ts)

    ljung_p   = resid_result.get("ljung_p", None)
    is_white  = resid_result.get("is_white_noise", False)
    interp    = resid_result.get("interpretation", "")

    if is_white:
        info_box(f"✅ Ljung-Box 검정 p-value: {ljung_p:.4f} → 잔차는 <b>백색잡음</b>으로 판단됩니다. {interp}", "success")
    else:
        info_box(f"⚠️ Ljung-Box 검정 p-value: {ljung_p:.4f} → 잔차에 <b>자기상관이 남아 있습니다</b>. {interp}", "warning")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown('<div class="chart-card"><div class="chart-card-title">잔차 시계열</div>', unsafe_allow_html=True)
        fig_resid = plot_residuals(resid_result)
        st.plotly_chart(fig_resid, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r2:
        st.markdown('<div class="chart-card"><div class="chart-card-title">잔차 ACF</div>', unsafe_allow_html=True)
        fig_resid_acf = plot_residuals(resid_result, mode="acf")
        st.plotly_chart(fig_resid_acf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 탭 9 : 백테스팅
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown('<div class="section-title">Rolling Window 백테스팅</div>', unsafe_allow_html=True)

    with st.spinner("백테스팅 실행 중..."):
        bt_result = run_backtest(
            models[best_model], ts_data,
            horizon=forecast_horizon,
            transform=transform_method,
        )

    bt_mae  = bt_result.get("mae",  "N/A")
    bt_rmse = bt_result.get("rmse", "N/A")

    kpi_cards({
        "백테스트 MAE":  (f"{bt_mae:.4f}"  if isinstance(bt_mae,  float) else bt_mae,  "Rolling Window", "info"),
        "백테스트 RMSE": (f"{bt_rmse:.4f}" if isinstance(bt_rmse, float) else bt_rmse, "Rolling Window", "info"),
        "사용 모델": (best_model, "최적 모델", "best"),
        "Forecast Horizon": (f"{forecast_horizon}step", "설정값", "info"),
    })

    st.markdown('<div class="chart-card"><div class="chart-card-title">백테스팅 결과 차트</div>', unsafe_allow_html=True)
    fig_bt = plot_backtest(bt_result, ts_data)
    st.plotly_chart(fig_bt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    info_box(
        "Rolling Window 백테스팅은 <b>darts historical_forecasts</b>를 활용합니다. "
        "각 시점에서 모델을 재학습하고 horizon 스텝 앞을 예측하여 실제값과 비교합니다.",
        "neutral"
    )
