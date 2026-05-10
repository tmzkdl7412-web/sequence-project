"""
app.py — 자동 시계열 분석 및 예측 대시보드
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.preprocessing import auto_detect_columns, preprocess, get_sample_data
from utils.analysis import (
    stl_decompose, stationarity_summary, difference,
    compute_acf_pacf, suggest_arima_order,
)
from utils.modeling import fit_predict_all, forecast_future, run_backtest, MODEL_NAMES
from utils.evaluation import (
    compute_metrics, find_best_model,
    compute_residuals, residual_acf, ljung_box_test,
    backtest_metrics,
)
from utils.visualization import (
    plot_timeseries, plot_stl, plot_differenced,
    plot_acf_pacf, plot_transform_comparison,
    plot_forecast_single, plot_forecast_overlay, plot_future_forecast,
    plot_metrics_bars, plot_residuals, plot_residual_acf,
    plot_backtest,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="시계열 분석 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── 전체 배경 ── */
.stApp { background: #F7F8FC; }

/* ── 사이드바 완전 숨김 ── */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── 헤더 ── */
.ts-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 12px;
    padding: 0.85rem 1.8rem;
    margin-bottom: 0.6rem;
    color: white;
}
.ts-header h1 { margin: 0; font-size: 1.25rem; font-weight: 700; }
.ts-header p  { margin: 0.15rem 0 0; font-size: 0.78rem; opacity: 0.85; }

/* ── KPI 카드 ── */
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
}
.kpi-card.green  { border-left-color: #10b981; }
.kpi-card.orange { border-left-color: #f59e0b; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.purple { border-left-color: #8b5cf6; }
.kpi-card.teal   { border-left-color: #14b8a6; }
.kpi-label { font-size: 0.7rem; color: #6b7280; font-weight: 600;
             text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 0.1rem 0 0; }
.kpi-sub   { font-size: 0.7rem; color: #9ca3af; }

/* ── 섹션 제목 ── */
.section-title {
    font-size: 1rem; font-weight: 700; color: #1e3a8a;
    border-bottom: 2px solid #dbeafe; padding-bottom: 0.3rem;
    margin: 1rem 0 0.7rem;
}

/* ── 해석 박스 ── */
.interp-box {
    background: #eff6ff; border-left: 3px solid #3b82f6;
    border-radius: 6px; padding: 0.6rem 0.9rem;
    font-size: 0.855rem; color: #1e40af; margin: 0.5rem 0;
}
.interp-warn {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    border-radius: 6px; padding: 0.6rem 0.9rem;
    font-size: 0.855rem; color: #92400e; margin: 0.5rem 0;
}
.interp-ok {
    background: #f0fdf4; border-left: 3px solid #10b981;
    border-radius: 6px; padding: 0.6rem 0.9rem;
    font-size: 0.855rem; color: #166534; margin: 0.5rem 0;
}

/* ── 파라미터 패널 sticky ── */
[data-testid="stVerticalBlock"]:has(#param-anchor) {
    position: sticky !important;
    top: 0 !important;
    z-index: 200 !important;
    background: #F7F8FC !important;
    padding-bottom: 4px !important;
    border-bottom: 2px solid #dbeafe !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.07) !important;
}

/* ── 위젯 레이블 검은색 ── */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
.stSlider label, .stSlider p,
.stSelectbox label, .stSelectbox p,
.stMultiSelect label, .stMultiSelect p,
.stCheckbox label, .stCheckbox p,
.stFileUploader label {
    font-size: 0.78rem !important;
    color: #1f2937 !important;
    font-weight: 500 !important;
}

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #e5e7eb; border-radius: 10px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-size: 0.8rem; font-weight: 600;
    padding: 0.35rem 0.65rem; color: #6b7280;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #1e3a8a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

/* ── 버튼 ── */
.stButton > button {
    background: #1e3a8a; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.4rem 1.1rem;
}
.stButton > button:hover { background: #1d4ed8; }

/* ── 테이블 ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  헤더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="ts-header">
  <h1>📈 자동 시계열 분석 및 예측 시스템</h1>
  <p>CSV 업로드 → 전처리 → 구조 분석(STL·정상성·ACF/PACF) → 모델링(ARIMA·AutoARIMA·ES·NMA) → 성능 평가 → 잔차 분석 → 백테스팅</p>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  파라미터 패널 (상단 가로 고정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<span id="param-anchor"></span>', unsafe_allow_html=True)

# 행 1: 데이터 소스 + 예측 파라미터
pc1, pc2, pc3, pc4, pc5, pc6 = st.columns([2.2, 1.0, 1.0, 0.75, 1.0, 1.8])

with pc1:
    uploaded = st.file_uploader(
        "CSV 파일 업로드", type=["csv"],
        help="날짜 + 값 컬럼이 있는 단변량 시계열 CSV",
    )
    use_sample = st.checkbox(
        "샘플 데이터 사용 (월별 소매 판매 지수)",
        value=(uploaded is None),
    )
with pc2:
    forecast_horizon = st.slider("Forecast H", 1, 120, 24, help="예측 시점 수")
with pc3:
    season_length = st.slider("계절 주기 m", 2, 52, 12, help="STL/모델 계절 주기")
with pc4:
    diff_d = st.slider("차분 d", 0, 2, 0, help="ARIMA d 파라미터")
with pc5:
    train_ratio = st.slider("Train 비율", 0.5, 0.95, 0.8, step=0.05)
with pc6:
    transform_list = st.multiselect(
        "변환 선택",
        ["StandardScaler", "MinMaxScaler", "Log", "Box-Cox"],
        default=[],
    )

# 데이터 로드
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 파싱 오류: {e}")
        st.stop()
elif use_sample:
    raw_df = get_sample_data()
else:
    st.info("CSV를 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

# 행 2: 컬럼 선택
detected_date, detected_value = auto_detect_columns(raw_df)
all_cols = list(raw_df.columns)

cc1, cc2, _ = st.columns([1.5, 1.5, 4.8])
with cc1:
    date_col = st.selectbox(
        "날짜 컬럼", all_cols,
        index=all_cols.index(detected_date) if detected_date in all_cols else 0,
    )
with cc2:
    value_candidates = [c for c in all_cols if c != date_col]
    value_col = st.selectbox(
        "값 컬럼", value_candidates,
        index=(value_candidates.index(detected_value)
               if detected_value in value_candidates else 0),
    )

st.markdown(
    '<div style="height:3px;background:linear-gradient(90deg,#3b82f6,#10b981);'
    'border-radius:2px;margin:4px 0 10px;"></div>',
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  전처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    df, ts, info = preprocess(raw_df, date_col, value_col)
except Exception as e:
    st.error(f"전처리 실패: {e}")
    st.stop()

effective_season = season_length
if season_length * 2 > info["length"]:
    effective_season = max(2, info["length"] // 4)
    st.warning(
        f"계절 주기({season_length})가 데이터 길이({info['length']})에 비해 큽니다. "
        f"자동으로 {effective_season}로 조정됩니다."
    )

split_idx = int(len(ts) * train_ratio)
split_idx = max(split_idx, 10)
split_idx = min(split_idx, len(ts) - 2)
train, test = ts.split_before(split_idx)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ACF/PACF → p, q 추정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
analysis_vals = (difference(df["value"], diff_d).values
                 if diff_d > 0 else df["value"].values)
acf_v, pacf_v, conf = compute_acf_pacf(analysis_vals)
p_est, q_est, order_text = suggest_arima_order(acf_v, pacf_v, conf)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  모델 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.spinner("모델 학습 중… ⏳"):
    predictions, models, errors = fit_predict_all(
        train, test, forecast_horizon,
        p=p_est, d=diff_d, q=q_est,
        season_length=effective_season,
    )

metrics_df = compute_metrics(test, predictions) if predictions else pd.DataFrame()
best_name  = find_best_model(metrics_df, "RMSE") if not metrics_df.empty else None
best_rmse  = (metrics_df.loc[metrics_df["모델"] == best_name, "RMSE"].values[0]
              if best_name else None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KPI 카드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
kpi_cols = st.columns(6)
_kpi_data = [
    ("blue",   "데이터 길이",  f"{info['length']}행",   f"Train {len(train)} / Test {len(test)}"),
    ("green",  "추정 빈도",    info["freq"],             "자동 탐지"),
    ("orange", "결측치 처리",  f"{info['missing']}개",   "선형 보간"),
    ("red",    "이상치 처리",  f"{info['outliers']}개",  "IQR 클리핑"),
    ("purple", "최적 모델",    best_name or "—",         "RMSE 기준"),
    ("teal",   "Best RMSE",   f"{best_rmse:.4f}" if best_rmse else "—", "테스트셋 기준"),
]
for col, (cls, label, val, sub) in zip(kpi_cols, _kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  탭 구성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(tab_data, tab_struct, tab_stat, tab_acf,
 tab_trans, tab_model, tab_eval, tab_resid, tab_bt) = st.tabs([
    "📋 데이터 개요",
    "📊 시계열 구조",
    "📐 정상성·차분",
    "📈 ACF/PACF",
    "🔄 변환",
    "🤖 모델링·예측",
    "📋 성능 평가",
    "🔍 잔차 분석",
    "🔁 백테스팅",
])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — 데이터 개요
# ══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">원본 시계열 시각화</div>',
                unsafe_allow_html=True)
    st.plotly_chart(plot_timeseries(df), use_container_width=True, key="ts_overview")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">기초 통계량</div>',
                    unsafe_allow_html=True)
        desc = df["value"].describe().rename("통계량").to_frame()
        desc.index.name = "항목"
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)
    with col_r:
        st.markdown('<div class="section-title">데이터 미리보기 (처음 20행)</div>',
                    unsafe_allow_html=True)
        preview = df.reset_index()
        preview.columns = ["날짜", "값"]
        st.dataframe(preview.head(20), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="interp-box">
    데이터 기간: <b>{df.index[0].date()}</b> ~ <b>{df.index[-1].date()}</b> &nbsp;|&nbsp;
    총 <b>{info['length']}</b>행 &nbsp;|&nbsp; 추정 빈도: <b>{info['freq']}</b><br>
    결측치 <b>{info['missing']}</b>개 선형 보간 처리, 이상치 <b>{info['outliers']}</b>개 IQR 클리핑 처리 완료
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — 시계열 구조 (STL)
# ══════════════════════════════════════════════════════════════════════════
with tab_struct:
    st.markdown('<div class="section-title">STL 분해 (Seasonal-Trend decomposition using LOESS)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="interp-box">
    STL은 시계열을 <b>Trend(추세)</b> + <b>Seasonal(계절성)</b> + <b>Residual(잔차)</b>로 분리합니다.
    현재 계절 주기 = <b>{effective_season}</b>
    </div>
    """, unsafe_allow_html=True)

    try:
        stl_result = stl_decompose(df["value"], effective_season)
        st.plotly_chart(plot_stl(stl_result, df.index), use_container_width=True,
                        key="stl_chart")

        trend_range = stl_result.trend.max() - stl_result.trend.min()
        seas_range  = stl_result.seasonal.max() - stl_result.seasonal.min()
        resid_std   = float(np.std(stl_result.resid))

        col1, col2, col3 = st.columns(3)
        col1.metric("추세 변화폭", f"{trend_range:.2f}")
        col2.metric("계절 진폭",   f"{seas_range:.2f}")
        col3.metric("잔차 표준편차", f"{resid_std:.4f}")

        st.markdown(f"""
        <div class="interp-box">
        • Trend: 장기적인 수준 변화를 나타냅니다. 변화폭이 크면 비정상 시계열일 가능성이 높습니다.<br>
        • Seasonal: 주기적 반복 패턴. 계절 진폭({seas_range:.2f})이 크면 계절 모델(SARIMA, ES)이 유리합니다.<br>
        • Residual: 추세·계절 제거 후 남은 성분. 표준편차({resid_std:.4f})가 작을수록 분해가 잘 된 것입니다.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"STL 분해 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 — 정상성 검정 & 차분
# ══════════════════════════════════════════════════════════════════════════
with tab_stat:
    st.markdown('<div class="section-title">정상성 검정 (ADF · KPSS)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="interp-box">
    ARIMA 모델링의 전제 조건은 <b>정상성(Stationarity)</b>입니다.
    ADF p &lt; 0.05 → 정상 지지 &nbsp;|&nbsp; KPSS p &gt; 0.05 → 정상 지지
    </div>
    """, unsafe_allow_html=True)

    test_series_raw = df["value"].values
    if diff_d > 0:
        diffed_series = difference(df["value"], diff_d)
        test_series = diffed_series.values

        st.markdown(f'<div class="section-title">차분 전/후 비교 (d={diff_d})</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            plot_differenced(df["value"], diffed_series, diff_d),
            use_container_width=True, key="diff_chart",
        )
        st.markdown("""
        <div class="interp-box">
        ACF가 천천히 감소하거나 추세가 뚜렷하면 차분이 필요합니다.
        차분 후 ACF가 빠르게 0에 수렴하면 정상성 확보된 것입니다.
        </div>
        """, unsafe_allow_html=True)
    else:
        test_series = test_series_raw
        st.info("상단 파라미터에서 차분 횟수(d)를 1 이상으로 설정하면 차분 비교 그래프가 표시됩니다.")

    adf_res, kpss_res, interpretation = stationarity_summary(test_series)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">ADF 검정 결과</div>',
                    unsafe_allow_html=True)
        adf_df = pd.DataFrame([adf_res]).T.rename(columns={0: "값"})
        st.dataframe(adf_df, use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">KPSS 검정 결과</div>',
                    unsafe_allow_html=True)
        kpss_df = pd.DataFrame([kpss_res]).T.rename(columns={0: "값"})
        st.dataframe(kpss_df, use_container_width=True)

    both_stat = ("정상" in adf_res["판정"] and "정상" in kpss_res["판정"])
    box_cls = "interp-ok" if both_stat else "interp-warn"
    st.markdown(f'<div class="{box_cls}">{interpretation}</div>',
                unsafe_allow_html=True)

    if diff_d == 0:
        st.markdown('<div class="section-title">원본 정상성 검정 (참고)</div>',
                    unsafe_allow_html=True)
        adf_raw, kpss_raw, _ = stationarity_summary(test_series_raw)
        col1, col2 = st.columns(2)
        col1.metric("ADF 판정 (원본)", adf_raw["판정"], delta=f"p={adf_raw['p-value']}")
        col2.metric("KPSS 판정 (원본)", kpss_raw["판정"], delta=f"p={kpss_raw['p-value']}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 4 — ACF / PACF
# ══════════════════════════════════════════════════════════════════════════
with tab_acf:
    st.markdown('<div class="section-title">ACF / PACF 분석</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="interp-box">
    분석 대상: {'차분 적용 시계열 (d=' + str(diff_d) + ')' if diff_d > 0 else '원본 시계열'}<br>
    • ACF(자기상관함수): MA 차수 q 판단 — 신뢰구간을 벗어난 lag까지가 q 후보<br>
    • PACF(편자기상관함수): AR 차수 p 판단 — 신뢰구간을 벗어난 lag까지가 p 후보<br>
    • 빨간 점선 = 95% 신뢰구간 (±{conf:.4f})
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(plot_acf_pacf(acf_v, pacf_v, conf), use_container_width=True,
                    key="acf_pacf_chart")

    st.markdown('<div class="section-title">AR(p) / MA(q) 후보 추정</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("추정 AR 차수 p", p_est, help="PACF 기반")
    col2.metric("추정 MA 차수 q", q_est, help="ACF 기반")

    for line in order_text.split("\n\n"):
        box_cls = "interp-ok" if "존재 가능" in line else "interp-box"
        st.markdown(f'<div class="{box_cls}">{line}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interp-warn">
    ARIMA 파라미터 가이드:<br>
    ACF가 천천히 감소 → 차분 필요 (d 증가) &nbsp;|&nbsp;
    PACF 급절단 → AR 모델 적합 &nbsp;|&nbsp;
    ACF 급절단 → MA 모델 적합 &nbsp;|&nbsp;
    둘 다 서서히 감소 → ARMA 혼합 모델
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 5 — 변환 파이프라인
# ══════════════════════════════════════════════════════════════════════════
with tab_trans:
    st.markdown('<div class="section-title">Darts 변환 파이프라인</div>',
                unsafe_allow_html=True)

    if not transform_list:
        st.markdown("""
        <div class="interp-box">
        상단 파라미터에서 변환을 선택하면 원본 시계열과 변환된 시계열을 비교합니다.<br>
        변환 목적: 분산 안정화(Log·Box-Cox), 값 범위 정규화(StandardScaler·MinMaxScaler)
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_timeseries(df), use_container_width=True,
                        key="ts_transform_raw")
        ts_transformed = ts
    else:
        ts_transformed = ts
        for tname in transform_list:
            try:
                if tname == "StandardScaler":
                    scaler = Scaler(StandardScaler())
                    ts_transformed = scaler.fit_transform(ts_transformed)
                elif tname == "MinMaxScaler":
                    scaler = Scaler(MinMaxScaler())
                    ts_transformed = scaler.fit_transform(ts_transformed)
                elif tname == "Log":
                    vals = ts_transformed.values()
                    shift = max(0, -vals.min() + 1) if vals.min() <= 0 else 0
                    ts_transformed = ts_transformed.with_values(np.log(vals + shift))
                    st.caption(f"Log 변환: shift={shift:.2f}")
                elif tname == "Box-Cox":
                    from scipy.stats import boxcox
                    vals = ts_transformed.values().flatten()
                    shift = max(0, -vals.min() + 1) if vals.min() <= 0 else 0
                    bc_vals, lmbda = boxcox(vals + shift)
                    ts_transformed = ts_transformed.with_values(bc_vals.reshape(-1, 1))
                    st.caption(f"Box-Cox λ = {lmbda:.4f}")

                st.plotly_chart(
                    plot_transform_comparison(ts, ts_transformed, tname),
                    use_container_width=True, key=f"transform_{tname}",
                )
                st.markdown(f'<div class="interp-ok">{tname} 변환 적용 완료</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"{tname} 변환 실패: {e}")

    st.markdown("""
    <div class="interp-box">
    변환은 시각화·비교 목적입니다. 모델 학습에는 원본 데이터를 사용합니다.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 6 — 모델링 & 예측
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    if errors:
        for mname, err in errors.items():
            st.warning(f"⚠️ {mname} 학습 실패 → 스킵: {err}")

    if not predictions:
        st.error("모든 모델이 실패했습니다. 파라미터를 조정해 주세요.")
        st.stop()

    st.markdown(f"""
    <div class="interp-box">
    Train: <b>{len(train)}</b>행 &nbsp;|&nbsp; Test: <b>{len(test)}</b>행 &nbsp;|&nbsp;
    Forecast Horizon: <b>{min(forecast_horizon, len(test))}</b> &nbsp;|&nbsp;
    추정 ARIMA 차수: p=<b>{p_est}</b>, d=<b>{diff_d}</b>, q=<b>{q_est}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">모델별 예측 결과</div>',
                unsafe_allow_html=True)
    pred_items = list(predictions.items())
    for i in range(0, len(pred_items), 2):
        cols = st.columns(2)
        for j, (mname, pred) in enumerate(pred_items[i:i+2]):
            with cols[j]:
                st.plotly_chart(
                    plot_forecast_single(train, test, pred, mname),
                    use_container_width=True, key=f"forecast_single_{mname}",
                )

    st.markdown('<div class="section-title">전체 모델 비교 오버레이</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        plot_forecast_overlay(train, test, predictions),
        use_container_width=True, key="forecast_overlay",
    )

    st.markdown('<div class="section-title">미래 예측 (데이터 종료 이후)</div>',
                unsafe_allow_html=True)

    future_model_name = best_name if best_name else list(predictions.keys())[0]
    future_model = models[future_model_name]
    future_pred, fut_err = forecast_future(
        future_model, ts, forecast_horizon, future_model_name
    )
    if future_pred is not None:
        st.plotly_chart(
            plot_future_forecast(ts, future_pred, future_model_name),
            use_container_width=True, key="future_forecast",
        )
        st.markdown(f"""
        <div class="interp-ok">
        최적 모델 <b>{future_model_name}</b>으로 <b>{forecast_horizon}</b>-step 미래 예측 완료.
        전체 데이터로 재학습 후 예측합니다 (Recursive forecasting).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"미래 예측 실패: {fut_err}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 7 — 성능 평가
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">모델 성능 비교 (MAE · RMSE · MAPE)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="interp-box">
    • MAE(평균절대오차): 예측 오차의 평균 크기<br>
    • RMSE(제곱근평균오차): 큰 오차에 더 민감 — 최적 모델 선정 기준<br>
    • MAPE(평균절대백분율오차): 상대적 오차율(%), 스케일 무관 비교 가능
    </div>
    """, unsafe_allow_html=True)

    def _highlight_best(row):
        if row["모델"] == best_name:
            return ["background-color:#dbeafe; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style
                  .apply(_highlight_best, axis=1)
                  .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    if best_name:
        st.markdown(f"""
        <div class="interp-ok">
        최적 모델 (RMSE 기준): <b>{best_name}</b> &nbsp;|&nbsp; RMSE = <b>{best_rmse:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    st.plotly_chart(
        plot_metrics_bars(metrics_df, best_name),
        use_container_width=True, key="metrics_bars",
    )


# ══════════════════════════════════════════════════════════════════════════
#  TAB 8 — 잔차 분석
# ══════════════════════════════════════════════════════════════════════════
with tab_resid:
    st.markdown('<div class="section-title">잔차 분석 — 최적 모델 기준</div>',
                unsafe_allow_html=True)

    if best_name and best_name in predictions:
        st.markdown(f"""
        <div class="interp-box">
        분석 모델: <b>{best_name}</b> (RMSE 최적)<br>
        좋은 모델의 잔차는 <b>백색잡음(White Noise)</b>에 가까워야 합니다 — 평균 0, 자기상관 없음, 등분산
        </div>
        """, unsafe_allow_html=True)

        best_pred = predictions[best_name]
        residuals = compute_residuals(test, best_pred)
        time_idx  = test[:len(residuals)].time_index

        st.plotly_chart(
            plot_residuals(residuals, best_name, time_idx),
            use_container_width=True, key="residuals_chart",
        )

        resid_mean = float(np.mean(residuals))
        resid_std  = float(np.std(residuals))
        col1, col2 = st.columns(2)
        col1.metric("잔차 평균", f"{resid_mean:.4f}", help="0에 가까울수록 편향 없음")
        col2.metric("잔차 표준편차", f"{resid_std:.4f}")

        st.markdown('<div class="section-title">잔차 ACF — 자기상관 잔존 여부</div>',
                    unsafe_allow_html=True)
        r_acf, r_conf = residual_acf(residuals)
        st.plotly_chart(
            plot_residual_acf(r_acf, r_conf, best_name),
            use_container_width=True, key="residual_acf_chart",
        )
        st.markdown("""
        <div class="interp-box">
        잔차 ACF가 신뢰구간(빨간 점선) 안에 있으면 자기상관 없음 → 백색잡음에 가까움<br>
        유의한 spike가 남아 있으면 모델 차수(p, q) 조정이 필요합니다.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Ljung-Box 검정</div>',
                    unsafe_allow_html=True)
        lb_df, lb_text = ljung_box_test(residuals)
        st.dataframe(
            lb_df.rename(columns={"lb_stat": "LB 통계량", "lb_pvalue": "p-value"})
                 .style.format({"LB 통계량": "{:.4f}", "p-value": "{:.4f}"}),
            use_container_width=True,
        )
        last_p = lb_df["lb_pvalue"].iloc[-1]
        box_cls = "interp-ok" if last_p > 0.05 else "interp-warn"
        st.markdown(f'<div class="{box_cls}">{lb_text}</div>', unsafe_allow_html=True)

    else:
        st.warning("예측 결과가 없습니다. 모델링·예측 탭을 먼저 확인하세요.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 9 — 백테스팅
# ══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-title">백테스팅 (Rolling Historical Forecasts)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="interp-box">
    Rolling 방식으로 과거 구간마다 반복 학습·예측하여 단순 Train/Test 분할보다 신뢰성 높은 성능 추정을 제공합니다.<br>
    start 비율(60~85%) 지점부터 슬라이딩 윈도우로 예측을 수행합니다.
    </div>
    """, unsafe_allow_html=True)

    if predictions:
        bt_model_name = st.selectbox(
            "백테스팅 모델 선택",
            [n for n in MODEL_NAMES if n in predictions],
        )

        if st.button("백테스팅 실행", type="primary"):
            with st.spinner(f"{bt_model_name} 백테스팅 수행 중… ⏳"):
                hist_fc, bt_err = run_backtest(
                    ts, bt_model_name, forecast_horizon,
                    p=p_est, d=diff_d, q=q_est,
                    season_length=effective_season,
                )

            if hist_fc is not None:
                st.plotly_chart(
                    plot_backtest(ts, hist_fc, bt_model_name),
                    use_container_width=True, key="backtest_chart",
                )

                bt_mae, bt_rmse = backtest_metrics(ts, hist_fc)
                if bt_mae is not None:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Backtest MAE",  f"{bt_mae:.4f}")
                    mc2.metric("Backtest RMSE", f"{bt_rmse:.4f}")

                    if best_rmse:
                        ratio = bt_rmse / best_rmse
                        if ratio < 1.5:
                            st.markdown(f"""
                            <div class="interp-ok">
                            Backtest RMSE({bt_rmse:.4f}) ≈ Test RMSE({best_rmse:.4f}) — 모델이 과적합 없이 안정적입니다.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="interp-warn">
                            Backtest RMSE({bt_rmse:.4f}) &gt;&gt; Test RMSE({best_rmse:.4f}) — 과적합 또는 파라미터 조정이 필요합니다.
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("백테스팅 지표 계산 실패")
            else:
                st.error(f"백테스팅 실패: {bt_err}")
    else:
        st.error("예측 결과가 없습니다.")

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
    "자동 시계열 분석 시스템 &nbsp;|&nbsp; Darts · statsmodels · Plotly · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
