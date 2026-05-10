"""
app.py — 자동 시계열 분석 및 예측 대시보드
"""

import warnings
import hashlib
import json as _json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: #FFFFFF !important; }

/* 화면 상단 잘림 방지 + 본문 중앙 정렬 */
.block-container {
    max-width: 1240px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    margin: 0 auto !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

.ts-header {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 0rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    overflow: visible !important;
}
.ts-header h1 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.35;
}
.ts-header p {
    margin: 0.35rem 0 0;
    font-size: 0.78rem;
    color: #6B7280;
    line-height: 1.5;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 0;
    min-height: 100px;
}
.kpi-card.green  { border-left-color: #10b981; }
.kpi-card.orange { border-left-color: #f59e0b; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.purple { border-left-color: #8b5cf6; }
.kpi-card.teal   { border-left-color: #14b8a6; }
.kpi-label { font-size: 0.7rem; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; line-height: 1.35; }
.kpi-value { font-size: 1.2rem; font-weight: 700; color: #111827; margin: 0.2rem 0 0; line-height: 1.35; }
.kpi-sub { font-size: 0.68rem; color: #9ca3af; margin-top: 0.1rem; line-height: 1.35; }

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 0.3rem;
    margin: 1rem 0 0.7rem;
    line-height: 1.4;
}

.interp-box {
    background: #EFF6FF;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 0.95rem;
    font-size: 0.855rem;
    color: #1E3A5F;
    margin: 0.5rem 0;
    line-height: 1.6;
    max-width: 100%;
    overflow-wrap: break-word;
    word-break: keep-all;
    white-space: normal;
}
.interp-warn {
    background: #FFFBEB;
    border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 0.95rem;
    font-size: 0.855rem;
    color: #78350F;
    margin: 0.5rem 0;
    line-height: 1.6;
    max-width: 100%;
    overflow-wrap: break-word;
    word-break: keep-all;
    white-space: normal;
}
.interp-ok {
    background: #F0FDF4;
    border-left: 3px solid #10b981;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 0.95rem;
    font-size: 0.855rem;
    color: #14532D;
    margin: 0.5rem 0;
    line-height: 1.6;
    max-width: 100%;
    overflow-wrap: break-word;
    word-break: keep-all;
    white-space: normal;
}

[data-testid="stVerticalBlock"]:has(#param-anchor) {
    position: sticky !important;
    top: 0 !important;
    z-index: 200 !important;
}

div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
.stSlider label, .stSlider p,
.stSelectbox label, .stSelectbox p,
.stMultiSelect label, .stMultiSelect p,
.stCheckbox label, .stCheckbox p,
.stFileUploader label {
    font-size: 0.78rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #F3F4F6; border-radius: 10px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-size: 0.8rem; font-weight: 600;
    padding: 0.35rem 0.65rem; color: #6B7280;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #2563EB !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

.stButton > button {
    background: #2563EB; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 0.4rem 1.1rem;
}
.stButton > button:hover { background: #1d4ed8; }

.stDataFrame { border-radius: 8px; overflow: hidden; }

[data-testid="stMetricLabel"] { color: #6B7280 !important; }
[data-testid="stMetricValue"] { color: #111827 !important; }

/* 파일 업로더 높이 압축 */
[data-testid="stFileUploader"] section {
    padding: 0.35rem 0.6rem !important;
    min-height: 50px !important;
}
[data-testid="stFileUploadDropzone"] {
    padding: 0.35rem !important;
    min-height: 50px !important;
}
[data-testid="stFileUploadDropzone"] div { font-size: 0.72rem !important; }
[data-testid="stFileUploadDropzone"] span { display: none !important; }

/* 슬라이더 수직 압축 */
[data-testid="stSlider"] {
    padding-top: 0.1rem !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

div[data-testid="stHorizontalBlock"],
div[data-testid="stVerticalBlock"] {
    max-width: 100% !important;
}

hr {
    margin-top: 1rem !important;
    margin-bottom: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── 헤더 ──
st.markdown("""
<div class="ts-header">
  <h1>자동 시계열 분석 및 예측 시스템</h1>
  <p>CSV 업로드 → 전처리 → 구조 분석(STL·정상성·ACF/PACF) → 모델링(ARIMA·AutoARIMA·ES·NMA) → 성능 평가 → 잔차 분석 → 백테스팅</p>
</div>
""", unsafe_allow_html=True)

# ── 파라미터 패널 ──
st.markdown('<span id="param-anchor"></span>', unsafe_allow_html=True)

pc1, pc2, pc3, pc4, pc5, pc6 = st.columns([2.5, 1.1, 1.1, 0.9, 1.1, 1.6])
with pc1:
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"],
                                help="날짜 + 값 컬럼이 있는 단변량 시계열 CSV")
    use_sample = st.checkbox("샘플 데이터 사용 (월별 소매 판매 지수)",
                             value=(uploaded is None))
with pc2:
    forecast_horizon = st.slider("Forecast H", 1, 120, 24)
with pc3:
    season_length = st.slider("계절 주기 m", 2, 52, 12)
with pc4:
    diff_d = st.slider("차분 d", 0, 2, 0)
with pc5:
    train_ratio = st.slider("Train 비율", 0.5, 0.95, 0.8, step=0.05)
with pc6:
    transform_list = st.multiselect("변환 선택",
                                    ["StandardScaler", "MinMaxScaler", "Log", "Box-Cox"],
                                    default=[])

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

detected_date, detected_value = auto_detect_columns(raw_df)
all_cols = list(raw_df.columns)
cc1, cc2, _ = st.columns([1.5, 1.5, 4.8])
with cc1:
    date_col = st.selectbox("날짜 컬럼", all_cols,
                            index=all_cols.index(detected_date) if detected_date in all_cols else 0)
with cc2:
    value_candidates = [c for c in all_cols if c != date_col]
    value_col = st.selectbox("값 컬럼", value_candidates,
                             index=(value_candidates.index(detected_value)
                                    if detected_value in value_candidates else 0))

st.markdown('<div style="height:3px;background:linear-gradient(90deg,#3b82f6,#10b981);'
            'border-radius:2px;margin:8px 0 12px;"></div>', unsafe_allow_html=True)

# JS: 파라미터 패널 흰색 배경
components.html("""
<script>
(function(){
  function run(){
    try{
      var doc=window.parent.document;
      var anchor=doc.getElementById('param-anchor');
      if(!anchor){setTimeout(run,400);return;}
      var el=anchor, target=null;
      while(el){
        var cls=el.className||'';
        if(cls.includes('block-container')||cls.includes('stMainBlock')){break;}
        if(el.getAttribute&&el.getAttribute('data-testid')==='stVerticalBlock'){target=el;}
        el=el.parentElement;
      }
      if(!target){setTimeout(run,400);return;}
      target.style.background='#FFFFFF';
      target.style.paddingBottom='4px';
      target.style.borderBottom='2px solid #E5E7EB';
      target.style.boxShadow='0 4px 16px rgba(0,0,0,0.06)';
      target.querySelectorAll('p,label,small').forEach(function(e){
        e.style.color='#374151';
      });
    }catch(e){console.warn(e);}
  }
  run();setTimeout(run,800);setTimeout(run,2500);
})();
</script>
""", height=0, scrolling=False)

# ── 전처리 ──
try:
    df, ts, info = preprocess(raw_df, date_col, value_col)
except Exception as e:
    st.error(f"전처리 실패: {e}")
    st.stop()

effective_season = season_length
if season_length * 2 > info["length"]:
    effective_season = max(2, info["length"] // 4)
    st.warning(f"계절 주기({season_length})가 데이터 길이({info['length']})에 비해 큽니다. "
               f"자동으로 {effective_season}로 조정됩니다.")

split_idx = max(10, min(int(len(ts) * train_ratio), len(ts) - 2))
train, test = ts.split_before(split_idx)

analysis_vals = (difference(df["value"], diff_d).values if diff_d > 0 else df["value"].values)
acf_v, pacf_v, conf = compute_acf_pacf(analysis_vals)
p_est, q_est, order_text = suggest_arima_order(acf_v, pacf_v, conf)

adf_res_global, kpss_res_global, _ = stationarity_summary(analysis_vals)

with st.spinner("모델 학습 중..."):
    predictions, models, errors = fit_predict_all(
        train, test, forecast_horizon,
        p=p_est, d=diff_d, q=q_est, season_length=effective_season,
    )

metrics_df = compute_metrics(test, predictions) if predictions else pd.DataFrame()
best_name  = find_best_model(metrics_df, "RMSE") if not metrics_df.empty else None
best_rmse  = (metrics_df.loc[metrics_df["모델"] == best_name, "RMSE"].values[0]
              if best_name else None)

# ── KPI 카드 ──
kpi_cols = st.columns(6)
for col, (cls, label, val, sub) in zip(kpi_cols, [
    ("blue",   "데이터 길이",  f"{info['length']}행",   f"Train {len(train)} / Test {len(test)}"),
    ("green",  "추정 빈도",    info["freq"],             "자동 탐지"),
    ("orange", "결측치 처리",  f"{info['missing']}개",   "선형 보간"),
    ("red",    "이상치 처리",  f"{info['outliers']}개",  "IQR 클리핑"),
    ("purple", "최적 모델",    best_name or "—",         "RMSE 기준"),
    ("teal",   "Best RMSE",   f"{best_rmse:.4f}" if best_rmse else "—", "테스트셋 기준"),
]):
    with col:
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',
                    unsafe_allow_html=True)

st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

# ── 탭 ──
(tab_data, tab_struct, tab_stat, tab_acf, tab_trans,
 tab_model, tab_eval, tab_resid, tab_bt, tab_ai) = st.tabs([
    "데이터 개요", "시계열 구조", "정상성·차분", "ACF/PACF", "변환",
    "모델링·예측", "성능 평가", "잔차 분석", "백테스팅", "결과 요약",
])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — 데이터 개요
# ══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">원본 시계열 시각화</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_timeseries(df), use_container_width=True, key="ts_overview")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">기초 통계량</div>', unsafe_allow_html=True)
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
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — 시계열 구조
# ══════════════════════════════════════════════════════════════════════════
with tab_struct:
    st.markdown('<div class="section-title">STL 분해 (Seasonal-Trend decomposition using LOESS)</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="interp-box">STL은 시계열을 <b>Trend(추세)</b> + <b>Seasonal(계절성)</b> + '
                f'<b>Residual(잔차)</b>로 분리합니다. 현재 계절 주기 = <b>{effective_season}</b></div>',
                unsafe_allow_html=True)
    try:
        stl_result = stl_decompose(df["value"], effective_season)
        st.plotly_chart(plot_stl(stl_result, df.index), use_container_width=True, key="stl_chart")
        tr = stl_result.trend.max() - stl_result.trend.min()
        sr = stl_result.seasonal.max() - stl_result.seasonal.min()
        rs = float(np.std(stl_result.resid))
        c1, c2, c3 = st.columns(3)
        c1.metric("추세 변화폭", f"{tr:.2f}")
        c2.metric("계절 진폭", f"{sr:.2f}")
        c3.metric("잔차 표준편차", f"{rs:.4f}")
        st.markdown(f"""<div class="interp-box">
        Trend: 장기적인 수준 변화. 변화폭이 크면 비정상 시계열일 가능성이 높습니다.<br>
        Seasonal: 주기적 반복 패턴. 계절 진폭({sr:.2f})이 크면 계절 모델(SARIMA, ES)이 유리합니다.<br>
        Residual: 추세·계절 제거 후 남은 성분. 표준편차({rs:.4f})가 작을수록 분해가 잘 된 것입니다.
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"STL 분해 실패: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 — 정상성·차분
# ══════════════════════════════════════════════════════════════════════════
with tab_stat:
    st.markdown('<div class="section-title">정상성 검정 (ADF · KPSS)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="interp-box">ARIMA 모델링의 전제 조건은 <b>정상성(Stationarity)</b>입니다.
    ADF p &lt; 0.05 → 정상 지지 &nbsp;|&nbsp; KPSS p &gt; 0.05 → 정상 지지</div>""",
                unsafe_allow_html=True)

    test_series_raw = df["value"].values
    if diff_d > 0:
        diffed_series = difference(df["value"], diff_d)
        test_series = diffed_series.values
        st.markdown(f'<div class="section-title">차분 전/후 비교 (d={diff_d})</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_differenced(df["value"], diffed_series, diff_d),
                        use_container_width=True, key="diff_chart")
        st.markdown("""<div class="interp-box">ACF가 천천히 감소하거나 추세가 뚜렷하면 차분이 필요합니다.
        차분 후 ACF가 빠르게 0에 수렴하면 정상성 확보된 것입니다.</div>""", unsafe_allow_html=True)
    else:
        test_series = test_series_raw
        st.info("상단에서 차분 횟수(d)를 1 이상으로 설정하면 차분 비교 그래프가 표시됩니다.")

    adf_res, kpss_res, interpretation = stationarity_summary(test_series)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">ADF 검정 결과</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([adf_res]).T.rename(columns={0: "값"}), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">KPSS 검정 결과</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([kpss_res]).T.rename(columns={0: "값"}), use_container_width=True)

    both_stat = "정상" in adf_res["판정"] and "정상" in kpss_res["판정"]
    st.markdown(f'<div class="{"interp-ok" if both_stat else "interp-warn"}">{interpretation}</div>',
                unsafe_allow_html=True)

    if diff_d == 0:
        st.markdown('<div class="section-title">원본 정상성 검정 (참고)</div>', unsafe_allow_html=True)
        ar, kr, _ = stationarity_summary(test_series_raw)
        c1, c2 = st.columns(2)
        c1.metric("ADF 판정 (원본)", ar["판정"], delta=f"p={ar['p-value']}")
        c2.metric("KPSS 판정 (원본)", kr["판정"], delta=f"p={kr['p-value']}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 4 — ACF/PACF
# ══════════════════════════════════════════════════════════════════════════
with tab_acf:
    st.markdown('<div class="section-title">ACF / PACF 분석</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="interp-box">
    분석 대상: {'차분 적용 시계열 (d=' + str(diff_d) + ')' if diff_d > 0 else '원본 시계열'}<br>
    ACF(자기상관함수): MA 차수 q 판단 — 신뢰구간을 벗어난 lag까지가 q 후보<br>
    PACF(편자기상관함수): AR 차수 p 판단 — 신뢰구간을 벗어난 lag까지가 p 후보<br>
    빨간 점선 = 95% 신뢰구간 (±{conf:.4f})</div>""", unsafe_allow_html=True)
    st.plotly_chart(plot_acf_pacf(acf_v, pacf_v, conf), use_container_width=True, key="acf_pacf_chart")

    st.markdown('<div class="section-title">AR(p) / MA(q) 후보 추정</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("추정 AR 차수 p", p_est, help="PACF 기반")
    c2.metric("추정 MA 차수 q", q_est, help="ACF 기반")
    for line in order_text.split("\n\n"):
        box_cls = "interp-ok" if "존재 가능" in line else "interp-box"
        st.markdown(f'<div class="{box_cls}">{line}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="interp-warn">ARIMA 파라미터 가이드:<br>
    ACF가 천천히 감소 → 차분 필요 (d 증가) &nbsp;|&nbsp;
    PACF 급절단 → AR 모델 적합 &nbsp;|&nbsp;
    ACF 급절단 → MA 모델 적합 &nbsp;|&nbsp;
    둘 다 서서히 감소 → ARMA 혼합 모델</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 5 — 변환
# ══════════════════════════════════════════════════════════════════════════
with tab_trans:
    st.markdown('<div class="section-title">Darts 변환 파이프라인</div>', unsafe_allow_html=True)
    if not transform_list:
        st.markdown("""<div class="interp-box">상단에서 변환을 선택하면 원본·변환 시계열을 비교합니다.<br>
        변환 목적: 분산 안정화(Log·Box-Cox), 값 범위 정규화(StandardScaler·MinMaxScaler)</div>""",
                    unsafe_allow_html=True)
        st.plotly_chart(plot_timeseries(df), use_container_width=True, key="ts_transform_raw")
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
                st.plotly_chart(plot_transform_comparison(ts, ts_transformed, tname),
                                use_container_width=True, key=f"transform_{tname}")
                st.markdown(f'<div class="interp-ok">{tname} 변환 적용 완료</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"{tname} 변환 실패: {e}")
    st.markdown('<div class="interp-box">변환은 시각화·비교 목적입니다. 모델 학습에는 원본 데이터를 사용합니다.</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 6 — 모델링·예측
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    for mname, err in errors.items():
        st.warning(f"{mname} 학습 실패 → 스킵: {err}")
    if not predictions:
        st.error("모든 모델이 실패했습니다. 파라미터를 조정해 주세요.")
        st.stop()

    st.markdown(f"""<div class="interp-box">
    Train: <b>{len(train)}</b>행 &nbsp;|&nbsp; Test: <b>{len(test)}</b>행 &nbsp;|&nbsp;
    Forecast Horizon: <b>{min(forecast_horizon, len(test))}</b> &nbsp;|&nbsp;
    추정 ARIMA 차수: p=<b>{p_est}</b>, d=<b>{diff_d}</b>, q=<b>{q_est}</b></div>""",
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">모델별 예측 결과</div>', unsafe_allow_html=True)
    pred_items = list(predictions.items())
    for i in range(0, len(pred_items), 2):
        cols = st.columns(2)
        for j, (mname, pred) in enumerate(pred_items[i:i+2]):
            with cols[j]:
                st.plotly_chart(plot_forecast_single(train, test, pred, mname),
                                use_container_width=True, key=f"forecast_single_{mname}")

    st.markdown('<div class="section-title">전체 모델 비교 오버레이</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_forecast_overlay(train, test, predictions),
                    use_container_width=True, key="forecast_overlay")

    st.markdown('<div class="section-title">미래 예측 (데이터 종료 이후)</div>', unsafe_allow_html=True)
    future_model_name = best_name if best_name else list(predictions.keys())[0]
    future_pred, fut_err = forecast_future(models[future_model_name], ts,
                                           forecast_horizon, future_model_name)
    if future_pred is not None:
        st.plotly_chart(plot_future_forecast(ts, future_pred, future_model_name),
                        use_container_width=True, key="future_forecast")
        st.markdown(f'<div class="interp-ok">최적 모델 <b>{future_model_name}</b>으로 '
                    f'<b>{forecast_horizon}</b>-step 미래 예측 완료. '
                    f'전체 데이터로 재학습 후 예측합니다 (Recursive forecasting).</div>',
                    unsafe_allow_html=True)
    else:
        st.warning(f"미래 예측 실패: {fut_err}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 7 — 성능 평가
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-title">모델 성능 비교 (MAE · RMSE · MAPE)</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="interp-box">
    MAE(평균절대오차): 예측 오차의 평균 크기<br>
    RMSE(제곱근평균오차): 큰 오차에 더 민감 — 최적 모델 선정 기준<br>
    MAPE(평균절대백분율오차): 상대적 오차율(%), 스케일 무관 비교 가능</div>""",
                unsafe_allow_html=True)

    def _highlight_best(row):
        if row["모델"] == best_name:
            return ["background-color:#dbeafe; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(metrics_df.style.apply(_highlight_best, axis=1)
                 .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}"}),
                 use_container_width=True, hide_index=True)

    if best_name:
        st.markdown(f'<div class="interp-ok">최적 모델 (RMSE 기준): <b>{best_name}</b> &nbsp;|&nbsp; '
                    f'RMSE = <b>{best_rmse:.4f}</b></div>', unsafe_allow_html=True)
    st.plotly_chart(plot_metrics_bars(metrics_df, best_name),
                    use_container_width=True, key="metrics_bars")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 8 — 잔차 분석
# ══════════════════════════════════════════════════════════════════════════
with tab_resid:
    st.markdown('<div class="section-title">잔차 분석 — 최적 모델 기준</div>', unsafe_allow_html=True)
    if best_name and best_name in predictions:
        st.markdown(f'<div class="interp-box">분석 모델: <b>{best_name}</b> (RMSE 최적)<br>'
                    '좋은 모델의 잔차는 <b>백색잡음(White Noise)</b>에 가까워야 합니다 — '
                    '평균 0, 자기상관 없음, 등분산</div>', unsafe_allow_html=True)

        residuals = compute_residuals(test, predictions[best_name])
        time_idx  = test[:len(residuals)].time_index
        st.plotly_chart(plot_residuals(residuals, best_name, time_idx),
                        use_container_width=True, key="residuals_chart")

        c1, c2 = st.columns(2)
        c1.metric("잔차 평균", f"{float(np.mean(residuals)):.4f}", help="0에 가까울수록 편향 없음")
        c2.metric("잔차 표준편차", f"{float(np.std(residuals)):.4f}")

        st.markdown('<div class="section-title">잔차 ACF — 자기상관 잔존 여부</div>',
                    unsafe_allow_html=True)
        r_acf, r_conf = residual_acf(residuals)
        st.plotly_chart(plot_residual_acf(r_acf, r_conf, best_name),
                        use_container_width=True, key="residual_acf_chart")
        st.markdown("""<div class="interp-box">잔차 ACF가 신뢰구간(빨간 점선) 안에 있으면
        자기상관 없음 → 백색잡음에 가까움<br>유의한 spike가 남아 있으면 모델 차수(p, q) 조정이 필요합니다.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Ljung-Box 검정</div>', unsafe_allow_html=True)
        lb_df, lb_text = ljung_box_test(residuals)
        st.dataframe(lb_df.rename(columns={"lb_stat": "LB 통계량", "lb_pvalue": "p-value"})
                     .style.format({"LB 통계량": "{:.4f}", "p-value": "{:.4f}"}),
                     use_container_width=True)
        last_p = lb_df["lb_pvalue"].iloc[-1]
        st.markdown(f'<div class="{"interp-ok" if last_p > 0.05 else "interp-warn"}">'
                    f'{lb_text}</div>', unsafe_allow_html=True)
    else:
        st.warning("예측 결과가 없습니다. 모델링·예측 탭을 먼저 확인하세요.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 9 — 백테스팅
# ══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-title">백테스팅 (Rolling Historical Forecasts)</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="interp-box">Rolling 방식으로 과거 구간마다 반복 학습·예측하여
    단순 Train/Test 분할보다 신뢰성 높은 성능 추정을 제공합니다.<br>
    start 비율(60~85%) 지점부터 슬라이딩 윈도우로 예측을 수행합니다.</div>""",
                unsafe_allow_html=True)

    if predictions:
        bt_model_name = st.selectbox("백테스팅 모델 선택",
                                     [n for n in MODEL_NAMES if n in predictions])
        if st.button("백테스팅 실행", type="primary"):
            with st.spinner(f"{bt_model_name} 백테스팅 수행 중..."):
                hist_fc, bt_err = run_backtest(ts, bt_model_name, forecast_horizon,
                                               p=p_est, d=diff_d, q=q_est,
                                               season_length=effective_season)
            if hist_fc is not None:
                st.plotly_chart(plot_backtest(ts, hist_fc, bt_model_name),
                                use_container_width=True, key="backtest_chart")
                bt_mae, bt_rmse = backtest_metrics(ts, hist_fc)
                if bt_mae is not None:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Backtest MAE",  f"{bt_mae:.4f}")
                    mc2.metric("Backtest RMSE", f"{bt_rmse:.4f}")
                    if best_rmse:
                        ratio = bt_rmse / best_rmse
                        cls = "interp-ok" if ratio < 1.5 else "interp-warn"
                        msg = (f"Backtest RMSE({bt_rmse:.4f}) ≈ Test RMSE({best_rmse:.4f}) — 모델이 과적합 없이 안정적입니다."
                               if ratio < 1.5 else
                               f"Backtest RMSE({bt_rmse:.4f}) &gt;&gt; Test RMSE({best_rmse:.4f}) — 과적합 또는 파라미터 조정이 필요합니다.")
                        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
                else:
                    st.warning("백테스팅 지표 계산 실패")
            else:
                st.error(f"백테스팅 실패: {bt_err}")
    else:
        st.error("예측 결과가 없습니다.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 10 — 결과 요약
# ══════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="section-title">결과 요약</div>', unsafe_allow_html=True)

    _best_model = best_name if best_name is not None else "N/A"
    _best_rmse = f"{best_rmse:.4f}" if best_rmse is not None else "N/A"

    if not metrics_df.empty:
        st.markdown('<div class="section-title">모델 성능 비교</div>', unsafe_allow_html=True)
        st.dataframe(
            metrics_df.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAPE (%)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="section-title">데이터 개요</div>', unsafe_allow_html=True)
    st.markdown(f"""
- 분석 기간: **{df.index[0].date()} ~ {df.index[-1].date()}**
- 총 데이터 수: **{info['length']}개**
- 추정 빈도: **{info['freq']}**
- Train/Test 분할: **{len(train)} / {len(test)}**
""")

    st.markdown('<div class="section-title">전처리 결과</div>', unsafe_allow_html=True)
    st.markdown(f"""
- 결측치 처리 개수: **{info['missing']}개**
- 이상치 처리 개수: **{info['outliers']}개**
""")

    st.markdown('<div class="section-title">정상성 및 차수 추정</div>', unsafe_allow_html=True)
    st.markdown(f"""
- ADF 검정 결과: **{adf_res_global['판정']}**
- KPSS 검정 결과: **{kpss_res_global['판정']}**
- 적용 차분 d: **{diff_d}**
- 추정 AR 차수 p: **{p_est}**
- 추정 MA 차수 q: **{q_est}**
- 계절 주기 m: **{effective_season}**
- 추천 ARIMA 형태: **ARIMA(p={p_est}, d={diff_d}, q={q_est})**
""")

    st.markdown('<div class="section-title">최적 모델 결과</div>', unsafe_allow_html=True)
    st.markdown(f"""
- 선택 모델: **{_best_model}**
- 평가 기준: **RMSE**
- Best RMSE: **{_best_rmse}**
""")

    st.markdown('<div class="section-title">종합 해석</div>', unsafe_allow_html=True)
    st.markdown(f"""
현재 시계열 데이터는 전처리, 정상성 검정, ACF/PACF 분석을 거쳐 모델링되었습니다.  
ACF/PACF 분석 결과를 기반으로 AR/MA 차수를 추정하였으며, 여러 모델 중 RMSE 기준 가장 우수한 모델은 **{_best_model}**입니다.

해당 모델은 현재 데이터 패턴을 가장 안정적으로 설명하는 것으로 판단됩니다.
""")

    st.markdown('<div class="section-title">결론</div>', unsafe_allow_html=True)
    st.markdown("""
본 시스템은 데이터 업로드, 전처리, 정상성 검정, 시계열 구조 분석, ACF/PACF 분석, 모델링 및 예측, 성능 평가, 잔차 분석, 백테스팅을 자동 수행합니다.  
이를 통해 사용자는 복잡한 시계열 분석 과정을 자동화하여 예측 결과와 모델 성능을 빠르게 확인할 수 있습니다.
""")