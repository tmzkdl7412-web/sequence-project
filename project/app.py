"""
app.py — 자동 시계열 분석 및 예측 대시보드
═══════════════════════════════════════════
Darts · statsmodels · Plotly · Streamlit
파일 업로드 시 전체 파이프라인이 자동으로 재실행됩니다.
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing import Pipeline
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  페이지 설정 & 스타일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="📈 시계열 분석 시스템", page_icon="📈", layout="wide")

st.markdown("""
<style>
.header-gradient {
    text-align:center; padding:0.8rem 0 0.2rem;
    background:linear-gradient(135deg,#4361EE,#3A0CA3);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:2.1rem; font-weight:800;
}
.sub-text {text-align:center;color:#666;font-size:0.95rem;margin-bottom:1.2rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-gradient">📈 자동 시계열 분석 및 예측 시스템</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-text">Darts · statsmodels · Plotly 기반 | '
            'CSV 업로드 → 전처리 → 분석 → 모델링 → 평가 자동 수행</div>',
            unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  사이드바
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.header("⚙️ 설정 패널")

    # ── 파일 업로드 ──
    uploaded = st.file_uploader("📁 CSV 파일 업로드", type=["csv"])
    use_sample = st.checkbox("📦 샘플 데이터 사용 (AirPassengers)", value=(uploaded is None))

    st.divider()

    # ── 컬럼 선택 (데이터 로드 후 동적 생성) ──
    st.subheader("📌 파라미터")
    forecast_horizon = st.slider("Forecast Horizon", 1, 365, 30,
                                 help="예측할 미래 시점 수")
    season_length = st.slider("계절 주기 (m)", 1, 365, 12,
                              help="STL 분해 및 계절성 모델링 주기")
    diff_d = st.slider("차분 횟수 (d)", 0, 2, 0,
                       help="ARIMA 차분 파라미터 / 정상성 확보용")
    train_ratio = st.slider("Train 비율", 0.5, 0.95, 0.8, step=0.05)

    st.divider()
    st.subheader("🔄 변환 옵션")
    transform_list = st.multiselect(
        "적용할 변환 선택",
        ["StandardScaler", "MinMaxScaler", "Log", "Box-Cox"],
        default=[],
        help="Darts Pipeline으로 적용",
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ CSV 파싱 오류: {e}")
        st.info("UTF-8 인코딩의 CSV 파일인지 확인해 주세요.")
        st.stop()
elif use_sample:
    raw_df = get_sample_data()
    st.sidebar.success("✅ AirPassengers 샘플 로드됨")
else:
    st.info("👈 사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 선택하세요.")
    st.stop()

# ── 컬럼 탐지 & 선택 ──
detected_date, detected_value = auto_detect_columns(raw_df)

with st.sidebar:
    st.divider()
    st.subheader("📊 컬럼 선택")
    all_cols = list(raw_df.columns)

    date_col = st.selectbox(
        "날짜 컬럼",
        all_cols,
        index=all_cols.index(detected_date) if detected_date in all_cols else 0,
    )
    value_candidates = [c for c in all_cols if c != date_col]
    value_col = st.selectbox(
        "값 컬럼",
        value_candidates,
        index=(value_candidates.index(detected_value)
               if detected_value in value_candidates else 0),
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1단계 — 전처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    df, ts, info = preprocess(raw_df, date_col, value_col)
except Exception as e:
    st.error(f"❌ 전처리 실패: {e}")
    st.stop()

# 데이터 길이 경고
if info["length"] < 50:
    st.warning(
        f"⚠️ 데이터가 {info['length']}행으로 매우 짧습니다. "
        f"일부 모델(AutoARIMA, 백테스팅)이 제한될 수 있습니다."
    )

# 계절 주기 자동 조정
effective_season = season_length
if season_length * 2 > info["length"]:
    effective_season = max(2, info["length"] // 4)
    st.warning(
        f"⚠️ 계절 주기({season_length})가 데이터 길이({info['length']})에 비해 "
        f"너무 큽니다. 자동으로 {effective_season}로 조정합니다."
    )

# 요약 메트릭
c1, c2, c3, c4 = st.columns(4)
c1.metric("📏 데이터 길이", f"{info['length']}행")
c2.metric("🔤 추정 빈도", info["freq"])
c3.metric("🕳️ 결측치 처리", f"{info['missing']}개")
c4.metric("📍 이상치 처리", f"{info['outliers']}개")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터 분할 (전역)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
split_idx = int(len(ts) * train_ratio)
split_idx = max(split_idx, 10)                     # 최소 train 10
split_idx = min(split_idx, len(ts) - 2)            # 최소 test 2
train, test = ts.split_before(split_idx)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 구성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 시계열 구조", "📈 ACF/PACF", "🔄 변환",
    "🤖 모델링·예측", "📋 성능 평가", "🔍 잔차 분석", "🔁 백테스팅",
])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — 시계열 구조 분석
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    # 원본 시계열
    st.subheader("📊 원본 시계열")
    st.plotly_chart(plot_timeseries(df), use_container_width=True)

    # STL 분해
    st.subheader("🔬 STL 분해")
    try:
        stl_result = stl_decompose(df["value"], effective_season)
        st.plotly_chart(plot_stl(stl_result, df.index), use_container_width=True)
        st.caption(f"계절 주기(period) = {effective_season}")
    except Exception as e:
        st.error(f"STL 분해 실패: {e}")

    # 정상성 검정
    st.subheader("📐 정상성 검정")
    test_series = df["value"].values
    if diff_d > 0:
        diffed = difference(df["value"], diff_d)
        test_series = diffed.values
        st.plotly_chart(
            plot_differenced(df["value"], diffed, diff_d),
            use_container_width=True,
        )

    adf_res, kpss_res, interpretation = stationarity_summary(test_series)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ADF 검정**")
        st.dataframe(
            pd.DataFrame([adf_res]).T.rename(columns={0: "값"}),
            use_container_width=True,
        )
    with col_b:
        st.markdown("**KPSS 검정**")
        st.dataframe(
            pd.DataFrame([kpss_res]).T.rename(columns={0: "값"}),
            use_container_width=True,
        )
    st.info(interpretation)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — 상관 구조 분석
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 ACF / PACF 분석")
    analysis_vals = (difference(df["value"], diff_d).values
                     if diff_d > 0 else df["value"].values)

    acf_v, pacf_v, conf = compute_acf_pacf(analysis_vals)
    st.plotly_chart(plot_acf_pacf(acf_v, pacf_v, conf), use_container_width=True)

    p_est, q_est, order_text = suggest_arima_order(acf_v, pacf_v, conf)
    st.markdown("### 🔍 AR(p) / MA(q) 후보 추정")
    st.markdown(order_text)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 — 변환 파이프라인
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔄 Darts 변환 파이프라인")

    if not transform_list:
        st.info("사이드바에서 적용할 변환을 선택하세요. 선택하지 않으면 원본 데이터 그대로 사용됩니다.")
        ts_transformed = ts
    else:
        ts_transformed = ts
        for tname in transform_list:
            try:
                if tname == "StandardScaler":
                    scaler = Scaler(StandardScaler())
                    ts_transformed = scaler.fit_transform(ts_transformed)
                    st.plotly_chart(
                        plot_transform_comparison(ts, ts_transformed, "StandardScaler"),
                        use_container_width=True,
                    )
                elif tname == "MinMaxScaler":
                    scaler = Scaler(MinMaxScaler())
                    ts_transformed = scaler.fit_transform(ts_transformed)
                    st.plotly_chart(
                        plot_transform_comparison(ts, ts_transformed, "MinMaxScaler"),
                        use_container_width=True,
                    )
                elif tname == "Log":
                    vals = ts_transformed.values()
                    shift = 0
                    if vals.min() <= 0:
                        shift = abs(vals.min()) + 1
                    log_vals = np.log(vals + shift)
                    ts_transformed = ts_transformed.with_values(log_vals)
                    st.plotly_chart(
                        plot_transform_comparison(ts, ts_transformed, f"Log (shift={shift})"),
                        use_container_width=True,
                    )
                elif tname == "Box-Cox":
                    from scipy.stats import boxcox
                    vals = ts_transformed.values().flatten()
                    shift = 0
                    if vals.min() <= 0:
                        shift = abs(vals.min()) + 1
                        vals = vals + shift
                    bc_vals, lmbda = boxcox(vals)
                    ts_transformed = ts_transformed.with_values(bc_vals.reshape(-1, 1))
                    st.plotly_chart(
                        plot_transform_comparison(
                            ts, ts_transformed, f"Box-Cox (λ={lmbda:.3f})"
                        ),
                        use_container_width=True,
                    )
                    st.caption(f"Box-Cox λ = {lmbda:.4f}")
                st.success(f"✅ {tname} 적용 완료")
            except Exception as e:
                st.error(f"{tname} 변환 실패: {e}")

    st.caption("ℹ️ 변환은 시각화 목적입니다. 모델 학습에는 원본 데이터를 사용합니다.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 4 — 모델링 & 예측
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 모델 학습 및 예측")
    st.markdown(
        f"Train: **{len(train)}**행 | Test: **{len(test)}**행 | "
        f"Forecast Horizon: **{min(forecast_horizon, len(test))}**"
    )

    with st.spinner("4개 모델 학습 중… ⏳"):
        predictions, models, errors = fit_predict_all(
            train, test, forecast_horizon,
            p=p_est, d=diff_d, q=q_est,
            season_length=effective_season,
        )

    # 모델별 에러 표시
    for mname, err in errors.items():
        st.warning(f"⚠️ {mname} 학습 실패 → 스킵됨: {err}")

    if not predictions:
        st.error("모든 모델이 실패했습니다. 파라미터를 조정해 주세요.")
        st.stop()

    # 개별 모델 그래프
    for mname, pred in predictions.items():
        st.plotly_chart(
            plot_forecast_single(train, test, pred, mname),
            use_container_width=True,
        )

    # 전체 오버레이
    st.subheader("📊 전체 모델 비교")
    st.plotly_chart(
        plot_forecast_overlay(train, test, predictions),
        use_container_width=True,
    )

    # 미래 예측 (최적 모델 기준)
    st.subheader("🔮 미래 예측")
    # 임시로 첫 번째 성공 모델 사용 (tab5에서 최적 모델 확정)
    first_model_name = list(predictions.keys())[0]
    first_model = models[first_model_name]
    future_pred, fut_err = forecast_future(
        first_model, ts, forecast_horizon, first_model_name
    )
    if future_pred is not None:
        st.plotly_chart(
            plot_future_forecast(ts, future_pred, first_model_name),
            use_container_width=True,
        )
        st.caption(
            "💡 Recursive forecasting: 1-step 예측 → 결과를 입력에 추가 → "
            "다음 step 예측 (ARIMA·ES 기본 동작)"
        )
    else:
        st.warning(f"미래 예측 실패: {fut_err}")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 5 — 성능 평가
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 모델 성능 비교")

    metrics_df = compute_metrics(test, predictions)

    # 최적 모델 하이라이트
    best_name = find_best_model(metrics_df, "RMSE")

    def _highlight_best(row):
        if row["모델"] == best_name:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style.apply(_highlight_best, axis=1)
                  .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    if best_name:
        st.success(f"🏆 최적 모델 (RMSE 기준): **{best_name}**")

    st.plotly_chart(plot_metrics_bars(metrics_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  TAB 6 — 잔차 분석 (최적 모델)
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔍 잔차 분석")

    if best_name and best_name in predictions:
        st.markdown(f"분석 대상 모델: **{best_name}** (RMSE 최적)")
        best_pred = predictions[best_name]
        residuals = compute_residuals(test, best_pred)

        # 잔차 시계열 플롯
        st.plotly_chart(plot_residuals(residuals, best_name), use_container_width=True)

        # 잔차 ACF
        r_acf, r_conf = residual_acf(residuals)
        st.plotly_chart(plot_residual_acf(r_acf, r_conf, best_name), use_container_width=True)

        # Ljung-Box 검정
        st.markdown("### 📊 Ljung-Box 검정")
        lb_df, lb_text = ljung_box_test(residuals)
        st.dataframe(lb_df, use_container_width=True)
        st.info(lb_text)
    else:
        st.warning("성능 평가 탭에서 최적 모델을 먼저 확인하세요.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 7 — 백테스팅
# ══════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🔁 백테스팅 (Historical Forecasts)")

    bt_model_name = st.selectbox(
        "백테스팅 모델 선택",
        [n for n in MODEL_NAMES if n in predictions],
        index=0,
    )

    if st.button("▶️ 백테스팅 실행", type="primary"):
        with st.spinner(f"{bt_model_name} 백테스팅 수행 중… ⏳"):
            hist_fc, bt_err = run_backtest(
                ts, bt_model_name, forecast_horizon,
                p=p_est, d=diff_d, q=q_est,
                season_length=effective_season,
            )

        if hist_fc is not None:
            st.plotly_chart(
                plot_backtest(ts, hist_fc, bt_model_name),
                use_container_width=True,
            )

            bt_mae, bt_rmse = backtest_metrics(ts, hist_fc)
            if bt_mae is not None:
                mc1, mc2 = st.columns(2)
                mc1.metric("Backtest MAE", f"{bt_mae:.4f}")
                mc2.metric("Backtest RMSE", f"{bt_rmse:.4f}")
            else:
                st.warning("백테스팅 지표 계산 실패")

            st.caption(
                "💡 `historical_forecasts`는 rolling window 방식으로 "
                "과거 데이터만 사용하여 반복 학습·예측하므로, "
                "단순 Train/Test 분할보다 신뢰성 높은 성능 추정을 제공합니다."
            )
        else:
            st.error(f"백테스팅 실패: {bt_err}")

# ── Footer ──
st.divider()
st.caption(
    "📈 시계열 분석 시스템 | "
    "Darts · statsmodels · Plotly · Streamlit | "
    "© 2025 Time Series Analysis Pipeline"
)
