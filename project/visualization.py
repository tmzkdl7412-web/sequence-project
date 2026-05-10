"""
utils/visualization.py
──────────────────────
모든 Plotly 인터랙티브 차트 생성 함수
(matplotlib 사용 금지 — 명세서 제약 준수)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts import TimeSeries


# ── 색상 팔레트 ──────────────────────────────────────────────────────────
C = {
    "primary":  "#4361EE",
    "dark":     "#3A0CA3",
    "accent":   "#F72585",
    "green":    "#06D6A0",
    "yellow":   "#FFD166",
    "teal":     "#118AB2",
    "gray":     "#6C757D",
    "train":    "#AAAAAA",
    "test":     "#4361EE",
    "pred":     "#E63946",
}

MODEL_COLORS = {
    "ARIMA":                C["accent"],
    "ExponentialSmoothing": C["green"],
    "AutoARIMA":            C["yellow"],
    "NaiveMovingAverage":   C["teal"],
}

_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — 원본 시계열 · STL · 차분
# ══════════════════════════════════════════════════════════════════════════

def plot_timeseries(df: pd.DataFrame) -> go.Figure:
    """원본 시계열 라인 차트"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["value"], mode="lines",
        line=dict(color=C["primary"], width=1.5), name="Value",
    ))
    fig.update_layout(title="원본 시계열", xaxis_title="날짜",
                      yaxis_title="값", height=380, **_LAYOUT)
    return fig


def plot_stl(stl_result, index) -> go.Figure:
    """STL 분해 4-panel 차트"""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
    )
    traces = [
        (stl_result.observed, C["primary"]),
        (stl_result.trend,    "#E63946"),
        (stl_result.seasonal, "#457B9D"),
        (stl_result.resid,    "#2A9D8F"),
    ]
    for i, (vals, color) in enumerate(traces, 1):
        fig.add_trace(go.Scatter(
            x=index, y=vals, mode="lines",
            line=dict(color=color, width=1.2), showlegend=False,
        ), row=i, col=1)
    fig.update_layout(
        height=620,
        title="STL 분해 (Seasonal-Trend decomposition using LOESS)",
        **_LAYOUT,
    )
    return fig


def plot_differenced(original: pd.Series, diffed: pd.Series,
                     d: int) -> go.Figure:
    """차분 전/후 비교"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
        subplot_titles=["원본 시계열", f"{d}차 차분 결과"],
    )
    fig.add_trace(go.Scatter(
        x=original.index, y=original.values, mode="lines",
        line=dict(color=C["primary"], width=1.2), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=diffed.index, y=diffed.values, mode="lines",
        line=dict(color=C["accent"], width=1.2), showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=420, **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — ACF / PACF
# ══════════════════════════════════════════════════════════════════════════

def plot_acf_pacf(acf_vals, pacf_vals, conf) -> go.Figure:
    """ACF · PACF 막대그래프 (Plotly)"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ACF (자기상관함수)", "PACF (편자기상관함수)"],
    )
    # ACF
    for i, v in enumerate(acf_vals):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, v], mode="lines",
            line=dict(color=C["primary"], width=2.5), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[i], y=[v], mode="markers",
            marker=dict(color=C["primary"], size=5), showlegend=False,
        ), row=1, col=1)
    fig.add_hline(y=conf,  line_dash="dash", line_color="red",
                  line_width=1, row=1, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red",
                  line_width=1, row=1, col=1)

    # PACF
    for i, v in enumerate(pacf_vals):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, v], mode="lines",
            line=dict(color=C["dark"], width=2.5), showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[i], y=[v], mode="markers",
            marker=dict(color=C["dark"], size=5), showlegend=False,
        ), row=1, col=2)
    fig.add_hline(y=conf,  line_dash="dash", line_color="red",
                  line_width=1, row=1, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red",
                  line_width=1, row=1, col=2)

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_layout(height=360, **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 3 — 변환 비교
# ══════════════════════════════════════════════════════════════════════════

def plot_transform_comparison(orig_ts: TimeSeries,
                              trans_ts: TimeSeries,
                              label: str) -> go.Figure:
    """변환 전/후 비교 차트"""
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=["변환 전 (Original)", f"변환 후 ({label})"],
    )
    fig.add_trace(go.Scatter(
        x=orig_ts.time_index, y=orig_ts.values().flatten(),
        mode="lines", line=dict(color=C["primary"], width=1.2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trans_ts.time_index, y=trans_ts.values().flatten(),
        mode="lines", line=dict(color=C["accent"], width=1.2),
        showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=420, title="변환 전/후 비교", **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 4 — 모델별 예측
# ══════════════════════════════════════════════════════════════════════════

def plot_forecast_single(train: TimeSeries, test: TimeSeries,
                         pred: TimeSeries, name: str) -> go.Figure:
    """단일 모델 train / test(실제) / 예측 그래프"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.time_index, y=train.values().flatten(),
        mode="lines", name="Train",
        line=dict(color=C["train"], width=1),
    ))
    fig.add_trace(go.Scatter(
        x=test.time_index, y=test.values().flatten(),
        mode="lines", name="Test (Actual)",
        line=dict(color=C["test"], width=2),
    ))
    n = min(len(pred), len(test))
    fig.add_trace(go.Scatter(
        x=pred[:n].time_index, y=pred[:n].values().flatten(),
        mode="lines", name=f"예측 ({name})",
        line=dict(color=C["pred"], width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"{name} — 실제 vs 예측",
        xaxis_title="날짜", yaxis_title="값",
        height=350,
        legend=dict(orientation="h", y=-0.18),
        **_LAYOUT,
    )
    return fig


def plot_forecast_overlay(train: TimeSeries, test: TimeSeries,
                          predictions: dict) -> go.Figure:
    """전체 모델 예측 오버레이"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.time_index, y=train.values().flatten(),
        mode="lines", name="Train",
        line=dict(color=C["train"], width=1),
    ))
    fig.add_trace(go.Scatter(
        x=test.time_index, y=test.values().flatten(),
        mode="lines", name="Test (Actual)",
        line=dict(color="#333", width=2.5, dash="dot"),
    ))
    for name, pred in predictions.items():
        n = min(len(pred), len(test))
        fig.add_trace(go.Scatter(
            x=pred[:n].time_index, y=pred[:n].values().flatten(),
            mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name, "#999"), width=2),
        ))
    fig.update_layout(
        title="전체 모델 예측 비교",
        xaxis_title="날짜", yaxis_title="값", height=420,
        legend=dict(orientation="h", y=-0.15),
        **_LAYOUT,
    )
    return fig


def plot_future_forecast(ts_full: TimeSeries,
                         future_pred: TimeSeries,
                         model_name: str) -> go.Figure:
    """미래 예측 시각화 (데이터 끝 이후)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_full.time_index, y=ts_full.values().flatten(),
        mode="lines", name="실측",
        line=dict(color=C["primary"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=future_pred.time_index, y=future_pred.values().flatten(),
        mode="lines", name=f"미래 예측 ({model_name})",
        line=dict(color=C["accent"], width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title=f"미래 {len(future_pred)}-step 예측 ({model_name})",
        xaxis_title="날짜", yaxis_title="값", height=400,
        legend=dict(orientation="h", y=-0.15),
        **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 5 — 성능 평가 차트
# ══════════════════════════════════════════════════════════════════════════

def plot_metrics_bars(metrics_df: pd.DataFrame) -> go.Figure:
    """MAE / RMSE / MAPE 막대 비교"""
    cols = ["MAE", "RMSE", "MAPE (%)"]
    colors = [C["primary"], C["accent"], C["green"]]
    fig = go.Figure()
    for col, color in zip(cols, colors):
        fig.add_trace(go.Bar(
            x=metrics_df["모델"], y=metrics_df[col],
            name=col, marker_color=color,
        ))
    fig.update_layout(
        barmode="group", title="모델별 성능 지표 비교",
        height=400, **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 6 — 잔차 분석
# ══════════════════════════════════════════════════════════════════════════

def plot_residuals(residuals: np.ndarray, model_name: str) -> go.Figure:
    """잔차 시계열 플롯"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=residuals, mode="lines",
        line=dict(color="#2A9D8F", width=1), name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
    fig.update_layout(
        title=f"잔차 (Residuals) — {model_name}",
        xaxis_title="Index", yaxis_title="Residual",
        height=320, **_LAYOUT,
    )
    return fig


def plot_residual_acf(acf_vals: np.ndarray, conf: float,
                      model_name: str) -> go.Figure:
    """잔차 ACF 차트"""
    fig = go.Figure()
    for i, v in enumerate(acf_vals):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, v], mode="lines",
            line=dict(color=C["dark"], width=2.5), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[i], y=[v], mode="markers",
            marker=dict(color=C["dark"], size=5), showlegend=False,
        ))
    fig.add_hline(y=conf,  line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", line_width=1)
    fig.update_layout(
        title=f"잔차 ACF — {model_name}",
        xaxis_title="Lag", yaxis_title="ACF",
        height=300, **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  TAB 7 — 백테스팅
# ══════════════════════════════════════════════════════════════════════════

def plot_backtest(ts: TimeSeries, hist_fc,
                  model_name: str) -> go.Figure:
    """백테스팅 실제 vs 예측 그래프"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.time_index, y=ts.values().flatten(),
        mode="lines", name="Actual",
        line=dict(color=C["primary"], width=1.5),
    ))
    if isinstance(hist_fc, list):
        times, vals = [], []
        for fc in hist_fc:
            times.extend(fc.time_index.tolist())
            vals.extend(fc.values().flatten().tolist())
        fig.add_trace(go.Scatter(
            x=times, y=vals, mode="lines+markers",
            name="Backtest", marker=dict(size=3),
            line=dict(color=C["accent"], width=1),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=hist_fc.time_index, y=hist_fc.values().flatten(),
            mode="lines", name="Backtest",
            line=dict(color=C["accent"], width=1.5),
        ))
    fig.update_layout(
        title=f"백테스팅 — {model_name} (Rolling Historical Forecasts)",
        xaxis_title="날짜", yaxis_title="값",
        height=400, **_LAYOUT,
    )
    return fig
