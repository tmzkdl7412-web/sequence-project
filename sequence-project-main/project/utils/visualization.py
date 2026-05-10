"""
utils/visualization.py
──────────────────────
모든 Plotly 인터랙티브 차트 생성 함수
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
    "yellow":   "#F4A261",
    "teal":     "#118AB2",
    "gray":     "#6C757D",
    "train":    "#B0B8C9",
    "test":     "#4361EE",
    "pred":     "#E63946",
    "bg":       "#F8F9FC",
}

MODEL_COLORS = {
    "ARIMA":                "#F72585",
    "ExponentialSmoothing": "#06D6A0",
    "AutoARIMA":            "#F4A261",
    "NaiveMovingAverage":   "#118AB2",
}

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=30, t=55, b=45),
    font=dict(family="Inter, sans-serif", size=12),
    paper_bgcolor="white",
    plot_bgcolor="white",
)


# ══════════════════════════════════════════════════════════════════════════
#  원본 시계열
# ══════════════════════════════════════════════════════════════════════════

def plot_timeseries(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["value"], mode="lines",
        line=dict(color=C["primary"], width=2),
        name="Value",
        fill="tozeroy",
        fillcolor="rgba(67,97,238,0.06)",
    ))
    fig.update_layout(
        title=dict(text="원본 시계열", font=dict(size=15)),
        xaxis_title="날짜", yaxis_title="값",
        height=380,
        **_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  STL 분해
# ══════════════════════════════════════════════════════════════════════════

def plot_stl(stl_result, index) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
    )
    traces = [
        (stl_result.observed, C["primary"],  "Observed"),
        (stl_result.trend,    "#E63946",     "Trend"),
        (stl_result.seasonal, "#457B9D",     "Seasonal"),
        (stl_result.resid,    "#2A9D8F",     "Residual"),
    ]
    for i, (vals, color, name) in enumerate(traces, 1):
        fig.add_trace(go.Scatter(
            x=index, y=vals, mode="lines",
            line=dict(color=color, width=1.5),
            name=name, showlegend=False,
        ), row=i, col=1)
    fig.update_layout(
        height=640,
        title=dict(text="STL 분해 (Seasonal-Trend decomposition using LOESS)",
                   font=dict(size=15)),
        **_LAYOUT,
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", row=i, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  차분
# ══════════════════════════════════════════════════════════════════════════

def plot_differenced(original: pd.Series, diffed: pd.Series, d: int) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.14,
        subplot_titles=["원본 시계열", f"{d}차 차분 결과"],
    )
    fig.add_trace(go.Scatter(
        x=original.index, y=original.values, mode="lines",
        line=dict(color=C["primary"], width=1.5), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=diffed.index, y=diffed.values, mode="lines",
        line=dict(color=C["accent"], width=1.5), showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=440, **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  ACF / PACF
# ══════════════════════════════════════════════════════════════════════════

def plot_acf_pacf(acf_vals, pacf_vals, conf) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["ACF (자기상관함수)", "PACF (편자기상관함수)"],
        horizontal_spacing=0.10,
    )

    def _add_stems(vals, color, row, col):
        for i, v in enumerate(vals):
            clr = C["accent"] if abs(v) > conf else color
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, v], mode="lines",
                line=dict(color=clr, width=2.5), showlegend=False,
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[i], y=[v], mode="markers",
                marker=dict(color=clr, size=5), showlegend=False,
            ), row=row, col=col)

    _add_stems(acf_vals,  C["primary"], 1, 1)
    _add_stems(pacf_vals, C["dark"],    1, 2)

    for col in [1, 2]:
        fig.add_hline(y=conf,  line_dash="dash", line_color="red",
                      line_width=1, row=1, col=col)
        fig.add_hline(y=-conf, line_dash="dash", line_color="red",
                      line_width=1, row=1, col=col)
        fig.add_hline(y=0, line_color="#CCC", line_width=0.8, row=1, col=col)

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_layout(height=380, **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  변환 비교
# ══════════════════════════════════════════════════════════════════════════

def plot_transform_comparison(orig_ts: TimeSeries, trans_ts: TimeSeries,
                               label: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.14,
        subplot_titles=["변환 전 (Original)", f"변환 후 ({label})"],
    )
    fig.add_trace(go.Scatter(
        x=orig_ts.time_index, y=orig_ts.values().flatten(),
        mode="lines", line=dict(color=C["primary"], width=1.5),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trans_ts.time_index, y=trans_ts.values().flatten(),
        mode="lines", line=dict(color=C["accent"], width=1.5),
        showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=440, title="변환 전/후 비교", **_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  모델 예측
# ══════════════════════════════════════════════════════════════════════════

def plot_forecast_single(train: TimeSeries, test: TimeSeries,
                         pred: TimeSeries, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.time_index, y=train.values().flatten(),
        mode="lines", name="Train",
        line=dict(color=C["train"], width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=test.time_index, y=test.values().flatten(),
        mode="lines", name="Test (실제)",
        line=dict(color=C["test"], width=2),
    ))
    n = min(len(pred), len(test))
    color = MODEL_COLORS.get(name, C["accent"])
    fig.add_trace(go.Scatter(
        x=pred[:n].time_index, y=pred[:n].values().flatten(),
        mode="lines", name=f"예측 ({name})",
        line=dict(color=color, width=2, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text=f"{name} — 실제 vs 예측", font=dict(size=14)),
        xaxis_title="날짜", yaxis_title="값",
        height=360,
        legend=dict(orientation="h", y=-0.20, x=0),
        **_LAYOUT,
    )
    return fig


def plot_forecast_overlay(train: TimeSeries, test: TimeSeries,
                          predictions: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.time_index, y=train.values().flatten(),
        mode="lines", name="Train",
        line=dict(color=C["train"], width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=test.time_index, y=test.values().flatten(),
        mode="lines", name="Test (실제)",
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
        title=dict(text="전체 모델 예측 비교", font=dict(size=14)),
        xaxis_title="날짜", yaxis_title="값", height=430,
        legend=dict(orientation="h", y=-0.16, x=0),
        **_LAYOUT,
    )
    return fig


def plot_future_forecast(ts_full: TimeSeries, future_pred: TimeSeries,
                         model_name: str) -> go.Figure:
    fig = go.Figure()
    # 최근 데이터만 표시 (전체 컨텍스트)
    x_all = ts_full.time_index
    y_all = ts_full.values().flatten()

    fig.add_trace(go.Scatter(
        x=x_all, y=y_all, mode="lines", name="실측",
        line=dict(color=C["primary"], width=1.8),
    ))
    # 연결선
    fig.add_trace(go.Scatter(
        x=[x_all[-1], future_pred.time_index[0]],
        y=[y_all[-1], future_pred.values().flatten()[0]],
        mode="lines", showlegend=False,
        line=dict(color=C["accent"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=future_pred.time_index, y=future_pred.values().flatten(),
        mode="lines+markers", name=f"미래 예측 ({model_name})",
        line=dict(color=C["accent"], width=2.5, dash="dash"),
        marker=dict(size=5, color=C["accent"]),
    ))
    fig.update_layout(
        title=dict(text=f"미래 {len(future_pred)}-step 예측 — {model_name}",
                   font=dict(size=14)),
        xaxis_title="날짜", yaxis_title="값", height=420,
        legend=dict(orientation="h", y=-0.16, x=0),
        **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  성능 평가
# ══════════════════════════════════════════════════════════════════════════

def plot_metrics_bars(metrics_df: pd.DataFrame, best_name: str = None) -> go.Figure:
    cols = ["MAE", "RMSE", "MAPE (%)"]
    colors = [C["primary"], C["accent"], C["green"]]
    fig = go.Figure()
    for col, color in zip(cols, colors):
        fig.add_trace(go.Bar(
            x=metrics_df["모델"], y=metrics_df[col],
            name=col, marker_color=color,
            text=metrics_df[col].round(3),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="모델별 성능 지표 비교", font=dict(size=14)),
        height=420, **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  잔차 분석
# ══════════════════════════════════════════════════════════════════════════

def plot_residuals(residuals: np.ndarray, model_name: str,
                   time_index=None) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["잔차 시계열", "잔차 히스토그램"],
        horizontal_spacing=0.10,
    )
    x = time_index if time_index is not None else list(range(len(residuals)))
    fig.add_trace(go.Scatter(
        x=x, y=residuals, mode="lines",
        line=dict(color="#2A9D8F", width=1.5), name="Residual",
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  line_width=1, row=1, col=1)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=20,
        marker_color="#2A9D8F", opacity=0.7,
        showlegend=False,
    ), row=1, col=2)
    fig.update_layout(
        height=340,
        title=dict(text=f"잔차 분석 — {model_name}", font=dict(size=14)),
        **_LAYOUT,
    )
    return fig


def plot_residual_acf(acf_vals: np.ndarray, conf: float,
                      model_name: str) -> go.Figure:
    fig = go.Figure()
    for i, v in enumerate(acf_vals):
        clr = C["accent"] if abs(v) > conf and i > 0 else C["dark"]
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, v], mode="lines",
            line=dict(color=clr, width=2.5), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[i], y=[v], mode="markers",
            marker=dict(color=clr, size=5), showlegend=False,
        ))
    fig.add_hline(y=conf,  line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=0, line_color="#CCC", line_width=0.8)
    fig.update_layout(
        title=dict(text=f"잔차 ACF — {model_name}", font=dict(size=14)),
        xaxis_title="Lag", yaxis_title="ACF",
        height=320, **_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  백테스팅
# ══════════════════════════════════════════════════════════════════════════

def plot_backtest(ts: TimeSeries, hist_fc, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.time_index, y=ts.values().flatten(),
        mode="lines", name="실측",
        line=dict(color=C["primary"], width=1.8),
    ))
    if isinstance(hist_fc, list):
        times, vals = [], []
        for fc in hist_fc:
            times.extend(fc.time_index.tolist())
            vals.extend(fc.values().flatten().tolist())
        fig.add_trace(go.Scatter(
            x=times, y=vals, mode="lines+markers",
            name="Backtest 예측",
            marker=dict(size=3, color=C["accent"]),
            line=dict(color=C["accent"], width=1.2),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=hist_fc.time_index, y=hist_fc.values().flatten(),
            mode="lines", name="Backtest 예측",
            line=dict(color=C["accent"], width=1.8),
        ))
    fig.update_layout(
        title=dict(
            text=f"백테스팅 — {model_name} (Rolling Historical Forecasts)",
            font=dict(size=14),
        ),
        xaxis_title="날짜", yaxis_title="값",
        height=420,
        legend=dict(orientation="h", y=-0.16, x=0),
        **_LAYOUT,
    )
    return fig
