"""
utils/modeling.py
─────────────────
모델 학습, 예측, Darts historical_forecasts 기반 백테스팅
지원 모델: ARIMA, ExponentialSmoothing, AutoARIMA, NaiveMovingAverage
"""

import warnings
import numpy as np
from darts import TimeSeries
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    AutoARIMA,
    NaiveMovingAverage,
)
from darts.utils.utils import ModelMode, SeasonalityMode

warnings.filterwarnings("ignore")


# ── 모델 팩토리 ──────────────────────────────────────────────────────────
def _build_model(name: str, p: int, d: int, q: int,
                 season_length: int, train_len: int):
    """모델 이름으로 적절한 Darts 모델 객체를 생성"""
    sl = season_length if season_length > 1 else None

    if name == "ARIMA":
        return ARIMA(p=p, d=d, q=q)

    elif name == "ExponentialSmoothing":
        # 계절 주기가 데이터보다 크면 비계절 모드로 폴백
        if sl and sl * 2 > train_len:
            sl = None
        return ExponentialSmoothing(
            seasonal_periods=sl,
            trend=ModelMode.ADDITIVE,
            seasonal=(SeasonalityMode.ADDITIVE if sl
                      else SeasonalityMode.NONE),
            damped=True,
        )

    elif name == "AutoARIMA":
        m = season_length if season_length > 1 else 1
        if m * 2 > train_len:
            m = 1
        return AutoARIMA(
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            seasonal=(m > 1),
            season_length=m,
            stepwise=True,
        )

    elif name == "NaiveMovingAverage":
        window = min(max(season_length, 5), train_len - 1)
        return NaiveMovingAverage(input_chunk_length=window)

    else:
        raise ValueError(f"지원하지 않는 모델: {name}")


# ── 개별 모델 학습 + 예측 ────────────────────────────────────────────────
def fit_predict_single(name: str, train: TimeSeries, n: int,
                       p: int, d: int, q: int,
                       season_length: int) -> tuple:
    """
    단일 모델 학습 → n-step 예측.
    Returns: (prediction_TimeSeries, model_object, error_msg_or_None)
    """
    try:
        model = _build_model(name, p, d, q, season_length, len(train))
        model.fit(train)
        pred = model.predict(n)
        return pred, model, None
    except Exception as e:
        return None, None, str(e)


# ── 전체 모델 학습 + 예측 ────────────────────────────────────────────────
MODEL_NAMES = ["ARIMA", "ExponentialSmoothing", "AutoARIMA", "NaiveMovingAverage"]


def fit_predict_all(train: TimeSeries, test: TimeSeries,
                    forecast_horizon: int,
                    p: int, d: int, q: int,
                    season_length: int):
    """
    4개 모델 학습 + 예측.
    Returns
    -------
    predictions : dict[str, TimeSeries]
    models      : dict[str, model]
    errors      : dict[str, str]   — 실패한 모델의 에러 메시지
    """
    n = min(forecast_horizon, len(test)) if len(test) > 0 else forecast_horizon
    predictions, models, errors = {}, {}, {}

    for name in MODEL_NAMES:
        pred, model, err = fit_predict_single(
            name, train, n, p, d, q, season_length
        )
        if pred is not None:
            predictions[name] = pred
            models[name] = model
        else:
            errors[name] = err

    return predictions, models, errors


# ── 미래 예측 (test 이후) ────────────────────────────────────────────────
def forecast_future(model, ts_full: TimeSeries, horizon: int,
                    model_name: str) -> tuple:
    """전체 데이터로 재학습 후 horizon만큼 미래 예측"""
    try:
        # NaiveMovingAverage는 retrain 불필요 (non-trainable baseline)
        if model_name != "NaiveMovingAverage":
            model.fit(ts_full)
        else:
            model.fit(ts_full)
        future_pred = model.predict(horizon)
        return future_pred, None
    except Exception as e:
        return None, str(e)


# ── 백테스팅 ─────────────────────────────────────────────────────────────
def run_backtest(ts: TimeSeries, model_name: str,
                 forecast_horizon: int,
                 p: int, d: int, q: int,
                 season_length: int):
    """
    Darts historical_forecasts 기반 rolling 백테스팅.
    Returns
    -------
    hist_fc : TimeSeries or list[TimeSeries]
    error   : str or None
    """
    try:
        model = _build_model(model_name, p, d, q, season_length, len(ts))

        # start 비율: 데이터의 60 ~ 85% 지점부터 시작
        start_ratio = max(0.6, min(0.85, 1 - (forecast_horizon * 5) / len(ts)))

        retrain = model_name != "NaiveMovingAverage"

        hist_fc = model.historical_forecasts(
            ts,
            start=start_ratio,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=retrain,
            verbose=False,
        )
        return hist_fc, None
    except Exception as e:
        return None, str(e)
