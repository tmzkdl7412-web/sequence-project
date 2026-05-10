"""
utils/evaluation.py
───────────────────
성능 지표 계산(MAE, RMSE, MAPE), 잔차 분석, Ljung-Box 검정
"""

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, mape
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── 모델별 성능 지표 ─────────────────────────────────────────────────────
def compute_metrics(test: TimeSeries, predictions: dict) -> pd.DataFrame:
    """
    모델별 MAE, RMSE, MAPE 계산 → DataFrame 반환.
    test와 prediction 길이가 다를 경우 짧은 쪽에 맞춤.
    """
    rows = []
    for name, pred in predictions.items():
        n = min(len(pred), len(test))
        actual = test[:n]
        forecast = pred[:n]
        try:
            m = mae(actual, forecast)
            r = rmse(actual, forecast)
            try:
                mp = mape(actual, forecast)
            except Exception:
                mp = np.nan
            rows.append({
                "모델": name,
                "MAE": round(float(m), 4),
                "RMSE": round(float(r), 4),
                "MAPE (%)": round(float(mp), 2),
            })
        except Exception:
            rows.append({
                "모델": name, "MAE": np.nan,
                "RMSE": np.nan, "MAPE (%)": np.nan,
            })

    df = pd.DataFrame(rows)
    return df


def find_best_model(metrics_df: pd.DataFrame,
                    criterion: str = "RMSE") -> str:
    """RMSE(기본) 기준 최적 모델명 반환"""
    valid = metrics_df.dropna(subset=[criterion])
    if valid.empty:
        return None
    idx = valid[criterion].idxmin()
    return valid.loc[idx, "모델"]


# ── 잔차 계산 ────────────────────────────────────────────────────────────
def compute_residuals(test: TimeSeries, pred: TimeSeries) -> np.ndarray:
    """test − pred 잔차 계산"""
    n = min(len(test), len(pred))
    return test[:n].values().flatten() - pred[:n].values().flatten()


# ── 잔차 ACF ─────────────────────────────────────────────────────────────
def residual_acf(residuals: np.ndarray, nlags: int = 20):
    """잔차의 ACF 값과 신뢰 구간 반환"""
    nlags = min(nlags, len(residuals) // 2 - 1)
    nlags = max(nlags, 1)
    acf_vals = acf(residuals, nlags=nlags, fft=True)
    conf = 1.96 / np.sqrt(len(residuals))
    return acf_vals, conf


# ── Ljung-Box 검정 ──────────────────────────────────────────────────────
def ljung_box_test(residuals: np.ndarray,
                   lags: int = 10) -> tuple[pd.DataFrame, str]:
    """
    Ljung-Box 검정 수행.
    Returns: (결과 DataFrame, 한국어 해석 텍스트)
    """
    lags = min(lags, len(residuals) // 2 - 1)
    lags = max(lags, 1)
    lb = acorr_ljungbox(residuals, lags=lags, return_df=True)

    # 마지막 lag의 p-value로 판정
    last_p = lb["lb_pvalue"].iloc[-1]
    if last_p > 0.05:
        text = (
            f"Ljung-Box 검정 p-value = {last_p:.4f} (> 0.05)\n\n"
            f"→ 귀무가설(잔차에 자기상관 없음)을 **기각하지 못합니다**.\n\n"
            f"잔차가 백색잡음에 가까우므로, 모델이 시계열의 구조를 "
            f"적절히 포착했다고 판단할 수 있습니다."
        )
    else:
        text = (
            f"Ljung-Box 검정 p-value = {last_p:.4f} (≤ 0.05)\n\n"
            f"→ 귀무가설을 **기각**합니다. 잔차에 유의미한 자기상관이 "
            f"남아 있습니다.\n\n모델 차수 조정(p, q 변경) 또는 "
            f"계절 차분을 고려하세요."
        )
    return lb, text


# ── 백테스팅 지표 ────────────────────────────────────────────────────────
def backtest_metrics(ts: TimeSeries, hist_fc) -> tuple:
    """
    백테스팅 결과에서 MAE, RMSE 계산.
    hist_fc가 list이면 concat 후 계산.
    """
    try:
        if isinstance(hist_fc, list):
            combined = hist_fc[0]
            for fc in hist_fc[1:]:
                try:
                    combined = combined.append(fc)
                except Exception:
                    pass
            bt_mae = float(mae(ts, combined))
            bt_rmse = float(rmse(ts, combined))
        else:
            bt_mae = float(mae(ts, hist_fc))
            bt_rmse = float(rmse(ts, hist_fc))
        return round(bt_mae, 4), round(bt_rmse, 4)
    except Exception:
        return None, None
