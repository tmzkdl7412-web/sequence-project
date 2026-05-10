"""
utils/analysis.py
─────────────────
STL 분해, 정상성 검정(ADF·KPSS), ACF/PACF 계산, 차분, p/q 후보 추정
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


# ── STL 분해 ─────────────────────────────────────────────────────────────
def stl_decompose(series: pd.Series, period: int):
    """
    STL(Seasonal-Trend decomposition using LOESS) 수행.
    Returns: STL result 객체 (observed, trend, seasonal, resid)
    """
    period = max(2, min(period, len(series) // 2))
    stl = STL(series, period=period, robust=True)
    return stl.fit()


# ── 정상성 검정 ──────────────────────────────────────────────────────────
def adf_test(series: np.ndarray) -> dict:
    """ADF(Augmented Dickey-Fuller) 검정"""
    result = adfuller(series, autolag="AIC")
    return {
        "통계량": round(result[0], 4),
        "p-value": round(result[1], 6),
        "사용 래그": result[2],
        "판정": "✅ 정상 (Stationary)" if result[1] < 0.05
                else "⚠️ 비정상 (Non-stationary)",
    }


def kpss_test(series: np.ndarray) -> dict:
    """KPSS 검정"""
    try:
        stat, pval, lags, crit = kpss(series, regression="c", nlags="auto")
    except Exception:
        stat, pval, lags, crit = kpss(series, regression="c")
    return {
        "통계량": round(stat, 4),
        "p-value": round(pval, 4),
        "사용 래그": lags,
        "판정": "✅ 정상 (Stationary)" if pval > 0.05
                else "⚠️ 비정상 (Non-stationary)",
    }


def stationarity_summary(series: np.ndarray) -> tuple[dict, dict, str]:
    """ADF + KPSS 검정 결과와 한국어 해석 텍스트 반환"""
    adf = adf_test(series)
    kp = kpss_test(series)

    # 종합 해석
    if "정상" in adf["판정"] and "정상" in kp["판정"]:
        interpretation = (
            f"ADF p-value = {adf['p-value']} (&lt; 0.05) → 단위근 없음, "
            f"KPSS p-value = {kp['p-value']} (&gt; 0.05) → 추세 정상. "
            f"<b>두 검정 모두 정상성을 지지합니다.</b> 차분 없이 모델링 가능합니다."
        )
    elif "정상" in adf["판정"] and "비정상" in kp["판정"]:
        interpretation = (
            f"ADF p-value = {adf['p-value']} → 단위근 없음, "
            f"KPSS p-value = {kp['p-value']} → 추세 비정상. "
            f"<b>결과가 상충합니다.</b> 추세 차분(d=1)을 고려하세요."
        )
    elif "비정상" in adf["판정"] and "정상" in kp["판정"]:
        interpretation = (
            f"ADF p-value = {adf['p-value']} → 단위근 존재, "
            f"KPSS p-value = {kp['p-value']} → 추세 정상. "
            f"<b>결과가 상충합니다.</b> 차분(d=1)을 시도해 보세요."
        )
    else:
        interpretation = (
            f"ADF p-value = {adf['p-value']} (≥ 0.05), "
            f"KPSS p-value = {kp['p-value']} (&lt; 0.05). "
            f"<b>두 검정 모두 비정상으로 판단합니다.</b> 차분(d=1 또는 d=2)이 필요합니다."
        )
    return adf, kp, interpretation


# ── 차분 ─────────────────────────────────────────────────────────────────
def difference(series: pd.Series, d: int) -> pd.Series:
    """d회 차분 수행"""
    result = series.copy()
    for _ in range(d):
        result = result.diff().dropna()
    return result


# ── ACF / PACF 계산 ─────────────────────────────────────────────────────
def compute_acf_pacf(series: np.ndarray, nlags: int = 40):
    """ACF, PACF 값과 신뢰 구간 계산"""
    nlags = min(nlags, len(series) // 2 - 1, 60)
    nlags = max(nlags, 1)
    acf_vals = acf(series, nlags=nlags, fft=True)
    pacf_vals = pacf(series, nlags=nlags, method="ywm")
    conf = 1.96 / np.sqrt(len(series))
    return acf_vals, pacf_vals, conf


# ── AR(p), MA(q) 후보 추정 ──────────────────────────────────────────────
def suggest_arima_order(acf_vals: np.ndarray, pacf_vals: np.ndarray,
                        conf: float) -> tuple[int, int, str]:
    """
    ACF/PACF 패턴 기반으로 AR(p), MA(q) 후보 추정.
    Returns: (p, q, 근거_텍스트)
    """
    # PACF에서 유의한 최대 lag → p 후보
    sig_pacf = []
    for i in range(1, len(pacf_vals)):
        if abs(pacf_vals[i]) > conf:
            sig_pacf.append(i)
        else:
            break  # 첫 비유의 lag에서 절단

    # ACF에서 유의한 최대 lag → q 후보
    sig_acf = []
    for i in range(1, len(acf_vals)):
        if abs(acf_vals[i]) > conf:
            sig_acf.append(i)
        else:
            break

    p = max(sig_pacf) if sig_pacf else 0
    q = max(sig_acf) if sig_acf else 0
    p = min(p, 5)
    q = min(q, 5)

    # 해석 텍스트
    lines = []
    if p > 0:
        lines.append(f"PACF가 lag {p}까지 유의 → <b>AR({p})</b> 성분 존재 가능")
    else:
        lines.append("PACF가 lag 1부터 유의하지 않음 → AR 성분 약함")
    if q > 0:
        lines.append(f"ACF가 lag {q}까지 유의 → <b>MA({q})</b> 성분 존재 가능")
    else:
        lines.append("ACF가 lag 1부터 유의하지 않음 → MA 성분 약함")
    lines.append(f"→ 추천 ARIMA 차수: <b>ARIMA(p={p}, d=상단 파라미터, q={q})</b>")

    return p, q, "\n\n".join(lines)
