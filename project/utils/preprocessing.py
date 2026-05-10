"""
utils/preprocessing.py
──────────────────────
데이터 로드, 결측치 처리, 이상치 탐지/클리핑, Darts TimeSeries 변환
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller


# ── 날짜·값 컬럼 자동 탐지 ──────────────────────────────────────────────
def auto_detect_columns(df: pd.DataFrame):
    """
    CSV DataFrame에서 날짜 컬럼과 값(숫자) 컬럼을 자동 탐지한다.
    Returns: (date_col_name, value_col_name)  — 탐지 실패 시 None
    """
    date_col, value_col = None, None

    # 1) 컬럼명 힌트로 날짜 컬럼 탐색
    date_hints = ["date", "datetime", "time", "timestamp", "날짜", "ds",
                  "period", "month", "year", "day"]
    for col in df.columns:
        if any(h in col.lower() for h in date_hints):
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except Exception:
                continue

    # 2) 힌트로 못 찾으면 dtype 기반 탐색
    if date_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
            if df[col].dtype == "object":
                try:
                    pd.to_datetime(df[col])
                    date_col = col
                    break
                except Exception:
                    continue

    # 3) 인덱스가 날짜인 경우
    if date_col is None and pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.reset_index()
        date_col = df.columns[0]

    # 값 컬럼: 날짜 컬럼이 아닌 첫 번째 숫자형 컬럼
    for col in df.columns:
        if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
            value_col = col
            break

    return date_col, value_col


# ── 이상치 탐지 (IQR) ───────────────────────────────────────────────────
def detect_outliers_iqr(series: pd.Series, factor: float = 1.5):
    """IQR 기반 이상치 마스크와 경계값 반환"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


# ── 전처리 파이프라인 ────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, date_col: str, value_col: str):
    """
    1. 날짜 파싱 · 정렬
    2. 결측치 보간
    3. IQR 이상치 클리핑
    4. 빈도 추론 · Darts TimeSeries 변환

    Returns
    -------
    clean_df : pd.DataFrame  (index=datetime, 컬럼='value')
    ts       : darts.TimeSeries
    info     : dict  (결측치 수, 이상치 수, 빈도, 길이)
    """
    sub = df[[date_col, value_col]].copy()
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.sort_values(date_col).reset_index(drop=True)
    sub = sub.rename(columns={date_col: "date", value_col: "value"})

    # 결측치 보간
    missing_count = int(sub["value"].isna().sum())
    sub["value"] = sub["value"].interpolate(method="linear").ffill().bfill()

    # IQR 이상치 클리핑
    outlier_mask, lower, upper = detect_outliers_iqr(sub["value"])
    outlier_count = int(outlier_mask.sum())
    sub.loc[outlier_mask, "value"] = sub["value"].clip(lower, upper)

    # 빈도 추론
    sub = sub.set_index("date")
    freq = pd.infer_freq(sub.index)
    if freq is None:
        median_diff = sub.index.to_series().diff().dropna().median()
        freq_map = [
            (pd.Timedelta(days=1), "D"),
            (pd.Timedelta(days=7), "W"),
            (pd.Timedelta(days=31), "MS"),
            (pd.Timedelta(days=92), "QS"),
        ]
        freq = "YS"
        for threshold, f in freq_map:
            if median_diff <= threshold:
                freq = f
                break

    sub = sub.asfreq(freq)
    sub["value"] = sub["value"].interpolate().ffill().bfill()

    # Darts TimeSeries 변환
    ts = TimeSeries.from_dataframe(sub, value_cols="value", freq=freq)

    info = {
        "missing": missing_count,
        "outliers": outlier_count,
        "freq": freq,
        "length": len(sub),
    }
    return sub, ts, info


# ── AirPassengers 샘플 데이터 생성 ──────────────────────────────────────
def get_sample_data() -> pd.DataFrame:
    """AirPassengers(1949-01 ~ 1960-12) 내장 샘플 데이터"""
    passengers = [
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140,
        145,150,178,163,172,178,199,199,184,162,146,166,
        171,180,193,181,183,218,230,242,209,191,172,194,
        196,196,236,235,229,243,264,272,237,211,180,201,
        204,188,235,227,234,264,302,293,259,229,203,229,
        242,233,267,269,270,315,364,347,312,274,237,278,
        284,277,317,313,318,374,413,405,355,306,271,306,
        315,301,356,348,355,422,465,467,404,347,305,336,
        340,318,362,348,363,435,491,505,404,359,310,337,
        360,342,406,396,420,472,548,559,463,407,362,405,
        417,391,419,461,472,535,622,606,508,461,390,432,
    ]
    dates = pd.date_range("1949-01-01", periods=len(passengers), freq="MS")
    return pd.DataFrame({"date": dates, "passengers": passengers})
