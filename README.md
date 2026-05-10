# 시계열 분석 및 예측 자동화 대시보드

CSV 파일을 업로드하면 전처리부터 모델링, 평가까지 전체 파이프라인이 자동으로 실행되는 Streamlit 기반 시계열 분석 웹 애플리케이션입니다.

## 주요 기능

### 1. 데이터 전처리
- CSV 파일 업로드 또는 AirPassengers 내장 샘플 데이터 사용
- 날짜/값 컬럼 자동 탐지
- 선형 보간을 통한 결측치 처리
- IQR 기반 이상치 탐지 및 클리핑
- 시계열 빈도 자동 추론 (일/주/월/분기/연)

### 2. 시계열 구조 분석
- STL 분해 (Trend / Seasonal / Residual)
- ADF · KPSS 정상성 검정 및 한국어 해석 자동 출력
- 차분(d=1, 2) 적용 후 재검정

### 3. 상관 구조 분석
- ACF / PACF 시각화
- PACF · ACF 패턴 기반 AR(p), MA(q) 차수 자동 추정

### 4. 변환 파이프라인
- StandardScaler, MinMaxScaler, Log, Box-Cox 변환 선택 적용
- Darts Pipeline 연동, 변환 전/후 비교 차트 제공

### 5. 모델링 및 예측
총 4가지 모델을 동시에 학습하고 결과를 비교합니다.

| 모델 | 설명 |
|------|------|
| ARIMA | ACF/PACF 기반 추정 차수 적용 |
| AutoARIMA | 자동 차수 탐색 (stepwise) |
| ExponentialSmoothing | Holt-Winters 계절성 모델 |
| NaiveMovingAverage | 이동평균 기반 베이스라인 |

- Train/Test 분할 예측 및 전체 모델 오버레이 비교
- 최적 모델 기준 미래 n-step 예측

### 6. 성능 평가
- MAE / RMSE / MAPE 자동 계산
- RMSE 기준 최적 모델 자동 선정 및 하이라이트

### 7. 잔차 분석
- 잔차 시계열 플롯
- 잔차 ACF 시각화
- Ljung-Box 검정 및 자동 해석 (백색잡음 여부 판단)

### 8. 백테스팅
- Darts `historical_forecasts` 기반 Rolling Window 백테스팅
- Backtest MAE / RMSE 지표 제공

## 기술 스택

- **Frontend**: Streamlit
- **시계열 모델링**: Darts (ARIMA, AutoARIMA, ExponentialSmoothing, NaiveMovingAverage)
- **통계 분석**: statsmodels (STL, ADF, KPSS, ACF/PACF, Ljung-Box)
- **시각화**: Plotly (인터랙티브 차트)
- **전처리**: pandas, numpy, scikit-learn, scipy

## 설치 및 실행

```bash
pip install -r project/requirements.txt
streamlit run project/app.py
```

## 프로젝트 구조

```
project/
├── app.py              # 메인 Streamlit 앱
├── analysis.py         # STL 분해, 정상성 검정, ACF/PACF
├── preprocessing.py    # 데이터 로드, 결측치/이상치 처리
├── modeling.py         # 모델 학습, 예측, 백테스팅
├── evaluation.py       # 성능 지표, 잔차 분석
├── visualization.py    # Plotly 차트 함수
└── requirements.txt    # 의존성 패키지
```
