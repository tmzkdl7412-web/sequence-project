# 📈 시계열 분석 및 예측 자동화 대시보드

CSV 파일을 업로드하면 전처리부터 모델링, 평가까지 전체 파이프라인이 자동으로 실행되는 Streamlit 기반 시계열 분석 웹 애플리케이션입니다.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Darts](https://img.shields.io/badge/Darts-time%20series-0B5394)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

```
CSV 업로드 → 전처리 → 구조 분석(STL·정상성·ACF/PACF) → 모델링(ARIMA·AutoARIMA·ES·NMA) → 성능 평가 → 잔차 분석 → 백테스팅
```

결과는 **데이터 개요 · 시계열 구조 · 정상성·차분 · ACF/PACF · 변환 · 모델링·예측 · 성능 평가 · 잔차 분석 · 백테스팅 · 결과 요약** 10개 탭으로 나뉘어 표시되며, 각 단계마다 수치에 대한 한국어 해석이 함께 출력됩니다.

## 주요 기능

### 1. 데이터 전처리
- CSV 파일 업로드 또는 내장 샘플 데이터(월별 소매 판매 지수, 120개월 가상 데이터) 사용
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
├── app.py                  # 메인 Streamlit 앱 (파라미터 패널, KPI 카드, 10개 탭)
├── requirements.txt        # 의존성 패키지
└── utils/
    ├── preprocessing.py    # 데이터 로드, 컬럼 자동 탐지, 결측치/이상치 처리, 샘플 데이터
    ├── analysis.py         # STL 분해, 정상성 검정, 차분, ACF/PACF, 차수 추정
    ├── modeling.py         # 모델 학습, 예측, 미래 예측, 백테스팅
    ├── evaluation.py       # 성능 지표, 잔차 분석, Ljung-Box 검정
    └── visualization.py    # Plotly 차트 함수
```

## 사용 방법

1. 날짜 컬럼과 값 컬럼이 있는 **단변량 시계열 CSV**를 업로드합니다. (없으면 샘플 데이터로 바로 체험 가능)
2. 상단 패널에서 예측 기간(H), 계절 주기(m), 차분 횟수(d), Train 비율, 변환 방식을 조절합니다.
3. 날짜·값 컬럼은 자동으로 탐지되며, 필요하면 직접 바꿀 수 있습니다.
4. 각 탭에서 분석 결과와 해석을 확인하고, 백테스팅 탭에서 모델을 골라 Rolling 검증을 실행합니다.
