# 시계열 분석·예측 대시보드

단변량 시계열 CSV를 올리면 전처리, 구조 분석, 모델 학습, 성능 평가, 잔차 분석, 백테스팅까지 한 화면에서 확인할 수 있는 Streamlit 앱입니다.

결과는 데이터 개요, 시계열 구조, 정상성·차분, ACF/PACF, 변환, 모델링·예측, 성능 평가, 잔차 분석, 백테스팅, 결과 요약 10개 탭으로 나뉘고, 검정 결과마다 어떻게 읽으면 되는지 한국어 설명을 같이 보여줍니다.

## 기능

**전처리**
- CSV 업로드 또는 내장 샘플 데이터(월별 소매 판매 지수, 120개월 가상 데이터) 사용
- 날짜·값 컬럼 자동 탐지, 시계열 빈도(일/주/월/분기/연) 추론
- 결측치는 선형 보간, 이상치는 IQR 기준으로 클리핑

**구조 분석**
- STL 분해(추세, 계절, 잔차)
- ADF·KPSS 정상성 검정, 1·2차 차분 후 재검정
- ACF·PACF를 보고 AR(p), MA(q) 차수 후보 추정
- StandardScaler, MinMaxScaler, Log, Box-Cox 변환 전후 비교

**모델링과 평가**

| 모델 | 설명 |
| --- | --- |
| ARIMA | ACF·PACF로 추정한 차수 사용 |
| AutoARIMA | stepwise로 차수 자동 탐색 |
| ExponentialSmoothing | Holt-Winters 계절 모델 |
| NaiveMovingAverage | 이동평균 기준 모델 |

- 네 모델을 같은 Train/Test 분할로 학습해 MAE, RMSE, MAPE 비교, RMSE가 가장 낮은 모델로 미래 구간 예측
- 잔차 플롯, 잔차 ACF, Ljung-Box 검정으로 잔차가 백색잡음인지 확인
- Darts `historical_forecasts`로 Rolling 백테스팅

## 사용 방법

1. 날짜 컬럼과 값 컬럼이 있는 CSV를 올립니다. 파일이 없으면 샘플 데이터로 바로 볼 수 있습니다.
2. 상단 패널에서 예측 기간(H), 계절 주기(m), 차분 횟수(d), Train 비율, 변환 방식을 정합니다.
3. 날짜·값 컬럼은 자동으로 잡히고, 필요하면 직접 바꿀 수 있습니다.
4. 탭을 넘기면서 결과를 보고, 백테스팅 탭에서 모델을 골라 Rolling 검증을 돌립니다.

## 실행 방법

```bash
pip install -r project/requirements.txt
streamlit run project/app.py
```

## 기술 스택

Streamlit, Darts, statsmodels, Plotly, pandas, NumPy, scikit-learn, SciPy

## 구조

```
project/
├── app.py                  # Streamlit 화면 (설정 패널, 요약 카드, 10개 탭)
├── requirements.txt
└── utils/
    ├── preprocessing.py    # 데이터 로드, 컬럼 탐지, 결측치·이상치 처리, 샘플 데이터
    ├── analysis.py         # STL 분해, 정상성 검정, 차분, ACF/PACF, 차수 추정
    ├── modeling.py         # 모델 학습, 예측, 백테스팅
    ├── evaluation.py       # 성능 지표, 잔차 분석, Ljung-Box 검정
    └── visualization.py    # Plotly 차트
```
