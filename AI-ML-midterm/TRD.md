# TRD: Can ML Beat the Market?
### 기술 요구사항 문서 — ML 기반 EMH 검증 시스템

**과목**: 전산물리 (Computational Physics) — 부산대학교 물리학과  
**Week 8 중간 프로젝트** | 기술 스택: TensorFlow · NumPy · SciPy · Matplotlib · uv

---

## 1. 시스템 아키텍처

```
[Data Collection]          yfinance API
        │
        ▼
[Preprocessing]            수익률, 로그수익률, Min-Max 정규화  (← Week2: 정규화)
        │
        ▼
[Feature Engineering]      기술지표 + 변동성 클러스터링 피처
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[Regime Detector]                             [Direction Predictor]
 K-Means (비지도, Week2)                       TensorFlow MLP (Week3-5)
 Bull / Bear / Sideways                        P(상승) 확률 출력
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
                  [Strategy Selector]
           Bull/Sideways → Markowitz (전략B)
           Bear          → Kelly+MLP (전략A)
                           │
                           ▼
                      [Backtester]
                   Walk-Forward Validation
                           │
                           ▼
                [Evaluation & Visualization]
                Sharpe, MDD, CAGR, 누적수익률
```

---

## 2. 데이터 사양

### 2.1 수집 대상

| 티커 | 설명 | 용도 |
|------|------|------|
| ^GSPC | S&P 500 | 전략B 구성, 벤치마크 |
| ^IXIC | NASDAQ Composite | 전략B 구성 |
| ^SOX | Philadelphia Semiconductor | 전략B 구성 (교수님 추천) |
| ^VIX | CBOE Volatility Index | 변동성 클러스터링 피처 |
| ^KS11 | KOSPI (선택) | 전략A 확장 시 사용 |

### 2.2 전처리 파이프라인

```python
# Week 2: 수익률 + 정규화 수업 내용 직접 적용
r_t      = (P_t - P_{t-1}) / P_{t-1}          # 일반 수익률
lr_t     = np.log(P_t / P_{t-1})              # 로그수익률
X_norm   = (X - X.min()) / (X.max() - X.min()) # Min-Max 정규화

# 결측치: forward-fill → drop
# 백테스트 분할: Train 2015–2021 | Test 2022–2024
```

---

## 3. 피처 엔지니어링

### 3.1 기술지표 피처

| 피처 | 계산 방법 |
|------|-----------|
| RSI(14) | 상대강도지수 |
| Bollinger %B | `(Close - Lower) / (Upper - Lower)` |
| MFI(14) | Money Flow Index |
| MACD Signal | MACD - Signal Line |
| MA200 비율 | `Close / MA(200)` |

### 3.2 변동성 클러스터링 피처 (Regime Detection 핵심)

금융 시장의 변동성은 **클러스터링(Volatility Clustering)** 특성을 가진다.  
ARCH 효과: 고변동성 구간 이후 고변동성이 지속된다.  
EMH는 "수익률 예측 불가"를 주장하지만, **변동성의 자기상관은 예측 가능**하다.  
→ Regime 감지에 핵심적으로 활용.

| 피처 | 계산 | 의미 |
|------|------|------|
| Realized Vol 5d | `std(r_t, 5) × √252` | 단기 실현변동성 |
| Realized Vol 21d | `std(r_t, 21) × √252` | 월간 실현변동성 |
| Vol Ratio | `RV_5d / RV_21d` | 변동성 가속/감속 |
| Vol Z-score | `(RV_21d - μ) / σ` | 변동성 수준 표준화 |
| VIX Level | VIX 원값 | 내재변동성 |
| VIX 5d Change | `VIX_t - VIX_{t-5}` | 변동성 방향 |
| Squared Returns r² | `r_t²` | ARCH 효과 직접 측정 |
| Lagged r² | `r_{t-1}², r_{t-2}²` | 변동성 자기상관 |

---

## 4. ML 모델 사양

### 4.1 Regime Detector — K-Means (비지도학습)

**Week 2 K-Means 클러스터링 직접 응용**

```python
from sklearn.cluster import KMeans

# 변동성 피처로 시장 상태 3분류
features = ['RV_21d', 'Vol_Zscore', 'VIX_Level', 'MA200_ratio']
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
regime_labels = kmeans.fit_predict(X_vol_features)
# → 클러스터 특성 분석 후 Bull/Bear/Sideways 레이블 할당
```

### 4.2 방향성 예측 모델 — TensorFlow MLP

**Week 3–5 핵심 개념 종합 적용**

```
입력: 기술지표 + 변동성 피처 (lookback=60일)
출력: P(상승) 확률 → Kelly f* 계산에 사용
```

**모델 구조 (Week 3: MLP + Week 5: Dropout)**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[n_features]),
    tf.keras.layers.Dropout(0.3),          # Week5: 과적합 방지
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),          # Week5: Regularization
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # P(상승)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Week4: Adam
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**학습 설정**

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,          # 검증셋으로 과적합 모니터링
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),  # Week5 응용
    ]
)
```

**목표 성능**: Test Accuracy > 55%  
(55% 이상이면 Kelly 공식에서 양수 기대값 → 장기적 수익 가능)

---

## 5. 전략 구현

### 5.1 전략 A — TensorFlow MLP + Kelly Criterion

```python
# MLP 예측 확률
p = model.predict(X_test)[:, 0]   # P(상승)
q = 1 - p

# 분수 Kelly (1/4 Kelly: 과도한 베팅 방지)
kelly_f = (p * 1.0 - q) / 1.0     # b=1 (1:1 배팅)
position_size = kelly_f * 0.25     # 1/4 Kelly
position_size = np.clip(position_size, 0, 0.5)  # 최대 50%
```

### 5.2 전략 B — Markowitz 포트폴리오 (샤프 최대화)

```python
from scipy.optimize import minimize

def neg_sharpe(weights, returns, cov_matrix, rf=0.02/252):
    port_return = np.dot(weights, returns.mean()) * 252
    port_vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    return -(port_return - rf) / port_vol

result = minimize(
    neg_sharpe, x0=[1/3, 1/3, 1/3],
    args=(returns, cov_matrix),
    method='SLSQP',
    bounds=[(0, 1)] * 3,
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
# 자산: S&P500 / NASDAQ / 필라델피아 반도체
```

### 5.3 Regime 기반 전략 전환

```python
# Bull/Sideways → 안정적 분산 전략
# Bear → ML 방향성 예측으로 현금 대피 또는 숏
if regime in ['Bull', 'Sideways']:
    use_strategy = 'Markowitz'
else:  # Bear
    use_strategy = 'Kelly_MLP'
```

---

## 6. 백테스터 설계

### 6.1 Walk-Forward Validation

```
Train Window : 252일 (1년)
Test Window  : 63일  (분기)
Step         : 63일씩 롤링 → Look-ahead bias 완전 차단
```

미래 데이터 참조 금지: 각 시점의 모델은 해당 시점 이전 데이터로만 학습.

### 6.2 성과 지표

```python
def sharpe_ratio(returns, rf=0.02/252):
    excess = returns - rf
    return excess.mean() / excess.std() * np.sqrt(252)

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    return ((cum_returns - peak) / peak).min()

def cagr(cum_returns, n_years):
    return (cum_returns.iloc[-1]) ** (1 / n_years) - 1
```

---

## 7. 기술 스택

| 분류 | 라이브러리 | 수업 연결 |
|------|-----------|----------|
| ML 프레임워크 | `tensorflow` | Week 1–5 표준 스택 |
| 수치 계산 | `numpy` | 전체 |
| 데이터 처리 | `pandas` | — |
| 최적화 | `scipy.optimize` | Markowitz |
| 군집화 | `scikit-learn` (KMeans) | Week 2 |
| 데이터 수집 | `yfinance` | — |
| 시각화 | `matplotlib`, `seaborn` | — |
| 패키지 관리 | `uv` | 수업 표준 |

**Python**: 3.12+  
**기존 코드 참고**:
- `trading_backup_20260324/kor/backtest.py` — 백테스트 로직
- `trading_backup_20260324/nas/screening.py` — 종목 스크리닝
- `AIandMLcourse/week2/01_linear_regression_spring.py` — TF 학습 패턴 참고

---

## 8. 파일 구조

```
AI-ML-midterm/
├── PRD.md
├── TRD.md
├── main_app.py          # PySide6 GUI 진입점 (4탭)
├── data.py              # yfinance 수집 + 전처리
├── features.py          # 피처 엔지니어링 (변동성 클러스터링)
├── regime.py            # K-Means Regime Detector
├── model_torch.py       # PyTorch MLP 방향성 예측 모델
├── strategy_kelly.py    # 전략 A: Kelly + MLP (4변형)
├── strategy_markowitz.py # 전략 B: Markowitz 최적화
├── backtest.py          # Backtester + 성과 지표
├── today_signal.py      # 실전 투자 신호 (매일 실행)
└── signal_history.csv   # 신호 히스토리 (자동 생성)
```

---

## 9. today_signal.py 아키텍처

```
yfinance (최신 데이터)
    │
    ▼
signal_history.csv 로드
    │
    ▼
이전 신호 수익률 검증 (전일 비중 × 오늘 가격변화)
    │
    ▼
Markowitz 최적 비중 계산 (1Y / 6M 윈도우)
    │
    ▼
5조건 체크 → 🟢BUY / 🟡CAUTION / 🔴AVOID
    │
    ├── 포트폴리오 원화 배분 (PORTFOLIO_KRW 기준)
    ├── 손절가 계산 (-10% 기준)
    └── 행동 강령 (신규/기존 보유자 분기)
    │
    ▼
signal_history.csv 저장  +  Discord Webhook 알림
```

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PORTFOLIO_KRW` | `10000000` | 투자 원금 (원화) |
| `DISCORD_WEBHOOK_URL` | (없음) | Discord Webhook URL |

---

## 10. GitHub Actions 자동화

```yaml
# .github/workflows/daily_signal.yml
on:
  schedule:
    - cron: '0 0 * * 1-5'   # 매일 09:00 KST (UTC+9), 평일
  workflow_dispatch:          # 수동 실행 허용
```

### 실행 흐름

```
GitHub Actions (ubuntu-latest)
    │
    ├── checkout 코드
    ├── uv 설치 (캐시 활용)
    ├── uv run python today_signal.py
    │       └── DISCORD_WEBHOOK_URL → Discord 알림 전송
    └── signal_history.csv → git commit & push
```

### Discord Webhook 설정 방법
1. Discord 서버 → 채널 설정 → Integrations → Webhooks → New Webhook
2. Webhook URL 복사
3. GitHub 저장소 → Settings → Secrets → Actions → `DISCORD_WEBHOOK_URL` 등록

### GitHub Actions 비용
- Public 저장소: **무료 (무제한)**
- Private 저장소: 월 2,000분 무료 — 1회 실행 약 2분 × 22일 = 44분/월 (여유)

---

## 9. 검증 계획

| 단계 | 방법 |
|------|------|
| 데이터 검증 | 결측치 0%, 날짜 정렬, 수익률 분포 확인 |
| 과적합 검증 | Train vs Test accuracy 차이 ≤ 5% (Week 5 기준) |
| 피처 검증 | 변동성 자기상관 ACF plot (클러스터링 존재 확인) |
| Regime 검증 | 3 클러스터 분리도, Silhouette Score |
| 백테스트 검증 | Look-ahead bias check, 거래 로그 검토 |
| 최종 검증 | 4전략 비교표 + 시각화 정상 출력 |

---

## 10. EMH 결론 판단 기준

| 결과 | 해석 |
|------|------|
| ML 전략 < Buy&Hold | **EMH 지지** — 교수님 주장 맞음, 공개 정보로 초과수익 불가 |
| ML 전략 > Buy&Hold | **약형 EMH 반례** — 단, 과적합 가능성 명시적으로 논의 필요 |
| Regime 전환 시만 개선 | 변동성 예측 가능성 → 부분적 시장 비효율성 존재 |

> **결론은 열린 질문이다.** 어느 결과가 나오든 데이터가 말하는 대로 보고하는 것이 목표. ML이 시장을 이기든 지든, Week 1–7의 핵심 개념(지도학습, MLP, Regularization, K-Means)을 실제 금융 데이터에 적용한 과정 자체가 이 프로젝트의 학문적 가치다.

---

## 11. YouTube 영상 구성안 (10분)

| 시간 | 내용 |
|------|------|
| 0:00–1:00 | 프로젝트 소개 + EMH 개념 (교수님 특강 연결) |
| 1:00–2:30 | 데이터 + 피처 설명 (변동성 클러스터링 시각화) |
| 2:30–4:00 | TF MLP 모델 구조 + 학습 과정 (Week 3–5 연결) |
| 4:00–5:30 | K-Means Regime 감지 결과 (Week 2 연결) |
| 5:30–8:00 | 백테스트 결과 — 4전략 비교 (누적수익률 그래프) |
| 8:00–9:30 | EMH 검증 결론 — 데이터가 말하는 것 |
| 9:30–10:00 | 한계점 + Transformer 기반 향후 개선 (Week 6) |
