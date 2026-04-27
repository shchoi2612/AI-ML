# TRD: Can ML Beat the Market?
### 기술 요구사항 문서 — ML 기반 EMH 검증 시스템

**과목**: 전산물리 (Computational Physics) — 부산대학교 물리학과  
**Week 8 중간 프로젝트** | 기술 스택: PyTorch · NumPy · SciPy · PySide6 · uv

---

## 1. 시스템 아키텍처

```
[Data Collection]          yfinance API
        │
        ▼
[Preprocessing]            수익률, 로그수익률, Min-Max 정규화  (← Week2: 정규화)
        │
        ▼
[Feature Engineering]      기술지표 + 변동성 클러스터링 피처 (13개)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[Regime Detector]                             [Direction Predictor]
 K-Means (비지도, Week2)                       PyTorch MLP (Week3-5)
 Bull / Sideways / Bear                        P(상승) 확률 출력
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

### 2.2 전처리 파이프라인

```python
# Week 2: 수익률 + 정규화 수업 내용 직접 적용
r_t      = (P_t - P_{t-1}) / P_{t-1}           # 일반 수익률
lr_t     = np.log(P_t / P_{t-1})               # 로그수익률
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
# → 클러스터 특성 분석 후 Bull/Sideways/Bear 레이블 할당
```

**실측 결과 (2015–2024)**

| 국면 | 일수 | 비율 |
|------|------|------|
| Bull | 2,708 | 77.3% |
| Sideways | 738 | 21.1% |
| Bear | 55 | 1.6% |

Silhouette Score: **0.50**

### 4.2 방향성 예측 모델 — PyTorch MLP

**Week 3–5 핵심 개념 종합 적용**

```
입력: 기술지표 + 변동성 피처 13개 (lookback=60일 시퀀스)
출력: P(상승) 확률 → Kelly f* 계산에 사용
```

**모델 구조 (Week 3: MLP + Week 5: Dropout)**

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),          # Week5: 과적합 방지
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()              # P(상승)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

**학습 설정**

```python
# EarlyStopping + 80/20 Train/Validation 분할
# Epoch: 100 (기본값, GUI에서 조정 가능)
# Batch size: 32
```

**실측 성능 (2022–2024 테스트 구간)**

| 지표 | 값 | 해석 |
|------|-----|------|
| ROC AUC | ~0.47 | 랜덤(0.5)보다 낮음 |
| Test Accuracy | ~58% | 강세장 편향 — 실질적 예측력 없음 |
| P(up) 최솟값 | 0.54 | 하락 예측 0회 |

> **발견**: 강세장 구간(2022–2024)에서 MLP가 "항상 오른다"로 수렴. 약형 EMH 지지.

---

## 5. 전략 구현

### 5.1 전략 A — PyTorch MLP + Kelly Criterion

```python
# MLP 예측 확률
p = predict_proba(model, scaler, X_test)  # P(상승)
q = 1 - p

# Kelly 공식 (b=1, Long-Only)
kelly_f = p - q                            # f* = p - q
position_size = np.clip(kelly_f, 0, 1)   # 음수 포지션 제거
```

**4가지 변형 비교**

| 변형 | 설명 | 결과 |
|------|------|------|
| Long-Only | f* > 0 시 매수 | Sharpe 0.075 |
| Long-Short | f* < 0 시 숏 | Sharpe 0.075 (숏 신호 無) |
| Confidence Filter | 확률 임계값 필터 | Sharpe 0.075 (전부 통과) |
| LS + CF | 두 조건 결합 | Sharpe 0.075 |

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
# 월별 리밸런싱, Look-ahead bias 방지
```

### 5.3 Regime 기반 전략 전환

```python
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

### 6.2 성과 지표

```python
def sharpe_ratio(returns, rf=0.02/252):
    excess = returns - rf
    return excess.mean() / excess.std() * np.sqrt(252)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def cagr(returns):
    cum = (1 + returns).cumprod()
    n_years = len(returns) / 252
    return cum.iloc[-1] ** (1 / n_years) - 1
```

### 6.3 실측 백테스트 결과 (2015–2024, 약 10년)

| 전략 | Sharpe | MDD | CAGR | 누적수익률 |
|------|--------|-----|------|----------|
| S&P500 Buy & Hold | 0.611 | -33.9% | 11.7% | 421% |
| NASDAQ Buy & Hold | 0.707 | -36.4% | 15.3% | 744% |
| SOX Buy & Hold | 0.675 | -46.5% | 19.1% | 1,274% |
| **Markowitz (월리밸)** | **0.637** | -39.8% | **14.5%** | **664%** |
| Kelly+MLP | 0.157 | -5.5% | 2.6% | 7.8% |

---

## 7. 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| ML 프레임워크 | `torch` (PyTorch) |
| 수치 계산 | `numpy` |
| 데이터 처리 | `pandas` |
| 최적화 | `scipy.optimize` |
| 군집화 | `scikit-learn` (KMeans) |
| 데이터 수집 | `yfinance` |
| GUI | `PySide6` |
| 시각화 | `matplotlib` |
| 패키지 관리 | `uv` |

**Python**: 3.12+

---

## 8. 파일 구조

```
AI-ML-midterm/
├── PRD.md
├── TRD.md
├── DESCRIPTION.md           # 프로젝트 전체 설명
├── main_app.py              # PySide6 GUI 진입점 (4탭)
├── data.py                  # yfinance 수집 + 전처리
├── features.py              # 피처 엔지니어링 (13개)
├── regime.py                # K-Means Regime Detector
├── model_torch.py           # PyTorch MLP 방향성 예측 모델
├── strategy_kelly.py        # 전략 A: Kelly + MLP (4변형)
├── strategy_markowitz.py    # 전략 B: Markowitz 최적화
├── backtest.py              # Backtester + 성과 지표
└── today_signal.py          # 실전 투자 신호 (매일 실행)
```

---

## 9. today_signal.py 아키텍처

```
yfinance (최신 데이터)
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
    └── 행동강령 (신규/기존 보유자 분기)
    │
    ▼
signal_history.csv 저장  +  Discord Webhook 알림
```

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PORTFOLIO_KRW` | `10000000` | 투자 원금 (원화) |
| `DISCORD_WEBHOOK_URL` | (없음) | Discord Webhook URL (미설정 시 알림 건너뜀) |

### 자동화 (cron)

```bash
# 평일 22:30 KST (미장 개장 30분 전) 자동 실행
30 13 * * 1-5 DISCORD_WEBHOOK_URL="..." /path/to/uv run python today_signal.py
```

---

## 10. 검증 계획

| 단계 | 방법 |
|------|------|
| 데이터 검증 | 결측치 0%, 날짜 정렬, 수익률 분포 확인 |
| 과적합 검증 | Loss/Accuracy Curve 시각화 (Train vs Val 괴리 확인) |
| 피처 검증 | 변동성 자기상관 ACF plot (클러스터링 존재 확인) |
| Regime 검증 | Silhouette Score, VIX vs RV 산점도 |
| 백테스트 검증 | Look-ahead bias check, Walk-Forward 롤링 구조 |
| 최종 검증 | 4전략 비교표 + 누적수익률 그래프 정상 출력 |

---

## 11. EMH 결론 판단 기준

| 결과 | 해석 |
|------|------|
| ML 전략 < Buy&Hold | **EMH 지지** — 공개 정보로 초과수익 불가 |
| ML 전략 > Buy&Hold | **약형 EMH 반례** — 과적합 가능성 명시적 논의 필요 |
| Markowitz > Buy&Hold (S&P500) | 타이밍 예측이 아닌 자산배분으로 초과수익 가능 |

**실측 결론**: Kelly+MLP는 Buy&Hold 미달 → **약형 EMH 지지**. Markowitz는 S&P500 대비 CAGR +2.8%p 초과 → **자산배분 이론 유효**.

---

## 12. YouTube 영상 구성안 (10분)

| 시간 | 내용 |
|------|------|
| 0:00–1:00 | 프로젝트 소개 + EMH 개념 (교수님 특강 연결) |
| 1:00–2:30 | 데이터 + 피처 설명 (변동성 클러스터링 시각화) |
| 2:30–4:00 | PyTorch MLP 모델 구조 + 학습 과정 (Week 3–5 연결) |
| 4:00–5:30 | K-Means Regime 감지 결과 (Week 2 연결) |
| 5:30–8:00 | 백테스트 결과 — 4전략 비교 (누적수익률 그래프) |
| 8:00–9:30 | EMH 검증 결론 — 데이터가 말하는 것 |
| 9:30–10:00 | 한계점 + Transformer 기반 향후 개선 (Week 6) |
