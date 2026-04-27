# Can ML Beat the Market?
**ML로 효율적 시장 가설(EMH) 검증 + Markowitz 포트폴리오 실전 신호**
**부산대학교 전산물리 중간 프로젝트**

---

## 핵심 질문

> *"교수님이 말씀하신 '차트로 돈 못 번다(EMH)'는 실제 데이터로도 맞는가?  
> 그리고 맞다면, 그래도 돈을 더 버는 방법은 없는가?"*

---

## 실행 방법

```bash
cd /home/dullear/install/AI-ML/AI-ML-midterm

# GUI (4탭 분석 도구)
uv run main_app.py

# 오늘 포트폴리오 신호 (터미널)
uv run python today_signal.py
```

---

## 프로그램 구조 (4탭 GUI)

### ① Data & Features
- **yfinance**로 2010~2024 주가 데이터 자동 다운로드
- 분석 대상: S&P500 / NASDAQ / 필라델피아 반도체(SOX) / VIX
- 피처 생성: RSI, 볼린저밴드 %B, MACD, MA200 비율, 실현변동성 등 13개

### ② Regime Detection
- **K-Means(비지도학습)** 으로 시장 국면 3분류
  - `Bull` / `Sideways` / `Bear`
- 분류 근거: VIX 수준, 변동성 클러스터링
- 평가 지표: Silhouette Score (~0.48)

### ③ MLP Training (PyTorch)
- PyTorch MLP로 "내일 S&P500이 오를지 내릴지" 이진 분류
- 구조: Dense(128) → Dropout(0.3) → Dense(64) → Dense(1, sigmoid)
- Lookback 60일 시퀀스 → 방향성 예측 확률 P(상승) 출력
- 실시간 Loss/Accuracy 그래프 + ROC 커브

### ④ Backtest Results
- 전략별 2010~2024 백테스트 결과 비교

---

## 백테스트 결과 (2010~2024, 14년)

| 전략 | Sharpe | MDD | CAGR | 누적수익률 |
|------|--------|-----|------|----------|
| S&P500 Buy & Hold | 0.611 | -33.9% | 11.7% | 421% |
| NASDAQ Buy & Hold | 0.707 | -36.4% | 15.3% | 744% |
| SOX Buy & Hold | 0.675 | -46.5% | 19.1% | 1,274% |
| **Markowitz (월리밸)** | **0.637** | -39.8% | **14.5%** | **664%** |
| Kelly+MLP | 0.157 | -5.5% | 2.6% | 7.8% |

### Kelly 전략 변형 비교 (테스트 구간 2022~2024)

| 전략 | Sharpe | 특이사항 |
|------|--------|---------|
| Kelly Long-Only (fraction=1) | 0.075 | 기본 전략 |
| Kelly Long-Short | 0.075 | P(up) 최솟값 0.54 → 숏 신호 無 |
| Kelly Confidence Filter | 0.075 | 모든 f* > threshold → 필터 무효 |
| Kelly LS+CF | 0.075 | 동일 이유로 LongOnly와 동일 |

> **발견:** MLP가 테스트 구간에서 하락을 단 한 번도 예측하지 않음 (P(up) 최솟값 0.54).
> 2022~2024가 강세장이라 모델이 "항상 오른다"로 수렴한 결과.

---

## 핵심 발견 및 결론

### EMH 검증 결과

| 질문 | 결과 |
|------|------|
| ML(Kelly)이 Buy&Hold를 이기나? | **No** → 약형 EMH 지지 |
| Markowitz가 Buy&Hold를 이기나? | **Yes** → 자산배분 이론은 유효 |

### MLP vs Markowitz — 왜 다른가?

| | MLP (Kelly) | Markowitz |
|--|-------------|-----------|
| 하는 일 | 시장 **타이밍** 예측 | 자산 **배분** 최적화 |
| EMH와의 관계 | EMH가 막음 | EMH와 무관 |
| 사용 자산 | S&P500 단일 | S&P500+NASDAQ+SOX 3개 |
| 왜 실패/성공 | 공개정보로 방향 예측 불가 | 분산투자로 리스크 줄임 |

---

## today_signal.py — 오늘 투자 신호

Markowitz 기반으로 매일 실행 가능한 실전 신호 스크립트.

### 출력 내용
1. **자산별 지표** — Sharpe, CAGR, MDD, 변동성, 1M/3M 수익률
2. **최적 비중** — 1Y / 6M 윈도우 Sharpe 최대화 비중
3. **시장 상태** — VIX 수준, 포트폴리오 모멘텀
4. **5가지 조건 체크** → 🟢BUY / 🟡CAUTION / 🔴AVOID
5. **행동 강령** — 신규/기존 보유자별 구체적 액션 플랜

### 샤프지수 해석 가이드

| 구간 | 의미 |
|------|------|
| > 2.0 | 비정상 강세 (과열 의심) |
| 1.0 ~ 2.0 | 훌륭한 전략 |
| 0.5 ~ 1.0 | 양호 (장기 S&P500 평균 0.5~0.6) |
| 0 ~ 0.5 | 위험 대비 수익 불충분 |
| < 0 | 시장보다 못함 |

> 상승장이 끝나면 각 자산의 Sharpe가 떨어지고 → 포트폴리오 조건 미충족 → 자동으로 CAUTION/AVOID 전환

---

## 데이터 파이프라인

```
yfinance (2010~현재)
    │
    ▼
수익률 · 로그수익률 · Min-Max 정규화
    │
    ▼
피처 엔지니어링 (RSI, Bollinger, MACD, 변동성 13개)
    │
    ├──────────────────────────┐
    ▼                          ▼
K-Means Regime 감지        PyTorch MLP 학습
(Bull/Sideways/Bear)       (P(상승) 확률)
    │                          │
    └──────────┬───────────────┘
               ▼
    Markowitz 최적화 (월별 리밸런싱)
    + Kelly Criterion 4변형 (비교용)
               │
               ▼
    전략 성과 비교 + 오늘 신호 + 행동 강령
```

---

## 사용 기술

| 분류 | 라이브러리 |
|------|-----------|
| ML 모델 | `torch` (PyTorch MLP) |
| 군집화 | `scikit-learn` KMeans |
| 포트폴리오 최적화 | `scipy.optimize` (Sharpe 최대화) |
| 데이터 수집 | `yfinance` |
| GUI | `PySide6` |
| 시각화 | `matplotlib` |
| 패키지 관리 | `uv` |

---

## 파일 설명

| 파일 | 역할 |
|------|------|
| `main_app.py` | PySide6 GUI 진입점 (4탭) |
| `data.py` | yfinance 수집 + 수익률/정규화 전처리 |
| `features.py` | RSI, Bollinger, MACD, 변동성 피처 생성 |
| `regime.py` | K-Means Regime 분류기 |
| `model_torch.py` | PyTorch MLP 모델 학습 및 예측 |
| `strategy_kelly.py` | Kelly Criterion 4가지 변형 (LongOnly/LS/CF/LS+CF) |
| `strategy_markowitz.py` | Markowitz 샤프 최대화 포트폴리오 |
| `backtest.py` | Buy&Hold 계산 + 성과 지표 테이블 |
| `today_signal.py` | Markowitz 기반 오늘 투자 신호 + 행동 강령 |
| `DESCRIPTION.md` | 이 파일 — 프로젝트 전체 설명 |

---

## 한계

- 거래 수수료 미반영
- Markowitz는 후행적 — 급변장 대응 느림
- MLP 테스트 구간(2022~2024)이 강세장 → Bear 국면 0일 → Kelly 전략 과소평가 가능성
- SOX처럼 단일 자산이 압도적일 때 몰빵 권고 → 집중 리스크
- 코로나 폭락(2020) 포함으로 Regime Bear 기준점 상승 → 2022 하락장을 Sideways로 분류
