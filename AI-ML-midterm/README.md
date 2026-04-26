# Can ML Beat the Market?
### 전산물리 중간 프로젝트 — ML로 효율적 시장 가설(EMH) 검증
**부산대학교 물리학과 | Week 8**

---

## 실행 방법

```bash
# 의존성 설치 (uv 사용)
uv sync

# GUI 앱 실행
uv run main_app.py
```

## 사용 순서

```
① 데이터 & 피처  →  ② Regime 감지  →  ③ MLP 학습  →  ④ 백테스트 결과
```

각 탭의 버튼을 순서대로 클릭하면 자동으로 다음 탭으로 이동합니다.

## 프로젝트 구조

```
AI-ML-midterm/
├── PRD.md              # 제품 요구사항
├── TRD.md              # 기술 요구사항
├── pyproject.toml      # uv 패키지 관리
├── main_app.py         # PySide6 GUI 진입점 ← 여기서 실행
├── data.py             # yfinance 수집 + 전처리 (Week 2: 정규화)
├── features.py         # 피처 엔지니어링 (RSI, Bollinger, 변동성 클러스터링)
├── regime.py           # K-Means Regime Detector (Week 2: 비지도학습)
├── model_torch.py      # PyTorch MLP (Week 3-5: MLP + Dropout + EarlyStopping)
├── strategy_kelly.py   # 전략 A: Kelly Criterion + MLP
├── strategy_markowitz.py # 전략 B: Markowitz 포트폴리오 최적화
└── backtest.py         # Walk-Forward 백테스터 + 성과 지표
```

## 기술 스택

| 분류 | 라이브러리 | 수업 연결 |
|------|-----------|----------|
| ML | `torch` (PyTorch) | Week 3–5 MLP |
| 군집화 | `scikit-learn` KMeans | Week 2 |
| 최적화 | `scipy.optimize` | Markowitz |
| GUI | `PySide6` | — |
| 시각화 | `matplotlib` | — |
| 패키지 | `uv` | 수업 표준 |

## 전략 비교

| 전략 | 설명 |
|------|------|
| Buy & Hold | S&P500 단순 보유 (벤치마크) |
| Markowitz | 샤프지수 최대화 포트폴리오 (월별 리밸런싱) |
| Kelly + MLP | PyTorch MLP 예측 확률 × Kelly 포지션 |
| Regime 전환 | Bull/Sideways → Markowitz, Bear → Kelly+MLP |
