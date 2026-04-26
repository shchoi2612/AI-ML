# PRD: Can ML Beat the Market?
### 전산물리 중간 프로젝트 — ML로 효율적 시장 가설(EMH) 검증

**과목**: 전산물리 (Computational Physics) — 부산대학교 물리학과  
**제출**: Week 8 중간 프로젝트 (Part I: Weeks 1–7 학습 내용 응용)  
**마감**: PRD/TRD 2026-04-21 | 유튜브 영상 2026-04-28 | GitHub 코드 TBD

---

## 1. 프로젝트 개요

### 배경 — 수업 연결

Week 1–7에서 배운 핵심 개념들:

| 주차 | 핵심 개념 | 이 프로젝트에서의 적용 |
|------|----------|----------------------|
| Week 2 | 선형 회귀, Gradient Descent, 지도학습 | 주가 방향성 예측 모델 기반 |
| Week 3 | MLP, Activation, Backpropagation | 방향성 예측 신경망 구조 |
| Week 4 | 물리 데이터 학습, Overfitting 개념 | 금융 시계열 = 복잡계 데이터 |
| Week 5 | Regularization, Dropout | 금융 데이터 과적합 방지 |
| Week 6 | Transformer, 시계열 예측 | 향후 개선 방향 |
| Week 2 | K-Means 군집화 (비지도학습) | Regime 감지 |

### 교수님 특강 — 투자의 과학

교수님은 별도 특강에서 다음을 강조하셨다:
- "차트 분석으로 돈 벌기 불가능" (약형 EMH)
- "공개 정보로도 초과 수익 불가"
- 최적 전략: 분산투자 + 샤프지수 최대화 (Markowitz, 1952)
- 참고 지수: S&P500, 나스닥, 필라델피아 반도체

이는 **효율적 시장 가설(Efficient Market Hypothesis, EMH)** — 약형(Weak-form)에 해당한다.

### 핵심 질문
> "Week 1–7에서 배운 ML 기법들을 금융 데이터에 적용했을 때, S&P500 단순 보유(Buy & Hold)를 능가할 수 있는가? — 교수님의 EMH 주장은 실제 데이터로 검증되는가?"

---

## 2. 목표 (Objectives)

| # | 목표 | 관련 수업 내용 |
|---|------|--------------|
| O1 | TensorFlow MLP로 주가 방향성 예측 | Week 3–4: Neural Network |
| O2 | Dropout/Regularization으로 과적합 방지 | Week 5: Regularization |
| O3 | K-Means로 시장 Regime 감지 (비지도학습) | Week 2: K-Means 군집화 |
| O4 | Kelly Criterion + ML 포지션 사이징 | Week 2: 지도학습 응용 |
| O5 | Markowitz 포트폴리오 최적화 (샤프 최대화) | 특강: Markowitz, Sharpe |
| O6 | 4가지 전략 비교 → EMH 지지/반박 결론 | 종합 분석 |

---

## 3. 사용자 및 활용 대상

- **전산물리 중간 프로젝트** 평가 대상 (Part I 종합 응용)
- Week 1–7 학습 내용을 실제 복잡계(금융 시장)에 적용한 사례 연구

---

## 4. 핵심 기능 요구사항

### F1. 데이터 수집 및 전처리
- Yahoo Finance(yfinance)로 주가 데이터 자동 수집
- 대상: S&P500(^GSPC), NASDAQ(^IXIC), 필라델피아 반도체(^SOX), VIX(^VIX)
- 백테스트 기간: 2015-01-01 ~ 2024-12-31 (약 10년)
- 전처리: 수익률, 로그수익률, Min-Max 정규화 (Week 2: 정규화)

### F2. 전략 A — TensorFlow MLP + Kelly Criterion
- **Week 3–5 직접 응용**: MLP로 다음 5일 방향성(상승/하락) 분류
- 예측 확률 p → Kelly 공식 `f* = (p - q) / b` 으로 포지션 크기 결정
- Dropout 적용 (Week 5: 과적합 방지)

### F3. 전략 B — Markowitz 포트폴리오 최적화
- 샤프지수 최대화 포트폴리오: S&P500 + NASDAQ + 필라델피아 반도체
- scipy.optimize로 최적 비중 월별 리밸런싱

### F4. ML Regime Detection (K-Means 비지도학습)
- **Week 2 K-Means 직접 응용**: 시장 상태를 Bull/Bear/Sideways 3분류
- 핵심 피처: **변동성 클러스터링** (고변동성 → 고변동성 지속, ARCH 효과)
- Regime에 따라 전략 A/B 자동 전환

### F5. 백테스트 & 비교 분석
- 4가지 비교: 전략A(Kelly+MLP) vs 전략B(Markowitz) vs ML+Regime vs Buy&Hold
- 성과 지표: 샤프지수, 최대낙폭(MDD), CAGR, 누적수익률
- 시각화: 누적수익률 곡선, Drawdown, 피처 중요도

---

## 5. 성공 기준 (Success Metrics)

| 지표 | 기준 | 근거 |
|------|------|------|
| MLP 방향성 예측 Accuracy | > 55% | Kelly 양수 기대값 조건 |
| 과적합 방지 | Test Accuracy ≥ Train - 5% | Week 5: Regularization 효과 |
| 결론 명확성 | EMH 지지 또는 반박 중 하나 | 어느 결과든 과학적으로 유효 |
| 코드 재현성 | GitHub에서 end-to-end 실행 | 과제 제출 기준 |
| 시각화 | 그래프 최소 4개 이상 | YouTube 영상 콘텐츠 |

---

## 6. 제약사항

- 공개 데이터만 사용 (No premium API)
- 미래 데이터 사용 금지 (Look-ahead bias 방지)
- 거래비용 미반영 (단순화 허용, 명시적 언급 필요)
- TensorFlow 사용 (수업 표준 스택 준수)

---

## 7. 제출 요건 및 타임라인

| 날짜 | 마일스톤 |
|------|----------|
| 2026-04-21 | PRD + TRD 제출 ✅ |
| 2026-04-28 (예정) | YouTube 소개영상 10분 제출 |
| TBD | GitHub 코드 링크 제출 |
