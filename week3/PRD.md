# PRD (Product Requirements Document)
# Week 3: 신경망 기초 인터랙티브 학습 도구

---

## 1. 개요 (Overview)

| 항목 | 내용 |
|------|------|
| 제품명 | Neural Networks Week 3 Interactive Viewer |
| 버전 | 1.0.0 |
| 작성일 | 2026-03-29 |
| 대상 사용자 | AI/ML 입문 수강생 |
| 플랫폼 | Python 3.12+, PySide6, Windows/macOS/Linux |

---

## 2. 목적 (Purpose)

Week 3 신경망 기초 실습 5개(Lab 1~5)를 **하나의 GUI 애플리케이션**으로 통합하여,
수강생이 코드를 수정하지 않고도 파라미터를 실시간으로 조작하고 결과를 시각적으로 확인할 수 있도록 한다.

---

## 3. 사용자 스토리 (User Stories)

| ID | As a... | I want to... | So that... |
|----|---------|--------------|------------|
| US-01 | 수강생 | Perceptron의 결정 경계를 시각적으로 확인하고 싶다 | AND/OR/XOR 문제를 직관적으로 이해한다 |
| US-02 | 수강생 | 활성화 함수(Sigmoid/Tanh/ReLU/LeakyReLU)를 한 화면에서 비교하고 싶다 | 각 함수의 특성과 미분값을 이해한다 |
| US-03 | 수강생 | 입력값을 바꿔가며 순전파 계산 과정을 단계별로 보고 싶다 | Forward Propagation의 수학적 구조를 체득한다 |
| US-04 | 수강생 | 은닉층 크기/학습률/에폭 수를 조절하며 MLP 학습을 실험하고 싶다 | Backpropagation과 하이퍼파라미터의 역할을 이해한다 |
| US-05 | 수강생 | 뉴런 수에 따른 함수 근사 능력 변화를 시각적으로 확인하고 싶다 | Universal Approximation Theorem을 직관적으로 이해한다 |

---

## 4. 기능 요구사항 (Functional Requirements)

### 4.1 공통 요구사항
- **FR-01**: 5개 탭(Tab)으로 구성된 단일 윈도우 앱
- **FR-02**: 각 탭에 이론 설명 패널과 시각화 캔버스를 함께 표시
- **FR-03**: 학습/계산은 별도 스레드에서 실행 (UI 블로킹 방지)
- **FR-04**: 한글 폰트 자동 설정 (Malgun Gothic / 시스템 폰트 fallback)

### 4.2 Tab 1 — Perceptron (핵심 키워드)

**Lab 1 핵심 키워드:**

| 키워드 | 정의 |
|--------|------|
| **Perceptron** | 1958년 Rosenblatt 발명, 최초의 학습 가능한 인공 뉴런 |
| **Step Function** | 활성화 함수: x≥0이면 1, 아니면 0 |
| **Linear Separability** | 하나의 직선(결정 경계)으로 데이터 분리 가능 여부 |
| **Decision Boundary** | `wx + b = 0`으로 정의되는 분류 경계 직선 |
| **Weight / Bias** | 학습 가능한 파라미터: 가중치(w)와 편향(b) |
| **Perceptron Learning Rule** | `w ← w + η·(y − ŷ)·x` |
| **XOR Problem** | 단일 퍼셉트론으로 해결 불가능 → Multi-Layer 필요 |

**기능 요구:**
- **FR-1-1**: AND / OR / XOR 세 가지 게이트를 동시에 학습하고 결정 경계를 표시
- **FR-1-2**: 학습 결과(예측값, 정확도)를 텍스트로 출력
- **FR-1-3**: "다시 학습" 버튼으로 랜덤 재학습

### 4.3 Tab 2 — Activation Functions

- **FR-2-1**: Sigmoid, Tanh, ReLU, Leaky ReLU 함수 및 미분 그래프 표시
- **FR-2-2**: Sigmoid vs Tanh 비교, ReLU vs Leaky ReLU 비교 그래프 표시
- **FR-2-3**: 각 함수의 특성 요약(범위, 장/단점, 용도) 텍스트 표시

### 4.4 Tab 3 — Forward Propagation (인터랙티브)

- **FR-3-1**: x1, x2 슬라이더(0.0~1.0)로 입력값 조절
- **FR-3-2**: 슬라이더 변경 시 순전파 자동 재계산 및 시각화 갱신
- **FR-3-3**: 단계별 계산 과정(`z₁`, `a₁`, `z₂`, `a₂`) 텍스트 출력
- **FR-3-4**: 네트워크 구조 다이어그램 + 레이어별 값 변화 그래프 표시

### 4.5 Tab 4 — MLP (Backpropagation)

- **FR-4-1**: 은닉층 뉴런 수(2~16), 학습률(0.01~1.0), 에폭 수(1000~50000) 조절
- **FR-4-2**: 학습 진행률 프로그레스 바 표시
- **FR-4-3**: 학습 완료 후 Loss 곡선, 결정 경계, 은닉층 활성화 히트맵 표시
- **FR-4-4**: 최종 정확도와 Loss를 텍스트로 출력

### 4.6 Tab 5 — Universal Approximation

- **FR-5-1**: 근사 대상 함수 선택 (Sine Wave / Step Function / Complex Function)
- **FR-5-2**: 3가지 뉴런 수(3, 10, 50)로 동시 학습 및 비교 시각화
- **FR-5-3**: 학습 진행 상태 메시지 표시
- **FR-5-4**: 각 모델의 MSE 값 그래프 제목에 표시

---

## 5. 비기능 요구사항 (Non-Functional Requirements)

| ID | 요구사항 | 목표 |
|----|---------|------|
| NF-01 | 응답성 | UI는 항상 반응 가능 (학습 중 버튼 비활성화) |
| NF-02 | 성능 | Tab 2 초기 렌더링 < 1초, Tab 4 학습(10000 에폭) < 30초 |
| NF-03 | 호환성 | Python 3.12+, PySide6 6.x, numpy, matplotlib |
| NF-04 | 가독성 | 한글 레이블 및 설명, 폰트 자동 감지 |

---

## 6. 제외 범위 (Out of Scope)

- GPU 가속
- 모델 저장/불러오기
- 사용자 정의 데이터 업로드
- MNIST 등 실제 데이터셋 적용

---

## 7. 성공 기준 (Success Criteria)

1. 5개 탭 모두 오류 없이 실행됨
2. MLP 학습 결과 XOR 정확도 100% 달성
3. Universal Approximation에서 50 뉴런 MSE < 0.01
4. UI가 학습 중에도 블로킹되지 않음
