# TRD (Technical Requirements Document)
# Week 3: 신경망 기초 인터랙티브 학습 도구

---

## 1. 기술 스택 (Tech Stack)

| 구성요소 | 선택 | 버전 | 이유 |
|---------|------|------|------|
| GUI 프레임워크 | PySide6 | 6.x | Qt6 공식 Python 바인딩, 크로스플랫폼 |
| 시각화 | matplotlib (FigureCanvasQTAgg) | 3.8+ | 기존 코드 재활용, Qt 임베드 지원 |
| 수치 계산 | NumPy | 1.26+ | 행렬 연산, ML 모델 구현 |
| Python | CPython | 3.12+ | 타입 힌트, 성능 |
| 패키지 관리 | uv | - | 빠른 의존성 관리 |

---

## 2. 아키텍처 (Architecture)

```
week3_app.py
│
├── MplCanvas               # matplotlib Figure를 QWidget으로 래핑
│
├── ML Models (순수 NumPy)
│   ├── Perceptron          # Step function 기반 단일 뉴런
│   ├── MLP                 # Sigmoid 기반 2-layer, Backpropagation
│   └── UniversalApproximator  # Tanh 기반 1-hidden-layer
│
├── Worker Threads (QObject + QThread)
│   ├── MLPWorker           # MLP 학습 (10000+ 에폭, 백그라운드)
│   └── UAWorker            # Universal Approximation 학습 (백그라운드)
│
└── Tab Widgets (QWidget)
    ├── Tab1_Perceptron     # Perceptron + 핵심 키워드
    ├── Tab2_Activation     # 활성화 함수 비교
    ├── Tab3_ForwardProp    # 인터랙티브 순전파
    ├── Tab4_MLP            # MLP 학습 실험
    └── Tab5_Universal      # Universal Approximation
```

---

## 3. 클래스 설계 (Class Design)

### 3.1 MplCanvas

```python
class MplCanvas(FigureCanvas):
    figure: Figure
    def __init__(self, figsize: tuple[float, float])
```

- matplotlib `Figure`를 `FigureCanvasQTAgg`로 래핑
- 각 탭에서 `canvas.figure.clear()` 후 서브플롯 재생성
- `canvas.draw()`로 갱신

### 3.2 ML Models

#### Perceptron
```
속성: weights(ndarray), bias(float), lr(float)
메서드: activation(x) → int, predict(inputs) → int, train(X, y, epochs)
```

#### MLP
```
속성: W1, b1, W2, b2, lr, loss_history
메서드: forward(X) → ndarray, backward(X, y, output), train_epoch(X, y) → float, predict(X) → ndarray
```
- Xavier 초기화: `W = randn * sqrt(2/fan_in)`
- 수치 안정성: `clip(x, -500, 500)` in sigmoid

#### UniversalApproximator
```
속성: W1, b1, W2, b2, activation
메서드: forward(x) → ndarray, train(X, y, epochs, lr)
```

### 3.3 Worker Threads

**패턴:** `QObject` 기반 worker를 `QThread`에 `moveToThread()`하여 실행

```python
# 사용 패턴
self._thread = QThread()
self._worker = MLPWorker(hidden_size, lr, epochs)
self._worker.moveToThread(self._thread)
self._thread.started.connect(self._worker.run)
self._worker.progress.connect(self._on_progress)
self._worker.finished.connect(self._on_finished)
self._thread.start()
```

**MLPWorker Signals:**
- `progress(int, float)` — (퍼센트, 현재 loss)
- `finished(object)` — 학습 완료된 MLP 객체

**UAWorker Signals:**
- `progress(str)` — 진행 메시지
- `finished(list)` — `[(n_neurons, x_test, y_true, y_pred, mse), ...]`

---

## 4. UI 레이아웃 (UI Layout)

### 공통 구조
```
QMainWindow
└── QTabWidget (중앙 위젯)
    ├── Tab 1~5: QWidget
    │   └── QHBoxLayout
    │       ├── 좌측 패널 (QWidget, maxWidth=300)
    │       │   └── QVBoxLayout
    │       │       ├── 설명/키워드 (QGroupBox)
    │       │       ├── 컨트롤 (버튼, 슬라이더, 스핀박스)
    │       │       └── 결과 텍스트 (QTextEdit)
    │       └── 우측 패널 (MplCanvas)
```

### Tab별 컨트롤

| Tab | 컨트롤 |
|-----|--------|
| 1 Perceptron | QPushButton "다시 학습" |
| 2 Activation | (정적, 컨트롤 없음) |
| 3 ForwardProp | QSlider x2 (x1, x2 입력값) |
| 4 MLP | QSpinBox(hidden), QDoubleSpinBox(lr), QSpinBox(epochs), QProgressBar, QPushButton |
| 5 Universal | QComboBox(함수 선택), QProgressBar, QPushButton |

---

## 5. 스레드 안전성 (Thread Safety)

- 모든 matplotlib 그리기는 **메인 스레드**에서만 수행
- Worker는 계산만 수행, UI 업데이트는 Signal/Slot으로 메인 스레드에 위임
- 학습 시작 시 버튼 비활성화 → 완료 시 재활성화
- 이전 학습 스레드가 남아 있으면 `quit()` → `wait()` 후 새 스레드 시작

---

## 6. 파일 구조 (File Structure)

```
week3/
├── week3_app.py          ← 메인 PySide6 애플리케이션 (신규)
├── PRD.md                ← 제품 요구사항 문서 (신규)
├── TRD.md                ← 기술 요구사항 문서 (신규, 이 파일)
├── 01_perceptron.py      ← 원본 실습 코드
├── 02_activation_functions.py
├── 03_forward_propagation.py
├── 04_mlp_numpy.py
├── 05_universal_approximation.py
├── week3.md              ← 이론 강의자료
└── outputs/              ← 원본 코드 출력 이미지
```

---

## 7. 실행 방법 (How to Run)

```bash
cd /home/dullear/AIandMLcourse
uv run week3/week3_app.py
```

**의존성 설치 (최초 1회):**
```bash
uv add pyside6
```

---

## 8. 주요 기술 결정 (Key Technical Decisions)

| 결정 | 이유 |
|------|------|
| matplotlib 임베드 방식 | 기존 실습 코드의 matplotlib 로직을 그대로 재사용 가능 |
| QObject + moveToThread | 서브클래스 QThread보다 더 Qt-idiomatic한 방식 |
| NumPy 순수 구현 | PyTorch/TensorFlow 없이 학습 원리를 명확하게 이해 |
| 탭 위젯 | 5개 Lab을 독립적으로 유지하되 하나의 앱으로 통합 |
| Xavier 초기화 | 랜덤 초기화보다 학습 안정성 향상 |

---

## 9. 에러 처리 (Error Handling)

- 한글 폰트 미설치 시: 영문 폰트로 fallback (레이블 깨짐 방지)
- MLP 학습 중 overflow: `np.clip(x, -500, 500)` in sigmoid
- 스레드 재시작: 기존 QThread `quit()` + `wait()` 후 새 인스턴스 생성
