# TRD (Technical Requirements Document)
# Week 4: 물리 데이터로 학습하기 - Neural Networks GUI

---

## 1. 기술 스택 (Tech Stack)

| 구성요소 | 선택 | 버전 | 이유 |
|---------|------|------|------|
| GUI 프레임워크 | PySide6 | 6.x | Qt6 공식 Python 바인딩, 크로스플랫폼 |
| 시각화 | matplotlib (FigureCanvasQTAgg) | 3.8+ | 기존 코드 재활용, Qt 임베드 지원 |
| ML 프레임워크 | TensorFlow/Keras | 2.x | Neural Network 학습 |
| 수치 계산 | NumPy | 1.26+ | 행렬 연산, 데이터 생성 |
| Python | CPython | 3.12+ | 타입 힌트, 성능 |
| 패키지 관리 | uv | - | 빠른 의존성 관리 |

---

## 2. 아키텍처 (Architecture)

```
week4_app.py
│
├── MplCanvas                   # matplotlib Figure → QWidget 래핑
│   └── FigureCanvasQTAgg
│
├── ProgressCallback            # Keras 콜백 → Qt Signal 브릿지
│   └── keras.callbacks.Callback
│
├── BaseTab (QWidget)           # 4개 탭 공통 레이아웃
│   ├── 좌측 패널 (250px 고정)
│   │   ├── 제목 / 설명 레이블
│   │   ├── [학습 시작] QPushButton
│   │   ├── QProgressBar
│   │   └── QTextEdit (상태 로그)
│   └── 우측 패널 (확장)
│       ├── NavigationToolbar2QT
│       └── MplCanvas
│
├── Worker Threads (QThread)
│   ├── Lab1Worker              # 1D 함수 근사 학습
│   ├── Lab2Worker              # 포물선 운동 학습
│   ├── Lab3Worker              # 과적합 학습 (3 모델)
│   └── Lab4Worker              # 진자 주기 학습
│
├── Tab Widgets
│   ├── FunctionApproxTab       # Lab 1
│   ├── ProjectileTab           # Lab 2
│   ├── OverfittingTab          # Lab 3
│   └── PendulumTab             # Lab 4
│
└── MainWindow (QMainWindow)
    └── QTabWidget (4개 탭)
```

---

## 3. 스레드 설계 (Thread Design)

### 문제: TF 학습 블로킹
TensorFlow `model.fit()`은 블로킹 콜 → 메인 스레드에서 실행 시 UI 응답 불가

### 해결: QThread + Keras Callback
```
메인 스레드 (Qt Event Loop)
    │
    ├── 버튼 클릭 → Worker.start()
    │
Worker 스레드 (QThread)
    │
    ├── model.fit(callbacks=[ProgressCallback])
    │
    ProgressCallback.on_epoch_end()
        │
        └── Signal.emit(epoch, total, loss, val_loss)
            │
            └── [Qt Signal/Slot 메커니즘으로 메인 스레드로 전달]
                │
                └── 메인 스레드: progress_bar.setValue() / log 업데이트
```

### Signal 정의
```python
class Worker(QThread):
    progress = Signal(int, int, float, float)  # epoch, total, loss, val_loss
    finished = Signal(dict)                     # 결과 딕셔너리
    error = Signal(str)                         # 오류 메시지
```

---

## 4. 학습 설정 (Training Config)

| Lab | Epochs | Batch | 비고 |
|-----|--------|-------|------|
| Lab 1 (함수 근사) | 1000 | 32 | EarlyStopping(patience=200), 3개 함수 순차 |
| Lab 2 (포물선) | 100 | 32 | validation_split=0.2 |
| Lab 3 (과적합) | 200 | 16 | 3개 모델 순차, validation_data 별도 |
| Lab 4 (진자) | 100 | 32 | validation_split=0.2, MAPE metric |

---

## 5. 데이터 흐름 (Data Flow)

```
[Worker.run()]
    │
    ├── 데이터 생성 (NumPy)
    ├── 모델 생성 (Keras Sequential)
    ├── model.fit() with ProgressCallback
    │   └── on_epoch_end → progress Signal emit
    ├── model.predict() → 결과 수집
    └── finished Signal emit(results dict)

[Tab Widget - finished 슬롯]
    │
    ├── results 수신
    ├── matplotlib Figure 업데이트
    └── canvas.draw()
```

---

## 6. 파일 구조 (File Structure)

```
/home/dullear/aicoursework/week4/
├── week4_app.py        # 메인 애플리케이션 (단일 파일)
├── PRD.md              # Product Requirements Document
├── TRD.md              # Technical Requirements Document
└── outputs/            # 그래프 저장 (선택)
```

---

## 7. 핵심 구현 패턴

### Matplotlib Qt 임베드
```python
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

fig = Figure(figsize=(12, 8), tight_layout=True)
canvas = FigureCanvasQTAgg(fig)
toolbar = NavigationToolbar2QT(canvas, parent_widget)
```

### QThread + Signal 패턴
```python
class Worker(QThread):
    progress = Signal(int, int, float, float)
    finished = Signal(dict)

    def run(self):
        cb = ProgressCallback(self.total_epochs, self.progress.emit)
        history = model.fit(..., callbacks=[cb])
        self.finished.emit({'history': history.history, ...})
```

### Keras → Qt 브릿지
```python
class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs, emit_fn):
        super().__init__()
        self.total_epochs = total_epochs
        self.emit_fn = emit_fn

    def on_epoch_end(self, epoch, logs=None):
        self.emit_fn(epoch + 1, self.total_epochs,
                     logs.get('loss', 0),
                     logs.get('val_loss', logs.get('loss', 0)))
```
