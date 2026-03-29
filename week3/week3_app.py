"""
Week 3: 신경망 기초 인터랙티브 학습 도구
PySide6 + matplotlib 임베드 방식
실행: uv run week3/week3_app.py
"""

import sys
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.font_manager as fm

# 한글 폰트 및 마이너스 기호 설정 (import 직후 즉시 적용)
def _apply_font():
    names = [f.name for f in fm.fontManager.ttflist]
    for name in ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang', 'AppleGothic']:
        if name in names:
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

_apply_font()

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QGroupBox,
    QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject




# ─────────────────────── Matplotlib 캔버스 ───────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(12, 5)):
        self.figure = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.figure)
        self.setMinimumSize(500, 350)


# ─────────────────────── ML 모델 (순수 NumPy) ───────────────────────
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(X, y):
                error = label - self.predict(inputs)
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate
        self.loss_history = []
        self.z1 = self.a1 = self.z2 = self.a2 = None

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_deriv(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_epoch(self, X, y):
        out = self.forward(X)
        loss = float(np.mean((out - y) ** 2))
        self.loss_history.append(loss)
        self.backward(X, y, out)
        return loss

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


class UniversalApproximator:
    def __init__(self, n_hidden, activation='tanh'):
        self.n_hidden = n_hidden
        self.activation = activation
        lim = np.sqrt(6 / (1 + n_hidden))
        self.W1 = np.random.uniform(-lim, lim, (1, n_hidden))
        self.b1 = np.zeros(n_hidden)
        lim = np.sqrt(6 / (n_hidden + 1))
        self.W2 = np.random.uniform(-lim, lim, (n_hidden, 1))
        self.b2 = np.zeros(1)

    def _act(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return relu(x)
        return sigmoid(x)

    def forward(self, x):
        return self._act(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def train(self, X, y, epochs=5000, lr=0.05):
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self._act(z1)
            out = a1 @ self.W2 + self.b2
            dL = 2 * (out - y) / len(X)
            self.W2 -= lr * (a1.T @ dL)
            self.b2 -= lr * np.sum(dL, axis=0)
            da1 = dL @ self.W2.T
            if self.activation == 'tanh':
                dz1 = da1 * (1 - a1**2)
            elif self.activation == 'relu':
                dz1 = da1 * (z1 > 0)
            else:
                dz1 = da1 * a1 * (1 - a1)
            self.W1 -= lr * (X.T @ dz1)
            self.b1 -= lr * np.sum(dz1, axis=0)


# ─────────────────────── Worker Threads ───────────────────────
class MLPWorker(QObject):
    progress = Signal(int, float)   # (퍼센트, loss)
    finished = Signal(object)       # 완성된 MLP

    def __init__(self, hidden_size, lr, epochs):
        super().__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs

    def run(self):
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]], dtype=float)
        mlp = MLP(2, self.hidden_size, 1, self.lr)
        step = max(1, self.epochs // 100)
        for epoch in range(self.epochs):
            loss = mlp.train_epoch(X, y)
            if (epoch + 1) % step == 0:
                pct = int((epoch + 1) / self.epochs * 100)
                self.progress.emit(pct, loss)
        self.finished.emit(mlp)


class UAWorker(QObject):
    progress = Signal(str)
    finished = Signal(list)   # [(n, x_test, y_true, y_pred, mse), ...]

    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name

    def run(self):
        x_tr = np.linspace(0, 1, 100).reshape(-1, 1)
        x_te = np.linspace(0, 1, 200).reshape(-1, 1)
        funcs = {
            'Sine Wave':      lambda x: np.sin(2*np.pi*x),
            'Step Function':  lambda x: np.where(x < 0.5, 0.0, 1.0),
            'Complex Function': lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.3*np.cos(6*np.pi*x),
        }
        target = funcs[self.func_name]
        y_tr = target(x_tr)
        y_te = target(x_te)
        results = []
        for n in [3, 10, 50]:
            self.progress.emit(f"{self.func_name} — {n}개 뉴런 학습 중...")
            lr = 0.05 if n < 20 else 0.01
            model = UniversalApproximator(n, activation='tanh')
            model.train(x_tr, y_tr, epochs=5000, lr=lr)
            y_pred = model.forward(x_te)
            mse = float(np.mean((y_pred - y_te)**2))
            results.append((n, x_te, y_te, y_pred, mse))
        self.finished.emit(results)


# ─────────────────────── Tab 1: Perceptron ───────────────────────
class Tab1_Perceptron(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # ── 좌측 패널 ──
        left = QWidget(); left.setMaximumWidth(310)
        lv = QVBoxLayout(left)

        kw_box = QGroupBox("Lab 1 — 핵심 키워드 (Perceptron)")
        kv = QVBoxLayout()
        kw_items = [
            ("<b>Perceptron</b>", "최초의 학습 가능한 인공 뉴런 (Rosenblatt, 1958)"),
            ("<b>Step Function</b>", "계단 활성화: x≥0 → 1, x&lt;0 → 0"),
            ("<b>Linear Separability</b>", "직선 하나로 데이터 분리 가능 여부"),
            ("<b>Decision Boundary</b>", "결정 경계: <i>w·x + b = 0</i>"),
            ("<b>Weight / Bias</b>", "학습 파라미터: 가중치(w), 편향(b)"),
            ("<b>Learning Rule</b>", "w ← w + η·(y−ŷ)·x"),
            ("<b>XOR Problem</b>", "선형 분리 불가능 → Multi-Layer 필요"),
        ]
        for bold, desc in kw_items:
            lbl = QLabel(f"{bold}<br><small style='color:#555'>{desc}</small>")
            lbl.setWordWrap(True)
            lbl.setContentsMargins(2, 3, 2, 3)
            kv.addWidget(lbl)
        kw_box.setLayout(kv)
        lv.addWidget(kw_box)

        self.run_btn = QPushButton("▶ 다시 학습")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        lv.addWidget(self.result_text)
        lv.addStretch()

        layout.addWidget(left)

        # ── 우측: 캔버스 ──
        self.canvas = MplCanvas(figsize=(11, 4))
        layout.addWidget(self.canvas)

        self._run()

    def _run(self):
        np.random.seed(int(np.random.randint(0, 9999)))
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        gates = [
            ('AND Gate\n(Linearly Sep.)', np.array([0,0,0,1]), 100),
            ('OR Gate\n(Linearly Sep.)',  np.array([0,1,1,1]), 100),
            ('XOR Gate\n(NOT Sep.!)',     np.array([0,1,1,0]), 1000),
        ]
        lines = []
        perceptrons = []
        for name, y, ep in gates:
            p = Perceptron(2)
            p.train(X, y, ep)
            preds = [p.predict(x) for x in X]
            acc = sum(a == b for a, b in zip(preds, y)) / 4 * 100
            lines.append(f"{name.split(chr(10))[0]}: {preds}  정확도 {acc:.0f}%")
            perceptrons.append((p, y, name))
        self.result_text.setPlainText('\n'.join(lines))

        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(1, 3)
        colors_bg = ['#aaaaff', '#ffaaaa']

        for ax, (p, y, title) in zip(axes, perceptrons):
            xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 120), np.linspace(-0.5, 1.5, 120))
            Z = np.array([p.predict(np.array([xi, yi])) for xi, yi in zip(xx.ravel(), yy.ravel())])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5], colors=colors_bg)
            for pt, lbl in zip(X, y):
                filled = lbl == 1
                kw = dict(s=200, zorder=3, linewidth=2)
                if filled:
                    ax.scatter(pt[0], pt[1], c='red',  marker='o', edgecolors='black', **kw)
                else:
                    ax.scatter(pt[0], pt[1], c='blue', marker='x', **kw)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('x1'); ax.set_ylabel('x2')
            ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
            ax.grid(True, alpha=0.3)
            leg = [mpatches.Patch(color='red', alpha=0.5, label='Output=1'),
                   mpatches.Patch(color='blue', alpha=0.5, label='Output=0')]
            ax.legend(handles=leg, fontsize=8)

        fig.tight_layout()
        self.canvas.draw()


# ─────────────────────── Tab 2: Activation Functions ───────────────────────
class Tab2_Activation(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # ── 좌측 ──
        left = QWidget(); left.setMaximumWidth(280)
        lv = QVBoxLayout(left)

        info_box = QGroupBox("활성화 함수 특성 요약")
        iv = QVBoxLayout()
        rows = [
            ("<b>Sigmoid</b> σ(x)=1/(1+e⁻ˣ)",   "범위 (0,1) | 이진 분류 출력층\n단점: Vanishing Gradient"),
            ("<b>Tanh</b> tanh(x)",               "범위 (-1,1) | 0 중심\n단점: Vanishing Gradient"),
            ("<b>ReLU</b> max(0,x)",              "범위 [0,∞) | 현대 표준\n단점: Dying ReLU"),
            ("<b>Leaky ReLU</b> max(αx,x)",       "범위 (-∞,∞) | Dying ReLU 해결\n단점: α 선택 필요"),
        ]
        for bold, desc in rows:
            lbl = QLabel(f"{bold}<br><small style='color:#555'>{desc}</small>")
            lbl.setWordWrap(True)
            lbl.setContentsMargins(2, 4, 2, 4)
            iv.addWidget(lbl)
        info_box.setLayout(iv)
        lv.addWidget(info_box)

        tip = QLabel("<b>사용 가이드:</b><br>"
                     "• 은닉층 → <b>ReLU</b><br>"
                     "• 이진 분류 출력 → <b>Sigmoid</b><br>"
                     "• 회귀 출력 → 없음(선형)<br>"
                     "• RNN/LSTM → <b>Tanh</b>")
        tip.setWordWrap(True)
        lv.addWidget(tip)
        lv.addStretch()
        layout.addWidget(left)

        # ── 우측: 캔버스 ──
        self.canvas = MplCanvas(figsize=(11, 8))
        layout.addWidget(self.canvas)
        self._plot()

    def _plot(self):
        x = np.linspace(-5, 5, 300)
        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(2, 2)

        # 함수 비교
        ax = axes[0, 0]
        ax.plot(x, sigmoid(x),    label='Sigmoid', lw=2)
        ax.plot(x, np.tanh(x),    label='Tanh',    lw=2)
        ax.plot(x, relu(x),       label='ReLU',    lw=2)
        ax.plot(x, leaky_relu(x), label='Leaky ReLU', lw=2, ls='--')
        ax.axhline(0, color='k', alpha=0.3); ax.axvline(0, color='k', alpha=0.3)
        ax.set_title('Activation Functions Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.legend(); ax.grid(True, alpha=0.3)

        # 미분 비교
        ax = axes[0, 1]
        sig_d = sigmoid(x) * (1 - sigmoid(x))
        tanh_d = 1 - np.tanh(x)**2
        relu_d = np.where(x > 0, 1.0, 0.0)
        lrelu_d = np.where(x > 0, 1.0, 0.01)
        ax.plot(x, sig_d,   label="Sigmoid'", lw=2)
        ax.plot(x, tanh_d,  label="Tanh'",    lw=2)
        ax.plot(x, relu_d,  label="ReLU'",    lw=2)
        ax.plot(x, lrelu_d, label="Leaky'",   lw=2, ls='--')
        ax.axhline(0, color='k', alpha=0.3); ax.axvline(0, color='k', alpha=0.3)
        ax.set_title('Derivatives (Gradients)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel("f'(x)"); ax.legend(); ax.grid(True, alpha=0.3)

        # Sigmoid vs Tanh
        ax = axes[1, 0]
        ax.plot(x, sigmoid(x), label='Sigmoid: (0,1)', lw=3)
        ax.plot(x, np.tanh(x), label='Tanh: (-1,1)',   lw=3)
        ax.axhline(0, color='k', alpha=0.3); ax.axhline(0.5, color='C0', ls='--', alpha=0.4, label='Sigmoid center(0.5)')
        ax.set_title('Sigmoid vs Tanh (Center Difference)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('Output'); ax.legend(); ax.grid(True, alpha=0.3)

        # ReLU vs Leaky ReLU
        ax = axes[1, 1]
        ax.plot(x, relu(x),       label='ReLU (neg->dead)', lw=3)
        ax.plot(x, leaky_relu(x), label='Leaky ReLU (neg->alive)', lw=3)
        ax.axvline(0, color='r', ls='--', alpha=0.5, label='Boundary x=0')
        ax.set_title('ReLU vs Leaky ReLU (Dying ReLU Problem)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('Output'); ax.legend(); ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.canvas.draw()


# ─────────────────────── Tab 3: Forward Propagation ───────────────────────
class Tab3_ForwardProp(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        np.random.seed(42)
        self._W1 = np.random.randn(2, 3) * 0.5
        self._b1 = np.random.randn(3) * 0.1
        self._W2 = np.random.randn(3, 1) * 0.5
        self._b2 = np.random.randn(1) * 0.1

        # ── 좌측 ──
        left = QWidget(); left.setMaximumWidth(300)
        lv = QVBoxLayout(left)

        ctrl_box = QGroupBox("입력값 조절 (2-3-1 네트워크)")
        cv = QVBoxLayout()

        cv.addWidget(QLabel("x1:"))
        self.sl_x1 = QSlider(Qt.Horizontal)
        self.sl_x1.setRange(0, 100); self.sl_x1.setValue(50)
        self.lbl_x1 = QLabel("0.50")
        self.sl_x1.valueChanged.connect(lambda v: (self.lbl_x1.setText(f"{v/100:.2f}"), self._update()))
        cv.addWidget(self.sl_x1); cv.addWidget(self.lbl_x1)

        cv.addWidget(QLabel("x2:"))
        self.sl_x2 = QSlider(Qt.Horizontal)
        self.sl_x2.setRange(0, 100); self.sl_x2.setValue(80)
        self.lbl_x2 = QLabel("0.80")
        self.sl_x2.valueChanged.connect(lambda v: (self.lbl_x2.setText(f"{v/100:.2f}"), self._update()))
        cv.addWidget(self.sl_x2); cv.addWidget(self.lbl_x2)

        ctrl_box.setLayout(cv)
        lv.addWidget(ctrl_box)

        step_box = QGroupBox("단계별 계산 과정")
        sv = QVBoxLayout()
        self.step_text = QTextEdit()
        self.step_text.setReadOnly(True)
        self.step_text.setFont(__import__('PySide6.QtGui', fromlist=['QFont']).QFont('Courier New', 9))
        sv.addWidget(self.step_text)
        step_box.setLayout(sv)
        lv.addWidget(step_box)
        lv.addStretch()
        layout.addWidget(left)

        # ── 우측 ──
        self.canvas = MplCanvas(figsize=(11, 8))
        layout.addWidget(self.canvas)

        self._update()

    def _compute(self, x1, x2):
        X = np.array([x1, x2])
        z1 = X @ self._W1 + self._b1
        a1 = relu(z1)
        z2 = a1 @ self._W2 + self._b2
        a2 = sigmoid(z2)
        return X, z1, a1, z2, a2

    def _update(self):
        x1 = self.sl_x1.value() / 100
        x2 = self.sl_x2.value() / 100
        X, z1, a1, z2, a2 = self._compute(x1, x2)

        lines = [
            "[Layer 1: Input → Hidden (ReLU)]",
            f"  z1 = X @ W1 + b1",
            f"     = [{x1:.2f}, {x2:.2f}] @ W1 + b1",
            f"     = [{z1[0]:.3f}, {z1[1]:.3f}, {z1[2]:.3f}]",
            f"  a1 = ReLU(z1)",
            f"     = [{a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.3f}]",
            "",
            "[Layer 2: Hidden → Output (Sigmoid)]",
            f"  z2 = a1 @ W2 + b2",
            f"     = {z2[0]:.4f}",
            f"  a2 = Sigmoid(z2)",
            f"     = {a2[0]:.4f}  ← 최종 출력",
        ]
        self.step_text.setPlainText('\n'.join(lines))

        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(2, 2)

        # 네트워크 다이어그램
        ax = axes[0, 0]
        ax.set_xlim(0, 4); ax.set_ylim(-0.5, 4.5); ax.axis('off')
        ax.set_title('Network Architecture (2-3-1)', fontsize=11, fontweight='bold')
        input_y = [1.0, 3.0]
        hidden_y = [0.5, 2.0, 3.5]
        out_y = [2.0]
        for i, y in enumerate(input_y):
            ax.add_patch(mpatches.Circle((0.5, y), 0.25, color='#aaddff', ec='black', lw=2))
            ax.text(0.5, y, f'x{i+1}\n{[x1,x2][i]:.2f}', ha='center', va='center', fontsize=8, fontweight='bold')
        for i, y in enumerate(hidden_y):
            ax.add_patch(mpatches.Circle((2.0, y), 0.25, color='#aaffaa', ec='black', lw=2))
            ax.text(2.0, y, f'h{i+1}\n{a1[i]:.2f}', ha='center', va='center', fontsize=8, fontweight='bold')
        ax.add_patch(mpatches.Circle((3.5, out_y[0]), 0.25, color='#ffaaaa', ec='black', lw=2))
        ax.text(3.5, out_y[0], f'y\n{a2[0]:.3f}', ha='center', va='center', fontsize=8, fontweight='bold')
        for iy in input_y:
            for hy in hidden_y:
                ax.plot([0.75, 1.75], [iy, hy], 'k-', alpha=0.2, lw=1)
        for hy in hidden_y:
            ax.plot([2.25, 3.25], [hy, out_y[0]], 'k-', alpha=0.2, lw=1)
        ax.text(0.5, -0.2, 'Input', ha='center', fontsize=9, fontweight='bold')
        ax.text(2.0, -0.2, 'Hidden\n(ReLU)', ha='center', fontsize=9, fontweight='bold')
        ax.text(3.5, -0.2, 'Output\n(Sigmoid)', ha='center', fontsize=9, fontweight='bold')

        # Layer 1 값 비교
        ax = axes[0, 1]
        n = len(z1)
        xpos = np.arange(n)
        w = 0.25
        ax.bar(xpos - w, [x1, x2, 0], w, label='Input', color='#4488ff', alpha=0.8)
        ax.bar(xpos,     z1,          w, label='z1 (pre-ReLU)', color='orange', alpha=0.8)
        ax.bar(xpos + w, a1,          w, label='a1 (post-ReLU)', color='green', alpha=0.8)
        ax.set_title('Layer 1: z1 vs a1 (ReLU)', fontsize=11, fontweight='bold')
        ax.set_xticks(xpos); ax.set_xticklabels([f'H{i+1}' for i in range(n)])
        ax.set_ylabel('Value'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Layer 2 값 비교
        ax = axes[1, 0]
        labels2 = ['a1[0] Input', 'z2 (pre-Sigmoid)', 'a2 (Final Output)']
        vals2 = [a1[0], float(z2[0]), float(a2[0])]
        colors2 = ['#88cc88', 'orange', '#ff6666']
        bars = ax.barh(labels2, vals2, color=colors2, alpha=0.8)
        for bar, val in zip(bars, vals2):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
        ax.set_title('Layer 2: Hidden→Output (Sigmoid)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Value'); ax.grid(True, alpha=0.3)

        # 수식 요약
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Forward Pass Summary', fontsize=11, fontweight='bold')
        eqs = [
            f"Input: x = [{x1:.2f}, {x2:.2f}]",
            "",
            "Layer 1 (2→3, ReLU):",
            f"  z₁ = X@W₁+b₁ = [{z1[0]:.3f},{z1[1]:.3f},{z1[2]:.3f}]",
            f"  a₁ = ReLU(z₁) = [{a1[0]:.3f},{a1[1]:.3f},{a1[2]:.3f}]",
            "",
            "Layer 2 (3→1, Sigmoid):",
            f"  z₂ = a₁@W₂+b₂ = {z2[0]:.4f}",
            f"  a₂ = σ(z₂)     = {a2[0]:.4f}",
            "",
            f">>> Final Output: {a2[0]:.4f}",
        ]
        yp = 0.95
        for eq in eqs:
            size = 10 if eq.startswith("Layer") or eq.startswith(">>>") else 9
            fw = 'bold' if eq.startswith("Layer") or eq.startswith(">>>") else 'normal'
            ax.text(0.05, yp, eq, fontsize=size, fontweight=fw,
                    family='monospace', transform=ax.transAxes)
            yp -= 0.082

        fig.tight_layout()
        self.canvas.draw()


# ─────────────────────── Tab 4: MLP (Backpropagation) ───────────────────────
class Tab4_MLP(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self._thread = None
        self._worker = None

        # ── 좌측 ──
        left = QWidget(); left.setMaximumWidth(300)
        lv = QVBoxLayout(left)

        ctrl_box = QGroupBox("하이퍼파라미터 설정")
        cv = QVBoxLayout()

        cv.addWidget(QLabel("은닉층 뉴런 수:"))
        self.sp_hidden = QSpinBox(); self.sp_hidden.setRange(2, 16); self.sp_hidden.setValue(4)
        cv.addWidget(self.sp_hidden)

        cv.addWidget(QLabel("학습률 (lr):"))
        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setRange(0.01, 1.0); self.sp_lr.setSingleStep(0.05); self.sp_lr.setValue(0.5)
        cv.addWidget(self.sp_lr)

        cv.addWidget(QLabel("에폭 수:"))
        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(1000, 50000); self.sp_epochs.setSingleStep(1000)
        self.sp_epochs.setValue(10000)
        cv.addWidget(self.sp_epochs)

        ctrl_box.setLayout(cv)
        lv.addWidget(ctrl_box)

        self.run_btn = QPushButton("▶ 학습 시작")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        lv.addWidget(self.progress_bar)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        lv.addWidget(self.result_text)

        theory = QLabel(
            "<b>Backpropagation 핵심:</b><br>"
            "δ₂ = (a₂−y) ⊙ σ'(z₂)<br>"
            "dW₂ = a₁ᵀ @ δ₂<br>"
            "δ₁ = (δ₂@W₂ᵀ) ⊙ σ'(z₁)<br>"
            "dW₁ = Xᵀ @ δ₁<br>"
            "W ← W − α·dW"
        )
        theory.setWordWrap(True)
        lv.addWidget(theory)
        lv.addStretch()
        layout.addWidget(left)

        # ── 우측 ──
        self.canvas = MplCanvas(figsize=(11, 4))
        layout.addWidget(self.canvas)

        self.result_text.setPlainText("'학습 시작' 버튼을 눌러 XOR 문제를 풀어보세요.")

    def _run(self):
        if self._thread and self._thread.isRunning():
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.setPlainText("학습 중...")

        self._thread = QThread()
        self._worker = MLPWorker(
            self.sp_hidden.value(),
            self.sp_lr.value(),
            self.sp_epochs.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, pct, loss):
        self.progress_bar.setValue(pct)
        self.result_text.setPlainText(f"학습 중... {pct}%\n현재 Loss: {loss:.6f}")

    def _on_finished(self, mlp):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]], dtype=float)
        preds = mlp.forward(X)
        labels = mlp.predict(X)
        acc = float(np.mean(labels == y.astype(int))) * 100
        final_loss = mlp.loss_history[-1]

        lines = ["XOR 학습 결과:", "입력    | 예측    | 정답"]
        for inp, p, t in zip(X, preds, y):
            lines.append(f"({int(inp[0])},{int(inp[1])}) | {p[0]:.4f} | {int(t[0])}")
        lines += [f"\n정확도: {acc:.0f}%", f"최종 Loss: {final_loss:.6f}"]
        self.result_text.setPlainText('\n'.join(lines))

        self._plot(mlp, X, y)

    def _plot(self, mlp, X, y):
        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(1, 3)

        # Loss 곡선
        ax = axes[0]
        ax.plot(mlp.loss_history, lw=2)
        ax.set_title('Training Loss (MSE)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        # 결정 경계
        ax = axes[1]
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cf = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
        fig.colorbar(cf, ax=ax, label='Output Prob')
        for pt, lbl in zip(X, y):
            filled = lbl[0] == 1
            kw = dict(s=300, lw=3, zorder=5)
            if filled:
                ax.scatter(pt[0], pt[1], c='red',  marker='o', edgecolors='black', **kw)
            else:
                ax.scatter(pt[0], pt[1], c='blue', marker='x', **kw)
        ax.set_title('XOR Decision Boundary', fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.grid(True, alpha=0.3)

        # 은닉층 활성화
        ax = axes[2]
        ha = mlp.a1  # (4, hidden_size)
        n_h = ha.shape[1]
        im = ax.imshow(ha.T, cmap='viridis', aspect='auto')
        ax.set_yticks(range(n_h)); ax.set_yticklabels([f'H{i+1}' for i in range(n_h)])
        ax.set_xticks(range(4)); ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
        ax.set_title('Hidden Layer Activations', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Activation')
        for i in range(n_h):
            for j in range(4):
                ax.text(j, i, f'{ha[j, i]:.2f}', ha='center', va='center',
                        color='white', fontsize=7, fontweight='bold')

        fig.tight_layout()
        self.canvas.draw()


# ─────────────────────── Tab 5: Universal Approximation ───────────────────────
class Tab5_Universal(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self._thread = None
        self._worker = None

        # ── 좌측 ──
        left = QWidget(); left.setMaximumWidth(300)
        lv = QVBoxLayout(left)

        ctrl_box = QGroupBox("Universal Approximation 실험")
        cv = QVBoxLayout()
        cv.addWidget(QLabel("근사 대상 함수:"))
        self.combo = QComboBox()
        self.combo.addItems(['Sine Wave', 'Step Function', 'Complex Function'])
        cv.addWidget(self.combo)
        ctrl_box.setLayout(cv)
        lv.addWidget(ctrl_box)

        self.run_btn = QPushButton("▶ 근사 시작")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        self.progress_label = QLabel("함수를 선택하고 '근사 시작'을 누르세요.")
        self.progress_label.setWordWrap(True)
        lv.addWidget(self.progress_label)

        theory = QLabel(
            "<b>Universal Approximation Theorem</b><br>"
            "(Cybenko, 1989)<br><br>"
            "하나의 은닉층을 가진 신경망은<br>"
            "충분히 많은 뉴런이 있다면<br>"
            "<b>어떤 연속 함수도</b> 임의의<br>"
            "정확도로 근사 가능.<br><br>"
            "<b>비교 뉴런 수:</b><br>"
            "• 3개 → 거친 근사<br>"
            "• 10개 → 대략적 형태<br>"
            "• 50개 → 거의 완벽!"
        )
        theory.setWordWrap(True)
        lv.addWidget(theory)
        lv.addStretch()
        layout.addWidget(left)

        # ── 우측 ──
        self.canvas = MplCanvas(figsize=(11, 4))
        layout.addWidget(self.canvas)

    def _run(self):
        if self._thread and self._thread.isRunning():
            return
        self.run_btn.setEnabled(False)
        self.progress_label.setText("학습 중...")

        self._thread = QThread()
        self._worker = UAWorker(self.combo.currentText())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda msg: self.progress_label.setText(msg))
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_finished(self, results):
        self.run_btn.setEnabled(True)
        msgs = []
        for n, _, _, _, mse in results:
            msgs.append(f"{n}개 뉴런: MSE={mse:.4f}")
        self.progress_label.setText("완료!\n" + "\n".join(msgs))

        fig = self.canvas.figure
        fig.clear()
        axes = fig.subplots(1, 3)
        func_name = self.combo.currentText()

        for ax, (n, x_te, y_true, y_pred, mse) in zip(axes, results):
            ax.plot(x_te, y_true, 'b-', lw=2, label='True Function', alpha=0.7)
            ax.plot(x_te, y_pred, 'r--', lw=2, label=f'NN ({n} neurons)')
            ax.set_title(f'{func_name}\n{n} Neurons  MSE={mse:.4f}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_xlabel('x'); ax.set_ylabel('y')

        fig.tight_layout()
        self.canvas.draw()


# ─────────────────────── Main Window ───────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3: 신경망 기초 인터랙티브 학습 도구")
        self.resize(1280, 700)

        tabs = QTabWidget()
        tabs.addTab(Tab1_Perceptron(),  "Lab 1: Perceptron")
        tabs.addTab(Tab2_Activation(),  "Lab 2: Activation Functions")
        tabs.addTab(Tab3_ForwardProp(), "Lab 3: Forward Propagation")
        tabs.addTab(Tab4_MLP(),         "Lab 4: MLP (Backprop)")
        tabs.addTab(Tab5_Universal(),   "Lab 5: Universal Approximation")
        self.setCentralWidget(tabs)


# ─────────────────────── Entry Point ───────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
