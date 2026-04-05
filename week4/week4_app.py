"""
Week 4: 물리 데이터로 학습하기 - Neural Networks
PySide6 + Matplotlib + TensorFlow/Keras

PRD/TRD 기반 구현:
- 4개 탭 (Lab 1~4) 통합 GUI
- TF 학습은 QThread 백그라운드 실행
- Keras 콜백 → Qt Signal로 실시간 진행률 표시
- Matplotlib FigureCanvasQTAgg 임베드
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QLabel, QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────

class ProgressCallback(keras.callbacks.Callback):
    """Keras epoch 종료 → Qt Signal emit 브릿지"""
    def __init__(self, total_epochs: int, emit_fn):
        super().__init__()
        self.total_epochs = total_epochs
        self.emit_fn = emit_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.emit_fn(
            epoch + 1,
            self.total_epochs,
            float(logs.get('loss', 0)),
            float(logs.get('val_loss', logs.get('loss', 0)))
        )


class BaseTab(QWidget):
    """4개 탭의 공통 레이아웃: 좌측 컨트롤 + 우측 Matplotlib"""

    def __init__(self, title: str, description: str,
                 fig_rows: int = 2, fig_cols: int = 3, parent=None):
        super().__init__(parent)
        self.worker = None
        self._build_ui(title, description, fig_rows, fig_cols)

    def _build_ui(self, title, description, fig_rows, fig_cols):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── 좌측 패널 ──────────────────────────
        left = QFrame()
        left.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        title_lbl = QLabel(title)
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title_lbl.setFont(f)
        title_lbl.setWordWrap(True)
        lv.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #555; font-size: 11px;")
        lv.addWidget(desc_lbl)

        lv.addSpacing(12)

        self.start_btn = QPushButton("▶  학습 시작")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover  { background-color: #1565C0; }
            QPushButton:disabled { background-color: #BDBDBD; color: #fff; }
        """)
        self.start_btn.clicked.connect(self._on_start)
        lv.addWidget(self.start_btn)

        lv.addSpacing(6)

        lv.addWidget(QLabel("진행률:"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        lv.addWidget(self.progress)

        lv.addSpacing(6)

        lv.addWidget(QLabel("학습 로그:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size: 11px; background: #fafafa;")
        lv.addWidget(self.log_box)

        lv.addStretch()
        root.addWidget(left)

        # ── 우측 패널 (Matplotlib) ──────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(2)

        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas)
        root.addWidget(right)

        self._draw_placeholder(fig_rows, fig_cols)

    def _draw_placeholder(self, rows, cols):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "▶  학습 시작 버튼을 눌러\n결과를 확인하세요",
                ha='center', va='center', fontsize=16, color='#9E9E9E',
                transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()

    def log(self, msg: str):
        self.log_box.append(msg)

    def _on_start(self):
        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()
        self.log("학습 준비 중...")
        self._start_worker()

    def _start_worker(self):
        raise NotImplementedError

    def _on_progress(self, epoch: int, total: int, loss: float, val_loss: float):
        pct = int(epoch / total * 100)
        self.progress.setValue(pct)
        step = max(1, total // 10)
        if epoch % step == 0 or epoch == total:
            self.log(f"  Epoch {epoch:4d}/{total}  loss={loss:.5f}  val_loss={val_loss:.5f}")

    def _on_error(self, msg: str):
        self.log(f"[오류] {msg}")
        self.start_btn.setEnabled(True)

    def _on_finished(self, results: dict):
        self.progress.setValue(100)
        self.log("\n학습 완료! 결과 그래프를 렌더링 중...")
        self._render(results)
        self.canvas.draw()
        self.log("완료.")
        self.start_btn.setEnabled(True)

    def _render(self, results: dict):
        raise NotImplementedError


# ─────────────────────────────────────────────
# Lab 1: 1D 함수 근사
# ─────────────────────────────────────────────

class Lab1Worker(QThread):
    progress = Signal(int, int, float, float)
    finished = Signal(dict)
    error = Signal(str)

    EPOCHS = 1000

    def run(self):
        try:
            results = {}
            x_test = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)

            funcs = {
                'sin(x)': (
                    np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1),
                    lambda x: np.sin(x)
                ),
                'cos(x)+0.5sin(2x)': (
                    np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1),
                    lambda x: np.cos(x) + 0.5 * np.sin(2 * x)
                ),
                'x·sin(x)': (
                    np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1),
                    lambda x: x * np.sin(x)
                ),
            }

            total_phases = len(funcs)
            for phase_idx, (fname, (x_tr, fn)) in enumerate(funcs.items()):
                y_tr = fn(x_tr)
                y_test = fn(x_test)

                # 데이터 셔플
                perm = np.random.permutation(len(x_tr))
                x_tr, y_tr = x_tr[perm], y_tr[perm]

                model = keras.Sequential([
                    keras.layers.Input(shape=(1,)),
                    keras.layers.Dense(128, activation='tanh'),
                    keras.layers.Dense(128, activation='tanh'),
                    keras.layers.Dense(64, activation='tanh'),
                    keras.layers.Dense(1, activation='linear'),
                ])
                model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mse')

                cb_progress = ProgressCallback(
                    self.EPOCHS,
                    lambda ep, tot, loss, vl, p=phase_idx, n=total_phases:
                        self.progress.emit(p * self.EPOCHS + ep, n * self.EPOCHS, loss, vl)
                )
                cb_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', factor=0.8, patience=100, min_lr=1e-5, verbose=0
                )
                cb_es = keras.callbacks.EarlyStopping(
                    monitor='loss', patience=200, restore_best_weights=True, verbose=0
                )

                hist = model.fit(
                    x_tr, y_tr,
                    epochs=self.EPOCHS,
                    batch_size=32,
                    verbose=0,
                    callbacks=[cb_progress, cb_lr, cb_es]
                )

                y_pred = model.predict(x_test, verbose=0)
                mse = float(np.mean((y_pred - y_test) ** 2))

                results[fname] = {
                    'x_test': x_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'x_train': x_tr,
                    'y_train': y_tr,
                    'loss': hist.history['loss'],
                    'mse': mse,
                }

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class FunctionApproxTab(BaseTab):
    DESC = ("Universal Approximation Theorem:\n"
            "충분히 넓은 NN은 임의의 연속 함수를 근사할 수 있습니다.\n\n"
            "3가지 함수를 [128,128,64] 네트워크로 학습합니다.")

    def __init__(self, parent=None):
        super().__init__("Lab 1: 1D 함수 근사", self.DESC, 2, 3, parent)

    def _start_worker(self):
        self.worker = Lab1Worker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.log(f"3개 함수 학습 (각 최대 {Lab1Worker.EPOCHS} epoch, EarlyStopping 적용)")
        self.worker.start()

    def _render(self, results: dict):
        self.fig.clear()
        names = list(results.keys())
        for i, fname in enumerate(names):
            r = results[fname]
            # 상단: 함수 근사
            ax1 = self.fig.add_subplot(2, 3, i + 1)
            ax1.plot(r['x_test'], r['y_test'], 'b-', lw=2, label='실제', alpha=0.7)
            ax1.plot(r['x_test'], r['y_pred'], 'r--', lw=2, label='NN 예측')
            ax1.scatter(r['x_train'][::10], r['y_train'][::10],
                        c='k', s=12, alpha=0.3)
            ax1.set_title(f"{fname}\nMSE={r['mse']:.6f}", fontsize=10, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 하단: 학습 loss 곡선
            ax2 = self.fig.add_subplot(2, 3, i + 4)
            ax2.plot(r['loss'], 'g-', lw=1.5)
            ax2.set_xlabel('Epoch', fontsize=9)
            ax2.set_ylabel('Loss (MSE)', fontsize=9)
            ax2.set_title('학습 Loss', fontsize=10, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        self.fig.suptitle('Lab 1: 1D 함수 근사 (TensorFlow/Keras)',
                          fontsize=13, fontweight='bold')


# ─────────────────────────────────────────────
# Lab 2: 포물선 운동
# ─────────────────────────────────────────────

class Lab2Worker(QThread):
    progress = Signal(int, int, float, float)
    finished = Signal(dict)
    error = Signal(str)

    EPOCHS = 100
    G = 9.81

    def run(self):
        try:
            # 데이터 생성
            n = 2000
            v0 = np.random.uniform(10, 50, n)
            theta = np.random.uniform(20, 70, n)
            theta_rad = np.deg2rad(theta)
            t_max = 2 * v0 * np.sin(theta_rad) / self.G
            t = np.random.uniform(0, t_max * 0.9, n)
            x = v0 * np.cos(theta_rad) * t + np.random.normal(0, 0.5, n)
            y = v0 * np.sin(theta_rad) * t - 0.5 * self.G * t**2 + np.random.normal(0, 0.5, n)
            valid = y >= 0
            X = np.column_stack([v0[valid], theta[valid], t[valid]])
            Y = np.column_stack([x[valid], y[valid]])

            model = keras.Sequential([
                keras.layers.Input(shape=(3,)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(2, activation='linear'),
            ])
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

            cb = ProgressCallback(self.EPOCHS, self.progress.emit)
            hist = model.fit(X, Y, validation_split=0.2, epochs=self.EPOCHS,
                             batch_size=32, verbose=0, callbacks=[cb])

            # 3가지 조건 예측
            conditions = [(20, 30), (30, 45), (40, 60)]
            preds = []
            for v, ang in conditions:
                ang_r = np.deg2rad(ang)
                t_end = 2 * v * np.sin(ang_r) / self.G
                ts = np.linspace(0, t_end, 60)
                X_in = np.column_stack([np.full(60, v), np.full(60, ang), ts])
                pred = model.predict(X_in, verbose=0)
                x_true = v * np.cos(ang_r) * ts
                y_true = v * np.sin(ang_r) * ts - 0.5 * self.G * ts**2
                preds.append({
                    'label': f'v₀={v} m/s, θ={ang}°',
                    'x_true': x_true, 'y_true': y_true,
                    'x_pred': pred[:, 0], 'y_pred': pred[:, 1],
                    'mse': float(np.mean((pred[:, 0]-x_true)**2 + (pred[:, 1]-y_true)**2))
                })

            self.finished.emit({'history': hist.history, 'conditions': preds})
        except Exception as e:
            self.error.emit(str(e))


class ProjectileTab(BaseTab):
    DESC = ("물리 법칙:\n"
            "x(t) = v₀·cos(θ)·t\n"
            "y(t) = v₀·sin(θ)·t - ½g·t²\n\n"
            "NN이 이 법칙을 데이터로부터 학습합니다.")

    def __init__(self, parent=None):
        super().__init__("Lab 2: 포물선 운동 회귀", self.DESC, 2, 3, parent)

    def _start_worker(self):
        self.worker = Lab2Worker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.log(f"포물선 운동 학습 ({Lab2Worker.EPOCHS} epochs)")
        self.worker.start()

    def _render(self, results: dict):
        self.fig.clear()
        history = results['history']
        conditions = results['conditions']

        # 상단: 3개 조건 궤적
        for i, c in enumerate(conditions):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.plot(c['x_true'], c['y_true'], 'b-', lw=2.5, label='실제 궤적', alpha=0.7)
            ax.plot(c['x_pred'], c['y_pred'], 'r--', lw=2, label='NN 예측')
            ax.set_xlabel('x (m)', fontsize=9)
            ax.set_ylabel('y (m)', fontsize=9)
            ax.set_title(f"{c['label']}\nMSE={c['mse']:.3f}", fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

        # 하단 좌: Loss 곡선
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax4.plot(history['loss'], 'b-', lw=2, label='Train')
        ax4.plot(history['val_loss'], 'r--', lw=2, label='Validation')
        ax4.set_xlabel('Epoch', fontsize=9)
        ax4.set_ylabel('MSE', fontsize=9)
        ax4.set_title('학습 곡선', fontsize=10, fontweight='bold')
        ax4.set_yscale('log')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 하단 중: MAE 곡선
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax5.plot(history['mae'], 'b-', lw=2, label='Train MAE')
        ax5.plot(history['val_mae'], 'r--', lw=2, label='Val MAE')
        ax5.set_xlabel('Epoch', fontsize=9)
        ax5.set_ylabel('MAE', fontsize=9)
        ax5.set_title('MAE 곡선', fontsize=10, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 하단 우: 최대 높이/거리 비교
        ax6 = self.fig.add_subplot(2, 3, 6)
        labels = [c['label'].replace(', ', '\n') for c in conditions]
        h_true = [max(c['y_true']) for c in conditions]
        h_pred = [max(c['y_pred']) for c in conditions]
        x_pos = np.arange(len(labels))
        w = 0.35
        ax6.bar(x_pos - w/2, h_true, w, label='실제 최대 높이', color='#1976D2', alpha=0.8)
        ax6.bar(x_pos + w/2, h_pred, w, label='NN 예측 최대 높이', color='#E53935', alpha=0.8)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(labels, fontsize=8)
        ax6.set_ylabel('최대 높이 (m)', fontsize=9)
        ax6.set_title('최대 높이 비교', fontsize=10, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

        self.fig.suptitle('Lab 2: 포물선 운동 회귀 (TensorFlow/Keras)',
                          fontsize=13, fontweight='bold')


# ─────────────────────────────────────────────
# Lab 3: 과적합 vs 과소적합
# ─────────────────────────────────────────────

class Lab3Worker(QThread):
    progress = Signal(int, int, float, float)
    finished = Signal(dict)
    error = Signal(str)

    EPOCHS = 200

    def _true_fn(self, x):
        return np.sin(2 * x) + 0.5 * x

    def run(self):
        try:
            np.random.seed(42)
            x_tr = np.random.uniform(-2, 2, 100).reshape(-1, 1)
            y_tr = self._true_fn(x_tr) + np.random.normal(0, 0.3, (100, 1))
            x_val = np.random.uniform(-2, 2, 50).reshape(-1, 1)
            y_val = self._true_fn(x_val) + np.random.normal(0, 0.3, (50, 1))
            x_test = np.linspace(-2, 2, 200).reshape(-1, 1)
            y_test = self._true_fn(x_test)

            model_defs = {
                '과소적합\n[4]': keras.Sequential([
                    keras.layers.Input(shape=(1,)),
                    keras.layers.Dense(4, activation='relu'),
                    keras.layers.Dense(1),
                ]),
                '적절한 학습\n[32,16]+Dropout': keras.Sequential([
                    keras.layers.Input(shape=(1,)),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(16, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(1),
                ]),
                '과적합\n[256,128,64,32]': keras.Sequential([
                    keras.layers.Input(shape=(1,)),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(1),
                ]),
            }

            total = len(model_defs) * self.EPOCHS
            results = {
                'x_train': x_tr, 'y_train': y_tr,
                'x_test': x_test, 'y_test': y_test,
                'models': {}
            }

            for m_idx, (mname, model) in enumerate(model_defs.items()):
                model.compile(optimizer=keras.optimizers.Adam(0.001),
                               loss='mse', metrics=['mae'])

                offset = m_idx * self.EPOCHS
                cb = ProgressCallback(
                    total,
                    lambda ep, tot, loss, vl, off=offset:
                        self.progress.emit(off + ep, tot, loss, vl)
                )
                hist = model.fit(
                    x_tr, y_tr,
                    validation_data=(x_val, y_val),
                    epochs=self.EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[cb]
                )
                y_pred = model.predict(x_test, verbose=0)
                results['models'][mname] = {
                    'y_pred': y_pred,
                    'train_loss': hist.history['loss'],
                    'val_loss': hist.history['val_loss'],
                    'final_train': hist.history['loss'][-1],
                    'final_val': hist.history['val_loss'][-1],
                    'test_mse': float(np.mean((y_pred - y_test) ** 2)),
                }

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class OverfittingTab(BaseTab):
    DESC = ("세 가지 모델 비교:\n"
            "• 과소적합 [4] - 너무 단순\n"
            "• 적절한 학습 [32,16]+Dropout\n"
            "• 과적합 [256,128,64,32] - 너무 복잡\n\n"
            "함수: y = sin(2x) + 0.5x + noise")

    def __init__(self, parent=None):
        super().__init__("Lab 3: 과적합 vs 과소적합", self.DESC, 2, 3, parent)

    def _start_worker(self):
        self.worker = Lab3Worker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.log(f"3개 모델 학습 (각 {Lab3Worker.EPOCHS} epochs)")
        self.worker.start()

    def _render(self, results: dict):
        self.fig.clear()
        colors = {'과소적합\n[4]': '#1976D2',
                  '적절한 학습\n[32,16]+Dropout': '#388E3C',
                  '과적합\n[256,128,64,32]': '#D32F2F'}

        x_tr = results['x_train']
        y_tr = results['y_train']
        x_test = results['x_test']
        y_test = results['y_test']
        models_data = results['models']
        names = list(models_data.keys())

        # 행 1: 각 모델의 예측
        for i, mname in enumerate(names):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.scatter(x_tr, y_tr, alpha=0.4, s=20, color='gray', label='학습 데이터')
            ax.plot(x_test, y_test, 'k-', lw=2.5, label='실제 함수', alpha=0.7)
            ax.plot(x_test, models_data[mname]['y_pred'],
                    color=colors[mname], lw=2, ls='--', label='NN 예측')
            short_name = mname.replace('\n', ' ')
            ax.set_title(f"{short_name}\nTest MSE={models_data[mname]['test_mse']:.4f}",
                         fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 행 2: 학습 곡선
        for i, mname in enumerate(names):
            ax = self.fig.add_subplot(2, 3, i + 4)
            d = models_data[mname]
            ep = range(1, len(d['train_loss']) + 1)
            ax.plot(ep, d['train_loss'], color=colors[mname], lw=2, label='Train')
            ax.plot(ep, d['val_loss'], color=colors[mname], lw=2, ls='--', label='Val')
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('MSE', fontsize=9)
            ax.set_title(f"학습 곡선\nTrain={d['final_train']:.4f} Val={d['final_val']:.4f}",
                         fontsize=9, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        self.fig.suptitle('Lab 3: 과적합 vs 과소적합 (TensorFlow/Keras)',
                          fontsize=13, fontweight='bold')


# ─────────────────────────────────────────────
# Lab 4: 진자 주기 예측
# ─────────────────────────────────────────────

class Lab4Worker(QThread):
    progress = Signal(int, int, float, float)
    finished = Signal(dict)
    error = Signal(str)

    EPOCHS = 100
    G = 9.81

    def _true_period(self, L, theta_deg):
        t0 = 2 * np.pi * np.sqrt(L / self.G)
        r = np.deg2rad(theta_deg)
        return t0 * (1 + r**2 / 16 + 11 * r**4 / 3072)

    def _rk4(self, L, theta0_deg, t_max, dt=0.02):
        theta = np.deg2rad(theta0_deg)
        omega = 0.0
        ts, th, om = [], [], []
        t = 0.0
        while t <= t_max:
            ts.append(t)
            th.append(np.rad2deg(theta))
            om.append(omega)
            k1t, k1o = omega, -(self.G / L) * np.sin(theta)
            k2t, k2o = omega + 0.5*dt*k1o, -(self.G/L)*np.sin(theta+0.5*dt*k1t)
            k3t, k3o = omega + 0.5*dt*k2o, -(self.G/L)*np.sin(theta+0.5*dt*k2t)
            k4t, k4o = omega + dt*k3o,      -(self.G/L)*np.sin(theta+dt*k3t)
            theta += dt / 6 * (k1t + 2*k2t + 2*k3t + k4t)
            omega += dt / 6 * (k1o + 2*k2o + 2*k3o + k4o)
            t += dt
        return np.array(ts), np.array(th), np.array(om)

    def run(self):
        try:
            n = 2000
            L_data = np.random.uniform(0.5, 3.0, n)
            th_data = np.random.uniform(5, 80, n)
            T_data = self._true_period(L_data, th_data)
            T_data *= (1 + np.random.normal(0, 0.01, n))

            X = np.column_stack([L_data, th_data])
            Y = T_data.reshape(-1, 1)

            model = keras.Sequential([
                keras.layers.Input(shape=(2,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(1, activation='linear'),
            ])
            model.compile(optimizer=keras.optimizers.Adam(0.001),
                          loss='mse', metrics=['mae', 'mape'])

            cb = ProgressCallback(self.EPOCHS, self.progress.emit)
            hist = model.fit(X, Y, validation_split=0.2, epochs=self.EPOCHS,
                             batch_size=32, verbose=0, callbacks=[cb])

            # 3가지 길이 예측
            lengths = [0.5, 1.0, 2.0]
            period_results = []
            for L in lengths:
                angles = np.linspace(5, 80, 50)
                X_in = np.column_stack([np.full(50, L), angles])
                T_pred = model.predict(X_in, verbose=0).flatten()
                T_true = self._true_period(L, angles)
                mape = float(np.mean(np.abs((T_pred - T_true) / T_true)) * 100)
                period_results.append({'L': L, 'angles': angles,
                                       'T_true': T_true, 'T_pred': T_pred, 'mape': mape})

            # RK4 시뮬레이션
            sims = []
            for L, th0 in [(1.0, 15), (1.0, 45), (1.0, 75)]:
                T_true = self._true_period(L, th0)
                X_in = np.array([[L, th0]])
                T_pred_val = float(model.predict(X_in, verbose=0)[0, 0])
                t_arr, th_arr, om_arr = self._rk4(L, th0, T_true * 3)
                sims.append({'label': f'θ₀={th0}°', 'L': L, 'theta0': th0,
                             't': t_arr, 'theta': th_arr, 'omega': om_arr,
                             'T_true': T_true, 'T_pred': T_pred_val})

            self.finished.emit({
                'history': hist.history,
                'period_results': period_results,
                'sims': sims,
            })
        except Exception as e:
            self.error.emit(str(e))


class PendulumTab(BaseTab):
    DESC = ("물리 법칙:\n"
            "T = 2π√(L/g)\n"
            "(작은 각도 근사)\n\n"
            "큰 각도: 타원 적분 근사\n"
            "운동 방정식 수치 적분 (RK4)")

    def __init__(self, parent=None):
        super().__init__("Lab 4: 진자 주기 예측", self.DESC, 2, 3, parent)

    def _start_worker(self):
        self.worker = Lab4Worker()
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.log(f"진자 주기 예측 학습 ({Lab4Worker.EPOCHS} epochs)")
        self.worker.start()

    def _render(self, results: dict):
        self.fig.clear()
        history = results['history']
        period_results = results['period_results']
        sims = results['sims']

        # 행 1: 주기 예측 (3가지 길이)
        colors_L = ['#1976D2', '#388E3C', '#F57C00']
        for i, pr in enumerate(period_results):
            ax = self.fig.add_subplot(2, 3, i + 1)
            ax.plot(pr['angles'], pr['T_true'], '-', lw=2.5,
                    color=colors_L[i], label='실제 주기', alpha=0.7)
            ax.plot(pr['angles'], pr['T_pred'], 'r--', lw=2, label='NN 예측')
            ax.set_xlabel('초기 각도 (°)', fontsize=9)
            ax.set_ylabel('주기 T (s)', fontsize=9)
            ax.set_title(f"L={pr['L']} m\nMAPE={pr['mape']:.2f}%",
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 행 2 좌: 학습 곡선
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax4.plot(history['loss'], 'b-', lw=2, label='Train')
        ax4.plot(history['val_loss'], 'r--', lw=2, label='Validation')
        ax4.set_xlabel('Epoch', fontsize=9)
        ax4.set_ylabel('MSE', fontsize=9)
        ax4.set_title('학습 곡선', fontsize=10, fontweight='bold')
        ax4.set_yscale('log')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 행 2 중: 각도별 RK4 시뮬레이션 (θ vs t)
        ax5 = self.fig.add_subplot(2, 3, 5)
        sim_colors = ['#1976D2', '#388E3C', '#D32F2F']
        for sim, sc in zip(sims, sim_colors):
            ax5.plot(sim['t'], sim['theta'], color=sc, lw=1.5,
                     label=f"{sim['label']} T_pred={sim['T_pred']:.3f}s")
        ax5.set_xlabel('시간 (s)', fontsize=9)
        ax5.set_ylabel('각도 (°)', fontsize=9)
        ax5.set_title('RK4 시뮬레이션 (L=1m)', fontsize=10, fontweight='bold')
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)

        # 행 2 우: 위상 공간 (θ vs ω)
        ax6 = self.fig.add_subplot(2, 3, 6)
        for sim, sc in zip(sims, sim_colors):
            ax6.plot(sim['theta'], sim['omega'], color=sc, lw=1.5,
                     alpha=0.8, label=sim['label'])
            ax6.plot(sim['theta'][0], sim['omega'][0], 'o', color=sc, ms=7)
        ax6.set_xlabel('각도 (°)', fontsize=9)
        ax6.set_ylabel('각속도 (°/s)', fontsize=9)
        ax6.set_title('위상 공간 (Phase Space)', fontsize=10, fontweight='bold')
        ax6.legend(fontsize=7)
        ax6.grid(True, alpha=0.3)

        self.fig.suptitle('Lab 4: 진자 주기 예측 + RK4 시뮬레이션',
                          fontsize=13, fontweight='bold')


# ─────────────────────────────────────────────
# 메인 윈도우
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 4: 물리 데이터로 학습하기 — Neural Networks")
        self.setMinimumSize(1280, 800)

        tabs = QTabWidget()
        tabs.addTab(FunctionApproxTab(), "Lab 1: 1D 함수 근사")
        tabs.addTab(ProjectileTab(),     "Lab 2: 포물선 운동")
        tabs.addTab(OverfittingTab(),    "Lab 3: 과적합/과소적합")
        tabs.addTab(PendulumTab(),       "Lab 4: 진자 주기 예측")

        tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 18px;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #1976D2;
                color: white;
                border-radius: 4px 4px 0 0;
            }
        """)

        self.setCentralWidget(tabs)

        # 상태바
        self.statusBar().showMessage(
            "Week 4 · 물리 데이터 Neural Networks  |  각 탭에서 [학습 시작]을 클릭하세요"
        )


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
