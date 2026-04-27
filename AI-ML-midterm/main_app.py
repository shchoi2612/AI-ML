"""
main_app.py — Can ML Beat the Market? 인터랙티브 GUI
전산물리 중간 프로젝트 — 부산대학교 물리학과

PyTorch MLP + K-Means Regime + Markowitz + Kelly
PySide6 GUI: 4탭 + 실시간 학습 진행률 + Matplotlib 임베드

실행:  uv run main_app.py
"""

import sys
import traceback
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.colors as mcolors
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QLabel, QTextEdit, QFrame, QSizePolicy, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSpinBox, QDoubleSpinBox, QFormLayout,
    QComboBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QColor


# ══════════════════════════════════════════════════════════
#  공통 헬퍼
# ══════════════════════════════════════════════════════════

REGIME_COLORS = {"Bull": "#2e7d32", "Sideways": "#f57c00", "Bear": "#c62828"}
STRAT_COLORS  = {
    "Buy&Hold":  "#1565C0",
    "Markowitz": "#00838f",
    "Kelly_MLP": "#e65100",
    "Regime":    "#6a1b9a",
}


def _btn(text: str, color: str = "#1976D2") -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(40)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{color}; color:white; border:none;
            border-radius:6px; font-size:13px; font-weight:bold;
        }}
        QPushButton:hover   {{ background:{_darken(color)}; }}
        QPushButton:disabled{{ background:#BDBDBD; color:#fff; }}
    """)
    return b


def _darken(hex_color: str, factor: float = 0.85) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _label(text: str, bold: bool = False, size: int = 11) -> QLabel:
    lbl = QLabel(text)
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    lbl.setFont(f)
    return lbl


def _section(title: str) -> QGroupBox:
    gb = QGroupBox(title)
    gb.setStyleSheet("QGroupBox { font-weight:bold; }")
    return gb


# ══════════════════════════════════════════════════════════
#  QThread Workers
# ══════════════════════════════════════════════════════════

class DataWorker(QObject):
    """Tab 1: 데이터 다운로드 + 피처 생성"""
    log_msg   = Signal(str)
    finished  = Signal(dict)
    error     = Signal(str)

    def __init__(self, start: str, end: str):
        super().__init__()
        self.start = start
        self.end = end

    def run(self):
        try:
            from data import build_dataset
            from features import build_features
            self.log_msg.emit("📥 Downloading data from yfinance...")
            ds = build_dataset(self.start, self.end)
            self.log_msg.emit(f"  ✔ prices: {ds['prices'].shape}")
            self.log_msg.emit("🔧 Engineering features...")
            feat = build_features(ds["prices"], ds["returns"])
            self.log_msg.emit(f"  ✔ features: {feat.shape}  ({', '.join(feat.columns[:4])} ...)")
            self.log_msg.emit("✅ Done!")
            self.finished.emit({"ds": ds, "features": feat})
        except Exception as e:
            self.error.emit(traceback.format_exc())


class RegimeWorker(QObject):
    """Tab 2: K-Means Regime Detection"""
    log_msg  = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, ds: dict, features: pd.DataFrame):
        super().__init__()
        self.ds = ds
        self.features = features

    def run(self):
        try:
            from regime import run_regime_detection
            self.log_msg.emit("🔍 Training K-Means (n=3)...")
            regimes, km, scaler, sil = run_regime_detection(self.features)
            counts = regimes.value_counts().to_dict()
            self.log_msg.emit(f"  Silhouette Score: {sil:.4f}")
            self.log_msg.emit(f"  Bull:     {counts.get('Bull', 0)} days")
            self.log_msg.emit(f"  Sideways: {counts.get('Sideways', 0)} days")
            self.log_msg.emit(f"  Bear:     {counts.get('Bear', 0)} days")
            self.log_msg.emit("✅ Regime detection complete!")
            self.finished.emit({"regimes": regimes, "km": km, "scaler": scaler, "sil": sil})
        except Exception as e:
            self.error.emit(traceback.format_exc())


class TrainWorker(QObject):
    """Tab 3: PyTorch MLP 학습 (실시간 epoch 신호)"""
    epoch_update = Signal(int, int, float, float, float, float)  # ep,tot,tl,vl,ta,va
    log_msg  = Signal(str)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, ds: dict, features: pd.DataFrame, epochs: int, lr: float):
        super().__init__()
        self.ds = ds
        self.features = features
        self.epochs = epochs
        self.lr = lr

    def run(self):
        try:
            from model_torch import prepare_sequences, train_model, predict_proba, split_train_test
            sp_ret = self.ds["returns"]["sp500_ret"].reindex(self.features.index)
            self.log_msg.emit("🔄 Preparing sequences (lookback=60)...")
            X, y, dates = prepare_sequences(self.features, sp_ret)
            X_tr, y_tr, X_te, y_te, d_tr, d_te = split_train_test(X, y, dates)
            self.log_msg.emit(f"  Train: {len(X_tr)} / Test: {len(X_te)} samples")
            self.log_msg.emit(f"  Up ratio — Train: {y_tr.mean():.2%}  Test: {y_te.mean():.2%}")
            self.log_msg.emit(f"\n🧠 Starting PyTorch MLP training (epochs={self.epochs})")

            def cb(ep, tot, tl, vl, ta, va):
                self.epoch_update.emit(ep, tot, tl, vl, ta, va)
                if ep % max(1, tot // 10) == 0 or ep == tot:
                    self.log_msg.emit(
                        f"  Epoch {ep:4d}/{tot}  "
                        f"loss={tl:.4f}  val_loss={vl:.4f}  "
                        f"acc={ta:.3f}  val_acc={va:.3f}"
                    )

            model, scaler_sc, history = train_model(
                X_tr, y_tr, epochs=self.epochs, lr=self.lr, epoch_callback=cb
            )
            probs_te = predict_proba(model, scaler_sc, X_te)
            acc = float(((probs_te >= 0.5).astype(float) == y_te).mean())
            self.log_msg.emit(f"\n📊 Test Accuracy: {acc:.4f}  ({'✅ >55%' if acc > 0.55 else '⚠️ ≤55%'})")
            self.finished.emit({
                "model": model, "scaler": scaler_sc, "history": history,
                "X_te": X_te, "y_te": y_te, "probs_te": probs_te,
                "d_te": d_te, "acc": acc,
            })
        except Exception as e:
            self.error.emit(traceback.format_exc())


class BacktestWorker(QObject):
    """Tab 4: 전체 백테스트"""
    log_msg  = Signal(str)
    progress = Signal(int)
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, ds: dict, features: pd.DataFrame,
                 model, scaler_sc, regimes: pd.Series):
        super().__init__()
        self.ds = ds
        self.features = features
        self.model = model
        self.scaler_sc = scaler_sc
        self.regimes = regimes

    def run(self):
        try:
            from backtest import buy_and_hold, performance_table
            from strategy_markowitz import backtest_markowitz
            from strategy_kelly import kelly_positions, backtest_kelly
            from model_torch import prepare_sequences, predict_proba

            self.log_msg.emit("📈 Calculating Buy & Hold...")
            self.progress.emit(10)
            bh = buy_and_hold(self.ds["prices"])

            self.log_msg.emit("📊 Calculating Markowitz portfolio...")
            self.progress.emit(30)
            mw = backtest_markowitz(self.ds["prices"])

            self.log_msg.emit("🤖 Applying Kelly+MLP strategy...")
            self.progress.emit(50)
            sp_ret = self.ds["returns"]["sp500_ret"].reindex(self.features.index)
            X, y, dates = prepare_sequences(self.features, sp_ret)
            probs = predict_proba(self.model, self.scaler_sc, X)
            pos = kelly_positions(probs)
            kelly_ret = backtest_kelly(sp_ret, pos, dates)

            self.log_msg.emit("🌐 Building regime-based strategy...")
            self.progress.emit(70)
            regime_ret = _build_regime_strategy(kelly_ret, mw, self.regimes)

            self.log_msg.emit("📋 Calculating performance metrics...")
            self.progress.emit(90)
            strategies = {
                "Buy&Hold":  bh,
                "Markowitz": mw,
                "Kelly_MLP": kelly_ret,
                "Regime":    regime_ret,
            }
            tbl = performance_table(strategies)
            self.progress.emit(100)
            self.log_msg.emit("\n" + tbl.to_string())
            self.log_msg.emit("\n✅ Backtest complete!")
            self.finished.emit({"strategies": strategies, "table": tbl})
        except Exception as e:
            self.error.emit(traceback.format_exc())


def _build_regime_strategy(kelly_ret: pd.Series, mw_ret: pd.Series,
                            regimes: pd.Series) -> pd.Series:
    """Bull/Sideways → Markowitz, Bear → Kelly+MLP"""
    idx = kelly_ret.index.union(mw_ret.index)
    result = pd.Series(index=idx, dtype=float)
    for date in idx:
        regime = regimes.get(date, "Bull")
        if regime == "Bear":
            result[date] = kelly_ret.get(date, 0.0)
        else:
            result[date] = mw_ret.get(date, 0.0)
    return result.dropna().rename("Regime")


# ══════════════════════════════════════════════════════════
#  Tab 1: 데이터 & 피처
# ══════════════════════════════════════════════════════════

class DataTab(QWidget):
    data_ready = Signal(dict)   # ds + features → 다른 탭에 전달

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        self._feat = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── 좌측 컨트롤 ──────────────────────────
        left = QFrame()
        left.setFrameStyle(QFrame.Shape.StyledPanel)
        left.setFixedWidth(230)
        lv = QVBoxLayout(left)

        lv.addWidget(_label("Data & Features", bold=True, size=13))
        lv.addWidget(_label("S&P500, NASDAQ, SOX, VIX\nvia yfinance (2015-2024)", size=10))
        lv.addSpacing(8)

        self.btn_load = _btn("📥  Download Data")
        self.btn_load.clicked.connect(self._on_load)
        lv.addWidget(self.btn_load)
        lv.addSpacing(4)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        lv.addWidget(_label("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size:11px;")
        lv.addWidget(self.log_box)
        lv.addStretch()
        root.addWidget(left)

        # ── 우측 Matplotlib ──────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas)
        root.addWidget(right)

        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Click Download Data to start",
                ha="center", va="center", fontsize=15, color="#9E9E9E",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _on_load(self):
        self.btn_load.setEnabled(False)
        self.progress.setVisible(True)
        self.log_box.clear()
        self._thread = QThread()
        self._worker = DataWorker("2015-01-01", "2024-12-31")
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log_msg.connect(self.log_box.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, result: dict):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_load.setEnabled(True)
        self._ds = result["ds"]
        self._feat = result["features"]
        self._draw(self._ds, self._feat)
        self.data_ready.emit(result)

    def _on_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_load.setEnabled(True)
        self.log_box.append(f"[Error]\n{msg}")

    def _draw(self, ds: dict, feat: pd.DataFrame):
        self.fig.clear()
        prices = ds["prices"]

        ax1 = self.fig.add_subplot(3, 2, 1)
        ax1.plot(prices["sp500"], color="#1565C0")
        ax1.set_title("S&P 500")
        ax1.set_ylabel("Price")

        ax2 = self.fig.add_subplot(3, 2, 2)
        ax2.plot(prices["nasdaq"], color="#00838f")
        ax2.set_title("NASDAQ")

        ax3 = self.fig.add_subplot(3, 2, 3)
        ax3.plot(prices["sox"], color="#e65100")
        ax3.set_title("Philadelphia Semi (SOX)")
        ax3.set_ylabel("Price")

        ax4 = self.fig.add_subplot(3, 2, 4)
        ax4.plot(prices["vix"], color="#c62828")
        ax4.set_title("VIX (Volatility Index)")

        ax5 = self.fig.add_subplot(3, 2, 5)
        ax5.plot(feat["RSI_14"], color="#6a1b9a")
        ax5.axhline(70, color="red", lw=0.7, ls="--", alpha=0.6)
        ax5.axhline(30, color="green", lw=0.7, ls="--", alpha=0.6)
        ax5.set_title("RSI(14)")
        ax5.set_ylabel("RSI")

        ax6 = self.fig.add_subplot(3, 2, 6)
        ax6.plot(feat["RV_21d"], color="#37474f")
        ax6.set_title("Realized Volatility 21d (Annualized)")
        ax6.set_ylabel("Vol")

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════
#  Tab 2: Regime 감지
# ══════════════════════════════════════════════════════════

class RegimeTab(QWidget):
    regime_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        self._feat = None
        self._build_ui()

    def set_data(self, result: dict):
        self._ds = result["ds"]
        self._feat = result["features"]
        self.btn_run.setEnabled(True)
        self.log_box.append("✔ Data ready. Run Regime detection.")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        left = QFrame()
        left.setFrameStyle(QFrame.Shape.StyledPanel)
        left.setFixedWidth(230)
        lv = QVBoxLayout(left)

        lv.addWidget(_label("Regime Detection (K-Means)", bold=True, size=13))
        lv.addWidget(_label("Week 2: K-Means Unsupervised\nUsing volatility features\nBull / Sideways / Bear", size=10))
        lv.addSpacing(8)

        self.btn_run = _btn("🔍  Run K-Means", color="#00695c")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        lv.addWidget(self.btn_run)
        lv.addSpacing(4)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        lv.addWidget(_label("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size:11px;")
        lv.addWidget(self.log_box)
        lv.addStretch()
        root.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(NavigationToolbar2QT(self.canvas, self))
        rv.addWidget(self.canvas)
        root.addWidget(right)

        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Download data in Tab 1 first",
                ha="center", va="center", fontsize=14, color="#9E9E9E",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _on_run(self):
        if self._feat is None:
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.log_box.clear()
        self._thread = QThread()
        self._worker = RegimeWorker(self._ds, self._feat)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log_msg.connect(self.log_box.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, result: dict):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        result["ds"] = self._ds
        result["features"] = self._feat
        self._draw(result["regimes"])
        self.regime_ready.emit(result)

    def _on_error(self, msg: str):
        self._thread.quit()
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.log_box.append(f"[Error]\n{msg}")

    def _draw(self, regimes: pd.Series):
        self.fig.clear()
        prices = self._ds["prices"]

        # ── S&P500 + Regime 배경색 ──
        ax1 = self.fig.add_subplot(2, 2, (1, 2))
        ax1.plot(prices["sp500"], color="#1565C0", lw=0.8, label="S&P500")
        for regime, color in REGIME_COLORS.items():
            mask = regimes == regime
            if mask.any():
                ax1.fill_between(prices.index,
                                 prices["sp500"].min(), prices["sp500"].max(),
                                 where=prices.index.isin(regimes[mask].index),
                                 alpha=0.15, color=color, label=regime)
        ax1.set_title("S&P500 Price + Regime Background")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.tick_params(axis="x", labelrotation=30, labelsize=8)

        # ── Regime 분포 파이차트 ──
        ax2 = self.fig.add_subplot(2, 2, 3)
        counts = regimes.value_counts()
        colors = [REGIME_COLORS.get(r, "#999") for r in counts.index]
        ax2.pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90)
        ax2.set_title("Regime Distribution")

        # ── VIX vs RV_21d 산점도 ──
        ax3 = self.fig.add_subplot(2, 2, 4)
        feat = self._feat.reindex(regimes.index).dropna()
        for regime, color in REGIME_COLORS.items():
            mask = (regimes == regime).reindex(feat.index, fill_value=False)
            ax3.scatter(feat.loc[mask, "VIX_Level"],
                        feat.loc[mask, "RV_21d"],
                        c=color, alpha=0.4, s=6, label=regime)
        ax3.set_xlabel("VIX Level")
        ax3.set_ylabel("RV_21d")
        ax3.set_title("VIX vs Realized Volatility (Regime colors)")
        ax3.legend(fontsize=8)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════
#  Tab 3: MLP 학습 (PyTorch)
# ══════════════════════════════════════════════════════════

class TrainTab(QWidget):
    model_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        self._feat = None
        self._history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self._build_ui()

    def set_data(self, result: dict):
        self._ds = result["ds"]
        self._feat = result["features"]
        self.btn_train.setEnabled(True)
        self.log_box.append("✔ Data ready. Click Train to start.")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── 좌측 컨트롤 ──────────────────────────
        left = QFrame()
        left.setFrameStyle(QFrame.Shape.StyledPanel)
        left.setFixedWidth(250)
        lv = QVBoxLayout(left)

        lv.addWidget(_label("PyTorch MLP Training", bold=True, size=13))
        lv.addWidget(_label("Week 3-5: MLP + Dropout\n+ EarlyStopping\nDirection classifier (Up/Down)", size=10))
        lv.addSpacing(8)

        # 하이퍼파라미터
        gb = _section("Hyperparameters")
        form = QFormLayout(gb)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 500)
        self.spin_epochs.setValue(100)
        self.dspin_lr = QDoubleSpinBox()
        self.dspin_lr.setRange(1e-5, 1e-1)
        self.dspin_lr.setSingleStep(0.0005)
        self.dspin_lr.setValue(0.001)
        self.dspin_lr.setDecimals(5)
        form.addRow("Epochs:", self.spin_epochs)
        form.addRow("LR:", self.dspin_lr)
        lv.addWidget(gb)
        lv.addSpacing(6)

        self.btn_train = _btn("🧠  Start Training", color="#e65100")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._on_train)
        lv.addWidget(self.btn_train)
        lv.addSpacing(4)

        lv.addWidget(_label("Progress:"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lv.addWidget(self.progress)
        lv.addSpacing(4)

        lv.addWidget(_label("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size:10px;")
        lv.addWidget(self.log_box)
        lv.addStretch()
        root.addWidget(left)

        # ── 우측 Matplotlib (실시간 업데이트) ──
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(NavigationToolbar2QT(self.canvas, self))
        rv.addWidget(self.canvas)
        root.addWidget(right)

        self._init_live_axes()

    def _init_live_axes(self):
        self.fig.clear()
        self._ax_loss = self.fig.add_subplot(2, 2, 1)
        self._ax_acc  = self.fig.add_subplot(2, 2, 2)
        self._ax_roc  = self.fig.add_subplot(2, 2, 3)
        self._ax_prob = self.fig.add_subplot(2, 2, 4)
        for ax, title in [
            (self._ax_loss, "Loss Curve"),
            (self._ax_acc,  "Accuracy Curve"),
            (self._ax_roc,  "ROC Curve"),
            (self._ax_prob, "Predicted Probability Distribution"),
        ]:
            ax.set_title(title)
            ax.text(0.5, 0.5, "Waiting for training...", ha="center", va="center",
                    fontsize=11, color="#9E9E9E", transform=ax.transAxes)
        self.canvas.draw()

    def _on_train(self):
        self.btn_train.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()
        self._history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self._init_live_axes()

        self._thread = QThread()
        self._worker = TrainWorker(
            self._ds, self._feat,
            self.spin_epochs.value(),
            self.dspin_lr.value()
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.log_msg.connect(self.log_box.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_epoch(self, ep: int, total: int, tl: float, vl: float, ta: float, va: float):
        self.progress.setValue(int(ep / total * 100))
        self._history["train_loss"].append(tl)
        self._history["val_loss"].append(vl)
        self._history["train_acc"].append(ta)
        self._history["val_acc"].append(va)
        # 실시간 업데이트 (5 epoch마다)
        if ep % 5 == 0 or ep == total:
            epochs = list(range(1, len(self._history["train_loss"]) + 1))
            self._ax_loss.cla()
            self._ax_loss.plot(epochs, self._history["train_loss"], label="train", color="#e65100")
            self._ax_loss.plot(epochs, self._history["val_loss"],   label="val",   color="#1565C0", ls="--")
            self._ax_loss.set_title("Loss Curve")
            self._ax_loss.legend(fontsize=8)

            self._ax_acc.cla()
            self._ax_acc.plot(epochs, self._history["train_acc"], label="train", color="#e65100")
            self._ax_acc.plot(epochs, self._history["val_acc"],   label="val",   color="#1565C0", ls="--")
            self._ax_acc.axhline(0.55, color="green", lw=0.8, ls=":", label="55% target")
            self._ax_acc.set_title("Accuracy Curve")
            self._ax_acc.legend(fontsize=8)

            self.canvas.draw()

    def _on_done(self, result: dict):
        self._thread.quit()
        self.btn_train.setEnabled(True)
        self.progress.setValue(100)
        self._draw_test_results(result)
        result["ds"] = self._ds
        result["features"] = self._feat
        self.model_ready.emit(result)

    def _on_error(self, msg: str):
        self._thread.quit()
        self.btn_train.setEnabled(True)
        self.log_box.append(f"[Error]\n{msg}")

    def _draw_test_results(self, result: dict):
        from sklearn.metrics import roc_curve, auc
        probs = result["probs_te"]
        y_te  = result["y_te"]

        # ROC
        self._ax_roc.cla()
        fpr, tpr, _ = roc_curve(y_te, probs)
        auc_val = auc(fpr, tpr)
        self._ax_roc.plot(fpr, tpr, color="#e65100", lw=1.5, label=f"AUC={auc_val:.3f}")
        self._ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
        self._ax_roc.set_xlabel("FPR")
        self._ax_roc.set_ylabel("TPR")
        self._ax_roc.set_title("ROC Curve (Test)")
        self._ax_roc.legend(fontsize=9)

        # 확률 분포
        self._ax_prob.cla()
        self._ax_prob.hist(probs[y_te == 1], bins=30, alpha=0.6, color="#2e7d32", label="Up(1)")
        self._ax_prob.hist(probs[y_te == 0], bins=30, alpha=0.6, color="#c62828", label="Down(0)")
        self._ax_prob.axvline(0.5, color="black", lw=1, ls="--")
        self._ax_prob.set_xlabel("P(Up)")
        self._ax_prob.set_title(f"Predicted Probability  (acc={result['acc']:.3f})")
        self._ax_prob.legend(fontsize=8)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════
#  Tab 4: 백테스트 결과
# ══════════════════════════════════════════════════════════

class BacktestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        self._feat = None
        self._model = None
        self._scaler_sc = None
        self._regimes = None
        self._build_ui()

    def set_data(self, result: dict):
        self._ds = result["ds"]
        self._feat = result["features"]
        self._check_ready()

    def set_model(self, result: dict):
        self._model = result["model"]
        self._scaler_sc = result["scaler"]
        if self._ds is None:
            self._ds = result.get("ds")
            self._feat = result.get("features")
        self._check_ready()

    def set_regime(self, result: dict):
        self._regimes = result["regimes"]
        if self._ds is None:
            self._ds = result.get("ds")
            self._feat = result.get("features")
        self._check_ready()

    def _check_ready(self):
        if self._model is not None and self._ds is not None:
            self.btn_run.setEnabled(True)
            self.log_box.append("✔ Model ready. Run backtest.")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        left = QFrame()
        left.setFrameStyle(QFrame.Shape.StyledPanel)
        left.setFixedWidth(230)
        lv = QVBoxLayout(left)

        lv.addWidget(_label("Backtest Results", bold=True, size=13))
        lv.addWidget(_label("4 Strategy Comparison:\nBuy&Hold / Markowitz\nKelly+MLP / Regime Switch", size=10))
        lv.addSpacing(8)

        self.btn_run = _btn("📊  Run Backtest", color="#6a1b9a")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        lv.addWidget(self.btn_run)
        lv.addSpacing(4)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lv.addWidget(self.progress)

        lv.addWidget(_label("Performance Metrics:"))
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lv.addWidget(self.table)

        lv.addWidget(_label("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-size:10px;")
        lv.addWidget(self.log_box)
        lv.addStretch()
        root.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(NavigationToolbar2QT(self.canvas, self))
        rv.addWidget(self.canvas)
        root.addWidget(right)

        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Complete Tab 3 training, then run backtest",
                ha="center", va="center", fontsize=14, color="#9E9E9E",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _on_run(self):
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()

        regimes = self._regimes if self._regimes is not None else pd.Series(dtype=object)

        self._thread = QThread()
        self._worker = BacktestWorker(
            self._ds, self._feat,
            self._model, self._scaler_sc,
            regimes
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log_msg.connect(self.log_box.append)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, result: dict):
        self._thread.quit()
        self.btn_run.setEnabled(True)
        self._fill_table(result["table"])
        self._draw(result["strategies"])

    def _on_error(self, msg: str):
        self._thread.quit()
        self.btn_run.setEnabled(True)
        self.log_box.append(f"[Error]\n{msg}")

    def _fill_table(self, df: pd.DataFrame):
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        self.table.setVerticalHeaderLabels(df.index.tolist())
        for r, idx in enumerate(df.index):
            for c, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.loc[idx, col]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _draw(self, strategies: dict):
        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        for name, ret in strategies.items():
            ret = ret.dropna()
            if len(ret) == 0:
                continue
            cum = (1 + ret).cumprod()
            color = STRAT_COLORS.get(name, "#555")
            ax1.plot(cum, label=name, color=color, lw=1.4)

            # Drawdown
            peak = cum.cummax()
            dd = (cum - peak) / peak * 100
            ax2.plot(dd, label=name, color=color, lw=1.0, alpha=0.8)

        ax1.set_title("Cumulative Returns (4 Strategies)")
        ax1.set_ylabel("Cumulative Return (x)")
        ax1.axhline(1, color="gray", lw=0.6, ls="--")
        ax1.legend(fontsize=9)
        ax1.tick_params(axis="x", labelrotation=30, labelsize=8)

        ax2.set_title("Drawdown (%)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.axhline(0, color="gray", lw=0.6, ls="--")
        ax2.legend(fontsize=9)
        ax2.tick_params(axis="x", labelrotation=30, labelsize=8)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════
#  메인 윈도우
# ══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Can ML Beat the Market? — Computational Physics Midterm")
        self.resize(1280, 820)
        self._build()

    def _build(self):
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setStyleSheet("QTabBar::tab { min-width: 160px; padding: 8px; font-size: 12px; }")

        self.tab_data    = DataTab()
        self.tab_regime  = RegimeTab()
        self.tab_train   = TrainTab()
        self.tab_backtest = BacktestTab()

        tabs.addTab(self.tab_data,     "① Data & Features")
        tabs.addTab(self.tab_regime,   "② Regime Detection")
        tabs.addTab(self.tab_train,    "③ MLP Training (PyTorch)")
        tabs.addTab(self.tab_backtest, "④ Backtest Results")

        # 탭 간 데이터 연결
        self.tab_data.data_ready.connect(self.tab_regime.set_data)
        self.tab_data.data_ready.connect(self.tab_train.set_data)
        self.tab_data.data_ready.connect(self.tab_backtest.set_data)
        self.tab_regime.regime_ready.connect(self.tab_backtest.set_regime)
        self.tab_train.model_ready.connect(self.tab_backtest.set_model)

        # 탭 완료 시 자동 이동
        self.tab_data.data_ready.connect(lambda _: tabs.setCurrentIndex(1))
        self.tab_regime.regime_ready.connect(lambda _: tabs.setCurrentIndex(2))
        self.tab_train.model_ready.connect(lambda _: tabs.setCurrentIndex(3))

        self.setCentralWidget(tabs)

        # 상태바
        self.statusBar().showMessage(
            "PNU Computational Physics Midterm | "
            "① Download Data → ② Regime Detection → ③ MLP Training → ④ Backtest"
        )


# ══════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
