"""
model_torch.py — PyTorch MLP 방향성 예측 모델
Week 3–5 핵심 개념 종합 적용:
  - MLP (Week 3: 신경망 구조)
  - Dropout (Week 5: 과적합 방지)
  - Adam optimizer (Week 4)
  - EarlyStopping (Week 5: Regularization 응용)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


LOOKBACK = 60       # 60일 히스토리 → 한 샘플
FORECAST_DAYS = 5   # 5일 후 방향성 예측


# ──────────────────────────────────────────────
# 모델 구조
# ──────────────────────────────────────────────

class DirectionMLP(nn.Module):
    """
    입력: lookback × n_features → Flatten → Dense layers
    출력: P(상승) sigmoid 확률
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),            # Week 5: 과적합 방지
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),            # Week 5: Regularization
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),               # P(상승)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────

def prepare_sequences(features: pd.DataFrame,
                       returns: pd.Series,
                       lookback: int = LOOKBACK,
                       forecast: int = FORECAST_DAYS):
    """
    X : (N, lookback * n_features)  — flattened window
    y : (N,)                         — 1 if 5d return > 0 else 0
    dates : DatetimeIndex of prediction target dates
    """
    feat_arr = features.values
    ret_arr = returns.values
    dates = features.index

    X_list, y_list, date_list = [], [], []
    for i in range(lookback, len(feat_arr) - forecast):
        window = feat_arr[i - lookback:i]           # (lookback, n_features)
        future_ret = ret_arr[i:i + forecast].sum()  # 5일 누적 수익률
        X_list.append(window.flatten())
        y_list.append(1.0 if future_ret > 0 else 0.0)
        date_list.append(dates[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DatetimeIndex(date_list)


def split_train_test(X, y, dates, train_end: str = "2021-12-31"):
    mask = dates <= train_end
    return X[mask], y[mask], X[~mask], y[~mask], dates[mask], dates[~mask]


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                lr: float = 1e-3, patience: int = 10,
                epoch_callback=None, device: str = "cpu"):
    """
    epoch_callback(epoch, total, train_loss, val_loss, train_acc, val_acc)
    가 주어지면 매 epoch 호출 (GUI 실시간 업데이트용).
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train).astype(np.float32)

    # 80/20 분할 (검증셋)
    split = int(len(X_sc) * 0.8)
    X_tr, X_val = X_sc[:split], X_sc[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DirectionMLP(X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            tr_correct += ((pred >= 0.5).float() == yb).sum().item()
            tr_total += len(xb)

        # ── val ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_correct += ((pred >= 0.5).float() == yb).sum().item()
                val_total += len(xb)

        tl = tr_loss / tr_total
        vl = val_loss / val_total
        ta = tr_correct / tr_total
        va = val_correct / val_total

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if epoch_callback:
            epoch_callback(epoch, epochs, tl, vl, ta, va)

        # Early stopping (Week 5)
        if vl < best_val_loss:
            best_val_loss = vl
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                if best_state:
                    model.load_state_dict(best_state)
                break

    return model, scaler, history


def predict_proba(model: DirectionMLP, scaler: StandardScaler,
                  X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """P(상승) 확률 배열 반환."""
    model.eval()
    X_sc = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        probs = model(torch.from_numpy(X_sc).to(device)).cpu().numpy()
    return probs


if __name__ == "__main__":
    from data import build_dataset
    from features import build_features
    ds = build_dataset()
    feat = build_features(ds["prices"], ds["returns"])
    sp_ret = ds["returns"]["sp500_ret"].reindex(feat.index)
    X, y, dates = prepare_sequences(feat, sp_ret)
    X_tr, y_tr, X_te, y_te, d_tr, d_te = split_train_test(X, y, dates)
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    def cb(ep, total, tl, vl, ta, va):
        if ep % 10 == 0:
            print(f"Epoch {ep}/{total}  loss={tl:.4f}  val_loss={vl:.4f}  acc={ta:.3f}  val_acc={va:.3f}")

    model, scaler, history = train_model(X_tr, y_tr, epochs=50, epoch_callback=cb)
    probs = predict_proba(model, scaler, X_te)
    acc = ((probs >= 0.5).astype(float) == y_te).mean()
    print(f"Test Accuracy: {acc:.4f}")
