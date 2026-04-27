"""
strategy_kelly.py — Kelly Criterion variants
  - kelly_positions       : Long-only (original)
  - kelly_positions_ls    : Long-Short (short when bear signal)
  - kelly_positions_cf    : Confidence Filter (trade only when |f*| > threshold)
"""

import numpy as np
import pandas as pd


def _kelly_f(probs: np.ndarray, b: float = 1.0) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    q = 1 - p
    return (p * b - q) / b


def kelly_positions(probs: np.ndarray,
                    b: float = 1.0,
                    fraction: float = 1.0,
                    max_pos: float = 1.0) -> np.ndarray:
    """Long-only Kelly."""
    pos = _kelly_f(probs, b) * fraction
    return np.clip(pos, 0.0, max_pos)


def kelly_positions_ls(probs: np.ndarray,
                       b: float = 1.0,
                       fraction: float = 1.0,
                       max_pos: float = 1.0) -> np.ndarray:
    """Long-Short Kelly: 하락 예측 시 숏 포지션 허용."""
    pos = _kelly_f(probs, b) * fraction
    return np.clip(pos, -max_pos, max_pos)


def kelly_positions_cf(probs: np.ndarray,
                       b: float = 1.0,
                       fraction: float = 1.0,
                       max_pos: float = 1.0,
                       threshold: float = 0.05) -> np.ndarray:
    """Confidence Filter Kelly: |f*| < threshold 이면 포지션 0."""
    f = _kelly_f(probs, b)
    pos = f * fraction
    pos[np.abs(f) < threshold] = 0.0
    return np.clip(pos, 0.0, max_pos)


def kelly_positions_ls_cf(probs: np.ndarray,
                          b: float = 1.0,
                          fraction: float = 1.0,
                          max_pos: float = 1.0,
                          threshold: float = 0.05) -> np.ndarray:
    """Long-Short + Confidence Filter 조합."""
    f = _kelly_f(probs, b)
    pos = f * fraction
    pos[np.abs(f) < threshold] = 0.0
    return np.clip(pos, -max_pos, max_pos)


def backtest_kelly(returns: pd.Series, positions: np.ndarray,
                   dates: pd.DatetimeIndex,
                   name: str = "Kelly_MLP") -> pd.Series:
    ret_aligned = returns.reindex(dates).values
    strat_returns = positions * ret_aligned
    return pd.Series(strat_returns, index=dates, name=name)
