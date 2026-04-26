"""
strategy_kelly.py — 전략 A: PyTorch MLP + Kelly Criterion
분수 Kelly (1/4) 포지션 사이징
"""

import numpy as np
import pandas as pd


def kelly_positions(probs: np.ndarray,
                    b: float = 1.0,
                    fraction: float = 0.25,
                    max_pos: float = 0.5) -> np.ndarray:
    """
    Kelly 공식: f* = (p*b - q) / b  (b=1이면 f* = p - q = 2p - 1)
    분수 Kelly: position = f* * fraction
    음수(하락 예측) → 0 (롱온리 전략)
    """
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    q = 1 - p
    kelly_f = (p * b - q) / b
    pos = kelly_f * fraction
    pos = np.clip(pos, 0.0, max_pos)
    return pos


def backtest_kelly(returns: pd.Series, positions: np.ndarray,
                   dates: pd.DatetimeIndex) -> pd.Series:
    """
    날짜별 Kelly 포지션으로 수익률 계산.
    position_t × return_{t+1}
    """
    ret_aligned = returns.reindex(dates).values
    strat_returns = positions * ret_aligned
    return pd.Series(strat_returns, index=dates, name="Kelly_MLP")
