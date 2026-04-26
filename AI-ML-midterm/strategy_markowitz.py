"""
strategy_markowitz.py — 전략 B: Markowitz 포트폴리오 최적화
샤프지수 최대화 + 월별 리밸런싱
자산: S&P500 / NASDAQ / 필라델피아 반도체 (교수님 추천)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RF_ANNUAL = 0.02        # 무위험이자율 2%


def neg_sharpe(weights: np.ndarray,
               mu: np.ndarray,
               cov: np.ndarray,
               rf: float = RF_ANNUAL / 252) -> float:
    port_ret = np.dot(weights, mu) * 252
    port_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
    if port_vol < 1e-9:
        return 0.0
    return -(port_ret - RF_ANNUAL) / port_vol


def optimal_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """샤프지수 최대화 포트폴리오 비중."""
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        args=(mu, cov),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x if result.success else np.ones(n) / n


def backtest_markowitz(prices: pd.DataFrame,
                       rebalance_freq: str = "ME") -> pd.Series:
    """
    월별 리밸런싱 Markowitz 전략 백테스트.
    prices: sp500 / nasdaq / sox 열 포함 DataFrame
    """
    assets = ["sp500", "nasdaq", "sox"]
    ret = prices[assets].pct_change().dropna()

    portfolio_ret = pd.Series(index=ret.index, dtype=float)
    months = ret.resample(rebalance_freq).last().index

    weights = np.ones(len(assets)) / len(assets)   # 초기 균등 비중

    for i, month_end in enumerate(months):
        # 이번 달 수익률 구간
        if i == 0:
            period_ret = ret[ret.index <= month_end]
        else:
            prev_end = months[i - 1]
            period_ret = ret[(ret.index > prev_end) & (ret.index <= month_end)]

        # 전달 데이터로 비중 갱신 (look-ahead bias 방지)
        if i > 0 and len(ret[ret.index <= months[i - 1]]) >= 60:
            hist = ret[ret.index <= months[i - 1]]
            weights = optimal_weights(hist)

        portfolio_ret[period_ret.index] = period_ret.values @ weights

    return portfolio_ret.dropna().rename("Markowitz")


if __name__ == "__main__":
    from data import build_dataset
    ds = build_dataset()
    strat = backtest_markowitz(ds["prices"])
    cum = (1 + strat).cumprod()
    print(f"Markowitz 최종 누적수익: {cum.iloc[-1]:.3f}x")
