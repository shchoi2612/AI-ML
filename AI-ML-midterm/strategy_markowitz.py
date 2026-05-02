"""
strategy_markowitz.py — 전략 B: Markowitz 포트폴리오 최적화
샤프지수 최대화 + 월별 리밸런싱
자산: S&P500 / NASDAQ / 필라델피아 반도체 (교수님 추천)

변경 이력:
  - MAX_WEIGHT 50% 캡 적용 (단일 종목 집중 방지)
  - VIX 기반 현금 비중 동적 조정 (리스크 관리)
  - 모멘텀 필터 추가 (Markowitz+Momentum 전략)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RF_ANNUAL  = 0.02     # 무위험이자율 2%
MAX_WEIGHT = 0.50     # 단일 종목 최대 50% 제한


def vix_cash_pct(vix: float) -> float:
    """
    VIX 수준에 따른 현금 비중 계산.
    VIX < 20  → 현금 0~10%  (선형)
    VIX 20~30 → 현금 10~30% (선형 보간)
    VIX > 30  → 현금 30~50% (선형, 최대 50%)
    """
    if vix < 20:
        return vix / 20 * 0.10
    elif vix <= 30:
        return 0.10 + (vix - 20) / 10 * 0.20
    else:
        return min(0.50, 0.30 + (vix - 30) / 30 * 0.20)


def neg_sharpe(weights: np.ndarray,
               mu: np.ndarray,
               cov: np.ndarray,
               rf: float = RF_ANNUAL / 252) -> float:
    port_ret = np.dot(weights, mu) * 252
    port_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
    if port_vol < 1e-9:
        return 0.0
    return -(port_ret - RF_ANNUAL) / port_vol


def optimal_weights(returns_df: pd.DataFrame,
                    max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """샤프지수 최대화 포트폴리오 비중. 단일 종목 max_weight 상한."""
    n = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        args=(mu, cov),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x if result.success else np.ones(n) / n


def backtest_markowitz(prices: pd.DataFrame,
                       rebalance_freq: str = "ME") -> pd.Series:
    """
    월별 리밸런싱 Markowitz 전략 백테스트.
    - 단일 종목 최대 50% 제한
    - VIX 기반 현금 비중 동적 조정
    prices: sp500 / nasdaq / sox / vix 열 포함 DataFrame
    """
    assets = ["sp500", "nasdaq", "sox"]
    ret = prices[assets].pct_change().dropna()
    vix = prices["vix"]

    portfolio_ret = pd.Series(index=ret.index, dtype=float)
    months = ret.resample(rebalance_freq).last().index

    weights  = np.ones(len(assets)) / len(assets)   # 초기 균등 비중
    cash_pct = 0.0

    for i, month_end in enumerate(months):
        # 이번 달 수익률 구간
        if i == 0:
            period_ret = ret[ret.index <= month_end]
        else:
            prev_end = months[i - 1]
            period_ret = ret[(ret.index > prev_end) & (ret.index <= month_end)]

        # 전달 데이터로 비중 갱신 (look-ahead bias 방지)
        if i > 0 and len(ret[ret.index <= months[i - 1]]) >= 60:
            hist       = ret[ret.index <= months[i - 1]]
            risky_w    = optimal_weights(hist, max_weight=MAX_WEIGHT)

            # VIX 기반 현금 비중 조정 (전달 말 VIX 사용)
            prev_vix   = float(vix.asof(months[i - 1]))
            cash_pct   = vix_cash_pct(prev_vix)
            weights    = risky_w * (1 - cash_pct)

        # 현금 수익률 = 0 (단순화); 위험자산 수익률만 합산
        portfolio_ret[period_ret.index] = period_ret.values @ weights

    return portfolio_ret.dropna().rename("Markowitz")


MOM_LOOKBACK    = 63    # 모멘텀 측정 기간 (거래일)
MOM_WEAK_EXTRA  = 0.20  # 모멘텀 약할 때 추가 현금 비중


def backtest_markowitz_momentum(prices: pd.DataFrame,
                                rebalance_freq: str = "ME",
                                lookback_mom: int = MOM_LOOKBACK,
                                weak_cash_extra: float = MOM_WEAK_EXTRA) -> pd.Series:
    """
    Markowitz + 모멘텀 필터 전략.
    - 기존 Markowitz (VIX 조정 포함)에 모멘텀 오버레이 추가
    - 63거래일 수익률 기준: 3개 중 2개 이상 음수 → "약세" → 현금 비중 +20%p 추가
    - 3개 모두 양수 → "강세" → 정상 운용
    - 현금 비중 최대 60% 캡
    """
    assets = ["sp500", "nasdaq", "sox"]
    ret = prices[assets].pct_change().dropna()
    vix = prices["vix"]

    portfolio_ret = pd.Series(index=ret.index, dtype=float)
    months = ret.resample(rebalance_freq).last().index

    weights  = np.ones(len(assets)) / len(assets)
    cash_pct = 0.0

    for i, month_end in enumerate(months):
        if i == 0:
            period_ret = ret[ret.index <= month_end]
        else:
            prev_end = months[i - 1]
            period_ret = ret[(ret.index > prev_end) & (ret.index <= month_end)]

        if i > 0 and len(ret[ret.index <= months[i - 1]]) >= 60:
            hist     = ret[ret.index <= months[i - 1]]
            risky_w  = optimal_weights(hist, max_weight=MAX_WEIGHT)

            prev_vix = float(vix.asof(months[i - 1]))
            cash_pct = vix_cash_pct(prev_vix)

            # 모멘텀 신호: 전달 말 기준 63거래일 전 → 전달 말 수익률
            mom_end   = months[i - 1]
            mom_start = mom_end - pd.offsets.BDay(lookback_mom)
            end_px    = prices[assets].asof(mom_end)
            start_px  = prices[assets].asof(mom_start)
            neg_count = int(sum(1 for a in assets if end_px[a] / start_px[a] - 1 < 0))
            if neg_count >= 2:
                cash_pct = min(0.60, cash_pct + weak_cash_extra)

            weights = risky_w * (1 - cash_pct)

        portfolio_ret[period_ret.index] = period_ret.values @ weights

    return portfolio_ret.dropna().rename("Markowitz+Momentum")


if __name__ == "__main__":
    from data import build_dataset
    ds = build_dataset()
    strat = backtest_markowitz(ds["prices"])
    cum = (1 + strat).cumprod()
    print(f"Markowitz 최종 누적수익: {cum.iloc[-1]:.3f}x")
    strat_m = backtest_markowitz_momentum(ds["prices"])
    cum_m = (1 + strat_m).cumprod()
    print(f"Markowitz+Momentum 최종 누적수익: {cum_m.iloc[-1]:.3f}x")
