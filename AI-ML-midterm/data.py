"""
data.py — yfinance 데이터 수집 + 전처리
Week 2: 수익률, 로그수익률, Min-Max 정규화 직접 적용
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "sox": "^SOX",
    "vix": "^VIX",
}

TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"


def download_prices(start: str = "2010-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """모든 티커의 종가 다운로드."""
    raw = yf.download(
        list(TICKERS.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].copy()
    close.columns = list(TICKERS.keys())
    close = close.ffill().dropna()
    return close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """일반 수익률 + 로그수익률 계산 (Week 2)."""
    ret = prices.pct_change()
    logret = np.log(prices / prices.shift(1))
    ret.columns = [f"{c}_ret" for c in prices.columns]
    logret.columns = [f"{c}_logret" for c in prices.columns]
    return pd.concat([ret, logret], axis=1).dropna()


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-Max 정규화 (Week 2: 정규화)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def build_dataset(start: str = "2010-01-01", end: str = "2025-12-31") -> dict:
    prices = download_prices(start, end)
    returns = compute_returns(prices)
    return {"prices": prices, "returns": returns}


if __name__ == "__main__":
    ds = build_dataset()
    print("prices shape :", ds["prices"].shape)
    print(ds["prices"].tail(3))
