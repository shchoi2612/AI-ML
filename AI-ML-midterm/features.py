"""
features.py — 피처 엔지니어링
TRD 3.1: RSI, Bollinger %B, MACD Signal, MA200 비율
TRD 3.2: 변동성 클러스터링 피처 (ARCH 효과 기반)
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI_14")


def bollinger_pct_b(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ((close - lower) / (upper - lower).replace(0, np.nan)).rename("BB_PctB")


def macd_signal(close: pd.Series,
                fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return (macd_line - macd_line.ewm(span=signal, adjust=False).mean()).rename("MACD_Sig")


def ma200_ratio(close: pd.Series) -> pd.Series:
    return (close / close.rolling(200).mean()).rename("MA200_Ratio")


def volatility_features(returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    변동성 클러스터링 피처 (TRD 3.2).
    EMH는 수익률 예측 불가를 주장하지만 변동성의 자기상관(ARCH 효과)은 예측 가능.
    """
    rv5 = returns.rolling(5).std() * np.sqrt(252)
    rv21 = returns.rolling(21).std() * np.sqrt(252)
    vol_ratio = rv5 / rv21.replace(0, np.nan)
    vol_zscore = (rv21 - rv21.rolling(252).mean()) / rv21.rolling(252).std()
    vix_5d_change = vix - vix.shift(5)
    sq_ret = returns ** 2

    return pd.DataFrame({
        "RV_5d": rv5,
        "RV_21d": rv21,
        "Vol_Ratio": vol_ratio,
        "Vol_Zscore": vol_zscore,
        "VIX_Level": vix,
        "VIX_5d_Change": vix_5d_change,
        "Sq_Ret": sq_ret,
        "Sq_Ret_Lag1": sq_ret.shift(1),
        "Sq_Ret_Lag2": sq_ret.shift(2),
    })


def build_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """통합 피처 빌더. prices/returns 는 data.build_dataset() 결과."""
    sp = prices["sp500"]
    vix = prices["vix"]
    sp_ret = returns["sp500_ret"]

    tech = pd.concat([
        rsi(sp),
        bollinger_pct_b(sp),
        macd_signal(sp),
        ma200_ratio(sp),
    ], axis=1)

    vol = volatility_features(sp_ret, vix)
    return pd.concat([tech, vol], axis=1).dropna()


if __name__ == "__main__":
    from data import build_dataset
    ds = build_dataset()
    feat = build_features(ds["prices"], ds["returns"])
    print("features shape:", feat.shape)
    print(feat.columns.tolist())
