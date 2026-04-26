"""
backtest.py — Walk-Forward 백테스터 + 성과 지표
TRD 6.1: Train 252일 / Test 63일 롤링 Walk-Forward
TRD 6.2: Sharpe, MDD, CAGR
"""

import numpy as np
import pandas as pd


RF_DAILY = 0.02 / 252


# ──────────────────────────────────────────────
# 성과 지표 (TRD 6.2)
# ──────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = RF_DAILY) -> float:
    excess = returns - rf
    std = excess.std()
    if std < 1e-9:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def cagr(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    n_years = len(returns) / 252
    if n_years < 1e-6 or cum.iloc[-1] <= 0:
        return 0.0
    return float(cum.iloc[-1] ** (1 / n_years) - 1)


def performance_table(strategies: dict) -> pd.DataFrame:
    """
    strategies: {name: pd.Series of daily returns}
    Returns DataFrame with Sharpe, MDD, CAGR, 누적수익률
    """
    rows = []
    for name, ret in strategies.items():
        ret = ret.dropna()
        cum_final = float((1 + ret).cumprod().iloc[-1]) if len(ret) else 0.0
        rows.append({
            "전략": name,
            "Sharpe": round(sharpe_ratio(ret), 3),
            "MDD": f"{max_drawdown(ret)*100:.1f}%",
            "CAGR": f"{cagr(ret)*100:.1f}%",
            "누적수익률": f"{(cum_final - 1)*100:.1f}%",
        })
    return pd.DataFrame(rows).set_index("전략")


# ──────────────────────────────────────────────
# Walk-Forward 백테스트
# ──────────────────────────────────────────────

def walk_forward_kelly(features: pd.DataFrame,
                       returns: pd.Series,
                       train_window: int = 252,
                       test_window: int = 63,
                       epoch_callback=None) -> pd.Series:
    """
    Walk-Forward Kelly+MLP 전략.
    각 폴드: 이전 train_window일로 PyTorch MLP 학습 → test_window일 예측
    Look-ahead bias 완전 차단.
    """
    from model_torch import prepare_sequences, train_model, predict_proba
    from strategy_kelly import kelly_positions, backtest_kelly

    all_returns = []
    feat_arr = features.values
    ret_arr = returns.reindex(features.index).values
    dates = features.index
    LOOKBACK = 60

    fold = 0
    start = train_window
    while start + test_window <= len(dates):
        # 학습 구간: [0, start)  테스트 구간: [start, start+test_window)
        tr_feat = features.iloc[:start]
        tr_ret = returns.reindex(tr_feat.index)
        X, y, seq_dates = prepare_sequences(tr_feat, tr_ret, lookback=LOOKBACK)
        if len(X) < 50:
            start += test_window
            continue

        # 간단 학습 (walk-forward는 epoch 줄여 속도 확보)
        model, scaler, _ = train_model(X, y, epochs=30, batch_size=32, patience=5)

        # 테스트 구간 피처
        te_feat = features.iloc[start:start + test_window + LOOKBACK]
        te_ret = returns.reindex(te_feat.index)
        X_te, y_te, te_dates = prepare_sequences(te_feat, te_ret, lookback=LOOKBACK)
        if len(X_te) == 0:
            start += test_window
            continue

        probs = predict_proba(model, scaler, X_te)
        pos = kelly_positions(probs)
        fold_ret = backtest_kelly(returns, pos, te_dates)
        all_returns.append(fold_ret)

        fold += 1
        if epoch_callback:
            epoch_callback(fold, max(1, (len(dates) - train_window) // test_window))

        start += test_window

    if not all_returns:
        return pd.Series(dtype=float, name="Kelly_MLP")
    return pd.concat(all_returns).rename("Kelly_MLP")


def buy_and_hold(prices: pd.DataFrame, ticker: str = "sp500") -> pd.Series:
    return prices[ticker].pct_change().dropna().rename("Buy&Hold")


if __name__ == "__main__":
    from data import build_dataset
    from features import build_features
    from strategy_markowitz import backtest_markowitz

    ds = build_dataset()
    feat = build_features(ds["prices"], ds["returns"])
    sp_ret = ds["returns"]["sp500_ret"].reindex(feat.index)

    bh = buy_and_hold(ds["prices"])
    mw = backtest_markowitz(ds["prices"])

    strategies = {"Buy&Hold": bh, "Markowitz": mw}
    tbl = performance_table(strategies)
    print(tbl)
