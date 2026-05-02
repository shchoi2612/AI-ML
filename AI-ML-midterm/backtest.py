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


def save_backtest_result(prices: pd.DataFrame = None,
                         path=None,
                         include_kelly: bool = True) -> dict:
    """
    백테스트 결과를 backtest_result.json으로 저장.

    저장 내용:
      - markowitz / buy_and_hold / kelly_mlp: Sharpe, MDD, CAGR, 누적수익률
      - daily_returns: 날짜별 일간 수익률
      - cumulative_returns: 날짜-누적수익률 매핑 테이블 (진입일 수익률 계산용)
      - excess_vs_bh: Markowitz ÷ Buy&Hold 상대 누적수익률

    반환: 저장된 결과 dict
    """
    import json
    from pathlib import Path as _Path
    from datetime import date as _date
    from strategy_markowitz import backtest_markowitz, backtest_markowitz_momentum

    _returns_df = None
    if prices is None:
        from data import build_dataset
        _ds = build_dataset()
        prices = _ds["prices"]
        _returns_df = _ds["returns"]
    else:
        from data import compute_returns
        _returns_df = compute_returns(prices)

    if path is None:
        path = _Path(__file__).parent / "backtest_result.json"

    bh   = buy_and_hold(prices)
    mw   = backtest_markowitz(prices)
    mwm  = backtest_markowitz_momentum(prices)

    cum_mw  = (1 + mw).cumprod()
    cum_bh  = (1 + bh).cumprod()
    cum_mwm = (1 + mwm).cumprod()

    common_idx = cum_mw.index.intersection(cum_bh.index)
    excess_cum = (cum_mw.reindex(common_idx) / cum_bh.reindex(common_idx)).dropna()

    result = {
        "generated_at": str(_date.today()),
        "markowitz": {
            "sharpe": round(sharpe_ratio(mw), 4),
            "mdd":    round(max_drawdown(mw), 4),
            "cagr":   round(cagr(mw), 4),
            "final_return": round(float(cum_mw.iloc[-1] - 1), 4),
        },
        "markowitz_momentum": {
            "sharpe": round(sharpe_ratio(mwm), 4),
            "mdd":    round(max_drawdown(mwm), 4),
            "cagr":   round(cagr(mwm), 4),
            "final_return": round(float(cum_mwm.iloc[-1] - 1), 4),
        },
        "buy_and_hold": {
            "sharpe": round(sharpe_ratio(bh), 4),
            "mdd":    round(max_drawdown(bh), 4),
            "cagr":   round(cagr(bh), 4),
            "final_return": round(float(cum_bh.iloc[-1] - 1), 4),
        },
        "daily_returns": {
            "markowitz":          {str(d.date()): round(float(v), 6) for d, v in mw.items()},
            "markowitz_momentum": {str(d.date()): round(float(v), 6) for d, v in mwm.items()},
            "buy_and_hold":       {str(d.date()): round(float(v), 6) for d, v in bh.items()},
        },
        "cumulative_returns": {
            "markowitz":          {str(d.date()): round(float(v), 6) for d, v in cum_mw.items()},
            "markowitz_momentum": {str(d.date()): round(float(v), 6) for d, v in cum_mwm.items()},
            "buy_and_hold":       {str(d.date()): round(float(v), 6) for d, v in cum_bh.items()},
        },
        "excess_vs_bh": {
            str(d.date()): round(float(v), 6) for d, v in excess_cum.items()
        },
    }

    # ── Kelly+MLP Walk-Forward (시간 소요: ~수 분) ──
    if include_kelly:
        try:
            from features import build_features
            print("  Kelly+MLP Walk-Forward 계산 중... (수 분 소요)")
            feat    = build_features(prices, _returns_df)
            sp_ret  = _returns_df["sp500_ret"].reindex(feat.index)
            kelly   = walk_forward_kelly(feat, sp_ret).dropna()
            if len(kelly) > 0:
                cum_kelly = (1 + kelly).cumprod()
                result["kelly_mlp"] = {
                    "sharpe": round(sharpe_ratio(kelly), 4),
                    "mdd":    round(max_drawdown(kelly), 4),
                    "cagr":   round(cagr(kelly), 4),
                    "final_return": round(float(cum_kelly.iloc[-1] - 1), 4),
                }
                result["cumulative_returns"]["kelly_mlp"] = {
                    str(d.date()): round(float(v), 6) for d, v in cum_kelly.items()
                }
                result["daily_returns"]["kelly_mlp"] = {
                    str(d.date()): round(float(v), 6) for d, v in kelly.items()
                }
                print(f"  ✅ Kelly+MLP: CAGR {result['kelly_mlp']['cagr']*100:.1f}%  "
                      f"Sharpe {result['kelly_mlp']['sharpe']:.3f}  "
                      f"MDD {result['kelly_mlp']['mdd']*100:.1f}%")
        except Exception as e:
            print(f"  ⚠️ Kelly+MLP 건너뜀: {e}")

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  ✅ backtest_result.json 저장 완료 → {path}")
    print(f"     Markowitz         : CAGR {result['markowitz']['cagr']*100:.1f}%  "
          f"Sharpe {result['markowitz']['sharpe']:.3f}  "
          f"MDD {result['markowitz']['mdd']*100:.1f}%")
    print(f"     Markowitz+Momentum: CAGR {result['markowitz_momentum']['cagr']*100:.1f}%  "
          f"Sharpe {result['markowitz_momentum']['sharpe']:.3f}  "
          f"MDD {result['markowitz_momentum']['mdd']*100:.1f}%")
    print(f"     Buy&Hold          : CAGR {result['buy_and_hold']['cagr']*100:.1f}%  "
          f"Sharpe {result['buy_and_hold']['sharpe']:.3f}  "
          f"MDD {result['buy_and_hold']['mdd']*100:.1f}%")
    return result


if __name__ == "__main__":
    from data import build_dataset
    from strategy_markowitz import backtest_markowitz, backtest_markowitz_momentum

    print("백테스트 실행 중...")
    ds = build_dataset()

    bh  = buy_and_hold(ds["prices"])
    mw  = backtest_markowitz(ds["prices"])
    mwm = backtest_markowitz_momentum(ds["prices"])

    strategies = {"Buy&Hold": bh, "Markowitz": mw, "Markowitz+Momentum": mwm}
    tbl = performance_table(strategies)
    print(tbl)

    print()
    save_backtest_result(ds["prices"])
