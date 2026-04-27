"""
today_signal.py — Markowitz 기반 오늘 포트폴리오 신호
  자산: S&P500(^GSPC) / NASDAQ(^IXIC) / 필라델피아 반도체(^SOX)
  - 최근 252일 데이터로 Sharpe 최대화 최적 비중 계산
  - 각 자산별 개별 지표 (Sharpe, 모멘텀, 변동성)
  - 포트폴리오 기대 Sharpe / 기대 수익률 / 기대 변동성
  - 오늘 리밸런싱 권고 비중 출력
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
from datetime import date
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from data import build_dataset
from backtest import sharpe_ratio, max_drawdown, cagr

RF_ANNUAL = 0.02
ASSETS = ["sp500", "nasdaq", "sox"]
ASSET_NAMES = {"sp500": "S&P500", "nasdaq": "NASDAQ", "sox": "SOX(필라델피아 반도체)"}


def optimal_weights(ret_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    n = ret_df.shape[1]
    mu  = ret_df.mean().values
    cov = ret_df.cov().values

    def neg_sharpe(w):
        pr = np.dot(w, mu) * 252
        pv = np.sqrt(w @ cov @ w) * np.sqrt(252)
        return -(pr - RF_ANNUAL) / pv if pv > 1e-9 else 0.0

    res = minimize(neg_sharpe, x0=np.ones(n)/n,
                   method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-12, "maxiter": 2000})
    w = res.x if res.success else np.ones(n)/n
    sharpe = -neg_sharpe(w)
    return w, sharpe


def portfolio_stats(weights: np.ndarray, ret_df: pd.DataFrame) -> dict:
    mu  = ret_df.mean().values
    cov = ret_df.cov().values
    ann_ret = float(np.dot(weights, mu) * 252)
    ann_vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(252))
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 1e-9 else 0.0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}


# ────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────

print("=" * 62)
print(f"  Markowitz Portfolio Signal  [{date.today()}]")
print("  Assets: S&P500 / NASDAQ / SOX")
print("=" * 62)
print("\n[1] Downloading latest data...")

ds  = build_dataset(end=str(date.today()))
prices = ds["prices"]
ret_all = prices[ASSETS].pct_change().dropna()

# 최근 252일 (약 1년) 기준으로 최적화
ret_1y  = ret_all.iloc[-252:]
ret_6m  = ret_all.iloc[-126:]

# ────────────────────────────────────────────
# 2. 자산별 개별 지표
# ────────────────────────────────────────────

print("\n── Individual Asset Metrics (Rolling 1Y) ─────────────")
print(f"  {'Asset':<26} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Vol(ann)':>9} {'1M%':>7} {'3M%':>7}")
print("  " + "-" * 59)

for a in ASSETS:
    r = ret_1y[a]
    sh  = sharpe_ratio(r)
    c   = cagr(r) * 100
    mdd = max_drawdown(r) * 100
    vol = r.std() * np.sqrt(252) * 100
    m1  = float((prices[a].iloc[-1] / prices[a].iloc[-22] - 1) * 100)
    m3  = float((prices[a].iloc[-1] / prices[a].iloc[-63] - 1) * 100)
    name = ASSET_NAMES[a]
    print(f"  {name:<26} {sh:>7.3f} {c:>6.1f}% {mdd:>6.1f}% {vol:>8.1f}% {m1:>+6.1f}% {m3:>+6.1f}%")

# ────────────────────────────────────────────
# 3. Markowitz 최적 비중 (1Y / 6M)
# ────────────────────────────────────────────

w_1y, sh_1y = optimal_weights(ret_1y)
w_6m, sh_6m = optimal_weights(ret_6m)
stats_1y = portfolio_stats(w_1y, ret_1y)
stats_6m = portfolio_stats(w_6m, ret_6m)

print(f"\n── Optimal Weights ────────────────────────────────────")
print(f"  {'Asset':<26} {'1Y window':>10} {'6M window':>10}")
print("  " + "-" * 48)
for i, a in enumerate(ASSETS):
    print(f"  {ASSET_NAMES[a]:<26} {w_1y[i]*100:>9.1f}% {w_6m[i]*100:>9.1f}%")

print(f"\n── Expected Portfolio Performance ─────────────────────")
print(f"  {'':26} {'1Y window':>10} {'6M window':>10}")
print("  " + "-" * 48)
print(f"  {'Sharpe Ratio':<26} {sh_1y:>10.3f} {sh_6m:>10.3f}")
print(f"  {'Ann. Return (est.)':<26} {stats_1y['ann_ret']*100:>9.1f}% {stats_6m['ann_ret']*100:>9.1f}%")
print(f"  {'Ann. Volatility':<26} {stats_1y['ann_vol']*100:>9.1f}% {stats_6m['ann_vol']*100:>9.1f}%")

# ────────────────────────────────────────────
# 4. 시장 전반 상태 판단
# ────────────────────────────────────────────

# VIX 수준
vix_now = float(prices["vix"].iloc[-1])
vix_avg = float(prices["vix"].iloc[-252:].mean())

# 포트폴리오 모멘텀 (1Y 가중 기준)
port_1m = sum(w_1y[i] * float((prices[a].iloc[-1]/prices[a].iloc[-22]-1)*100) for i,a in enumerate(ASSETS))
port_3m = sum(w_1y[i] * float((prices[a].iloc[-1]/prices[a].iloc[-63]-1)*100) for i,a in enumerate(ASSETS))

print(f"\n── Market Context ─────────────────────────────────────")
print(f"  VIX now / 1Y avg   : {vix_now:.2f} / {vix_avg:.2f}  ", end="")
if vix_now < 15:
    print("(극도 안정 — 과열 주의)")
elif vix_now < 20:
    print("(정상 수준)")
elif vix_now < 30:
    print("(불안 구간 — 변동성 상승)")
else:
    print("(공포 구간 ⚠)")

print(f"  Portfolio 1M return: {port_1m:+.2f}%")
print(f"  Portfolio 3M return: {port_3m:+.2f}%")

# ────────────────────────────────────────────
# 5. 신호 판단
# ────────────────────────────────────────────

conds = {
    "Portfolio Sharpe (1Y) > 0.5" : sh_1y > 0.5,
    "Portfolio 1M momentum > 0"   : port_1m > 0,
    "Portfolio 3M momentum > 0"   : port_3m > 0,
    "VIX < 25 (no fear)"          : vix_now < 25,
    "Min asset CAGR > 0"          : all(cagr(ret_1y[a]) > 0 for a in ASSETS),
}
passed = sum(conds.values())

if passed >= 4:
    signal  = "🟢  BUY / HOLD — 권고 비중대로 보유"
    suggest = w_1y
elif passed == 3:
    signal  = "🟡  CAUTION — 축소 보유 (비중 절반)"
    suggest = w_1y * 0.5
else:
    signal  = "🔴  AVOID — 현금 보유"
    suggest = np.zeros(3)

print(f"\n── Signal Conditions ──────────────────────────────────")
for cond, ok in conds.items():
    print(f"  [{'✓' if ok else '✗'}] {cond}")
print(f"\n  Passed: {passed}/5")

print(f"\n{'='*62}")
print(f"  SIGNAL: {signal}")
print(f"\n  Recommended allocation:")
for i, a in enumerate(ASSETS):
    bar = "█" * int(suggest[i] * 30)
    print(f"    {ASSET_NAMES[a]:<26} {suggest[i]*100:>5.1f}%  {bar}")
cash_pct = max(0.0, 1 - suggest.sum())
print(f"    {'CASH':<26} {cash_pct*100:>5.1f}%")

# ────────────────────────────────────────────
# 6. 오늘 행동 강령
# ────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"  ACTION PLAN — 오늘 해야 할 일")
print(f"{'='*62}")

# 리스크 레벨 분류
if vix_now > 30:
    risk_level = "HIGH"
elif vix_now > 20:
    risk_level = "ELEVATED"
else:
    risk_level = "NORMAL"

# SOX 과열 여부 (1M +20% 이상이면 단기 과열)
sox_1m = float((prices["sox"].iloc[-1] / prices["sox"].iloc[-22] - 1) * 100)
sox_overheated = sox_1m > 20

# 6M vs 1Y 비중 불일치 여부 (방향성 전환 신호)
weight_shift = any(abs(w_1y[i] - w_6m[i]) > 0.15 for i in range(3))

print(f"\n  [시장 상태]")
print(f"  • 리스크 레벨  : {risk_level}  (VIX {vix_now:.1f})")
if sox_overheated:
    print(f"  • SOX 단기 과열: 1개월 +{sox_1m:.1f}% — 추격 매수 주의")
if weight_shift:
    print(f"  • 최적 비중 변화 감지 (6M vs 1Y 차이 큼) — 리밸런싱 시점")

print(f"\n  [포지션 없는 경우 — 신규 진입]")
if passed >= 4:
    print(f"  ① 투자금의 {int((1-cash_pct)*100)}%를 아래 비중으로 분할 매수")
    for i, a in enumerate(ASSETS):
        if suggest[i] > 0.01:
            print(f"     - {ASSET_NAMES[a]}: 투자금의 {suggest[i]*100:.0f}%")
    if sox_overheated:
        print(f"  ② SOX 과열 중 → 오늘 한 번에 사지 말고 2~3일에 걸쳐 분할 매수")
    else:
        print(f"  ② 오늘 바로 진입 가능 (모멘텀·VIX 모두 정상)")
    print(f"  ③ 손절 기준: 개별 자산 -10% 이탈 시 재검토")
elif passed == 3:
    print(f"  ① 투자금의 50% 이하로 소규모만 진입")
    print(f"  ② 나머지는 현금 대기, 다음 주 재평가")
else:
    print(f"  ① 진입하지 마세요. 현금 보유.")
    print(f"  ② 조건이 3개 이상 충족될 때까지 대기")

print(f"\n  [포지션 있는 경우 — 기존 보유자]")
if passed >= 4:
    if weight_shift:
        print(f"  ① 비중 점검 후 리밸런싱 (6M/1Y 비중 불일치)")
        for i, a in enumerate(ASSETS):
            diff = (w_1y[i] - w_6m[i]) * 100
            if abs(diff) > 5:
                direction = "늘리기" if diff > 0 else "줄이기"
                print(f"     - {ASSET_NAMES[a]}: {direction} ({diff:+.0f}%p 차이)")
    else:
        print(f"  ① 현재 비중 유지 (리밸런싱 불필요)")
    print(f"  ② 월말에 비중 재계산 후 5%p 이상 차이나면 조정")
elif passed == 3:
    print(f"  ① 수익 난 자산 일부 익절 고려")
    print(f"  ② 비중을 권고의 50%로 축소")
else:
    print(f"  ① 전량 또는 대부분 매도 후 현금화")
    print(f"  ② 시장 안정화(VIX < 20, 모멘텀 회복) 후 재진입")

print(f"\n  [공통 주의사항]")
print(f"  • 이 신호는 Markowitz 이론 기반 — 단기 급등락 예측 불가")
print(f"  • 백테스트 14년 CAGR: Markowitz 14.5% / S&P500 11.7%")
print(f"  • SOX Sharpe 3.09는 비정상 — 과거 패턴이 지속된다는 보장 없음")
print(f"  • 거래 수수료 미반영. 월 1회 이상 리밸런싱은 비용 손실 주의")
print(f"{'='*62}\n")
