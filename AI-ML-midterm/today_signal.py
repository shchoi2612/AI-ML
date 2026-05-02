"""
today_signal.py — Markowitz 기반 오늘 포트폴리오 신호
  자산: S&P500(^GSPC) / NASDAQ(^IXIC) / 필라델피아 반도체(^SOX)
  - Sharpe 최대화 최적 비중 계산 (1Y / 6M 윈도우)
  - 포트폴리오 금액 기준 원화 투자금액 계산
  - 신호 히스토리 저장 및 이전 신호 수익률 검증
  - Discord Webhook 알림 (DISCORD_WEBHOOK_URL 환경변수)

환경변수:
  DISCORD_WEBHOOK_URL  Discord Webhook URL
  PORTFOLIO_KRW        투자 원금 (기본: 10000000 = 1000만원)

실행:
  uv run python today_signal.py
  uv run python today_signal.py --entry-date 2024-01-01 --entry-krw 10000000
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from data import build_dataset
from backtest import sharpe_ratio, max_drawdown, cagr

# ────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────
RF_ANNUAL       = 0.02
MIN_WEIGHT      = 0.10   # 자산별 최소 비중 (10%)
MAX_WEIGHT      = 0.50   # 자산별 최대 비중 (50%) — 단일 종목 집중 방지
ASSETS          = ["sp500", "nasdaq", "sox"]
ASSET_NAMES     = {"sp500": "S&P500", "nasdaq": "NASDAQ", "sox": "SOX"}
PORTFOLIO_KRW   = int(os.environ.get("PORTFOLIO_KRW", 10_000_000))
HISTORY_FILE    = Path(__file__).parent / "signal_history.csv"
BACKTEST_FILE   = Path(__file__).parent / "backtest_result.json"
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
STOP_LOSS_PCT   = 0.10  # 개별 자산 -10% 손절 기준


# ────────────────────────────────────────────
# 수학 함수
# ────────────────────────────────────────────
def optimal_weights(ret_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Sharpe 최대화 최적 비중.
    각 자산: MIN_WEIGHT ~ MAX_WEIGHT 제약, 합계 = 1.
    반환: (weights, sharpe)
    """
    n = ret_df.shape[1]
    mu  = ret_df.mean().values
    cov = ret_df.cov().values

    def neg_sharpe(w):
        pr = np.dot(w, mu) * 252
        pv = np.sqrt(w @ cov @ w) * np.sqrt(252)
        return -(pr - RF_ANNUAL) / pv if pv > 1e-9 else 0.0

    res = minimize(neg_sharpe, x0=np.ones(n) / n, method="SLSQP",
                   bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * n,   # ← 수정: 0~1 → MIN~MAX
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-12, "maxiter": 2000})
    w = res.x if res.success else np.ones(n) / n
    return w, -neg_sharpe(w)


def portfolio_stats(weights: np.ndarray, ret_df: pd.DataFrame) -> dict:
    mu  = ret_df.mean().values
    cov = ret_df.cov().values
    ann_ret = float(np.dot(weights, mu) * 252)
    ann_vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(252))
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 1e-9 else 0.0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}


# ────────────────────────────────────────────
# 히스토리 관리
# ────────────────────────────────────────────
_HIST_COLS = ["date", "signal", "sp500_w", "nasdaq_w", "sox_w", "cash_w",
              "sharpe_1y", "vix", "passed", "sp500_px", "nasdaq_px", "sox_px"]

def load_history() -> pd.DataFrame:
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=_HIST_COLS)


def save_signal(signal_code: str, w1y: np.ndarray,
                sharpe_1y: float, vix: float, passed: int,
                prices: pd.DataFrame) -> None:
    today_str = str(date.today())
    hist = load_history()
    hist = hist[hist["date"].astype(str) != today_str]
    row = {
        "date":      today_str,
        "signal":    signal_code,
        "sp500_w":   round(float(w1y[0]), 4),
        "nasdaq_w":  round(float(w1y[1]), 4),
        "sox_w":     round(float(w1y[2]), 4),
        "cash_w":    round(float(max(0.0, 1.0 - w1y.sum())), 4),
        "sharpe_1y": round(float(sharpe_1y), 3),
        "vix":       round(float(vix), 2),
        "passed":    int(passed),
        "sp500_px":  round(float(prices["sp500"].iloc[-1]), 2),
        "nasdaq_px": round(float(prices["nasdaq"].iloc[-1]), 2),
        "sox_px":    round(float(prices["sox"].iloc[-1]), 2),
    }
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False)
    print(f"  ✅ 신호 저장 완료 → {HISTORY_FILE.name}")


def validate_last_signal(prices: pd.DataFrame) -> tuple:
    """이전 신호 이후 수익률 계산 및 누적 적중률 반환."""
    hist = load_history()
    today_str = str(date.today())
    past = hist[hist["date"].astype(str) < today_str]
    if past.empty:
        return None, None

    last = past.iloc[-1]
    last_date = str(last["date"])[:10]
    px_cols = {"sp500": "sp500_px", "nasdaq": "nasdaq_px", "sox": "sox_px"}
    w_cols  = {"sp500": "sp500_w",  "nasdaq": "nasdaq_w",  "sox": "sox_w"}

    port_ret = 0.0
    for a in ASSETS:
        w = float(last[w_cols[a]])
        if w > 0.01:
            px_then = float(last[px_cols[a]])
            px_now  = float(prices[a].iloc[-1])
            port_ret += w * (px_now / px_then - 1)

    port_ret_pct = port_ret * 100
    ok = "✅" if port_ret_pct > 0 else "❌"
    msg = (f"이전 신호 ({last_date}, {last['signal']}) 이후 "
           f"포트폴리오: {port_ret_pct:+.2f}%  {ok}")

    buy_rows = past[past["signal"] == "BUY"].reset_index(drop=True)
    if len(buy_rows) >= 2:
        hits = total = 0
        for i in range(len(buy_rows) - 1):
            r, r_next = buy_rows.iloc[i], buy_rows.iloc[i + 1]
            ret = sum(
                float(r[w_cols[a]]) * (float(r_next[px_cols[a]]) / float(r[px_cols[a]]) - 1)
                for a in ASSETS if float(r[w_cols[a]]) > 0.01
            )
            total += 1
            hits  += int(ret > 0)
        if total > 0:
            msg += f"\n  BUY 신호 누적 적중률: {hits}/{total} ({hits/total*100:.0f}%)"

    return msg, port_ret_pct


# ────────────────────────────────────────────
# Discord 알림
# ────────────────────────────────────────────
def send_discord(signal_code: str, signal_line: str,
                 suggest: np.ndarray, sharpe_1y: float,
                 vix: float, passed: int, port_1m: float,
                 validation: str | None,
                 action: str | None = None) -> None:
    if not DISCORD_WEBHOOK:
        print("  ℹ️  DISCORD_WEBHOOK_URL 미설정 — 알림 건너뜀")
        return

    COLOR = {"BUY": 0x2ecc71, "CAUTION": 0xf39c12, "AVOID": 0xe74c3c}
    color = COLOR.get(signal_code, 0x95a5a6)

    weight_lines, krw_lines = [], []
    for i, a in enumerate(ASSETS):
        weight_lines.append(f"`{ASSET_NAMES[a]:6}` {suggest[i]*100:5.1f}%")
        amt = int(suggest[i] * PORTFOLIO_KRW)
        if amt > 0:
            krw_lines.append(f"{ASSET_NAMES[a]}: **{amt:,}원**")
    cash_pct = max(0.0, 1 - suggest.sum())
    weight_lines.append(f"`{'CASH':6}` {cash_pct*100:5.1f}%")
    krw_lines.append(f"현금: **{int(cash_pct * PORTFOLIO_KRW):,}원**")

    fields = [
        {"name": "📊 추천 비중",
         "value": "\n".join(weight_lines), "inline": True},
        {"name": f"💰 배분 ({PORTFOLIO_KRW // 10000:,}만원)",
         "value": "\n".join(krw_lines) if krw_lines else "전액 현금", "inline": True},
        {"name": "📈 주요 지표",
         "value": (f"Sharpe(1Y): **{sharpe_1y:.2f}**\n"
                   f"VIX: **{vix:.1f}**\n"
                   f"조건 충족: **{passed}/5**\n"
                   f"1M 모멘텀: **{port_1m:+.1f}%**"),
         "inline": False},
    ]
    if validation:
        fields.append({"name": "🔍 이전 신호 검증",
                       "value": validation, "inline": False})
    if action:
        fields.append({"name": "📋 오늘 행동강령",
                       "value": action, "inline": False})

    payload = json.dumps({
        "embeds": [{
            "title": f"{signal_line[:40]}  [{date.today()}]",
            "color": color,
            "fields": fields,
            "footer": {"text": "AI-ML-midterm | PNU 전산물리 | Markowitz"}
        }]
    }).encode("utf-8")

    import urllib.request
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK, data=payload,
            headers={"Content-Type": "application/json",
                     "User-Agent": "DiscordBot (custom, 1.0)"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10):
            print("  ✅ Discord 알림 전송 완료")
    except Exception as e:
        print(f"  ⚠️  Discord 알림 실패: {e}")


# ────────────────────────────────────────────
# 진입일 수익률 계산 (STEP 3)
# ────────────────────────────────────────────
def compute_entry_return(entry_date: str, entry_krw: int) -> dict:
    """
    backtest_result.json의 누적수익률 테이블로 진입일 대비 현재 수익률 계산.
    backtest.py를 먼저 실행해야 함.
    """
    if not BACKTEST_FILE.exists():
        return {"error": "backtest_result.json 없음. 먼저 uv run python backtest.py를 실행하세요."}

    with open(BACKTEST_FILE) as f:
        bt = json.load(f)

    cum_table = bt["cumulative_returns"]["markowitz"]
    dates_sorted = sorted(cum_table.keys())

    # 진입일 찾기 (당일 또는 그 이후 첫 영업일)
    entry_cum = None
    actual_entry = None
    for d in dates_sorted:
        if d >= entry_date:
            entry_cum = cum_table[d]
            actual_entry = d
            break

    if entry_cum is None:
        return {"error": f"진입일 {entry_date} 이후 백테스트 데이터가 없습니다."}

    latest_date = dates_sorted[-1]
    latest_cum = cum_table[latest_date]
    period_return = latest_cum / entry_cum - 1
    current_krw = int(entry_krw * latest_cum / entry_cum)
    profit_krw = current_krw - entry_krw

    return {
        "entry_date":       actual_entry,
        "entry_krw":        entry_krw,
        "latest_date":      latest_date,
        "entry_cum":        round(entry_cum, 6),
        "latest_cum":       round(latest_cum, 6),
        "period_return_pct": round(period_return * 100, 2),
        "current_krw":      current_krw,
        "profit_krw":       profit_krw,
    }


# ────────────────────────────────────────────
# VIX 기반 현금 비중 조정
# ────────────────────────────────────────────
def _vix_cash_pct(vix: float) -> float:
    """
    VIX 수준에 따른 추가 현금 비중.
    VIX < 20  → 0~10%   (선형)
    VIX 20~30 → 10~30%  (선형 보간)
    VIX > 30  → 30~50%  (선형, 최대 50%)
    """
    if vix < 20:
        return vix / 20 * 0.10
    elif vix <= 30:
        return 0.10 + (vix - 20) / 10 * 0.20
    else:
        return min(0.50, 0.30 + (vix - 30) / 30 * 0.20)


# ════════════════════════════════════════════
# ■ 신호 계산 메인 함수 (CLI + FastAPI 공용)
# ════════════════════════════════════════════

def run_signal() -> dict:
    """
    Markowitz 포트폴리오 신호를 계산하고 출력·저장.
    반환값 dict는 FastAPI /api/signal 등에서 활용 가능.
    """
    print("=" * 62)
    print(f"  Markowitz Portfolio Signal  [{date.today()}]")
    print(f"  Assets: S&P500 / NASDAQ / SOX")
    print(f"  Portfolio: {PORTFOLIO_KRW:,}원")
    print(f"  Weight bounds: {MIN_WEIGHT*100:.0f}% ~ {MAX_WEIGHT*100:.0f}% per asset")
    print("=" * 62)

    # ────────────────────────────────────────
    # 1. 데이터 로드
    # ────────────────────────────────────────
    print("\n[1] Downloading latest data...")
    ds      = build_dataset(end=str(date.today()))
    prices  = ds["prices"]
    ret_all = prices[ASSETS].pct_change().dropna()
    ret_1y  = ret_all.iloc[-252:]
    ret_6m  = ret_all.iloc[-126:]

    validation_msg, _validation_ret = validate_last_signal(prices)
    if validation_msg:
        print(f"\n── Previous Signal Check ──────────────────────────────")
        for line in validation_msg.split("\n"):
            print(f"  {line}")

    # ────────────────────────────────────────
    # 2. 자산별 지표
    # ────────────────────────────────────────
    print(f"\n── Individual Asset Metrics (Rolling 1Y) ─────────────")
    print(f"  {'Asset':<26} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Vol':>8} {'1M%':>7} {'3M%':>7}")
    print("  " + "-" * 59)

    for a in ASSETS:
        r   = ret_1y[a]
        sh  = sharpe_ratio(r)
        c   = cagr(r) * 100
        mdd = max_drawdown(r) * 100
        vol = r.std() * np.sqrt(252) * 100
        m1  = float((prices[a].iloc[-1] / prices[a].iloc[-22] - 1) * 100)
        m3  = float((prices[a].iloc[-1] / prices[a].iloc[-63] - 1) * 100)
        print(f"  {ASSET_NAMES[a]:<26} {sh:>7.3f} {c:>6.1f}% {mdd:>6.1f}% {vol:>7.1f}% {m1:>+6.1f}% {m3:>+6.1f}%")

    # ────────────────────────────────────────
    # 3. Markowitz 최적 비중
    # ────────────────────────────────────────
    w_1y, sh_1y = optimal_weights(ret_1y)
    w_6m, sh_6m = optimal_weights(ret_6m)
    stats_1y    = portfolio_stats(w_1y, ret_1y)
    stats_6m    = portfolio_stats(w_6m, ret_6m)

    print(f"\n── Optimal Weights (bounds: {MIN_WEIGHT*100:.0f}%~{MAX_WEIGHT*100:.0f}%) ──────────────────")
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

    # ────────────────────────────────────────
    # 4. 시장 전반 상태
    # ────────────────────────────────────────
    vix_now = float(prices["vix"].iloc[-1])
    vix_avg = float(prices["vix"].iloc[-252:].mean())
    port_1m = sum(w_1y[i] * float((prices[a].iloc[-1] / prices[a].iloc[-22] - 1) * 100)
                  for i, a in enumerate(ASSETS))
    port_3m = sum(w_1y[i] * float((prices[a].iloc[-1] / prices[a].iloc[-63] - 1) * 100)
                  for i, a in enumerate(ASSETS))

    print(f"\n── Market Context ─────────────────────────────────────")
    vix_label = ("(극도 안정 — 과열 주의)" if vix_now < 15
                 else "(정상 수준)" if vix_now < 20
                 else "(불안 구간)" if vix_now < 30 else "(공포 구간 ⚠)")
    print(f"  VIX now / 1Y avg   : {vix_now:.2f} / {vix_avg:.2f}  {vix_label}")
    print(f"  Portfolio 1M return: {port_1m:+.2f}%")
    print(f"  Portfolio 3M return: {port_3m:+.2f}%")

    # ────────────────────────────────────────
    # 5. 신호 조건
    # ────────────────────────────────────────
    conds = {
        "Portfolio Sharpe (1Y) > 0.5": sh_1y > 0.5,
        "Portfolio 1M momentum > 0"  : port_1m > 0,
        "Portfolio 3M momentum > 0"  : port_3m > 0,
        "VIX < 25 (no fear)"         : vix_now < 25,
        "Min asset CAGR > 0"         : all(cagr(ret_1y[a]) > 0 for a in ASSETS),
    }
    passed = sum(conds.values())

    if passed >= 4:
        signal_code = "BUY"
        signal_line = "🟢  BUY / HOLD"
        suggest     = w_1y
    elif passed == 3:
        signal_code = "CAUTION"
        signal_line = "🟡  CAUTION — 축소 보유 (비중 절반)"
        suggest     = w_1y * 0.5
    else:
        signal_code = "AVOID"
        signal_line = "🔴  AVOID — 현금 보유"
        suggest     = np.zeros(3)

    # VIX 기반 현금 비중 동적 조정 (AVOID 제외)
    _vix_adj = _vix_cash_pct(vix_now)
    if signal_code != "AVOID" and _vix_adj > 0.01:
        suggest = suggest * (1 - _vix_adj)
        print(f"\n  ⚠️  VIX {vix_now:.1f} → 현금 비중 +{_vix_adj*100:.0f}% 추가 적용 (위험 조정)")

    cash_pct = max(0.0, 1 - suggest.sum())

    print(f"\n── Signal Conditions ──────────────────────────────────")
    for cond, ok in conds.items():
        print(f"  [{'✓' if ok else '✗'}] {cond}")
    print(f"\n  Passed: {passed}/5")

    print(f"\n{'='*62}")
    print(f"  SIGNAL: {signal_line}")
    print(f"\n  Recommended allocation:")
    for i, a in enumerate(ASSETS):
        bar = "█" * int(suggest[i] * 30)
        print(f"    {ASSET_NAMES[a]:<26} {suggest[i]*100:>5.1f}%  {bar}")
    print(f"    {'CASH':<26} {cash_pct*100:>5.1f}%")

    # ────────────────────────────────────────
    # 6. 포트폴리오 금액 배분
    # ────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PORTFOLIO ALLOCATION  ({PORTFOLIO_KRW:,}원 기준)")
    print(f"{'='*62}")
    print(f"  {'자산':<26} {'비중':>6}  {'금액':>14}")
    print("  " + "-" * 50)
    for i, a in enumerate(ASSETS):
        amt = int(suggest[i] * PORTFOLIO_KRW)
        print(f"  {ASSET_NAMES[a]:<26} {suggest[i]*100:>5.1f}%  {amt:>12,}원")
    print(f"  {'현금(CASH)':<26} {cash_pct*100:>5.1f}%  {int(cash_pct * PORTFOLIO_KRW):>12,}원")
    print("  " + "-" * 50)
    print(f"  {'합계':<26} {'100.0%':>6}  {PORTFOLIO_KRW:>12,}원")

    # ────────────────────────────────────────
    # 7. 행동 강령
    # ────────────────────────────────────────
    sox_1m     = float((prices["sox"].iloc[-1] / prices["sox"].iloc[-22] - 1) * 100)
    sox_ovheat = sox_1m > 20
    wt_shift   = any(abs(w_1y[i] - w_6m[i]) > 0.15 for i in range(3))

    if vix_now > 30:   risk_level = "HIGH"
    elif vix_now > 20: risk_level = "ELEVATED"
    else:              risk_level = "NORMAL"

    print(f"\n{'='*62}")
    print(f"  ACTION PLAN — 오늘 해야 할 일")
    print(f"{'='*62}")

    print(f"\n  [시장 상태]")
    print(f"  • 리스크 레벨: {risk_level}  (VIX {vix_now:.1f})")
    if sox_ovheat:
        print(f"  • SOX 단기 과열: 1M {sox_1m:+.1f}% — 추격매수 주의")
    if wt_shift:
        print(f"  • 최적 비중 변화 감지 (6M vs 1Y 차이 큼) — 리밸런싱 검토")

    print(f"\n  [손절 기준 — 개별 자산 -{STOP_LOSS_PCT*100:.0f}% 이탈 시 재검토]")
    for i, a in enumerate(ASSETS):
        if suggest[i] > 0.01:
            px_now  = float(prices[a].iloc[-1])
            px_stop = px_now * (1 - STOP_LOSS_PCT)
            loss_krw = int(suggest[i] * PORTFOLIO_KRW * STOP_LOSS_PCT)
            print(f"  • {ASSET_NAMES[a]}: 현재 ${px_now:,.1f} → 손절가 ${px_stop:,.1f}  "
                  f"(최대 손실 약 {loss_krw:,}원)")

    print(f"\n  [신규 진입 — 포지션 없는 경우]")
    if passed >= 4:
        invest_amt = int((1 - cash_pct) * PORTFOLIO_KRW)
        print(f"  ① 총 {int((1-cash_pct)*100)}% 투자  → {invest_amt:,}원")
        for i, a in enumerate(ASSETS):
            if suggest[i] > 0.01:
                print(f"     - {ASSET_NAMES[a]}: {int(suggest[i]*PORTFOLIO_KRW):,}원")
        if sox_ovheat:
            print(f"  ② SOX 과열 중 → 2~3일 분할 매수 권고")
        else:
            print(f"  ② 오늘 바로 진입 가능 (모멘텀·VIX 정상)")
        print(f"  ③ 손절 -{STOP_LOSS_PCT*100:.0f}% 이탈 시 즉시 재검토")
        print(f"  ④ 다음 신호: 내일 오전 9시 (또는 uv run python today_signal.py)")
    elif passed == 3:
        print(f"  ① 최대 50%만 진입 → {int(0.5*PORTFOLIO_KRW):,}원 이하")
        print(f"  ② 나머지 {int(0.5*PORTFOLIO_KRW):,}원 현금 대기, 다음 주 재평가")
        print(f"  ③ 추가 조건 충족 시 비중 확대")
    else:
        print(f"  ① 진입하지 마세요 → 현금 {PORTFOLIO_KRW:,}원 전액 보유")
        print(f"  ② 조건 3개 이상 충족 시까지 대기")
        print(f"  ③ 단기채/예금으로 현금 운용 고려")

    print(f"\n  [기존 보유자]")
    if passed >= 4:
        if wt_shift:
            print(f"  ① 리밸런싱 필요 (6M vs 1Y 비중 불일치)")
            for i, a in enumerate(ASSETS):
                diff = (w_1y[i] - w_6m[i]) * 100
                if abs(diff) > 5:
                    direction = "늘리기" if diff > 0 else "줄이기"
                    diff_krw  = int(abs(diff) / 100 * PORTFOLIO_KRW)
                    print(f"     - {ASSET_NAMES[a]}: {direction}  ({diff:+.0f}%p, 약 {diff_krw:,}원)")
        else:
            print(f"  ① 현재 비중 유지, 리밸런싱 불필요")
        print(f"  ② 월말 비중 재계산 후 5%p 이상 차이 시 조정")
        print(f"  ③ 수익 실현 기준: 단일 자산 +30% 이상 시 익절 검토")
    elif passed == 3:
        print(f"  ① 수익 난 자산 일부 익절 고려")
        print(f"  ② 총 비중 50% 이하로 축소 → 절반 매도")
        half_amt = int(0.5 * PORTFOLIO_KRW)
        print(f"     목표: 투자금 {half_amt:,}원, 현금 {half_amt:,}원")
    else:
        print(f"  ① 전량 매도 후 현금화  → 목표: {PORTFOLIO_KRW:,}원 현금")
        print(f"  ② VIX < 20 + 모멘텀 회복 확인 후 재진입")
        print(f"  ③ 급락 시 분할 매수 기회 탐색")

    print(f"\n  [공통 주의사항]")
    print(f"  • 이 신호는 Markowitz 이론 기반 — 단기 급등락 예측 불가")
    print(f"  • 14Y 백테스트 CAGR: Markowitz 14.5% / S&P500 11.7%")
    if sh_1y > 2.0:
        print(f"  • Sharpe {sh_1y:.2f} → 비정상 강세 구간. 과거 패턴 지속 보장 없음")
    print(f"  • 거래 수수료 미반영. 월 1회 초과 리밸런싱 시 비용 손실 주의")
    print(f"  • 환율 미반영 — 실제 원화 투자 시 달러 환헤지 고려")
    print(f"  • 이 신호는 참고용이며 투자 손실에 대한 책임은 본인에게 있음")

    # ────────────────────────────────────────
    # 8. 히스토리 저장 + Discord 알림
    # ────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  SAVE & NOTIFY")
    print(f"{'='*62}")
    save_signal(signal_code, w_1y, sh_1y, vix_now, passed, prices)

    _action_lines = [f"리스크: **{risk_level}** (VIX {vix_now:.1f})"]
    if sox_ovheat:
        _action_lines.append(f"⚠️ SOX 단기 과열 {sox_1m:+.1f}% — 분할매수 권고")
    if wt_shift:
        _action_lines.append("⚠️ 6M vs 1Y 비중 차이 큼 — 리밸런싱 검토")
    if passed >= 4:
        invest_amt = int((1 - cash_pct) * PORTFOLIO_KRW)
        _action_lines.append(f"**신규:** {int((1-cash_pct)*100)}% 투자 → {invest_amt:,}원")
        _action_lines.append("SOX 과열 중 → 2~3일 분할매수" if sox_ovheat else "오늘 바로 진입 가능")
        _action_lines.append(f"손절 기준: -{STOP_LOSS_PCT*100:.0f}% 이탈 시 재검토")
    elif passed == 3:
        _action_lines.append(f"**신규:** 최대 50%만 진입 → {int(0.5*PORTFOLIO_KRW):,}원 이하")
    else:
        _action_lines.append("**신규:** 진입 금지 — 전액 현금 보유")
    action_msg = "\n".join(_action_lines)

    send_discord(signal_code, signal_line, suggest,
                 sh_1y, vix_now, passed, port_1m, validation_msg, action_msg)
    print(f"{'='*62}\n")

    return {
        "date":        str(date.today()),
        "signal_code": signal_code,
        "signal_line": signal_line,
        "w_1y":        w_1y.tolist(),
        "w_6m":        w_6m.tolist(),
        "suggest":     suggest.tolist(),
        "cash_pct":    float(cash_pct),
        "sharpe_1y":   float(sh_1y),
        "sharpe_6m":   float(sh_6m),
        "vix":         float(vix_now),
        "passed":      int(passed),
        "conditions":  {k: bool(v) for k, v in conds.items()},
        "prices":      {a: float(prices[a].iloc[-1]) for a in ASSETS},
        "stats_1y":    stats_1y,
        "stats_6m":    stats_6m,
        "port_1m":     float(port_1m),
        "port_3m":     float(port_3m),
        "risk_level":  risk_level,
        "portfolio_krw": PORTFOLIO_KRW,
    }


# ════════════════════════════════════════════
# ■ 진입점
# ════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Markowitz Portfolio Signal Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  uv run python today_signal.py
  uv run python today_signal.py --entry-date 2024-01-01 --entry-krw 10000000
        """
    )
    parser.add_argument(
        "--entry-date", type=str, default=None,
        metavar="YYYY-MM-DD",
        help="Markowitz 전략으로 진입한 날짜 (backtest_result.json 필요)"
    )
    parser.add_argument(
        "--entry-krw", type=int, default=None,
        metavar="금액",
        help="진입 금액 (원), --entry-date와 함께 사용"
    )
    args = parser.parse_args()

    # 오늘 신호 계산
    run_signal()

    # 진입일 수익률 계산 (두 인자 모두 주어진 경우)
    if args.entry_date is not None and args.entry_krw is not None:
        result = compute_entry_return(args.entry_date, args.entry_krw)
        print(f"\n{'='*62}")
        print(f"  진입 시뮬레이션 (Markowitz 누적수익률 기준)")
        print(f"{'='*62}")
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            pct    = result["period_return_pct"]
            profit = result["profit_krw"]
            sign   = "+" if pct >= 0 else ""
            p_sign = "+" if profit >= 0 else ""
            print(f"  진입일   : {result['entry_date']}")
            print(f"  현재일   : {result['latest_date']}")
            print(f"  진입금액 : {result['entry_krw']:,}원")
            print(f"  수익률   : {sign}{pct:.2f}%")
            print(f"  현재금액 : {result['current_krw']:,}원")
            print(f"  손익금액 : {p_sign}{profit:,}원")
        print(f"{'='*62}\n")
    elif (args.entry_date is None) != (args.entry_krw is None):
        print("  ⚠️  --entry-date와 --entry-krw는 함께 사용해야 합니다.\n")
