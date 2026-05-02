"""
discord_notify.py — 조건부 Discord 알림

전송 조건 (하나라도 해당하면 전송):
  1. 신호 코드 변경   : BUY / CAUTION / AVOID 전환
  2. VIX 기준선 교차  : 20 또는 30 을 넘거나 내려올 때
  3. 월말 정기 보고   : 해당 월의 마지막 거래일
  4. 비중 큰 변화     : sp500/nasdaq/sox/cash 중 하나라도 ±5%p 이상

비교 기준: signal_history.csv 의 직전 행 (오늘 행 제외)

cron (매일 오전 9시):
  0 9 * * * cd /home/dullear/aicoursework/AI-ML-midterm && \
    uv run python today_signal.py && \
    uv run python discord_notify.py

사용법:
  uv run python discord_notify.py          # 조건 판단 후 전송
  uv run python discord_notify.py --force  # 무조건 전송
"""

import argparse
import calendar
import csv
import sys
from datetime import date
from pathlib import Path

import requests

# ──────────────────────────────────────────────
#  설정
# ──────────────────────────────────────────────

import os as _os
WEBHOOK_URL = _os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "",   # 환경변수 미설정 시 전송 불가 — .env 또는 cron 환경에 설정 필요
)

BASE_DIR        = Path(__file__).parent
HISTORY_FILE    = BASE_DIR / "signal_history.csv"
PORTFOLIO_FILE  = BASE_DIR / "portfolio_state.csv"

REBALANCE_THRESHOLD_PCT = 5.0   # 리밸런싱 임계값 (%p)
WEIGHT_CHANGE_THRESHOLD = 5.0   # 비중 변화 알림 임계값 (%p)
VIX_LEVELS              = [20.0, 30.0]  # 기준선 교차 체크


# ──────────────────────────────────────────────
#  데이터 로드
# ──────────────────────────────────────────────

def _read_history() -> list[dict]:
    """signal_history.csv 전체 반환 (최신 행이 마지막)."""
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE) as f:
        return list(csv.DictReader(f))


def load_latest_signal() -> dict | None:
    rows = _read_history()
    return rows[-1] if rows else None


def load_prev_signal() -> dict | None:
    """오늘 날짜를 제외한 가장 최근 행 반환."""
    rows = _read_history()
    today_str = str(date.today())
    # 오늘 날짜 행(today_signal.py가 방금 추가한 것) 제외
    prev_rows = [r for r in rows if not r["date"].startswith(today_str)]
    return prev_rows[-1] if prev_rows else None


def load_portfolio_state() -> dict | None:
    if not PORTFOLIO_FILE.exists():
        return None
    with open(PORTFOLIO_FILE) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    return {
        "date":       row.get("date", ""),
        "total_krw":  int(float(row.get("total_krw", 0))),
        "sp500_pct":  float(row.get("sp500_pct", 0)),
        "nasdaq_pct": float(row.get("nasdaq_pct", 0)),
        "sox_pct":    float(row.get("sox_pct", 0)),
        "cash_pct":   float(row.get("cash_pct", 0)),
    }


# ──────────────────────────────────────────────
#  전송 여부 판단
# ──────────────────────────────────────────────

def _is_last_trading_day_of_month() -> bool:
    """오늘이 해당 월의 마지막 거래일(월~금)인지 확인."""
    today = date.today()
    last_day = calendar.monthrange(today.year, today.month)[1]
    # 마지막 날부터 역으로 탐색해 첫 번째 평일(월~금)을 찾음
    for d in range(last_day, 0, -1):
        candidate = today.replace(day=d)
        if candidate.weekday() < 5:   # 0=월 … 4=금
            return today == candidate
    return False


def _vix_crossed(vix_now: float, vix_prev: float) -> str | None:
    """
    VIX 기준선(20, 30) 교차 여부.
    넘어올 때 또는 내려올 때 모두 감지.
    반환: 설명 문자열 또는 None
    """
    for level in VIX_LEVELS:
        was_above = vix_prev >= level
        now_above = vix_now  >= level
        if was_above != now_above:
            direction = "상향 돌파" if now_above else "하향 복귀"
            return f"VIX {level:.0f} {direction} ({vix_prev:.1f} → {vix_now:.1f})"
    return None


def _weight_changed(sig_now: dict, sig_prev: dict) -> list[str]:
    """추천 비중 ±5%p 이상 변화한 항목 반환."""
    keys = [("sp500_w", "S&P500"), ("nasdaq_w", "NASDAQ"),
            ("sox_w", "SOX"), ("cash_w", "CASH")]
    changes = []
    for col, name in keys:
        now  = float(sig_now.get(col, 0)) * 100
        prev = float(sig_prev.get(col, 0)) * 100
        diff = now - prev
        if abs(diff) >= WEIGHT_CHANGE_THRESHOLD:
            sign = "+" if diff > 0 else ""
            changes.append(f"{name} {sign}{diff:.0f}%p ({prev:.0f}%→{now:.0f}%)")
    return changes


def should_send(sig: dict, prev: dict | None,
                force: bool = False) -> tuple[bool, list[str]]:
    """
    전송 여부와 사유 목록 반환.
    Returns: (send: bool, reasons: list[str])
    """
    if force:
        return True, ["강제 전송 (--force)"]

    reasons = []

    # 이전 신호 없음 → 첫 실행, 전송
    if prev is None:
        return True, ["첫 신호 기록"]

    vix_now  = float(sig.get("vix", 0))
    vix_prev = float(prev.get("vix", 0))

    # 조건 1: 신호 코드 변경
    if sig.get("signal") != prev.get("signal"):
        reasons.append(
            f"신호 변경: {prev['signal']} → {sig['signal']}"
        )

    # 조건 2: VIX 기준선 교차
    crossed = _vix_crossed(vix_now, vix_prev)
    if crossed:
        reasons.append(crossed)

    # 조건 3: 월말 정기 보고
    if _is_last_trading_day_of_month():
        reasons.append(f"월말 정기 보고 ({date.today().strftime('%Y-%m-%d')})")

    # 조건 4: 비중 큰 변화
    weight_changes = _weight_changed(sig, prev)
    if weight_changes:
        reasons.append("비중 변화: " + ", ".join(weight_changes))

    return bool(reasons), reasons


# ──────────────────────────────────────────────
#  리밸런싱 계산
# ──────────────────────────────────────────────

def calc_rebalance(signal: dict, portfolio: dict) -> list[dict]:
    rec = {
        "S&P500": float(signal.get("sp500_w",  0)) * 100,
        "NASDAQ": float(signal.get("nasdaq_w", 0)) * 100,
        "SOX":    float(signal.get("sox_w",    0)) * 100,
        "CASH":   float(signal.get("cash_w",   0)) * 100,
    }
    cur = {
        "S&P500": portfolio["sp500_pct"],
        "NASDAQ": portfolio["nasdaq_pct"],
        "SOX":    portfolio["sox_pct"],
        "CASH":   portfolio["cash_pct"],
    }
    total = portfolio["total_krw"]

    actions = []
    for asset in ["S&P500", "NASDAQ", "SOX", "CASH"]:
        diff = rec[asset] - cur[asset]
        if abs(diff) < REBALANCE_THRESHOLD_PCT:
            continue
        amount = int(total * abs(diff) / 100)
        actions.append({
            "asset":  asset,
            "cur":    cur[asset],
            "rec":    rec[asset],
            "diff":   diff,
            "amount": amount,
            "action": "매수" if diff > 0 else "매도",
        })
    return sorted(actions, key=lambda x: abs(x["diff"]), reverse=True)


# ──────────────────────────────────────────────
#  DCA 월간 매수 가이드
# ──────────────────────────────────────────────

def calc_dca_guide(signal: dict, portfolio: dict | None,
                   monthly_add: int = 1_000_000) -> str:
    rec = {
        "S&P500": float(signal.get("sp500_w",  0)) * 100,
        "NASDAQ": float(signal.get("nasdaq_w", 0)) * 100,
        "SOX":    float(signal.get("sox_w",    0)) * 100,
    }

    if portfolio is None or portfolio["total_krw"] == 0:
        lines = [f"**📅 이번 달 추가 매수 가이드** (추가 {monthly_add:,}원)"]
        rec_sum = sum(rec.values())
        for asset, pct in rec.items():
            if pct > 0:
                amt = int(monthly_add * pct / rec_sum) if rec_sum > 0 else 0
                lines.append(f"  • `{asset:<6}` **{amt:,}원** ({pct:.0f}% 비중 기준)")
        lines.append("  *포트폴리오 저장 시 갭 기반 정밀 계산 가능*")
        return "\n".join(lines)

    total     = portfolio["total_krw"]
    new_total = total + monthly_add
    cur = {
        "S&P500": portfolio["sp500_pct"],
        "NASDAQ": portfolio["nasdaq_pct"],
        "SOX":    portfolio["sox_pct"],
    }
    gaps = {}
    for asset in rec:
        gap = new_total * rec[asset] / 100 - total * cur[asset] / 100
        if gap > 0:
            gaps[asset] = gap

    if not gaps:
        return "**📅 이번 달 추가 매수 가이드**: 추천 비중 초과 보유 — 이번 달 추가 매수 불필요"

    total_gap = sum(gaps.values())
    lines = [f"**📅 이번 달 추가 매수 가이드** (추가 {monthly_add:,}원 / 총 {new_total:,}원)"]
    remaining = monthly_add
    for asset, gap in sorted(gaps.items(), key=lambda x: -x[1]):
        alloc = int(min(gap, monthly_add * gap / total_gap))
        alloc = min(alloc, remaining)
        remaining -= alloc
        if alloc > 0:
            lines.append(
                f"  📈 `{asset:<6}` **{alloc:,}원** 매수  "
                f"({cur[asset]:.0f}% → {rec[asset]:.0f}%)"
            )
    if remaining > 0:
        lines.append(f"  💵 잔여 {remaining:,}원 → 현금 보유")
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  메시지 조립
# ──────────────────────────────────────────────

SIGNAL_EMOJI = {"BUY": "🟢", "CAUTION": "🟡", "AVOID": "🔴"}
VIX_WARN_THRESHOLD   = 20.0
VIX_DANGER_THRESHOLD = 30.0

# embed 왼쪽 컬러 바
SIGNAL_COLOR = {"BUY": 0x4ade80, "CAUTION": 0xfacc15, "AVOID": 0xf87171}
VIX_COLOR    = {30: 0xef4444, 20: 0xf97316}   # danger / warn


def _embed_color(sig_code: str, vix: float) -> int:
    if vix >= VIX_DANGER_THRESHOLD:
        return VIX_COLOR[30]
    if vix >= VIX_WARN_THRESHOLD:
        return VIX_COLOR[20]
    return SIGNAL_COLOR.get(sig_code, 0x94a3b8)


def build_embed(signal: dict, prev: dict | None,
                portfolio: dict | None, actions: list[dict],
                send_reasons: list[str],
                monthly_add: int = 1_000_000) -> dict:
    """Discord embed dict 반환."""
    sig_code  = signal.get("signal", "?")
    sig_emoji = SIGNAL_EMOJI.get(sig_code, "⚪")
    sharpe    = float(signal.get("sharpe_1y", 0))
    vix       = float(signal.get("vix", 0))
    sig_date  = str(signal.get("date", ""))[:10]
    passed    = signal.get("passed", "?")
    today     = date.today()

    # 제목
    if vix >= VIX_DANGER_THRESHOLD:
        title = f"🚨 방어 모드 긴급 알림 — {sig_emoji} {sig_code}"
    elif vix >= VIX_WARN_THRESHOLD:
        title = f"⚠️ VIX 경고 — {sig_emoji} {sig_code}"
    else:
        title = f"📊 {today.month}월 포트폴리오 신호 — {sig_emoji} {sig_code}"

    # 전송 사유 (description)
    reasons_str = " · ".join(send_reasons)
    description = f"*{reasons_str}*"

    fields = []

    # ── 지표 요약 ──
    fields.append({
        "name": "📈 지표",
        "value": (
            f"Sharpe `{sharpe:.3f}` · VIX `{vix:.2f}`\n"
            f"조건 충족 `{passed}/5` · 기준일 `{sig_date}`"
        ),
        "inline": False,
    })

    # ── 전일 대비 변화 ──
    if prev is not None:
        prev_sig = prev.get("signal", "?")
        prev_vix = float(prev.get("vix", 0))
        change_parts = []
        if sig_code != prev_sig:
            change_parts.append(
                f"신호 {SIGNAL_EMOJI.get(prev_sig,'⚪')} {prev_sig} → {sig_emoji} {sig_code}"
            )
        vix_diff = vix - prev_vix
        vix_sign = "+" if vix_diff >= 0 else ""
        change_parts.append(f"VIX {prev_vix:.1f} → {vix:.1f} ({vix_sign}{vix_diff:.1f})")
        for wc in _weight_changed(signal, prev):
            change_parts.append(wc)
        fields.append({
            "name": "🔀 전일 대비",
            "value": "\n".join(change_parts),
            "inline": False,
        })

    # ── VIX 구간 경고 ──
    if vix >= VIX_DANGER_THRESHOLD:
        rec_cash = min(50, int(30 + (vix - 30) / 30 * 20))
        fields.append({
            "name": "🚨 VIX 공포 구간",
            "value": f"즉시 현금 비중 확대 검토\n현금 권고 `{rec_cash}%` (VIX {vix:.1f})",
            "inline": False,
        })
    elif vix >= VIX_WARN_THRESHOLD:
        rec_cash = int(10 + (vix - 20) / 10 * 20)
        fields.append({
            "name": "⚠️ VIX 불안 구간",
            "value": f"현금 비중 `{rec_cash}%` 권고 (VIX {vix:.1f})",
            "inline": False,
        })

    # ── 추천 비중 (inline 4개) ──
    for col, name in [("sp500_w", "S&P500"), ("nasdaq_w", "NASDAQ"),
                      ("sox_w", "SOX"), ("cash_w", "CASH")]:
        pct = float(signal.get(col, 0)) * 100
        fields.append({"name": f"🎯 {name}", "value": f"`{pct:.0f}%`", "inline": True})

    # ── 리밸런싱 ──
    if portfolio is None:
        fields.append({
            "name": "ℹ️ 리밸런싱",
            "value": "포트폴리오 미저장 — 웹 대시보드에서 현재 비중을 저장하세요",
            "inline": False,
        })
    elif not actions:
        fields.append({
            "name": "✅ 리밸런싱 불필요",
            "value": "모든 비중이 추천 범위 내 (±5%p)",
            "inline": False,
        })
    else:
        rebal_lines = []
        for a in actions:
            sign  = "+" if a["diff"] > 0 else "-"
            emoji = "📈" if a["diff"] > 0 else "📉"
            rebal_lines.append(
                f"{emoji} **{a['asset']}** {a['cur']:.0f}%→{a['rec']:.0f}%"
                f" ({sign}{abs(a['diff']):.0f}%p) **{a['action']} {a['amount']:,}원**"
            )
        footer_note = f"\n총 자산 {portfolio['total_krw']:,}원 (저장: {portfolio['date']})"
        fields.append({
            "name": "🔄 리밸런싱 필요",
            "value": "\n".join(rebal_lines) + footer_note,
            "inline": False,
        })

    # ── DCA (월말에만) ──
    if _is_last_trading_day_of_month():
        dca_text = calc_dca_guide(signal, portfolio, monthly_add)
        # embed field value 최대 1024자
        if len(dca_text) > 1020:
            dca_text = dca_text[:1020] + "…"
        fields.append({"name": "📅 적립식 매수 가이드", "value": dca_text, "inline": False})

    return {
        "title":       title,
        "description": description,
        "color":       _embed_color(sig_code, vix),
        "fields":      fields,
        "footer":      {"text": f"Markowitz 자동 신호 · {today}"},
    }


# ──────────────────────────────────────────────
#  전송
# ──────────────────────────────────────────────

def send_discord(embed: dict) -> bool:
    if not WEBHOOK_URL:
        print("❌ DISCORD_WEBHOOK_URL 환경변수가 설정되지 않았습니다.")
        return False
    try:
        resp = requests.post(
            WEBHOOK_URL,
            json={"embeds": [embed]},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"❌ Discord 전송 실패: {e}")
        return False


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discord 조건부 알림 전송")
    parser.add_argument("--force",       action="store_true", help="조건 무시하고 무조건 전송")
    parser.add_argument("--monthly-add", type=int, default=1_000_000,
                        help="이번 달 추가 투자금 (기본: 100만원)")
    args = parser.parse_args()

    signal = load_latest_signal()
    if signal is None:
        print("❌ signal_history.csv 없음. 먼저 today_signal.py를 실행하세요.")
        sys.exit(1)

    prev = load_prev_signal()
    do_send, reasons = should_send(signal, prev, force=args.force)

    vix_now = float(signal.get("vix", 0))
    print(f"VIX: {vix_now:.1f} | 전송: {'✅' if do_send else '⏭'}")
    if reasons:
        for r in reasons:
            print(f"  → {r}")
    else:
        print("  → 전송 조건 미충족, 건너뜀")

    if not do_send:
        return

    portfolio = load_portfolio_state()
    if portfolio is None:
        print("⚠ portfolio_state.csv 없음 — 리밸런싱 계산 생략")
        actions = []
    else:
        actions = calc_rebalance(signal, portfolio)

    embed = build_embed(signal, prev, portfolio, actions, reasons,
                        monthly_add=args.monthly_add)
    print("─── 전송 embed ───")
    import json as _json
    print(_json.dumps(embed, ensure_ascii=False, indent=2))
    print("──────────────────")

    if send_discord(embed):
        print("✅ Discord 전송 완료")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
