"""
main_app.py — Can ML Beat the Market? FastAPI 웹 서버
PNU 전산물리 중간 프로젝트

엔드포인트:
  GET  /              → 오늘 신호 대시보드 (Tailwind HTML)
  GET  /backtest      → 백테스트 결과 차트 페이지 (Chart.js)
  POST /simulate      → entry_date, entry_krw 받아서 수익률 JSON 반환
  GET  /api/signal    → 오늘 신호 JSON (signal_history.csv 최신 행)
  GET  /api/backtest  → backtest_result.json 반환

사전 준비:
  uv run python today_signal.py   # signal_history.csv 생성
  uv run python backtest.py       # backtest_result.json 생성

실행:
  uv run uvicorn main_app:app --reload --port 8000
"""

import sys, os, json
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR        = Path(__file__).parent
HISTORY_FILE    = BASE_DIR / "signal_history.csv"
BACKTEST_FILE   = BASE_DIR / "backtest_result.json"
PORTFOLIO_FILE  = BASE_DIR / "portfolio_state.csv"

app = FastAPI(title="Can ML Beat the Market?", version="1.0.0")


# ══════════════════════════════════════════════════════════
#  데이터 로더
# ══════════════════════════════════════════════════════════

def _load_latest_signal() -> dict | None:
    if not HISTORY_FILE.exists():
        return None
    df = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    row["date"] = str(row["date"])[:10]
    return row


def _load_backtest() -> dict | None:
    if not BACKTEST_FILE.exists():
        return None
    with open(BACKTEST_FILE) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
#  HTML 헬퍼
# ══════════════════════════════════════════════════════════

def _shell(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>{title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body {{ font-family: 'Inter', sans-serif; }}
    .mono {{ font-family: 'JetBrains Mono', 'Courier New', monospace; }}
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">
  <nav class="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center gap-6 text-sm sticky top-0 z-10">
    <span class="font-bold text-white">Can ML Beat the Market?</span>
    <a href="/" class="text-blue-400 hover:text-blue-300 transition-colors">Dashboard</a>
    <a href="/backtest" class="text-blue-400 hover:text-blue-300 transition-colors">Backtest</a>
    <span class="text-gray-600 ml-auto text-xs">PNU Computational Physics</span>
  </nav>
  <main class="max-w-5xl mx-auto px-4 py-8">
    {body}
  </main>
</body>
</html>"""


def _no_data_page(filename: str, command: str) -> str:
    return f"""<div class="text-center py-24 text-gray-500">
  <p class="text-5xl mb-4">📭</p>
  <p class="text-xl font-semibold text-gray-300">{filename} 없음</p>
  <p class="mt-3 text-sm">먼저 터미널에서 실행하세요:</p>
  <code class="mt-2 inline-block bg-gray-800 text-green-400 px-4 py-2 rounded text-sm">{command}</code>
</div>"""


def _bt_card(name: str, m: dict) -> str:
    cagr_cls  = "text-green-400" if m.get("cagr", 0) > 0 else "text-red-400"
    final_cls = "text-blue-400"  if m.get("final_return", 0) > 0 else "text-red-400"
    return f"""
<div class="bg-gray-900 rounded-2xl p-5 border border-gray-800 hover:border-gray-700 transition-colors">
  <h3 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">{name}</h3>
  <div class="space-y-2 text-sm">
    <div class="flex justify-between">
      <span class="text-gray-400">Sharpe</span>
      <span class="mono font-semibold">{m.get('sharpe', 0):.3f}</span>
    </div>
    <div class="flex justify-between">
      <span class="text-gray-400">MDD</span>
      <span class="mono font-semibold text-red-400">{m.get('mdd', 0)*100:.1f}%</span>
    </div>
    <div class="flex justify-between">
      <span class="text-gray-400">CAGR</span>
      <span class="mono font-semibold {cagr_cls}">{m.get('cagr', 0)*100:.1f}%</span>
    </div>
    <div class="flex justify-between border-t border-gray-800 pt-2 mt-2">
      <span class="text-gray-400">누적수익률</span>
      <span class="mono font-bold {final_cls}">{m.get('final_return', 0)*100:.1f}%</span>
    </div>
  </div>
</div>"""


# ══════════════════════════════════════════════════════════
#  API 엔드포인트
# ══════════════════════════════════════════════════════════

@app.get("/api/signal")
def api_signal():
    """오늘 신호 JSON (signal_history.csv 최신 행)."""
    sig = _load_latest_signal()
    if sig is None:
        return JSONResponse(
            {"error": "signal_history.csv 없음. today_signal.py를 먼저 실행하세요."},
            status_code=404
        )
    return sig


@app.get("/api/backtest")
def api_backtest():
    """backtest_result.json 전체 반환."""
    bt = _load_backtest()
    if bt is None:
        return JSONResponse(
            {"error": "backtest_result.json 없음. backtest.py를 먼저 실행하세요."},
            status_code=404
        )
    return bt


class PortfolioState(BaseModel):
    total_krw: int
    sp500_pct:  float
    nasdaq_pct: float
    sox_pct:    float
    cash_pct:   float


@app.get("/api/portfolio")
def api_portfolio_get():
    """마지막 저장된 포트폴리오 상태 반환."""
    if not PORTFOLIO_FILE.exists():
        return JSONResponse({"error": "저장된 포트폴리오 없음"}, status_code=404)
    df = pd.read_csv(PORTFOLIO_FILE)
    if df.empty:
        return JSONResponse({"error": "저장된 포트폴리오 없음"}, status_code=404)
    row = df.iloc[-1].to_dict()
    row["date"] = str(row.get("date", ""))[:10]
    return row


@app.post("/api/portfolio")
def api_portfolio_save(req: PortfolioState):
    """포트폴리오 상태 저장 (portfolio_state.csv에 추가)."""
    from datetime import date as _date
    new_row = {
        "date":       str(_date.today()),
        "total_krw":  req.total_krw,
        "sp500_pct":  req.sp500_pct,
        "nasdaq_pct": req.nasdaq_pct,
        "sox_pct":    req.sox_pct,
        "cash_pct":   req.cash_pct,
    }
    if PORTFOLIO_FILE.exists():
        df = pd.read_csv(PORTFOLIO_FILE)
    else:
        df = pd.DataFrame(columns=new_row.keys())
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PORTFOLIO_FILE, index=False)
    return {"saved": True, "date": new_row["date"]}


class SimulateRequest(BaseModel):
    entry_date: str   # YYYY-MM-DD
    entry_krw:  int   # 원화 금액


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    진입일·금액으로 전략별 수익률 계산.
    backtest_result.json의 cumulative_returns 테이블 사용.
    """
    bt = _load_backtest()
    if bt is None:
        return JSONResponse({"error": "backtest_result.json 없음"}, status_code=404)

    cum_data = bt.get("cumulative_returns", {})
    strategies_out = {}

    for key in ["markowitz", "markowitz_momentum", "buy_and_hold", "kelly_mlp"]:
        if key not in cum_data:
            continue
        table = cum_data[key]
        dates_sorted = sorted(table.keys())

        entry_cum = None
        actual_entry = None
        for d in dates_sorted:
            if d >= req.entry_date:
                entry_cum = table[d]
                actual_entry = d
                break

        if entry_cum is None:
            continue

        latest_date   = dates_sorted[-1]
        latest_cum    = table[latest_date]
        period_return = latest_cum / entry_cum - 1
        current_krw   = int(req.entry_krw * latest_cum / entry_cum)

        strategies_out[key] = {
            "entry_date":        actual_entry,
            "latest_date":       latest_date,
            "period_return_pct": round(period_return * 100, 2),
            "multiplier":        round(latest_cum / entry_cum, 3),
            "current_krw":       current_krw,
            "profit_krw":        current_krw - req.entry_krw,
        }

    if not strategies_out:
        return JSONResponse(
            {"error": f"진입일 {req.entry_date} 이후 백테스트 데이터가 없습니다. (데이터 기간: 2010-01-01 ~ 2025-12-31)"},
            status_code=400
        )

    return {"entry_krw": req.entry_krw, "strategies": strategies_out}


# ══════════════════════════════════════════════════════════
#  HTML 페이지
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """오늘 신호 대시보드."""
    sig = _load_latest_signal()

    if sig is None:
        body = _no_data_page("signal_history.csv", "uv run python today_signal.py")
        return HTMLResponse(_shell("Dashboard", body))

    sig_code = str(sig.get("signal", ""))
    badge = {
        "BUY":     ("🟢 BUY / HOLD",  "border-green-700 bg-green-950",  "text-green-300"),
        "CAUTION": ("🟡 CAUTION",      "border-yellow-700 bg-yellow-950","text-yellow-300"),
        "AVOID":   ("🔴 AVOID",        "border-red-700 bg-red-950",      "text-red-300"),
    }.get(sig_code, (sig_code, "border-gray-700 bg-gray-900", "text-gray-300"))
    sig_text, banner_cls, text_cls = badge

    sharpe = float(sig.get("sharpe_1y", 0))
    vix    = float(sig.get("vix", 0))
    passed = int(sig.get("passed", 0))
    sig_date = str(sig.get("date", ""))[:10]

    # VIX 구간 계산
    if vix < 20:
        _vix_zone      = "정상"
        _vix_zone_cls  = "text-green-400"
        _vix_bar_cls   = "bg-green-500"
        _vix_cash_adj  = vix / 20 * 10
        _vix_desc      = "시장 안정 — 정상 운용"
    elif vix <= 30:
        _vix_zone      = "경계"
        _vix_zone_cls  = "text-yellow-400"
        _vix_bar_cls   = "bg-yellow-500"
        _vix_cash_adj  = 10 + (vix - 20) / 10 * 20
        _vix_desc      = "불안 구간 — 현금 비중 증가 적용 중"
    else:
        _vix_zone      = "위험"
        _vix_zone_cls  = "text-red-400"
        _vix_bar_cls   = "bg-red-500"
        _vix_cash_adj  = min(50, 30 + (vix - 30) / 30 * 20)
        _vix_desc      = "공포 구간 🚨 즉시 방어 모드"
    _vix_bar_pct = min(100, int(vix / 50 * 100))  # 0~50 범위를 0~100%로
    _z1_cls = "bg-green-950 border-green-800"  if vix < 20          else "bg-gray-800 border-gray-700"
    _z2_cls = "bg-yellow-950 border-yellow-800" if 20 <= vix <= 30  else "bg-gray-800 border-gray-700"
    _z3_cls = "bg-red-950 border-red-800"       if vix > 30         else "bg-gray-800 border-gray-700"

    sp500_w  = float(sig.get("sp500_w", 0))
    nasdaq_w = float(sig.get("nasdaq_w", 0))
    sox_w    = float(sig.get("sox_w", 0))
    cash_w   = float(sig.get("cash_w", 0))

    assets_data = [
        ("S&P500", sp500_w,  float(sig.get("sp500_px",  0))),
        ("NASDAQ", nasdaq_w, float(sig.get("nasdaq_px", 0))),
        ("SOX",    sox_w,    float(sig.get("sox_px",    0))),
        ("CASH",   cash_w,   0.0),
    ]

    weight_rows = ""
    for name, w, _ in assets_data:
        bar_pct = int(w * 100)
        bar_color = "bg-blue-500" if name != "CASH" else "bg-gray-600"
        weight_rows += f"""
        <div class="flex items-center gap-3 py-1.5">
          <span class="w-16 text-xs text-gray-400 shrink-0">{name}</span>
          <div class="flex-1 bg-gray-800 rounded-full h-2.5">
            <div class="{bar_color} h-2.5 rounded-full transition-all" style="width:{bar_pct}%"></div>
          </div>
          <span class="w-14 text-right mono text-sm font-semibold">{w*100:.1f}%</span>
        </div>"""

    price_rows = ""
    for name, _, px in assets_data[:3]:
        price_rows += f"""
        <div class="flex justify-between text-sm py-1.5 border-b border-gray-800 last:border-0">
          <span class="text-gray-400">{name}</span>
          <span class="mono font-semibold">${px:,.2f}</span>
        </div>"""

    cond_dots = "".join(
        f'<span class="inline-block w-2 h-2 rounded-full {"bg-green-500" if i < passed else "bg-gray-700"}"></span>'
        for i in range(5)
    )

    # 추천 비중을 JS 변수로 임베드 (리밸런싱 계산기용)
    rec_weights_js = json.dumps({
        "S&P500": round(sp500_w * 100, 1),
        "NASDAQ": round(nasdaq_w * 100, 1),
        "SOX":    round(sox_w   * 100, 1),
        "CASH":   round(cash_w  * 100, 1),
    })

    body = f"""
    <div class="mb-8">
      <h1 class="text-3xl font-bold">Signal Dashboard</h1>
      <p class="text-gray-500 mt-1 text-sm">Markowitz Portfolio Optimization · S&P500 / NASDAQ / SOX</p>
    </div>

    <!-- Signal Banner -->
    <div class="rounded-2xl border {banner_cls} p-6 mb-6">
      <div class="flex items-start justify-between flex-wrap gap-4">
        <div>
          <p class="text-4xl font-black {text_cls}">{sig_text}</p>
          <p class="text-gray-500 mt-2 text-sm">
            기준일: <span class="text-gray-300">{sig_date}</span>
            &ensp;|&ensp; 조건 충족: <span class="text-gray-300">{passed}/5</span>
            &ensp; {cond_dots}
          </p>
        </div>
        <div class="text-right">
          <p class="text-3xl mono font-black">{sharpe:.3f}</p>
          <p class="text-gray-500 text-xs mt-1">Sharpe (1Y)</p>
          <p class="text-gray-400 text-sm mt-2">VIX <span class="text-white font-semibold">{vix:.2f}</span></p>
        </div>
      </div>
    </div>

    <!-- VIX 방어 상태 카드 -->
    <div class="bg-gray-900 rounded-2xl border border-gray-800 p-5 mb-6">
      <div class="flex items-start justify-between flex-wrap gap-4 mb-3">
        <div>
          <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider">VIX 방어 상태</h2>
          <p class="text-xs text-gray-600 mt-0.5">현재 시장 공포 지수 → 자동 현금 비중 조정</p>
        </div>
        <span class="text-xs font-bold {_vix_zone_cls} bg-gray-800 rounded-full px-3 py-1 border border-gray-700">
          {_vix_zone} 구간 · VIX {vix:.2f}
        </span>
      </div>

      <!-- VIX 게이지 바 -->
      <div class="relative mb-3">
        <div class="flex w-full h-3 rounded-full overflow-hidden bg-gray-800">
          <div class="h-full bg-green-600" style="width:40%"></div>
          <div class="h-full bg-yellow-600" style="width:20%"></div>
          <div class="h-full bg-red-700" style="flex:1"></div>
        </div>
        <!-- 현재 VIX 위치 마커 -->
        <div class="absolute top-0 h-3 w-0.5 bg-white" style="left:{_vix_bar_pct}%"></div>
        <!-- 구간 레이블 -->
        <div class="flex justify-between text-xs text-gray-600 mt-1">
          <span>0</span>
          <span class="text-green-600">VIX 20</span>
          <span class="text-yellow-600">VIX 30</span>
          <span class="text-red-600">50+</span>
        </div>
      </div>

      <!-- 구간별 현금 비중 기준 -->
      <div class="grid grid-cols-3 gap-3 mb-3">
        <div class="text-center p-2 rounded-lg border {_z1_cls}">
          <p class="text-xs text-green-400 font-semibold">VIX &lt; 20</p>
          <p class="text-xs text-gray-400">현금 0~10%</p>
          <p class="text-xs text-gray-600">정상 운용</p>
        </div>
        <div class="text-center p-2 rounded-lg border {_z2_cls}">
          <p class="text-xs text-yellow-400 font-semibold">VIX 20~30</p>
          <p class="text-xs text-gray-400">현금 10~30%</p>
          <p class="text-xs text-gray-600">선형 보간</p>
        </div>
        <div class="text-center p-2 rounded-lg border {_z3_cls}">
          <p class="text-xs text-red-400 font-semibold">VIX &gt; 30</p>
          <p class="text-xs text-gray-400">현금 30~50%</p>
          <p class="text-xs text-gray-600">방어 모드 🚨</p>
        </div>
      </div>

      <div class="flex items-center justify-between text-sm">
        <span class="{_vix_zone_cls}">{_vix_desc}</span>
        <span class="mono font-bold {_vix_zone_cls}">현금 비중 +{_vix_cash_adj:.0f}% 자동 적용</span>
      </div>
    </div>

    <!-- Weights + Prices -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
      <div class="bg-gray-900 rounded-2xl p-5 border border-gray-800">
        <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">추천 비중 (1Y window)</h2>
        {weight_rows}
      </div>
      <div class="bg-gray-900 rounded-2xl p-5 border border-gray-800">
        <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">현재 자산 가격</h2>
        {price_rows}
        <a href="/backtest" class="mt-4 block text-center text-xs text-blue-400 hover:text-blue-300 transition-colors py-2 border border-gray-700 rounded-lg">
          백테스트 결과 보기 →
        </a>
      </div>
    </div>

    <!-- 진입일 수익률 계산 + 72의 법칙 + D-day -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">

      <!-- 진입일 수익률 계산기 -->
      <div class="md:col-span-2 bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-5">진입일 수익률 계산</h2>
        <form id="simForm" class="flex flex-wrap gap-4 items-end">
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">진입일 (YYYY-MM-DD)</label>
            <input type="text" id="sim_date" placeholder="2022-01-01"
              class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500 transition-colors mono"/>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1.5">금액 (원)</label>
            <input type="number" id="sim_krw" value="10000000"
              class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500 transition-colors"/>
          </div>
          <button type="submit"
            class="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-2 rounded-lg text-sm transition-colors">
            계산
          </button>
        </form>
        <div id="simResult" class="mt-5 hidden"></div>
      </div>

      <!-- 72의 법칙 + 다음 리밸런싱 D-day -->
      <div class="flex flex-col gap-5">
        <!-- 72의 법칙 -->
        <div class="bg-gray-900 rounded-2xl p-5 border border-gray-800">
          <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">72의 법칙</h2>
          <p class="text-xs text-gray-600 mb-3">연수익률 입력 → 원금 2배 기간</p>
          <div class="flex gap-2 items-center">
            <input type="number" id="rule72_rate" value="14.5" step="0.1" min="0.1"
              class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-24 focus:outline-none focus:border-blue-500 transition-colors mono"
              oninput="calcRule72()"/>
            <span class="text-gray-400 text-sm">%</span>
          </div>
          <div id="rule72_result" class="mt-3"></div>
        </div>

        <!-- 다음 리밸런싱 D-day -->
        <div class="bg-gray-900 rounded-2xl p-5 border border-gray-800">
          <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-1">리밸런싱 D-day</h2>
          <p class="text-xs text-gray-600 mb-3">매주 월요일 기준</p>
          <div id="dday_result" class="text-3xl font-black mono text-blue-400"></div>
          <p id="dday_date" class="text-xs text-gray-500 mt-1"></p>
        </div>
      </div>
    </div>

    <!-- 리밸런싱 계산기 -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 mb-6">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-1">리밸런싱 계산기</h2>
      <p class="text-xs text-gray-600 mb-5">현재 보유 비중 입력 → 추천 비중까지 매수/매도 금액 계산 (서버 통신 없이 즉시 계산)</p>
      <div class="flex flex-wrap gap-4 items-end mb-5">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">총 자산 (원)</label>
          <input type="number" id="rb_total" value="10000000"
            class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500 transition-colors"/>
        </div>
        <p class="text-xs text-gray-600 self-center pb-1">합계가 100%여야 계산 가능</p>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4" id="rb_inputs">
        <!-- JS로 생성 -->
      </div>
      <div id="rb_sum_warn" class="hidden text-xs text-red-400 mb-3">⚠ 현재 비중 합계가 100%가 아닙니다.</div>
      <div class="flex gap-3">
        <button onclick="calcRebalance()"
          class="bg-green-700 hover:bg-green-600 text-white font-semibold px-6 py-2 rounded-lg text-sm transition-colors">
          계산
        </button>
        <button onclick="savePortfolio()"
          class="bg-gray-700 hover:bg-gray-600 text-white font-semibold px-6 py-2 rounded-lg text-sm transition-colors border border-gray-600">
          💾 저장
        </button>
        <span id="rb_save_msg" class="hidden self-center text-xs text-green-400"></span>
      </div>
      <div id="rb_result" class="mt-5 hidden"></div>
    </div>

    <!-- 적립식 매수 계산기 (DCA) -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 mb-6">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-1">적립식 매수 계산기 (DCA)</h2>
      <p class="text-xs text-gray-600 mb-5">이번 달 추가 투자금을 추천 비중 갭 기반으로 배분 — 서버 통신 없이 즉시 계산</p>

      <!-- 초기 투입 플랜 안내 -->
      <div class="bg-gray-800 rounded-xl p-4 border border-yellow-900 mb-5">
        <p class="text-xs font-bold text-yellow-400 mb-2">💡 초기 투입 플랜 (1000만원 기준)</p>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-gray-300">
          <div class="bg-gray-700 rounded p-2 text-center">
            <p class="text-yellow-400 font-bold">1달차</p>
            <p class="mono font-semibold">400만원</p>
            <p class="text-gray-500">40% 투입</p>
          </div>
          <div class="bg-gray-700 rounded p-2 text-center">
            <p class="text-blue-400 font-bold">2~7달차</p>
            <p class="mono font-semibold">100만원/월</p>
            <p class="text-gray-500">6개월 분할</p>
          </div>
          <div class="bg-gray-700 rounded p-2 text-center">
            <p class="text-green-400 font-bold">합계</p>
            <p class="mono font-semibold">1,000만원</p>
            <p class="text-gray-500">7개월 완성</p>
          </div>
          <div class="bg-gray-700 rounded p-2 text-center">
            <p class="text-purple-400 font-bold">목적</p>
            <p class="text-gray-300 leading-tight">타이밍 리스크 분산</p>
          </div>
        </div>
        <div class="mt-3 flex gap-2">
          <button onclick="setDcaMonthly(4000000)"
            class="text-xs bg-yellow-800 hover:bg-yellow-700 border border-yellow-700 rounded px-3 py-1.5 transition-colors">
            1달차 (400만)
          </button>
          <button onclick="setDcaMonthly(1000000)"
            class="text-xs bg-blue-900 hover:bg-blue-800 border border-blue-700 rounded px-3 py-1.5 transition-colors">
            2~7달차 (100만)
          </button>
        </div>
      </div>

      <div class="flex flex-wrap gap-4 items-end mb-5">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">현재 총 자산 (원)</label>
          <input type="number" id="dca_total" value="0"
            class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500"
            placeholder="리밸런싱 계산기와 공유"/>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">이번 달 추가 투자금 (원)</label>
          <input type="number" id="dca_monthly" value="1000000"
            class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500 mono"/>
        </div>
        <button onclick="calcDCA()"
          class="bg-purple-700 hover:bg-purple-600 text-white font-semibold px-6 py-2 rounded-lg text-sm transition-colors">
          계산
        </button>
      </div>
      <p class="text-xs text-gray-700 mb-3">현재 비중은 위 리밸런싱 계산기의 입력값을 자동 참조</p>
      <div id="dca_result" class="hidden"></div>
    </div>

    <script>
    // ── 진입일 수익률 계산 ──
    document.getElementById("simForm").addEventListener("submit", async (e) => {{
      e.preventDefault();
      const entry_date = document.getElementById("sim_date").value.trim();
      const entry_krw  = parseInt(document.getElementById("sim_krw").value);
      if (!entry_date || !entry_krw) return;

      const res  = await fetch("/simulate", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{entry_date, entry_krw}})
      }});
      const data = await res.json();
      const el   = document.getElementById("simResult");
      el.classList.remove("hidden");

      if (data.error) {{
        el.innerHTML = `<p class="text-red-400 text-sm">❌ ${{data.error}}</p>`;
        return;
      }}
      const nameMap  = {{ markowitz: "Markowitz", markowitz_momentum: "Markowitz+Momentum", buy_and_hold: "Buy & Hold", kelly_mlp: "Kelly+MLP" }};
      const colorMap = {{ markowitz: "text-cyan-400", markowitz_momentum: "text-green-400", buy_and_hold: "text-gray-400", kelly_mlp: "text-orange-400" }};
      let cards = "";
      for (const [key, s] of Object.entries(data.strategies)) {{
        const pct  = s.period_return_pct;
        const pCls = pct >= 0 ? "text-green-400" : "text-red-400";
        const nCls = colorMap[key] || "text-white";
        cards += `
          <div class="bg-gray-800 rounded-xl p-4">
            <p class="text-xs font-semibold ${{nCls}} mb-2">${{nameMap[key] || key}}</p>
            <p class="text-3xl font-black mono ${{pCls}}">${{s.multiplier.toFixed(2)}}x</p>
            <p class="text-sm mono ${{pCls}}">${{pct >= 0 ? "+" : ""}}${{pct.toFixed(1)}}%</p>
            <p class="text-xs text-gray-500 mt-1">${{s.current_krw.toLocaleString()}}원</p>
          </div>`;
      }}
      el.innerHTML = `<div class="grid grid-cols-2 md:grid-cols-3 gap-3">${{cards}}</div>`;
    }});

    // ── 72의 법칙 ──
    function calcRule72() {{
      const rate = parseFloat(document.getElementById("rule72_rate").value);
      const el = document.getElementById("rule72_result");
      if (!rate || rate <= 0) {{ el.innerHTML = ""; return; }}
      const years = (72 / rate).toFixed(1);
      el.innerHTML = `
        <p class="text-2xl font-black mono text-yellow-400">${{years}}년</p>
        <p class="text-xs text-gray-500 mt-1">연 ${{rate}}% 수익률로 원금 2배</p>`;
    }}
    calcRule72();

    // ── 다음 리밸런싱 D-day ──
    (function() {{
      const today = new Date();
      const dow = today.getDay(); // 0=Sun, 1=Mon, ..., 6=Sat
      const daysToMonday = dow === 1 ? 7 : (8 - dow) % 7;
      const nextMonday = new Date(today);
      nextMonday.setDate(today.getDate() + daysToMonday);
      const mm = String(nextMonday.getMonth() + 1).padStart(2, "0");
      const dd = String(nextMonday.getDate()).padStart(2, "0");
      document.getElementById("dday_result").textContent = `D-${{daysToMonday}}`;
      document.getElementById("dday_date").textContent = `다음 월요일: ${{nextMonday.getFullYear()}}-${{mm}}-${{dd}}`;
    }})();

    // ── 리밸런싱 계산기 ──
    const REC_WEIGHTS = {rec_weights_js};
    const ASSETS = ["S&P500", "NASDAQ", "SOX", "CASH"];
    const ASSET_COLORS = {{"S&P500":"text-blue-400","NASDAQ":"text-purple-400","SOX":"text-green-400","CASH":"text-gray-400"}};

    // 입력 칸 생성
    const rbContainer = document.getElementById("rb_inputs");
    ASSETS.forEach(asset => {{
      const rec = REC_WEIGHTS[asset] || 0;
      rbContainer.innerHTML += `
        <div class="bg-gray-800 rounded-xl p-3 border border-gray-700">
          <p class="text-xs font-semibold ${{ASSET_COLORS[asset] || "text-white"}} mb-1">${{asset}}</p>
          <p class="text-xs text-gray-500 mb-2">추천: <span class="mono text-gray-300">${{rec}}%</span></p>
          <div class="flex items-center gap-1">
            <input type="number" id="rb_${{asset.replace("&","n")}}" value="0" min="0" max="100" step="1"
              class="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm w-full mono focus:outline-none focus:border-blue-500"
              oninput="checkRbSum()"/>
            <span class="text-gray-400 text-xs shrink-0">%</span>
          </div>
        </div>`;
    }});

    function getRbInputs() {{
      return ASSETS.map(a => parseFloat(document.getElementById("rb_" + a.replace("&","n")).value) || 0);
    }}

    function checkRbSum() {{
      const sum = getRbInputs().reduce((a, b) => a + b, 0);
      const warn = document.getElementById("rb_sum_warn");
      if (Math.abs(sum - 100) > 0.5) {{
        warn.textContent = `⚠ 현재 비중 합계 ${{sum.toFixed(1)}}% (100%여야 합니다)`;
        warn.classList.remove("hidden");
      }} else {{
        warn.classList.add("hidden");
      }}
    }}

    // ── 포트폴리오 상태 저장/로드 ──
    async function savePortfolio() {{
      const total = parseInt(document.getElementById("rb_total").value) || 0;
      const [sp500, nasdaq, sox, cash] = getRbInputs();
      const msg = document.getElementById("rb_save_msg");
      const res = await fetch("/api/portfolio", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          total_krw: total,
          sp500_pct: sp500, nasdaq_pct: nasdaq,
          sox_pct: sox, cash_pct: cash,
        }})
      }});
      const data = await res.json();
      msg.classList.remove("hidden");
      if (data.saved) {{
        msg.textContent = `✅ 저장 완료 (${{data.date}})`;
        setTimeout(() => msg.classList.add("hidden"), 3000);
      }} else {{
        msg.textContent = "❌ 저장 실패";
      }}
    }}

    // 페이지 로드 시 마지막 저장값 복원
    (async function loadPortfolio() {{
      try {{
        const res = await fetch("/api/portfolio");
        if (!res.ok) return;
        const data = await res.json();
        if (data.total_krw) document.getElementById("rb_total").value = data.total_krw;
        const mapping = {{"S&P500": data.sp500_pct, "NASDAQ": data.nasdaq_pct, "SOX": data.sox_pct, "CASH": data.cash_pct}};
        ASSETS.forEach(a => {{
          const el = document.getElementById("rb_" + a.replace("&","n"));
          if (el && mapping[a] != null) el.value = mapping[a];
        }});
        checkRbSum();
        const msg = document.getElementById("rb_save_msg");
        msg.classList.remove("hidden");
        msg.textContent = `📂 마지막 저장: ${{data.date}}`;
        setTimeout(() => msg.classList.add("hidden"), 4000);
      }} catch(e) {{}}
    }})();

    // ── 적립식 매수 계산기 (DCA) ──
    function setDcaMonthly(amount) {{
      document.getElementById("dca_monthly").value = amount;
      // 리밸런싱 총자산도 동기화
      const rbTotal = document.getElementById("rb_total").value;
      document.getElementById("dca_total").value = rbTotal || 0;
    }}

    function calcDCA() {{
      const total   = parseInt(document.getElementById("dca_total").value) || 0;
      const monthly = parseInt(document.getElementById("dca_monthly").value) || 0;
      const el      = document.getElementById("dca_result");

      if (monthly <= 0) {{
        el.classList.remove("hidden");
        el.innerHTML = `<p class="text-red-400 text-sm">⚠ 추가 투자금을 입력해주세요.</p>`;
        return;
      }}

      const currents = getRbInputs();  // [S&P500, NASDAQ, SOX, CASH]
      const RISKY_ASSETS = ["S&P500", "NASDAQ", "SOX"];
      const newTotal = total + monthly;

      // 갭 계산: 목표 배분액 - 현재 보유액
      const gaps = {{}};
      let totalGap = 0;
      RISKY_ASSETS.forEach((asset, i) => {{
        const recPct = REC_WEIGHTS[asset] || 0;
        const targetAmt  = newTotal * recPct / 100;
        const currentAmt = total > 0 ? total * currents[i] / 100 : 0;
        const gap = targetAmt - currentAmt;
        if (gap > 0) {{
          gaps[asset] = gap;
          totalGap += gap;
        }}
      }});

      let rows = "";
      let allocated = 0;

      if (totalGap <= 0) {{
        el.classList.remove("hidden");
        el.innerHTML = `<p class="text-green-400 text-sm">✅ 모든 종목이 이미 추천 비중 초과 — 이번 달 추가 매수 불필요 (현금 보유 권장)</p>`;
        return;
      }}

      const ASSET_COLORS2 = {{"S&P500":"text-blue-400","NASDAQ":"text-purple-400","SOX":"text-green-400"}};
      const sortedGaps = Object.entries(gaps).sort((a, b) => b[1] - a[1]);

      sortedGaps.forEach(([asset, gap]) => {{
        const alloc = Math.min(gap, Math.round(monthly * gap / totalGap));
        const capped = Math.min(alloc, monthly - allocated);
        if (capped <= 0) return;
        allocated += capped;
        const curPct = currents[RISKY_ASSETS.indexOf(asset)];
        const recPct = REC_WEIGHTS[asset] || 0;
        const c = ASSET_COLORS2[asset] || "text-white";
        rows += `
          <div class="flex items-center justify-between py-2 border-b border-gray-700 last:border-0">
            <span class="text-sm font-semibold ${{c}}">${{asset}}</span>
            <span class="text-xs text-gray-500 mono">${{curPct.toFixed(1)}}% → ${{recPct.toFixed(1)}}%</span>
            <span class="text-sm font-bold mono text-green-400">매수 ${{capped.toLocaleString()}}원</span>
          </div>`;
      }});

      const remainder = monthly - allocated;
      if (remainder > 0) {{
        rows += `
          <div class="flex items-center justify-between py-2">
            <span class="text-sm text-gray-400">CASH (잔여)</span>
            <span class="text-xs text-gray-500">보유</span>
            <span class="text-sm mono text-gray-400">${{remainder.toLocaleString()}}원</span>
          </div>`;
      }}

      el.classList.remove("hidden");
      el.innerHTML = `
        <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <p class="text-xs text-gray-500 mb-3">추가 ${{monthly.toLocaleString()}}원 배분 (현재 총자산 ${{total.toLocaleString()}}원 → 신규 ${{newTotal.toLocaleString()}}원)</p>
          ${{rows}}
        </div>`;
    }}

    function calcRebalance() {{
      const total = parseInt(document.getElementById("rb_total").value) || 0;
      const currents = getRbInputs();
      const sum = currents.reduce((a, b) => a + b, 0);
      const el = document.getElementById("rb_result");

      if (Math.abs(sum - 100) > 0.5) {{
        el.classList.remove("hidden");
        el.innerHTML = `<p class="text-red-400 text-sm">⚠ 현재 비중 합계가 ${{sum.toFixed(1)}}%입니다. 100%로 맞춰주세요.</p>`;
        return;
      }}
      if (total <= 0) {{
        el.classList.remove("hidden");
        el.innerHTML = `<p class="text-red-400 text-sm">⚠ 총 자산을 입력해주세요.</p>`;
        return;
      }}

      let rows = "";
      let hasAction = false;
      ASSETS.forEach((asset, i) => {{
        const cur = currents[i];
        const rec = REC_WEIGHTS[asset] || 0;
        const diff = rec - cur;
        const amount = Math.round(total * Math.abs(diff) / 100);
        if (amount < 1000) return; // 1000원 미만 무시
        hasAction = true;
        const isBuy = diff > 0;
        const actionCls = isBuy ? "text-green-400" : "text-red-400";
        const action = isBuy ? "매수" : "매도";
        const sign = isBuy ? "+" : "-";
        rows += `
          <div class="flex items-center justify-between py-2 border-b border-gray-700 last:border-0">
            <span class="text-sm font-semibold ${{ASSET_COLORS[asset] || "text-white"}}">${{asset}}</span>
            <span class="text-xs text-gray-500 mono">${{cur.toFixed(1)}}% → ${{rec.toFixed(1)}}%</span>
            <span class="text-xs text-gray-500">(${{sign}}${{Math.abs(diff).toFixed(1)}}%p)</span>
            <span class="text-sm font-bold mono ${{actionCls}}">${{action}} ${{amount.toLocaleString()}}원</span>
          </div>`;
      }});

      el.classList.remove("hidden");
      if (!hasAction) {{
        el.innerHTML = `<p class="text-green-400 text-sm">✅ 이미 추천 비중에 가깝습니다. 리밸런싱 불필요.</p>`;
      }} else {{
        el.innerHTML = `
          <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <p class="text-xs text-gray-500 mb-3">총 자산 ${{total.toLocaleString()}}원 기준</p>
            ${{rows}}
          </div>`;
      }}
    }}
    </script>
    """

    # 행동강령 섹션
    conduct = """
    <div class="mt-6 bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-800">
        <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider">행동강령</h2>
        <p class="text-xs text-gray-600 mt-0.5">Markowitz 전략 기반 — 이 규칙을 어기면 백테스트 성과를 재현할 수 없음</p>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-gray-800">

        <!-- 절대 금지 -->
        <div class="p-5">
          <p class="text-xs font-bold text-red-400 uppercase tracking-wider mb-3">🚫 절대 금지</p>
          <ul class="space-y-2 text-sm text-gray-300">
            <li class="flex gap-2">
              <span class="text-red-500 shrink-0 mt-0.5">✕</span>
              <span><strong>샤프지수 무시</strong>하고 단일 종목 집중 — Sharpe가 낮아도 "느낌상 좋아 보이면" 올인하는 행위 금지</span>
            </li>
            <li class="flex gap-2">
              <span class="text-red-500 shrink-0 mt-0.5">✕</span>
              <span>신호 나오기 전 <strong>선매수</strong> — 신호일 이전 진입은 백테스트 기반 없는 도박</span>
            </li>
            <li class="flex gap-2">
              <span class="text-red-500 shrink-0 mt-0.5">✕</span>
              <span>Markowitz 비중 무시하고 <strong>임의 비율 변경</strong> — 최적화 결과를 수동으로 덮어쓰지 않는다</span>
            </li>
          </ul>
        </div>

        <!-- 필수 준수 -->
        <div class="p-5">
          <p class="text-xs font-bold text-green-400 uppercase tracking-wider mb-3">✅ 필수 준수</p>
          <ul class="space-y-2 text-sm text-gray-300">
            <li class="flex gap-2">
              <span class="text-green-500 shrink-0 mt-0.5">✓</span>
              <span>포지션은 <strong>Markowitz 최적 비중대로만</strong> 유지 — Sharpe 최대화 가중치 그대로 집행</span>
            </li>
            <li class="flex gap-2">
              <span class="text-green-500 shrink-0 mt-0.5">✓</span>
              <span><strong>매주 월요일 오전</strong> 신호 확인 후 비중 5%p 이상 이탈 시 리밸런싱</span>
            </li>
            <li class="flex gap-2">
              <span class="text-green-500 shrink-0 mt-0.5">✓</span>
              <span>AVOID 신호 시 <strong>즉시 현금 비중 확대</strong> — Bear 레짐에서 손실 방어 최우선</span>
            </li>
          </ul>
        </div>

        <!-- 참고용 (Kelly) -->
        <div class="p-5">
          <p class="text-xs font-bold text-yellow-400 uppercase tracking-wider mb-3">📊 참고용 (Kelly+MLP)</p>
          <ul class="space-y-2 text-sm text-gray-300">
            <li class="flex gap-2">
              <span class="text-yellow-500 shrink-0 mt-0.5">!</span>
              <span>Kelly+MLP 백테스트 결과는 <strong>전략 비교 참고용</strong>이며 실제 매매에 사용하지 않는다</span>
            </li>
            <li class="flex gap-2">
              <span class="text-yellow-500 shrink-0 mt-0.5">!</span>
              <span><strong>Sharpe -0.062, CAGR 1.7%</strong> — ML 예측이 Buy&amp;Hold조차 못 이김. 과신 금지</span>
            </li>
            <li class="flex gap-2">
              <span class="text-yellow-500 shrink-0 mt-0.5">!</span>
              <span>메인 전략: <strong>Markowitz</strong> · 검증 지표: Kelly+MLP · 벤치마크: Buy &amp; Hold</span>
            </li>
          </ul>
        </div>

      </div>
    </div>
    """
    return HTMLResponse(_shell("Dashboard — Can ML Beat the Market?", body + conduct))


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page():
    """백테스트 결과 차트 페이지."""
    bt = _load_backtest()

    if bt is None:
        body = _no_data_page("backtest_result.json", "uv run python backtest.py")
        return HTMLResponse(_shell("Backtest", body))

    mw        = bt["markowitz"]
    mwm       = bt.get("markowitz_momentum")   # optional (구버전 JSON 호환)
    bh        = bt["buy_and_hold"]
    km        = bt.get("kelly_mlp")            # optional
    generated = bt.get("generated_at", "")

    # ── 가상 시뮬레이션 기본값 계산 (2022-01-01, Markowitz, 10M) ──
    _SIM_DATE = "2022-01-01"
    _SIM_KRW  = 10_000_000
    _cum_mw   = bt["cumulative_returns"]["markowitz"]
    _dates_s  = sorted(_cum_mw.keys())
    _entry_d  = next((d for d in _dates_s if d >= _SIM_DATE), None)
    _default_hint = ""
    if _entry_d:
        _entry_cum  = _cum_mw[_entry_d]
        _latest_d   = _dates_s[-1]
        _latest_cum = _cum_mw[_latest_d]
        _mult       = _latest_cum / _entry_cum
        _ret_pct    = (_mult - 1) * 100
        _cur_krw    = int(_SIM_KRW * _mult)
        _sign       = "+" if _ret_pct >= 0 else ""
        _default_hint = f"기본값 2022-01-01 결과: <span class=\"mono text-cyan-400\">{_mult:.3f}x / {_sign}{_ret_pct:.1f}% / {_cur_krw:,}원</span>"

    # Chart.js용 데이터 준비
    cum_mw  = bt["cumulative_returns"]["markowitz"]
    cum_mwm = bt["cumulative_returns"].get("markowitz_momentum", {})
    cum_bh  = bt["cumulative_returns"]["buy_and_hold"]
    cum_km  = bt["cumulative_returns"].get("kelly_mlp", {})
    excess  = bt.get("excess_vs_bh", {})

    all_dates = sorted(set(cum_mw.keys()) & set(cum_bh.keys()))
    sampled   = [d for i, d in enumerate(all_dates) if i % 21 == 0 or i == len(all_dates) - 1]

    labels_js    = json.dumps(sampled)
    mw_vals_js   = json.dumps([cum_mw.get(d, 1.0) for d in sampled])
    mwm_vals_js  = json.dumps([cum_mwm.get(d) for d in sampled]) if cum_mwm else "null"
    bh_vals_js   = json.dumps([cum_bh.get(d, 1.0) for d in sampled])
    exc_vals_js  = json.dumps([excess.get(d, 1.0) for d in sampled])
    km_vals_js   = json.dumps([cum_km.get(d) for d in sampled]) if cum_km else "null"

    # 성과 카드 열 수: Markowitz + Momentum + B&H + Kelly(optional)
    _n_cards   = 2 + (1 if mwm else 0) + (1 if km else 0)
    card_cols  = {2: "md:grid-cols-2", 3: "md:grid-cols-3", 4: "md:grid-cols-4"}.get(_n_cards, "md:grid-cols-2")
    mom_card   = _bt_card("Markowitz+Momentum", mwm) if mwm else ""
    kelly_card = _bt_card("Kelly+MLP (Walk-Forward)", km) if km else ""

    # 데이터 기간 표기
    data_start = all_dates[0] if all_dates else "2010-01-01"
    data_end   = all_dates[-1] if all_dates else "2025-12-31"

    body = f"""
    <div class="mb-8">
      <h1 class="text-3xl font-bold">Backtest Results</h1>
      <p class="text-gray-500 mt-1 text-sm">전략 비교: Markowitz / Markowitz+모멘텀 / Buy&Hold / Kelly+MLP · 기준: {generated}</p>
    </div>

    <!-- Performance Cards -->
    <div class="grid grid-cols-1 {card_cols} gap-5 mb-8">
      {_bt_card("Markowitz (월별 리밸런싱)", mw)}
      {mom_card}
      {_bt_card("Buy & Hold (S&P500)", bh)}
      {kelly_card}
    </div>

    <!-- Cumulative Return Chart -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 mb-2">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-5">누적 수익률 비교</h2>
      <div style="position:relative; height:340px;">
        <canvas id="cumChart"></canvas>
      </div>
    </div>
    <p class="text-xs text-gray-600 text-right mb-5 px-1">데이터 기간: {data_start} ~ {data_end}</p>

    <!-- Excess Return Chart -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 mb-8">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-5">
        Markowitz ÷ Buy&Hold (초과 누적수익)
      </h2>
      <div style="position:relative; height:200px;">
        <canvas id="excChart"></canvas>
      </div>
    </div>

    <!-- 가상 시뮬레이션 (통합) -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 mb-8">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-1">가상 시뮬레이션</h2>
      <p class="text-xs text-gray-600 mb-1">특정 날짜에 진입했다면 지금 얼마가 됐을까? · {_default_hint}</p>
      <p class="text-xs text-gray-700 mb-5">백테스트 기간: {data_start} ~ {data_end}</p>
      <div class="flex flex-wrap gap-3 mb-5">
        <button onclick="quickSim('2020-01-02')"
          class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-3 py-1.5 transition-colors">
          2020-01-02 (코로나 전)
        </button>
        <button onclick="quickSim('2022-01-03')"
          class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-3 py-1.5 transition-colors">
          2022-01-03 (금리 인상 시작)
        </button>
        <button onclick="quickSim('2023-01-03')"
          class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg px-3 py-1.5 transition-colors">
          2023-01-03 (AI 랠리 전)
        </button>
      </div>
      <form id="simForm2" class="flex flex-wrap gap-4 items-end mb-5">
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">진입일 (YYYY-MM-DD)</label>
          <input type="text" id="sim2_date" placeholder="2022-01-01"
            class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500 mono"/>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1.5">초기 자산 (원)</label>
          <input type="number" id="sim2_krw" value="10000000"
            class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-44 focus:outline-none focus:border-blue-500"/>
        </div>
        <button type="submit"
          class="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-6 py-2 rounded-lg text-sm transition-colors">
          계산
        </button>
      </form>
      <div id="simResult2" class="hidden"></div>
    </div>

    <script>
    // ── 누적수익률 차트 ──
    const chartDefaults = {{
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: "#94a3b8", font: {{ size: 11 }} }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => ctx.dataset.label + ": " + ((ctx.parsed.y - 1) * 100).toFixed(1) + "%",
          }},
        }},
      }},
      scales: {{
        x: {{ ticks: {{ color: "#64748b", maxTicksLimit: 8, maxRotation: 0 }}, grid: {{ color: "#1e293b" }} }},
        y: {{ ticks: {{ color: "#64748b", callback: (v) => ((v - 1) * 100).toFixed(0) + "%" }}, grid: {{ color: "#1e293b" }} }},
      }},
    }};

    const cumDatasets = [
      {{
        label: "Markowitz",
        data:  {mw_vals_js},
        borderColor: "#22d3ee",
        backgroundColor: "rgba(34,211,238,0.04)",
        borderWidth: 1.8,
        pointRadius: 0,
        fill: true,
      }},
      {{
        label: "Buy & Hold (S&P500)",
        data:  {bh_vals_js},
        borderColor: "#64748b",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        borderDash: [5, 3],
      }},
    ];
    const mwmVals = {mwm_vals_js};
    if (mwmVals) {{
      cumDatasets.push({{
        label: "Markowitz+Momentum",
        data:  mwmVals,
        borderColor: "#4ade80",
        backgroundColor: "rgba(74,222,128,0.04)",
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false,
        borderDash: [4, 2],
      }});
    }}
    const kmVals = {km_vals_js};
    if (kmVals) {{
      cumDatasets.push({{
        label: "Kelly+MLP",
        data:  kmVals,
        borderColor: "#fb923c",
        backgroundColor: "rgba(251,146,60,0.04)",
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false,
        borderDash: [3, 2],
      }});
    }}

    new Chart(document.getElementById("cumChart").getContext("2d"), {{
      type: "line",
      data: {{ labels: {labels_js}, datasets: cumDatasets }},
      options: chartDefaults,
    }});

    // ── 초과수익 차트 ──
    const excOpts = JSON.parse(JSON.stringify(chartDefaults));
    new Chart(document.getElementById("excChart").getContext("2d"), {{
      type: "line",
      data: {{
        labels: {labels_js},
        datasets: [{{
          label: "초과 누적수익 (Markowitz ÷ B&H)",
          data: {exc_vals_js},
          borderColor: "#a78bfa",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: "origin",
        }}],
      }},
      options: excOpts,
    }});

    // ── 시뮬레이션 ──
    function renderSimResult(data) {{
      const el = document.getElementById("simResult2");
      el.classList.remove("hidden");
      if (data.error) {{
        el.innerHTML = `<p class="text-red-400 text-sm">❌ ${{data.error}}</p>`;
        return;
      }}
      const nameMap  = {{ markowitz: "Markowitz", markowitz_momentum: "Markowitz+Momentum", buy_and_hold: "Buy & Hold", kelly_mlp: "Kelly+MLP" }};
      const colorMap = {{ markowitz: "text-cyan-400", markowitz_momentum: "text-green-400", buy_and_hold: "text-gray-400", kelly_mlp: "text-orange-400" }};
      let cards = "";
      for (const [key, s] of Object.entries(data.strategies)) {{
        const pct = s.period_return_pct;
        const pCls = pct >= 0 ? "text-green-400" : "text-red-400";
        const nameCls = colorMap[key] || "text-white";
        cards += `
          <div class="bg-gray-800 rounded-2xl p-5 border border-gray-700">
            <p class="text-xs font-semibold uppercase tracking-wider ${{nameCls}} mb-3">${{nameMap[key] || key}}</p>
            <p class="text-4xl font-black mono ${{pCls}} mb-1">${{s.multiplier.toFixed(2)}}x</p>
            <p class="text-sm mono ${{pCls}} mb-3">${{pct >= 0 ? "+" : ""}}${{pct.toFixed(1)}}%</p>
            <div class="space-y-1 text-xs text-gray-400">
              <div class="flex justify-between">
                <span>현재 자산</span>
                <span class="mono font-semibold text-white">${{s.current_krw.toLocaleString()}}원</span>
              </div>
              <div class="flex justify-between">
                <span>손익</span>
                <span class="mono font-semibold ${{pCls}}">${{s.profit_krw >= 0 ? "+" : ""}}${{s.profit_krw.toLocaleString()}}원</span>
              </div>
              <div class="flex justify-between pt-1 border-t border-gray-700">
                <span>기간</span>
                <span class="mono">${{s.entry_date}} → ${{s.latest_date}}</span>
              </div>
            </div>
          </div>`;
      }}
      const colCls = Object.keys(data.strategies).length === 3
        ? "grid-cols-1 md:grid-cols-3" : "grid-cols-1 md:grid-cols-2";
      el.innerHTML = `<div class="grid ${{colCls}} gap-4">${{cards}}</div>`;
    }}

    function quickSim(date) {{
      document.getElementById("sim2_date").value = date;
      document.getElementById("simForm2").requestSubmit();
    }}

    document.getElementById("simForm2").addEventListener("submit", async (e) => {{
      e.preventDefault();
      const entry_date = document.getElementById("sim2_date").value.trim();
      const entry_krw  = parseInt(document.getElementById("sim2_krw").value);
      if (!entry_date || !entry_krw) return;
      const res  = await fetch("/simulate", {{
        method: "POST", headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{entry_date, entry_krw}})
      }});
      renderSimResult(await res.json());
    }});
    </script>
    """
    return HTMLResponse(_shell("Backtest — Can ML Beat the Market?", body))


# ══════════════════════════════════════════════════════════
#  직접 실행
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)
