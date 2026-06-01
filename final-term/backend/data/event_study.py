"""B-17: 이벤트 스터디 — 게임 이벤트의 정책→섹터 인과를 실제 역사로 고증.

게임의 대표 이벤트와 대응되는 실제 역사적 사건 윈도우를 잡아,
각 섹터 ETF의 누적수익률을 측정한다. "전쟁 나면 방산·에너지 오른다"가
진짜인지 데이터로 확인 → B-16 매핑 테이블의 근거가 된다.

산출: data/event_study.json
"""
import json
import os

SECTOR_PROXY = {
    "semiconductor": "SOXX",
    "energy": "XLE",
    "finance": "XLF",
    "defense": "ITA",
    "consumer": "XLP",
}

# 게임 이벤트 ↔ 실제 역사 사건 윈도우 (start, end, inclusive)
# 누적수익률 = (end종가 / start종가 - 1) × 100
EPISODES = [
    {
        "game_event": "전쟁 발발 / 국제긴장 고조",
        "real_event": "러시아 우크라이나 침공",
        "start": "2022-02-23", "end": "2022-03-08",
        "expect": "방산↑ 에너지↑ / 성장주(반도체)↓ 소비재 약세",
    },
    {
        "game_event": "팬데믹 / 민심 급락 패닉",
        "real_event": "COVID-19 시장 붕괴",
        "start": "2020-02-19", "end": "2020-03-23",
        "expect": "전 섹터↓, 소비재(방어주)가 상대적으로 덜 하락",
    },
    {
        "game_event": "금리 인상 / 긴축",
        "real_event": "2022 연준 공격적 금리 인상기",
        "start": "2022-01-03", "end": "2022-10-13",
        "expect": "금융 상대적 선방 / 성장주(반도체) 급락",
    },
    {
        "game_event": "유가 급등 충격",
        "real_event": "2014~2016 유가 붕괴 (역방향 참고)",
        "start": "2014-06-20", "end": "2016-02-11",
        "expect": "에너지 폭락 — 유가↔에너지 섹터 직결 확인용",
    },
    {
        "game_event": "팬데믹 회복 / 경기 반등",
        "real_event": "COVID 저점 이후 회복 랠리",
        "start": "2020-03-23", "end": "2020-08-31",
        "expect": "반도체(성장주)↑ 주도, 에너지 회복 지연",
    },
]

OUT_PATH = os.path.join(os.path.dirname(__file__), "event_study.json")


def nearest_close(close, ticker, date_str):
    """해당 날짜 또는 직후 거래일의 종가."""
    import pandas as pd
    s = close[ticker]
    ts = pd.Timestamp(date_str)
    sub = s[s.index >= ts]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0])


def main():
    import yfinance as yf

    tickers = list(SECTOR_PROXY.values())
    raw = yf.download(tickers, start="2014-01-01", end="2023-01-01",
                      auto_adjust=True, progress=False)
    close = raw["Close"]

    results = []
    for ep in EPISODES:
        sector_returns = {}
        for sector, ticker in SECTOR_PROXY.items():
            p0 = nearest_close(close, ticker, ep["start"])
            p1 = nearest_close(close, ticker, ep["end"])
            if p0 and p1:
                sector_returns[sector] = round((p1 / p0 - 1) * 100, 1)
            else:
                sector_returns[sector] = None
        ranked = sorted(
            [(s, r) for s, r in sector_returns.items() if r is not None],
            key=lambda x: x[1], reverse=True,
        )
        results.append({
            "game_event": ep["game_event"],
            "real_event": ep["real_event"],
            "window": f"{ep['start']} ~ {ep['end']}",
            "expectation": ep["expect"],
            "sector_returns_pct": sector_returns,
            "ranking_best_to_worst": [s for s, _ in ranked],
        })

    out = {
        "_meta": {
            "source": "yfinance",
            "sector_proxy": SECTOR_PROXY,
            "metric": "윈도우 누적수익률 % = (end종가/start종가 - 1)×100",
        },
        "episodes": results,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[event_study] 저장 완료: {OUT_PATH}\n")
    for ep in results:
        print(f"■ {ep['game_event']}  ({ep['real_event']}, {ep['window']})")
        print(f"  예상: {ep['expectation']}")
        for s in ep["ranking_best_to_worst"]:
            r = ep["sector_returns_pct"][s]
            print(f"    {s:14s} {r:+.1f}%")
        print()


if __name__ == "__main__":
    main()
