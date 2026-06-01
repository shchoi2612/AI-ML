"""B-15: 실제 섹터 ETF 데이터 수집 + 변동성·상관관계 추출.

게임의 가상 ETF를 실제 미국 섹터 ETF로 보정하기 위한 통계를 뽑는다.
게임 섹터 → 실제 ETF 프록시 매핑:
    semiconductor → SOXX (iShares Semiconductor)
    energy        → XLE  (Energy Select Sector SPDR)
    finance       → XLF  (Financial Select Sector SPDR)
    defense       → ITA  (iShares Aerospace & Defense)
    consumer      → XLP  (Consumer Staples Select Sector SPDR)

산출: data/market_stats.json
    - 섹터별 일별 변동성(표준편차) + 연율화 변동성
    - 변동성 상대배수 (가장 잔잔한 섹터=1.0 기준) → ETF_NOISE_RANGE 보정용
    - 섹터 간 상관행렬

엔진(config.py)에 값을 꽂는 것은 B-18 (W2末 hero 확인 후). 여기선 수집·분석까지만.
"""
import json
import os

# 게임 섹터 → 실제 ETF 티커
SECTOR_PROXY = {
    "semiconductor": "SOXX",
    "energy": "XLE",
    "finance": "XLF",
    "defense": "ITA",
    "consumer": "XLP",
}

START_DATE = "2015-01-01"
OUT_PATH = os.path.join(os.path.dirname(__file__), "market_stats.json")
TRADING_DAYS = 252  # 연율화 계수


def fetch_and_analyze() -> dict:
    import yfinance as yf
    import numpy as np

    tickers = list(SECTOR_PROXY.values())
    raw = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)

    # auto_adjust=True 이면 'Close'가 수정종가
    close = raw["Close"]
    # 일별 로그수익률
    returns = np.log(close / close.shift(1)).dropna()

    # 섹터별 통계 (게임 섹터 키로 환산)
    vol_daily = {}
    for sector, ticker in SECTOR_PROXY.items():
        vol_daily[sector] = float(returns[ticker].std())

    # 상대 변동성 배수 (가장 잔잔한 섹터 = 1.0)
    base_vol = min(vol_daily.values())
    rel_vol = {s: round(v / base_vol, 3) for s, v in vol_daily.items()}

    vol_annual = {s: round(v * np.sqrt(TRADING_DAYS), 4) for s, v in vol_daily.items()}
    vol_daily = {s: round(v, 5) for s, v in vol_daily.items()}

    # 섹터 간 상관행렬 (게임 섹터 키로 라벨 변환)
    corr = returns.corr()
    inv = {v: k for k, v in SECTOR_PROXY.items()}
    corr_matrix = {}
    for t1 in tickers:
        corr_matrix[inv[t1]] = {
            inv[t2]: round(float(corr.loc[t1, t2]), 3) for t2 in tickers
        }

    return {
        "_meta": {
            "source": "yfinance",
            "start_date": START_DATE,
            "sector_proxy": SECTOR_PROXY,
            "n_observations": int(len(returns)),
            "note": "auto_adjust=True (수정종가), 로그수익률 기준. B-18에서 엔진 반영.",
        },
        "volatility_daily": vol_daily,
        "volatility_annual": vol_annual,
        "volatility_relative": rel_vol,
        "correlation_matrix": corr_matrix,
    }


def main():
    try:
        stats = fetch_and_analyze()
    except Exception as e:
        print(f"[fetch_etf] 데이터 수집 실패: {e}")
        print("yfinance 설치 또는 네트워크 확인 필요: uv pip install yfinance")
        raise

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[fetch_etf] 저장 완료: {OUT_PATH}")
    print(f"  관측치: {stats['_meta']['n_observations']}일")
    print("  섹터별 상대 변동성 (잔잔한 섹터=1.0):")
    for s, v in sorted(stats["volatility_relative"].items(), key=lambda x: x[1]):
        print(f"    {s:14s} ×{v}")


if __name__ == "__main__":
    main()
