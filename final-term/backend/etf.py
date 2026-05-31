"""ETF 가격 계산 모듈 — 게이지 변화에 따른 5종 ETF 가격 산출."""
import random
from config import ETF_KEYS, ETF_SENSITIVITY, ETF_NOISE_RANGE, GAUGE_KEYS


def calculate_etf_changes(
    gauge_deltas: dict,
    current_prices: dict,
) -> tuple[dict, dict]:
    """게이지 변화량을 기반으로 ETF 가격을 갱신한다.

    Returns:
        (new_prices, pct_changes): 새 가격 dict, 퍼센트 변화 dict
    """
    new_prices = {}
    pct_changes = {}

    for etf in ETF_KEYS:
        sensitivity = ETF_SENSITIVITY[etf]
        weighted_sum = sum(
            gauge_deltas.get(g, 0) * sensitivity.get(g, 0)
            for g in GAUGE_KEYS
        )
        noise = random.uniform(-ETF_NOISE_RANGE, ETF_NOISE_RANGE)
        pct = weighted_sum + noise

        old_price = current_prices[etf]
        new_price = round(old_price * (1 + pct / 100), 1)
        new_price = max(1.0, new_price)  # 최소 1

        new_prices[etf] = new_price
        pct_changes[etf] = round(pct, 2)

    return new_prices, pct_changes
