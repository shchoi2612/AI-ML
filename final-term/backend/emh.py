"""EMH 검증 분석 — 임기말 성적표."""
from config import GAUGE_KEYS, GAUGE_NAMES, ETF_KEYS, ETF_NAMES


def analyze_emh(state: dict) -> dict:
    """게임 종료 후 EMH 분석 결과를 반환한다.

    Pearson 상관: 정책 카테고리(게이지 변화 방향) × 섹터 ETF 변동률.
    숫자 3개 + 한 문장 요약.
    """
    gauge_history = state.get("gauge_history", [])
    etf_history = state.get("etf_history", [])
    turn = state.get("turn", 1) - 1

    # 안정도 점수
    stability = (
        max(0, 100 - state["debt"])
        + max(0, 100 - state["inflation"])
        + state["morale"]
        + max(0, 100 - state["tension"])
    )

    if len(etf_history) < 3:
        return {
            "total_turns": turn,
            "stability_score": stability,
            "predictability_score": 0.0,
            "top_correlation": {"sector": "N/A", "value": 0.0},
            "avg_etf_volatility": 0.0,
            "summary_text": f"임기 {turn}개월 동안의 데이터가 부족하여 시장 예측 가능성을 분석할 수 없습니다.",
        }

    # 게이지 변화량 시계열 구성
    gauge_deltas = []
    for i in range(1, len(gauge_history)):
        delta = sum(
            abs(gauge_history[i][k] - gauge_history[i - 1][k])
            for k in GAUGE_KEYS
        )
        gauge_deltas.append(delta)

    # ETF 변동률 시계열 (섹터별)
    sector_changes = {etf: [] for etf in ETF_KEYS}
    for entry in etf_history:
        for etf in ETF_KEYS:
            sector_changes[etf].append(entry.get(etf, 0.0))

    # Pearson 상관: 정책 강도(게이지 총 변화량) vs 각 섹터 ETF 변동
    n = min(len(gauge_deltas), len(etf_history))
    if n < 3:
        return {
            "total_turns": turn,
            "stability_score": stability,
            "predictability_score": 0.0,
            "top_correlation": {"sector": "N/A", "value": 0.0},
            "avg_etf_volatility": 0.0,
            "summary_text": f"임기 {turn}개월 동안의 데이터가 부족하여 시장 예측 가능성을 분석할 수 없습니다.",
        }

    correlations = {}
    for etf in ETF_KEYS:
        r = _pearson(gauge_deltas[:n], sector_changes[etf][:n])
        correlations[etf] = round(r, 3)

    # 가장 높은 상관 섹터
    top_etf = max(correlations, key=lambda k: abs(correlations[k]))
    top_val = correlations[top_etf]

    # 평균 예측 가능성 (전체 상관 절대값 평균)
    predictability = round(
        sum(abs(v) for v in correlations.values()) / len(correlations), 3
    )

    # 평균 ETF 변동성
    all_changes = [abs(c) for etf in ETF_KEYS for c in sector_changes[etf][:n]]
    avg_vol = round(sum(all_changes) / len(all_changes), 2) if all_changes else 0.0

    # 요약 텍스트
    if predictability > 0.6:
        verdict = "시장이 정책에 강하게 반응했습니다 — EMH 약형식이 위반되는 패턴입니다."
    elif predictability > 0.3:
        verdict = "시장이 정책에 부분적으로 반응했습니다 — 일부 예측 가능한 패턴이 존재합니다."
    else:
        verdict = "시장이 정책과 독립적으로 움직였습니다 — EMH 약형식이 성립하는 패턴입니다."

    summary = (
        f"임기 {turn}개월 동안 당신의 정책은 시장에 {predictability*100:.0f}% 예측 가능한 패턴을 만들었습니다. "
        f"가장 민감한 섹터는 {ETF_NAMES[top_etf]}(상관계수 {top_val:+.2f})입니다. "
        f"{verdict}"
    )

    return {
        "total_turns": turn,
        "stability_score": stability,
        "predictability_score": predictability,
        "top_correlation": {"sector": ETF_NAMES[top_etf], "value": top_val},
        "avg_etf_volatility": avg_vol,
        "summary_text": summary,
    }


def _pearson(x: list, y: list) -> float:
    """두 시계열의 Pearson 상관계수를 계산한다."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return 0.0
    return cov / denom
