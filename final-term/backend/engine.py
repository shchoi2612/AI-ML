"""경제 게임 엔진 — 게임 상태, 게이지 계산, 이벤트 선택."""
import random
from config import (
    INITIAL_STATE, GAUGE_KEYS, GAUGE_NAMES, MAX_TURNS,
    DIFFICULTY_TIERS, CASCADE_THRESHOLDS, ETF_KEYS,
    HINT_MAGNITUDE, HINT_DIRECTION,
)
from etf import calculate_etf_changes


def new_game() -> dict:
    """초기 게임 상태를 반환한다."""
    return {
        "turn": 1,
        **{k: v for k, v in INITIAL_STATE.items()},
        "etf_prices": {k: 100.0 for k in ETF_KEYS},
        "etf_history": [],
        "gauge_history": [{k: v for k, v in INITIAL_STATE.items()}],
        "log": [],
        "event_history": [],
        "pending_chain": None,
    }


def _get_difficulty_multiplier(turn: int) -> float:
    """현재 턴에 해당하는 난이도 분산 배수를 반환한다."""
    for start, end, mult in DIFFICULTY_TIERS:
        if start <= turn <= end:
            return mult
    return 1.0


def apply_choice(state: dict, choice: dict, label: str) -> dict:
    """선택지의 효과를 랜덤 분산과 함께 적용하고, ETF를 갱신한다.

    Returns:
        실제 적용된 gauge_deltas dict
    """
    base = choice["base_effects"]
    variance = choice.get("variance", 0)
    mult = _get_difficulty_multiplier(state["turn"])
    scaled_var = int(variance * mult)

    gauge_deltas = {}
    changes_text = []

    for key in GAUGE_KEYS:
        base_val = base.get(key, 0)
        if base_val == 0 and scaled_var == 0:
            continue
        delta = base_val + random.randint(-scaled_var, scaled_var)
        state[key] = max(0, min(100, state[key] + delta))
        gauge_deltas[key] = delta
        sign = "+" if delta > 0 else ""
        changes_text.append(f"{GAUGE_NAMES[key]} {sign}{delta}")

    # ETF 갱신
    new_prices, pct_changes = calculate_etf_changes(gauge_deltas, state["etf_prices"])
    state["etf_prices"] = new_prices
    state["etf_history"].append(pct_changes)

    # 히스토리 기록
    state["gauge_history"].append({k: state[k] for k in GAUGE_KEYS})
    state["log"].append(f"턴 {state['turn']}: [{label}] → {', '.join(changes_text)}")
    state["turn"] += 1

    return gauge_deltas


def check_game_over(state: dict) -> str | None:
    """게임 오버 조건을 확인한다. None이면 계속 진행."""
    if state["debt"] >= 100:
        return "국가 부도 — 부채가 한계를 초과했습니다."
    if state["inflation"] >= 100:
        return "초인플레이션 — 경제가 붕괴했습니다."
    if state["morale"] <= 0:
        return "혁명 발생 — 민심이 바닥을 쳤습니다."
    if state["tension"] >= 100:
        return "전쟁 발발 — 국제 긴장이 폭발했습니다."
    if state["turn"] > MAX_TURNS:
        # 완주(승리): 리터럴 "WIN" 대신 표시용 한국어 메시지를 반환한다.
        # game_over 계약은 string|null이며, 프론트는 이 문자열을 그대로 노출한다.
        return f"임기 완주 — 당신의 {MAX_TURNS}개월이 끝났습니다."
    return None


def _get_tier(turn: int) -> int:
    """턴 번호로 이벤트 티어를 결정한다."""
    if turn <= 7:
        return 1
    if turn <= 14:
        return 2
    return 3


def _check_cascades(state: dict) -> list[str]:
    """현재 게이지 상태에서 해금된 캐스케이드 태그를 반환한다."""
    tags = []
    for c in CASCADE_THRESHOLDS:
        val = state[c["gauge"]]
        if c["direction"] == "above" and val >= c["threshold"]:
            tags.extend(c["tags"])
        elif c["direction"] == "below" and val <= c["threshold"]:
            tags.extend(c["tags"])
    return tags


def select_event(state: dict, events: list) -> dict:
    """현재 게임 상태에 맞는 이벤트를 선택한다."""
    turn = state["turn"]
    tier = _get_tier(turn)
    recent_ids = state["event_history"][-5:]
    cascade_tags = _check_cascades(state)

    # 체인 이벤트 우선 처리
    if state["pending_chain"]:
        chain_id = state["pending_chain"]
        state["pending_chain"] = None
        for e in events:
            if e["id"] == chain_id:
                state["event_history"].append(e["id"])
                return e

    # 후보 풀 구성: 해당 티어 + 해금된 캐스케이드
    pool = []
    for e in events:
        if e["id"] in recent_ids:
            continue
        tags = e.get("tags", [])
        is_cascade = any(t.startswith("cascade_") for t in tags)
        if is_cascade:
            if any(t in cascade_tags for t in tags):
                pool.append(e)
        elif e["tier"] <= tier:
            pool.append(e)

    if not pool:
        pool = [e for e in events if e["tier"] <= tier]

    event = random.choice(pool)

    # 체인 설정
    if event.get("chain_to") and random.random() < event.get("chain_prob", 0):
        state["pending_chain"] = event["chain_to"]

    state["event_history"].append(event["id"])
    return event


def generate_hint(effects: dict) -> str:
    """효과 dict를 한국어 정성적 힌트 문자열로 변환한다."""
    parts = []
    for key in GAUGE_KEYS:
        val = effects.get(key, 0)
        if val == 0:
            continue
        mag = abs(val)
        size_word = "소폭"
        for lo, hi, word in HINT_MAGNITUDE:
            if lo <= mag <= hi:
                size_word = word
                break
        pos_word, neg_word = HINT_DIRECTION[key]
        direction = pos_word if val > 0 else neg_word
        parts.append(f"{GAUGE_NAMES[key]} {size_word} {direction}")
    return " / ".join(parts) + " 예상" if parts else "영향 미미"
