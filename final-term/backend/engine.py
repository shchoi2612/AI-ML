"""경제 게임 엔진 — 게임 상태, 게이지 계산, 이벤트 선택."""
import random
from config import (
    INITIAL_STATE, GAUGE_KEYS, GAUGE_NAMES, MAX_TURNS,
    DIFFICULTY_TIERS, CASCADE_THRESHOLDS, ETF_KEYS,
    HINT_MAGNITUDE, HINT_DIRECTION,
    BASE_FISCAL_CAPACITY, DEBT_CAPACITY_BASELINE, DEBT_CAPACITY_DIVISOR,
    MIN_FISCAL_CAPACITY, SECTOR_KEYS, SECTOR_ACCRUAL_DIVISOR, SECTOR_RESOURCE_CAP,
    PASSIVE_DRIFT, GAUGE_FLOOR, DEBT_REDUCTION_FACTOR, EVENT_IMPACT_SCALE,
    EVENT_RESPONSE_DISCOUNT,
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
        # 코스트/자원 레이어 (v2)
        "sector_resources": {s: 0 for s in SECTOR_KEYS},
        "fiscal_capacity": compute_capacity(INITIAL_STATE),
        # 이번 턴 이벤트의 직접 충격(게이지) + 강도. select_event가 채우고 apply_cards가 소비.
        "pending_impact": {},
        "pending_severity": "light",
    }


# ── 위기 대응 핸들 (event response handle) ────────────────────────────
def _threatened_gauges(impact: dict) -> set:
    """이벤트가 '악화'시키는 게이지 집합 (부채/인플레/긴장↑, 민심↓)."""
    out = set()
    for g, v in (impact or {}).items():
        if g == "morale":
            if v < 0:
                out.add(g)
        elif v > 0:
            out.add(g)
    return out


def _card_counters(card: dict, threatened: set) -> bool:
    """카드가 위협받는 게이지를 '돕는' 방향으로 움직이면 그 위기의 대응 카드다."""
    eff = card.get("base_effects", {})
    for g in threatened:
        v = eff.get(g, 0)
        if g == "morale" and v > 0:
            return True
        if g != "morale" and v < 0:
            return True
    return False


def effective_cost(card: dict, state: dict) -> tuple[int, int]:
    """이번 턴 위기 대응 할인을 반영한 (재정 코스트, 섹터 코스트).

    위기가 위협하는 게이지를 돕는 카드는 강도별로 코스트가 깎인다(클램프 0).
    할인은 코스트(게이팅)에만 적용 — base_effects/ETF/ρ 파이프라인은 불변(firewall).
    """
    fiscal = card["fiscal_cost"]
    sector = card.get("sector_cost", 0)
    threatened = _threatened_gauges(state.get("pending_impact", {}))
    if threatened and _card_counters(card, threatened):
        df, ds = EVENT_RESPONSE_DISCOUNT.get(state.get("pending_severity", "light"), (0, 0))
        fiscal = max(0, fiscal - df)
        sector = max(0, sector - ds)
    return fiscal, sector


# ── 코스트/자원 레이어 (v2) ────────────────────────────────────────────
def compute_capacity(state: dict) -> int:
    """현재 부채를 반영한 이번 턴의 재정 여력(use-it-or-lose-it)."""
    penalty = max(0, state["debt"] - DEBT_CAPACITY_BASELINE) // DEBT_CAPACITY_DIVISOR
    return max(MIN_FISCAL_CAPACITY, BASE_FISCAL_CAPACITY - penalty)


def refresh_sector_resources(state: dict) -> dict:
    """섹터 ETF 성과로 섹터 자원을 적립(누적)하고 상한으로 클램프한다."""
    res = state["sector_resources"]
    for s in SECTOR_KEYS:
        accrue = max(0, int((state["etf_prices"].get(s, 100.0) - 100.0) // SECTOR_ACCRUAL_DIVISOR))
        res[s] = min(SECTOR_RESOURCE_CAP, res[s] + accrue)
    return res


def card_affordable(card: dict, capacity: int, resources: dict, state: dict | None = None) -> bool:
    """단일 카드가 현재 여력/섹터자원으로 감당 가능한지(위기 대응 할인 반영)."""
    fiscal, sector_cost = effective_cost(card, state) if state is not None else (
        card["fiscal_cost"], card.get("sector_cost", 0))
    if fiscal > capacity:
        return False
    sector = card.get("sector")
    if sector is not None and sector_cost > resources.get(sector, 0):
        return False
    return True


def validate_selection(state: dict, cards: list) -> str | None:
    """선택한 카드 묶음의 총코스트가 예산 내인지 검증(할인 반영). 위반 시 사유, OK면 None."""
    capacity = compute_capacity(state)
    total_fiscal = sum(effective_cost(c, state)[0] for c in cards)
    if total_fiscal > capacity:
        return f"재정 여력 초과 (필요 {total_fiscal} > 여력 {capacity})"
    sector_need: dict = {}
    for c in cards:
        s = c.get("sector")
        if s is not None:
            sector_need[s] = sector_need.get(s, 0) + effective_cost(c, state)[1]
    for s, need in sector_need.items():
        have = state["sector_resources"].get(s, 0)
        if need > have:
            return f"{s} 섹터 자원 초과 (필요 {need} > 보유 {have})"
    return None


def _get_difficulty_multiplier(turn: int) -> float:
    """현재 턴에 해당하는 난이도 분산 배수를 반환한다."""
    for start, end, mult in DIFFICULTY_TIERS:
        if start <= turn <= end:
            return mult
    return 1.0


def _roll_deltas(base: dict, variance: int, turn: int) -> dict:
    """base_effects + 난이도 스케일 랜덤 분산으로 gauge_deltas를 굴린다."""
    scaled_var = int(variance * _get_difficulty_multiplier(turn))
    deltas = {}
    for key in GAUGE_KEYS:
        base_val = base.get(key, 0)
        if base_val == 0 and scaled_var == 0:
            continue
        deltas[key] = base_val + random.randint(-scaled_var, scaled_var)
    return deltas


def _commit(state: dict, gauge_deltas: dict, label: str, etf_deltas: dict | None = None) -> dict:
    """gauge_deltas를 게이지에 적용 → ETF 갱신 → 히스토리 기록 → 턴 증가.

    이 부분이 validity 파이프라인(deltas → calculate_etf_changes →
    gauge_history/etf_history)이다. etf.py / 기록 방식은 불변.

    etf_deltas가 주어지면 ETF는 그 값으로만 갱신한다(게이지는 gauge_deltas).
    패시브 드리프트를 게이지엔 반영하되 ETF 신호엔 섞지 않기 위한 분리:
    ETF는 '플레이어 정책'에만 반응해야 EMH ρ(정책→섹터) 측정이 깨끗하다.
    None이면 gauge_deltas로 ETF를 갱신한다(v1 호환).
    """
    changes_text = []
    for key, delta in gauge_deltas.items():
        # 게이지별 하한 클램프 — 부채는 MIN_DEBT 밑으로 못 내려감(0으로 "해결" 불가).
        state[key] = max(GAUGE_FLOOR.get(key, 0), min(100, state[key] + delta))
        sign = "+" if delta > 0 else ""
        changes_text.append(f"{GAUGE_NAMES[key]} {sign}{delta}")

    etf_input = gauge_deltas if etf_deltas is None else etf_deltas
    new_prices, pct_changes = calculate_etf_changes(etf_input, state["etf_prices"])
    state["etf_prices"] = new_prices
    state["etf_history"].append(pct_changes)

    state["gauge_history"].append({k: state[k] for k in GAUGE_KEYS})
    state["log"].append(f"턴 {state['turn']}: [{label}] → {', '.join(changes_text) or '변화 없음'}")
    state["turn"] += 1
    return gauge_deltas


def apply_choice(state: dict, choice: dict, label: str) -> dict:
    """[v1 호환] 단일 선택지의 효과를 적용하고 gauge_deltas를 반환한다."""
    deltas = _roll_deltas(choice["base_effects"], choice.get("variance", 0), state["turn"])
    return _commit(state, deltas, label)


def apply_cards(state: dict, cards: list) -> dict:
    """[v2] 선택한 카드 묶음의 효과 합산을 적용하고, 섹터 자원을 차감한다.

    재정 여력은 매 턴 새로 계산되는 예산이라 차감 대상이 아니다(use-it-or-lose-it).
    섹터 자원은 누적되므로 사용분만큼 차감한다.
    카드 0장(패스)도 허용: 게이지 변화 없이 시장 노이즈만 반영하며 턴이 진행된다.

    Returns:
        합산 적용된 gauge_deltas dict
    """
    # 섹터 차감액을 '할인 반영(effective_cost)'으로 먼저 확정한다 — pending_* 비우기 전에.
    sector_spend: dict = {}
    for c in cards:
        s = c.get("sector")
        if s is not None:
            scost = effective_cost(c, state)[1]
            if scost:
                sector_spend[s] = sector_spend.get(s, 0) + scost

    # 정책(카드) 효과부터 합산. 이게 ETF 신호의 유일한 입력이다(EMH ρ 보호).
    policy: dict = {}
    for c in cards:
        d = _roll_deltas(c["base_effects"], c.get("variance", 0), state["turn"])
        for k, v in d.items():
            policy[k] = policy.get(k, 0) + v

    # 부채 감소 댐핑: 정책의 '순 부채 감소'분만 약화(증가는 그대로). 갚기는 더디다.
    if policy.get("debt", 0) < 0:
        policy["debt"] = int(round(policy["debt"] * DEBT_REDUCTION_FACTOR))

    # 게이지엔 드리프트 + 이벤트 충격 + 정책을 모두 합산(무위=손해, 위기는 직격).
    # ETF엔 정책(policy)만 넘긴다 → 드리프트/이벤트 같은 외생 충격은 ρ를 오염시키지 않음.
    combined: dict = dict(PASSIVE_DRIFT)
    impact = state.get("pending_impact", None) or {}
    for k, v in impact.items():
        combined[k] = combined.get(k, 0) + int(round(v * EVENT_IMPACT_SCALE))
    for k, v in policy.items():
        combined[k] = combined.get(k, 0) + v
    state["pending_impact"] = {}            # 소비 후 비움
    state["pending_severity"] = "light"

    label = " + ".join(c.get("title", c.get("label", "?")) for c in cards) or "정책 보류(패스)"
    deltas = _commit(state, combined, label, etf_deltas=policy)

    # 섹터 자원 차감 (누적 자원만, 할인 반영분)
    for s, amt in sector_spend.items():
        state["sector_resources"][s] = max(0, state["sector_resources"][s] - amt)

    return deltas


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
                state["pending_impact"] = dict(e.get("impact", {}))
                state["pending_severity"] = e.get("severity", "light")
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
    state["pending_impact"] = dict(event.get("impact", {}))
    state["pending_severity"] = event.get("severity", "light")
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
