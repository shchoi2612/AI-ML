"""밸런스 시뮬레이터 — 봇으로 임기를 자동 플레이해 난이도를 측정한다.

봇:
  random  : 매 턴 감당 가능한 카드를 무작위로 0~2장.
  greedy  : 1-ply 룩어헤드(base_effects 결정값)로 '적용 후 안정도'가 최대가 되는
            감당 가능 조합 선택 (생존 최적화 탐욕봇).
  onetrack: 한 카드만 매 턴 반복(감당 가능하면) — 단일 전략 지배 여부 측정.

측정: 완주율 / 평균 생존 턴 / 사망 원인 분포 / (외길) 지배전략 유무.

엔진/카드/config만 사용. validity firewall(etf.py/emh.py/ETF_SENSITIVITY) 안 건드림.
실행: backend/ 에서  .venv/bin/python balance_sim.py
"""
import random
import itertools
from collections import Counter

from engine import (
    new_game, compute_capacity, refresh_sector_resources,
    card_affordable, apply_cards, check_game_over,
)
from cards import CARDS
from config import GAUGE_KEYS

GAMES = 500
SEED = 1234


def stability(state) -> int:
    return (100 - state["debt"]) + (100 - state["inflation"]) + state["morale"] + (100 - state["tension"])


def affordable_singles(state, cap):
    return [c for c in CARDS if card_affordable(c, cap, state["sector_resources"])]


def fits(subset, cap, resources) -> bool:
    if sum(c["fiscal_cost"] for c in subset) > cap:
        return False
    need = {}
    for c in subset:
        if c["sector"]:
            need[c["sector"]] = need.get(c["sector"], 0) + c["sector_cost"]
    return all(need[s] <= resources.get(s, 0) for s in need)


def affordable_subsets(state, cap, max_cards=3):
    singles = affordable_singles(state, cap)
    subsets = [[]]
    for r in range(1, min(max_cards, len(singles)) + 1):
        for combo in itertools.combinations(singles, r):
            if fits(combo, cap, state["sector_resources"]):
                subsets.append(list(combo))
    return subsets


def _apply_base(state, subset):
    """드리프트 + 카드 base_effects를 결정값으로 적용한 결과 게이지(클램프)."""
    from config import PASSIVE_DRIFT
    g = {k: state[k] for k in GAUGE_KEYS}
    for k, v in PASSIVE_DRIFT.items():
        g[k] = max(0, min(100, g[k] + v))
    for c in subset:
        for k, v in c["base_effects"].items():
            g[k] = max(0, min(100, g[k] + v))
    return g


def survival_score(state, subset) -> float:
    """생존 우선: 죽음(0/100)에서 가장 가까운 게이지의 여유를 최대화, 동률은 안정도로."""
    g = _apply_base(state, subset)
    margin = min(100 - g["debt"], 100 - g["inflation"], g["morale"], 100 - g["tension"])
    stab = (100 - g["debt"]) + (100 - g["inflation"]) + g["morale"] + (100 - g["tension"])
    return margin * 1000 + stab


# ── 봇 정책 ──
def bot_random(state, cap):
    subs = affordable_subsets(state, cap, max_cards=2)
    return random.choice(subs)


def bot_greedy(state, cap):
    best, best_score = [], survival_score(state, [])
    for sub in affordable_subsets(state, cap, max_cards=3):
        sc = survival_score(state, sub)
        if sc > best_score:
            best, best_score = sub, sc
    return best


def make_onetrack(card_id):
    def bot(state, cap):
        for c in affordable_singles(state, cap):
            if c["id"] == card_id:
                return [c]
        return []
    return bot


CAUSE = [
    ("국가 부도", "부채"),
    ("초인플레이션", "인플레"),
    ("혁명", "민심"),
    ("전쟁", "긴장"),
    ("임기 완주", "완주"),
]


def classify(msg):
    for key, label in CAUSE:
        if msg.startswith(key):
            return label
    return "기타"


def play_one(bot):
    state = new_game()
    while True:
        cap = compute_capacity(state)
        sel = bot(state, cap)
        apply_cards(state, sel)
        over = check_game_over(state)
        if over:
            return classify(over), state["turn"] - 1
        refresh_sector_resources(state)


def run(name, bot, games=GAMES):
    causes = Counter()
    turns = []
    for _ in range(games):
        cause, t = play_one(bot)
        causes[cause] += 1
        turns.append(t)
    comp = causes["완주"] / games
    avg = sum(turns) / len(turns)
    dist = ", ".join(f"{c} {causes[c]/games*100:.0f}%" for c, _ in
                     sorted(causes.items(), key=lambda x: -x[1]))
    return {"name": name, "completion": comp, "avg_turns": avg, "dist": dist}


def main():
    random.seed(SEED)
    print("=" * 68)
    print(f"EconSim 밸런스 시뮬레이션 — 봇당 {GAMES}회 (seed={SEED})")
    print("=" * 68)

    rows = [run("랜덤봇", bot_random), run("탐욕봇", bot_greedy)]
    print(f"\n{'봇':<10}{'완주율':>8}{'평균턴':>8}   사망원인 분포")
    print("-" * 68)
    for r in rows:
        print(f"{r['name']:<10}{r['completion']*100:>6.0f}%{r['avg_turns']:>8.1f}   {r['dist']}")

    print("\n[외길봇 — 단일 카드 반복] 지배전략 탐지")
    print("-" * 68)
    ot = []
    for c in CARDS:
        r = run(c["title"], make_onetrack(c["id"]), games=GAMES)
        ot.append((c["title"], r["completion"], r["avg_turns"]))
    ot.sort(key=lambda x: -x[1])
    for title, comp, avg in ot:
        print(f"  {title:<22}{comp*100:>5.0f}%  (평균 {avg:.1f}턴)")

    greedy = rows[1]["completion"]
    top_ot = ot[0]
    print("\n" + "=" * 68)
    print(f"탐욕봇 완주율 {greedy*100:.0f}%  (목표 40~60%)  →  ", end="")
    if greedy > 0.60:
        print("너무 쉬움 — 코스트/효과 더 조일 것")
    elif greedy < 0.40:
        print("너무 어려움 — 살짝 완화")
    else:
        print("목표 범위 ✓")
    if top_ot[1] >= 0.50:
        print(f"⚠ 지배전략 의심: '{top_ot[0]}' 단일 반복으로 완주율 {top_ot[1]*100:.0f}%")
    else:
        print(f"단일 카드 지배전략 없음 (최고 외길 '{top_ot[0]}' {top_ot[1]*100:.0f}%)")
    print("=" * 68)


if __name__ == "__main__":
    main()
