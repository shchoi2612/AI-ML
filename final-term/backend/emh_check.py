"""EMH ρ 측정 하니스 — 패시브 드리프트의 ETF 분리가 정책→섹터 상관을 지키는지 검증.

emh.py(firewall)는 호출만 한다. 탐욕봇으로 임기를 끝까지 플레이한 뒤
analyze_emh(state)의 예측가능성(평균 |ρ|)과 섹터별 ρ를 모은다.

세 모드 비교:
  분리(현재)      : 드리프트는 게이지에만, ETF는 정책(카드)만        ← 지금 engine
  결합(이전)      : 드리프트가 ETF에도 섞임 (_commit 몽키패치로 재현)
  드리프트없음(기준): PASSIVE_DRIFT={} → 드리프트 도입 전 ρ

실행: backend/ 에서  .venv/bin/python emh_check.py
"""
import random
from collections import defaultdict

import engine
from engine import new_game, compute_capacity, refresh_sector_resources, apply_cards, check_game_over
from emh import analyze_emh
from balance_sim import bot_greedy
from config import ETF_KEYS, ETF_NAMES

GAMES = 300
SEED = 1234


def play_to_end(bot):
    """봇으로 끝까지 플레이하고 최종 state를 반환(분석용)."""
    state = new_game()
    while True:
        cap = compute_capacity(state)
        sel = bot(state, cap)
        apply_cards(state, sel)
        if check_game_over(state):
            return state
        refresh_sector_resources(state)


def measure(label, games=GAMES):
    preds, tops = [], []
    sector_r = defaultdict(list)
    for _ in range(games):
        state = play_to_end(bot_greedy)
        res = analyze_emh(state)
        preds.append(res["predictability_score"])
        tops.append(abs(res["top_correlation"]["value"]))
        # 섹터별 ρ 재계산(분석과 동일 입력)
        for etf in ETF_KEYS:
            sector_r[etf].append(_sector_rho(state, etf))
    avg_pred = sum(preds) / len(preds)
    avg_top = sum(tops) / len(tops)
    print(f"\n[{label}]  게임 {games}회")
    print(f"  평균 예측가능성(평균 |ρ|) : {avg_pred:.3f}")
    print(f"  평균 최고 상관 |ρ|        : {avg_top:.3f}")
    print("  섹터별 평균 ρ:")
    for etf in ETF_KEYS:
        vals = sector_r[etf]
        print(f"    {ETF_NAMES[etf]:<10} {sum(vals)/len(vals):+.3f}")
    return avg_pred


def _sector_rho(state, etf):
    from config import GAUGE_KEYS
    from emh import _pearson
    gh = state["gauge_history"]
    eh = state["etf_history"]
    gd = [sum(abs(gh[i][k] - gh[i - 1][k]) for k in GAUGE_KEYS) for i in range(1, len(gh))]
    ev = [e.get(etf, 0.0) for e in eh]
    n = min(len(gd), len(ev))
    return _pearson(gd[:n], ev[:n]) if n >= 3 else 0.0


def main():
    print("=" * 60)
    print(f"EMH ρ 측정 — 탐욕봇 {GAMES}회 (seed={SEED})")
    print("=" * 60)

    # A. 분리 (현재 engine)
    random.seed(SEED)
    a = measure("분리 (현재) — ETF=정책만")

    # B. 결합 (이전) — _commit이 etf_deltas를 무시하게 몽키패치
    orig = engine._commit
    def coupled(state, gauge_deltas, label, etf_deltas=None):
        return orig(state, gauge_deltas, label, etf_deltas=None)
    engine._commit = coupled
    random.seed(SEED)
    b = measure("결합 (이전) — ETF=드리프트+정책")
    engine._commit = orig

    # C. 드리프트 없음 (기준)
    saved = engine.PASSIVE_DRIFT
    engine.PASSIVE_DRIFT = {}
    random.seed(SEED)
    c = measure("드리프트 없음 (기준) — 도입 전")
    engine.PASSIVE_DRIFT = saved

    print("\n" + "=" * 60)
    print("요약 (평균 예측가능성, 1에 가까울수록 정책→섹터 신호 강함)")
    print(f"  드리프트없음(기준) : {c:.3f}")
    print(f"  분리(현재)         : {a:.3f}   ← 기준에 가까울수록 OK")
    print(f"  결합(이전)         : {b:.3f}   ← 희석된 값")
    print("=" * 60)


if __name__ == "__main__":
    main()
