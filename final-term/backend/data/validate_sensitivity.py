"""B-17: 대표 이벤트/게이지 인과를 실제 역사 데이터로 고증 (측정 가능한 검증).

게임 인과 흐름: 정책 선택 → 게이지 변화 → ETF_SENSITIVITY → 섹터 변동.
따라서 "현실 고증"의 핵심 단위는 게이지→섹터 민감도(ETF_SENSITIVITY)다.
각 거시 게이지를 가장 잘 분리하는 역사적 에피소드와 매칭해, 모델이 만드는
섹터 순위가 실제 섹터 순위와 얼마나 닮았는지 Spearman 순위상관으로 측정한다.

PART A: 게이지→섹터 민감도 검증 (현재값 vs 수정안)
PART B: 대표 이벤트 3개를 엔진에 실제로 돌려 end-to-end 섹터 반응 측정

산출: data/validation_report.json
엔진 반영(B-18)은 W2末 hero CHECKPOINT 후. 여기선 측정·근거까지만.
"""
import json
import os
import sys

# 백엔드 모듈 import 경로 추가 (data/ → backend/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ETF_KEYS, ETF_SENSITIVITY  # noqa: E402

SECTORS = list(ETF_KEYS)
OUT_PATH = os.path.join(os.path.dirname(__file__), "validation_report.json")

# ── 실제 역사 에피소드의 섹터 순위 (event_study.json에서, best→worst) ──
# 각 게이지를 가장 잘 분리하는 에피소드와 매칭
REAL_RANKINGS = {
    "tension": {
        "episode": "러우 침공 (2022-02~03)",
        "order": ["energy", "defense", "consumer", "semiconductor", "finance"],
    },
    "inflation": {
        "episode": "2022 인플레/긴축기",
        "order": ["energy", "defense", "consumer", "finance", "semiconductor"],
    },
    "morale": {  # 민심=위험선호. 회복 랠리가 위험선호 상승을 가장 잘 분리
        "episode": "COVID 회복 랠리 (2020-03~08)",
        "order": ["semiconductor", "energy", "defense", "finance", "consumer"],
    },
}

# ── morale 열 수정안 (B-18 후보) ──
# 근거: 위험선호 상승(회복 랠리)에서 성장주(반도체)가 주도, 방어주(소비재)가 후행.
# 현재값은 consumer 0.7 > semiconductor 0.4 로 순위가 반대였음.
PROPOSED_MORALE = {
    "semiconductor": 0.7,  # 성장주 — 위험선호 시 주도
    "energy": 0.4,
    "finance": 0.3,
    "defense": 0.2,
    "consumer": 0.2,       # 방어주 — 위험선호 시 후행
}


def model_order(gauge: str, sensitivity: dict) -> list:
    """해당 게이지에 +쇼크를 줬을 때 모델이 만드는 섹터 순위 (best→worst)."""
    scored = [(etf, sensitivity[etf][gauge]) for etf in SECTORS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [etf for etf, _ in scored]


def spearman(order_a: list, order_b: list) -> float:
    """두 순위(섹터 리스트)의 Spearman 순위상관. 순위에 Pearson 적용."""
    rank_a = {s: i for i, s in enumerate(order_a)}
    rank_b = {s: i for i, s in enumerate(order_b)}
    xs = [rank_a[s] for s in SECTORS]
    ys = [rank_b[s] for s in SECTORS]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(cov / (vx * vy), 3) if vx and vy else 0.0


def part_a():
    """게이지→섹터 민감도 검증: 현재값 vs 수정안."""
    results = {}
    # 수정안 sensitivity 사본
    proposed = {etf: dict(ETF_SENSITIVITY[etf]) for etf in SECTORS}
    for etf in SECTORS:
        proposed[etf]["morale"] = PROPOSED_MORALE[etf]

    for gauge, real in REAL_RANKINGS.items():
        cur_order = model_order(gauge, ETF_SENSITIVITY)
        cur_rho = spearman(cur_order, real["order"])
        entry = {
            "real_episode": real["episode"],
            "real_order": real["order"],
            "current_model_order": cur_order,
            "current_spearman": cur_rho,
        }
        if gauge == "morale":
            prop_order = model_order(gauge, proposed)
            entry["proposed_model_order"] = prop_order
            entry["proposed_spearman"] = spearman(prop_order, real["order"])
        results[gauge] = entry
    return results


def part_b():
    """대표 이벤트 3개를 엔진에 실제로 돌려 섹터 반응 측정 (노이즈 평균)."""
    from engine import new_game, apply_choice

    # (event_id, choice_index, 매칭 게이지, 설명)
    REP_EVENTS = [
        ("nuclear_crisis", 2, "tension",
         "독자적 핵 억제력 개발 선언 — 긴장 고조(전쟁 패턴)"),
        ("global_recession", 2, "inflation",
         "통화 완화(양적완화) — 인플레 급등(긴축기 패턴)"),
        ("trade_boom", 1, "morale",
         "복지 예산 확대 — 민심/위험선호 상승(회복 랠리 패턴)"),
    ]
    from events import EVENTS
    event_map = {e["id"]: e for e in EVENTS}

    N = 3000  # 노이즈/분산 평균용 시행 횟수
    out = []
    for eid, cidx, gauge, desc in REP_EVENTS:
        event = event_map[eid]
        choice = event["choices"][cidx]
        # N회 시행, ETF % 변화 평균 (turn=1 고정 → 난이도 배수 1.0)
        sums = {etf: 0.0 for etf in SECTORS}
        for _ in range(N):
            st = new_game()
            apply_choice(st, choice, choice["label"])
            ch = st["etf_history"][-1]
            for etf in SECTORS:
                sums[etf] += ch[etf]
        avg = {etf: round(sums[etf] / N, 2) for etf in SECTORS}
        model_rank = sorted(SECTORS, key=lambda s: avg[s], reverse=True)
        real = REAL_RANKINGS[gauge]
        out.append({
            "event_id": eid,
            "choice": choice["label"],
            "description": desc,
            "matched_gauge": gauge,
            "avg_etf_pct": avg,
            "model_order": model_rank,
            "real_episode": real["episode"],
            "real_order": real["order"],
            "spearman": spearman(model_rank, real["order"]),
        })
    return out


def _patch_proposed_morale():
    """ETF_SENSITIVITY 객체의 morale 열을 수정안으로 in-place 패치.
    etf.py가 같은 dict 객체를 참조하므로 엔진 재실행에 즉시 반영된다."""
    for etf in SECTORS:
        ETF_SENSITIVITY[etf]["morale"] = PROPOSED_MORALE[etf]


def main():
    a = part_a()
    b = part_b()
    # 수정안을 엔진에 임시 적용 후 대표 이벤트 재실행 (end-to-end 개선 확인)
    _patch_proposed_morale()
    b_proposed = part_b()
    report = {
        "_meta": {
            "method": "게이지→섹터 모델 순위 vs 실제 역사 에피소드 순위, Spearman 순위상관",
            "note": "ρ=1 완벽일치, 0 무관, -1 정반대. B-18 반영 전 측정.",
        },
        "part_a_gauge_sensitivity": a,
        "part_b_representative_events": b,
        "part_b_with_proposed_morale": b_proposed,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[validate] 저장: {OUT_PATH}\n")
    print("═══ PART A: 게이지→섹터 민감도 검증 (Spearman ρ) ═══")
    for g, e in a.items():
        line = f"  {g:10s} vs {e['real_episode']:24s} : 현재 ρ={e['current_spearman']:+.3f}"
        if "proposed_spearman" in e:
            line += f"  →  수정안 ρ={e['proposed_spearman']:+.3f}"
        print(line)
        print(f"             실제: {e['real_order']}")
        print(f"             현재: {e['current_model_order']}")
        if "proposed_model_order" in e:
            print(f"             수정: {e['proposed_model_order']}")
    print("\n═══ PART B: 대표 이벤트 end-to-end (엔진 {}회 평균) ═══".format(3000))
    for e in b:
        print(f"  ■ {e['event_id']} / {e['choice']}  (ρ={e['spearman']:+.3f})")
        print(f"     {e['description']}")
        ranked = " > ".join(f"{s}({e['avg_etf_pct'][s]:+.1f})" for s in e["model_order"])
        print(f"     모델: {ranked}")
        print(f"     실제: {' > '.join(e['real_order'])} ({e['real_episode']})")
    print("\n═══ PART B': 수정안(morale) 적용 후 대표 이벤트 재측정 ═══")
    for e0, e1 in zip(b, b_proposed):
        delta = e1["spearman"] - e0["spearman"]
        mark = "→ 개선" if delta > 0.001 else ("→ 동일" if abs(delta) <= 0.001 else "→ 하락")
        print(f"  ■ {e1['event_id']:18s} ρ {e0['spearman']:+.2f} → {e1['spearman']:+.2f}  {mark}")


if __name__ == "__main__":
    main()
