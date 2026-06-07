"""정책 카드 정의 — 코스트 기반 카드 시스템 (v2).

상시 카드풀: 모든 카드는 항상 보이고, 매 턴 '재정 여력 + 섹터 자원'으로
감당 가능한 부분집합을 골라 쓴다 (손패 드로우 아님).

각 카드:
    id           : 고유 id
    title        : 표시 라벨
    sector       : 섹터 자원을 쓰는 섹터 (None이면 일반 카드, 섹터 자원 불필요)
                   값은 config.SECTOR_KEYS 중 하나: energy / defense / semiconductor
    fiscal_cost  : 재정 여력에서 소비 (모든 카드)
    sector_cost  : 해당 섹터 자원에서 소비 (sector=None이면 0)
    base_effects : 게이지 효과 (기존 events.py 선택지 스키마 그대로 → 인과 보존)
    variance     : 랜덤 분산 (engine이 난이도 배수로 스케일)
    tags         : 분류용

base_effects/variance는 기존 events.py 선택지에서 가져와 인과 의미를 보존한다.
이 값이 gauge_deltas → calculate_etf_changes → Pearson ρ 파이프라인을 그대로 탄다.
(validity firewall: etf.py / emh.py / ETF_SENSITIVITY 불변)
"""

CARDS = [
    # ── 일반 카드 (sector=None, 섹터 자원 불필요) ──
    {
        "id": "interest_rate_hike", "title": "기준금리 인상",
        "sector": None, "fiscal_cost": 2, "sector_cost": 0,
        "base_effects": {"debt": -3, "inflation": -6, "morale": -7}, "variance": 2,
        "tags": ["monetary"],
    },
    {
        "id": "issue_bonds", "title": "국채 발행 경기부양",
        "sector": None, "fiscal_cost": 1, "sector_cost": 0,
        "base_effects": {"debt": 10, "inflation": 3, "morale": 5}, "variance": 3,
        "tags": ["fiscal"],
    },
    {
        "id": "welfare_expansion", "title": "복지 예산 확대",
        "sector": None, "fiscal_cost": 4, "sector_cost": 0,
        "base_effects": {"debt": 6, "inflation": 4, "morale": 10}, "variance": 2,
        "tags": ["fiscal", "social"],
    },
    {
        "id": "austerity", "title": "긴축 재정 패키지",
        "sector": None, "fiscal_cost": 1, "sector_cost": 0,
        "base_effects": {"debt": -8, "inflation": -3, "morale": -12}, "variance": 3,
        "tags": ["fiscal"],
    },

    # ── 에너지 섹터 카드 ──
    {
        "id": "strategic_oil_release", "title": "전략 비축유 방출",
        "sector": "energy", "fiscal_cost": 3, "sector_cost": 2,
        "base_effects": {"debt": 5, "inflation": -8, "morale": 3, "tension": -2}, "variance": 2,
        "tags": ["energy"],
    },
    {
        "id": "renewable_investment", "title": "신재생 에너지 대규모 투자",
        "sector": "energy", "fiscal_cost": 4, "sector_cost": 3,
        "base_effects": {"debt": 8, "inflation": -2, "morale": 6, "tension": -3}, "variance": 2,
        "tags": ["energy"],
    },

    # ── 방위 섹터 카드 ──
    {
        "id": "defense_buildup", "title": "군비 증강·방위비 확대",
        "sector": "defense", "fiscal_cost": 5, "sector_cost": 3,
        "base_effects": {"debt": 12, "inflation": 3, "morale": -5, "tension": -8}, "variance": 3,
        "tags": ["defense", "military"],
    },
    {
        "id": "diplomatic_mediation", "title": "다자간 외교 중재",
        "sector": "defense", "fiscal_cost": 2, "sector_cost": 1,
        "base_effects": {"morale": 3, "inflation": 2, "tension": -10}, "variance": 3,
        "tags": ["defense", "diplomacy"],
    },

    # ── 반도체 섹터 카드 ──
    {
        "id": "chip_subsidy", "title": "반도체 산업 긴급 보조금",
        "sector": "semiconductor", "fiscal_cost": 5, "sector_cost": 3,
        "base_effects": {"debt": 10, "inflation": -3, "morale": 5, "tension": 3}, "variance": 2,
        "tags": ["semiconductor", "tech"],
    },
    {
        "id": "fab_construction", "title": "반도체 팹 국비 건설",
        "sector": "semiconductor", "fiscal_cost": 6, "sector_cost": 4,
        "base_effects": {"debt": 15, "inflation": -2, "morale": 6, "tension": -3}, "variance": 3,
        "tags": ["semiconductor", "tech"],
    },
]

CARDS_BY_ID = {c["id"]: c for c in CARDS}
