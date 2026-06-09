"""게임 상수 및 설정 — 모든 튜닝 가능한 값을 한 곳에서 관리."""

MAX_TURNS = 20

INITIAL_STATE = {
    "debt": 50,
    "inflation": 40,
    "morale": 60,
    "tension": 30,
}

GAUGE_NAMES = {
    "debt": "부채",
    "inflation": "인플레이션",
    "morale": "민심",
    "tension": "국제긴장도",
}

GAUGE_KEYS = ("debt", "inflation", "morale", "tension")

# 게이지별 위험 방향 — high: 높을수록 위험, low: 낮을수록 위험
GAUGE_DANGER = {
    "debt": "high",
    "inflation": "high",
    "morale": "low",
    "tension": "high",
}

# ── ETF ──
ETF_NAMES = {
    "semiconductor": "반도체 ETF",
    "energy": "에너지 ETF",
    "finance": "금융 ETF",
    "defense": "방산 ETF",
    "consumer": "소비재 ETF",
}

ETF_KEYS = ("semiconductor", "energy", "finance", "defense", "consumer")

# 게이지 변화 → ETF 가격 민감도 (가중치)
ETF_SENSITIVITY = {
    "semiconductor": {"debt": -0.3, "inflation": -0.5, "morale": 0.4, "tension": -0.6},
    "energy":        {"debt": -0.2, "inflation": 0.6,  "morale": 0.1, "tension": 0.8},
    "finance":       {"debt": -0.7, "inflation": -0.4, "morale": 0.3, "tension": -0.3},
    "defense":       {"debt": 0.1,  "inflation": 0.0,  "morale": -0.2, "tension": 0.9},
    "consumer":      {"debt": -0.3, "inflation": -0.6, "morale": 0.7, "tension": -0.2},
}

ETF_NOISE_RANGE = 1.5  # ETF 가격 랜덤 노이즈 ±

# ── 난이도 곡선 ──
# (시작턴, 끝턴, 분산 배수)
DIFFICULTY_TIERS = [
    (1, 7, 1.0),    # 초반: base variance 그대로
    (8, 14, 1.5),   # 중반: variance × 1.5
    (15, 20, 2.0),  # 후반: variance × 2.0
]

# ── 캐스케이드 임계값 ──
# 게이지가 threshold를 넘으면 해당 이벤트 풀 해금
CASCADE_THRESHOLDS = [
    {"gauge": "debt", "threshold": 70, "direction": "above", "tags": ["cascade_debt"]},
    {"gauge": "morale", "threshold": 30, "direction": "below", "tags": ["cascade_morale"]},
    {"gauge": "inflation", "threshold": 75, "direction": "above", "tags": ["cascade_inflation"]},
    {"gauge": "tension", "threshold": 80, "direction": "above", "tags": ["cascade_tension"]},
]

# ── 힌트 매핑 ──
# 효과 크기 → 한국어 정성적 표현
HINT_MAGNITUDE = [
    (1, 3, "소폭"),
    (4, 7, "상당히"),
    (8, 12, "대폭"),
    (13, 100, "극심한"),
]

HINT_DIRECTION = {
    "debt": ("증가", "감소"),
    "inflation": ("상승", "하락"),
    "morale": ("개선", "악화"),
    "tension": ("고조", "완화"),
}

# ── 코스트/자원 레이어 (v2 카드 시스템) ──
# 매 턴 재정 여력(fiscal capacity): 부채가 높을수록 줄어든다 (현실 제약 직역).
#   capacity = BASE_FISCAL_CAPACITY - max(0, debt - DEBT_CAPACITY_BASELINE)//DEBT_CAPACITY_DIVISOR
#   (최소 MIN_FISCAL_CAPACITY로 클램프)
#   예: debt 50→4, debt 70→2, debt 90→1  (빠듯하게: 매 턴 1~2장만)
BASE_FISCAL_CAPACITY = 4
DEBT_CAPACITY_BASELINE = 50
DEBT_CAPACITY_DIVISOR = 10
MIN_FISCAL_CAPACITY = 1

# 섹터 자원: 섹터 ETF가 100을 넘으면 매 턴 적립 → 그 섹터 카드 발동 재원.
#   accrue = max(0, (etf_price - 100)//SECTOR_ACCRUAL_DIVISOR), 상한 SECTOR_RESOURCE_CAP
# SECTOR_KEYS는 ETF_KEYS의 부분집합 (자원 게이팅 대상, 데모는 3섹터)
SECTOR_KEYS = ("energy", "defense", "semiconductor")
SECTOR_ACCRUAL_DIVISOR = 5
SECTOR_RESOURCE_CAP = 20

# 패시브 드리프트: 매 턴 자동으로 가벼운 악화. 이벤트가 주(主)압력을 담당하므로
# 드리프트는 '잔잔한 배경'으로 약화(과거 3/2/-2/2 → 절반). 무위=손해는 이벤트 충격이 보장.
PASSIVE_DRIFT = {"debt": 2, "inflation": 1, "morale": -1, "tension": 1}

# ── 부채 억제 메커닉 ──
# 현실의 국가부채처럼: 0으로 "해결"되지 않고, 최선이 악화 둔화/억제다.
#   (1) 부채 바닥(floor): 부채는 MIN_DEBT 밑으로 못 내려간다 → 절대 0이 안 됨.
#   (2) 감소 댐핑: 정책의 '부채 감소'분은 DEBT_REDUCTION_FACTOR로 약화 → 갚기 어렵다.
#       (드리프트/이벤트의 부채 증가는 그대로. 즉 쌓기는 쉽고 줄이기는 더디다.)
MIN_DEBT = 25
DEBT_REDUCTION_FACTOR = 0.8
# 게이지별 하한 (기본 0, 부채만 MIN_DEBT). _commit 클램프에서 사용.
GAUGE_FLOOR = {"debt": MIN_DEBT, "inflation": 0, "morale": 0, "tension": 0}

# ── 이벤트 강도(severity) 시스템 ──
# 매 턴 뜨는 이벤트가 게이지를 직접 때린다(드리프트/정책과 별개의 외생 충격).
# 강도: light(가벼운 사건) / medium(중간 위기) / major(큰 위기 — "이번 턴 큰일났다").
# 이벤트 충격은 게이지에만 반영하고 ETF 신호엔 넣지 않는다(EMH ρ 보호: ETF=정책만).
# 전역 튜닝 노브 — 완주율이 너무 낮으면 이 값을 낮춰 충격을 완화한다.
# 0.80: 탐욕봇 완주율 50%(목표 40~60% 중앙), 사망 원인 균형(부채/민심), 외길 비지배. balance_sim 검증.
EVENT_IMPACT_SCALE = 0.80
SEVERITY_LABELS = {"light": "사건", "medium": "위기", "major": "중대 위기"}

# ── 위기 대응 핸들 (event response handle) ──
# 위기가 게이지를 때리는 동시에, 그 위기를 '막는' 카드를 그 턴 싸게 만든다.
# 위기가 위협하는 게이지(impact)를 돕는 방향으로 움직이는 카드 = 그 위기의 '대응'.
# 강도별 코스트 할인 (재정, 섹터). 클램프 0. → 빠듯한 여력으로도 위기 대응이 가능.
# 섹터-프리 기본 대응(긴급 외교 성명 등)이 있어 초반에도 답이 막히지 않는다(cards 참조).
EVENT_RESPONSE_DISCOUNT = {"light": (0, 0), "medium": (1, 1), "major": (2, 2)}

# ── LLM ──
GROQ_MODEL = "llama-3.3-70b-versatile"
NARRATION_MAX_TOKENS = 300
NARRATION_TEMPERATURE = 0.7
