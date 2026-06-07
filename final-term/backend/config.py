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
#   예: debt 50→10, debt 80→7, debt 100→5
BASE_FISCAL_CAPACITY = 10
DEBT_CAPACITY_BASELINE = 50
DEBT_CAPACITY_DIVISOR = 10
MIN_FISCAL_CAPACITY = 2

# 섹터 자원: 섹터 ETF가 100을 넘으면 매 턴 적립 → 그 섹터 카드 발동 재원.
#   accrue = max(0, (etf_price - 100)//SECTOR_ACCRUAL_DIVISOR), 상한 SECTOR_RESOURCE_CAP
# SECTOR_KEYS는 ETF_KEYS의 부분집합 (자원 게이팅 대상, 데모는 3섹터)
SECTOR_KEYS = ("energy", "defense", "semiconductor")
SECTOR_ACCRUAL_DIVISOR = 5
SECTOR_RESOURCE_CAP = 20

# ── LLM ──
GROQ_MODEL = "llama-3.3-70b-versatile"
NARRATION_MAX_TOKENS = 300
NARRATION_TEMPERATURE = 0.7
