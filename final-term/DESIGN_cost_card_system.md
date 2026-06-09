# Design Doc: 코스트 기반 카드 시스템 (EconSim v2 정책 레이어)

> 상태: **설계 단계 (코딩 전)**. 이 문서가 합의되면 영향 범위(아래 §5)대로 구현.
> 전제: 현재 hero 루프(정책→게이지/ETF→차트)는 LIVE로 작동 확인됨. 이 재설계는 **작동하는 토대 위에 자원 레이어를 얹는 것**이지, 갈아엎는 게 아니다.
> 최우선 원칙: **검증 파이프라인(validity firewall, §4)을 건드리지 않는다.**

---

## 1. 왜 바꾸나 (문제의식)

현재: `1 이벤트 → 선택지 3개 중 1개`. 매 턴 강제로 하나를 고르는 구조라 "예산 제약 하에서 우선순위를 정한다"는 경제 수장의 핵심 의사결정이 없다. 선택은 있지만 **자원 제약**이 없다.

목표: **코스트(자원) 레이어**를 도입해 "정책은 많지만 예산이 한정돼 무엇을 먼저 할지 고른다"를 게임의 중심 결정으로 만든다. 단, 이 레이어는 기존 인과/검증 골격 위에 **얹기만** 한다.

---

## 2. 핵심 결정 4개 (확정 — 문서에 박음)

### 결정 1 — 코스트 = 현실 경제 제약으로 프레이밍
두 종류의 자원으로 코스트를 표현한다.

- **재정 여력 (fiscal capacity)** — 매 턴 주어지는 기본 예산.
  **부채가 높을수록 여력이 줄어든다.**
  ```
  capacity(turn) = BASE_CAPACITY - debt_penalty(state.debt)
  # 예: BASE_CAPACITY=10,  debt_penalty = round(max(0, debt-50)/10)
  #     debt 50 → 여력 10,  debt 80 → 여력 7,  debt 100 → 여력 5
  ```
  → "빚이 많으면 쓸 수 있는 돈이 줄어든다"는 현실 제약을 게임 자원으로 직역.

- **섹터 자원 (sector resources)** — 섹터 ETF가 강하면 그 섹터 전용 자원이 쌓인다.
  ```
  sector_resource[s] += accrue(etf_price[s])
  # 예: accrue = max(0, round((etf_price[s]-100)/5))
  #     에너지 ETF 120 → 매 턴 +4 에너지 자원 적립
  ```
  → "잘 나가는 섹터가 그 섹터 정책의 재원을 댄다"는 프레이밍. 섹터 카드는 이 자원을 써야 발동.

각 카드는 **(재정 코스트, 섹터 코스트)** 두 축의 가격을 가진다. 일반 카드는 재정만, 섹터 카드는 재정+해당 섹터 자원을 소비.

### 결정 2 — 상시 카드풀 + 코스트 제한형 (손패 드로우 아님)
- 카드는 **항상 전부 보인다** (랜덤 손패 없음). 제약은 **예산**이다.
- 매 턴 플레이어는 **여력/섹터자원으로 감당 가능한 카드들의 부분집합**을 골라 쓴다.
  - 한 턴에 0~N장 가능 (총 재정 코스트 ≤ capacity, 각 섹터 카드의 섹터 코스트 ≤ 해당 섹터 자원).
- 프레이밍: **"정책은 많지만 예산이 한정돼 우선순위를 정한다."**
- 기존 "이벤트"는 폐기하지 않고 **상황/모디파이어 레이어**로 재배치(§3): 그 턴의 위기 상황을 제시하고, 특정 섹터 카드를 할인/할증하거나 여력을 일시 가감.

### 결정 3 — EMH 검증 보존 (validity firewall)
- 카드 → 게이지/섹터 인과 골격은 **그대로 유지**. 코스트는 그 위에 얹는 **자원 회계 레이어일 뿐**이다.
- 카드도 기존 선택지와 **동일한 `base_effects` 스키마**로 게이지에 작용한다. 따라서
  `base_effects → gauge_deltas → calculate_etf_changes(ETF_SENSITIVITY) → gauge_history/etf_history → Pearson ρ`
  파이프라인은 **한 줄도 안 바뀐다**. ρ(예측가능성) 검증 그대로 살아있음. (상세 §4)
- **패시브 드리프트의 ETF 분리 (밸런스 1단계 추가):** 매 턴 자동 악화(드리프트)는 게이지에만 반영하고 **ETF 신호에는 넣지 않는다**. 측정으로 검증된 근거 — 드리프트를 ETF에 섞은 '결합' 모드는 평균 |ρ|가 더 높아 보이지만(0.380 vs 분리 0.348), 그 상승분은 드리프트의 상수 푸시가 만든 **인공물**이었다(예: 방산 ETF ρ가 무드리프트 기준 −0.02 → 결합 −0.40으로 가짜 부풀림). ETF를 정책-only로 분리하면 ETF 가격이 검증된 `ETF_SENSITIVITY × 정책`만으로 생성돼 섹터별 ρ가 기준값에 수렴한다(방산 +0.14). **결합의 높은 ρ는 드리프트 인공물이었고, 분리가 정직한 설계다.** (`engine._commit`의 `etf_deltas` 분리; `emh.py`/`etf.py`/`ETF_SENSITIVITY` 무변경. 측정 하니스 `emh_check.py`)

### 결정 4 — 데모 범위
- **대표 카드 8~12장 + 섹터 2~3개(에너지·방위·반도체)부터.**
- 금융·소비재 ETF는 ρ 계산용으로 계속 산출하되, **섹터 자원 게이팅은 3섹터만** 우선.
- 카드는 기존 이벤트 선택지(약 72개)에서 대표 8~12개를 골라 `base_effects`를 재활용.
- 확장(카드 30+, 5섹터 전면 게이팅)은 나중.

---

## 3. 새 턴 루프

```
매 턴:
 1. capacity = BASE_CAPACITY - debt_penalty(debt)        # 여력 산출(부채 반영)
 2. sector_resources 갱신  (현재 etf_price로 적립)        # 섹터 자원 적립
 3. 카드풀 제시: 각 카드의 (재정코스트, 섹터, 섹터코스트, 힌트, 감당가능?) 표시
 4. (선택) 상황 이벤트 표시 — 여력/섹터코스트 모디파이어 적용
 5. 플레이어가 감당 가능한 카드 부분집합 선택
 6. 적용:
     gauge_deltas = Σ(선택 카드 base_effects) + variance      # ← 기존 effect 적용부 재사용
     게이지 클램프(0~100) → calculate_etf_changes → 히스토리   # ← 변경 없음(validity firewall)
     capacity/sector_resources에서 코스트 차감
 7. check_game_over → select_situation(다음 상황) → turn++
```

핵심: **6번의 게이지/ETF/히스토리 처리부는 현재 `apply_choice`의 후반부를 그대로 재사용.** 코스트는 5~6번 앞단의 게이팅·차감으로만 존재.

---

## 4. Validity Firewall — 절대 안 바꾸는 것

ρ 검증(루브릭 차별점)을 깨지 않으려면 아래는 **불변**:

| 건드리지 않음 | 이유 |
|---|---|
| `etf.py :: calculate_etf_changes` | gauge_deltas→ETF 인과. 섹터 자원은 etf_price를 *읽기만* 하지 이 계산을 안 바꿈 |
| `config.ETF_SENSITIVITY` | 검증된 섹터 민감도 행렬 |
| `emh.py` 전체 (Pearson, predictability) | gauge_history/etf_history만 읽음 — 이 둘은 §3-6에서 동일하게 채워짐 |
| `state.gauge_history` / `state.etf_history` 기록 방식 | 턴당 1 스냅샷. 멀티카드여도 "턴당 합산 1회"라 구조 동일 |

> 멀티카드 효과는 ρ에 **유리**할 수도 있다: 턴별 정책 강도(Σ게이지변화) 분산이 커져 상관 신호가 또렷해짐. emh.py는 그대로 두면 됨.

코스트 레이어가 새로 *읽는* 값은 `state.debt`(여력 계산), `state.etf_prices`(섹터 자원 적립)뿐 — 둘 다 읽기 전용이라 인과 파이프라인에 역류 없음.

---

## 5. 영향 범위 분석 (engine.py / events.py / config.py + α)

### 5.1 `config.py` — 추가만 (기존 상수 불변)
추가:
```python
BASE_FISCAL_CAPACITY = 10
DEBT_CAPACITY_PENALTY = ...        # debt→여력 차감 함수 파라미터
SECTOR_KEYS = ("energy", "defense", "semiconductor")   # ETF_KEYS의 부분집합(자원 게이팅용)
SECTOR_ACCRUAL = ...               # etf_price→섹터자원 적립 파라미터/상한
```
**불변:** `GAUGE_KEYS, ETF_KEYS, ETF_SENSITIVITY, DIFFICULTY_TIERS, CASCADE_THRESHOLDS, HINT_*`. (ETF_KEYS는 5종 유지 — ρ용. 자원은 그중 3종만.)

### 5.2 `cards.py` — 신규 파일
8~12장 카드 정의. 스키마(기존 choice + 코스트 필드):
```python
CARDS = [
  {"id":"strategic_oil_release", "title":"전략 비축유 방출",
   "sector":"energy", "fiscal_cost":3, "sector_cost":2,
   "base_effects":{"debt":5,"inflation":-8,"morale":3,"tension":-2}, "variance":2,
   "tags":["energy"]},
  ...
]
```
`base_effects`/`variance`는 기존 events.py 선택지에서 그대로 복사 → 인과 의미 보존.

### 5.3 `engine.py` — 부분 변경 (effect 적용부 재사용)
- `new_game()`: state에 `fiscal_capacity`, `sector_resources`(dict) 추가.
- 신규 `compute_capacity(state)`, `refresh_sector_resources(state)`, `affordable(state, card)`.
- `apply_choice(state, choice, label)` → **`apply_cards(state, card_ids)`** 로 일반화:
  - 선택 카드들의 `base_effects` 합산 → **기존 게이지 루프/ETF/히스토리 코드 그대로 호출**.
  - 코스트 차감(여력·섹터자원).
  - log/label은 카드 라벨들 join.
- `check_game_over`: **변경 없음**(선택). (원하면 "여력 0 연속 N턴" 소프트 패널티 추가 가능 — 데모 범위 밖.)
- `select_event` → `select_situation`으로 **역할 축소**(상황/모디파이어 제공). 티어/캐스케이드/체인 로직 재활용 가능.
- `generate_hint`: 카드 힌트에 **그대로 재사용**.

### 5.4 `events.py` — 유지하되 역할 변경
- 데이터 폐기 아님. "상황(situation)" 으로 재배치 — 위기 서사 + (옵션)모디파이어. 데모에선 일부만 써도 됨.
- 선택지(`choices`)는 카드 추출 소스로 사용.

### 5.5 `main.py` (API) — 계약 변경 (예고된 부분)
- `/game/new`: 응답에 `fiscal_capacity`, `sector_resources`, `card_pool`(각 카드 코스트/감당가능 플래그) 추가.
- `/game/action`: 요청 `{game_id, choice_index}` → **`{game_id, card_ids:[...]}`**. 응답에 `spent`, 갱신된 `fiscal_capacity`/`sector_resources` 추가.
- `narration`: ctx의 `choice_label` → 카드 라벨 목록으로. (스트림 구조 동일)
- `emh-summary`: **변경 없음.**

### 5.6 변경 없음 (firewall)
`etf.py`, `emh.py`, `config.ETF_SENSITIVITY` — **그대로**.

### 영향도 요약
| 파일 | 변경 강도 | 비고 |
|---|---|---|
| `config.py` | 낮음 (추가) | 기존 상수 불변 |
| `cards.py` | 신규 | 8~12장 |
| `engine.py` | 중간 | effect 적용 후반부 재사용, 앞단에 코스트 게이팅 |
| `events.py` | 낮음 | 역할만 상황/소스로 재배치 |
| `main.py` | 중간 | 계약 v2 (요청/응답 필드 추가) |
| `etf.py` `emh.py` | **0** | validity firewall |
| 프론트(석원) | 계약 v2 반영 | 카드풀 UI + 코스트 표시 + 다중선택 (석원 작업) |

---

## 6. PRD / TRD 업데이트 필요 항목

### PRD.md
- **입력 모델** ("이벤트마다 3개 선택지 버튼") → **"상시 카드풀 + 코스트 제한 다중선택"** 으로 교체.
- **게임 플로우** 다이어그램 3번("정책 선택지 3개") → "예산 내 카드 다중선택"으로.
- **신규 섹션 "코스트/자원 모델"** 추가 (재정 여력 + 섹터 자원).
- **EMH 검증** 섹션에 한 줄 추가: "코스트 레이어는 ρ 파이프라인에 영향 없음(validity 보존)". → 루브릭 validity 서사 강화.
- **범위 밖**에 "손패 드로우/덱빌딩" 명시적 제외 추가.

### TRD.md
- **API Contract**: `/game/new`·`/game/action` 스키마에 코스트/카드 필드 추가, 요청 바디 `choice_index → card_ids`. (계약 v2)
- **파일 구조**: `cards.py` 추가.
- **아키텍처 원칙(엔진↔뷰 분리)**: 유지. "카드는 이벤트처럼 데이터" 한 줄.
- **태스크 분해**: 신규 태스크 (engine 코스트 함수, cards.py, 계약 v2, 프론트 카드 UI). 기존 B-14 밸런싱은 카드 코스트 밸런싱까지 흡수.
- **명시**: `etf.py`/`emh.py` 불변(validity firewall).

> 결론: PRD/TRD **둘 다 업데이트 필요**하나, **추가·치환 수준**이지 재작성 아님. EMH/ETF 검증 서사는 오히려 강화됨.

---

## 7. 미해결/후속 (메모)

- **카드 밸런싱**: 코스트 ↔ effect 크기 매트릭스. 데모는 손으로 튜닝, 추후 데이터(B-15~18) 반영.
- **"한 턴 0장 허용?"**: 패스(자원 비축) 허용 여부 — 데모는 허용(전략적 비축) 권장.
- **상황 이벤트 모디파이어 깊이**: 데모는 서사만, 모디파이어는 옵션.
- **프론트 계약 v2**: 코스트 재설계 시작 시 석원님과 새 계약(카드/코스트 포함) 재공유.

---

## 부록 A. 베이스라인에서 발견된 핸드오프 항목 (석원님 프론트 / 백엔드 cosmetic)
코스트 작업과 별개로, 계약 검증 중 확인된 것:
- **(석원) 진행바 `.repeat(20 - progress)` 크래시** — `page.tsx`. **완주(turn 21) 경로에서 발동**. `Math.max(0, 20 - progress)` 또는 progress를 20으로 clamp. (역할분담: 프론트=석원)
- **(백엔드 cosmetic) `check_game_over`가 완주 시 리터럴 `"WIN"` 반환** — 프론트가 "WIN"을 게임오버 메시지로 그대로 노출(프론트의 "임기 완주!" 메시지가 가려짐). 계약 v2 손볼 때 승리 메시지 문자열로 교체하거나 null+프론트 처리로 정리. (지금은 베이스라인 유지 위해 미수정)
- 계약 4개(ETF 키/game_over 타입/요청 바디/응답 키)는 **이미 전부 일치 — 수정 불필요**.
