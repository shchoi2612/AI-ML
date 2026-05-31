# TRD: EconSim — 기술 설계 및 태스크 분해

## 스택

| 레이어 | 기술 | 역할 |
|--------|------|------|
| 백엔드 | FastAPI (Python 3.12) | 게임 엔진, API, LLM 연동 |
| 프론트엔드 | Next.js 14 (React/TypeScript) | UI, 차트, VN 렌더링 |
| 차트 | Lightweight Charts (TradingView OSS) | 실시간 ETF 캔들/라인 |
| 애니메이션 | Framer Motion | 게이지 전환, 말풍선 등장 |
| LLM | Groq API (llama-3.3-70b-versatile) | 한국어 뉴스 나레이션 |
| 포트레이트 | AI 생성 (DALL-E/Midjourney) | VN 페르소나 3~5장 |

## 아키텍처 원칙: 엔진 ↔ 뷰 분리

엔진(Python)은 JSON만 반환. 뷰(React)는 JSON만 소비. 둘 사이 계약은 API contract(아래).

확장 시 영향:
- 이벤트 추가 → events.py에 dict 추가 (엔진/뷰 변경 없음)
- VN 아트 추가 → portraits/에 이미지 추가 (엔진/뷰 변경 없음)
- 차트 라이브러리 교체 → EtfChart.tsx 1개만 교체

## API Contract

### POST /game/new → GameState

```json
{
  "game_id": "uuid",
  "turn": 1,
  "gauges": {"debt": 50, "inflation": 40, "morale": 60, "tension": 30},
  "etf_prices": {"semiconductor": 100, "energy": 100, "finance": 100, "defense": 100, "consumer": 100},
  "event": {
    "id": "oil_crisis",
    "title": "중동 군사 충돌로 유가 급등",
    "desc": "...",
    "choices": [
      {"label": "전략 비축유 방출", "hint": "부채 소폭 증가, 인플레이션 대폭 하락 예상"},
      ...
    ]
  }
}
```

### POST /game/action → 즉시 반환 (<1초)

Request: `{"game_id": "uuid", "choice_index": 0}`

```json
{
  "turn": 2,
  "gauges": {"debt": 55, "inflation": 32, "morale": 63, "tension": 28},
  "gauge_deltas": {"debt": 5, "inflation": -8, "morale": 3, "tension": -2},
  "etf_prices": {"semiconductor": 99.2, ...},
  "etf_changes": {"semiconductor": -0.8, ...},
  "next_event": { ... },
  "game_over": null
}
```

### GET /game/{game_id}/narration → SSE 스트리밍

차트 업데이트 후 별도 요청. Groq 레이턴시를 hero 순간에서 분리.

```json
{
  "narration": "정부의 전략 비축유 방출 결정으로...",
  "vn_dialogue": {
    "speaker": "재무장관",
    "portrait": "finance_minister",
    "text": "각하, 비축유 방출은 단기적으로..."
  }
}
```

### GET /game/{game_id}/emh-summary → EmhReport

임기 완주 시에만 호출.

```json
{
  "total_turns": 20,
  "predictability_score": 0.73,
  "stability_score": 312,
  "sector_correlations": {"semiconductor": 0.65, ...},
  "summary_text": "당신의 정책은 시장에 73% 예측 가능한 패턴을..."
}
```

## 파일 구조

```
econ-game/
├── backend/
│   ├── main.py              # FastAPI 라우터 + CORS
│   ├── engine.py            # 게이지 계산, 이벤트 선택 (v1 재활용)
│   ├── events.py            # 22개 이벤트 데이터 (v1 재활용)
│   ├── config.py            # 상수/임계값/민감도 (v1 재활용)
│   ├── etf.py               # ETF 가격 계산 (v1 재활용)
│   ├── narration.py         # Groq LLM 스트리밍 (v1 재활용)
│   ├── emh.py               # EMH Pearson 분석 (신규)
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── package.json
│   ├── app/                  # Next.js App Router
│   │   ├── layout.tsx
│   │   ├── page.tsx          # 메인 게임 화면
│   │   └── globals.css
│   ├── components/
│   │   ├── PolicyInput.tsx   # 선택지 피커 (채팅 스타일)
│   │   ├── EtfChart.tsx      # Lightweight Charts 캔들/라인
│   │   ├── GaugePanel.tsx    # 4개 게이지 바 (색상 위험구간)
│   │   ├── VNDialogue.tsx    # 말풍선 + 포트레이트
│   │   └── EmhSummary.tsx    # 임기말 성적표
│   ├── lib/
│   │   └── api.ts            # fetch wrapper
│   └── public/
│       └── portraits/        # AI 생성 포트레이트
├── PRD.md
├── TRD.md
├── ref/                      # 기존 레포 + Streamlit v1
└── v1-backup/
```

## 의존성 (버전 고정)

### backend/requirements.txt
```
fastapi==0.115.0
uvicorn==0.30.0
groq==0.12.0
python-dotenv==1.0.1
numpy==1.26.4
scipy==1.14.0
pydantic==2.9.0
```

### frontend/package.json (핵심)
```
next: ^14.2
react: ^18.3
lightweight-charts: ^4.2
framer-motion: ^11.0
```

---

# 태스크 분해

## 마일스톤

| 마일스톤 | 기한 | 정의 |
|----------|------|------|
| M0: One-Click Loop | W1 D3 | /game/new → 선택지 1개 클릭 → /game/action → ETF 1종 차트에 새 점 추가 (정책→차트 한 루프 증명) |
| M1: Hero Loop | W2 末 | 5종 ETF 멀티차트 + 4게이지 + 힌트 + 애니메이션 (hero 순간 완성) |
| **CHECKPOINT** | **W2 末 / W3 D1** | **hero 순간이 데모 가능한가? NO → Streamlit fallback (FB-01~05)** |
| M2: Full Loop | W3 末 | M1 + LLM 나레이션 (SSE 또는 템플릿 폴백) + VN 말풍선 포트레이트 1장 |
| M3: Polish | W4 末 | EMH 성적표 (숫자 3개 + 한 문장), 게임오버 화면, cross-review |
| M4: Ship | W5 末 | PPT, 포스터, 데모 영상, 최종 제출 |

---

## 태스크 목록

우선순위: P0 = M0 블로커, P1 = M1 블로커, P2 = M2 블로커, P3 = M3 이후

### Phase 0: Skeleton (W1 D1~D3)

| ID | 담당 | 태스크 | 우선순위 | 산출물 |
|----|------|--------|----------|--------|
| B-01 | Backend | git init, backend/ 폴더 생성, venv, requirements.txt | P0 | 빈 FastAPI 서버 기동 |
| B-02 | Backend | v1 로직 이전: engine.py, events.py, config.py, etf.py를 backend/에 복사 + import 정리 | P0 | 모듈 import 성공 |
| B-03 | Backend | main.py: POST /game/new 엔드포인트 (new_game() + select_event() 호출, JSON 반환) | P0 | curl로 GameState JSON 수신 |
| B-04 | Backend | main.py: POST /game/action 엔드포인트 (apply_choice() + ETF 계산, game_id로 세션 관리) | P0 | curl로 deltas+etf JSON 수신 |
| B-05 | Backend | CORS 설정 (localhost:3000 허용) | P0 | 프론트에서 fetch 성공 |
| B-06 | Both | Contract fixtures 생성: fixtures/new_game.json, fixtures/action_response.json. 프론트 NEXT_PUBLIC_API_MODE=mock\|live 지원 | P0 | 백/프론트 독립 개발 가능 |
| F-01 | Frontend | npx create-next-app, 기본 레이아웃, 다크 테마 CSS | P0 | localhost:3000 빈 페이지 |
| F-02 | Frontend | lib/api.ts: fetch wrapper + mock mode (fixture JSON 반환) | P0 | mock 모드에서 API 응답 확인 |
| F-03 | Frontend | EtfChart.tsx: Lightweight Charts 라인 차트 1개 렌더 | P0 | 차트 프레임 + mock 데이터 표시 |
| F-04 | Frontend | PolicyInput.tsx: 선택지 버튼 3개 (최소 버전) | P0 | 버튼 클릭 → API 호출 |
| **M0** | **Both** | **/game/new → 선택지 1개 클릭 → /game/action → ETF 1종 차트에 새 점 추가** | **P0** | **one-click policy-to-chart 증명** |

### Phase 1: Hero Loop (W1 D3 ~ W2 末)

| ID | 담당 | 태스크 | 우선순위 | 산출물 |
|----|------|--------|----------|--------|
| B-07 | Backend | engine.py: 랜덤 분산(base ± variance x difficulty) 적용 | P1 | 같은 선택지도 매번 다른 결과 |
| B-08 | Backend | 이벤트 티어/체인/캐스케이드 선택 로직 검증 (v1에서 이전한 select_event) | P1 | 22개 이벤트 정상 순환 |
| B-09 | Backend | generate_hint() 검증 (정성적 힌트 문자열 생성) | P1 | "부채 소폭 증가 예상" 등 |
| F-05 | Frontend | EtfChart.tsx: ETF 5종 멀티 라인/캔들 차트 + 실시간 데이터 추가 애니메이션 | P1 | 선택 시 차트가 살아있게 움직임 |
| F-06 | Frontend | GaugePanel.tsx: 4개 게이지 바 + 색상 위험구간 (초록→노랑→빨강) + Framer Motion 전환 | P1 | 게이지 시각적 반응 |
| F-07 | Frontend | PolicyInput.tsx 확장: 이벤트 타이틀/설명 + 힌트 표시 + 채팅 스타일 | P1 | 이벤트 카드 완성 |
| F-08 | Frontend | 게임 루프 통합: 선택 → API 호출 → 게이지+차트 업데이트 → 다음 이벤트 | P1 | hero 순간 완성 |
| **M1** | **Both** | **정책 선택 → 게이지+ETF 5종 변동 → 차트 실시간 반응 + 애니메이션** | **P1** | **데모 가능한 hero 순간** |

### Phase 2: Full Loop — Narrowest Wedge (W3)

| ID | 담당 | 태스크 | 우선순위 | 산출물 |
|----|------|--------|----------|--------|
| B-10 | Backend | GET /game/{id}/turn/{turn}/narration: Groq 스트리밍 SSE + vn_dialogue JSON. 실패 시 템플릿 폴백 반환 | P2 | SSE 스트림으로 나레이션+대사 수신 (또는 폴백 텍스트) |
| B-11 | Backend | narration.py: v1 재활용 + 2-phase 분리 (action 응답에서 분리) | P2 | 나레이션이 차트 업데이트 후 도착 |
| F-09 | Frontend | VNDialogue.tsx: 포트레이트 이미지 1장 + 말풍선 + 타이핑 애니메이션 | P2 | 페르소나가 말풍선으로 반응 |
| F-10 | Frontend | 나레이션 표시: SSE 수신 → 뉴스 텍스트 실시간 렌더 (SSE 실패 시 폴백 텍스트 표시) | P2 | LLM 해설이 타이핑처럼 나타남 |
| F-11 | Frontend | 포트레이트 에셋: AI 생성 1장 (재무장관). 추가 캐릭터는 M3에서 | P2 | portraits/ 폴더에 이미지 1장 |
| **M2** | **Both** | **정책 → 차트 → 말풍선 1개 → 나레이션 (또는 폴백)** | **P2** | **narrowest wedge 완성** |

### CHECKPOINT (W2 末 / W3 D1 아침)

```
hero 순간이 데모 가능한가? (정책 선택 → ETF 5종 차트 실시간 반응)
├── YES → Phase 2 (Full Loop) 진행
└── NO  → Streamlit fallback 발동 (FB-01~05)
```

**Fallback 태스크 (NO 판정 시에만 실행):**

| ID | 담당 | 태스크 | 기한 | 산출물 |
|----|------|--------|------|--------|
| FB-01 | Frontend | v1-backup/app.py 기반 Streamlit 앱 복원 | W3 D2 | Streamlit 서버 기동 |
| FB-02 | Frontend | st.components.v1.html()로 Lightweight Charts HTML 임베드 | W3 D3 | 차트가 Streamlit 안에서 렌더 |
| FB-03 | Both | FastAPI 유지 or 직접 Python 호출 결정 | W3 D2 | 통신 방식 확정 |
| FB-04 | Frontend | VN을 st.chat_message()로 대체 | W3 D4 | 말풍선 대체 완료 |
| FB-05 | Both | W4 중반 데모 녹화 가능 상태 검증 | W3 末 | go/no-go |

참고: 실제 파일 위치는 `/home/dullear/econ-game/v1-backup/` (ref/ 아님)

### Phase 3: Polish (W4)

| ID | 담당 | 태스크 | 우선순위 | 산출물 |
|----|------|--------|----------|--------|
| B-12 | Backend | emh.py: Pearson 상관 분석 (숫자 3개 + 한 문장 요약, 히트맵 아님) | P3 | JSON으로 상관계수 3개 + summary_text 반환 |
| B-13 | Backend | GET /game/{id}/emh-summary 엔드포인트 | P3 | 임기말 성적표 API |
| B-14 | Backend | 게임 밸런싱: 이벤트 효과값 조정, 난이도 곡선 튜닝 | P3 | 20개월 완주 가능하되 긴장감 |
| F-12 | Frontend | EmhSummary.tsx: 임기말 성적표 (안정도 점수 + 상관계수 3개 + 한 문장, 히트맵 없음) | P3 | 임기 완주 시 1페이지 렌더 |
| F-13 | Frontend | 게임오버 화면: 파면 사유 + 최종 게이지 + "다시 시작" | P3 | 파면 시 화면 |
| F-14 | Frontend | 임기 시작 화면: 타이틀 + "임기 시작" 버튼 | P3 | 진입 화면 |
| F-15 | Frontend | 추가 포트레이트 2~4장 (M2에서 1장으로 시작한 것 확장) | P3 | portraits/ 이미지 추가 |
| X-01 | Both | Cross-review: 백엔드↔프론트 코드 리뷰 | P3 | 리뷰 코멘트 반영 |

### Phase 4: Ship (W5)

| ID | 담당 | 태스크 | 우선순위 | 산출물 |
|----|------|--------|----------|--------|
| X-02 | Backend | PPT 학술 섹션: EMH 분석 결과 슬라이드, Democracy 4 차별화 슬라이드 | P3 | PPT 2~3장 |
| X-03 | Frontend | PPT 디자인: 나머지 슬라이드, 아키텍처 포스터 | P3 | PPT + 포스터 |
| X-04 | Both | 데모 영상 녹화 (hero 순간 중심, 2~3분) | P3 | mp4 파일 |
| X-05 | Both | README.md 작성 (설치/실행 방법) | P3 | README |

---

## 의존성 그래프

```
B-01 → B-02 → B-03 ─┐
                     ├→ B-06 (fixtures) ─┐
B-04 ────────────────┘                    │
                                          ├→ M0 (one-click loop)
F-01 → F-02 (mock mode) → F-03 ─────────┤
                           F-04 ─────────┘
                                          │
                          B-07,B-08,B-09 ─┤
                     F-05,F-06,F-07 → F-08 ┤→ M1 (hero) → CHECKPOINT (W2末)
                                          │
                                          ├→ YES → B-10,B-11 ──┐
                                          │         F-09,F-10,F-11 ┤→ M2 (full loop)
                                          │                        │
                                          │         B-12~14 ───────┤→ M3 (polish)
                                          │         F-12~15 ───────┘
                                          │
                                          └→ NO  → FB-01~05 (Streamlit fallback)
```

## Week 3 Fallback Plan

CHECKPOINT에서 NO 판정 시 FB-01~05 태스크 발동 (위 Phase 2 직후 참조).
실제 파일: `/home/dullear/econ-game/v1-backup/`
목표: W4 중반까지 데모 가능 상태
