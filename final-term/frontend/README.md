# EconSim — 국가경제 시뮬레이터(가제)

DOS 터미널 스타일의 경제 정책 시뮬레이션 게임.  
경제장관이 되어 20턴 동안 부채·인플레이션·민심·국제긴장 4개 지표를 관리한다.

---

## 현재 진행 상황

| 영역 | 상태 | 비고 |
|------|------|------|
| 프론트엔드 UI | ✅ 완성 | Next.js 14, DOS/CRT 스타일 |
| 게이지 시스템 | ✅ 완성 | 4개 독립 지표 + 델타 표시 |
| 이벤트·정책 카드 | ✅ 완성 | 8개 이벤트, 중복 방지 |
| ETF 차트 | ✅ 완성 | 5개 섹터, 턴별 히스토리 |
| 도시 씬 + 시민 여론 | ✅ 완성 | 게이지별 독립 말풍선 |
| 뉴스 티커 | ✅ 완성 | Breaking News 스크롤 |
| Mock API | ✅ 완성 | 백엔드 없이 전 기능 동작 |
| 백엔드 연결 | 🔲 미착수 | API 스펙은 정의됨 (아래 참고) |
| 엔딩 분기 | 🔲 미착수 | 게이지 조건 + 이벤트 트리거 조합 예정 |
| 콘텐츠 확장 | 🔲 미착수 | 이벤트 8개 → 30개+ 목표 |
| 게임 밸런스 | 🔲 미착수 | 현재 고정 delta값, 수치 조정 필요 |

---

## 실행 방법

```bash
# 의존성 설치
npm install

# Mock 모드로 개발 서버 실행 (백엔드 불필요)
NEXT_PUBLIC_API_MODE=mock npm run dev

# 백엔드 연결 모드
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

`http://localhost:3000` 접속

---

## 프로젝트 구조

```
frontend/
├── app/
│   ├── page.tsx          # 메인 게임 루프 (화면 전환: chart ↔ policy ↔ debug)
│   ├── globals.css       # CRT 효과, 키프레임 애니메이션
│   └── layout.tsx
├── components/
│   ├── GaugePanel.tsx    # 우측 4개 게이지 바
│   ├── PolicyInput.tsx   # 이벤트 제목 + 3개 정책 카드
│   ├── CityScene.tsx     # ASCII 도시 씬 + 시민 말풍선
│   ├── NewsTicker.tsx    # 하단 Breaking News 스크롤
│   ├── EtfChart.tsx      # ETF 가격 히스토리 차트
│   └── DebugPanel.tsx    # 게이지 직접 조작 (개발용)
├── lib/
│   └── api.ts            # API 타입 정의 + Mock/실서버 전환 로직
└── public/
    └── fixtures/
        ├── events.json         # 이벤트 목록
        ├── new_game.json       # 게임 초기 상태
        └── action_response.json # 정책 선택 응답 템플릿
```

---

## API 스펙 (백엔드 연동 기준)

### `POST /game/new` → `GameState`

```ts
{
  game_id: string
  turn: number          // 1부터 시작
  gauges: Gauges        // { debt, inflation, morale, tension } 각 0~100
  etf_prices: EtfPrices // { semiconductor, energy, finance, defense, consumer }
  event: GameEvent      // { id, title, desc, choices: [{ label, hint }] }
}
```

### `POST /game/action` → `ActionResponse`

```ts
// Request
{ game_id: string, choice_index: number }  // choice_index: 0~2

// Response
{
  turn: number
  gauges: Gauges
  gauge_deltas: Gauges   // 이번 턴 변화량 (UI 델타 표시용)
  etf_prices: EtfPrices
  etf_changes: EtfPrices
  next_event: GameEvent
  game_over: string | null  // null이면 계속 진행
}
```

> **TODO:** `game_over`를 `string | null`에서 `{ type: string, message: string } | null`로 확장해 엔딩 분기 연출 차별화 예정

---

## 남은 과제

### 🔴 높은 우선순위

- **백엔드 연결** — `lib/api.ts`의 fetch 로직은 준비됨. 서버 구현 후 `NEXT_PUBLIC_API_URL` 설정만 하면 전환 가능
- **엔딩 분기 구현** — `game_over` 타입 확장 후 게이지 임계값(예: 부채 90+) 및 특정 이벤트 트리거 조합으로 멀티 엔딩 설계
- **게임 밸런스 수치화** — 현재 Mock `action_response.json`의 delta값이 고정. 이벤트×선택지별 delta 매트릭스를 JSON 또는 스프레드시트로 정의 필요

### 🟡 중간 우선순위

- **이벤트 콘텐츠 확장** — 현재 8개 → 30개+ 목표. `public/fixtures/events.json` 형식에 맞춰 추가하면 됨
- **엔딩 화면 연출** — 현재 단순 텍스트 오버레이. 엔딩 타입별 ASCII 아트 + 결과 리포트 (재임 기간 통계)
- **턴 연계 이벤트** — 이전 정책 선택이 이후 이벤트 출현에 영향을 주는 인과관계 시스템

### 🟢 낮은 우선순위

- **사운드 이펙트** — 키 입력, 이벤트 발생 시 레트로 효과음
- **세이브/로드** — localStorage 기반 게임 상태 저장
- **모바일 대응** — 현재 데스크탑 전용 레이아웃

---

## 기여 방법

이벤트 추가는 `public/fixtures/events.json`에 아래 형식으로 작성:

```json
{
  "id": "unique_snake_case_id",
  "title": "이벤트 제목 (20자 내외)",
  "desc": "상황 설명 (2~3문장)",
  "choices": [
    { "label": "선택지 라벨", "hint": "게이지 영향 힌트 (예: 부채↑, 민심↓)" },
    { "label": "선택지 라벨", "hint": "..." },
    { "label": "선택지 라벨", "hint": "..." }
  ]
}
```
