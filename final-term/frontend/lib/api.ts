const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// NEXT_PUBLIC_ 변수는 빌드 타임에 인라인됨. typeof window 체크 불필요.
const PREFER_MOCK = process.env.NEXT_PUBLIC_API_MODE === "mock";

export type Gauges = {
  debt: number;
  inflation: number;
  morale: number;
  tension: number;
};

export type EtfPrices = {
  semiconductor: number;
  energy: number;
  finance: number;
  defense: number;
  consumer: number;
};

export type Choice = { label: string; hint: string };

export type GameEvent = {
  id: string;
  title: string;
  desc: string;
  choices: Choice[];
};

export type GameState = {
  game_id: string;
  turn: number;
  gauges: Gauges;
  etf_prices: EtfPrices;
  event: GameEvent;
};

export type ActionResponse = {
  turn: number;
  gauges: Gauges;
  gauge_deltas: Gauges;
  etf_prices: EtfPrices;
  etf_changes: EtfPrices;
  next_event: GameEvent;
  game_over: string | null;
};

async function loadFixture<T>(name: string): Promise<T> {
  const res = await fetch(`/fixtures/${name}.json`);
  if (!res.ok) throw new Error(`fixture not found: ${name}`);
  return res.json();
}

// 이미 나온 이벤트 추적 (같은 탭 세션 내)
const _seenIds = new Set<string>();

async function pickNextEvent(currentId?: string): Promise<GameEvent> {
  const events = await loadFixture<GameEvent[]>("events");
  const pool = events.filter((e) => e.id !== currentId && !_seenIds.has(e.id));
  const candidates = pool.length > 0 ? pool : events.filter((e) => e.id !== currentId);
  const picked = candidates[Math.floor(Math.random() * candidates.length)];
  _seenIds.add(picked.id);
  return picked;
}

// 백엔드 없어도 동작: mock 우선, 실패 시 fixture fallback
export async function newGame(): Promise<GameState> {
  if (PREFER_MOCK) {
    _seenIds.clear();
    const base = await loadFixture<GameState>("new_game");
    const event = await pickNextEvent();
    return { ...base, event };
  }
  try {
    const res = await fetch(`${BASE_URL}/game/new`, { method: "POST" });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  } catch {
    _seenIds.clear();
    const base = await loadFixture<GameState>("new_game");
    const event = await pickNextEvent();
    return { ...base, event };
  }
}

export async function sendAction(
  gameId: string,
  choiceIndex: number,
  currentEventId?: string
): Promise<ActionResponse> {
  if (PREFER_MOCK) {
    const base = await loadFixture<ActionResponse>("action_response");
    const next_event = await pickNextEvent(currentEventId);
    return { ...base, next_event };
  }
  try {
    const res = await fetch(`${BASE_URL}/game/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, choice_index: choiceIndex }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  } catch {
    const base = await loadFixture<ActionResponse>("action_response");
    const next_event = await pickNextEvent(currentEventId);
    return { ...base, next_event };
  }
}
