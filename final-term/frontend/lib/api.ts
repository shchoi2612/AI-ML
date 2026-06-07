// 계약 v2 — 코스트 기반 카드 시스템
const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// NEXT_PUBLIC_ 변수는 빌드 타임에 인라인됨.
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

// 섹터 자원 게이팅 대상 (ETF의 부분집합)
export type SectorResources = {
  energy: number;
  defense: number;
  semiconductor: number;
};

export type Situation = {
  id: string;
  title: string;
  desc: string;
};

export type Card = {
  id: string;
  title: string;
  sector: keyof SectorResources | null;
  fiscal_cost: number;
  sector_cost: number;
  hint: string;
  affordable: boolean;
  tags: string[];
};

export type GameState = {
  game_id: string;
  turn: number;
  gauges: Gauges;
  etf_prices: EtfPrices;
  situation: Situation;
  fiscal_capacity: number;
  sector_resources: SectorResources;
  card_pool: Card[];
};

export type ActionResponse = {
  turn: number;
  gauges: Gauges;
  gauge_deltas: Gauges;
  etf_prices: EtfPrices;
  etf_changes: EtfPrices;
  next_situation: Situation | null;
  game_over: string | null;
  // 진행 중일 때만 포함 (다음 턴 예산 + 카드풀)
  fiscal_capacity?: number;
  sector_resources?: SectorResources;
  card_pool?: Card[];
};

async function loadFixture<T>(name: string): Promise<T> {
  const res = await fetch(`/fixtures/${name}.json`);
  if (!res.ok) throw new Error(`fixture not found: ${name}`);
  return res.json();
}

// 백엔드 없어도 동작: mock 우선, 실패 시 fixture fallback
export async function newGame(): Promise<GameState> {
  if (PREFER_MOCK) return loadFixture<GameState>("new_game");
  try {
    const res = await fetch(`${BASE_URL}/game/new`, { method: "POST" });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  } catch {
    return loadFixture<GameState>("new_game");
  }
}

export async function sendAction(
  gameId: string,
  cardIds: string[],
): Promise<ActionResponse> {
  if (PREFER_MOCK) return loadFixture<ActionResponse>("action_response");
  try {
    const res = await fetch(`${BASE_URL}/game/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: gameId, card_ids: cardIds }),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  } catch {
    return loadFixture<ActionResponse>("action_response");
  }
}
