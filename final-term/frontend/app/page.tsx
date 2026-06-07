"use client";
import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { GaugePanel } from "@/components/GaugePanel";
import { PolicyInput } from "@/components/PolicyInput";
import { CityScene } from "@/components/CityScene";
import { DebugPanel } from "@/components/DebugPanel";
import { NewsTicker } from "@/components/NewsTicker";
import {
  newGame,
  sendAction,
  type GameState,
  type EtfPrices,
  type Gauges,
} from "@/lib/api";

const EtfChart = dynamic(
  () => import("@/components/EtfChart").then((m) => m.EtfChart),
  { ssr: false }
);

type EtfPoint = { turn: number; prices: EtfPrices };
type Screen = "chart" | "policy" | "debug";

const MAX_TURNS = 20;
const IS_MOCK = process.env.NEXT_PUBLIC_API_MODE === "mock";

function HLine({ char = "═", left = "╔", right = "╗", label = "" }: {
  char?: string; left?: string; right?: string; label?: string;
}) {
  return (
    <div className="flex items-center overflow-hidden leading-none text-dos-green select-none" style={{ fontSize: "0.7rem" }}>
      <span className="shrink-0">{left}{char}{char}</span>
      {label && <span className="shrink-0 text-dos-amber font-bold px-1">{label}</span>}
      <span className="flex-1 min-w-0 overflow-hidden text-dos-green-dim" style={{ letterSpacing: 0 }}>
        {char.repeat(300)}
      </span>
      <span className="shrink-0">{right}</span>
    </div>
  );
}

function HeaderRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center overflow-hidden leading-none text-dos-green select-none px-1" style={{ fontSize: "0.7rem" }}>
      <span className="shrink-0 mr-2">║</span>
      <span className="flex-1 min-w-0">{children}</span>
      <span className="shrink-0 ml-2">║</span>
    </div>
  );
}

/* 정책 화면 하단 한 줄 게이지 */
function GaugeStrip({ gauges }: { gauges: Gauges }) {
  const items: { label: string; key: keyof Gauges; dangerHigh: boolean }[] = [
    { label: "DEBT", key: "debt",      dangerHigh: true  },
    { label: "INFL", key: "inflation", dangerHigh: true  },
    { label: "MORL", key: "morale",    dangerHigh: false },
    { label: "TENS", key: "tension",   dangerHigh: true  },
  ];

  return (
    <div
      className="ascii-panel shrink-0 font-mono"
      style={{ fontSize: "0.6rem", whiteSpace: "pre" }}
    >
      <span className="ascii-panel-title">STATUS</span>
      {items.map(({ label, key, dangerHigh }) => {
        const v = gauges[key];
        const d = dangerHigh ? v : 100 - v;
        const color = d >= 80 ? "#ff3333" : d >= 60 ? "#ffaa00" : "#39ff14";
        const f = Math.round(v / 10);
        const bar = "#".repeat(f) + ".".repeat(10 - f);
        return (
          <span key={key}>
            <span style={{ color: "#1a7a0a" }}>{label} </span>
            <span style={{ color: "#1a7a0a" }}>{"["}</span>
            <span style={{ color }}>{bar}</span>
            <span style={{ color: "#1a7a0a" }}>{"]"}</span>
            <span style={{ color }}>{" " + String(v).padStart(3) + "  "}</span>
          </span>
        );
      })}
    </div>
  );
}

export default function GamePage() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [etfHistory, setEtfHistory] = useState<EtfPoint[]>([]);
  const [deltas, setDeltas] = useState<Partial<Gauges>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gameOver, setGameOver] = useState<string | null>(null);
  const [screen, setScreen] = useState<Screen>("chart");

  const startGame = useCallback(async () => {
    setLoading(true);
    setError(null);
    setGameOver(null);
    setDeltas({});
    setEtfHistory([]);
    try {
      const state = await newGame();
      setGameState(state);
      setEtfHistory([{ turn: state.turn, prices: state.etf_prices }]);
      setScreen("policy"); // 게임 시작 → 정책 선택 화면
    } catch {
      setError("CONNECTION FAILED — check NEXT_PUBLIC_API_MODE=mock");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { startGame(); }, [startGame]);

  const handleChoice = useCallback(
    async (choiceIndex: number) => {
      if (!gameState || loading) return;
      setLoading(true);
      setDeltas({});
      try {
        const res = await sendAction(gameState.game_id, choiceIndex, gameState.event.id);
        const nextTurn = gameState.turn + 1;
        setDeltas(res.gauge_deltas);
        setGameState({
          ...gameState,
          turn: nextTurn,
          gauges: res.gauges,
          etf_prices: res.etf_prices,
          event: res.next_event,
        });
        setEtfHistory((prev) => [
          ...prev,
          { turn: nextTurn, prices: res.etf_prices },
        ]);
        if (res.game_over) {
          setGameOver(res.game_over);
        } else if (nextTurn > MAX_TURNS) {
          setGameOver("임기 완주! 경제를 성공적으로 이끌었습니다.");
        }
        setScreen("chart"); // 선택 후 → 차트 화면
      } catch {
        setError("TRANSMISSION ERROR — action failed");
      } finally {
        setLoading(false);
      }
    },
    [gameState, loading]
  );

  const handleDebugApply = useCallback((newGauges: Gauges) => {
    if (!gameState) return;
    setGameState({ ...gameState, gauges: newGauges });
    setDeltas({});
    setScreen("chart");
  }, [gameState]);

  const month = gameState ? `2024.${String(gameState.turn).padStart(2, "0")}` : "----.-";
  const turnStr = gameState ? `TURN ${String(gameState.turn).padStart(2, "0")}/${MAX_TURNS}` : "BOOT...";
  // 완주 시 turn이 MAX_TURNS를 넘어(turn 21) progress가 20을 초과하면
  // 아래 진행바의 "░".repeat(20 - progress)가 음수 인자로 RangeError를 던진다.
  // progress를 [0, 20]으로 clamp해 두 repeat 모두 음수/오버플로를 방지한다.
  const progress = gameState
    ? Math.max(0, Math.min(20, Math.round((gameState.turn / MAX_TURNS) * 20)))
    : 0;

  return (
    <div className="h-screen flex flex-col bg-black text-dos-green font-mono overflow-hidden" style={{ fontSize: "0.7rem" }}>

      {/* 헤더 */}
      <div className="shrink-0 select-none">
        <HLine left="╔" right="╗" label="ECONSIM v1.0 // 국가경제 시뮬레이터" />
        <HeaderRow>
          <span className="text-dos-green-dim">경제장관 집무실</span>
          <span className="text-dos-green-dim mx-3">·</span>
          <span className="text-dos-amber">{month}</span>
          <span className="text-dos-green-dim mx-3">·</span>
          <span>{turnStr}</span>
          <span className="text-dos-green-dim mx-3">·</span>
          <span className="text-dos-green-dim">[{"▓".repeat(progress)}{"░".repeat(20 - progress)}]</span>
          <span className="flex-1" />
          <span className={IS_MOCK ? "text-dos-amber" : "text-dos-green"}>
            {IS_MOCK ? "◈ MOCK" : "◈ LIVE"}
          </span>
        </HeaderRow>
        <HLine left="╠" right="╣" char="═" />
      </div>

      {/* 에러 배너 */}
      {error && (
        <div className="shrink-0 border-b border-dos-red px-3 py-1 text-dos-red font-mono flex justify-between"
          style={{ fontSize: "0.65rem", background: "#1a0000" }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)} className="underline ml-4">[DISMISS]</button>
        </div>
      )}

      {/* 게임 오버 오버레이 */}
      {gameOver && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/90">
          <div className="ascii-panel max-w-sm w-full text-center px-8 py-6">
            <div className="text-dos-red font-bold mb-3 tracking-widest" style={{ fontSize: "1rem" }}>
              ■■ GAME OVER ■■
            </div>
            <div className="text-dos-amber mb-5 leading-relaxed" style={{ fontSize: "0.65rem" }}>
              {gameOver}
            </div>
            {gameState && (
              <div className="text-dos-green-dim mb-5" style={{ fontSize: "0.6rem" }}>
                SURVIVED {gameState.turn} / {MAX_TURNS} TURNS
              </div>
            )}
            <button onClick={startGame}
              className="font-mono font-bold px-4 py-1 border border-dos-green text-dos-green hover:bg-dos-green hover:text-black transition-colors"
              style={{ fontSize: "0.65rem" }}>
              [ RESTART ]
            </button>
          </div>
        </div>
      )}

      {/* 부팅 */}
      {!gameState && loading && (
        <div className="flex-1 flex items-center justify-center">
          <span className="text-dos-green-dim" style={{ fontSize: "0.65rem" }}>
            INITIALIZING SYSTEM<span className="blink">_</span>
          </span>
        </div>
      )}

      {/* 메인 */}
      {gameState && (
        <main className="flex-1 flex flex-col gap-1 overflow-hidden min-h-0 px-1 pb-1">

          {/* ── 차트 화면 ── */}
          {screen === "chart" && (
            <>
              <div className="flex gap-1 flex-1 min-h-0 mt-1">
                <div className="flex-1 min-w-0 flex flex-col gap-1">
                  <EtfChart history={etfHistory} />
                  {/* 도시 씬 */}
                  <div className="flex-1 min-h-0 overflow-hidden">
                    <CityScene gauges={gameState.gauges} />
                  </div>
                </div>
                <div className="w-52 shrink-0 overflow-hidden min-h-0">
                  <GaugePanel gauges={gameState.gauges} deltas={deltas} />
                </div>
              </div>

              {/* 뉴스 티커 */}
              <NewsTicker gauges={gameState.gauges} turn={gameState.turn} />

              {/* 다음 정책 결정 / DEBUG 버튼 */}
              {!gameOver && (
                <div className="shrink-0 flex justify-center items-center gap-3 py-1">
                  <button
                    onClick={() => setScreen("policy")}
                    className="font-mono font-bold border border-dos-amber text-dos-amber px-8 py-1 hover:bg-dos-amber hover:text-black transition-colors"
                    style={{ fontSize: "0.7rem", letterSpacing: "0.1em" }}
                  >
                    ▶▶ 다음 정책 결정
                  </button>
                  <button
                    onClick={() => setScreen("debug")}
                    className="font-mono border border-dos-green-dim text-dos-green-dim px-4 py-1 hover:border-dos-green hover:text-dos-green transition-colors"
                    style={{ fontSize: "0.6rem", letterSpacing: "0.05em" }}
                  >
                    [ DEBUG ]
                  </button>
                </div>
              )}
            </>
          )}

          {/* ── 정책 선택 화면 ── */}
          {screen === "policy" && (
            <>
              <div className="flex-1 min-h-0 mt-1">
                <PolicyInput
                  event={gameState.event}
                  onChoice={handleChoice}
                  disabled={loading}
                />
              </div>
              <GaugeStrip gauges={gameState.gauges} />
            </>
          )}

          {/* ── 디버그 화면 ── */}
          {screen === "debug" && (
            <div className="flex-1 min-h-0 mt-1">
              <DebugPanel
                gauges={gameState.gauges}
                onApply={handleDebugApply}
                onCancel={() => setScreen("chart")}
              />
            </div>
          )}

        </main>
      )}

      <HLine left="╚" right="╝" char="═" />
    </div>
  );
}
