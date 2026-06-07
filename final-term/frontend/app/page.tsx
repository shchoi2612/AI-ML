"use client";
import { useEffect, useState, useCallback } from "react";
import { StatusBars } from "@/components/StatusBars";
import { MarketSignal } from "@/components/MarketSignal";
import { NationalFace } from "@/components/NationalFace";
import { BreakingNews } from "@/components/BreakingNews";
import { CardPool } from "@/components/CardPool";
import { Scorecard } from "@/components/Scorecard";
import { NewsTicker } from "@/components/NewsTicker";
import {
  newGame,
  sendAction,
  type GameState,
  type EtfPrices,
} from "@/lib/api";
import { moodFromGauges } from "@/lib/expression";

type EtfPoint = { turn: number; prices: EtfPrices };

const MAX_TURNS = 20;
const IS_MOCK = process.env.NEXT_PUBLIC_API_MODE === "mock";

export default function GamePage() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [etfHistory, setEtfHistory] = useState<EtfPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gameOver, setGameOver] = useState<string | null>(null);
  const [narrationTurn, setNarrationTurn] = useState<number | null>(null);
  const [fx, setFx] = useState(0);          // 결정 반응 트리거
  const [crisisFx, setCrisisFx] = useState(false);
  const [shaking, setShaking] = useState(false);

  const startGame = useCallback(async () => {
    setLoading(true);
    setError(null);
    setGameOver(null);
    setNarrationTurn(null);
    setEtfHistory([]);
    try {
      const state = await newGame();
      setGameState(state);
      setEtfHistory([{ turn: state.turn, prices: state.etf_prices }]);
    } catch {
      setError("연결 실패 — 백엔드(:8000) 확인 또는 NEXT_PUBLIC_API_MODE=mock");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { startGame(); }, [startGame]);

  const handleCommit = useCallback(
    async (cardIds: string[]) => {
      if (!gameState || loading) return;
      setLoading(true);
      try {
        const res = await sendAction(gameState.game_id, cardIds);
        const next: GameState = {
          ...gameState,
          turn: res.turn,
          gauges: res.gauges,
          etf_prices: res.etf_prices,
          situation: res.next_situation ?? gameState.situation,
          fiscal_capacity: res.fiscal_capacity ?? gameState.fiscal_capacity,
          sector_resources: res.sector_resources ?? gameState.sector_resources,
          card_pool: res.card_pool ?? gameState.card_pool,
        };
        setGameState(next);
        setEtfHistory((prev) => [...prev, { turn: res.turn, prices: res.etf_prices }]);
        setNarrationTurn(res.turn - 1);   // 방금 결정한 턴의 속보

        // 위험 구간 진입 시 크림슨 플래시
        const g = res.gauges;
        const crisis = g.debt >= 80 || g.inflation >= 80 || g.morale <= 20 || g.tension >= 80;
        setCrisisFx(crisis);
        setFx((k) => k + 1);
        setShaking(true);
        setTimeout(() => setShaking(false), 340);

        if (res.game_over) setGameOver(res.game_over);
      } catch {
        setError("전송 오류 — 정책 집행 실패");
      } finally {
        setLoading(false);
      }
    },
    [gameState, loading]
  );

  const month = gameState ? `2024.${String(((gameState.turn - 1) % 12) + 1).padStart(2, "0")}` : "----.--";
  const progress = gameState ? Math.max(0, Math.min(1, gameState.turn / MAX_TURNS)) : 0;

  return (
    <div className={`h-screen flex flex-col overflow-hidden ${shaking ? "sr-shake" : ""}`} style={{ color: "var(--sr-ink)" }}>

      {/* 결정 반응 플래시 */}
      {fx > 0 && <div key={fx} className={`sr-flash ${crisisFx ? "sr-flash-crisis" : ""}`} />}

      {/* ── 헤더 (상황실 콘솔) ── */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-2" style={{ borderBottom: "1px solid var(--sr-border)", background: "var(--sr-bg2)" }}>
        <span className="sr-display" style={{ fontWeight: 800, fontSize: "0.95rem", letterSpacing: "0.04em", color: "var(--sr-ink)" }}>
          ECON<span style={{ color: "var(--crisis)" }}>SIM</span>
        </span>
        <span className="sr-label" style={{ color: "var(--sr-dim)" }}>경제장관 상황실</span>
        <span className="ml-2 sr-label" style={{ color: "var(--gold)" }}>{month}</span>
        <span className="sr-label" style={{ color: "var(--sr-mut)" }}>임기 {gameState?.turn ?? 0}/{MAX_TURNS}</span>
        {/* 진행 바 */}
        <div className="flex-1 h-[5px] rounded-sm overflow-hidden" style={{ background: "#0a121b", maxWidth: 260 }}>
          <div className="h-full" style={{ width: `${progress * 100}%`, background: "var(--gold)", transition: "width 0.4s" }} />
        </div>
        <span className="ml-auto sr-label" style={{ color: IS_MOCK ? "var(--gold)" : "var(--stable)" }}>
          ● {IS_MOCK ? "MOCK" : "LIVE"}
        </span>
      </div>

      {/* 에러 배너 */}
      {error && (
        <div className="shrink-0 px-4 py-1 flex justify-between" style={{ background: "var(--crisis-deep)", color: "#fff", fontSize: "0.66rem" }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)} className="underline ml-4">[닫기]</button>
        </div>
      )}

      {/* 부팅 */}
      {!gameState && (
        <div className="flex-1 flex items-center justify-center">
          <span className="sr-label blink" style={{ fontSize: "0.8rem" }}>상황실 가동 중…</span>
        </div>
      )}

      {/* ── 메인 ── */}
      {gameState && (
        <main className="flex-1 flex flex-col gap-2 overflow-hidden min-h-0 px-3 py-2">

          {/* 속보 자막 */}
          <BreakingNews gameId={gameState.game_id} turn={narrationTurn} />

          {/* 좌: 국가 지표 / 우: 국가 표정(작게) + 시장 신호 상태 */}
          <div className="shrink-0 flex gap-2 items-stretch">
            <div className="flex-1 min-w-0"><StatusBars gauges={gameState.gauges} /></div>
            <div className="w-[34%] min-w-0 flex flex-col gap-2">
              <div className="sr-panel flex-1 flex items-center justify-center py-2 min-h-0">
                <NationalFace mood={moodFromGauges(gameState.gauges)} />
              </div>
              <MarketSignal history={etfHistory} />
            </div>
          </div>

          {/* 카드 주인공 */}
          <div className="flex-1 min-h-0">
            <CardPool
              situation={gameState.situation}
              cards={gameState.card_pool}
              capacity={gameState.fiscal_capacity}
              resources={gameState.sector_resources}
              onCommit={handleCommit}
              disabled={loading || !!gameOver}
            />
          </div>
        </main>
      )}

      {/* 임기말 성적표 */}
      {gameOver && gameState && (
        <Scorecard gameId={gameState.game_id} gameOverMsg={gameOver} history={etfHistory} onRestart={startGame} />
      )}

      {/* 하단 앰비언트 속보 스크롤 */}
      {gameState && <NewsTicker gauges={gameState.gauges} turn={gameState.turn} />}
    </div>
  );
}
