"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import type { EtfPrices } from "@/lib/api";

const EtfChart = dynamic(() => import("@/components/EtfChart").then((m) => m.EtfChart), { ssr: false });

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const IS_MOCK = process.env.NEXT_PUBLIC_API_MODE === "mock";

type Emh = {
  total_turns: number;
  stability_score: number;
  predictability_score: number;
  top_correlation: { sector: string; value: number };
  avg_etf_volatility: number;
  summary_text: string;
};

interface ScorecardProps {
  gameId: string;
  gameOverMsg: string;
  history: Array<{ turn: number; prices: EtfPrices }>;
  onRestart: () => void;
}

// 임기말 성적표: 플레이 중 숨겼던 raw 숫자 + 차트 + 상관계수를 여기서 공개(지연된 보상).
export function Scorecard({ gameId, gameOverMsg, history, onRestart }: ScorecardProps) {
  const [emh, setEmh] = useState<Emh | null>(null);

  useEffect(() => {
    if (IS_MOCK) return;
    let on = true;
    fetch(`${BASE_URL}/game/${gameId}/emh-summary`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (on && d) setEmh(d); })
      .catch(() => {});
    return () => { on = false; };
  }, [gameId]);

  const isWin = gameOverMsg.startsWith("임기 완주");

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(6,9,13,0.94)" }}>
      <div className="sr-panel w-full" style={{ maxWidth: 720, padding: 0 }}>
        {/* 헤더 */}
        <div className="px-5 py-3" style={{ borderBottom: "1px solid var(--sr-border)" }}>
          <div className="sr-label" style={{ color: isWin ? "var(--gold)" : "var(--crisis)" }}>
            {isWin ? "임기 종료 · 성적표" : "파면 · 성적표"}
          </div>
          <div className="sr-display" style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--sr-ink)", marginTop: 2 }}>
            {gameOverMsg}
          </div>
        </div>

        <div className="px-5 py-4 flex flex-col gap-4">
          {/* 숨겨뒀던 진짜 차트 */}
          <div>
            <div className="sr-label" style={{ marginBottom: 4 }}>섹터 ETF 실적 (전 임기)</div>
            <EtfChart history={history} />
          </div>

          {/* EMH 수치 — 여기서만 공개 */}
          {emh ? (
            <>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "안정도", val: `${emh.stability_score} / 400`, color: "var(--stable)" },
                  { label: "시장 예측가능성", val: `${(emh.predictability_score * 100).toFixed(0)}%`, color: "var(--gold)" },
                  { label: `최민감 · ${emh.top_correlation.sector}`, val: emh.top_correlation.value.toFixed(2), color: "var(--blue)" },
                ].map((m) => (
                  <div key={m.label} className="sr-panel px-3 py-2 text-center">
                    <div className="sr-label" style={{ fontSize: "0.5rem" }}>{m.label}</div>
                    <div className="sr-display" style={{ fontSize: "1.2rem", fontWeight: 800, color: m.color, marginTop: 2 }}>
                      {m.val}
                    </div>
                  </div>
                ))}
              </div>
              <div className="sr-panel px-3 py-2" style={{ fontSize: "0.72rem", lineHeight: 1.7, color: "var(--sr-mut)" }}>
                {emh.summary_text}
              </div>
            </>
          ) : (
            <div className="sr-label">{IS_MOCK ? "EMH 분석은 LIVE 모드에서만 제공됩니다." : "EMH 분석 불러오는 중…"}</div>
          )}

          <button onClick={onRestart}
            className="sr-display self-center px-6 py-2 rounded"
            style={{ background: "var(--gold)", color: "#1a1206", fontWeight: 800, letterSpacing: "0.05em", fontSize: "0.8rem" }}>
            다시 임기 시작
          </button>
        </div>
      </div>
    </div>
  );
}
