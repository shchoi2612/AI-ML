"use client";
import { useMemo } from "react";
import type { EtfPrices } from "@/lib/api";

// 시장 = 차트 없이 "상태(태그)"만. 숫자/그래프 없음.
// 진짜 수치·차트·상관계수는 임기말 EMH 성적표에서만 공개.
function meanPrice(p: EtfPrices): number {
  return (p.semiconductor + p.energy + p.finance + p.defense + p.consumer) / 5;
}

export function MarketSignal({ history }: { history: Array<{ turn: number; prices: EtfPrices }> }) {
  const { tag, color, desc } = useMemo(() => {
    const series = history.map((h) => meanPrice(h.prices));
    const n = series.length;
    const change = n >= 2 ? series[n - 1] - series[n - 2] : 0;
    if (change > 1.5) return { tag: "환호", color: "var(--stable)", desc: "시장이 정책을 반긴다" };
    if (change < -1.5) return { tag: "패닉", color: "var(--crisis)", desc: "투자자들이 등을 돌린다" };
    if (Math.abs(change) > 0.6) return { tag: "동요", color: "var(--gold)", desc: "시장이 술렁인다" };
    return { tag: "관망", color: "var(--sr-mut)", desc: "시장은 숨을 고른다" };
  }, [history]);

  return (
    <div className="sr-panel px-3 py-2 flex items-center justify-between gap-3">
      <div className="flex flex-col">
        <span className="sr-label">시장 신호</span>
        <span style={{ fontSize: "0.58rem", color: "var(--sr-mut)" }}>{desc}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span style={{ width: 9, height: 9, borderRadius: 9, background: color, boxShadow: `0 0 8px ${color}` }} />
        <span className="sr-display" style={{ color, fontSize: "1.05rem", fontWeight: 800, letterSpacing: "0.06em" }}>
          {tag}
        </span>
      </div>
    </div>
  );
}
