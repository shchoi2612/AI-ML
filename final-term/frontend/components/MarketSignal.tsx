"use client";
import { useMemo } from "react";
import type { EtfPrices } from "@/lib/api";

// 시장 = 큰 차트가 아니라 한 줄 "신호". 숫자 없이 스파크라인 + 정성 태그만.
// 진짜 수치/상관계수는 임기말 EMH 성적표에서만 공개.
function meanPrice(p: EtfPrices): number {
  return (p.semiconductor + p.energy + p.finance + p.defense + p.consumer) / 5;
}

export function MarketSignal({ history }: { history: Array<{ turn: number; prices: EtfPrices }> }) {
  const { points, tag, color } = useMemo(() => {
    const series = history.map((h) => meanPrice(h.prices));
    const n = series.length;
    const lo = Math.min(...series, 100), hi = Math.max(...series, 100);
    const span = Math.max(1, hi - lo);
    const W = 100, H = 24;
    const points = series
      .map((v, i) => {
        const x = n <= 1 ? W : (i / (n - 1)) * W;
        const y = H - ((v - lo) / span) * H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const change = n >= 2 ? series[n - 1] - series[n - 2] : 0;
    let tag = "관망", color = "var(--sr-mut)";
    if (change > 1.5) { tag = "환호"; color = "var(--stable)"; }
    else if (change < -1.5) { tag = "패닉"; color = "var(--crisis)"; }
    else if (Math.abs(change) > 0.6) { tag = "동요"; color = "var(--gold)"; }
    return { points, tag, color };
  }, [history]);

  return (
    <div className="sr-panel px-3 py-2 flex items-center gap-3">
      <div className="sr-label shrink-0">시장 신호</div>
      <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="flex-1" style={{ height: 24 }}>
        <polyline points={points} fill="none" stroke={color} strokeWidth={1.6}
                  vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <div className="shrink-0 sr-display" style={{ color, fontSize: "0.78rem", fontWeight: 800, letterSpacing: "0.04em" }}>
        {tag}
      </div>
    </div>
  );
}
