"use client";
import { useEffect, useRef, memo } from "react";
import {
  createChart,
  ColorType,
  IChartApi,
  ISeriesApi,
} from "lightweight-charts";
import type { EtfPrices } from "@/lib/api";

const ETF_META: { key: keyof EtfPrices; label: string; color: string }[] = [
  { key: "semiconductor", label: "반도체", color: "#33ff66" },
  { key: "energy", label: "에너지", color: "#ffaa00" },
  { key: "finance", label: "금융", color: "#6699ff" },
  { key: "defense", label: "방산", color: "#ff6666" },
  { key: "consumer", label: "소비재", color: "#cc66ff" },
];

function turnToDate(turn: number): string {
  const year = 2024 + Math.floor((turn - 1) / 12);
  const month = ((turn - 1) % 12) + 1;
  return `${year}-${String(month).padStart(2, "0")}-01`;
}

interface EtfChartProps {
  history: Array<{ turn: number; prices: EtfPrices }>;
}

export const EtfChart = memo(function EtfChart({ history }: EtfChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const seriesRef = useRef<Record<string, ISeriesApi<any>>>({});

  useEffect(() => {
    if (!containerRef.current) return;

    chartRef.current = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#000000" },
        textColor: "#1a7a0a",
        fontFamily: "var(--font-geist-mono), monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "#0a1f05" },
        horzLines: { color: "#0a1f05" },
      },
      timeScale: {
        borderColor: "#1a7a0a",
        timeVisible: false,
        rightOffset: 1,
        barSpacing: 40,
        fixLeftEdge: true,
        fixRightEdge: true,
        lockVisibleTimeRangeOnResize: true,
      },
      rightPriceScale: {
        borderColor: "#1a7a0a",
      },
      crosshair: {
        vertLine: { color: "#39ff14", style: 1 },
        horzLine: { color: "#39ff14", style: 1 },
      },
      width: containerRef.current.clientWidth,
      height: 180,
    });

    for (const { key, color } of ETF_META) {
      seriesRef.current[key] = chartRef.current.addLineSeries({
        color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });
    }

    const ro = new ResizeObserver(() => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chartRef.current?.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chartRef.current || history.length === 0) return;
    for (const { key } of ETF_META) {
      const data = history.map((h) => ({
        time: turnToDate(h.turn),
        value: h.prices[key],
      }));
      seriesRef.current[key]?.setData(data);
    }
    // 데이터가 항상 화면 전체를 가득 채우도록 자동 스케일
    chartRef.current.timeScale().fitContent();
  }, [history]);

  const latest = history[history.length - 1]?.prices;

  return (
    <div className="ascii-panel">
      <span className="ascii-panel-title">ETF MARKET FEED</span>
      <div ref={containerRef} style={{ width: "100%" }} />
      {/* 범례 + 현재값을 차트 밖(아래)에 표시 */}
      <div className="flex gap-4 flex-wrap mt-2" style={{ fontSize: "0.6rem" }}>
        {ETF_META.map(({ key, label, color }) => (
          <span key={key} className="font-mono" style={{ color }}>
            ─ {label}
            {latest && (
              <span style={{ color: "#1a7a0a" }}>
                {" "}({latest[key].toFixed(1)})
              </span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
});
