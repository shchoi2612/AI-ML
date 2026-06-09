"use client";
import { useState } from "react";
import type { Gauges } from "@/lib/api";

const PARAMS: { key: keyof Gauges; label: string; dangerHigh: boolean }[] = [
  { key: "debt",      label: "DEBT",      dangerHigh: true  },
  { key: "inflation", label: "INFLATION", dangerHigh: true  },
  { key: "morale",    label: "MORALE",    dangerHigh: false },
  { key: "tension",   label: "TENSION",   dangerHigh: true  },
];

function gaugeColor(v: number, dangerHigh: boolean): string {
  const d = dangerHigh ? v : 100 - v;
  if (d >= 80) return "#ff3333";
  if (d >= 60) return "#ffaa00";
  return "#39ff14";
}

interface DebugPanelProps {
  gauges: Gauges;
  onApply: (gauges: Gauges) => void;
  onCancel: () => void;
}

export function DebugPanel({ gauges, onApply, onCancel }: DebugPanelProps) {
  const [vals, setVals] = useState<Gauges>({ ...gauges });

  const set = (key: keyof Gauges, raw: number) => {
    const v = Math.max(0, Math.min(100, isNaN(raw) ? 0 : raw));
    setVals((p) => ({ ...p, [key]: v }));
  };

  return (
    <div className="flex flex-col gap-2 h-full min-h-0">

      {/* 헤더 */}
      <div className="ascii-panel shrink-0">
        <span className="ascii-panel-title">DEBUG CONSOLE</span>
        <div className="text-dos-amber font-bold mb-1" style={{ fontSize: "0.8rem" }}>
          ▶▶ DIRECT GAUGE OVERRIDE
        </div>
        <div className="text-dos-green-dim" style={{ fontSize: "0.58rem" }}>
          WARNING: frontend-only — backend state unchanged.
          Values reset on next policy action.
        </div>
      </div>

      {/* 슬라이더 영역 */}
      <div className="ascii-panel flex-1 min-h-0 flex flex-col justify-around"
           style={{ padding: "0.75rem" }}>
        <span className="ascii-panel-title">GAUGE CONTROL</span>

        {PARAMS.map(({ key, label, dangerHigh }) => {
          const v     = vals[key];
          const color = gaugeColor(v, dangerHigh);
          const filled = Math.round(v / 10);
          const bar   = "#".repeat(filled) + ".".repeat(10 - filled);

          return (
            <div key={key} className="flex items-center gap-2"
                 style={{ fontSize: "0.65rem", fontFamily: "var(--font-geist-mono)" }}>

              {/* 라벨 */}
              <span style={{ color: "#1a7a0a", minWidth: "5.5rem" }}>{label}</span>

              {/* ASCII 바 */}
              <span style={{ color: "#1a7a0a" }}>{"["}</span>
              <span style={{ color }}>{bar}</span>
              <span style={{ color: "#1a7a0a" }}>{"]"}</span>

              {/* 수치 */}
              <span style={{ color, minWidth: "2rem", textAlign: "right" }}>
                {String(v).padStart(3)}
              </span>

              {/* 슬라이더 */}
              <input
                type="range"
                min={0} max={100} value={v}
                onChange={(e) => set(key, Number(e.target.value))}
                className="flex-1"
                style={{ accentColor: color, cursor: "pointer" }}
              />

              {/* 숫자 직접 입력 */}
              <input
                type="number"
                min={0} max={100} value={v}
                onChange={(e) => set(key, Number(e.target.value))}
                style={{
                  width: "3.2rem",
                  background: "#000",
                  border: `1px solid ${color}`,
                  color,
                  fontFamily: "var(--font-geist-mono)",
                  fontSize: "0.65rem",
                  padding: "2px 4px",
                  textAlign: "right",
                }}
              />
            </div>
          );
        })}
      </div>

      {/* 버튼 */}
      <div className="shrink-0 flex gap-3 justify-between items-center">
        <button
          onClick={() => setVals({ debt: 50, inflation: 50, morale: 50, tension: 50 })}
          className="font-mono border border-dos-green-dim text-dos-green-dim px-4 py-1
                     hover:border-dos-green hover:text-dos-green transition-colors"
          style={{ fontSize: "0.6rem" }}
        >
          [ RESET ALL → 50 ]
        </button>
        <div className="flex gap-2">
          <button
            onClick={onCancel}
            className="font-mono border border-dos-green-dim text-dos-green-dim px-5 py-1
                       hover:border-dos-green hover:text-dos-green transition-colors"
            style={{ fontSize: "0.65rem" }}
          >
            [ CANCEL ]
          </button>
          <button
            onClick={() => onApply(vals)}
            className="font-mono font-bold border border-dos-amber text-dos-amber px-5 py-1
                       hover:bg-dos-amber hover:text-black transition-colors"
            style={{ fontSize: "0.65rem" }}
          >
            [ APPLY ]
          </button>
        </div>
      </div>
    </div>
  );
}
