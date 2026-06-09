"use client";
import { motion } from "framer-motion";
import type { Gauges } from "@/lib/api";

const GAUGES: {
  key: keyof Gauges;
  short: string;
  dangerHigh: boolean;
}[] = [
  { key: "debt",      short: "DEBT", dangerHigh: true  },
  { key: "inflation", short: "INFL", dangerHigh: true  },
  { key: "morale",    short: "MORL", dangerHigh: false },
  { key: "tension",   short: "TENS", dangerHigh: true  },
];

function dangerLevel(value: number, dangerHigh: boolean): 0 | 1 | 2 {
  const d = dangerHigh ? value : 100 - value;
  if (d >= 80) return 2;
  if (d >= 60) return 1;
  return 0;
}

const DANGER_COLOR = ["#39ff14", "#ffaa00", "#ff3333"] as const;
const DANGER_ICON  = ["  ", " !", "!!"] as const;

const FS = "0.65rem";

interface GaugePanelProps {
  gauges: Gauges;
  deltas?: Partial<Gauges>;
}

export function GaugePanel({ gauges, deltas }: GaugePanelProps) {
  return (
    <div className="ascii-panel font-mono overflow-hidden" style={{ fontSize: FS, height: "calc(100% - 0.75rem)" }}>
      <span className="ascii-panel-title">NATIONAL STATUS</span>

      {/* white-space:pre 로 열 정렬 보장 */}
      <div style={{ whiteSpace: "pre", lineHeight: "1.55" }}>

        {/* 헤더 */}
        <span style={{ color: "#1a7a0a" }}>{"NAME [##########] VAL\n"}</span>
        <span style={{ color: "#1a7a0a" }}>{"─────────────────────\n"}</span>

        {GAUGES.map(({ key, short, dangerHigh }) => {
          const value  = gauges[key];
          const delta  = deltas?.[key] ?? 0;
          const lvl    = dangerLevel(value, dangerHigh);
          const color  = DANGER_COLOR[lvl];
          const icon   = DANGER_ICON[lvl];
          const filled = Math.round(value / 10);
          const bar    = "#".repeat(filled) + ".".repeat(10 - filled);

          return (
            <motion.div
              key={key}
              initial={false}
              animate={{ opacity: 1 }}
              style={{ display: "inline" }}
            >
              {/* 라벨: 4자 고정 */}
              <span style={{ color: "#1a7a0a" }}>{short}</span>
              {/* 바 */}
              <span style={{ color: "#1a7a0a" }}>{" ["}</span>
              <span style={{ color }}>{bar}</span>
              <span style={{ color: "#1a7a0a" }}>{"]"}</span>
              {/* 값: 3자 고정 */}
              <span style={{ color }}>{" " + String(value).padStart(3)}</span>
              {/* 위험 아이콘 */}
              <span style={{ color, fontWeight: "bold" }}>{icon}</span>
              {/* 델타 */}
              {delta !== 0 && (
                <span style={{ color: delta > 0 ? "#ff3333" : "#39ff14" }}>
                  {delta > 0 ? "+" : ""}{delta}
                </span>
              )}
              {"\n"}
            </motion.div>
          );
        })}

        {/* 하단 구분선 */}
        <span style={{ color: "#1a7a0a" }}>{"─────────────────────\n"}</span>
        <span style={{ color: "#1a7a0a" }}>{" ! WARN  >=60\n"}</span>
        <span style={{ color: "#ff3333" }}>{"!! CRIT  >=80\n"}</span>
      </div>
    </div>
  );
}
