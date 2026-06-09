"use client";
import type { Gauges } from "@/lib/api";

// 국가 지표 — 숫자 없이 아이콘 + 채움 막대. 위험 구간이면 크림슨 + "위험" 점멸.
const METER: { key: keyof Gauges; label: string; glyph: string; dangerHigh: boolean }[] = [
  { key: "debt",      label: "국가부채",   glyph: "₩", dangerHigh: true  },
  { key: "inflation", label: "인플레이션", glyph: "▲", dangerHigh: true  },
  { key: "morale",    label: "민심",       glyph: "♥", dangerHigh: false },
  { key: "tension",   label: "국제긴장",   glyph: "⚡", dangerHigh: true  },
];

function danger(value: number, dangerHigh: boolean): 0 | 1 | 2 {
  const d = dangerHigh ? value : 100 - value;
  if (d >= 80) return 2;
  if (d >= 60) return 1;
  return 0;
}

const FILL = ["var(--stable)", "var(--gold)", "var(--crisis)"];

export function StatusBars({ gauges }: { gauges: Gauges }) {
  return (
    <div className="sr-panel px-3 py-2 flex flex-col gap-2">
      <div className="sr-label">국가 지표</div>
      {METER.map(({ key, label, glyph, dangerHigh }) => {
        const v = gauges[key];
        const lvl = danger(v, dangerHigh);
        const color = FILL[lvl];
        const crit = lvl === 2;
        return (
          <div key={key} className="flex items-center gap-2">
            <span style={{ color, width: "1.1em", textAlign: "center", fontSize: "0.85rem" }}>{glyph}</span>
            <span className="sr-display" style={{ color: "var(--sr-ink)", fontSize: "0.66rem", width: "4.6em", letterSpacing: "-0.02em" }}>
              {label}
            </span>
            <div className="flex-1 h-[10px] rounded-sm overflow-hidden"
                 style={{ background: "#0a121b", border: "1px solid var(--sr-border)" }}>
              <div className={crit ? "sr-danger h-full" : "h-full"}
                   style={{ width: `${v}%`, background: color, transition: "width 0.5s cubic-bezier(0.2,0.9,0.2,1)" }} />
            </div>
            {crit ? (
              <span className="blink sr-display" style={{ color: "var(--crisis)", fontSize: "0.58rem", fontWeight: 800, width: "2.4em" }}>위험</span>
            ) : (
              <span style={{ width: "2.4em" }} />
            )}
          </div>
        );
      })}
    </div>
  );
}
