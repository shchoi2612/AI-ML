"use client";
import type { GameEvent } from "@/lib/api";

const KEYS = ["A", "B", "C"] as const;

interface PolicyInputProps {
  event: GameEvent;
  onChoice: (index: number) => void;
  disabled?: boolean;
}

export function PolicyInput({ event, onChoice, disabled }: PolicyInputProps) {
  return (
    <div className="flex flex-col gap-2 h-full min-h-0">

      {/* 이벤트 패널 */}
      <div className="relative shrink-0 border border-dos-green mt-3 px-3 pb-3 pt-4">
        <span
          className="absolute -top-2 left-3 bg-black px-1 text-dos-amber font-bold tracking-widest"
          style={{ fontSize: "0.6rem" }}
        >
          CRISIS ALERT
        </span>
        <div style={{ fontSize: "0.82rem", color: "#ffaa00", fontWeight: "bold", lineHeight: 1.4, fontFamily: "var(--font-geist-mono)" }}>
          ▶ {event.title}
        </div>
      </div>

      {/* 이벤트 설명 */}
      <div className="shrink-0 px-1" style={{ fontSize: "0.62rem", color: "#1a7a0a", lineHeight: 1.7 }}>
        {event.desc}
      </div>

      {/* ── 3개 선택 카드 ── */}
      <div className="flex gap-2 flex-1 min-h-0">
        {event.choices.map((choice, i) => (
          <button
            key={i}
            onClick={() => onChoice(i)}
            disabled={disabled}
            style={{ flex: 1, minHeight: 0, textAlign: "left", cursor: disabled ? "not-allowed" : "pointer" }}
            className={[
              "ascii-panel flex flex-col gap-2 transition-colors duration-100",
              disabled ? "opacity-40" : "hover:border-dos-amber",
            ].join(" ")}
          >
            <div>
              <span style={{ fontSize: "0.9rem", fontWeight: "bold", color: "#000", background: "#ffaa00", padding: "1px 7px", fontFamily: "var(--font-geist-mono)" }}>
                [{KEYS[i]}]
              </span>
            </div>
            <div style={{ fontSize: "0.82rem", color: "#39ff14", fontWeight: "bold", lineHeight: 1.3, fontFamily: "var(--font-geist-mono)" }}>
              {choice.label}
            </div>
            <div style={{ fontSize: "0.58rem", color: "#1a7a0a", overflow: "hidden", whiteSpace: "nowrap" }}>
              {"─".repeat(40)}
            </div>
            <div style={{ fontSize: "0.62rem", color: "#1a7a0a", lineHeight: 1.7, fontFamily: "var(--font-geist-mono)" }}>
              {choice.hint.split(", ").map((line, j) => (
                <div key={j}>{"  › " + line}</div>
              ))}
            </div>
          </button>
        ))}
      </div>

      {disabled && (
        <div className="shrink-0 text-center text-dos-green-dim blink font-mono" style={{ fontSize: "0.6rem" }}>
          PROCESSING...
        </div>
      )}
    </div>
  );
}
