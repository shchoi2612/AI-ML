"use client";
import type { Mood } from "@/lib/expression";
import { MOOD_META } from "@/lib/expression";

// 국가의 표정 — SVG 선화(터미널 톤). 표정 4종. 표정 전환은 부드럽게(face-in).
function FacePaths({ mood, color }: { mood: Mood; color: string }) {
  const s = { fill: "none", stroke: color, strokeWidth: 4, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (mood) {
    case "joy":
      return (
        <>
          {/* 웃는 눈 ^ ^ */}
          <path d="M62 96 Q78 80 94 96" {...s} />
          <path d="M126 96 Q142 80 158 96" {...s} />
          {/* 눈썹 (편안히 올라감) */}
          <path d="M60 74 Q78 66 96 74" {...s} strokeWidth={3} />
          <path d="M124 74 Q142 66 160 74" {...s} strokeWidth={3} />
          {/* 큰 미소 */}
          <path d="M66 136 Q110 184 154 136" {...s} />
        </>
      );
    case "neutral":
      return (
        <>
          <circle cx={78} cy={94} r={7} {...s} />
          <circle cx={142} cy={94} r={7} {...s} />
          <path d="M62 74 L96 74" {...s} strokeWidth={3} />
          <path d="M124 74 L158 74" {...s} strokeWidth={3} />
          <path d="M80 150 L140 150" {...s} />
        </>
      );
    case "sad":
      return (
        <>
          <circle cx={78} cy={98} r={7} {...s} />
          <circle cx={142} cy={98} r={7} {...s} />
          {/* 눈썹 안쪽 올라감 (\  /) */}
          <path d="M62 80 L96 70" {...s} strokeWidth={3} />
          <path d="M124 70 L158 80" {...s} strokeWidth={3} />
          {/* 찌푸린 입 */}
          <path d="M70 162 Q110 126 150 162" {...s} />
        </>
      );
    case "angry":
      return (
        <>
          {/* 가늘게 뜬 눈 */}
          <path d="M64 96 L96 102" {...s} />
          <path d="M124 102 L156 96" {...s} />
          {/* 눈썹 안쪽 내려감 (V) */}
          <path d="M60 70 L98 88" {...s} strokeWidth={4} />
          <path d="M122 88 L160 70" {...s} strokeWidth={4} />
          {/* 악문 입 + 이 */}
          <path d="M68 160 Q110 132 152 160" {...s} />
          <path d="M86 150 L86 158" {...s} strokeWidth={2.5} />
          <path d="M110 146 L110 156" {...s} strokeWidth={2.5} />
          <path d="M134 150 L134 158" {...s} strokeWidth={2.5} />
        </>
      );
  }
}

export function NationalFace({ mood }: { mood: Mood }) {
  const { label, color } = MOOD_META[mood];
  return (
    <div className="flex flex-col items-center justify-center gap-1 h-full select-none">
      <div className="sr-label">국가의 표정</div>
      <div key={mood} className="face-in" style={{ filter: `drop-shadow(0 0 8px ${color}55)`, transition: "filter 0.4s" }}>
        <svg viewBox="0 0 220 220" width="100%" height="100%"
             style={{ maxWidth: 132, maxHeight: 132, display: "block" }}>
          <circle cx={110} cy={110} r={86} fill="none" stroke={color} strokeWidth={5} style={{ transition: "stroke 0.4s" }} />
          <FacePaths mood={mood} color={color} />
        </svg>
      </div>
      <div key={`${mood}-label`} className="face-in sr-display" style={{ color, fontSize: "0.95rem", fontWeight: 800, letterSpacing: "0.08em" }}>
        {label}
      </div>
    </div>
  );
}
