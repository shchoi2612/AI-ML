import type { Gauges } from "@/lib/api";

export type Mood = "joy" | "neutral" | "sad" | "angry";

// 위험 임계 (이 값 이상이면 위험)
export const DANGER = 80;

// 표정은 프론트에서만 계산한다 (백엔드 게이지 기반, 계약 불변).
// 부채·인플레·긴장 중 하나라도 위험 임계 넘으면 화남.
// 아니면 안정도 점수로 즐거움/그저그럼/슬픔.
export function stability(g: Gauges): number {
  return (100 - g.debt) + (100 - g.inflation) + g.morale + (100 - g.tension); // 0..400
}

export function moodFromGauges(g: Gauges): Mood {
  if (g.debt >= DANGER || g.inflation >= DANGER || g.tension >= DANGER) return "angry";
  const s = stability(g);
  if (s >= 280) return "joy";
  if (s >= 180) return "neutral";
  return "sad";
}

// 나빠지는 방향 판정용 순위 (높을수록 나쁨)
export const MOOD_RANK: Record<Mood, number> = { joy: 0, neutral: 1, sad: 2, angry: 3 };

// 상황실 팔레트: 안정=teal, 그저그럼=골드, 슬픔=오렌지, 화남=크림슨
export const MOOD_META: Record<Mood, { label: string; color: string }> = {
  joy:     { label: "안정",     color: "#3fb6a8" },
  neutral: { label: "그저그럼", color: "#ffb627" },
  sad:     { label: "침체",     color: "#ff8a3d" },
  angry:   { label: "위기",     color: "#ff3b46" },
};
