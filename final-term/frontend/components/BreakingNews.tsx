"use client";
import { useEffect, useState, useRef } from "react";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const IS_MOCK = process.env.NEXT_PUBLIC_API_MODE === "mock";

// 엔진 narration(SSE)을 "속보 자막(chyron)"으로 띄운다.
// GROQ 키가 비어 있으면 백엔드가 폴백 문장을 보내고, 키를 꽂으면 자동으로 LLM 문장이 흐른다.
interface BreakingNewsProps {
  gameId: string;
  turn: number | null;   // 내레이션을 받을 액션 턴 (없으면 대기 문구)
}

export function BreakingNews({ gameId, turn }: BreakingNewsProps) {
  const [text, setText] = useState("국정 브리핑 대기 중…");
  const [slamKey, setSlamKey] = useState(0);
  const [live, setLive] = useState(false);
  const acc = useRef("");

  useEffect(() => {
    if (turn == null) return;
    let aborted = false;
    const ctrl = new AbortController();
    acc.current = "";
    setText("");
    setLive(true);
    setSlamKey((k) => k + 1);

    async function run() {
      if (IS_MOCK) {
        setText("속보 — 정부가 새 정책을 단행했다. 시장이 반응하기 시작한다.");
        setLive(false);
        return;
      }
      try {
        const res = await fetch(`${BASE_URL}/game/${gameId}/turn/${turn}/narration`, { signal: ctrl.signal });
        if (!res.ok || !res.body) throw new Error(String(res.status));
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const ev = JSON.parse(line.slice(6));
              if (ev.type === "narration_chunk" && ev.text) {
                acc.current += ev.text;
                if (!aborted) setText(acc.current);
              }
            } catch { /* 부분 프레임 무시 */ }
          }
        }
      } catch {
        if (!aborted) setText("속보 — 정부가 정책을 단행했다. 시장이 술렁인다.");
      } finally {
        if (!aborted) setLive(false);
      }
    }
    run();
    return () => { aborted = true; ctrl.abort(); };
  }, [gameId, turn]);

  return (
    <div className="sr-panel overflow-hidden" style={{ borderColor: "var(--crisis-deep)" }}>
      <div className="flex items-stretch">
        <div className="shrink-0 flex items-center gap-1 px-3"
             style={{ background: "var(--crisis)", color: "#fff" }}>
          <span style={{ width: 7, height: 7, borderRadius: 9, background: "#fff" }}
                className={live ? "blink" : ""} />
          <span className="sr-display" style={{ fontWeight: 800, fontSize: "0.74rem", letterSpacing: "0.06em" }}>
            속보
          </span>
        </div>
        <div key={slamKey} className="sr-slam flex-1 px-3 py-2 flex items-center"
             style={{ minHeight: "2.4em" }}>
          <span className="sr-display" style={{ color: "var(--sr-ink)", fontSize: "0.82rem", lineHeight: 1.35 }}>
            {text || "…"}
          </span>
        </div>
      </div>
    </div>
  );
}
