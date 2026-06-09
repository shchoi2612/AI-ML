"use client";
import { useEffect, useState } from "react";
import type { Situation } from "@/lib/api";

// 큰 위기(major)가 뜨면 화면 가운데 "속보" 풀스크린으로 슬램 + 제목 타이핑.
// 클릭하거나 잠시 뒤 자동으로 닫힌다. (light/medium은 띄우지 않음 — CardPool 배너로 충분)
interface CrisisAlertProps {
  situation: Situation;
  onDismiss: () => void;
}

export function CrisisAlert({ situation, onDismiss }: CrisisAlertProps) {
  const [typed, setTyped] = useState("");
  const title = situation.title;

  // 제목 타이핑
  useEffect(() => {
    setTyped("");
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setTyped(title.slice(0, i));
      if (i >= title.length) clearInterval(id);
    }, 45);
    return () => clearInterval(id);
  }, [title]);

  // 자동 닫힘 (5.5초)
  useEffect(() => {
    const t = setTimeout(onDismiss, 5500);
    return () => clearTimeout(t);
  }, [onDismiss]);

  return (
    <div
      onClick={onDismiss}
      className="fixed inset-0 z-50 flex items-center justify-center sr-crisis-in"
      style={{ background: "rgba(6,8,12,0.86)", cursor: "pointer", backdropFilter: "blur(2px)" }}
    >
      <div className="text-center px-8" style={{ maxWidth: 720 }}>
        {/* 속보 헤더 */}
        <div className="inline-flex items-center gap-2 px-4 py-1 mb-5"
             style={{ background: "var(--crisis)", color: "#fff" }}>
          <span style={{ width: 9, height: 9, borderRadius: 9, background: "#fff" }} className="blink" />
          <span className="sr-display" style={{ fontWeight: 900, letterSpacing: "0.22em", fontSize: "1.1rem" }}>
            속보 · BREAKING
          </span>
        </div>

        {/* 강도 라벨 */}
        <div className="sr-label mb-2" style={{ color: "var(--crisis)", letterSpacing: "0.18em", fontWeight: 800 }}>
          ■ 중대 위기 발생
        </div>

        {/* 제목 (타이핑) */}
        <div className="sr-display" style={{
          fontSize: "2rem", fontWeight: 900, color: "#fff", lineHeight: 1.2,
          textShadow: "0 0 22px rgba(255,59,70,0.5)",
        }}>
          {typed}<span className="blink" style={{ color: "var(--crisis)" }}>▌</span>
        </div>

        {/* 설명 */}
        <div style={{ fontSize: "0.82rem", color: "var(--sr-mut)", lineHeight: 1.7, marginTop: 16 }}>
          {situation.desc}
        </div>

        <div className="sr-label" style={{ color: "var(--sr-dim)", marginTop: 26 }}>
          [ 화면을 누르면 대응에 들어갑니다 ]
        </div>
      </div>
    </div>
  );
}
