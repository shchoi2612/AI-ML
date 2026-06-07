"use client";
import { useState, useMemo } from "react";
import type { Card, Situation, SectorResources } from "@/lib/api";

const SECTOR_LABEL: Record<string, string> = {
  energy: "에너지",
  defense: "방위",
  semiconductor: "반도체",
};

interface CardPoolProps {
  situation: Situation;
  cards: Card[];
  capacity: number;
  resources: SectorResources;
  onCommit: (cardIds: string[]) => void;
  disabled?: boolean;
}

export function CardPool({
  situation, cards, capacity, resources, onCommit, disabled,
}: CardPoolProps) {
  const [selected, setSelected] = useState<string[]>([]);

  // 선택분 합산 비용
  const spent = useMemo(() => {
    let fiscal = 0;
    const sectors: Record<string, number> = { energy: 0, defense: 0, semiconductor: 0 };
    for (const id of selected) {
      const c = cards.find((x) => x.id === id);
      if (!c) continue;
      fiscal += c.fiscal_cost;
      if (c.sector) sectors[c.sector] += c.sector_cost;
    }
    return { fiscal, sectors };
  }, [selected, cards]);

  const remainingFiscal = capacity - spent.fiscal;
  const remainingSector = (s: keyof SectorResources) => resources[s] - spent.sectors[s];

  // 지금 추가로 고를 수 있는가 (이미 고른 건 항상 해제 가능)
  function canSelect(c: Card): boolean {
    if (selected.includes(c.id)) return true;
    if (c.fiscal_cost > remainingFiscal) return false;
    if (c.sector && c.sector_cost > remainingSector(c.sector)) return false;
    return true;
  }

  function toggle(c: Card) {
    if (disabled) return;
    setSelected((prev) =>
      prev.includes(c.id) ? prev.filter((x) => x !== c.id) : canSelect(c) ? [...prev, c.id] : prev
    );
  }

  function commit() {
    if (disabled) return;
    onCommit(selected);
    setSelected([]);
  }

  return (
    <div className="flex flex-col gap-2 h-full min-h-0">

      {/* 상황 배너 */}
      <div className="relative shrink-0 border border-dos-green mt-3 px-3 pb-2 pt-4">
        <span className="absolute -top-2 left-3 bg-black px-1 text-dos-amber font-bold tracking-widest" style={{ fontSize: "0.6rem" }}>
          SITUATION
        </span>
        <div style={{ fontSize: "0.82rem", color: "#ffaa00", fontWeight: "bold", lineHeight: 1.4 }}>
          ▶ {situation.title}
        </div>
        <div style={{ fontSize: "0.6rem", color: "#1a7a0a", lineHeight: 1.6, marginTop: 4 }}>
          {situation.desc}
        </div>
      </div>

      {/* 예산 바: 재정 여력 + 섹터 자원 */}
      <div className="shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 px-1" style={{ fontSize: "0.62rem" }}>
        <span style={{ color: remainingFiscal < 0 ? "#ff3333" : "#39ff14" }}>
          재정 여력 <b>{remainingFiscal}</b>/{capacity}
        </span>
        {(["energy", "defense", "semiconductor"] as const).map((s) => (
          <span key={s} style={{ color: "#1a7a0a" }}>
            {SECTOR_LABEL[s]} 자원 <b style={{ color: "#39ff14" }}>{remainingSector(s)}</b>/{resources[s]}
          </span>
        ))}
        <span className="ml-auto text-dos-green-dim">선택 {selected.length}장</span>
      </div>

      {/* 카드풀 (상시, 코스트 제한) */}
      <div className="flex-1 min-h-0 overflow-y-auto grid gap-2 pr-1"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", alignContent: "start" }}>
        {cards.map((c) => {
          const isSel = selected.includes(c.id);
          const usable = canSelect(c);
          return (
            <button
              key={c.id}
              onClick={() => toggle(c)}
              disabled={disabled || (!isSel && !usable)}
              style={{ textAlign: "left", cursor: disabled ? "not-allowed" : "pointer" }}
              className={[
                "ascii-panel flex flex-col gap-1 transition-colors duration-100",
                isSel ? "border-dos-amber" : "",
                !isSel && !usable ? "opacity-30" : "hover:border-dos-amber",
              ].join(" ")}
            >
              <div className="flex items-center justify-between">
                <span style={{ fontSize: "0.5rem", color: c.sector ? "#ffaa00" : "#1a7a0a" }}>
                  {c.sector ? SECTOR_LABEL[c.sector] : "일반"}
                </span>
                <span style={{ fontSize: "0.55rem", color: "#39ff14" }}>
                  {isSel ? "■ 선택" : "□"}
                </span>
              </div>
              <div style={{ fontSize: "0.72rem", color: "#39ff14", fontWeight: "bold", lineHeight: 1.25 }}>
                {c.title}
              </div>
              <div style={{ fontSize: "0.56rem", color: "#ffaa00" }}>
                재정 {c.fiscal_cost}{c.sector ? ` · ${SECTOR_LABEL[c.sector]} ${c.sector_cost}` : ""}
              </div>
              <div style={{ fontSize: "0.56rem", color: "#1a7a0a", lineHeight: 1.5 }}>
                {c.hint}
              </div>
            </button>
          );
        })}
      </div>

      {/* 집행 버튼 (0장=패스 허용) */}
      <div className="shrink-0 flex justify-center items-center gap-3 py-1">
        <button
          onClick={commit}
          disabled={disabled}
          className="font-mono font-bold border border-dos-amber text-dos-amber px-8 py-1 hover:bg-dos-amber hover:text-black transition-colors disabled:opacity-40"
          style={{ fontSize: "0.7rem", letterSpacing: "0.08em" }}
        >
          {selected.length > 0 ? `▶▶ 정책 집행 (${selected.length}장, 재정 ${spent.fiscal})` : "▶▶ 이번 턴 패스"}
        </button>
        {disabled && (
          <span className="text-dos-green-dim blink font-mono" style={{ fontSize: "0.6rem" }}>PROCESSING...</span>
        )}
      </div>
    </div>
  );
}
