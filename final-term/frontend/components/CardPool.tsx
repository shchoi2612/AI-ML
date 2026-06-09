"use client";
import { useState, useMemo } from "react";
import type { Card, Situation, SectorResources } from "@/lib/api";
import { flavorFor, SECTOR_META } from "@/lib/cardFlavor";

function Pips({ filled, total, color, dim = "var(--sr-dim)", size = 7 }: {
  filled: number; total: number; color: string; dim?: string; size?: number;
}) {
  return (
    <span style={{ display: "inline-flex", gap: 2, alignItems: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <span key={i} style={{
          width: size, height: size, borderRadius: size,
          background: i < filled ? color : "transparent",
          border: `1px solid ${i < filled ? color : dim}`,
        }} />
      ))}
    </span>
  );
}

interface CardPoolProps {
  situation: Situation;
  cards: Card[];
  capacity: number;
  resources: SectorResources;
  onCommit: (cardIds: string[]) => void;
  disabled?: boolean;
}

export function CardPool({ situation, cards, capacity, resources, onCommit, disabled }: CardPoolProps) {
  const [selected, setSelected] = useState<string[]>([]);

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

  const remFiscal = capacity - spent.fiscal;
  const remSector = (s: keyof SectorResources) => resources[s] - spent.sectors[s];

  function canSelect(c: Card): boolean {
    if (selected.includes(c.id)) return true;
    if (c.fiscal_cost > remFiscal) return false;
    if (c.sector && c.sector_cost > remSector(c.sector)) return false;
    return true;
  }

  function toggle(c: Card) {
    if (disabled) return;
    setSelected((p) => (p.includes(c.id) ? p.filter((x) => x !== c.id) : canSelect(c) ? [...p, c.id] : p));
  }

  function commit() {
    if (disabled) return;
    onCommit(selected);
    setSelected([]);
  }

  return (
    <div className="flex flex-col gap-2 h-full min-h-0">

      {/* ── 상황 배너 (강도별 색/라벨) ── */}
      {(() => {
        const sev = situation.severity ?? "light";
        const sc = sev === "major" ? "var(--crisis)" : sev === "medium" ? "var(--gold)" : "var(--sr-dim)";
        const label = situation.severity_label ?? "사건";
        return (
          <div className="sr-panel shrink-0 px-4 py-3" style={{ borderLeft: `3px solid ${sc}` }}>
            <div className="flex items-center gap-2">
              <span className="sr-label" style={{ color: sc, fontWeight: 800, letterSpacing: "0.1em" }}>
                {sev === "major" ? "■ 중대 위기" : sev === "medium" ? "▲ 위기" : "· 사건"}
              </span>
              <span className="sr-label" style={{ color: "var(--sr-dim)" }}>SITUATION · {label}</span>
            </div>
            <div className="sr-display" style={{ fontSize: "1.15rem", fontWeight: 800, color: "var(--sr-ink)", lineHeight: 1.25, marginTop: 3 }}>
              {situation.title}
            </div>
            <div style={{ fontSize: "0.66rem", color: "var(--sr-mut)", lineHeight: 1.6, marginTop: 5 }}>
              {situation.desc}
            </div>
          </div>
        );
      })()}

      {/* ── 예산 바 (핍, 숫자 없음) ── */}
      <div className="sr-panel shrink-0 px-4 py-2 flex flex-wrap items-center gap-x-5 gap-y-1">
        <span className="flex items-center gap-2">
          <span className="sr-label" style={{ color: remFiscal < 0 ? "var(--crisis)" : "var(--gold)" }}>재정 여력</span>
          <Pips filled={Math.max(0, remFiscal)} total={capacity} color="var(--gold)" size={6} />
        </span>
        {(["energy", "defense", "semiconductor"] as const).map((s) => (
          <span key={s} className="flex items-center gap-2">
            <span className="sr-label">{SECTOR_META[s].label}</span>
            <Pips filled={Math.max(0, remSector(s))} total={Math.max(resources[s], 1)} color={SECTOR_META[s].color} size={6} />
          </span>
        ))}
        <span className="ml-auto sr-label" style={{ color: "var(--sr-mut)" }}>선택 {selected.length}</span>
      </div>

      {/* ── 정책 카드풀 (주인공) ── */}
      <div className="flex-1 min-h-0 overflow-y-auto grid gap-2 pr-1"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(188px, 1fr))", alignContent: "start" }}>
        {cards.map((c) => {
          const isSel = selected.includes(c.id);
          const usable = canSelect(c);
          const locked = !isSel && !usable;
          const sm = c.sector ? SECTOR_META[c.sector] : null;
          return (
            <button
              key={c.id}
              onClick={() => toggle(c)}
              disabled={disabled || locked}
              className={isSel ? "sr-panel" : "sr-panel"}
              style={{
                textAlign: "left", padding: "10px 12px",
                display: "flex", flexDirection: "column", gap: 6,
                cursor: disabled ? "not-allowed" : locked ? "not-allowed" : "pointer",
                opacity: locked ? 0.34 : 1,
                borderColor: isSel ? "var(--gold)" : "var(--sr-border)",
                boxShadow: isSel ? "0 0 0 1px var(--gold), 0 0 16px -4px var(--gold)" : "none",
                transition: "border-color 0.12s, box-shadow 0.12s, opacity 0.12s",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="sr-label" style={{ color: sm ? sm.color : "var(--sr-mut)", letterSpacing: "0.12em" }}>
                    {sm ? sm.label : "일반 정책"}
                  </span>
                  {c.discounted && (
                    <span className="sr-label" style={{
                      color: "var(--stable)", border: "1px solid var(--stable)",
                      borderRadius: 3, padding: "0 3px", fontSize: "0.5rem", letterSpacing: 0,
                    }}>
                      위기대응
                    </span>
                  )}
                </span>
                <span style={{ fontSize: "0.62rem", color: isSel ? "var(--gold)" : "var(--sr-dim)" }}>
                  {locked ? "🔒" : isSel ? "■ 선정" : "□"}
                </span>
              </div>

              <div className="sr-display" style={{ fontSize: "0.92rem", fontWeight: 800, color: "var(--sr-ink)", lineHeight: 1.2 }}>
                {c.title}
              </div>

              <div style={{ fontSize: "0.64rem", color: "var(--sr-mut)", lineHeight: 1.5, flex: 1 }}>
                {flavorFor(c.id)}
              </div>

              {/* 코스트 (핍, 숫자 없음) — 할인 시 원래 칸은 빈 윤곽으로 남겨 절감 표시 */}
              <div className="flex items-center gap-3" style={{ marginTop: 2 }}>
                <span className="flex items-center gap-1">
                  <span style={{ fontSize: "0.52rem", color: "var(--sr-dim)" }}>재정</span>
                  <Pips filled={c.fiscal_cost}
                        total={Math.max(c.fiscal_cost, c.base_fiscal_cost ?? c.fiscal_cost)}
                        color="var(--gold)" size={5} />
                </span>
                {sm && (c.sector_cost > 0 || (c.base_sector_cost ?? 0) > 0) && (
                  <span className="flex items-center gap-1">
                    <span style={{ fontSize: "0.52rem", color: "var(--sr-dim)" }}>{sm.label}</span>
                    <Pips filled={c.sector_cost}
                          total={Math.max(c.sector_cost, c.base_sector_cost ?? c.sector_cost)}
                          color={sm.color} size={5} />
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* ── 집행 버튼 ── */}
      <div className="shrink-0 flex justify-center items-center gap-3 py-1">
        <button
          onClick={commit}
          disabled={disabled}
          className="sr-display px-8 py-2 rounded"
          style={{
            background: selected.length > 0 ? "var(--crisis)" : "transparent",
            color: selected.length > 0 ? "#fff" : "var(--sr-mut)",
            border: `1px solid ${selected.length > 0 ? "var(--crisis)" : "var(--sr-border)"}`,
            fontWeight: 800, letterSpacing: "0.06em", fontSize: "0.82rem",
            opacity: disabled ? 0.4 : 1, cursor: disabled ? "not-allowed" : "pointer",
          }}
        >
          {selected.length > 0 ? `▶ 정책 집행 (${selected.length})` : "이번 달 관망"}
        </button>
        {disabled && <span className="blink sr-label" style={{ color: "var(--crisis)" }}>집행 중…</span>}
      </div>
    </div>
  );
}
