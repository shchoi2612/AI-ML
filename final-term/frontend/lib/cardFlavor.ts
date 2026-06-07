// 정책 카드의 "정성적 분위기 텍스트" — 결과 수치(±%, 게이지 변화)는 절대 노출하지 않는다.
// 카드는 "무엇을 하는 정책인가"의 무드만 보여주고, 결과는 플레이어가 직접 당해서 안다.
// (백엔드 card id 기준. 새 카드가 생기면 여기에 한 줄 추가.)
export const CARD_FLAVOR: Record<string, string> = {
  interest_rate_hike: "돈줄을 죈다. 시장이 숨을 죽인다.",
  issue_bonds: "빚을 내어 경기에 불을 지핀다.",
  welfare_expansion: "곳간을 열어 민심을 산다.",
  austerity: "허리띠를 졸라맨다 — 고통은 국민의 몫.",
  strategic_oil_release: "비축유를 푼다. 기름값에 손을 댄다.",
  renewable_investment: "미래에 베팅한다. 대가는 지금 치른다.",
  defense_buildup: "칼을 간다. 이웃이 긴장한다.",
  diplomatic_mediation: "협상 테이블로 적을 불러 앉힌다.",
  chip_subsidy: "전략 산업에 실탄을 쏟아붓는다.",
  fab_construction: "국운을 건 한 방 — 거대한 도박.",
};

export function flavorFor(id: string): string {
  return CARD_FLAVOR[id] ?? "내각이 펜을 든다. 결과는 시장이 말해줄 것이다.";
}

// 섹터 한글 라벨 + 색
export const SECTOR_META: Record<string, { label: string; color: string }> = {
  energy: { label: "에너지", color: "#ffb627" },
  defense: { label: "방위", color: "#ff7a59" },
  semiconductor: { label: "반도체", color: "#5b9bd5" },
};
