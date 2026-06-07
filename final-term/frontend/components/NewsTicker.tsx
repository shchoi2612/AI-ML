"use client";
import { useState, useEffect } from "react";
import type { Gauges } from "@/lib/api";

const SILLY: string[] = [
  "포크가 부엌에서 발견... '예상된 위치'",
  "한 남성, 어젯밤 숙면 취했다고 밝혀",
  "전국 구름, 오늘도 하늘에 떠 있는 것으로 확인",
  "전문가: '밥은 먹어야 산다'... 국민들 '공감'",
  "지역 고양이, 창문 밖 바라보며 사색에 잠겨",
  "과학자들, 물이 여전히 축축하다는 사실 재확인",
  "시민 A씨 '오늘 점심 먹었다'... 주변 반응 무덤덤",
  "어젯밤 달이 떴다는 사실 밝혀져... 전문가 '예상된 일'",
  "한 어린이, 바나나 껍질 미끄러운지 실험 중",
  "지역 빵집 빵 맛있다는 제보 접수... 취재진 확인 중",
  "냉장고 문 닫으면 불 꺼진다는 설 '또 제기'",
  "최고 수면 전문가 인터뷰 중 꾸벅... '졸린다'",
  "비 맞은 사람들, '비 맞지 말걸' 공통 반응",
  "어느 직장인 '월요일이 또 왔다'... 전국적 현상",
];

const BAD: Record<keyof Gauges, string[]> = {
  debt: [
    "국가부채 사상 최고치 경신... '재정 위기 임박' 경고",
    "정부 채무 급증... 전문가 '신용등급 하락 우려'",
    "국채 이자 폭발... 복지예산 삭감 불가피",
    "재정 적자 눈덩이... '미래 세대에 짐 떠넘기나'",
  ],
  inflation: [
    "물가 고공행진 지속... '서민 생계 위협'",
    "인플레이션 통제 불능 수준... 중앙은행 긴급 회의",
    "마트 장바구니 물가 폭등... '장 보기 무섭다'",
    "실질임금 하락 지속... '월급은 그대로, 물가만 올라'",
  ],
  morale: [
    "국민 불만 극에 달해... 곳곳서 시위 발생",
    "정부 신뢰도 사상 최저... '이 정부 못 믿겠다'",
    "민심 이반 가속화... 차기 선거 여권 '암울'",
    "국민 행복지수 최하위권... '이 나라 탈출하고 싶다'",
  ],
  tension: [
    "국제 긴장 고조... 군 대응 태세 강화",
    "외교 충돌 격화... '전쟁 불사' 발언 논란",
    "징병 검사 대상 확대 검토... 젊은 층 불안 확산",
    "국경 근처 군사 훈련 급증... 전문가 '우려스럽다'",
  ],
};

const GOOD: Record<keyof Gauges, string[]> = {
  debt: [
    "국가부채 감소 추세... '재정 건전화 청신호'",
    "정부 흑자 달성... '경제 정책 효과 나타나'",
    "신용등급 상향 검토... '재정 운용 모범적'",
  ],
  inflation: [
    "물가 안정세 진입... '인플레 잡혔다'",
    "소비자 물가 하향 안정... 가계 숨통 트여",
    "인플레 둔화에 중앙은행 '안도'... 금리 인하 시사",
  ],
  morale: [
    "국민 신뢰도 급상승... '이 정부 해낼 수 있겠다'",
    "행복지수 개선... '요즘 살 것 같아'",
    "정부 지지율 반등... 여당 '민심 잡았다'",
  ],
  tension: [
    "국제 긴장 완화... '평화 무드 조성'",
    "외교 협상 타결... 양국 관계 정상화",
    "군사 충돌 가능성 낮아져... 국민 안도",
  ],
};

const NEUTRAL: string[] = [
  "경제 동향 점검... '전반적 양호'",
  "정부, 경제 안정화 정책 추진 중",
  "전문가 '현재 경제 지표 무난한 수준'",
  "경제장관실 '상황 예의주시 중'",
  "주식시장, 보합세 유지",
  "오늘의 환율... 전일 대비 소폭 변동",
];

const INAUGURATION: string[] = [
  "속보: 신임 경제장관 취임식 성황리에 마쳐... 국민들 기대 속 새 출발",
  "경제장관 취임 첫날... '국민과 함께 경제를 살리겠다' 포부 밝혀",
  "새 경제장관 첫 출근... 취재진 인파 몰려 '기대감 고조'",
];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function makeHeadline(g: Gauges, turn: number): string {
  if (turn <= 1) return pick(INAUGURATION);
  if (Math.random() < 0.33) return pick(SILLY);

  const badness: Record<keyof Gauges, number> = {
    debt:      g.debt,
    inflation: g.inflation,
    morale:    100 - g.morale,
    tension:   g.tension,
  };
  const goodness: Record<keyof Gauges, number> = {
    debt:      100 - g.debt,
    inflation: 100 - g.inflation,
    morale:    g.morale,
    tension:   100 - g.tension,
  };

  type Note = { key: keyof Gauges; type: "bad" | "good"; score: number };
  const notes: Note[] = [];
  for (const k of ["debt", "inflation", "morale", "tension"] as (keyof Gauges)[]) {
    if (badness[k]  >= 60) notes.push({ key: k, type: "bad",  score: badness[k]  });
    if (goodness[k] >= 65) notes.push({ key: k, type: "good", score: goodness[k] });
  }

  if (notes.length === 0) return pick(NEUTRAL);

  notes.sort((a, b) => b.score - a.score);
  const chosen = Math.random() < 0.6 ? notes[0] : pick(notes);
  return pick(chosen.type === "bad" ? BAD[chosen.key] : GOOD[chosen.key]);
}

interface NewsTickerProps {
  gauges: Gauges;
  turn: number;
}

const SEP = "          ◆  ◈  ◆          ";

function makeBatch(g: Gauges, turn: number): string {
  // 4개 헤드라인을 구분자로 이어붙여 밀도 확보
  const items = Array.from({ length: 4 }, (_, i) =>
    i === 0 && turn <= 1 ? makeHeadline(g, 1) : makeHeadline(g, turn + 1)
  );
  return items.join(SEP);
}

export function NewsTicker({ gauges, turn }: NewsTickerProps) {
  const [state, setState] = useState(() => {
    const text = makeBatch(gauges, turn);
    return { text, animKey: 0, duration: Math.round(text.length * 0.22) };
  });

  useEffect(() => {
    const text = makeBatch(gauges, turn);
    setState({ text, animKey: Date.now(), duration: Math.round(text.length * 0.22) });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [turn]);

  const onEnd = () => {
    const text = makeBatch(gauges, turn);
    setState({ text, animKey: Date.now(), duration: Math.round(text.length * 0.22) });
  };

  const hasBad  = gauges.debt >= 60 || gauges.inflation >= 60 || gauges.morale <= 40 || gauges.tension >= 60;
  const hasGood = !hasBad && (gauges.debt <= 35 || gauges.inflation <= 35 || gauges.morale >= 65 || gauges.tension <= 35);
  const labelColor = hasBad ? "#ff4444" : hasGood ? "#5599ff" : "#ffaa00";
  const textColor  = hasBad ? "#ff9900" : hasGood ? "#88bbff" : "#39ff14";

  return (
    <div className="shrink-0 flex items-center overflow-hidden font-mono select-none"
         style={{ fontSize: "0.6rem", height: "1.5em",
                  background: "#010801",
                  borderTop: "1px solid #1a4010", borderBottom: "1px solid #1a4010" }}>

      {/* 고정 라벨 */}
      <div className="shrink-0 font-bold px-2"
           style={{ color: labelColor, borderRight: "1px solid #1a4010",
                    whiteSpace: "nowrap", lineHeight: "1.5em" }}>
        ◈ BREAKING NEWS
      </div>

      {/* 스크롤 영역 */}
      <div className="flex-1 overflow-hidden" style={{ height: "1.5em" }}>
        <div
          key={state.animKey}
          onAnimationEnd={onEnd}
          style={{
            display: "inline-block",
            whiteSpace: "nowrap",
            color: textColor,
            paddingLeft: "100%",
            animation: `ticker-scroll ${state.duration}s linear forwards`,
            lineHeight: "1.5em",
          }}
        >
          {state.text}
        </div>
      </div>
    </div>
  );
}
