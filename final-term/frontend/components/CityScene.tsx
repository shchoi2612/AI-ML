"use client";
import { useState, useEffect, useRef } from "react";
import type { Gauges } from "@/lib/api";

// ── 댓글 ─────────────────────────────────────────────────────
type Tone = "silly" | "neutral" | "angry" | "happy";

const ANGRY_POOL: Record<keyof Gauges, string[]> = {
  debt: [
    `나라 빚이 ${"{debt}"}%래... 내 세금은 어디로?`,
    "이 부채 누가 갚냐? 다음 세대한테 떠넘기는 거잖아",
    "재정 적자가 심각하대. 이래도 되는 거야?",
    "빚으로 나라 운영하는 거야? 진짜 답답하다",
    "국가부채 이 속도면 파산 아냐?",
  ],
  inflation: [
    `인플레 ${"{inflation}"}%... 장 보기 무섭다`,
    "월급은 그대로인데 물가만 미쳐 날뛰네",
    "라면 한 봉지가 이 가격이라고? 진짜",
    "인플레 때문에 저축이 다 녹아버려. 미치겠다",
    "물가 폭등에 서민만 죽어나가고",
  ],
  morale: [
    `민심이 ${"{morale}"}%야... 다들 지쳤나봐`,
    "주변 사람들 다 힘들다고 하더라고",
    "불만이 폭발 직전이야. 언제 터지려나",
    "이 나라 탈출하고 싶다는 친구가 늘었어",
    "이 정부 진짜 믿을 수가 없어",
  ],
  tension: [
    `긴장도 ${"{tension}"}%... 징집되는 거 아니야?`,
    "전쟁 나는 거 아냐? 요즘 너무 불안해",
    "국제 정세가 심상치 않대. 무섭다",
    "이러다 진짜 큰일 나는 거 아닌지 걱정돼",
    "군사 충돌 뉴스가 왜 이렇게 많아?",
  ],
};

const HAPPY_POOL: Record<keyof Gauges, string[]> = {
  debt: [
    `나라 부채가 ${"{debt}"}%까지 줄었대! 재정 건전해지나?`,
    "요즘 정부가 재정관리를 잘하는 것 같아",
    "나라 빚 갚아나가는 거 보니까 믿음직하네",
    "재정이 이 정도면 꽤 건실한 거 아닌가?",
  ],
  inflation: [
    `물가 ${"{inflation}"}%로 안정됐대. 장보기 편해졌어`,
    "인플레가 잡히니까 월급 가치가 올라간 느낌",
    "요즘 물가가 얌전하네. 오랜만에 여유가 생겼어",
    "물가 안정되니까 저축할 맛이 나네",
  ],
  morale: [
    `민심이 ${"{morale}"}%래. 요즘 분위기 좋은 것 같더라`,
    "주변 사람들 표정이 밝아진 것 같아",
    "나라가 안정되니 사람들도 웃음이 늘었네",
    "친구가 취직했대! 경기가 풀리나봐",
  ],
  tension: [
    `긴장도 ${"{tension}"}%래. 요즘 평화롭다 진짜`,
    "전쟁 걱정 없이 살 수 있는 게 얼마나 좋은지",
    "국제 관계가 안정되니까 마음이 편하네",
    "요즘 뉴스 보면 세상이 평화로운 것 같아",
  ],
};

const NEUTRAL_POOL: string[] = [
  "그래서 내 월급은 언제 오르냐고",
  "이 나라에서 집 살 수 있기나 한 건지",
  "주식이 요즘 애매하네. 사야 하나 말아야 하나",
  "ETF 지금 사도 되냐? 친구가 사라는데",
  "그래서 결국 세금은 또 오르는 건가?",
  "또 무슨 정책 바꿨다는 거야",
  "경제장관은 뭐 하는 사람이래?",
  "뭐, 그냥저냥 사는 거지 뭐",
];

const SILLY_POOL: string[] = [
  "오늘 점심 뭐 먹지?","버스 언제 오냐...","아 배고프다","지갑 두고 왔나?",
  "커피 한 잔 마시고 싶다","퇴근하고 싶다","저녁은 치킨이지","....","흠","날씨 좋다",
];

function fillTemplate(s: string, g: Gauges): string {
  return s
    .replace("{debt}",      String(g.debt))
    .replace("{inflation}", String(g.inflation))
    .replace("{morale}",    String(g.morale))
    .replace("{tension}",   String(g.tension));
}

function makeComment(g: Gauges): { text: string; tone: Tone } {
  if (Math.random() < 0.2) {
    return { text: SILLY_POOL[Math.floor(Math.random() * SILLY_POOL.length)], tone: "silly" };
  }

  // 각 게이지의 "나쁜 정도" / "좋은 정도" (0~100)
  const badness:  Record<keyof Gauges, number> = {
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

  type Note = { key: keyof Gauges; tone: "angry" | "happy"; urgency: number };
  const notes: Note[] = [];
  for (const k of ["debt","inflation","morale","tension"] as (keyof Gauges)[]) {
    if (badness[k]  >= 60) notes.push({ key: k, tone: "angry", urgency: badness[k]  });
    if (goodness[k] >= 65) notes.push({ key: k, tone: "happy", urgency: goodness[k] });
  }

  if (notes.length === 0) {
    return { text: NEUTRAL_POOL[Math.floor(Math.random() * NEUTRAL_POOL.length)], tone: "neutral" };
  }

  notes.sort((a, b) => b.urgency - a.urgency);
  // 60% 확률로 가장 심각한 게이지 선택, 40%는 랜덤
  const chosen = Math.random() < 0.6
    ? notes[0]
    : notes[Math.floor(Math.random() * notes.length)];

  const pool = chosen.tone === "angry" ? ANGRY_POOL[chosen.key] : HAPPY_POOL[chosen.key];
  const raw  = pool[Math.floor(Math.random() * pool.length)];
  return { text: fillTemplate(raw, g), tone: chosen.tone };
}

function bubbleColor(tone: Tone): string {
  if (tone === "silly")  return "#ffffff";
  if (tone === "angry")  return "#ff4444";
  if (tone === "happy")  return "#5599ff";
  return "#39ff14";
}

// ── 스틱맨 ──────────────────────────────────────────────────
interface Stickman {
  id: number; x: number; dir: 1 | -1; frame: 0 | 1; speed: number;
  stopped: boolean; bubble: { text: string; tone: Tone } | null; ticks: number;
}

function getFrameLines(m: Stickman): [string, string, string] {
  if (m.stopped) return [" o ", "/|\\", "/ \\"];
  if (m.dir === 1) return m.frame === 0 ? [" o>","/| ","/ \\"] : [" o>","/| "," /\\"];
  return m.frame === 0 ? ["<o "," |\\","/ \\"] : ["<o "," |\\","/\\ "];
}

const INIT_MEN: Stickman[] = [
  { id:0, x: 3, dir: 1,frame:0,speed:0.22,stopped:false,bubble:null,ticks:0},
  { id:1, x:12, dir:-1,frame:1,speed:0.16,stopped:false,bubble:null,ticks:0},
  { id:2, x:21, dir: 1,frame:0,speed:0.28,stopped:false,bubble:null,ticks:0},
  { id:3, x:33, dir:-1,frame:1,speed:0.19,stopped:false,bubble:null,ticks:0},
  { id:4, x:44, dir: 1,frame:0,speed:0.24,stopped:false,bubble:null,ticks:0},
  { id:5, x:55, dir:-1,frame:1,speed:0.13,stopped:false,bubble:null,ticks:0},
  { id:6, x:64, dir: 1,frame:0,speed:0.30,stopped:false,bubble:null,ticks:0},
  { id:7, x:74, dir:-1,frame:1,speed:0.17,stopped:false,bubble:null,ticks:0},
  { id:8, x:83, dir: 1,frame:0,speed:0.21,stopped:false,bubble:null,ticks:0},
  { id:9, x:91, dir:-1,frame:1,speed:0.25,stopped:false,bubble:null,ticks:0},
];

// ── 차량 ──────────────────────────────────────────────────────
interface Vehicle { id:number; x:number; dir:1|-1; speed:number; lane:0|1; }

const INIT_VEHICLES: Vehicle[] = [
  {id:0,x: 10,dir: 1,speed:1.0,lane:0},{id:1,x:44,dir: 1,speed:0.8,lane:0},
  {id:2,x: 76,dir: 1,speed:1.2,lane:0},{id:3,x:22,dir:-1,speed:0.9,lane:1},
  {id:4,x: 58,dir:-1,speed:1.1,lane:1},{id:5,x:87,dir:-1,speed:0.7,lane:1},
];

// 차: 3줄, 탱크: 4줄 (사용자 제공 아트)
function getVehicleLines(dir: 1|-1, isTank: boolean): string[] {
  if (isTank) return dir === 1
    ? ["     _____",
       "  ___/  ___ |=========>",
       " |___________\\",
       " (O)(O)(O)(O)(O)"]
    : ["     _____",
       "<=========| ___\\___  ",
       "/___________| ",
       "(O)(O)(O)(O)(O)"];
  return dir === 1
    ? ["      ______   ","   __/  __  \\__ ","  |__o______o__|"]
    : ["      ______   ","   __\\  __  /__ ","  |__o______o__|"];
}

// ── 전투기 ──────────────────────────────────────────────────
interface Jet { id:number; x:number; dir:1|-1; speed:number; alt:0|1|2; }
const INIT_JETS: Jet[] = [
  {id:0,x: 15,dir: 1,speed:3.5,alt:1},
  {id:1,x: 70,dir:-1,speed:2.8,alt:0},
  {id:2,x: 40,dir: 1,speed:4.2,alt:2},
];
// 사용자 제공 3줄 전투기 아트
function getJetLines(dir:1|-1): [string,string,string] {
  if (dir === -1) return ["    /", "<==//==//", "   /"];
  // 우향: 좌향 미러 (/ → \)
  return ["\\    ", "\\\\==\\\\==>", " \\   "];
}

// ── 하늘 ─────────────────────────────────────────────────────
function getSkyLines(tension:number): {lines:string[];color:string} {
  if (tension>=80) return {color:"#4a1010",lines:[
    "                                                                              ",
    "  ████████████                    ████████████████              ████████     ",
    "         █████████████████████               ████████████████████            ",
    "                                                                              ",
  ]};
  if (tension>=50) return {color:"#2a3010",lines:[
    "                                                                              ",
    "  (░░░░░░)              (░░░░)                       (░░░░░░)                ",
    "             (░░░░░)                  (░░░)                                  ",
    "                                                                              ",
  ]};
  return {color:"#0d3010",lines:[
    "                                                                              ",
    "  (~~~~)               (~)                         (~~~~~)                   ",
    "           (~~~)                    (~~~~)                                   ",
    "                                                                              ",
  ]};
}

// ── 건물 ─────────────────────────────────────────────────────
const BUILDINGS = `\
                                              /\\
            _____          ___________       /  \\           __________
   ______  |     |   ___  |           |  ___/ /\\ \\___  ____|          |
  |      | | [_] |  |   | |  [_] [_]  | |   |    |  ||    |  [_] [_]  |
  | [_]  | | [_] |  |[_]| |  [_] [_]  | |   | /\\ |  ||    |  [_] [_]  |
  | [_]  | | [_] |  |[_]| |  [_] [_]  | |   |/  \\|  ||    |  [_] [_]  |
  | [_]  | | [_] |  |[_]| |  [_] [_]  |_|   |    |  ||    |  [_] [_]  |
  | [_]  |_| [_] |__|[_]|_|  [_] [_]  |_|[_]|    |[_]|[_] |  [_] [_]  |
  | [_]  |_| [_] |__|[_]|_|___________|_|[_]|____|[_]|[_]_|__________|_|
  |______|_|_____|__|___|_|___________|_|___|____|___|______|__________|_|`;

// ── 도로 (15줄 2차선 대로) ────────────────────────────────────
// 각 차선: 5줄 (margin 1 + car 3 + margin 1), 중앙·상하 경계: ═══
// 인도: 2줄
//
// [0] ░░░ 인도 1  ← 스틱맨 몸
// [1] ░░░ 인도 2  ← 스틱맨 발   (road-container top = 0)
// [2] ═══ 상단 경계
// [3]     상행 차/탱크 line0  top=3×1.3em=3.9em ←┐ UPPER_CAR_TOP
// [4]     상행 차/탱크 line1                      │ 탱크=4줄, 차=3줄
// [5]     상행 차/탱크 line2                      │
// [6]     상행 탱크 line3(탱크만)                ←┘
// [7]     상행 여백 아래
// [8] ═══ 중앙선
// [9]     하행 차/탱크 line0  top=9×1.3em=11.7em ←┐ LOWER_CAR_TOP
//[10]     하행 차/탱크 line1                      │
//[11]     하행 차/탱크 line2                      │
//[12]     하행 탱크 line3(탱크만)                ←┘
//[13]     하행 여백 아래
//[14] ═══ 하단 경계
const DASH = ("   -   ").repeat(50);
const ROAD = [
  {text:"░".repeat(350), color:"#0d3d08"},  // [0] 인도 1
  {text:"░".repeat(350), color:"#0d3d08"},  // [1] 인도 2
  {text:"═".repeat(350), color:"#39ff14"},  // [2] 상단 경계
  {text:" ".repeat(350), color:"#060e04"},  // [3]
  {text:" ".repeat(350), color:"#060e04"},  // [4]
  {text:DASH,            color:"#132b08"},  // [5] 상행 차선 중앙선 (매우 어둡게)
  {text:" ".repeat(350), color:"#060e04"},  // [6]
  {text:" ".repeat(350), color:"#060e04"},  // [7]
  {text:"═".repeat(350), color:"#39ff14"},  // [8] 중앙선
  {text:" ".repeat(350), color:"#060e04"},  // [9]
  {text:" ".repeat(350), color:"#060e04"},  // [10]
  {text:DASH,            color:"#132b08"},  // [11] 하행 차선 중앙선
  {text:" ".repeat(350), color:"#060e04"},  // [12]
  {text:" ".repeat(350), color:"#060e04"},  // [13]
  {text:"═".repeat(350), color:"#39ff14"},  // [14] 하단 경계
] as const;

// ── 포지션 상수 (road-container 기준 top) ─────────────────────
// 1em = 0.6rem (CityScene fontSize), lineHeight 1.3 → 1 line = 1.3em
//
// 스틱맨: 3줄짜리. 발(line3) 이 road[1] 구역(1.3~2.6em)에 오려면
//   div top = road[1] bottom - div height = 2.6em - 3.9em = -1.3em
// 상행차 (lane=0): top = road[4] start = 4×1.3em = 5.2em
// 하행차 (lane=1): top = road[10] start = 10×1.3em = 13.0em
// 전투기: CityScene 기준 top, 하늘(0~5.2em) 구역
const STICKMAN_TOP  = "-1.3em";
// 탱크(4줄)도 차(3줄)도 road[3] 시작에 놓음. 탱크=lane 꽉참, 차=아래 1줄 여유
const UPPER_CAR_TOP = "3.9em";   // road[3] top (4×1.3em 아님, 3×1.3=3.9em)
const LOWER_CAR_TOP = "11.7em";  // road[9] top = 9×1.3em
// 전투기: 3줄짜리(3.9em), 하늘(4줄=5.2em) 안에 위치
const JET_TOPS: ["0.0em","0.65em","1.3em"] = ["0.0em","0.65em","1.3em"];

// ── 컴포넌트 ─────────────────────────────────────────────────
interface CitySceneProps { gauges: Gauges; }
const FS = "0.6rem";
const LH = 1.3;

export function CityScene({ gauges }: CitySceneProps) {
  const [men,      setMen]      = useState<Stickman[]>(INIT_MEN);
  const [vehicles, setVehicles] = useState<Vehicle[]>(INIT_VEHICLES);
  const [jets,     setJets]     = useState<Jet[]>(INIT_JETS);
  const gRef = useRef(gauges);
  gRef.current = gauges;

  useEffect(() => {
    const id = setInterval(() => {
      setMen(prev => prev.map(m => {
        if (m.stopped) {
          const t = m.ticks - 1;
          if (t <= 0) return {...m, stopped:false, bubble:null, ticks:0};
          return {...m, ticks:t};
        }
        let x = m.x + m.dir * m.speed, dir = m.dir;
        if (x >= 93) { x = 93; dir = -1; }
        if (x <=  1) { x =  1; dir =  1; }
        return {...m, x, dir, frame:(m.frame===0?1:0) as 0|1};
      }));
      setVehicles(prev => prev.map(v => {
        let x = v.x + v.dir * v.speed;
        if (x > 112) x = -12;
        if (x < -12) x = 112;
        return {...v, x};
      }));
      setJets(prev => prev.map(j => {
        let x = j.x + j.dir * j.speed;
        if (x > 115) x = -15;
        if (x < -15) x = 115;
        return {...j, x};
      }));
    }, 150);
    return () => clearInterval(id);
  }, []);

  const handleClick = (id: number) => {
    setMen(prev => prev.map(m => {
      if (m.id !== id) return m.stopped ? {...m,stopped:false,bubble:null,ticks:0} : m;
      if (m.stopped) return m;
      return {...m, stopped:true, bubble:makeComment(gRef.current), ticks:28};
    }));
  };

  const tension   = gauges.tension;
  const isTense   = tension >= 60;
  const jetCount  = tension>=85?3:tension>=75?2:tension>=65?1:0;
  const sky       = getSkyLines(tension);
  const bldgColor = tension>=80 ? "#5a2a10" : "#1a7a0a";

  return (
    <div className="relative font-mono overflow-hidden select-none"
         style={{ fontSize: FS, lineHeight: LH }}>

      {/* 하늘 */}
      <pre className="m-0 p-0 overflow-hidden" style={{ color: sky.color }}>
        {sky.lines.join("\n")}
      </pre>

      {/* 건물 */}
      <pre className="m-0 p-0 overflow-hidden" style={{ color: bldgColor }}>
        {BUILDINGS}
      </pre>

      {/* 전투기 — CityScene 기준 절대 위치 (하늘 영역), 3줄 아트 */}
      {jets.slice(0, jetCount).map(j => {
        const [jl0,jl1,jl2] = getJetLines(j.dir);
        return (
          <div key={j.id} className="absolute"
               style={{ left:`${j.x}%`, top:JET_TOPS[j.alt],
                        transform:"translateX(-50%)", zIndex:6 }}>
            <pre className="m-0 p-0"
                 style={{ color:"#ff6666", fontSize:FS, lineHeight:LH }}>
              {jl0}{"\n"}{jl1}{"\n"}{jl2}
            </pre>
          </div>
        );
      })}

      {/* ── 도로 컨테이너 ── 스틱맨·차량 위치 기준점 */}
      <div className="relative" style={{ overflow:"visible" }}>

        {/* 도로 줄 (overflow hidden으로 가로 클리핑) */}
        {ROAD.map((row, i) => (
          <div key={i} className="overflow-hidden"
               style={{ color:row.color, lineHeight:LH, whiteSpace:"nowrap" }}>
            {row.text}
          </div>
        ))}

        {/* 차량 / 탱크 */}
        {vehicles.map(v => {
          const lines = getVehicleLines(v.dir, isTense);
          const color = isTense ? "#ffaa00" : "#39ff14";
          const top   = v.lane===0 ? UPPER_CAR_TOP : LOWER_CAR_TOP;
          return (
            <div key={v.id} className="absolute"
                 style={{ left:`${v.x}%`, top, transform:"translateX(-50%)", zIndex:3 }}>
              <pre className="m-0 p-0"
                   style={{ color, fontSize:FS, lineHeight:LH }}>
                {lines.join("\n")}
              </pre>
            </div>
          );
        })}

        {/* 스틱맨 — top:-1.3em → 발이 road[1](인도) 위에 정확히 올라감 */}
        {men.map(m => {
          const [l0,l1,l2] = getFrameLines(m);
          const bc = m.bubble ? bubbleColor(m.bubble.tone) : "#39ff14";
          return (
            <div key={m.id} onClick={() => handleClick(m.id)}
                 className="absolute cursor-pointer"
                 style={{ left:`${m.x}%`, top:STICKMAN_TOP,
                          transform:"translateX(-50%)", zIndex:5 }}>
              {m.bubble && (
                <div className="absolute"
                     style={{ bottom:"calc(100% + 3px)", left:"50%",
                              transform:"translateX(-50%)", whiteSpace:"nowrap",
                              zIndex:7, pointerEvents:"none" }}>
                  <div style={{ border:`1px solid ${bc}`, color:bc, background:"#000",
                                padding:"2px 6px", fontSize:"0.55rem", lineHeight:1.5,
                                fontFamily:"var(--font-geist-mono)" }}>
                    {m.bubble.text}
                  </div>
                  <div style={{ textAlign:"center", color:bc,
                                fontSize:"0.5rem", lineHeight:1 }}>▼</div>
                </div>
              )}
              <pre className="m-0 p-0"
                   style={{ color:m.stopped?"#ffaa00":"#39ff14",
                            fontSize:FS, lineHeight:LH }}>
                {l0}{"\n"}{l1}{"\n"}{l2}
              </pre>
            </div>
          );
        })}
      </div>
    </div>
  );
}
