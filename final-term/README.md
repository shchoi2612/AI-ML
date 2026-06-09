# EconSim — 국가 경제 시뮬레이션 웹게임

20개월 임기의 경제장관이 되어 매 턴 닥치는 위기에 정책 카드로 대응하는 게임.
정책 선택이 4개 게이지(부채·인플레이션·민심·국제긴장)와 5개 섹터 ETF를 움직이고,
임기말 EMH 성적표(정책→섹터 예측가능성 ρ)로 시장 효율성을 채점한다.

- **백엔드**: FastAPI (Python) — 게임 엔진 / ETF / EMH 분석 / LLM 나레이션
- **프론트엔드**: Next.js 14 + TypeScript + Tailwind

> 백엔드(`:8000`)와 프론트엔드(`:3000`)를 **둘 다** 띄워야 동작한다.
> (프론트는 `localhost:8000` API를 호출. 백엔드가 없으면 mock fixture로 폴백된다.)

---

## 1. 백엔드 실행 (`final-term/backend/`)

```bash
cd final-term/backend

# 가상환경 + 의존성
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 서버 기동 (http://localhost:8000)
uvicorn main:app --reload --port 8000
```

### 환경변수 (선택)
LLM 속보 나레이션을 쓰려면 [Groq](https://console.groq.com/) API 키가 필요하다.
없어도 게임은 폴백 문장으로 정상 작동한다.

```bash
cp .env.example .env               # 그리고 .env 안에 GROQ_API_KEY=... 채우기
```

---

## 2. 프론트엔드 실행 (`final-term/frontend/`)

새 터미널에서:

```bash
cd final-term/frontend

npm install
npm run dev                        # http://localhost:3000
```

브라우저로 **http://localhost:3000** 접속 → 게임 시작.

---

## 3. 테스트 / 밸런스 측정 (백엔드)

```bash
cd final-term/backend
source .venv/bin/activate

pytest                             # 단위 테스트
python balance_sim.py              # 봇 자동 플레이로 난이도(완주율) 측정
python emh_check.py                # EMH ρ(정책→섹터 상관) 측정
```

---

## 포트 요약

| 서비스 | 포트 | 디렉터리 |
|---|---|---|
| 백엔드 (FastAPI) | 8000 | `final-term/backend` |
| 프론트엔드 (Next.js) | 3000 | `final-term/frontend` |
