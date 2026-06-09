"""EconSim FastAPI 백엔드 — 계약 v2 (코스트 기반 카드 시스템)."""
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from engine import (
    new_game, check_game_over, select_event, generate_hint,
    apply_cards, compute_capacity, refresh_sector_resources,
    card_affordable, validate_selection, effective_cost,
)
from config import SEVERITY_LABELS
from cards import CARDS, CARDS_BY_ID
from events import EVENTS
from narration import stream_narration
from emh import analyze_emh

app = FastAPI(title="EconSim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인메모리 세션 저장소 (M0: 단일 플레이어)
sessions: dict[str, dict] = {}

GAUGE_OUT = ("debt", "inflation", "morale", "tension")


def _gauges(state: dict) -> dict:
    return {k: state[k] for k in GAUGE_OUT}


def _format_situation(event: dict) -> dict:
    """이벤트를 '상황(situation)' 서사로 변환 (v2: 선택지는 카드풀이 대체)."""
    sev = event.get("severity", "light")
    return {
        "id": event["id"],
        "title": event["title"],
        "desc": event["desc"],
        "severity": sev,                              # light / medium / major
        "severity_label": SEVERITY_LABELS.get(sev, "사건"),
    }


def _format_card(card: dict, capacity: int, resources: dict, state: dict) -> dict:
    """카드를 프론트용 JSON으로 (할인 반영 코스트 + 감당가능 + 정성 힌트 + 대응 할인 표시)."""
    eff_fiscal, eff_sector = effective_cost(card, state)
    base_fiscal = card["fiscal_cost"]
    base_sector = card.get("sector_cost", 0)
    return {
        "id": card["id"],
        "title": card["title"],
        "sector": card.get("sector"),
        "fiscal_cost": eff_fiscal,                    # 이번 턴 실제 코스트(할인 반영)
        "sector_cost": eff_sector,
        "base_fiscal_cost": base_fiscal,              # 원래 코스트(취소선 표시용)
        "base_sector_cost": base_sector,
        "discounted": (eff_fiscal < base_fiscal) or (eff_sector < base_sector),
        "hint": generate_hint(card["base_effects"]),
        "affordable": card_affordable(card, capacity, resources, state),
        "tags": card.get("tags", []),
    }


def _budget_payload(state: dict) -> dict:
    """이번 턴의 예산(재정 여력 + 섹터 자원) + 감당가능/할인 반영 카드풀."""
    capacity = compute_capacity(state)
    state["fiscal_capacity"] = capacity
    resources = state["sector_resources"]
    return {
        "fiscal_capacity": capacity,
        "sector_resources": dict(resources),
        "card_pool": [_format_card(c, capacity, resources, state) for c in CARDS],
    }


class ActionRequest(BaseModel):
    game_id: str
    card_ids: list[str] = []   # 이번 턴에 쓸 카드 id 목록 (빈 리스트 = 패스)


# 백엔드 검증용 최소 페이지 (디자인 X, 석원님 프론트와 별개). 같은 오리진이라 CORS 불필요.
@app.get("/verify")
def verify_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "verify.html"))


@app.post("/game/new")
def create_game():
    game_id = str(uuid.uuid4())
    state = new_game()
    refresh_sector_resources(state)              # 초기 ETF=100 → 적립 0
    situation = select_event(state, EVENTS)
    sessions[game_id] = {"state": state, "current_situation": situation}
    return {
        "game_id": game_id,
        "turn": state["turn"],
        "gauges": _gauges(state),
        "etf_prices": state["etf_prices"],
        "situation": _format_situation(situation),
        **_budget_payload(state),
    }


@app.post("/game/action")
def take_action(req: ActionRequest):
    session = sessions.get(req.game_id)
    if not session:
        raise HTTPException(404, "game not found")

    state = session["state"]
    situation = session["current_situation"]

    # 카드 id 검증
    selected = []
    for cid in req.card_ids:
        card = CARDS_BY_ID.get(cid)
        if card is None:
            raise HTTPException(400, f"unknown card_id: {cid}")
        selected.append(card)

    # 코스트(여력/섹터자원) 검증
    err = validate_selection(state, selected)
    if err:
        raise HTTPException(400, f"카드 선택 불가: {err}")

    action_turn = state["turn"]
    gauge_deltas = apply_cards(state, selected)

    game_over = check_game_over(state)
    next_situation = None
    if not game_over:
        refresh_sector_resources(state)          # 이번 턴 ETF 성과로 다음 턴 자원 적립
        next_situation = select_event(state, EVENTS)
        session["current_situation"] = next_situation

    card_labels = " + ".join(c["title"] for c in selected) or "정책 보류(패스)"
    session.setdefault("turn_ctx", {})[action_turn] = {
        "event_title": situation["title"],
        "choice_label": card_labels,
        "gauge_deltas": gauge_deltas,
        "current_state": _gauges(state),
        "etf_changes": state["etf_history"][-1] if state["etf_history"] else {},
    }

    resp = {
        "turn": state["turn"],
        "gauges": _gauges(state),
        "gauge_deltas": gauge_deltas,
        "etf_prices": state["etf_prices"],
        "etf_changes": state["etf_history"][-1] if state["etf_history"] else {},
        "next_situation": _format_situation(next_situation) if next_situation else None,
        "game_over": game_over,
    }
    if not game_over:
        resp.update(_budget_payload(state))      # 다음 턴 예산 + 카드풀
    return resp


# ── Phase 2: 나레이션 (2-phase 분리) ──

VN_SPEAKERS = [
    {"speaker": "재무장관", "portrait": "finance_minister"},
    {"speaker": "중앙은행 총재", "portrait": "central_bank_governor"},
    {"speaker": "야당 대표", "portrait": "opposition_leader"},
]


@app.get("/game/{game_id}/turn/{turn}/narration")
def get_narration(game_id: str, turn: int):
    session = sessions.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")

    ctx = session.get("turn_ctx", {}).get(turn)
    if not ctx:
        raise HTTPException(404, "turn not found")

    import json, random

    speaker = random.choice(VN_SPEAKERS)

    def sse_generator():
        narration_text = ""
        for chunk in stream_narration(
            event_title=ctx["event_title"],
            choice_label=ctx["choice_label"],
            gauge_deltas=ctx["gauge_deltas"],
            current_state=ctx["current_state"],
            etf_changes=ctx["etf_changes"],
            turn=turn,
        ):
            narration_text += chunk
            yield f"data: {json.dumps({'type': 'narration_chunk', 'text': chunk}, ensure_ascii=False)}\n\n"

        vn = {
            "type": "vn_dialogue",
            "speaker": speaker["speaker"],
            "portrait": speaker["portrait"],
            "text": narration_text[:100] + "..." if len(narration_text) > 100 else narration_text,
        }
        yield f"data: {json.dumps(vn, ensure_ascii=False)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ── Phase 3: EMH 성적표 ──

@app.get("/game/{game_id}/emh-summary")
def get_emh_summary(game_id: str):
    session = sessions.get(game_id)
    if not session:
        raise HTTPException(404, "game not found")
    return analyze_emh(session["state"])
