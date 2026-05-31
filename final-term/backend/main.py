"""EconSim FastAPI 백엔드."""
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from engine import new_game, apply_choice, check_game_over, select_event, generate_hint
from events import EVENTS
from narration import stream_narration

app = FastAPI(title="EconSim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인메모리 세션 저장소 (M0: 단일 플레이어)
sessions: dict[str, dict] = {}


def _format_event(event: dict) -> dict:
    """이벤트를 프론트용 JSON으로 변환."""
    return {
        "id": event["id"],
        "title": event["title"],
        "desc": event["desc"],
        "choices": [
            {"label": c["label"], "hint": generate_hint(c["base_effects"])}
            for c in event["choices"]
        ],
    }


class ActionRequest(BaseModel):
    game_id: str
    choice_index: int


@app.post("/game/new")
def create_game():
    game_id = str(uuid.uuid4())
    state = new_game()
    event = select_event(state, EVENTS)
    sessions[game_id] = {"state": state, "current_event": event}
    return {
        "game_id": game_id,
        "turn": state["turn"],
        "gauges": {k: state[k] for k in ("debt", "inflation", "morale", "tension")},
        "etf_prices": state["etf_prices"],
        "event": _format_event(event),
    }


@app.post("/game/action")
def take_action(req: ActionRequest):
    session = sessions.get(req.game_id)
    if not session:
        raise HTTPException(404, "game not found")

    state = session["state"]
    event = session["current_event"]

    if req.choice_index < 0 or req.choice_index >= len(event["choices"]):
        raise HTTPException(400, "invalid choice_index")

    choice = event["choices"][req.choice_index]
    action_turn = state["turn"]
    gauge_deltas = apply_choice(state, choice, choice["label"])

    game_over = check_game_over(state)
    next_event = None
    if not game_over:
        next_event = select_event(state, EVENTS)
        session["current_event"] = next_event

    # 나레이션용 컨텍스트 저장 (턴별)
    session.setdefault("turn_ctx", {})[action_turn] = {
        "event_title": event["title"],
        "choice_label": choice["label"],
        "gauge_deltas": gauge_deltas,
        "current_state": {k: state[k] for k in ("debt", "inflation", "morale", "tension")},
        "etf_changes": state["etf_history"][-1] if state["etf_history"] else {},
    }

    return {
        "turn": state["turn"],
        "gauges": {k: state[k] for k in ("debt", "inflation", "morale", "tension")},
        "gauge_deltas": gauge_deltas,
        "etf_prices": state["etf_prices"],
        "etf_changes": state["etf_history"][-1] if state["etf_history"] else {},
        "next_event": _format_event(next_event) if next_event else None,
        "game_over": game_over,
    }


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
