"""EconSim FastAPI 백엔드 — M0: one-click loop만 지원."""
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import new_game, apply_choice, check_game_over, select_event, generate_hint
from events import EVENTS

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
    gauge_deltas = apply_choice(state, choice, choice["label"])

    game_over = check_game_over(state)
    next_event = None
    if not game_over:
        next_event = select_event(state, EVENTS)
        session["current_event"] = next_event

    return {
        "turn": state["turn"],
        "gauges": {k: state[k] for k in ("debt", "inflation", "morale", "tension")},
        "gauge_deltas": gauge_deltas,
        "etf_prices": state["etf_prices"],
        "etf_changes": state["etf_history"][-1] if state["etf_history"] else {},
        "next_event": _format_event(next_event) if next_event else None,
        "game_over": game_over,
    }
