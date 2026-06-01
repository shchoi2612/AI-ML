"""API 계약 회귀 테스트 — TRD 계약 ↔ 실제 구현 드리프트 방지.

이 테스트가 green이면 프론트(석원님)가 보는 계약이 실제 백엔드와 일치함이 보장된다.
foundation 점검(2026-06-01)에서 발견한 계약 드리프트 4건의 재발 방지가 목적.

실행: backend/ 에서  .venv/bin/python -m pytest tests/ -q
나레이션 테스트는 Groq를 monkeypatch해 네트워크 없이 SSE 프레이밍만 검증.
"""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402

client = TestClient(main.app)


def _new_game():
    return client.post("/game/new").json()


def test_new_game_contract():
    """POST /game/new 응답 형태가 TRD 계약과 일치."""
    j = _new_game()
    assert set(j.keys()) == {"game_id", "turn", "gauges", "etf_prices", "event"}
    assert j["turn"] == 1
    assert set(j["gauges"]) == {"debt", "inflation", "morale", "tension"}
    assert set(j["etf_prices"]) == {
        "semiconductor", "energy", "finance", "defense", "consumer"
    }
    ev = j["event"]
    assert set(ev.keys()) == {"id", "title", "desc", "choices"}
    assert len(ev["choices"]) == 3
    for c in ev["choices"]:
        assert set(c.keys()) == {"label", "hint"}  # 정확한 숫자 노출 금지


def test_action_flow_contract():
    """POST /game/action 응답 형태 + 턴 진행."""
    gid = _new_game()["game_id"]
    r = client.post("/game/action", json={"game_id": gid, "choice_index": 0})
    assert r.status_code == 200
    j = r.json()
    assert set(j.keys()) == {
        "turn", "gauges", "gauge_deltas", "etf_prices",
        "etf_changes", "next_event", "game_over",
    }
    assert j["turn"] == 2
    assert j["game_over"] is None
    assert set(j["etf_changes"]) == set(j["etf_prices"])


def test_invalid_choice_index():
    gid = _new_game()["game_id"]
    r = client.post("/game/action", json={"game_id": gid, "choice_index": 99})
    assert r.status_code == 400


def test_emh_summary_contract():
    """GET /emh-summary 응답이 실제 emh.py 반환 형태와 일치 (드리프트 #2 방지)."""
    gid = _new_game()["game_id"]
    for _ in range(4):
        client.post("/game/action", json={"game_id": gid, "choice_index": 0})
    j = client.get(f"/game/{gid}/emh-summary").json()
    assert set(j.keys()) == {
        "total_turns", "stability_score", "predictability_score",
        "top_correlation", "avg_etf_volatility", "summary_text",
    }
    assert set(j["top_correlation"].keys()) == {"sector", "value"}
    assert 0.0 <= j["predictability_score"] <= 1.0
    assert isinstance(j["summary_text"], str) and j["summary_text"]


def test_narration_sse_contract(monkeypatch):
    """GET /turn/{turn}/narration SSE 이벤트 순서 (드리프트 #1 방지).

    Groq를 monkeypatch해 네트워크 없이 narration_chunk×N → vn_dialogue → done 검증.
    """
    monkeypatch.setattr(main, "stream_narration",
                        lambda **kw: iter(["테스트 ", "뉴스 ", "브리핑"]))
    gid = _new_game()["game_id"]
    client.post("/game/action", json={"game_id": gid, "choice_index": 0})

    types, text, vn = [], "", None
    with client.stream("GET", f"/game/{gid}/turn/1/narration") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            types.append(ev["type"])
            if ev["type"] == "narration_chunk":
                text += ev["text"]
            elif ev["type"] == "vn_dialogue":
                vn = ev
            if ev["type"] == "done":
                break

    assert types[0] == "narration_chunk"
    assert types[-1] == "done"
    assert "vn_dialogue" in types
    assert text == "테스트 뉴스 브리핑"
    assert set(vn.keys()) == {"type", "speaker", "portrait", "text"}


def test_404_paths():
    assert client.post(
        "/game/action", json={"game_id": "nope", "choice_index": 0}
    ).status_code == 404
    assert client.get("/game/nope/emh-summary").status_code == 404
    assert client.get("/game/nope/turn/1/narration").status_code == 404


@pytest.mark.parametrize("fname,endpoint", [
    ("new_game.json", None),
])
def test_fixtures_match_live(fname, endpoint):
    """fixtures(프론트 mock 계약)의 키가 실제 /game/new 응답과 일치."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "fixtures", fname), encoding="utf-8") as f:
        fixture = json.load(f)
    live = _new_game()
    assert set(fixture.keys()) == set(live.keys())
    assert set(fixture["event"].keys()) == set(live["event"].keys())
    assert set(fixture["event"]["choices"][0].keys()) == \
        set(live["event"]["choices"][0].keys())
