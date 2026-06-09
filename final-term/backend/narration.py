"""LLM 경제 뉴스 나레이션 — Groq API (스트리밍)."""
import os
from config import GAUGE_NAMES, ETF_NAMES, GROQ_MODEL, NARRATION_MAX_TOKENS, NARRATION_TEMPERATURE

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY")
            if key:
                _client = Groq(api_key=key)
        except Exception:
            pass
    return _client

SYSTEM_PROMPT = (
    "당신은 대한민국 경제 뉴스 앵커입니다. "
    "주어진 경제 지표 변화를 바탕으로 2~3문장의 뉴스 브리핑을 작성하세요. "
    "전문적이되 일반 시청자가 이해할 수 있는 어조로 작성하세요. "
    "숫자는 이미 계산되어 있으므로 그대로 인용하세요. 새로운 숫자를 만들지 마세요."
)


def _build_user_prompt(
    event_title: str,
    choice_label: str,
    gauge_deltas: dict,
    current_state: dict,
    etf_changes: dict,
    turn: int,
) -> str:
    gauge_lines = "\n".join(
        f"- {GAUGE_NAMES[k]}: {gauge_deltas.get(k, 0):+d} → 현재 {current_state[k]}"
        for k in GAUGE_NAMES
    )
    etf_lines = "\n".join(
        f"- {ETF_NAMES[k]}: {etf_changes.get(k, 0):+.1f}%"
        for k in ETF_NAMES
    )
    return (
        f"[임기 {turn}개월차/20개월 경제 뉴스 브리핑]\n"
        f"상황: {event_title}\n"
        f"정부 결정: {choice_label}\n\n"
        f"지표 변화:\n{gauge_lines}\n\n"
        f"ETF 동향:\n{etf_lines}\n\n"
        "위 데이터를 바탕으로 뉴스 브리핑을 작성하세요."
    )


def stream_narration(
    event_title: str,
    choice_label: str,
    gauge_deltas: dict,
    current_state: dict,
    etf_changes: dict,
    turn: int,
):
    """스트리밍 나레이션 제너레이터. 청크 단위로 텍스트를 yield한다."""
    client = _get_client()
    if client is None:
        yield _fallback(choice_label)
        return

    user_prompt = _build_user_prompt(
        event_title, choice_label, gauge_deltas, current_state, etf_changes, turn,
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=NARRATION_TEMPERATURE,
            max_tokens=NARRATION_MAX_TOKENS,
            stream=True,
        )
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                yield text
    except Exception:
        yield _fallback(choice_label)


def _fallback(choice_label: str) -> str:
    return f"[{choice_label}] 정책이 시행되었습니다. 경제 지표가 변동하고 있습니다."
