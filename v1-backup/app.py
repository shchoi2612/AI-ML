"""국가 경제 시뮬레이션 게임 — 대시보드 UI."""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from engine import new_game, apply_choice, check_game_over, select_event, generate_hint
from events import EVENTS
from narration import stream_narration
from config import GAUGE_NAMES, GAUGE_KEYS, GAUGE_DANGER, ETF_NAMES, ETF_KEYS, MAX_TURNS

st.set_page_config(page_title="국가 경제 시뮬레이션", page_icon="🏛️", layout="wide")

# ── 세션 초기화 ──
if "game" not in st.session_state:
    st.session_state.game = new_game()
    st.session_state.phase = "event"       # "event" | "narration"
    st.session_state.narration_data = None  # 나레이션용 데이터
    st.session_state.current_event = None
    st.session_state.game_over = None

game = st.session_state.game


# ── 헬퍼 함수 ──
def gauge_color(key: str, value: int) -> str:
    """게이지 값에 따라 색상을 반환한다."""
    if GAUGE_DANGER[key] == "high":
        if value < 60:
            return "#28a745"
        return "#ffc107" if value < 80 else "#dc3545"
    else:  # morale: low is bad
        if value > 40:
            return "#28a745"
        return "#ffc107" if value > 20 else "#dc3545"


def render_gauge(col, key: str, value: int):
    """커스텀 색상 게이지를 렌더링한다."""
    color = gauge_color(key, value)
    with col:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<p style='margin-bottom:4px;font-size:14px;color:#aaa'>{GAUGE_NAMES[key]}</p>"
            f"<div style='background:#333;border-radius:10px;height:18px;width:100%;overflow:hidden'>"
            f"<div style='background:{color};width:{value}%;height:100%;border-radius:10px;"
            f"transition:width 0.5s ease'></div></div>"
            f"<p style='font-size:28px;font-weight:bold;margin:4px 0;color:{color}'>{value}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── 사이드바 ──
with st.sidebar:
    st.markdown("### 📊 ETF 현황")
    etf_prices = game["etf_prices"]
    etf_hist = game["etf_history"]
    for etf_key in ETF_KEYS:
        delta_str = None
        if etf_hist:
            last_pct = etf_hist[-1].get(etf_key, 0)
            delta_str = f"{last_pct:+.1f}%"
        st.metric(ETF_NAMES[etf_key], f"{etf_prices[etf_key]:.1f}", delta=delta_str)

    st.divider()

    if len(game["gauge_history"]) > 1:
        st.markdown("### 📈 지표 추이")
        df = pd.DataFrame(game["gauge_history"])
        df.columns = [GAUGE_NAMES[k] for k in GAUGE_KEYS]
        st.line_chart(df, height=200)

    if game["log"]:
        st.divider()
        st.markdown("### 📋 진행 기록")
        for entry in reversed(game["log"][-10:]):
            st.caption(entry)


# ── 게임 오버 체크 ──
over = check_game_over(game)
if over:
    st.session_state.game_over = over

if st.session_state.game_over:
    if st.session_state.game_over == "WIN":
        st.balloons()
        st.success("🎉 축하합니다! 20턴을 생존하여 국가를 안정적으로 이끌었습니다!")
        score = max(0, 100 - game["debt"]) + max(0, 100 - game["inflation"]) + game["morale"] + max(0, 100 - game["tension"])
        st.metric("안정도 점수", f"{score} / 400")
    else:
        st.error(f"💀 GAME OVER — {st.session_state.game_over}")

    st.subheader("최종 게이지")
    cols = st.columns(4)
    for col, key in zip(cols, GAUGE_KEYS):
        render_gauge(col, key, game[key])

    if st.button("🔄 다시 시작", use_container_width=True):
        st.session_state.game = new_game()
        st.session_state.phase = "event"
        st.session_state.narration_data = None
        st.session_state.current_event = None
        st.session_state.game_over = None
        st.rerun()
    st.stop()


# ── 헤더 ──
header_cols = st.columns([3, 1])
with header_cols[0]:
    st.title("🏛️ 국가 경제 시뮬레이션")
with header_cols[1]:
    st.markdown(
        f"<div style='text-align:right;padding-top:20px'>"
        f"<span style='font-size:24px;font-weight:bold'>턴 {game['turn']} / {MAX_TURNS}</span></div>",
        unsafe_allow_html=True,
    )

# ── 게이지 표시 ──
gauge_cols = st.columns(4)
for col, key in zip(gauge_cols, GAUGE_KEYS):
    render_gauge(col, key, game[key])

st.divider()


# ══════════════════════════════════════
# Phase: 이벤트 선택
# ══════════════════════════════════════
if st.session_state.phase == "event":
    # 이벤트 선택 (캐시)
    if st.session_state.current_event is None:
        st.session_state.current_event = select_event(game, EVENTS)
    event = st.session_state.current_event

    st.subheader(f"⚡ {event['title']}")
    st.write(event["desc"])

    st.markdown("**정책을 선택하세요:**")
    btn_cols = st.columns(len(event["choices"]))
    for i, (btn_col, choice) in enumerate(zip(btn_cols, event["choices"])):
        with btn_col:
            hint = generate_hint(choice["base_effects"])
            st.caption(hint)
            if st.button(choice["label"], key=f"choice_{i}", use_container_width=True):
                # 선택 적용
                gauge_deltas = apply_choice(game, choice, choice["label"])
                etf_changes = game["etf_history"][-1] if game["etf_history"] else {}

                st.session_state.narration_data = {
                    "event_title": event["title"],
                    "choice_label": choice["label"],
                    "gauge_deltas": gauge_deltas,
                    "current_state": {k: game[k] for k in GAUGE_KEYS},
                    "etf_changes": etf_changes,
                    "turn": game["turn"] - 1,
                }
                st.session_state.phase = "narration"
                st.session_state.current_event = None
                st.rerun()


# ══════════════════════════════════════
# Phase: 나레이션
# ══════════════════════════════════════
elif st.session_state.phase == "narration":
    data = st.session_state.narration_data

    # 결과 요약
    st.subheader(f"📋 {data['choice_label']}")
    delta_cols = st.columns(4)
    for col, key in zip(delta_cols, GAUGE_KEYS):
        d = data["gauge_deltas"].get(key, 0)
        if d != 0:
            col.metric(GAUGE_NAMES[key], data["current_state"][key], delta=f"{d:+d}", delta_color="inverse" if GAUGE_DANGER[key] == "high" else "normal")
        else:
            col.metric(GAUGE_NAMES[key], data["current_state"][key])

    st.divider()

    # LLM 나레이션 스트리밍
    st.markdown("### 📰 뉴스 브리핑")
    st.write_stream(stream_narration(
        event_title=data["event_title"],
        choice_label=data["choice_label"],
        gauge_deltas=data["gauge_deltas"],
        current_state=data["current_state"],
        etf_changes=data["etf_changes"],
        turn=data["turn"],
    ))

    st.divider()

    if st.button("다음 턴 →", use_container_width=True, type="primary"):
        st.session_state.phase = "event"
        st.session_state.narration_data = None
        st.rerun()
