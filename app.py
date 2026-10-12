"""Chat interface: `streamlit run app.py`.

Type or record a question; the agent answers from its tools. Under each answer,
"How I got this" shows the model, every tool call and the tool's raw result —
so each number in the answer can be checked against what the code returned.

Streamlit reruns this script on every interaction, so the agent and the
conversation live in st.session_state.
"""

import uuid

import pandas as pd
import streamlit as st

from airport_agent.agent import ask, build_agent
from airport_agent.config import ConfigError, settings
from airport_agent.db import connect
from airport_agent.voice import VoiceError, transcribe

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long-haul flights out of Anchorage airport?",
    "What is the unmet flight demand at SFO, and why?",
]

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")


def setup_problem() -> str | None:
    """A setup step the user still has to do, or None if ready."""
    if not settings.db_path.exists():
        return "No data yet. Run `python -m airport_agent.ingest` first (about 3 minutes), then reload."
    try:
        settings.llm_api_key()
    except ConfigError as error:
        return str(error)
    return None


def start_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


def show_how_i_got_this(entry: dict) -> None:
    """Model, tool calls and raw tool results behind one answer."""
    with st.expander("How I got this"):
        st.caption(f"Answered by **{entry.get('model') or 'unknown model'}**")
        if not entry["tool_calls"]:
            st.write("No tools were called for this answer.")
        for call in entry["tool_calls"]:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            st.markdown(f"**`{call['name']}({args})`**")
            st.json(call["result"], expanded=False)


def show_entry(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        if entry.get("heard_by"):
            st.caption(f"🎤 heard (transcribed by {entry['heard_by']}):")
        st.markdown(entry["text"])
        if entry["role"] == "assistant" and "tool_calls" in entry:
            show_how_i_got_this(entry)


# --- setup ----------------------------------------------------------------------
problem = setup_problem()
if problem:
    st.error(problem)
    st.stop()
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
    start_conversation()

# --- sidebar --------------------------------------------------------------------
with st.sidebar:
    st.subheader("Try an example")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if st.button("🔄 New conversation", width="stretch"):
        start_conversation()
        st.rerun()
    st.divider()
    st.caption(
        "Numbers and scores come from deterministic code over public BTS, DOT and FAA data; "
        "the AI chooses tools and explains their results. Passenger counts are departing passengers."
    )
    with st.expander("Data sources"):
        with connect() as conn:
            sources = pd.read_sql(
                "SELECT table_name AS data, period, fetched_on, source FROM data_sources ORDER BY table_name", conn
            )
        st.dataframe(sources, hide_index=True)

# --- conversation ---------------------------------------------------------------
st.title("✈️ Airport Investment Intelligence")
st.caption("Ask where US airport demand is outgrowing capacity - type or record your question. Follow-ups welcome.")

for entry in st.session_state.history:
    show_entry(entry)

submitted = st.chat_input("Ask about US airports…", accept_audio=True)
question, heard_by = st.session_state.pop("pending", None), None
if submitted is not None:
    audio = getattr(submitted, "audio", None)
    if audio is not None:
        try:
            transcript = transcribe(audio.getvalue())
            question, heard_by = transcript.text, transcript.engine
        except VoiceError as error:
            st.error(f"{error} Please type the question instead.")
    else:
        question = getattr(submitted, "text", submitted)

if question:
    user_entry = {"role": "user", "text": question, "heard_by": heard_by}
    st.session_state.history.append(user_entry)
    show_entry(user_entry)
    with st.chat_message("assistant"):
        with st.spinner("Looking it up…"):
            try:
                reply = ask(st.session_state.agent, question, st.session_state.thread_id)
                entry = {"role": "assistant", "text": reply.text, "model": reply.model, "tool_calls": reply.tool_calls}
            except Exception as error:  # provider errors vary (quota, overload, network)
                entry = {"role": "assistant", "text": f"⚠️ The AI service failed ({type(error).__name__}). "
                                                      "Try again in a minute, or change the model in `.env`."}
        st.markdown(entry["text"])
        if "tool_calls" in entry:
            show_how_i_got_this(entry)
    st.session_state.history.append(entry)
