"""The LangChain agent: an LLM that answers analyst questions by calling our tools.

The LLM chooses tools and explains their results; it never produces numbers
itself. Conversation memory is kept in memory per thread_id, so follow-up
questions see the earlier turns (lost on restart). In long conversations,
older tool results are replaced with "[cleared]" in what is sent to the model;
the stored history keeps them. A tool that crashes returns an error message to
the model instead of ending the turn (traceback in the terminal log).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelFallbackMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from airport_agent.config import settings
from airport_agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Kept short on purpose: it is sent with every model call (free-tier token budget).
SYSTEM_PROMPT = """You help analysts at a US airport-investment firm find US airports where demand \
is outgrowing capacity, so renovation (more flight/passenger capacity) is most likely to pay off.

Numbers:
- Every fact and number must come from a tool result. Never estimate, invent or compute numbers \
(no differences, sums or averages) — use compare_airports for differences.
- Passenger counts are departing passengers; say so.
- A higher percentile or score always means more pressure or need; no metric is reversed.
- Terminal size and true airport capacity are not in the data; never describe them.

Tools: rankings/candidates -> rank_airports (region = Census division/region, e.g. "New England"; \
state = 2-letter code); "why is X above Y" -> compare_airports; congestion -> get_congestion_profile; \
long-haul share -> get_flight_distance_mix; unmet demand -> get_unmet_demand; trend/size/details -> \
get_airport_profile. Use airport codes (LA = LAX, Santa Ana = SNA). If a tool returns an error with \
a suggestion, use it or ask the analyst.

Answers:
- Rankings: lead with strong candidates (high score AND airport-constrained), show each airport's size.
- If the question is about terminals or runways specifically, also show that group's score \
(terminal_pressure / runway_pressure) and say the ranking is by overall score.
- Congestion: describe the profile; don't reduce it to one number.
- Unmet demand: call it an estimate; explain "why" from the signals and known constraints.
- Be concise; use a small table for 3+ airports.
- End with "Sources & limits": source, data year and the 2-3 most relevant tool caveats. Never add \
caveats of your own.
- Follow-ups: resolve "it"/"those" from the conversation; call tools again for new numbers, or \
when an earlier tool result shows [cleared]. If the analyst changed the weights, pass the same \
weights to compare_airports.
- If no tool covers the question (e.g. finances, non-US airports), say so."""

# Fail fast instead of appearing frozen: the provider SDKs otherwise retry
# overloaded requests quietly with growing waits. Healthy answers take ~2–5 s.
MAX_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 30

# Long-conversation safety net: every model call resends the whole conversation.
# Once it passes this many tokens (system prompt and tool schemas not counted),
# all but the latest few tool results are sent as "[cleared]", saving quota and
# keeping stale raw numbers out of view. Short demo sessions (~5 turns) never reach it.
CLEAR_TOOL_RESULTS_ABOVE_TOKENS = 6_000
TOOL_RESULTS_KEPT = 3


@dataclass
class AgentReply:
    """The agent's answer to one question, plus the tool calls behind it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # name, args, result
    model: str | None = None  # which model wrote the answer (the fallback, if the main one failed)


def build_model(provider: str, model_name: str, max_retries: int = MAX_RETRIES) -> BaseChatModel:
    """Create a chat model for a provider ("gemini" or "groq").

    Gemini: temperature 1.0, as Google strongly recommends for Gemini 3 models
    (lower values can cause looping); set explicitly because the LangChain
    connector defaults to 0.7. Thinking level comes from GEMINI_THINKING_LEVEL.
    Groq: temperature 0 for repeatable tool choices; reasoning hidden so it stays
    out of the history (Gemini can't replay it, and it costs tokens).

    Args:
        provider: "gemini" or "groq".
        model_name: The provider's model id, e.g. "gemini-3.6-flash".
        max_retries: Quiet retries inside the provider SDK before giving up.

    Raises:
        ConfigError: if the provider's API key is missing.
    """
    api_key = settings.api_key(provider)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=1.0,
            thinking_config={"thinking_level": settings.gemini_thinking_level},
            retries=max_retries,
            request_timeout=REQUEST_TIMEOUT_SECONDS,
        )
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        reasoning_format="hidden",
        max_retries=max_retries,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _tool_error_message(error: Exception, request: ToolCallRequest) -> str:
    """Turn a tool crash into a message the model relays to the analyst.

    Without this the exception ends the turn and leaves an unanswered tool call
    in the stored conversation, which the providers reject on every later turn.
    The model sees only the tool name and error type; the traceback goes to the log.
    """
    name = request.tool_call["name"]
    logger.error("Tool %s failed", name, exc_info=error)
    return (
        f"{name} hit an internal error ({type(error).__name__}) and returned no data. "
        "Tell the analyst this tool could not answer; do not estimate the numbers yourself."
    )


def build_agent():
    """Create the agent: model + our six tools + system prompt + in-memory conversation memory.

    Context editing comes first, so every model — fallbacks included — gets the
    trimmed conversation (see CLEAR_TOOL_RESULTS_ABOVE_TOKENS). It edits a copy
    of the request, so the stored history (and "How I got this") stays complete.
    Tool crashes are returned to the model as error messages (_tool_error_message).

    If fallbacks are configured (and their providers have keys), LangChain's
    ModelFallbackMiddleware retries a failed model call (e.g. "high demand" or
    quota errors) on each fallback in order — by default Gemini Flash-Lite
    (separate quota), then Groq (separate provider). Only the last model gets
    SDK retries of its own: for the others, the next fallback is the retry.
    """
    middleware = [
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=CLEAR_TOOL_RESULTS_ABOVE_TOKENS, keep=TOOL_RESULTS_KEPT)]
        ),
        ToolErrorMiddleware(_tool_error_message),
    ]
    main_retries = MAX_RETRIES
    fallbacks = settings.usable_fallbacks
    if fallbacks:
        models = [
            build_model(provider, model, max_retries=MAX_RETRIES if i == len(fallbacks) - 1 else 0)
            for i, (provider, model) in enumerate(fallbacks)
        ]
        middleware.append(ModelFallbackMiddleware(*models))
        main_retries = 0
    return create_agent(
        build_model(settings.llm_provider, settings.llm_model, max_retries=main_retries),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=middleware,
    )


def _parse_result(content: Any) -> Any:
    """Tool results arrive as JSON text; turn them back into a dict when possible."""
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def ask(agent, question: str, thread_id: str) -> AgentReply:
    """Send one question to the agent and return its answer.

    Args:
        agent: An agent from build_agent().
        question: The analyst's message.
        thread_id: Conversation id; reuse it so follow-ups see earlier turns.

    Returns:
        The final answer text, the tool calls made for this question only
        (name, arguments and the tool's result — the basis for a "how I got
        this" view), and the model that wrote the answer.
    """
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]

    # Keep only this turn: everything after the latest user message.
    last_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    this_turn = messages[last_user + 1:]
    results = {m.tool_call_id: _parse_result(m.content) for m in this_turn if isinstance(m, ToolMessage)}
    tool_calls = [
        {"name": call["name"], "args": call["args"], "result": results.get(call["id"])}
        for m in this_turn
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    final = messages[-1]
    return AgentReply(
        text=final.text,
        tool_calls=tool_calls,
        model=final.response_metadata.get("model_name"),
    )
