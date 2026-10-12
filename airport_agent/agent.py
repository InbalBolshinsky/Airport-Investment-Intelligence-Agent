"""The LangChain agent: an LLM that answers analyst questions by calling our tools.

The LLM chooses tools and explains their results; it never produces numbers
itself. Conversation memory is kept in memory per thread_id, so follow-up
questions see the earlier turns (lost on restart).
"""

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from airport_agent.config import settings
from airport_agent.tools import ALL_TOOLS

# Placeholder: core rules only. The full prompt is written with the real tools.
SYSTEM_PROMPT = """You help analysts at a US airport-investment firm find airports \
where added flight and passenger capacity is most needed.

Rules:
- Answer facts only from tool results. Never state a number that is not in a tool result.
- Always mention the caveats a tool returns, and its source and data year.
- If a result has is_estimate = true, say clearly that it is an estimate.
- If no tool can answer the question, say so instead of guessing."""

# Fail fast instead of appearing frozen: the provider SDKs otherwise retry
# overloaded requests quietly with growing waits. Healthy answers take ~2–5 s.
MAX_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class AgentReply:
    """The agent's answer to one question, plus the tool calls behind it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None  # which model wrote the answer (the fallback, if the main one failed)


def build_model(provider: str, model_name: str, max_retries: int = MAX_RETRIES) -> BaseChatModel:
    """Create a chat model for a provider ("gemini" or "groq").

    Gemini: temperature 1.0, as Google strongly recommends for Gemini 3 models
    (lower values can cause looping); set explicitly because the LangChain
    connector defaults to 0.7. Thinking level comes from GEMINI_THINKING_LEVEL.
    Groq: temperature 0 for repeatable tool choices; its private reasoning is
    hidden so it doesn't enter the conversation history (Gemini can't replay it
    when it takes over again, and it wastes free-tier tokens).

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


def build_agent():
    """Create the agent: model + our six tools + system prompt + in-memory conversation memory.

    If a fallback is configured and its provider has a key, LangChain's
    ModelFallbackMiddleware retries a failed model call (e.g. "high demand"
    errors) on the fallback — by default Groq, so it doesn't share Google's
    outages. The main model then gets no retries of its own: the fallback is
    the retry.
    """
    middleware = []
    main_retries = MAX_RETRIES
    if settings.fallback_enabled:
        fallback = build_model(settings.fallback_provider, settings.fallback_model)
        middleware.append(ModelFallbackMiddleware(fallback))
        main_retries = 0
    return create_agent(
        build_model(settings.llm_provider, settings.llm_model, max_retries=main_retries),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=middleware,
    )


def ask(agent, question: str, thread_id: str) -> AgentReply:
    """Send one question to the agent and return its answer.

    Args:
        agent: An agent from build_agent().
        question: The analyst's message.
        thread_id: Conversation id; reuse it so follow-ups see earlier turns.

    Returns:
        The final answer text, the tool calls (name + arguments) made for this
        question only — the basis for a "how I got this" view — and the model
        that wrote the answer.
    """
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]

    # Keep only this turn: everything after the latest user message.
    last_user = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    this_turn = messages[last_user + 1:]
    tool_calls = [
        {"name": call["name"], "args": call["args"]}
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
