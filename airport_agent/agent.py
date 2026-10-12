"""The LangChain agent: an LLM that answers analyst questions by calling our tools.

The LLM chooses tools and explains their results; it never produces numbers
itself. Conversation memory is kept in memory per thread_id, so follow-up
questions see the earlier turns (lost on restart).
"""

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
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


@dataclass
class AgentReply:
    """The agent's answer to one question, plus the tool calls behind it."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_model() -> BaseChatModel:
    """Create the chat model for the provider selected in .env (LLM_PROVIDER).

    Gemini: temperature 1.0, as Google strongly recommends for Gemini 3 models
    (lower values can cause looping); set explicitly because the LangChain
    connector defaults to 0.7. Thinking level comes from GEMINI_THINKING_LEVEL.
    Groq: temperature 0 for repeatable tool choices.

    Raises:
        ConfigError: if the selected provider's API key is missing.
    """
    api_key = settings.llm_api_key()
    if settings.llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=api_key,
            temperature=1.0,
            thinking_config={"thinking_level": settings.gemini_thinking_level},
        )
    return ChatGroq(model=settings.llm_model, api_key=api_key, temperature=0)


def build_agent():
    """Create the agent: model + our six tools + system prompt + in-memory conversation memory."""
    return create_agent(
        build_model(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def ask(agent, question: str, thread_id: str) -> AgentReply:
    """Send one question to the agent and return its answer.

    Args:
        agent: An agent from build_agent().
        question: The analyst's message.
        thread_id: Conversation id; reuse it so follow-ups see earlier turns.

    Returns:
        The final answer text and the tool calls (name + arguments) made for
        this question only — the basis for a "how I got this" view.
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
    return AgentReply(text=messages[-1].text, tool_calls=tool_calls)
