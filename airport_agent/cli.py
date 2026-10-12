"""Command-line chat with the agent.

Usage:
    python -m airport_agent.cli "What % of flights out of Anchorage are long-haul?"
    python -m airport_agent.cli          # interactive chat; follow-ups share one conversation
"""

import sys
import uuid

from airport_agent.agent import AgentReply, ask, build_agent
from airport_agent.config import ConfigError
from airport_agent.db import data_problem


def format_reply(reply: AgentReply) -> str:
    """Answer text followed by the model that wrote it and the tools it called."""
    calls = ", ".join(
        f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
        for c in reply.tool_calls
    )
    return f"{reply.text}\n  [model: {reply.model or 'unknown'} | tools: {calls or 'none'}]"


def answer(agent, question: str, thread_id: str) -> str:
    """Ask the agent; turn an AI-service failure into a readable message instead of a traceback."""
    try:
        return format_reply(ask(agent, question, thread_id))
    except Exception as error:  # provider errors vary by SDK (503 overload, 429 quota, timeouts)
        return (
            f"The AI service failed: {type(error).__name__}: {str(error)[:200]}\n"
            "Try again in a minute, or change GEMINI_MODEL / LLM_PROVIDER in .env."
        )


def main() -> int:
    """Answer a single question from the command line, or run an interactive chat."""
    problem = data_problem()
    if problem:
        print(problem)
        return 1
    try:
        agent = build_agent()
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 1

    thread_id = str(uuid.uuid4())  # one conversation per run

    if len(sys.argv) > 1:
        print(answer(agent, " ".join(sys.argv[1:]), thread_id))
        return 0

    print("Airport investment agent — ask a question (empty line to quit).")
    while question := input("\nyou> ").strip():
        print(f"\nagent> {answer(agent, question, thread_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
