"""Command-line chat with the agent.

Usage:
    python -m airport_agent.cli "What % of flights out of Anchorage are long-haul?"
    python -m airport_agent.cli          # interactive chat; follow-ups share one conversation
"""

import sys
import uuid

from airport_agent.agent import AgentReply, ask, build_agent
from airport_agent.config import ConfigError


def format_reply(reply: AgentReply) -> str:
    """Answer text followed by the tools called to produce it."""
    calls = ", ".join(
        f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
        for c in reply.tool_calls
    )
    return f"{reply.text}\n  [tools: {calls or 'none'}]"


def main() -> int:
    """Answer a single question from the command line, or run an interactive chat."""
    try:
        agent = build_agent()
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 1

    thread_id = str(uuid.uuid4())  # one conversation per run

    if len(sys.argv) > 1:
        print(format_reply(ask(agent, " ".join(sys.argv[1:]), thread_id)))
        return 0

    print("Airport investment agent — ask a question (empty line to quit).")
    while question := input("\nyou> ").strip():
        print(f"\nagent> {format_reply(ask(agent, question, thread_id))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
