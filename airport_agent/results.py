"""The shape every tool returns to the agent.

Architecture rule: every result carries its provenance — where the numbers
came from, how recent they are, the caveats the agent must surface, and
whether the value is measured or estimated.
"""

from typing import Any, TypedDict


class ToolResult(TypedDict):
    """What a tool hands back to the LLM.

    data: the numbers (and labels) the answer is built from.
    source: where the numbers came from, e.g. "BTS T-100 Segment Summary (API)".
    data_year: how recent the data is, e.g. "2025" or "2019–2025".
    caveats: assumptions and limits the agent must mention in its answer.
    is_estimate: True when the value is inferred from proxies, not measured.
    """

    data: dict[str, Any]
    source: str
    data_year: str
    caveats: list[str]
    is_estimate: bool


def make_result(
    data: dict[str, Any],
    *,
    source: str,
    data_year: str,
    caveats: list[str],
    is_estimate: bool = False,
) -> ToolResult:
    """Build a ToolResult. All tools go through this helper.

    source, data_year and caveats are required keyword arguments, so a tool
    that forgets its provenance fails immediately instead of silently
    returning an answer without caveats.
    """
    return ToolResult(
        data=data,
        source=source,
        data_year=data_year,
        caveats=caveats,
        is_estimate=is_estimate,
    )
