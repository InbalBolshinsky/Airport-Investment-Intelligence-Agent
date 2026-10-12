"""The shape every tool returns: the data plus its provenance (source, period,
caveats, and whether it is measured or estimated)."""

from typing import Any, TypedDict


class ToolResult(TypedDict):
    """What a tool hands back to the LLM.

    data: the numbers (and labels) the answer is built from.
    source: where the numbers came from, e.g. "BTS T-100 Segment Summary (API)".
    data_year: period of each table used, e.g. "airport_monthly: 2014-01 to 2026-04; routes: 2019, 2025".
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
    """Build a ToolResult. Provenance args are required keywords, so a tool can't omit them."""
    return ToolResult(
        data=data,
        source=source,
        data_year=data_year,
        caveats=caveats,
        is_estimate=is_estimate,
    )
