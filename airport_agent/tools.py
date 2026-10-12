"""The six tools the LLM can call — the only part of the system it sees.

Each tool is a thin wrapper: it validates inputs, calls deterministic code,
and returns a ToolResult with provenance. Docstrings are sent to the LLM as
the tool's instructions, so they say when to use the tool and what each
input means.

SKELETON: every tool currently returns fake data marked with STUB_CAVEAT.
The real metric and scoring code replaces the stub bodies; signatures stay.
"""

from langchain.tools import tool

from airport_agent.results import ToolResult, make_result

STUB_SOURCE = "stub data (skeleton)"
STUB_CAVEAT = "STUB DATA — not real; the skeleton returns fixed example values."


def _clean_code(airport: str) -> str:
    """Normalize an airport code (' sfo ' -> 'SFO'). Real validation comes later."""
    return airport.strip().upper()


@tool(parse_docstring=True)
def rank_airports(
    region: str | None = None,
    state: str | None = None,
    min_passengers: int = 100_000,
    max_passengers: int | None = None,
    runway_weight: float = 30,
    terminal_weight: float = 30,
    unmet_demand_weight: float = 30,
    congestion_weight: float = 10,
    top_n: int = 10,
) -> ToolResult:
    """Rank US airports by renovation score: how fast demand is outgrowing capacity.

    Use for questions like "which airports are strong candidates for expansion?".
    Returns airports ordered by total score (0-100), each with its four group
    scores (runway pressure, terminal pressure, unmet demand, congestion),
    national percentile, regional rank and bottleneck label. Scores are
    percentiles against all US airports; weights are relative and need not sum
    to 100.

    Args:
        region: US Census region or division, e.g. "New England" or "Pacific". Omit for all US.
        state: Two-letter US state code, e.g. "CA". Omit for no state filter.
        min_passengers: Exclude airports with fewer annual passengers than this.
        max_passengers: Exclude airports with more annual passengers than this. Omit for no upper limit.
        runway_weight: Weight of the runway-pressure group (flight growth, flights per runway).
        terminal_weight: Weight of the terminal-pressure group (passenger growth, passengers-per-flight growth).
        unmet_demand_weight: Weight of the unmet-demand group (load factor, average fare).
        congestion_weight: Weight of the congestion group (capacity-caused delay share).
        top_n: Number of airports to return.
    """
    ranked = [
        {"airport": "BOS", "total_score": 81.0, "regional_rank": 1, "bottleneck": "airport-constrained",
         "groups": {"runway": 84.0, "terminal": 79.0, "unmet_demand": 83.0, "congestion": 72.0}},
        {"airport": "BDL", "total_score": 64.0, "regional_rank": 2, "bottleneck": "airline-side",
         "groups": {"runway": 55.0, "terminal": 68.0, "unmet_demand": 70.0, "congestion": 51.0}},
    ]
    return make_result(
        {"region": region, "state": state, "airports": ranked[:top_n]},
        source=STUB_SOURCE,
        data_year="2019–2025",
        caveats=[STUB_CAVEAT],
    )


@tool(parse_docstring=True)
def compare_airports(airports: list[str]) -> ToolResult:
    """Compare two or more airports' renovation scores side by side.

    Use for follow-ups like "why is BOS above BDL?". Returns each airport's
    total and group scores plus the differences between them, computed in code
    — explain the ranking from these differences, never compute your own.

    Args:
        airports: Airport codes (IATA), e.g. ["BOS", "BDL"].
    """
    codes = [_clean_code(a) for a in airports]
    return make_result(
        {"airports": codes, "scores": {c: {"total_score": 70.0} for c in codes}, "differences": {}},
        source=STUB_SOURCE,
        data_year="2019–2025",
        caveats=[STUB_CAVEAT],
    )


@tool(parse_docstring=True)
def get_congestion_profile(airports: list[str]) -> ToolResult:
    """Describe how congested one or more airports are, side by side.

    Use for questions like "compare LAX and SNA congestion". Returns, per
    airport: flights per runway, share of flights delayed, share of delays
    caused by the aviation system (capacity), known constraints such as
    curfews or caps, and the live FAA status. Congestion is a profile, not a
    single number — describe the pattern.

    Args:
        airports: Airport codes (IATA), e.g. ["LAX", "SNA"].
    """
    codes = [_clean_code(a) for a in airports]
    profile = {
        c: {"flights_per_runway": 50_000, "pct_flights_delayed": 20.0, "nas_delay_share_pct": 30.0,
            "known_constraints": [], "live_faa_status": "no active delays"}
        for c in codes
    }
    return make_result(profile, source=STUB_SOURCE, data_year="2025", caveats=[STUB_CAVEAT])


@tool(parse_docstring=True)
def get_flight_distance_mix(
    airport: str,
    short_max_miles: int = 800,
    long_min_miles: int = 3000,
) -> ToolResult:
    """Share of an airport's passenger departures that are short, medium and long-haul.

    Use for questions like "what % of flights out of Anchorage are long-haul?".
    Returns the percentage of passenger departures in each distance band and
    the band limits used. Cargo flights are excluded.

    Args:
        airport: Airport code (IATA), e.g. "ANC".
        short_max_miles: Flights shorter than this many miles count as short-haul.
        long_min_miles: Flights longer than this many miles count as long-haul.
    """
    return make_result(
        {"airport": _clean_code(airport), "short_pct": 40.0, "medium_pct": 45.0, "long_pct": 15.0,
         "bands_miles": {"short_below": short_max_miles, "long_above": long_min_miles}},
        source=STUB_SOURCE,
        data_year="2025",
        caveats=[STUB_CAVEAT],
    )


@tool(parse_docstring=True)
def get_unmet_demand(airport: str) -> ToolResult:
    """Estimate unmet demand at an airport and show the signals behind it.

    Use for questions like "what is the unmet demand at SFO, and why?". Returns
    an unmet-demand index (0-100, strength of signal — not a passenger count),
    the signals behind it (load factor, average fare, runway and terminal
    pressure), a bottleneck label ("airport-constrained", "airline-side" or
    "no unmet demand") and known constraints. Always present it as an estimate.

    Args:
        airport: Airport code (IATA), e.g. "SFO".
    """
    return make_result(
        {"airport": _clean_code(airport), "unmet_demand_index": 75.0, "bottleneck": "airport-constrained",
         "signals": {"load_factor_pct": 85.0, "avg_fare_usd": 400.0}, "known_constraints": []},
        source=STUB_SOURCE,
        data_year="2025",
        caveats=[STUB_CAVEAT],
        is_estimate=True,
    )


@tool(parse_docstring=True)
def get_airport_profile(airport: str) -> ToolResult:
    """Full detail for one airport: every metric and its yearly history.

    Use for follow-ups like "show me SFO's trend" or "how big is BDL?".
    Returns raw metric values, national percentiles, annual passengers, and
    the yearly series 2014-2025 with COVID years (2020-2022) flagged.

    Args:
        airport: Airport code (IATA), e.g. "SFO".
    """
    return make_result(
        {"airport": _clean_code(airport), "annual_passengers": 1_000_000,
         "yearly": [{"year": 2019, "passengers": 1_000_000, "covid_year": False}]},
        source=STUB_SOURCE,
        data_year="2014–2025",
        caveats=[STUB_CAVEAT],
    )


ALL_TOOLS = [
    rank_airports,
    compare_airports,
    get_congestion_profile,
    get_flight_distance_mix,
    get_unmet_demand,
    get_airport_profile,
]
