"""The six tools the LLM can call — the only part of the system it sees.

Each tool validates its inputs, calls the deterministic code and returns a
ToolResult with provenance and caveats. The docstrings are the LLM's tool
instructions: when to use the tool and what each input means.

Output keys: `_pct` = percentage (82.3 = 82.3%), `_percentile` = rank among
ranked US airports (0 lowest, 100 highest). Scores are 0-100.
"""

import math
from typing import Any

import pandas as pd
from langchain.tools import tool

from airport_agent import analysis, metrics, scoring
from airport_agent.analysis import UnknownAirportError
from airport_agent.db import connect
from airport_agent.results import ToolResult, make_result

SCORE_TABLES = ("airport_monthly", "routes", "fares", "delays", "runways", "airports", "regions")
DEPARTING_CAVEAT = "Passengers = departing passengers only (~half of the total airports usually report)."
BUSY_CAVEAT = (
    "'Airport-constrained' = full planes at one of the US airports with the most flights per long runway "
    "(inferred), not measured capacity. Delays don't count toward it: one month of them is noisy and includes weather."
)
DEFAULT_WEIGHTS_CAVEAT = (
    f"Total score, national rank and strong-candidate flag use the default weights {scoring.DEFAULT_WEIGHTS}."
)

# --- formatting helpers ---------------------------------------------------------

def _num(value: Any, digits: int = 1) -> float | None:
    """Round a number for the LLM; missing values become None."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return round(float(value), digits)


def _pct(fraction: Any) -> float | None:
    """0.823 -> 82.3"""
    return _num(None if pd.isna(fraction) else 100 * fraction)


def _rank(value: Any) -> int | None:
    """A rank for the LLM; None when the airport has no total score (scoring.NO_SCORE_CAVEAT)."""
    return None if pd.isna(value) else int(value)


def _error(message: str) -> ToolResult:
    """A result the LLM can relay (e.g. 'did you mean SNA?') instead of a crash."""
    return make_result({"error": message}, source="input check", data_year="n/a", caveats=[])


def _score_caveat() -> str:
    count = len(analysis.national_metrics())
    return f"Scores are percentiles vs. all {count} ranked US airports; filters don't change scores."


def _scored_result(data: dict, caveats: list[str], *, is_estimate: bool = False) -> ToolResult:
    source, period = analysis.provenance(*SCORE_TABLES)
    return make_result(data, source=source, data_year=period, caveats=caveats, is_estimate=is_estimate)


def _group_scores(row: pd.Series) -> dict[str, float | None]:
    return {group: _num(row[group]) for group in scoring.GROUPS}


def _raw_metrics(row: pd.Series) -> dict[str, float | None]:
    """The measured values behind the percentiles, with units in the key names."""
    return {
        "flight_growth_pct_per_year": _pct(row["flight_growth"]),
        "flights_per_long_runway": _num(row["flights_per_runway"], 0),
        "passenger_growth_pct_per_year": _pct(row["passenger_growth"]),
        "passengers_per_flight_growth_pct_per_year": _pct(row["passengers_per_flight_growth"]),
        "load_factor_pct": _pct(row["load_factor"]),
        "fare_premium_pct": _pct(row["fare_premium"]),
        "aviation_system_delayed_arrivals_pct": _pct(row["nas_delay_rate"]),
    }


def _percentiles(row: pd.Series) -> dict[str, float | None]:
    return {f"{m}_percentile": _num(row[f"{m}_pct"]) for m in scoring.METRICS}


def _summary(code: str, row: pd.Series) -> dict[str, Any]:
    """One airport in a ranking: who, how big, how it scores, and its own caveats."""
    return {
        "airport": code,
        "name": row["name"].title(),
        "state": row["state"],
        "departing_passengers_millions": _num(row["passengers"] / 1e6, 2),
        "total_score": _num(row["total_score"]),
        "national_rank": _rank(row["national_rank"]),
        "group_scores": _group_scores(row),
        "bottleneck": row["bottleneck"],
        "strong_candidate": bool(row["strong_candidate"]),
        "caveats": row["caveats"],
    }


# --- tools ----------------------------------------------------------------------

@tool(parse_docstring=True)
def rank_airports(
    region: str | None = None,
    state: str | None = None,
    min_passengers: int = metrics.MIN_PASSENGERS,
    max_passengers: int | None = None,
    runway_weight: float = scoring.DEFAULT_WEIGHTS["runway_pressure"],
    terminal_weight: float = scoring.DEFAULT_WEIGHTS["terminal_pressure"],
    unmet_demand_weight: float = scoring.DEFAULT_WEIGHTS["unmet_demand"],
    congestion_weight: float = scoring.DEFAULT_WEIGHTS["congestion"],
    top_n: int = 5,
) -> ToolResult:
    """Rank US airports by renovation score: how fast demand is outgrowing capacity.

    Use for questions like "which airports are strong candidates for expansion?".
    Returns airports ordered by total score (0-100), each with its four group
    scores (runway pressure, terminal pressure, unmet demand, congestion),
    national rank, bottleneck label and strong_candidate flag, plus the list of
    strong candidates in the whole selection. A strong candidate has BOTH a high
    score and an airport-constrained bottleneck — lead with these. Weights are
    relative and need not sum to 100.

    Args:
        region: US Census region or division, e.g. "New England" or "West". Omit for all US.
        state: Two-letter US state code, e.g. "CA". Omit for no state filter.
        min_passengers: Exclude airports with fewer departing passengers per year than this.
        max_passengers: Exclude airports with more departing passengers per year than this. Omit for no upper limit.
        runway_weight: Weight of the runway-pressure group (flights per runway, flight growth).
        terminal_weight: Weight of the terminal-pressure group (passenger growth, passengers-per-flight growth).
        unmet_demand_weight: Weight of the unmet-demand group (load factor, fare premium).
        congestion_weight: Weight of the congestion group (delays caused by the aviation system).
        top_n: Number of airports to return.
    """
    weights = {"runway_pressure": runway_weight, "terminal_pressure": terminal_weight,
               "unmet_demand": unmet_demand_weight, "congestion": congestion_weight}
    try:
        scores = analysis.scores(None if weights == scoring.DEFAULT_WEIGHTS else weights)
        selection = scoring.rank_airports(
            scores, area=region, state=state, min_passengers=min_passengers, max_passengers=max_passengers,
        )
    except ValueError as error:
        return _error(str(error))
    if selection.empty:
        return _error("No ranked airports match these filters.")

    top = selection.head(top_n)
    data = {
        "selection": {"region": region, "state": state, "min_departing_passengers": min_passengers,
                      "max_departing_passengers": max_passengers},
        "airports_in_selection": len(selection),
        "weights": weights,
        "strong_candidates": selection.index[selection["strong_candidate"]].tolist(),
        "airports": [
            {**_summary(code, row), "regional_rank": _rank(row["regional_rank"])} for code, row in top.iterrows()
        ],
    }
    caveats = [_score_caveat(), DEPARTING_CAVEAT, BUSY_CAVEAT,
               f"Rankings favor fast-growing smaller airports; many big hubs are still below "
               f"{metrics.START_YEAR} traffic."]
    return _scored_result(data, caveats)


@tool(parse_docstring=True)
def compare_airports(
    airports: list[str],
    runway_weight: float = scoring.DEFAULT_WEIGHTS["runway_pressure"],
    terminal_weight: float = scoring.DEFAULT_WEIGHTS["terminal_pressure"],
    unmet_demand_weight: float = scoring.DEFAULT_WEIGHTS["unmet_demand"],
    congestion_weight: float = scoring.DEFAULT_WEIGHTS["congestion"],
) -> ToolResult:
    """Compare two or more airports' renovation scores side by side.

    Use for follow-ups like "why is BOS above BDL?". Returns each airport's
    total, group scores and metric percentiles, the differences relative to the
    first airport (computed in code — explain the ranking from these, never
    compute your own), and the measured values behind them. If the analyst set
    custom weights earlier, pass the same weights so the comparison matches.

    Args:
        airports: Airport codes (IATA), e.g. ["BOS", "BDL"].
        runway_weight: Weight of the runway-pressure group.
        terminal_weight: Weight of the terminal-pressure group.
        unmet_demand_weight: Weight of the unmet-demand group.
        congestion_weight: Weight of the congestion group.
    """
    weights = {"runway_pressure": runway_weight, "terminal_pressure": terminal_weight,
               "unmet_demand": unmet_demand_weight, "congestion": congestion_weight}
    try:
        codes = list(dict.fromkeys(analysis.resolve_airport(a) for a in airports))  # each airport once
        scores = analysis.scores(None if weights == scoring.DEFAULT_WEIGHTS else weights)
    except ValueError as error:  # unknown airport or invalid weights
        return _error(str(error))
    if len(codes) < 2:
        return _error("Give at least two different airports to compare.")
    table = scoring.compare_airports(scores, codes).rename(index=lambda r: r.replace("_pct", "_percentile"))
    first = codes[0]
    data = {
        "scores": {c: {r: _num(v) for r, v in table[c].items()} for c in codes},
        "differences_vs_first": {
            col: {r: _num(v) for r, v in table[col].items()} for col in table.columns if col not in codes
        },
        "measured_values": {c: _raw_metrics(scores.loc[c]) for c in codes},
        "national_rank": {c: _rank(scores.loc[c, "national_rank"]) for c in codes},
        "bottleneck": {c: scores.loc[c, "bottleneck"] for c in codes},
        "strong_candidate": {c: bool(scores.loc[c, "strong_candidate"]) for c in codes},
        "first_airport": first,
        "weights": weights,
    }
    caveats = [_score_caveat()] + [f"{c}: {note}" for c in codes for note in scores.loc[c, "caveats"]]
    return _scored_result(data, caveats)


@tool(parse_docstring=True)
def get_congestion_profile(airports: list[str]) -> ToolResult:
    """Describe how congested one or more airports are, side by side.

    Use for questions like "compare LAX and SNA congestion". Returns, per
    airport: flights per long runway (and its national percentile), share of
    arriving flights delayed, share delayed by the aviation system (traffic,
    airport operations, non-extreme weather), and known constraints such as
    curfews or caps. Congestion is a profile, not a single number — describe
    the pattern.

    Args:
        airports: Airport codes (IATA), e.g. ["LAX", "SNA"].
    """
    try:
        codes = [analysis.resolve_airport(a) for a in airports]
    except UnknownAirportError as error:
        return _error(str(error))
    scores = analysis.scores()
    with connect() as conn:
        delays = metrics.delay_profile(conn)
        runways = metrics.long_runway_counts(conn)

    profiles = {}
    for c in codes:
        row = scores.loc[c]
        delay = delays.loc[c] if c in delays.index else None
        profiles[c] = {
            "long_runways": int(runways.get(c, 0)),
            "departures_2025": _num(row["flights_per_runway"] * runways.get(c, float("nan")), 0),
            "flights_per_long_runway": _num(row["flights_per_runway"], 0),
            "flights_per_long_runway_percentile": _num(row["flights_per_runway_pct"]),
            "delay_month": None if delay is None else delay["month"],
            "arrivals_in_month": None if delay is None else _num(delay["arrivals"], 0),
            "delayed_arrivals_pct": None if delay is None else _num(delay["pct_delayed"]),
            "aviation_system_delayed_arrivals_pct": None if delay is None else _num(delay["nas_delay_rate_pct"]),
            "aviation_system_share_of_delays_pct": None if delay is None else _num(delay["nas_share_of_delays_pct"]),
            "aviation_system_delay_percentile": _num(row["nas_delay_rate_pct"]),
            "congestion_score": _num(row["congestion"]),
            "known_constraints": analysis.known_constraints(c),
        }
    caveats = [
        "Delays are one month (the latest BTS publishes), arrivals only, larger reporting airlines only — seasonal.",
        "'Aviation-system delays' = delays BTS attributes to the National Aviation System (air traffic control, "
        "airport operations, heavy traffic, non-extreme weather) — the closest BTS cause to a capacity "
        "shortage, but not capacity alone.",
        f"Runway count (>= {metrics.MIN_RUNWAY_FT:,} ft) is a proxy for runway capacity; "
        "it ignores layout and procedures.",
        analysis.CONSTRAINTS_COVERAGE_CAVEAT,
    ] + [f"{c}: no delay data for this airport." for c in codes if profiles[c]["delay_month"] is None]
    return _scored_result(profiles, caveats)


@tool(parse_docstring=True)
def get_flight_distance_mix(
    airport: str,
    short_max_miles: int = metrics.SHORT_HAUL_MAX_MILES,
    long_min_miles: int = metrics.LONG_HAUL_MIN_MILES,
) -> ToolResult:
    """Share of an airport's departures that are short, medium and long-haul.

    Use for questions like "what % of flights out of Anchorage are long-haul?".
    Returns the percentage of 2025 departures in each distance band, two ways:
    passenger_flights (passenger aircraft only — what matters for terminals) and
    all_flights (cargo included). "Flights" is ambiguous, so report both and say
    which is which; they can differ a lot at cargo hubs. State the band limits
    in the answer — there is no official definition.

    Args:
        airport: Airport code (IATA), e.g. "ANC".
        short_max_miles: Flights shorter than this many miles count as short-haul.
        long_min_miles: Flights longer than this many miles count as long-haul.
    """
    try:
        code = analysis.resolve_airport(airport)
        with connect() as conn:
            mixes = {
                view: metrics.distance_mix(conn, code, short_max_miles, long_min_miles, passenger_only=passenger_only)
                for view, passenger_only in (("passenger_flights", True), ("all_flights", False))
            }
    except ValueError as error:  # unknown airport or overlapping bands
        return _error(str(error))
    if mixes["all_flights"] is None:
        return _error(f"No departures recorded for {code} in {metrics.END_YEAR}.")

    data = {
        "airport": code,
        "year": metrics.END_YEAR,
        "bands_miles": {"short_below": short_max_miles, "long_above": long_min_miles},
        **{
            view: None if mix is None else {
                "short_haul_pct": _num(mix["short_pct"]),
                "medium_haul_pct": _num(mix["medium_pct"]),
                "long_haul_pct": _num(mix["long_pct"]),
                "departures": _num(mix["departures"], 0),
            }
            for view, mix in mixes.items()
        },
    }
    caveats = [
        "Two views: passenger_flights counts passenger aircraft only; all_flights adds cargo, combined "
        "passenger/cargo and seaplane flights.",
        "Distance bands are a convention, not an official definition; distances are great-circle miles.",
    ]
    source, period = analysis.provenance("routes")
    return make_result(data, source=source, data_year=f"{metrics.END_YEAR} ({period})", caveats=caveats)


@tool(parse_docstring=True)
def get_unmet_demand(airport: str) -> ToolResult:
    """Estimate unmet demand at an airport and show the signals behind it.

    Use for questions like "what is the unmet demand at SFO, and why?". Returns
    an unmet-demand index (0-100: strength of signal — NOT a passenger count),
    the signals behind it (load factor, fare premium vs. similar-distance trips,
    how busy the runways are, aviation-system delays), a bottleneck label
    ("airport-constrained", "airline-side", "no unmet-demand signal" or "unknown"), the
    strong_candidate flag and known constraints. Always present it as an
    estimate, and explain the "why" from the signals and constraints.

    Args:
        airport: Airport code (IATA), e.g. "SFO".
    """
    try:
        code = analysis.resolve_airport(airport)
    except UnknownAirportError as error:
        return _error(str(error))
    row = analysis.scores().loc[code]
    data = {
        "airport": code,
        "name": row["name"].title(),
        "unmet_demand_index": _num(row["unmet_demand"]),
        "signals": {
            "load_factor_pct": _pct(row["load_factor"]),
            "load_factor_percentile": _num(row["load_factor_pct"]),
            "fare_premium_pct": _pct(row["fare_premium"]),
            "fare_premium_percentile": _num(row["fare_premium_pct"]),
            "flights_per_long_runway": _num(row["flights_per_runway"], 0),
            "flights_per_long_runway_percentile": _num(row["flights_per_runway_pct"]),
            "aviation_system_delayed_arrivals_pct": _pct(row["nas_delay_rate"]),
            "aviation_system_delay_percentile": _num(row["nas_delay_rate_pct"]),
        },
        "bottleneck": row["bottleneck"],
        "strong_candidate": bool(row["strong_candidate"]),
        "total_score": _num(row["total_score"]),
        "national_rank": _rank(row["national_rank"]),
        "known_constraints": analysis.known_constraints(code),
    }
    caveats = [
        "Unmet demand can't be measured, only inferred: the index is the strength of signals, not a count "
        "of passengers who couldn't fly.",
        "Full planes are a hint, not proof — airlines want full planes; they point to a renovation "
        "opportunity only when the airport itself is busy ('airport-constrained').",
        BUSY_CAVEAT,
        f"Fare premium compares fares with the national typical fare for the same trip distance "
        f"({metrics.END_YEAR}).",
        analysis.CONSTRAINTS_COVERAGE_CAVEAT,
        DEFAULT_WEIGHTS_CAVEAT,
        *row["caveats"],
    ]
    return _scored_result(data, caveats, is_estimate=True)


@tool(parse_docstring=True)
def get_airport_profile(airport: str) -> ToolResult:
    """Full detail for one airport: every metric and its yearly history.

    Use for follow-ups like "show me SFO's trend" or "how big is BDL?".
    Returns departing passengers, measured metric values and their national
    percentiles, group scores and rank, and the yearly series 2014-2025 with
    COVID years (2020-2022) flagged.

    Args:
        airport: Airport code (IATA), e.g. "SFO".
    """
    try:
        code = analysis.resolve_airport(airport)
    except UnknownAirportError as error:
        return _error(str(error))
    row = analysis.scores().loc[code]
    with connect() as conn:
        yearly = metrics.yearly_series(conn, code)
    data = {
        "airport": code,
        "name": row["name"].title(),
        "city": row["city"].title(),
        "state": row["state"],
        "census_division": row["division"],
        "departing_passengers_2025": _num(row["passengers"], 0),
        "measured_values": _raw_metrics(row),
        "percentiles": _percentiles(row),
        "group_scores": _group_scores(row),
        "total_score": _num(row["total_score"]),
        "national_rank": _rank(row["national_rank"]),
        "bottleneck": row["bottleneck"],
        "strong_candidate": bool(row["strong_candidate"]),
        "yearly": [
            {"year": int(y.year), "departures": int(y.departures), "departing_passengers": int(y.passengers),
             "load_factor_pct": _num(y.load_factor_pct), "covid_year": bool(y.covid_year)}
            for y in yearly.itertuples()
        ],
    }
    caveats = [
        DEPARTING_CAVEAT,
        f"Growth rates compare {metrics.START_YEAR} with {metrics.END_YEAR} (COVID years skipped).",
        DEFAULT_WEIGHTS_CAVEAT,
        *row["caveats"],
    ]
    return _scored_result(data, caveats)


ALL_TOOLS = [
    rank_airports,
    compare_airports,
    get_congestion_profile,
    get_flight_distance_mix,
    get_unmet_demand,
    get_airport_profile,
]
