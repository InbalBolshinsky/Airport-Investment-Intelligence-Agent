"""Scoring: metrics -> national percentiles -> group scores -> weighted total -> ranking.

Rules (decision numbers refer to the design log):
- Each metric becomes a percentile (0-100) against all ranked US airports (D-6, D-7).
  For every metric, higher = more pressure, so none is flipped.
- A group score is the weighted mean of the group's available metric
  percentiles (METRIC_WEIGHTS; default 1); the total is the weighted mean of
  the available groups, weights rescaled over what the airport has. Nothing
  is filled in (D-48).
- Filters (region, state, size) are applied after national scoring, so an
  airport's score never depends on the question asked (D-7).
"""

import numpy as np
import pandas as pd

GROUPS: dict[str, list[str]] = {
    "runway_pressure": ["flight_growth", "flights_per_runway"],
    "terminal_pressure": ["passenger_growth", "passengers_per_flight_growth"],
    "unmet_demand": ["load_factor", "fare_premium"],
    "congestion": ["nas_delay_rate"],
}
METRICS = [metric for group in GROUPS.values() for metric in group]
DEFAULT_WEIGHTS = {"runway_pressure": 30, "terminal_pressure": 30, "unmet_demand": 30, "congestion": 10}  # D-9
# Weights of metrics inside their group (default 1). An airport that is busy
# today is a surer bet than one growing fast from a small base (D-52).
METRIC_WEIGHTS = {"flights_per_runway": 2}
BOTTLENECK_PERCENTILE = 70  # D-50
STRONG_CANDIDATE_PERCENTILE = 70  # national percentile of total score (D-53)

METRIC_LABELS = {
    "flight_growth": "flight growth",
    "flights_per_runway": "flights per runway",
    "passenger_growth": "passenger growth",
    "passengers_per_flight_growth": "passengers-per-flight growth",
    "load_factor": "load factor",
    "fare_premium": "fare premium",
    "nas_delay_rate": "capacity-related delays",
}
SMALL_BASE_CAVEAT = (
    "Growth starts from a small 2019 base (under 100k passengers); likely a one-time step "
    "(e.g. new airline service), not a steady trend."
)


def percentile_ranks(values: pd.Series) -> pd.Series:
    """Percentile of each value among the non-missing values: 0 = lowest, 100 = highest.

    Percentile = share of the other airports this one beats. Ties share the
    average rank. Missing stays missing. A lone value gets 50.
    """
    valid = values.dropna()
    if len(valid) <= 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=values.index)
    ranks = valid.rank(method="average")
    return ((ranks - 1) / (len(valid) - 1) * 100).reindex(values.index)


def _weighted_mean(values: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Row-wise weighted mean over the non-missing columns; NaN if all are missing."""
    available_weight = values.notna().mul(weights, axis=1).sum(axis=1)
    return values.fillna(0).mul(weights, axis=1).sum(axis=1) / available_weight.replace(0, np.nan)


def _check_weights(weights: dict[str, float] | None) -> pd.Series:
    """Merge user weights over the defaults and validate them."""
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    unknown = set(merged) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown weight group(s) {sorted(unknown)}; valid: {list(GROUPS)}")
    if any(w < 0 for w in merged.values()) or not any(w > 0 for w in merged.values()):
        raise ValueError("Weights must be non-negative and at least one must be positive.")
    return pd.Series(merged)[list(GROUPS)]


def bottleneck_label(scored: pd.DataFrame) -> pd.Series:
    """Who is the bottleneck? (D-15, D-50, D-51)

    The question is "could airlines add flights here today?", so the airport
    side is judged on how busy it is NOW — flights per runway or
    capacity-caused delays — not on growth. "Busy" is inferred from these
    proxies relative to other airports; true capacity isn't in public data.

    - "airport-constrained": planes are full AND the airport is among the busiest
      today (load factor and max(flights per runway, capacity delays) percentiles
      >= BOTTLENECK_PERCENTILE) — a renovation could capture this demand.
    - "airline-side": planes are full but the airport has room — airlines could
      add flights themselves; a renovation is unlikely to help.
    - "no unmet-demand signal": planes are not unusually full.
    - "unknown": no load factor.
    """
    full = scored["load_factor_pct"] >= BOTTLENECK_PERCENTILE
    full_now = scored[["flights_per_runway_pct", "nas_delay_rate_pct"]].max(axis=1)
    pressured = full_now >= BOTTLENECK_PERCENTILE
    labels = np.select(
        [scored["load_factor_pct"].isna(), full & pressured, full],
        ["unknown", "airport-constrained", "airline-side"],
        default="no unmet-demand signal",
    )
    return pd.Series(labels, index=scored.index)


def airport_caveats(row: pd.Series) -> list[str]:
    """Caveats that belong to one airport's result: small-base growth, missing metrics."""
    caveats = [SMALL_BASE_CAVEAT] if row.get("small_base") else []
    caveats += [
        f"No {METRIC_LABELS[m]} data for this airport; scored on its other metrics."
        for m in METRICS if pd.isna(row[m])
    ]
    return caveats


def score(metrics: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score every airport nationally.

    Args:
        metrics: one row per airport with the METRICS columns (from metrics.all_metrics).
        weights: group weights overriding DEFAULT_WEIGHTS, e.g. {"congestion": 0}.
            Relative; they need not sum to 100.

    Returns:
        `metrics` plus: <metric>_pct per metric, one column per group (0-100),
        total_score (0-100), total_score_pct (national percentile of the total),
        national_rank (1 = best), bottleneck, strong_candidate (high score AND
        airport-constrained), missing_metrics, caveats.

    Raises:
        ValueError: for unknown groups or invalid weights.
    """
    w = _check_weights(weights)
    out = metrics.copy()
    for metric in METRICS:
        out[f"{metric}_pct"] = percentile_ranks(out[metric])
    for group, members in GROUPS.items():
        member_weights = pd.Series({f"{m}_pct": METRIC_WEIGHTS.get(m, 1) for m in members})
        out[group] = _weighted_mean(out[member_weights.index], member_weights)

    out["total_score"] = _weighted_mean(out[list(GROUPS)], w)
    out["total_score_pct"] = percentile_ranks(out["total_score"])
    out["national_rank"] = out["total_score"].rank(ascending=False, method="min")
    out["bottleneck"] = bottleneck_label(out)
    # Both signals an investor needs: demand growing into capacity (high score)
    # AND demand blocked by the airport today (D-53).
    out["strong_candidate"] = (out["total_score_pct"] >= STRONG_CANDIDATE_PERCENTILE) & (
        out["bottleneck"] == "airport-constrained"
    )
    out["missing_metrics"] = out[METRICS].isna().apply(lambda r: [m for m in METRICS if r[m]], axis=1)
    out["caveats"] = out.apply(airport_caveats, axis=1)
    return out


def rank_airports(
    scored: pd.DataFrame,
    *,
    area: str | None = None,
    state: str | None = None,
    min_passengers: float | None = None,
    max_passengers: float | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Filter nationally scored airports and order them by total score.

    Args:
        scored: output of score().
        area: US Census division or region, e.g. "New England" or "West" (case-insensitive).
        state: two-letter state code.
        min_passengers / max_passengers: size range on annual passengers (D-5).
        top_n: keep only the first n.

    Returns:
        The matching rows, best first, with regional_rank (1 = best within this
        selection) and regional_count (airports in the selection).

    Raises:
        ValueError: if `area` is not a Census division or region.
    """
    selected = scored
    if area:
        key = area.strip().lower()
        in_area = (selected["division"].str.lower() == key) | (selected["region"].str.lower() == key)
        if not in_area.any():
            valid = sorted(set(scored["division"]) | set(scored["region"]))
            raise ValueError(f"Unknown area '{area}'. Use a Census division or region: {valid}")
        selected = selected[in_area]
    if state:
        selected = selected[selected["state"] == state.strip().upper()]
    if min_passengers is not None:
        selected = selected[selected["passengers"] >= min_passengers]
    if max_passengers is not None:
        selected = selected[selected["passengers"] <= max_passengers]

    ranked = selected.sort_values("total_score", ascending=False).copy()
    ranked["regional_rank"] = ranked["total_score"].rank(ascending=False, method="min")
    ranked["regional_count"] = len(ranked)
    return ranked.head(top_n) if top_n else ranked


def compare_airports(scored: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    """Side-by-side scores for several airports, with differences computed here.

    Returns:
        Rows = total_score, each group, each metric percentile; one column per
        airport, plus "<other> - <first>" difference columns relative to the
        first airport. This is what answers "why is X above Y?".

    Raises:
        KeyError: if an airport is not among the ranked airports.
    """
    missing = [a for a in airports if a not in scored.index]
    if missing:
        raise KeyError(f"Not among ranked airports: {missing}")
    rows = ["total_score", *GROUPS, *(f"{m}_pct" for m in METRICS)]
    table = scored.loc[airports, rows].T
    first = airports[0]
    for other in airports[1:]:
        table[f"{other} - {first}"] = table[other] - table[first]
    return table
