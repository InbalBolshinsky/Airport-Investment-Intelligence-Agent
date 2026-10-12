"""What the tools stand on: cached national scores, provenance, airport lookup.

Metrics are computed once per process (the data only changes when ingest is
re-run); scores are recomputed only when the analyst asks for other weights.
"""

import difflib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from airport_agent import metrics, scoring
from airport_agent.db import connect

CONSTRAINTS_FILE = Path(__file__).resolve().parent / "reference" / "constraints.csv"
CONSTRAINTS_COVERAGE_CAVEAT = (
    "Known constraints come from a small hand-checked list (SNA, SFO, LAX); "
    "an airport without an entry may still have constraints."
)


class UnknownAirportError(ValueError):
    """An airport code that isn't ranked. The message suggests what to use instead."""


@lru_cache(maxsize=1)
def national_metrics() -> pd.DataFrame:
    """Every metric for every ranked airport (cached for the process)."""
    with connect() as conn:
        return metrics.all_metrics(conn)


@lru_cache(maxsize=1)
def _default_scores() -> pd.DataFrame:
    return scoring.score(national_metrics())


def scores(weights: dict[str, float] | None = None) -> pd.DataFrame:
    """National scores with the default weights (cached) or the analyst's weights."""
    return _default_scores() if not weights else scoring.score(national_metrics(), weights)


def provenance(*tables: str) -> tuple[str, str]:
    """(source, data period) for the given tables, read from data_sources.

    Returns e.g. ("BTS T-100 ...; FAA runway records ...",
                  "airport_monthly: 2014-01 to 2026-04; runways: FAA data effective 2026/10/01").
    """
    with connect() as conn:
        rows = conn.execute(
            f"SELECT table_name, source, period FROM data_sources WHERE table_name IN ({','.join('?' * len(tables))})",
            tables,
        ).fetchall()
    sources = list(dict.fromkeys(source for _, source, _ in rows))  # unique, in order
    return "; ".join(sources), "; ".join(f"{table}: {period}" for table, _, period in rows)


@lru_cache(maxsize=1)
def _constraints() -> pd.DataFrame:
    return pd.read_csv(CONSTRAINTS_FILE)


def known_constraints(airport: str) -> list[dict[str, str]]:
    """Hand-checked, sourced and dated constraints for an airport (D-14), e.g. curfews or caps.

    Returns:
        [{"constraint", "source_url", "checked_on"}, ...]; empty if none listed.
    """
    rows = _constraints()[_constraints()["airport"] == airport]
    return rows[["constraint", "source_url", "checked_on"]].to_dict("records")


def resolve_airport(code: str) -> str:
    """Normalize an airport code and check that it is ranked.

    Returns:
        The upper-case code.

    Raises:
        UnknownAirportError: why it isn't ranked (it has traffic), or close
            matches by code, name or city (it isn't found).
    """
    clean = code.strip().upper()
    ranked = national_metrics()
    if clean in ranked.index:
        return clean

    with connect() as conn:
        has_traffic = conn.execute("SELECT 1 FROM airport_monthly WHERE airport = ? LIMIT 1", (clean,)).fetchone()
    if has_traffic:
        raise UnknownAirportError(
            f"{clean} is not ranked: under {metrics.MIN_PASSENGERS:,} passengers in {metrics.END_YEAR}, "
            "outside the 50 states + DC, or missing from the FAA airport list."
        )

    text = code.strip().lower()
    by_name = ranked[
        ranked["name"].str.lower().str.contains(text, regex=False)
        | ranked["city"].str.lower().str.contains(text, regex=False)
    ].index.tolist()
    by_code = difflib.get_close_matches(clean, ranked.index.tolist(), n=3)
    suggestions = list(dict.fromkeys(by_name + by_code))[:5]
    hint = ", ".join(f"{a} ({ranked.loc[a, 'name'].title()}, {ranked.loc[a, 'city'].title()})" for a in suggestions)
    raise UnknownAirportError(f"Unknown airport '{code}'." + (f" Did you mean: {hint}?" if hint else ""))
