"""Deterministic airport metrics, computed from the local database.

Each function returns one value per airport: a pandas Series indexed by the
BTS airport code. Nothing here is scored or ranked — that's scoring.py.
A value that can't be computed (no runways, no fare markets, division by
zero) is missing (NaN), never zero; scoring treats missing explicitly.

Ratios and growth rates are fractions (0.823 = 82.3%).
"""

import sqlite3

import numpy as np
import pandas as pd

# --- Assumptions (all in one place) -------------------------------------------
START_YEAR = 2019               # growth window start: last pre-COVID year (D-20)
END_YEAR = 2025                 # growth window end: latest full year
MIN_PASSENGERS = 100_000        # airports below this in END_YEAR aren't ranked (D-5)
MIN_RUNWAY_FT = 5_000           # runways long enough for airliners (D-42)
PASSENGER_AIRCRAFT_CONFIG = 1   # BTS aircraft_config 1 = passenger aircraft (D-10, D-18)
SMALL_BASE_PASSENGERS = 100_000 # START_YEAR traffic below this = "small base" growth (D-46)


def annualized_growth(start: pd.Series, end: pd.Series, years: int) -> pd.Series:
    """Average yearly growth between two values: (end / start) ** (1 / years) - 1.

    Missing where `start` is missing or not positive (growth from nothing is undefined).
    """
    start = start.where(start > 0)
    return (end / start) ** (1 / years) - 1


def scored_airports(conn: sqlite3.Connection) -> pd.DataFrame:
    """The airports that get ranked, with name, location and END_YEAR passengers.

    Ranked = in the FAA airport list (so foreign airports drop out), in one of
    the 50 states + DC (Census regions), and >= MIN_PASSENGERS in END_YEAR.

    Returns:
        DataFrame indexed by airport code with name, city, state, division,
        region, passengers.
    """
    return pd.read_sql(
        """SELECT m.airport, a.airport_name AS name, a.city, a.state, r.division, r.region,
                  SUM(m.passengers) AS passengers
           FROM airport_monthly m
           JOIN airports a ON a.bts_code = m.airport
           JOIN regions r ON r.state = a.state
           WHERE m.year = :year
           GROUP BY m.airport
           HAVING SUM(m.passengers) >= :min_passengers""",
        conn,
        params={"year": END_YEAR, "min_passengers": MIN_PASSENGERS},
        index_col="airport",
    )


def _annual_total(conn: sqlite3.Connection, column: str, year: int) -> pd.Series:
    """Sum of one airport_monthly column per airport for one year."""
    df = pd.read_sql(
        f"SELECT airport, SUM({column}) AS total FROM airport_monthly WHERE year = ? GROUP BY airport",
        conn, params=(year,), index_col="airport",
    )
    return df["total"]


def flight_growth(conn: sqlite3.Connection) -> pd.Series:
    """Annualized growth in departures, START_YEAR -> END_YEAR (runway pressure).

    Departures include cargo flights: runways serve every aircraft (D-11).
    """
    return annualized_growth(
        _annual_total(conn, "departures", START_YEAR),
        _annual_total(conn, "departures", END_YEAR),
        END_YEAR - START_YEAR,
    ).rename("flight_growth")


def passenger_growth(conn: sqlite3.Connection) -> pd.Series:
    """Annualized growth in passengers, START_YEAR -> END_YEAR (terminal pressure)."""
    return annualized_growth(
        _annual_total(conn, "passengers", START_YEAR),
        _annual_total(conn, "passengers", END_YEAR),
        END_YEAR - START_YEAR,
    ).rename("passenger_growth")


def flights_per_runway(conn: sqlite3.Connection) -> pd.Series:
    """END_YEAR departures per runway of at least MIN_RUNWAY_FT (runway pressure).

    Missing for airports with no such runway in the FAA data (e.g. PBI, D-43).
    Assumes runway count is a fair proxy for runway capacity; it ignores
    layout (e.g. closely spaced parallel runways) and air-traffic procedures.
    """
    runways = pd.read_sql(
        """SELECT a.bts_code AS airport, COUNT(*) AS runways
           FROM runways r JOIN airports a ON a.airport = r.airport
           WHERE r.length_ft >= ?
           GROUP BY a.bts_code""",
        conn, params=(MIN_RUNWAY_FT,), index_col="airport",
    )["runways"]
    departures = _annual_total(conn, "departures", END_YEAR)
    return (departures / runways).rename("flights_per_runway")


def _passengers_per_flight(conn: sqlite3.Connection, year: int) -> pd.Series:
    """Passengers per passenger-aircraft departure for one year (route data)."""
    df = pd.read_sql(
        """SELECT origin AS airport, SUM(passengers) AS passengers, SUM(departures) AS departures
           FROM routes WHERE year = ? AND aircraft_config = ?
           GROUP BY origin""",
        conn, params=(year, PASSENGER_AIRCRAFT_CONFIG), index_col="airport",
    )
    return df["passengers"] / df["departures"].where(df["departures"] > 0)


def passengers_per_flight_growth(conn: sqlite3.Connection) -> pd.Series:
    """Annualized growth in passengers per passenger flight (terminal pressure, D-18).

    Rising = bigger or fuller planes: more people through the terminal per
    flight, which runway metrics don't see. Cargo flights excluded.
    """
    return annualized_growth(
        _passengers_per_flight(conn, START_YEAR),
        _passengers_per_flight(conn, END_YEAR),
        END_YEAR - START_YEAR,
    ).rename("passengers_per_flight_growth")


def load_factor(conn: sqlite3.Connection) -> pd.Series:
    """END_YEAR passengers / seats (unmet demand: a hint, not proof — D-15)."""
    passengers = _annual_total(conn, "passengers", END_YEAR)
    seats = _annual_total(conn, "seats", END_YEAR)
    return (passengers / seats.where(seats > 0)).rename("load_factor")


def fare_curve(fares: pd.DataFrame) -> tuple[float, float]:
    """Fit the national "typical fare for this distance" curve (D-47).

    log(fare) = intercept + slope * log(distance), least squares weighted by
    passengers so busy markets count more.

    Args:
        fares: rows with avg_fare_usd, distance_mi, passengers_per_day (> 0).

    Returns:
        (intercept, slope). Typical fare at distance d = exp(intercept) * d ** slope.
    """
    slope, intercept = np.polyfit(
        np.log(fares["distance_mi"]),
        np.log(fares["avg_fare_usd"]),
        deg=1,
        w=np.sqrt(fares["passengers_per_day"]),  # polyfit squares the weights
    )
    return float(intercept), float(slope)


def fare_premium(conn: sqlite3.Connection) -> pd.Series:
    """How far an airport's fares sit above the typical fare for the same distance.

    Per market: actual fare / typical fare - 1. Per airport: passenger-weighted
    average over every market it is part of (either end). +0.12 = fares 12%
    above what similar-distance trips cost nationally — a sign seats are scarce.

    Missing for airports with no listed market (D-44: DOT lists only markets
    touching a multi-airport city; no Alaska/Hawaii).
    """
    fares = pd.read_sql(
        """SELECT airport_1, airport_2, avg_fare_usd, distance_mi, passengers_per_day FROM fares
           WHERE avg_fare_usd > 0 AND distance_mi > 0 AND passengers_per_day > 0""",
        conn,
    )
    intercept, slope = fare_curve(fares)
    fares["premium"] = fares["avg_fare_usd"] / (np.exp(intercept) * fares["distance_mi"] ** slope) - 1

    both_ends = pd.concat([
        fares.rename(columns={"airport_1": "airport"})[["airport", "premium", "passengers_per_day"]],
        fares.rename(columns={"airport_2": "airport"})[["airport", "premium", "passengers_per_day"]],
    ])
    both_ends["weighted"] = both_ends["premium"] * both_ends["passengers_per_day"]
    sums = both_ends.groupby("airport")[["weighted", "passengers_per_day"]].sum()
    return (sums["weighted"] / sums["passengers_per_day"]).rename("fare_premium")


def nas_delay_rate(conn: sqlite3.Connection) -> pd.Series:
    """Share of arriving flights delayed by the National Aviation System (D-49).

    NAS delays = air traffic control, airport operations, heavy traffic — the
    capacity-related cause. Latest published month only (D-36).
    """
    df = pd.read_sql(
        "SELECT airport, nas_ct, arr_flights FROM delays WHERE arr_flights > 0",
        conn, index_col="airport",
    )
    return (df["nas_ct"] / df["arr_flights"]).rename("nas_delay_rate")


def small_base(conn: sqlite3.Connection) -> pd.Series:
    """True where START_YEAR passengers were below SMALL_BASE_PASSENGERS (D-46).

    Growth from such a base is likely a one-time step (e.g. new airline
    service), not a steady trend. No START_YEAR traffic at all counts as small.
    """
    start = _annual_total(conn, "passengers", START_YEAR)
    airports = _annual_total(conn, "passengers", END_YEAR).index
    return (start.reindex(airports).fillna(0) < SMALL_BASE_PASSENGERS).rename("small_base")


def all_metrics(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every metric for every ranked airport: one row per airport.

    Returns:
        scored_airports() columns plus one column per metric (NaN = missing).
    """
    table = scored_airports(conn)
    for metric in (
        flight_growth, flights_per_runway, passenger_growth, passengers_per_flight_growth,
        load_factor, fare_premium, nas_delay_rate, small_base,
    ):
        table = table.join(metric(conn))
    return table
