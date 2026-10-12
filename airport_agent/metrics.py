"""Deterministic airport metrics, computed from the local database.

Each metric returns a pandas Series indexed by BTS airport code. A value that
can't be computed (no runway, no fare market, division by zero) is NaN, never 0.
Ratios and growth rates are fractions (0.823 = 82.3%). Scoring is in scoring.py.
"""

import sqlite3

import numpy as np
import pandas as pd

# --- Assumptions ---------------------------------------------------------------
START_YEAR = 2019               # growth window start: last pre-COVID year
END_YEAR = 2025                 # growth window end: latest full year
MIN_PASSENGERS = 100_000        # airports below this in END_YEAR aren't ranked
MIN_RUNWAY_FT = 5_000           # runways long enough for airliners
PASSENGER_AIRCRAFT_CONFIG = 1   # BTS aircraft_config 1 = passenger aircraft
SMALL_BASE_PASSENGERS = 100_000 # START_YEAR traffic below this = "small base" growth
COVID_YEARS = (2020, 2021, 2022)  # flagged in yearly series; excluded from growth
SHORT_HAUL_MAX_MILES = 800      # default distance bands; adjustable per question
LONG_HAUL_MIN_MILES = 3_000


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

    Departures include cargo flights: runways serve every aircraft.
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


def long_runway_counts(conn: sqlite3.Connection) -> pd.Series:
    """Number of runways of at least MIN_RUNWAY_FT per airport (BTS code)."""
    return pd.read_sql(
        """SELECT a.bts_code AS airport, COUNT(*) AS runways
           FROM runways r JOIN airports a ON a.airport = r.airport
           WHERE r.length_ft >= ?
           GROUP BY a.bts_code""",
        conn, params=(MIN_RUNWAY_FT,), index_col="airport",
    )["runways"].rename("long_runways")


def flights_per_runway(conn: sqlite3.Connection) -> pd.Series:
    """END_YEAR departures per runway of at least MIN_RUNWAY_FT (runway pressure).

    Missing for airports with no such runway in the FAA data.
    Assumes runway count is a fair proxy for runway capacity; it ignores
    layout (e.g. closely spaced parallel runways) and air-traffic procedures.
    """
    departures = _annual_total(conn, "departures", END_YEAR)
    return (departures / long_runway_counts(conn)).rename("flights_per_runway")


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
    """Annualized growth in passengers per passenger flight (terminal pressure).

    Rising = bigger or fuller planes: more people through the terminal per
    flight, which runway metrics don't see. Cargo flights excluded.
    """
    return annualized_growth(
        _passengers_per_flight(conn, START_YEAR),
        _passengers_per_flight(conn, END_YEAR),
        END_YEAR - START_YEAR,
    ).rename("passengers_per_flight_growth")


def load_factor(conn: sqlite3.Connection) -> pd.Series:
    """END_YEAR passengers / seats (unmet demand: a hint, not proof)."""
    passengers = _annual_total(conn, "passengers", END_YEAR)
    seats = _annual_total(conn, "seats", END_YEAR)
    return (passengers / seats.where(seats > 0)).rename("load_factor")


def fare_curve(fares: pd.DataFrame) -> tuple[float, float]:
    """Fit the national "typical fare for this distance" curve.

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
    average over its markets (either end). +0.12 = 12% above the national
    typical fare, a sign seats are scarce.

    Missing when DOT lists no market for the airport (it lists only markets
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
    """Share of arriving flights delayed by the National Aviation System.

    NAS delays = air traffic control, airport operations, heavy traffic and
    non-extreme weather: the closest BTS cause to a capacity shortage, but not
    capacity alone. Latest published month only.
    """
    df = pd.read_sql(
        "SELECT airport, nas_ct, arr_flights FROM delays WHERE arr_flights > 0",
        conn, index_col="airport",
    )
    return (df["nas_ct"] / df["arr_flights"]).rename("nas_delay_rate")


def small_base(conn: sqlite3.Connection) -> pd.Series:
    """True where START_YEAR passengers were below SMALL_BASE_PASSENGERS.

    Growth from such a base is likely a one-time step (e.g. new airline
    service), not a steady trend. No START_YEAR traffic at all counts as small.
    """
    start = _annual_total(conn, "passengers", START_YEAR)
    airports = _annual_total(conn, "passengers", END_YEAR).index
    return (start.reindex(airports).fillna(0) < SMALL_BASE_PASSENGERS).rename("small_base")


def distance_mix(
    conn: sqlite3.Connection,
    airport: str,
    short_max_miles: float = SHORT_HAUL_MAX_MILES,
    long_min_miles: float = LONG_HAUL_MIN_MILES,
    passenger_only: bool = True,
) -> dict[str, float] | None:
    """Share of an airport's END_YEAR departures in each distance band.

    Bands: short < short_max_miles; long > long_min_miles; medium in between.
    passenger_only=True counts passenger aircraft only; False counts every
    aircraft type (passenger, cargo, combined passenger/cargo, seaplane).

    Returns:
        short_pct, medium_pct, long_pct (0-100) and departures,
        or None if the airport has no such departures.

    Raises:
        ValueError: if short_max_miles is above long_min_miles (the bands would overlap).
    """
    if short_max_miles > long_min_miles:
        raise ValueError(
            f"Short-haul limit ({short_max_miles:,} mi) must not be above the long-haul limit ({long_min_miles:,} mi)."
        )
    query = "SELECT distance_mi, departures FROM routes WHERE origin = ? AND year = ?"
    params: tuple = (airport, END_YEAR)
    if passenger_only:
        query += " AND aircraft_config = ?"
        params += (PASSENGER_AIRCRAFT_CONFIG,)
    df = pd.read_sql(query, conn, params=params)
    total = df["departures"].sum()
    if not total:
        return None
    short = df.loc[df["distance_mi"] < short_max_miles, "departures"].sum()
    long = df.loc[df["distance_mi"] > long_min_miles, "departures"].sum()
    return {
        "short_pct": 100 * short / total,
        "medium_pct": 100 * (total - short - long) / total,
        "long_pct": 100 * long / total,
        "departures": float(total),
    }


def delay_profile(conn: sqlite3.Connection) -> pd.DataFrame:
    """Delay picture per airport for the latest published month.

    Returns:
        DataFrame indexed by airport: month ("YYYY-MM"), arrivals, pct_delayed
        (15+ min late), nas_delay_rate_pct (arrivals delayed by the aviation
        system), nas_share_of_delays_pct.
    """
    df = pd.read_sql(
        "SELECT airport, year, month, arr_flights, arr_del15, nas_ct FROM delays WHERE arr_flights > 0",
        conn, index_col="airport",
    )
    return pd.DataFrame({
        "month": df["year"].astype(str) + "-" + df["month"].astype(int).map("{:02d}".format),
        "arrivals": df["arr_flights"],
        "pct_delayed": 100 * df["arr_del15"] / df["arr_flights"],
        "nas_delay_rate_pct": 100 * df["nas_ct"] / df["arr_flights"],
        "nas_share_of_delays_pct": 100 * df["nas_ct"] / df["arr_del15"].where(df["arr_del15"] > 0),
    })


def yearly_series(conn: sqlite3.Connection, airport: str) -> pd.DataFrame:
    """Departures, passengers and load factor per full year up to END_YEAR, COVID years flagged."""
    df = pd.read_sql(
        """SELECT year, SUM(departures) AS departures, SUM(passengers) AS passengers, SUM(seats) AS seats
           FROM airport_monthly WHERE airport = ? AND year <= ? GROUP BY year ORDER BY year""",
        conn, params=(airport, END_YEAR),
    )
    df["load_factor_pct"] = 100 * df["passengers"] / df["seats"].where(df["seats"] > 0)
    df["covid_year"] = df["year"].isin(COVID_YEARS)
    return df.drop(columns="seats")


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
