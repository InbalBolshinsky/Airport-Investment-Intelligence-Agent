"""Metric functions on a tiny in-memory database where every answer is worked out by hand."""

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from airport_agent import metrics


@pytest.fixture
def conn() -> sqlite3.Connection:
    """Three airports:
    AAA — established, in Massachusetts; one long and one short runway.
    BBB — new since 2019 (no 2019 traffic); FAA code differs (XBB); only a short runway.
    CCC — too small to be ranked (50k passengers in 2025).
    """
    c = sqlite3.connect(":memory:")
    pd.DataFrame(
        [  # airport, year, month, departures, passengers, seats
            ("AAA", 2019, 1, 1_000, 100_000, 125_000),
            ("AAA", 2025, 1, 2_000, 400_000, 500_000),
            ("BBB", 2025, 1, 500, 150_000, 200_000),
            ("CCC", 2025, 1, 100, 50_000, 60_000),
        ],
        columns=["airport", "year", "month", "departures", "passengers", "seats"],
    ).to_sql("airport_monthly", c, index=False)
    pd.DataFrame(
        [("AAA", "AAA", "Alpha", "Town A", "MA"), ("XBB", "BBB", "Beta", "Town B", "CA"),
         ("CCC", "CCC", "Gamma", "Town C", "MA")],
        columns=["airport", "bts_code", "airport_name", "city", "state"],
    ).to_sql("airports", c, index=False)
    pd.DataFrame(
        [("MA", "New England", "Northeast"), ("CA", "Pacific", "West")],
        columns=["state", "division", "region"],
    ).to_sql("regions", c, index=False)
    pd.DataFrame(
        [("AAA", "09/27", 6_000), ("AAA", "15/33", 4_000), ("XBB", "01/19", 3_000)],
        columns=["airport", "runway", "length_ft"],
    ).to_sql("runways", c, index=False)
    pd.DataFrame(
        [  # year, origin, dest, aircraft_config, distance_mi, departures, seats, passengers
            (2019, "AAA", "ZZZ", 1, 500, 1_000, 110_000, 90_000),
            (2019, "AAA", "ZZZ", 2, 500, 100, 0, 0),  # cargo: must not count
            (2025, "AAA", "ZZZ", 1, 500, 1_800, 400_000, 360_000),
        ],
        columns=["year", "origin", "dest", "aircraft_config", "distance_mi", "departures", "seats", "passengers"],
    ).to_sql("routes", c, index=False)
    pd.DataFrame(
        [("AAA", 1_000, 50)], columns=["airport", "arr_flights", "nas_ct"]
    ).to_sql("delays", c, index=False)
    return c


def test_annualized_growth_by_hand():
    start = pd.Series([100.0, 100.0, 0.0, np.nan])
    end = pd.Series([200.0, 400.0, 50.0, 10.0])
    one_year = metrics.annualized_growth(start, end, years=1)
    assert one_year[0] == pytest.approx(1.0)       # doubled in one year = +100%
    assert one_year[[2, 3]].isna().all()           # growth from zero or nothing is undefined
    two_years = metrics.annualized_growth(start, end, years=2)
    assert two_years[1] == pytest.approx(1.0)      # x4 over two years = doubling each year


def test_scored_airports_filters_size_and_translates_codes(conn):
    ranked = metrics.scored_airports(conn)
    assert set(ranked.index) == {"AAA", "BBB"}      # CCC is under 100k passengers
    assert ranked.loc["BBB", "state"] == "CA"       # joined through bts_code, not the FAA code
    assert ranked.loc["AAA", "division"] == "New England"


def test_flight_growth(conn):
    growth = metrics.flight_growth(conn)
    assert growth["AAA"] == pytest.approx(2 ** (1 / 6) - 1)  # 1,000 -> 2,000 over 6 years
    assert pd.isna(growth.get("BBB"))                       # no 2019 traffic


def test_flights_per_runway_counts_only_long_runways(conn):
    per_runway = metrics.flights_per_runway(conn)
    assert per_runway["AAA"] == pytest.approx(2_000)   # 2,000 departures / 1 runway >= 5,000 ft
    assert pd.isna(per_runway.get("BBB"))               # only a 3,000 ft runway


def test_passengers_per_flight_growth_excludes_cargo(conn):
    growth = metrics.passengers_per_flight_growth(conn)
    # passenger aircraft only: 90 per flight in 2019 (not 90,000 / 1,100), 200 in 2025
    assert growth["AAA"] == pytest.approx((200 / 90) ** (1 / 6) - 1)


def test_load_factor(conn):
    assert metrics.load_factor(conn)["AAA"] == pytest.approx(0.8)   # 400k / 500k


def test_nas_delay_rate(conn):
    assert metrics.nas_delay_rate(conn)["AAA"] == pytest.approx(0.05)  # 50 / 1,000


def test_small_base(conn):
    small = metrics.small_base(conn)
    assert bool(small["BBB"]) is True    # no 2019 traffic at all
    assert bool(small["AAA"]) is False   # exactly 100,000 is not below the threshold


def test_fare_curve_recovers_a_known_curve():
    distances = np.array([200.0, 500.0, 1_000.0, 2_000.0, 3_000.0])
    fares = pd.DataFrame({
        "distance_mi": distances,
        "avg_fare_usd": 30 * distances ** 0.3,
        "passengers_per_day": [10, 50, 100, 50, 10],
    })
    intercept, slope = metrics.fare_curve(fares)
    assert math.exp(intercept) == pytest.approx(30)
    assert slope == pytest.approx(0.3)


def test_fare_premium_ranks_pricier_airport_higher():
    c = sqlite3.connect(":memory:")
    rows = [  # busy markets exactly on the curve 30 * d^0.3 pin the curve down
        ("YYY", "ZZZ", 30 * d ** 0.3, d, 10_000) for d in (300.0, 800.0, 1_500.0, 2_500.0)
    ] + [
        ("XXX", "AAA", 30 * 500 ** 0.3 * 1.2, 500.0, 10),  # AAA: 20% above typical
        ("XXX", "BBB", 30 * 500 ** 0.3 * 0.8, 500.0, 10),  # BBB: 20% below typical
    ]
    pd.DataFrame(rows, columns=["airport_1", "airport_2", "avg_fare_usd", "distance_mi", "passengers_per_day"]) \
        .to_sql("fares", c, index=False)
    premium = metrics.fare_premium(c)
    assert premium["AAA"] == pytest.approx(0.2, abs=0.01)
    assert premium["BBB"] == pytest.approx(-0.2, abs=0.01)
