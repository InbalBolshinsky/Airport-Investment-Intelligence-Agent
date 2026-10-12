"""Average fares by airport-pair market, 2025.

Source: DOT "Consumer Airfare Report: Table 1a — All U.S. Airport Pair Markets"
on the datahub.transportation.gov Socrata API (dataset tfrh-tu9e), built from
the DB1B 10% ticket sample.

Cleaning:
- Keep 2025 (all four quarters): airports, average fare, passengers per day,
  non-stop market distance.
- Converted to numbers. Each airport's average fare is computed in the metric
  code (passenger-weighted over its markets), not here.

Known gaps:
- DOT lists only markets where at least one end is in a city with several
  commercial airports (e.g. New York, Chicago, LA, Boston). Airports with no
  such market have no fare (143 of 226 airports with >= 100k passengers are
  covered in 2025).
- Markets are directionless (both directions added together).
- Contiguous US only: no Alaska or Hawaii.
"""

import sqlite3

import pandas as pd

from airport_agent.db import record_source, replace_table
from airport_agent.ingest._http import socrata_rows

DOMAIN = "datahub.transportation.gov"
DATASET_ID = "tfrh-tu9e"
YEAR = 2025


def ingest(conn: sqlite3.Connection) -> int:
    """Load `fares`: one row per airport pair per quarter of YEAR."""
    df = socrata_rows(
        DOMAIN,
        DATASET_ID,
        select="year,quarter,airport_1,airport_2,fare,passengers,nsmiles",
        where=f"year={YEAR}",
    ).rename(columns={"passengers": "passengers_per_day", "nsmiles": "distance_mi", "fare": "avg_fare_usd"})

    numeric = ["year", "quarter", "avg_fare_usd", "passengers_per_day", "distance_mi"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    count = replace_table(conn, "fares", df)
    record_source(
        conn,
        table="fares",
        source="DOT Consumer Airfare Report, Table 1a: airport-pair markets (datahub.transportation.gov API)",
        url=f"https://{DOMAIN}/resource/{DATASET_ID}.json",
        period=f"{YEAR} Q1–Q4",
        row_count=count,
        known_gaps=(
            "Only markets touching a multi-airport city are listed, so some airports have no fare. "
            "Directionless markets. No Alaska or Hawaii."
        ),
    )
    return count
