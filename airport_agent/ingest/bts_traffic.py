"""Monthly traffic per airport: departures, passengers, seats, freight.

Source: BTS "T-100 Segment Summary by Origin Airport" on the data.bts.gov
Socrata API (dataset r495-tyji), January 2014 onwards.

Cleaning:
- Keep only the columns the metrics use; rename to plain names.
- `reporting_month` (a date) -> integer `year` and `month`.
- Counts converted to numbers; a missing value stays missing (NULL), it is
  not treated as zero.

Known gaps:
- Departures include ALL aircraft (passenger and cargo), which is what runway
  metrics want (D-11); passengers per passenger flight comes from the route table.
- "Origin airport" also includes foreign airports with flights to the US;
  they are dropped later by joining with the FAA list of US airports.
- The most recent months may be preliminary and revised by BTS.
"""

import sqlite3

import pandas as pd

from airport_agent.db import record_source, replace_table
from airport_agent.ingest._http import socrata_rows

DOMAIN = "data.bts.gov"
DATASET_ID = "r495-tyji"

COLUMNS = {
    "origin_airport_code": "airport",
    "origin_airport_name": "airport_name",
    "origin_city_name": "city",
    "reporting_month": "reporting_month",
    "total_departures": "departures",
    "total_passengers": "passengers",
    "total_seats": "seats",
    "total_freight_lbs": "freight_lbs",
}
NUMERIC = ["departures", "passengers", "seats", "freight_lbs"]


def ingest(conn: sqlite3.Connection) -> int:
    """Load `airport_monthly`: one row per airport per month."""
    raw = socrata_rows(DOMAIN, DATASET_ID, select=",".join(COLUMNS))
    df = raw.rename(columns=COLUMNS)

    months = pd.to_datetime(df.pop("reporting_month"))
    df["year"] = months.dt.year
    df["month"] = months.dt.month
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    df = df[["airport", "airport_name", "city", "year", "month", *NUMERIC]]

    count = replace_table(conn, "airport_monthly", df)
    first, last = months.min(), months.max()
    record_source(
        conn,
        table="airport_monthly",
        source="BTS T-100 Segment Summary by Origin Airport (data.bts.gov API)",
        url=f"https://{DOMAIN}/resource/{DATASET_ID}.json",
        period=f"{first:%Y-%m} to {last:%Y-%m}",
        row_count=count,
        known_gaps=(
            "Departures include cargo flights. Includes foreign origin airports (filtered via FAA list). "
            "Latest months may be preliminary."
        ),
    )
    return count
