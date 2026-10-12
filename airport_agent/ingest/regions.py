"""US Census regions and divisions, by state (static reference data).

Source: US Census Bureau, "Census Regions and Divisions of the United States".
Used to answer regional questions ("airports in New England") with a
standard, citable definition. 50 states + DC; territories are out of scope.
"""

import sqlite3

import pandas as pd

from airport_agent.db import record_source, replace_table

SOURCE_URL = "https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf"

# division -> (region, states)
DIVISIONS = {
    "New England": ("Northeast", ["CT", "ME", "MA", "NH", "RI", "VT"]),
    "Middle Atlantic": ("Northeast", ["NJ", "NY", "PA"]),
    "East North Central": ("Midwest", ["IL", "IN", "MI", "OH", "WI"]),
    "West North Central": ("Midwest", ["IA", "KS", "MN", "MO", "NE", "ND", "SD"]),
    "South Atlantic": ("South", ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"]),
    "East South Central": ("South", ["AL", "KY", "MS", "TN"]),
    "West South Central": ("South", ["AR", "LA", "OK", "TX"]),
    "Mountain": ("West", ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"]),
    "Pacific": ("West", ["AK", "CA", "HI", "OR", "WA"]),
}


def ingest(conn: sqlite3.Connection) -> int:
    """Load the `regions` table: one row per state with its Census division and region."""
    rows = [
        {"state": state, "division": division, "region": region}
        for division, (region, states) in DIVISIONS.items()
        for state in states
    ]
    count = replace_table(conn, "regions", pd.DataFrame(rows))
    record_source(
        conn,
        table="regions",
        source="US Census Bureau regions and divisions (static reference)",
        url=SOURCE_URL,
        period="current",
        row_count=count,
        known_gaps="US territories (Puerto Rico, Guam, etc.) not included.",
    )
    return count
