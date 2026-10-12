"""SQLite access: one local database file (DB_PATH), rebuilt by the ingest scripts.

Every table is loaded together with a row in `data_sources` saying where it
came from, when it was fetched, which period it covers and its known gaps —
tools read their provenance from there.
"""

import sqlite3
from datetime import date

import pandas as pd

from airport_agent.config import settings


def connect() -> sqlite3.Connection:
    """Open the project database, creating its folder if needed."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.db_path)


def replace_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> int:
    """Replace table `name` with the contents of `df`. Returns the number of rows written."""
    df.to_sql(name, conn, if_exists="replace", index=False)
    return len(df)


def record_source(
    conn: sqlite3.Connection,
    *,
    table: str,
    source: str,
    url: str,
    period: str,
    row_count: int,
    known_gaps: str = "",
) -> None:
    """Record (or update) the provenance of one table in `data_sources`.

    Args:
        table: The table this describes, e.g. "airport_monthly".
        source: Human-readable source, e.g. "BTS T-100 Segment Summary by Origin Airport (API)".
        url: Where the data was fetched from.
        period: Data period covered, e.g. "2014-01 to 2026-04".
        row_count: Rows loaded.
        known_gaps: Limits a user of this data should know about.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS data_sources (
               table_name TEXT PRIMARY KEY, source TEXT, url TEXT, period TEXT,
               fetched_on TEXT, row_count INTEGER, known_gaps TEXT)"""
    )
    conn.execute(
        "INSERT OR REPLACE INTO data_sources VALUES (?, ?, ?, ?, ?, ?, ?)",
        (table, source, url, period, date.today().isoformat(), row_count, known_gaps),
    )
    conn.commit()
