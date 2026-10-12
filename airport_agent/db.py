"""SQLite access: one local database file (DB_PATH), rebuilt by the ingest scripts.

Each table has a row in `data_sources` (source, fetch date, period, known gaps);
tools read their provenance from there.
"""

import sqlite3
from contextlib import closing
from datetime import date

import pandas as pd

from airport_agent.config import settings

# Tables the agent reads, and the ingest source that loads each one.
REQUIRED_TABLES = {
    "airport_monthly": "bts_traffic",
    "routes": "bts_routes",
    "fares": "dot_fares",
    "delays": "bts_delays",
    "airports": "faa",
    "runways": "faa",
    "regions": "regions",
}


def connect() -> sqlite3.Connection:
    """Open the project database, creating its folder if needed."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.db_path)


def missing_tables() -> list[str]:
    """Required tables not in the database (all of them if there is no database file).

    Opens the file read-only: connect() would create an empty database, which
    then looks like data that exists.
    """
    if not settings.db_path.exists():
        return list(REQUIRED_TABLES)
    with closing(sqlite3.connect(f"{settings.db_path.as_uri()}?mode=ro", uri=True)) as conn:
        present = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return [table for table in REQUIRED_TABLES if table not in present]


def data_problem() -> str | None:
    """What to run before the agent can answer, or None if every required table is loaded."""
    missing = missing_tables()
    if not missing:
        return None
    if len(missing) == len(REQUIRED_TABLES):
        return "No data yet. Run `python -m airport_agent.ingest` first (about 3 minutes)."
    sources = " ".join(dict.fromkeys(REQUIRED_TABLES[table] for table in missing))
    return (
        f"Missing data tables {missing} (an ingest source failed). "
        f"Re-run it with `python -m airport_agent.ingest {sources}`."
    )


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
