"""Arrival delays and their causes per airport, for the latest month BTS publishes.

Source: BTS "Airline On-Time Statistics and Delay Causes", via the TranStats
"Download Raw Data" link: a zip with the latest month (a full year would take
~2 h of flight-level downloads).

Cleaning:
- Summed over airlines to one row per airport and month.
- Kept: arrivals, arrivals 15+ min late, delay counts by cause (airline,
  weather, National Aviation System, security, late incoming aircraft),
  cancellations, and delay minutes (total and NAS).

Known gaps:
- One month only, so seasonal; re-running ingest gets the newest month.
- Arrivals only; only the larger airlines that report on-time data.
- Cause counts are fractional: BTS splits a delayed flight across causes.
"""

import io
import sqlite3
import zipfile

import pandas as pd

from airport_agent.db import record_source, replace_table
from airport_agent.ingest._http import TIMEOUT_SECONDS, session

DOWNLOAD_URL = (
    "https://www.transtats.bts.gov/OT_Delay/ot_delaycause1_DL.aspx"
    "?8n4=9ur4r%20Brn4z106u%20or69rr0%20FHGEM%20NaQ%20FHGEM"
)
DATA_FILE = "Airline_Delay_Cause.csv"
SUMMED = [
    "arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct", "security_ct",
    "late_aircraft_ct", "arr_cancelled", "arr_delay", "nas_delay",
]


def ingest(conn: sqlite3.Connection) -> int:
    """Load `delays`: one row per airport for the latest published month.

    Raises:
        RuntimeError: if BTS does not return a zip (e.g. the link changed).
    """
    response = session.get(DOWNLOAD_URL, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise RuntimeError("BTS delay-cause link returned no zip; the link may have changed.")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw = pd.read_csv(archive.open(DATA_FILE))

    raw[SUMMED] = raw[SUMMED].apply(pd.to_numeric, errors="coerce")
    delays = raw.groupby(["airport", "year", "month"], as_index=False)[SUMMED].sum(min_count=1)

    count = replace_table(conn, "delays", delays)
    periods = sorted({f"{y}-{m:02d}" for y, m in zip(delays.year, delays.month)})
    record_source(
        conn,
        table="delays",
        source="BTS Airline On-Time Statistics and Delay Causes (TranStats download link)",
        url=DOWNLOAD_URL,
        period=", ".join(periods),
        row_count=count,
        known_gaps=(
            "Latest month only (seasonal). Arrivals; reporting (larger) airlines only. "
            "Downloaded file, not an API."
        ),
    )
    return count
