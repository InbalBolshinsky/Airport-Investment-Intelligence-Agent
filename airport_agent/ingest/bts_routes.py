"""Route-level traffic: departures, seats and passengers per origin–destination pair.

Source: BTS "T-100 Segment (All Carriers)" on TranStats. BTS publishes this
table only as a download form (no API), so we submit the form from code: load
the form page, copy its hidden ASP.NET fields, tick the columns we need, and
post it back; the response is a zip with one CSV.

Years: START_YEAR and END_YEAR, the endpoints of the growth window (D-20).
END_YEAR gives the long-haul share; both give passengers per passenger flight (D-18).

Cleaning:
- Summed over carriers and months to one row per year, origin, destination
  and aircraft configuration (keeps the table small).
- aircraft_config: 1 = passenger, 2 = freight only, 3 = combined
  passenger/freight, 4 = seaplane. Which ones count as "passenger flights" is
  decided in the metric code.

Known gaps:
- All carriers, including foreign airlines on US routes.
- Distance is the published great-circle distance between the airports.
"""

import io
import re
import sqlite3
import zipfile

import pandas as pd

from airport_agent.db import record_source, replace_table
from airport_agent.ingest._http import session
from airport_agent.metrics import END_YEAR, START_YEAR

FORM_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
YEARS = (START_YEAR, END_YEAR)
FIELDS = ["YEAR", "ORIGIN", "DEST", "DISTANCE", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "AIRCRAFT_CONFIG"]
DATA_FILE = "T_T100_SEGMENT_ALL_CARRIER.csv"
DOWNLOAD_TIMEOUT_SECONDS = 600  # BTS builds the file on request; a year takes ~1.5 min


def download_year(year: int) -> pd.DataFrame:
    """Submit the TranStats download form for one year and return the CSV as a DataFrame.

    Raises:
        RuntimeError: if TranStats does not return a zip (e.g. the form changed).
    """
    page = session.get(FORM_URL, timeout=60).text
    form = dict(re.findall(r'<input[^>]*type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page))
    form.update({
        "cboGeography": "All", "cboYear": str(year), "cboPeriod": "All",
        "chkDownloadZip": "on", "btnDownload": "Download",
    })
    form.update({field: "on" for field in FIELDS})

    response = session.post(FORM_URL, data=form, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise RuntimeError(f"TranStats returned no zip for {year}; the download form may have changed.")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        return pd.read_csv(archive.open(DATA_FILE))


def ingest(conn: sqlite3.Connection) -> int:
    """Load `routes`: one row per year, origin, destination and aircraft configuration."""
    raw = pd.concat([download_year(year) for year in YEARS], ignore_index=True)

    routes = (
        raw.groupby(["YEAR", "ORIGIN", "DEST", "AIRCRAFT_CONFIG"], as_index=False)
        .agg(
            distance_mi=("DISTANCE", "max"),
            departures=("DEPARTURES_PERFORMED", "sum"),
            seats=("SEATS", "sum"),
            passengers=("PASSENGERS", "sum"),
        )
        .rename(columns={"YEAR": "year", "ORIGIN": "origin", "DEST": "dest", "AIRCRAFT_CONFIG": "aircraft_config"})
    )

    count = replace_table(conn, "routes", routes)
    record_source(
        conn,
        table="routes",
        source="BTS T-100 Segment, All Carriers (TranStats download form)",
        url=FORM_URL,
        period=", ".join(str(y) for y in YEARS),
        row_count=count,
        known_gaps="Downloaded file, not an API. Summed over carriers and months. Includes foreign carriers.",
    )
    return count
